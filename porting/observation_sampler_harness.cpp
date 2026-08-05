#include "../virne/solver/learning/rl_core/observation_sampler.h"

#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rl = virne::solver::learning::rl_core;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    std::string device = "cpu";
    int threads = 1;
    int warmups = 1;
    int repetitions = 3;
    int iterations = 10;
};

struct TensorSummary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::string device;
    std::uint64_t checksum = fnv_offset;
    std::size_t bytes = 0U;
    std::vector<std::string> bits;
};

struct CaseResult {
    std::string name;
    TensorSummary output;
};

struct DenseSummary {
    TensorSummary p_net_x;
    TensorSummary v_node_x;
    TensorSummary curr_v_node_id;
    TensorSummary action_mask;
    TensorSummary v_net_size;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    DenseSummary output;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: vne_observation_sampler_harness [--benchmark]"
          " [--device cpu|cuda|auto] [--threads N] [--warmups N]"
          " [--repetitions N] [--iterations N]");
}

int positive_argument(const char* value, const char* option) {
    try {
        const auto parsed = std::stoll(value);
        if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            usage_error(std::string(option) + " requires a positive integer");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " requires a positive integer");
    }
}

Arguments parse_arguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto read_positive = [&](const char* option) {
            if (++index >= argc) {
                usage_error(std::string(option) + " requires a value");
            }
            return positive_argument(argv[index], option);
        };
        auto read_string = [&](const char* option) {
            if (++index >= argc) {
                usage_error(std::string(option) + " requires a value");
            }
            return std::string(argv[index]);
        };
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--device") {
            result.device = read_string("--device");
            if (result.device != "cpu" && result.device != "cuda"
                && result.device != "auto") {
                usage_error("--device must be cpu, cuda, or auto");
            }
        } else if (argument == "--threads") {
            result.threads = read_positive("--threads");
        } else if (argument == "--warmups") {
            result.warmups = read_positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = read_positive("--repetitions");
        } else if (argument == "--iterations") {
            result.iterations = read_positive("--iterations");
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

torch::Device select_device(const std::string& requested) {
    if (requested == "cuda"
        || (requested == "auto" && torch::cuda::is_available())) {
        if (!torch::cuda::is_available()) {
            usage_error("CUDA requested but unavailable");
        }
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

std::string dtype_name(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    case torch::kInt64: return "int64";
    case torch::kBool: return "bool";
    default: throw std::runtime_error("unsupported sampler summary dtype");
    }
}

template <typename UInt>
std::string bits_hex(const UInt bits) {
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(static_cast<int>(sizeof(UInt) * 2U))
           << static_cast<std::uint64_t>(bits);
    return output.str();
}

template <typename Value, typename UInt>
void append_bits(
    const torch::Tensor& tensor,
    std::vector<std::string>& output) {
    const auto* values = tensor.const_data_ptr<Value>();
    for (std::int64_t index = 0; index < tensor.numel(); ++index) {
        UInt bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        output.push_back(bits_hex(bits));
    }
}

TensorSummary summarize(
    const torch::Tensor& source,
    const bool include_bits = true) {
    const torch::Tensor tensor = source.detach().to(torch::kCPU).contiguous();
    TensorSummary result;
    result.shape = tensor.sizes().vec();
    result.dtype = dtype_name(tensor.scalar_type());
    result.device = source.device().str();
    result.bytes = static_cast<std::size_t>(tensor.nbytes());
    const auto* bytes = static_cast<const std::uint8_t*>(tensor.const_data_ptr());
    for (std::size_t index = 0U; index < result.bytes; ++index) {
        result.checksum ^= bytes[index];
        result.checksum *= fnv_prime;
    }
    if (!include_bits) {
        return result;
    }
    result.bits.reserve(static_cast<std::size_t>(tensor.numel()));
    if (tensor.scalar_type() == torch::kFloat32) {
        append_bits<float, std::uint32_t>(tensor, result.bits);
    } else if (tensor.scalar_type() == torch::kFloat64) {
        append_bits<double, std::uint64_t>(tensor, result.bits);
    } else if (tensor.scalar_type() == torch::kInt64) {
        append_bits<std::int64_t, std::uint64_t>(tensor, result.bits);
    } else {
        const auto* values = tensor.const_data_ptr<bool>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            result.bits.push_back(bits_hex(
                static_cast<std::uint8_t>(values[index])));
        }
    }
    return result;
}

rl::DenseObservationBatch dense_fixture(const torch::Device& device) {
    rl::DenseObservationBatch result;
    result.p_net_x = torch::arange(
        24, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                         .reshape({4, 3, 2});
    result.v_node_x = torch::arange(
        12, torch::TensorOptions().dtype(torch::kFloat64).device(device))
                          .reshape({4, 3});
    result.curr_v_node_id = torch::tensor(
        {10, 11, 12, 13},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    result.action_mask = torch::tensor(
        {1, 0, 0, 1, 1, 1, 0, 0},
        torch::TensorOptions().dtype(torch::kBool).device(device))
                             .reshape({4, 2});
    result.v_net_size = torch::tensor(
        {2.0F, 3.0F, 4.0F, 5.0F},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return result;
}

pyg::Data graph(
    const std::initializer_list<float> nodes,
    const std::initializer_list<std::int64_t> edges) {
    pyg::Data result;
    result.x = torch::tensor(std::vector<float>(nodes)).reshape(
        {static_cast<std::int64_t>(nodes.size()), 1});
    result.edge_index = torch::tensor(
        std::vector<std::int64_t>(edges), torch::kInt64).reshape({2, -1});
    return result;
}

std::vector<CaseResult> run_cases(const torch::Device& device) {
    const torch::Tensor selected = torch::tensor(
        {3, 1, 3, -4}, torch::kInt64);
    const torch::Tensor plain = torch::arange(
        8, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                                    .reshape({4, 2});
    const auto dense = dense_fixture(device);
    const auto sampled_dense = rl::sample_observations(dense, selected, device);

    rl::MlpTensorObservation mlp;
    mlp.p_net_x = dense.p_net_x;
    mlp.general.curr_v_node_id = dense.curr_v_node_id;
    mlp.general.action_mask = dense.action_mask;
    const auto sampled_mlp = rl::sample_observations(mlp, selected, device);

    auto first = graph({1.0F, 2.0F}, {0, 1, 1, 0});
    first.edge_attr = torch::tensor({0.25F, 0.5F}).reshape({2, 1});
    auto second = graph({5.0F, 6.0F, 7.0F}, {0, 1, 1, 2});
    second.edge_attr = torch::tensor({0.75F, 1.0F}).reshape({2, 1});
    const auto graphs = pyg::Batch::from_data_list({first, second}).to(device);
    const auto graph_indices = torch::tensor({1, 0, 1}, torch::kInt64);
    const auto sampled_graphs =
        rl::sample_observations(graphs, graph_indices, device);

    return {
        {"tensor", summarize(rl::sample_observations(plain, selected, device))},
        {"dense_p_net_x", summarize(sampled_dense.p_net_x)},
        {"dense_v_node_x", summarize(sampled_dense.v_node_x)},
        {"dense_curr_v_node_id", summarize(*sampled_dense.curr_v_node_id)},
        {"dense_action_mask", summarize(*sampled_dense.action_mask)},
        {"dense_v_net_size", summarize(*sampled_dense.v_net_size)},
        {"mlp_p_net_x", summarize(sampled_mlp.p_net_x)},
        {"mlp_curr_v_node_id", summarize(*sampled_mlp.general.curr_v_node_id)},
        {"graph_x", summarize(sampled_graphs.x)},
        {"graph_edge_index", summarize(sampled_graphs.edge_index)},
        {"graph_edge_attr", summarize(*sampled_graphs.edge_attr)},
        {"graph_batch", summarize(*sampled_graphs.batch)},
        {"graph_ptr", summarize(*sampled_graphs.ptr)},
        {"graph_edge_ptr", summarize(sampled_graphs.edge_ptr)},
        {"graph_max_num_nodes", summarize(torch::tensor(
            {sampled_graphs.max_num_nodes}, torch::kInt64))},
    };
}

rl::DenseObservationBatch benchmark_fixture(const torch::Device& device) {
    constexpr std::int64_t rows = 4096;
    rl::DenseObservationBatch result;
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    result.p_net_x = torch::arange(rows * 64, f32).reshape({rows, 64})
                         .mul_(0.001F);
    result.v_node_x = torch::arange(rows * 16, f32).reshape({rows, 16})
                          .mul_(0.002F);
    result.curr_v_node_id = torch::arange(rows, i64);
    result.action_mask = torch::arange(rows * 128, i64)
                             .remainder(3).ne(0).reshape({rows, 128});
    result.v_net_size = torch::arange(rows, f32).mul_(0.25F);
    return result;
}

DenseSummary summarize_dense(const rl::DenseObservationBatch& source) {
    return DenseSummary{
        summarize(source.p_net_x, false),
        summarize(source.v_node_x, false),
        summarize(*source.curr_v_node_id, false),
        summarize(*source.action_mask, false),
        summarize(*source.v_net_size, false),
    };
}

BenchmarkResult run_benchmark(
    const Arguments& arguments,
    const torch::Device& device) {
    constexpr std::int64_t rows = 4096;
    constexpr std::int64_t selected_count = 1024;
    const auto source = benchmark_fixture(device);
    const auto indices = torch::arange(selected_count, torch::kInt64)
                             .mul(37).remainder(rows);
    torch::NoGradGuard guard;
    rl::DenseObservationBatch output;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = rl::sample_observations(source, indices, device);
        }
    }
    if (device.is_cuda()) {
        torch::cuda::synchronize(device.index());
    }
    BenchmarkResult result;
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        if (device.is_cuda()) {
            torch::cuda::synchronize(device.index());
        }
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = rl::sample_observations(source, indices, device);
        }
        if (device.is_cuda()) {
            torch::cuda::synchronize(device.index());
        }
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started)
                .count());
    }
    result.output = summarize_dense(output);
    return result;
}

void emit_shape(
    std::ostream& output,
    const std::vector<std::int64_t>& shape) {
    output << '[';
    for (std::size_t index = 0U; index < shape.size(); ++index) {
        if (index != 0U) output << ',';
        output << shape[index];
    }
    output << ']';
}

void emit_summary(std::ostream& output, const TensorSummary& summary) {
    output << "{\"shape\":";
    emit_shape(output, summary.shape);
    output << ",\"dtype\":\"" << summary.dtype
           << "\",\"device\":\"" << summary.device
           << "\",\"checksum\":\"" << summary.checksum
           << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0U; index < summary.bits.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << summary.bits[index] << '"';
    }
    output << "]}";
}

void emit_dense(std::ostream& output, const DenseSummary& summary) {
    output << "{\"p_net_x\":";
    emit_summary(output, summary.p_net_x);
    output << ",\"v_node_x\":";
    emit_summary(output, summary.v_node_x);
    output << ",\"curr_v_node_id\":";
    emit_summary(output, summary.curr_v_node_id);
    output << ",\"action_mask\":";
    emit_summary(output, summary.action_mask);
    output << ",\"v_net_size\":";
    emit_summary(output, summary.v_net_size);
    output << '}';
}

void emit_report(
    const Arguments& arguments,
    const torch::Device& device,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult* benchmark) {
    std::cout << "{\"component\":\"observation_sampler\",\"device\":\""
              << (device.is_cuda() ? "cuda" : "cpu")
              << "\",\"threads\":" << arguments.threads << ",\"cases\":[";
    for (std::size_t index = 0U; index < cases.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << "{\"name\":\"" << cases[index].name << "\",\"output\":";
        emit_summary(std::cout, cases[index].output);
        std::cout << '}';
    }
    std::cout << ']';
    if (benchmark != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"dense_sample_4096_to_1024\""
                  << ",\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"iterations\":" << arguments.iterations
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < benchmark->samples_ns.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << benchmark->samples_ns[index];
        }
        std::cout << "],\"output\":";
        emit_dense(std::cout, benchmark->output);
        std::cout << '}';
    }
    std::cout << "}\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto device = select_device(arguments.device);
        torch::set_num_threads(arguments.threads);
        torch::set_num_interop_threads(arguments.threads);
        const auto cases = run_cases(device);
        BenchmarkResult benchmark;
        if (arguments.benchmark) {
            benchmark = run_benchmark(arguments, device);
        }
        emit_report(arguments, device, cases,
                    arguments.benchmark ? &benchmark : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "observation-sampler harness error: " << error.what() << '\n';
        return 2;
    }
}
