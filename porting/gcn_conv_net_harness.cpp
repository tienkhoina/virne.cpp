#include "../virne/solver/learning/neural_network/gnn.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nn = virne::solver::learning::neural_network;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments { bool benchmark = false; int warmups = 1; int repetitions = 3; };
struct Summary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::vector<std::string> bits;
};
struct StateEntry { std::string name; Summary value; };
struct CaseReport { std::string name; std::vector<StateEntry> state; Summary output; };

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: vne_gcn_conv_net_harness [--benchmark]"
        " [--warmups N] [--repetitions N]");
}

Arguments parse_arguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto positive = [&](const char* option) {
            if (++index >= argc) usage_error(std::string(option) + " needs N");
            const int value = std::stoi(argv[index]);
            if (value <= 0) usage_error(std::string(option) + " must be positive");
            return value;
        };
        if (argument == "--benchmark") result.benchmark = true;
        else if (argument == "--warmups") result.warmups = positive("--warmups");
        else if (argument == "--repetitions") result.repetitions = positive("--repetitions");
        else usage_error("unknown argument: " + std::string(argument));
    }
    return result;
}

Summary summarize(const torch::Tensor& input, const bool include_bits = true) {
    const auto value = input.detach().to(torch::kCPU).contiguous();
    Summary result;
    result.shape = value.sizes().vec();
    if (value.scalar_type() == torch::kFloat32) result.dtype = "float32";
    else if (value.scalar_type() == torch::kInt64) result.dtype = "int64";
    else throw std::runtime_error("unexpected GCN state dtype");
    const auto bytes = static_cast<std::size_t>(value.numel()) * value.element_size();
    const auto* raw = reinterpret_cast<const std::uint8_t*>(value.data_ptr());
    for (std::size_t index = 0; index < bytes; ++index) {
        result.checksum ^= raw[index];
        result.checksum *= fnv_prime;
    }
    if (include_bits) {
        result.bits.reserve(static_cast<std::size_t>(value.numel()));
        for (std::int64_t index = 0; index < value.numel(); ++index) {
            std::ostringstream stream;
            if (value.scalar_type() == torch::kFloat32) {
                std::uint32_t bits = 0U;
                std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
                stream << std::hex << std::setfill('0') << std::setw(8) << bits;
            } else {
                std::uint64_t bits = 0U;
                std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
                stream << std::hex << std::setfill('0') << std::setw(16) << bits;
            }
            result.bits.push_back(stream.str());
        }
    }
    return result;
}

torch::Tensor node_input(const std::int64_t count, const std::int64_t features,
                         const double offset = 0.0) {
    return torch::arange(count * features, torch::kFloat32)
        .reshape({count, features})
        .remainder(29.0)
        .sub(14.0)
        .div(9.0)
        .add(offset);
}

pyg::Batch make_batch(const std::vector<std::int64_t>& node_counts) {
    std::vector<pyg::Data> graphs;
    graphs.reserve(node_counts.size());
    for (std::size_t graph_id = 0; graph_id < node_counts.size(); ++graph_id) {
        const auto nodes = node_counts[graph_id];
        pyg::Data graph;
        graph.x = node_input(nodes, 3, static_cast<double>(graph_id) * 0.25);
        std::vector<std::int64_t> rows;
        std::vector<std::int64_t> cols;
        for (std::int64_t node = 0; node + 1 < nodes; ++node) {
            rows.push_back(node);
            cols.push_back(node + 1);
            rows.push_back(node + 1);
            cols.push_back(node);
        }
        graph.edge_index = torch::stack({
            torch::tensor(rows, torch::kInt64),
            torch::tensor(cols, torch::kInt64)}, 0);
        graphs.push_back(std::move(graph));
    }
    return pyg::Batch::from_data_list(graphs);
}

std::vector<StateEntry> summarize_state(nn::GCNConvNet& model) {
    std::vector<StateEntry> result;
    for (const auto& item : model->named_parameters(true)) {
        result.push_back(StateEntry{item.key(), summarize(item.value())});
    }
    for (const auto& item : model->named_buffers(true)) {
        result.push_back(StateEntry{item.key(), summarize(item.value())});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.name < right.name;
    });
    return result;
}

CaseReport run_case(
    const std::string& name,
    const nn::GCNConvNetOptions options,
    const std::vector<std::int64_t>& node_counts) {
    const std::uint64_t seed = name == "sparse" ? 31 : name == "dense_bn" ? 37
        : name == "sum" ? 41 : name == "mean" ? 43 : 47;
    torch::manual_seed(seed);
    nn::GCNConvNet network(options);
    network->eval();
    const auto batch = make_batch(node_counts);
    return CaseReport{name, summarize_state(network), summarize(network->forward(batch))};
}

struct BenchmarkReport { std::vector<std::int64_t> samples_ns; std::uint64_t checksum = 0U; };

BenchmarkReport benchmark(const Arguments& arguments) {
    torch::manual_seed(101);
    nn::GCNConvNet network(nn::GCNConvNetOptions{
        3, 32, 64, 3, false, 1.0, false, nn::GraphPoolingKind::none});
    network->eval();
    const auto batch = make_batch(std::vector<std::int64_t>(32, 16));
    BenchmarkReport result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        result.checksum = summarize(network->forward(batch), false).checksum;
    }
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto output = network->forward(batch);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (summarize(output, false).checksum != result.checksum) {
            throw std::runtime_error("GCN wrapper checksum drift");
        }
    }
    return result;
}

void emit_summary(const Summary& value) {
    std::cout << "{\"shape\":[";
    for (std::size_t index = 0; index < value.shape.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << value.shape[index];
    }
    std::cout << "],\"dtype\":\"" << value.dtype
              << "\",\"checksum\":\"" << value.checksum << "\",\"bits\":[";
    for (std::size_t index = 0; index < value.bits.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << value.bits[index] << '"';
    }
    std::cout << "]}";
}

void emit_report(
    const Arguments& arguments,
    const std::vector<CaseReport>& cases,
    const BenchmarkReport* benchmark_report) {
    std::cout << "{\"component\":\"gcn_conv_net\",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) std::cout << ',';
        const auto& value = cases[index];
        std::cout << "{\"name\":\"" << value.name << "\",\"state\":[";
        for (std::size_t state_id = 0; state_id < value.state.size(); ++state_id) {
            if (state_id != 0) std::cout << ',';
            std::cout << "{\"name\":\"" << value.state[state_id].name
                      << "\",\"value\":";
            emit_summary(value.state[state_id].value);
            std::cout << '}';
        }
        std::cout << "],\"output\":";
        emit_summary(value.output);
        std::cout << '}';
    }
    std::cout << ']';
    if (benchmark_report != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"batch32_nodes16_dim3_to32\""
                  << ",\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0; index < benchmark_report->samples_ns.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << benchmark_report->samples_ns[index];
        }
        std::cout << "],\"checksum\":\"" << benchmark_report->checksum << "\"}";
    }
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        torch::NoGradGuard no_grad;
        const std::vector<CaseReport> cases{
            run_case("sparse", {3, 4, 6, 1, false, 1.0, false,
                                nn::GraphPoolingKind::none}, {3, 5}),
            run_case("dense_bn", {3, 4, 6, 3, true, 1.0, true,
                                   nn::GraphPoolingKind::none}, {3, 5}),
            run_case("sum", {3, 4, 6, 3, false, 1.0, false,
                              nn::GraphPoolingKind::add}, {3, 5}),
            run_case("mean", {3, 4, 6, 3, false, 1.0, false,
                               nn::GraphPoolingKind::mean}, {3, 5}),
            run_case("max", {3, 4, 6, 3, false, 1.0, false,
                              nn::GraphPoolingKind::max}, {3, 5}),
        };
        BenchmarkReport benchmark_report;
        if (arguments.benchmark) benchmark_report = benchmark(arguments);
        emit_report(arguments, cases, arguments.benchmark ? &benchmark_report : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gcn_conv_net_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
