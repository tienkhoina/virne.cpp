#include "../virne/solver/learning/rl_core/action_mask.h"

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

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    std::string device = "cpu";
    int threads = 1;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 25;
};

struct TensorSummary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::size_t bytes = 0U;
    std::vector<std::string> bits;
};

struct CaseResult {
    std::string name;
    TensorSummary output;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    TensorSummary output;
};

[[noreturn]] void usage_error(const std::string_view message) {
    throw std::invalid_argument(
        std::string(message) +
        "\nusage: vne_action_mask_harness [--benchmark]"
        " [--device cpu|cuda|auto] [--threads N] [--warmups N]"
        " [--repetitions N] [--iterations N]");
}

int positive_int(const char* value, const char* option) {
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
            if (++index >= argc) usage_error(std::string(option) + " requires a value");
            return positive_int(argv[index], option);
        };
        auto read_string = [&](const char* option) {
            if (++index >= argc) usage_error(std::string(option) + " requires a value");
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

torch::Device select_device(const std::string& request) {
    if (request == "cuda" || (request == "auto" && torch::cuda::is_available())) {
        if (!torch::cuda::is_available()) usage_error("CUDA requested but unavailable");
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

std::string dtype_name(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat32: return "float32";
    case torch::kFloat64: return "float64";
    case torch::kBool: return "bool";
    default: throw std::runtime_error("unsupported action-mask output dtype");
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

TensorSummary summarize(const torch::Tensor& source, const bool include_bits = true) {
    const auto tensor = source.detach().to(torch::kCPU).contiguous();
    TensorSummary result;
    result.shape = tensor.sizes().vec();
    result.dtype = dtype_name(tensor.scalar_type());
    result.bytes = static_cast<std::size_t>(tensor.nbytes());
    const auto* bytes = static_cast<const std::uint8_t*>(tensor.const_data_ptr());
    for (std::size_t index = 0U; index < result.bytes; ++index) {
        result.checksum ^= bytes[index];
        result.checksum *= fnv_prime;
    }
    if (!include_bits) return result;
    result.bits.reserve(static_cast<std::size_t>(tensor.numel()));
    if (tensor.scalar_type() == torch::kFloat32) {
        const auto* values = tensor.const_data_ptr<float>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            std::uint32_t bits = 0U;
            std::memcpy(&bits, values + index, sizeof(bits));
            result.bits.push_back(bits_hex(bits));
        }
    } else if (tensor.scalar_type() == torch::kFloat64) {
        const auto* values = tensor.const_data_ptr<double>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            std::uint64_t bits = 0U;
            std::memcpy(&bits, values + index, sizeof(bits));
            result.bits.push_back(bits_hex(bits));
        }
    } else {
        const auto* values = tensor.const_data_ptr<bool>();
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            result.bits.push_back(bits_hex(static_cast<std::uint8_t>(values[index])));
        }
    }
    return result;
}

torch::Tensor values32(const torch::Device& device) {
    return torch::tensor({1.5F, -2.25F, 0.0F, 4.75F},
                         torch::TensorOptions().dtype(torch::kFloat32)
                             .device(device)).reshape({2, 2});
}

torch::Tensor values64(const torch::Device& device) {
    return torch::tensor({1.25, -3.5, 2.75, 8.0},
                         torch::TensorOptions().dtype(torch::kFloat64)
                             .device(device)).reshape({2, 2});
}

std::vector<CaseResult> run_cases(const torch::Device& device) {
    const auto f32 = values32(device);
    const auto f64 = values64(device);
    const auto mixed_bool = torch::tensor(
        {1, 0, 1, 0}, torch::TensorOptions().dtype(torch::kBool)
                              .device(device)).reshape({1, 4});
    const auto integer_mask = torch::tensor(
        {2, 0, -1, 0}, torch::TensorOptions().dtype(torch::kInt64)
                                .device(device));
    const auto numeric_mask = torch::tensor(
        {1.0, 0.0, -2.0, 0.25}, torch::TensorOptions().dtype(torch::kFloat64)
                                       .device(device));
    const auto all_true = torch::ones({2, 2},
                                      torch::TensorOptions().dtype(torch::kBool)
                                          .device(device));
    const auto all_false = torch::zeros({2, 2},
                                        torch::TensorOptions().dtype(torch::kBool)
                                            .device(device));
    const std::vector<std::uint8_t> vector_mask{1U, 0U, 1U, 0U};
    return {
        {"f32_logit_bool_reshape", summarize(rl::apply_mask_to_logit(f32, mixed_bool))},
        {"f32_prob_int", summarize(rl::apply_mask_to_prob(f32, integer_mask))},
        {"f32_logit_vector", summarize(rl::apply_mask_to_logit(f32, vector_mask))},
        {"f32_prob_all_false", summarize(rl::apply_mask_to_prob(f32, all_false))},
        {"f32_logit_all_true", summarize(rl::apply_mask_to_logit(f32, all_true))},
        {"f32_logit_identity", summarize(rl::apply_mask_to_logit(f32))},
        {"f32_prob_identity", summarize(rl::apply_mask_to_prob(f32))},
        {"f64_logit_float", summarize(rl::apply_mask_to_logit(f64, numeric_mask))},
        {"f64_prob_float", summarize(rl::apply_mask_to_prob(f64, numeric_mask))},
    };
}

BenchmarkResult run_benchmark(const Arguments& arguments, const torch::Device& device) {
    constexpr std::int64_t rows = 256;
    constexpr std::int64_t columns = 1024;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto values = torch::arange(rows * columns, options).reshape({rows, columns})
                            .mul_(0.0007F).sub_(0.3F);
    const auto indices = torch::arange(
        rows * columns,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    const auto mask = indices.remainder(5).ne(0).reshape({rows, columns});
    torch::NoGradGuard guard;
    torch::Tensor output;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = rl::apply_mask_to_logit(values, mask);
        }
    }
    if (device.is_cuda()) torch::cuda::synchronize();
    BenchmarkResult result;
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        if (device.is_cuda()) torch::cuda::synchronize();
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            output = rl::apply_mask_to_logit(values, mask);
        }
        if (device.is_cuda()) torch::cuda::synchronize();
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count());
    }
    result.output = summarize(output, false);
    return result;
}

void emit_shape(std::ostream& output, const std::vector<std::int64_t>& shape) {
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
           << "\",\"checksum\":\"" << summary.checksum
           << "\",\"bytes\":" << summary.bytes << ",\"bits\":[";
    for (std::size_t index = 0U; index < summary.bits.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << summary.bits[index] << '"';
    }
    output << "]}";
}

void emit_report(
    const Arguments& arguments,
    const torch::Device& device,
    const std::vector<CaseResult>& cases,
    const BenchmarkResult* benchmark) {
    std::cout << "{\"component\":\"action_mask\",\"device\":\""
              << (device.is_cuda() ? "cuda" : "cpu") << "\",\"threads\":"
              << arguments.threads << ",\"cases\":[";
    for (std::size_t index = 0U; index < cases.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << "{\"name\":\"" << cases[index].name << "\",\"output\":";
        emit_summary(std::cout, cases[index].output);
        std::cout << '}';
    }
    std::cout << ']';
    if (benchmark != nullptr) {
        std::cout << ",\"benchmark\":{\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"iterations\":" << arguments.iterations
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < benchmark->samples_ns.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << benchmark->samples_ns[index];
        }
        std::cout << "],\"output\":";
        emit_summary(std::cout, benchmark->output);
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
        if (arguments.benchmark) benchmark = run_benchmark(arguments, device);
        emit_report(arguments, device, cases,
                    arguments.benchmark ? &benchmark : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "action-mask harness error: " << error.what() << '\n';
        return 2;
    }
}
