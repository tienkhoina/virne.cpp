#include "../virne/solver/learning/neural_network/neural_tensor.h"

#include <torch/torch.h>

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
struct CaseReport {
    std::string name;
    Summary weight_matrix;
    Summary weight_matrix_block;
    Summary bias;
    Summary output;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: vne_neural_tensor_harness [--benchmark]"
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
    if (value.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("unexpected neural tensor dtype");
    }
    Summary result;
    result.shape = value.sizes().vec();
    result.dtype = "float32";
    const auto bytes = static_cast<std::size_t>(value.numel()) * value.element_size();
    const auto* raw = reinterpret_cast<const std::uint8_t*>(value.data_ptr());
    for (std::size_t index = 0; index < bytes; ++index) {
        result.checksum ^= raw[index];
        result.checksum *= fnv_prime;
    }
    if (include_bits) {
        result.bits.reserve(static_cast<std::size_t>(value.numel()));
        for (std::int64_t index = 0; index < value.numel(); ++index) {
            std::uint32_t bits = 0U;
            std::memcpy(&bits, raw + index * sizeof(bits), sizeof(bits));
            std::ostringstream stream;
            stream << std::hex << std::setfill('0') << std::setw(8) << bits;
            result.bits.push_back(stream.str());
        }
    }
    return result;
}

torch::Tensor input(
    const std::int64_t batch,
    const std::int64_t features,
    const double offset = 0.0) {
    return torch::arange(batch * features, torch::kFloat32)
        .reshape({batch, features})
        .remainder(19.0)
        .sub(9.0)
        .div(7.0)
        .add(offset);
}

CaseReport run_case(
    const std::string& name,
    const std::int64_t input_features,
    const std::int64_t tensor_neurons,
    const std::int64_t batch) {
    const std::uint64_t seed = name == "small" ? 17 : name == "wide" ? 19 : 23;
    torch::manual_seed(seed);
    nn::NeuralTensorNetwork network(
        nn::NeuralTensorNetworkOptions{input_features, tensor_neurons});
    const auto result = network->forward(
        input(batch, input_features), input(batch, input_features, 0.25));
    return CaseReport{
        name,
        summarize(network->weight_matrix),
        summarize(network->weight_matrix_block),
        summarize(network->bias),
        summarize(result),
    };
}

struct BenchmarkReport { std::vector<std::int64_t> samples_ns; std::uint64_t checksum = 0U; };

BenchmarkReport benchmark(const Arguments& arguments) {
    torch::manual_seed(101);
    nn::NeuralTensorNetwork network(nn::NeuralTensorNetworkOptions{64, 32});
    const auto first = input(32, 64);
    const auto second = input(32, 64, 0.25);
    BenchmarkReport result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        result.checksum = summarize(network->forward(first, second), false).checksum;
    }
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto output = network->forward(first, second);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (summarize(output, false).checksum != result.checksum) {
            throw std::runtime_error("neural tensor checksum drift");
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
    std::cout << "{\"component\":\"neural_tensor_network\",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) std::cout << ',';
        const auto& value = cases[index];
        std::cout << "{\"name\":\"" << value.name
                  << "\",\"weight_matrix\":";
        emit_summary(value.weight_matrix);
        std::cout << ",\"weight_matrix_block\":";
        emit_summary(value.weight_matrix_block);
        std::cout << ",\"bias\":";
        emit_summary(value.bias);
        std::cout << ",\"output\":";
        emit_summary(value.output);
        std::cout << '}';
    }
    std::cout << ']';
    if (benchmark_report != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"batch32_features64_neurons32\""
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
            run_case("small", 3, 2, 2),
            run_case("wide", 4, 3, 3),
            run_case("scalar", 1, 1, 1),
        };
        BenchmarkReport benchmark_report;
        if (arguments.benchmark) benchmark_report = benchmark(arguments);
        emit_report(arguments, cases, arguments.benchmark ? &benchmark_report : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neural_tensor_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
