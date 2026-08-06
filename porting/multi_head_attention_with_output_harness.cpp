#include "../virne/solver/learning/neural_network/attention.h"

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

struct Arguments {
    bool benchmark = false;
    int warmups = 1;
    int repetitions = 3;
};

struct Summary {
    std::vector<std::int64_t> shape;
    std::string dtype;
    std::uint64_t checksum = fnv_offset;
    std::vector<std::string> bits;
};

struct CaseReport {
    std::string name;
    Summary query_weight;
    Summary key_weight;
    Summary value_weight;
    Summary output_weight;
    Summary output;
    Summary weights;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: vne_multi_head_attention_with_output_harness"
        " [--benchmark] [--warmups N] [--repetitions N]");
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
    else throw std::runtime_error("unexpected attention tensor dtype");
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

torch::Tensor input(
    const std::int64_t batch,
    const std::int64_t length,
    const std::int64_t features,
    const double offset = 0.0) {
    return torch::arange(batch * length * features, torch::kFloat32)
        .reshape({batch, length, features})
        .remainder(47.0)
        .sub(23.0)
        .div(16.0)
        .add(offset);
}

torch::Tensor make_mask(
    const std::int64_t batch,
    const std::int64_t query_length,
    const std::int64_t key_length,
    const bool broadcast) {
    auto mask = broadcast
        ? torch::ones({query_length, key_length}, torch::kFloat32)
        : torch::ones({batch, 1, query_length, key_length}, torch::kFloat32);
    if (broadcast) {
        mask.index_put_({1, 0}, 0.0);
    } else {
        mask.index_put_({0, 0, 1, 2}, 0.0);
        mask.index_put_({1, 0, 0, 1}, 0.0);
    }
    return mask;
}

CaseReport run_case(
    const std::string& name,
    const nn::MultiHeadAttentionWithOutputOptions options,
    const std::int64_t batch,
    const std::int64_t query_length,
    const std::int64_t key_length,
    const std::int64_t value_length,
    const bool masked,
    const bool broadcast_mask) {
    const std::uint64_t seed = name == "default" ? 71 : name == "masked" ? 73 : 79;
    torch::manual_seed(seed);
    nn::MultiHeadAttentionWithOutput attention(options);
    attention->eval();
    const auto query = input(batch, query_length, options.embedding_dim);
    const auto key = input(batch, key_length, options.embedding_dim, 0.25);
    const auto value = input(batch, value_length, options.embedding_dim, -0.5);
    const auto result = masked
        ? attention->forward(query, key, value,
                            make_mask(batch, query_length, key_length, broadcast_mask))
        : attention->forward(query, key, value);
    return CaseReport{
        name,
        summarize(attention->W_q->weight),
        summarize(attention->W_k->weight),
        summarize(attention->W_v->weight),
        summarize(attention->fc->weight),
        summarize(result.output),
        summarize(result.weights),
    };
}

struct BenchmarkReport {
    std::vector<std::int64_t> samples_ns;
    std::uint64_t checksum = 0U;
};

BenchmarkReport benchmark(const Arguments& arguments) {
    torch::manual_seed(101);
    nn::MultiHeadAttentionWithOutput attention(
        nn::MultiHeadAttentionWithOutputOptions{64, 4, 16, 16, 0.1});
    attention->eval();
    const auto query = input(32, 64, 64);
    const auto key = input(32, 64, 64, 0.25);
    const auto value = input(32, 64, 64, -0.5);
    BenchmarkReport result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        result.checksum = summarize(
            attention->forward(query, key, value).output, false).checksum;
    }
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto output = attention->forward(query, key, value).output;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (summarize(output, false).checksum != result.checksum) {
            throw std::runtime_error("multi-head output attention checksum drift");
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
    std::cout << "{\"component\":\"multi_head_attention_with_output\",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) std::cout << ',';
        const auto& value = cases[index];
        std::cout << "{\"name\":\"" << value.name
                  << "\",\"query_weight\":";
        emit_summary(value.query_weight);
        std::cout << ",\"key_weight\":";
        emit_summary(value.key_weight);
        std::cout << ",\"value_weight\":";
        emit_summary(value.value_weight);
        std::cout << ",\"output_weight\":";
        emit_summary(value.output_weight);
        std::cout << ",\"output\":";
        emit_summary(value.output);
        std::cout << ",\"weights\":";
        emit_summary(value.weights);
        std::cout << '}';
    }
    std::cout << ']';
    if (benchmark_report != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"batch32_seq64_dim64_heads4\""
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
            run_case("default", {4, 2, 0, 0, 0.1}, 2, 3, 4, 4, false, false),
            run_case("masked", {5, 2, 3, 2, 0.0}, 2, 2, 3, 3, true, false),
            run_case("broadcast", {6, 3, 2, 1, 0.0}, 1, 3, 2, 2, true, true),
        };
        BenchmarkReport benchmark_report;
        if (arguments.benchmark) benchmark_report = benchmark(arguments);
        emit_report(arguments, cases, arguments.benchmark ? &benchmark_report : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multi_head_attention_with_output_harness: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
