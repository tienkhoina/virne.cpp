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

struct Arguments { bool benchmark = false; int warmups = 1; int repetitions = 3; };
struct Summary { std::vector<std::int64_t> shape; std::string dtype; std::uint64_t checksum = fnv_offset; std::vector<std::string> bits; };
struct CaseReport { std::string name; Summary query_weight; Summary key_weight; Summary output; };

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: vne_multi_head_attention_harness [--benchmark]"
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
    result.dtype = value.scalar_type() == torch::kFloat32 ? "float32" : "int64";
    const auto bytes = static_cast<std::size_t>(value.numel()) * value.element_size();
    const auto* raw = reinterpret_cast<const std::uint8_t*>(value.data_ptr());
    for (std::size_t index = 0; index < bytes; ++index) {
        result.checksum ^= raw[index];
        result.checksum *= fnv_prime;
    }
    if (include_bits) {
        result.bits.reserve(static_cast<std::size_t>(value.numel()));
        for (std::int64_t index = 0; index < value.numel(); ++index) {
            if (value.scalar_type() == torch::kFloat32) {
                std::uint32_t bits = 0U;
                std::memcpy(&bits, raw + index * 4, 4);
                std::ostringstream stream;
                stream << std::hex << std::setfill('0') << std::setw(8) << bits;
                result.bits.push_back(stream.str());
            } else {
                std::uint64_t bits = 0U;
                std::memcpy(&bits, raw + index * 8, 8);
                std::ostringstream stream;
                stream << std::hex << std::setfill('0') << std::setw(16) << bits;
                result.bits.push_back(stream.str());
            }
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

CaseReport run_case(
    const std::string& name,
    const nn::MultiHeadSelfAttentionOptions options,
    const std::int64_t batch,
    const std::int64_t query_length,
    const std::int64_t key_length,
    const bool cross) {
    torch::manual_seed(name == "self" ? 73 : name == "cross" ? 79 : 83);
    nn::MultiHeadSelfAttention attention(options);
    const auto query = input(batch, query_length, options.input_dim);
    const auto key = input(batch, key_length, options.input_dim, 0.25);
    const auto output = cross ? attention->forward(query, key)
                              : attention->forward(query);
    return CaseReport{
        name,
        summarize(attention->W_query),
        summarize(attention->W_key),
        summarize(output),
    };
}

struct BenchmarkReport { std::vector<std::int64_t> samples_ns; std::uint64_t checksum = 0U; };

BenchmarkReport benchmark(const Arguments& arguments) {
    torch::manual_seed(101);
    nn::MultiHeadSelfAttention attention(nn::MultiHeadSelfAttentionOptions{
        64, 128, 0, 0, 4});
    const auto query = input(32, 64, 64);
    BenchmarkReport result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        result.checksum = summarize(attention->forward(query), false).checksum;
    }
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto output = attention->forward(query);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (summarize(output, false).checksum != result.checksum) {
            throw std::runtime_error("multi-head attention checksum drift");
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

void emit_report(const Arguments& arguments, const std::vector<CaseReport>& cases,
                 const BenchmarkReport* benchmark_report) {
    std::cout << "{\"component\":\"multi_head_self_attention\",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << "{\"name\":\"" << cases[index].name << "\",\"query_weight\":";
        emit_summary(cases[index].query_weight);
        std::cout << ",\"key_weight\":";
        emit_summary(cases[index].key_weight);
        std::cout << ",\"output\":";
        emit_summary(cases[index].output);
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
        const std::vector<CaseReport> cases{
            run_case("self", {4, 8, 0, 0, 2}, 2, 3, 3, false),
            run_case("cross", {5, 0, 3, 2, 2}, 3, 2, 4, true),
            run_case("custom", {3, 0, 2, 4, 1}, 2, 4, 3, true),
        };
        BenchmarkReport benchmark_report;
        if (arguments.benchmark) benchmark_report = benchmark(arguments);
        emit_report(arguments, cases, arguments.benchmark ? &benchmark_report : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multi_head_attention_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
