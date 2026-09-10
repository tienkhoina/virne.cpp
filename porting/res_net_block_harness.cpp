#include "../virne/solver/learning/neural_network/res.h"

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

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments { bool benchmark = false; int warmups = 1; int repetitions = 3; };
struct Summary {
    std::vector<std::int64_t> shape;
    std::uint64_t checksum = fnv_offset;
    std::vector<std::string> bits;
};
struct StateEntry { std::string name; Summary value; };
struct CaseReport { std::string name; std::vector<StateEntry> state; Summary output; };

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: vne_res_net_block_harness [--benchmark]"
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
        throw std::runtime_error("unexpected ResNet tensor dtype");
    }
    Summary result;
    result.shape = value.sizes().vec();
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

torch::Tensor input(const std::int64_t batch, const std::int64_t channels,
                    const std::int64_t height, const std::int64_t width) {
    return torch::arange(batch * channels * height * width, torch::kFloat32)
        .reshape({batch, channels, height, width})
        .remainder(37.0)
        .sub(18.0)
        .div(11.0);
}

std::vector<StateEntry> state(nn::ResNetBlock& block) {
    std::vector<StateEntry> result;
    for (const auto& item : block->named_parameters(true)) {
        result.push_back(StateEntry{item.key(), summarize(item.value())});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.name < right.name;
    });
    return result;
}

CaseReport run_case(const std::string& name, const std::int64_t channels,
                    const std::int64_t batch, const std::int64_t height,
                    const std::int64_t width) {
    torch::manual_seed(name == "single" ? 31 : name == "rgb" ? 37 : 41);
    nn::ResNetBlock block(nn::ResNetBlockOptions{channels});
    block->eval();
    return CaseReport{name, state(block), summarize(block->forward(
        input(batch, channels, height, width)))};
}

struct BenchmarkReport { std::vector<std::int64_t> samples_ns; std::uint64_t checksum = 0U; };

BenchmarkReport benchmark(const Arguments& arguments) {
    torch::manual_seed(101);
    nn::ResNetBlock block(nn::ResNetBlockOptions{64});
    block->eval();
    const auto value = input(16, 64, 16, 16);
    BenchmarkReport result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        result.checksum = summarize(block->forward(value), false).checksum;
    }
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto output = block->forward(value);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        result.samples_ns.push_back(elapsed);
        if (summarize(output, false).checksum != result.checksum) {
            throw std::runtime_error("ResNet block checksum drift");
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
    std::cout << "],\"dtype\":\"float32\",\"checksum\":\""
              << value.checksum << "\",\"bits\":[";
    for (std::size_t index = 0; index < value.bits.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << value.bits[index] << '"';
    }
    std::cout << "]}";
}

void emit_report(const Arguments& arguments, const std::vector<CaseReport>& cases,
                 const BenchmarkReport* benchmark_report) {
    std::cout << "{\"component\":\"res_net_block\",\"cases\":[";
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
        std::cout << ",\"benchmark\":{\"name\":\"batch16_channels64_16x16\""
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
            run_case("single", 1, 2, 3, 4),
            run_case("rgb", 3, 2, 4, 5),
            run_case("wide", 8, 1, 2, 2),
        };
        BenchmarkReport benchmark_report;
        if (arguments.benchmark) benchmark_report = benchmark(arguments);
        emit_report(arguments, cases, arguments.benchmark ? &benchmark_report : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "res_net_block_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
