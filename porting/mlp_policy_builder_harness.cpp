#include "../virne/solver/learning/rl_core/mlp_policy_builder.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace core = virne::solver::learning::rl_core;
namespace policy = virne::solver::learning::rl_policy;

namespace {

constexpr std::size_t benchmark_count = 65536U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    int warmups = 1;
    int repetitions = 3;
};

struct NamedOptions {
    const char* name;
    policy::MlpActorCriticOptions value;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    std::uint64_t checksum = 0U;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: vne_mlp_policy_builder_harness [--benchmark]"
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
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--warmups") {
            result.warmups = positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = positive("--repetitions");
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

core::PolicyConfiguration configuration(
    const std::int64_t p_net_x,
    const std::int64_t v_node_x,
    const std::int64_t p_nodes,
    const std::int64_t embedding,
    const double dropout,
    const bool batch_norm) {
    core::PolicyConfiguration result;
    result.feature_dimensions.p_net_x_dim = p_net_x;
    result.feature_dimensions.v_node_x_dim = v_node_x;
    result.feature_dimensions.p_net_num_nodes = p_nodes;
    result.neural_network.embedding_dim = embedding;
    result.neural_network.dropout_prob = dropout;
    result.neural_network.batch_norm = batch_norm;
    return result;
}

std::vector<NamedOptions> exact_cases() {
    std::vector<NamedOptions> result;
    result.push_back({
        "multi_resource_default",
        core::make_mlp_actor_critic_options(
            configuration(14, 11, 500, 128, 0.0, false))});
    result.push_back({
        "custom_network",
        core::make_mlp_actor_critic_options(
            configuration(7, 5, 17, 256, 0.25, true))});
    result.push_back({
        "minimal",
        core::make_mlp_actor_critic_options(
            configuration(1, 1, 1, 1, 1.0, false))});
    return result;
}

void mix(std::uint64_t& checksum, const std::uint64_t value) noexcept {
    for (unsigned byte = 0U; byte < 8U; ++byte) {
        checksum ^= (value >> (byte * 8U)) & 0xffU;
        checksum *= fnv_prime;
    }
}

void mix_options(
    std::uint64_t& checksum,
    const policy::MlpActorCriticOptions& value) noexcept {
    mix(checksum, static_cast<std::uint64_t>(value.feature_dim));
    mix(checksum, static_cast<std::uint64_t>(value.action_dim));
    mix(checksum, static_cast<std::uint64_t>(value.num_layers));
    mix(checksum, static_cast<std::uint64_t>(value.embedding_dim));
    std::uint64_t dropout_bits = 0U;
    std::memcpy(&dropout_bits, &value.dropout_prob, sizeof(dropout_bits));
    mix(checksum, dropout_bits);
    mix(checksum, value.batch_norm ? 1U : 0U);
}

std::vector<core::PolicyConfiguration> prepared_configurations() {
    std::vector<core::PolicyConfiguration> result;
    result.reserve(benchmark_count);
    for (std::size_t index = 0U; index < benchmark_count; ++index) {
        result.push_back(configuration(
            1 + static_cast<std::int64_t>(index % 24U),
            1 + static_cast<std::int64_t>((index / 3U) % 20U),
            1 + static_cast<std::int64_t>(index % 2048U),
            32 + static_cast<std::int64_t>(index % 8U) * 32,
            static_cast<double>(index % 9U) * 0.125,
            (index & 1U) != 0U));
    }
    return result;
}

std::uint64_t derive_batch(
    const std::vector<core::PolicyConfiguration>& inputs) {
    std::uint64_t checksum = fnv_offset;
    for (const auto& input : inputs) {
        mix_options(
            checksum, core::make_mlp_actor_critic_options(input));
    }
    return checksum;
}

BenchmarkResult benchmark(const Arguments& arguments) {
    const auto inputs = prepared_configurations();
    BenchmarkResult result;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        const auto checksum = derive_batch(inputs);
        if (warmup == 0) result.checksum = checksum;
        if (checksum != result.checksum) {
            throw std::runtime_error("MLP policy option checksum drift");
        }
    }
    result.samples_ns.reserve(
        static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto checksum = derive_batch(inputs);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (checksum != result.checksum) {
            throw std::runtime_error("MLP policy option checksum drift");
        }
        result.samples_ns.push_back(elapsed);
    }
    return result;
}

void emit_options(const policy::MlpActorCriticOptions& value) {
    std::uint64_t dropout_bits = 0U;
    std::memcpy(&dropout_bits, &value.dropout_prob, sizeof(dropout_bits));
    std::cout
        << "\"feature_dim\":" << value.feature_dim
        << ",\"action_dim\":" << value.action_dim
        << ",\"num_layers\":" << value.num_layers
        << ",\"embedding_dim\":" << value.embedding_dim
        << ",\"dropout_bits\":\"" << dropout_bits << '"'
        << ",\"batch_norm\":" << (value.batch_norm ? "true" : "false");
}

void emit_report(
    const Arguments& arguments,
    const std::vector<NamedOptions>& cases,
    const BenchmarkResult* timed) {
    std::cout << "{\"component\":\"mlp_policy_builder\",\"cases\":[";
    for (std::size_t index = 0U; index < cases.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << "{\"name\":\"" << cases[index].name << "\",";
        emit_options(cases[index].value);
        std::cout << '}';
    }
    std::cout << ']';
    if (timed != nullptr) {
        std::cout
            << ",\"benchmark\":{\"name\":\"prepared_options_65536\""
            << ",\"warmups\":" << arguments.warmups
            << ",\"repetitions\":" << arguments.repetitions
            << ",\"configurations\":" << benchmark_count
            << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < timed->samples_ns.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << timed->samples_ns[index];
        }
        std::cout << "],\"checksum\":\"" << timed->checksum << "\"}";
    }
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto cases = exact_cases();
        BenchmarkResult timed;
        if (arguments.benchmark) timed = benchmark(arguments);
        emit_report(arguments, cases, arguments.benchmark ? &timed : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_policy_builder_harness: " << error.what() << '\n';
        return 1;
    }
}
