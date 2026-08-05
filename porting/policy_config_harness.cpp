#include "../virne/solver/learning/rl_core/policy_config.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rl = virne::solver::learning::rl_core;
namespace utils = virne::utils;

namespace {

constexpr std::size_t benchmark_configuration_count = 65536U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    int warmups = 1;
    int repetitions = 3;
};

struct PreparedConfiguration {
    utils::FeatureConstructorSummary summary;
    rl::PolicyFeatureFlags flags;
    rl::PolicyGeneralNetworkConfig network;
};

struct ExactFixture {
    std::string name;
    PreparedConfiguration input;
};

struct ExactResult {
    std::string name;
    rl::PolicyFeatureDimensions helper_dimensions;
    rl::PolicyConfiguration configuration;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    std::uint64_t checksum = fnv_offset;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message
        + "\nusage: vne_policy_config_harness [--benchmark]"
          " [--warmups N] [--repetitions N]");
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
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--warmups") {
            result.warmups = read_positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = read_positive("--repetitions");
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

utils::FeatureConstructorSummary make_summary(
    const std::size_t p_node_attrs,
    const std::size_t p_link_attrs,
    const std::size_t v_node_attrs,
    const std::size_t v_link_attrs,
    const std::int64_t p_nodes) {
    utils::FeatureConstructorSummary result;
    result.num_extracted_p_node_attrs = p_node_attrs;
    result.num_extracted_p_link_attrs = p_link_attrs;
    result.num_extracted_v_node_attrs = v_node_attrs;
    result.num_extracted_v_link_attrs = v_link_attrs;
    result.p_num_nodes = p_nodes;
    return result;
}

rl::PolicyFeatureFlags flags_from_mask(const unsigned mask) {
    rl::PolicyFeatureFlags result;
    result.use_node_status_flags = (mask & 1U) != 0U;
    result.use_aggregated_link_attrs = (mask & 2U) != 0U;
    result.use_degree_metric = (mask & 4U) != 0U;
    result.use_more_topological_metrics = (mask & 8U) != 0U;
    return result;
}

std::vector<ExactFixture> exact_fixtures() {
    std::vector<ExactFixture> result;
    result.reserve(20U);

    result.push_back({
        "defaults",
        {make_summary(3U, 1U, 3U, 1U, 500), {}, {}},
    });
    for (unsigned mask = 0U; mask < 16U; ++mask) {
        std::ostringstream name;
        name << "flags_" << std::setw(2) << std::setfill('0') << mask;
        result.push_back({
            name.str(),
            {make_summary(2U, 3U, 5U, 2U, 17),
             flags_from_mask(mask), {}},
        });
    }

    rl::PolicyFeatureFlags disabled;
    disabled.use_node_status_flags = false;
    disabled.use_aggregated_link_attrs = false;
    disabled.use_degree_metric = false;
    disabled.use_more_topological_metrics = false;
    result.push_back({
        "zero_attrs",
        {make_summary(0U, 0U, 0U, 0U, 1), disabled, {}},
    });

    rl::PolicyFeatureFlags mixed;
    mixed.use_node_status_flags = true;
    mixed.use_aggregated_link_attrs = false;
    mixed.use_degree_metric = true;
    mixed.use_more_topological_metrics = false;
    result.push_back({
        "multi_attrs",
        {make_summary(7U, 4U, 11U, 5U, 1024), mixed, {}},
    });

    rl::PolicyGeneralNetworkConfig forwarded;
    forwarded.embedding_dim = 257;
    forwarded.dropout_prob = 0.375;
    forwarded.batch_norm = true;
    result.push_back({
        "general_nn_forwarding",
        {make_summary(1U, 2U, 3U, 4U, 33), {}, forwarded},
    });
    return result;
}

std::vector<ExactResult> run_exact_cases() {
    const auto fixtures = exact_fixtures();
    std::vector<ExactResult> result;
    result.reserve(fixtures.size());
    for (const auto& fixture : fixtures) {
        result.push_back({
            fixture.name,
            rl::derive_policy_feature_dimensions(
                fixture.input.summary, fixture.input.flags),
            rl::derive_policy_configuration(
                fixture.input.summary,
                fixture.input.flags,
                fixture.input.network),
        });
    }
    return result;
}

std::string double_bits(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << bits;
    return output.str();
}

void emit_dimensions(
    std::ostream& output,
    const rl::PolicyFeatureDimensions& value) {
    output << "{\"p_net_x_dim\":" << value.p_net_x_dim
           << ",\"p_net_edge_dim\":" << value.p_net_edge_dim
           << ",\"v_net_x_dim\":" << value.v_net_x_dim
           << ",\"v_net_edge_dim\":" << value.v_net_edge_dim
           << ",\"v_node_x_dim\":" << value.v_node_x_dim
           << ",\"p_net_num_nodes\":" << value.p_net_num_nodes
           << '}';
}

void emit_helpers(
    std::ostream& output,
    const rl::PolicyFeatureDimensions& value) {
    output << "{\"get_p_net_x_dim\":" << value.p_net_x_dim
           << ",\"get_p_net_edge_dim\":" << value.p_net_edge_dim
           << ",\"get_v_node_x_dim\":" << value.v_node_x_dim
           << ",\"get_v_net_x_dim\":" << value.v_net_x_dim
           << ",\"get_v_net_edge_dim\":" << value.v_net_edge_dim
           << '}';
}

void emit_general(
    std::ostream& output,
    const rl::PolicyGeneralNetworkConfig& value) {
    output << "{\"embedding_dim\":" << value.embedding_dim
           << ",\"dropout_prob_bits\":\""
           << double_bits(value.dropout_prob)
           << "\",\"batch_norm\":"
           << (value.batch_norm ? "true" : "false") << '}';
}

void mix(std::uint64_t& checksum, const std::uint64_t value) noexcept {
    checksum ^= value;
    checksum *= fnv_prime;
}

void mix_configuration(
    std::uint64_t& checksum,
    const rl::PolicyConfiguration& value) noexcept {
    const auto& dimensions = value.feature_dimensions;
    mix(checksum, static_cast<std::uint64_t>(dimensions.p_net_x_dim));
    mix(checksum, static_cast<std::uint64_t>(dimensions.p_net_edge_dim));
    mix(checksum, static_cast<std::uint64_t>(dimensions.v_net_x_dim));
    mix(checksum, static_cast<std::uint64_t>(dimensions.v_net_edge_dim));
    mix(checksum, static_cast<std::uint64_t>(dimensions.v_node_x_dim));
    mix(checksum, static_cast<std::uint64_t>(dimensions.p_net_num_nodes));
    mix(checksum,
        static_cast<std::uint64_t>(value.neural_network.embedding_dim));
    std::uint64_t dropout_bits = 0U;
    std::memcpy(
        &dropout_bits,
        &value.neural_network.dropout_prob,
        sizeof(dropout_bits));
    mix(checksum, dropout_bits);
    mix(checksum, value.neural_network.batch_norm ? 1U : 0U);
}

std::vector<PreparedConfiguration> prepare_benchmark_configurations() {
    std::vector<PreparedConfiguration> result;
    result.reserve(benchmark_configuration_count);
    for (std::size_t index = 0U;
         index < benchmark_configuration_count;
         ++index) {
        PreparedConfiguration item;
        item.summary = make_summary(
            index % 8U,
            (index / 8U) % 6U,
            (index / 48U) % 9U,
            (index / 432U) % 7U,
            1 + static_cast<std::int64_t>(index % 2048U));
        item.flags = flags_from_mask(
            static_cast<unsigned>((index / 3U) & 15U));
        item.network.embedding_dim =
            32 + static_cast<std::int64_t>(index % 8U) * 32;
        item.network.dropout_prob =
            static_cast<double>(index % 9U) * 0.125;
        item.network.batch_norm = (index & 1U) != 0U;
        result.push_back(item);
    }
    return result;
}

std::uint64_t derive_prepared_batch(
    const std::vector<PreparedConfiguration>& inputs) {
    std::uint64_t checksum = fnv_offset;
    for (const auto& input : inputs) {
        const auto configuration = rl::derive_policy_configuration(
            input.summary, input.flags, input.network);
        mix_configuration(checksum, configuration);
    }
    return checksum;
}

BenchmarkResult run_benchmark(const Arguments& arguments) {
    const auto inputs = prepare_benchmark_configurations();
    std::uint64_t expected_checksum = fnv_offset;
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        const auto checksum = derive_prepared_batch(inputs);
        if (warmup == 0) {
            expected_checksum = checksum;
        } else if (checksum != expected_checksum) {
            throw std::runtime_error("policy configuration checksum drift");
        }
    }

    BenchmarkResult result;
    result.checksum = expected_checksum;
    result.samples_ns.reserve(
        static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0;
         repetition < arguments.repetitions;
         ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto checksum = derive_prepared_batch(inputs);
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
        if (checksum != expected_checksum) {
            throw std::runtime_error("policy configuration checksum drift");
        }
        result.samples_ns.push_back(elapsed);
    }
    return result;
}

void emit_report(
    const Arguments& arguments,
    const std::vector<ExactResult>& exact,
    const BenchmarkResult* benchmark) {
    std::cout << "{\"component\":\"policy_config\",\"cases\":[";
    for (std::size_t index = 0U; index < exact.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        const auto& item = exact[index];
        std::cout << "{\"name\":\"" << item.name << "\",\"helpers\":";
        emit_helpers(std::cout, item.helper_dimensions);
        std::cout << ",\"feature_config\":";
        emit_dimensions(std::cout, item.configuration.feature_dimensions);
        std::cout << ",\"general_nn\":";
        emit_general(std::cout, item.configuration.neural_network);
        std::cout << '}';
    }
    std::cout << ']';
    if (benchmark != nullptr) {
        std::cout << ",\"benchmark\":{\"name\":\"prepared_65536\""
                  << ",\"warmups\":" << arguments.warmups
                  << ",\"repetitions\":" << arguments.repetitions
                  << ",\"configurations\":"
                  << benchmark_configuration_count
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U;
             index < benchmark->samples_ns.size();
             ++index) {
            if (index != 0U) {
                std::cout << ',';
            }
            std::cout << benchmark->samples_ns[index];
        }
        std::cout << "],\"checksum\":\"" << benchmark->checksum
                  << "\"}";
    }
    std::cout << "}\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto exact = run_exact_cases();
        BenchmarkResult benchmark;
        if (arguments.benchmark) {
            benchmark = run_benchmark(arguments);
        }
        emit_report(
            arguments,
            exact,
            arguments.benchmark ? &benchmark : nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "policy-config harness error: " << error.what() << '\n';
        return 2;
    }
}
