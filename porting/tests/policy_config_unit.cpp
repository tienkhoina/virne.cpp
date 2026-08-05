#include "../../virne/solver/learning/rl_core/policy_config.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace rl = virne::solver::learning::rl_core;

namespace {

static_assert(std::is_trivially_copyable_v<rl::PolicyFeatureFlags>);
static_assert(std::is_trivially_copyable_v<rl::PolicyFeatureDimensions>);
static_assert(std::is_trivially_copyable_v<rl::PolicyGeneralNetworkConfig>);
static_assert(std::is_trivially_copyable_v<rl::PolicyConfiguration>);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

virne::utils::FeatureConstructorSummary make_summary(
    const std::size_t p_node_attrs,
    const std::size_t p_link_attrs,
    const std::size_t v_node_attrs,
    const std::size_t v_link_attrs,
    const std::int64_t p_nodes = 500) {
    virne::utils::FeatureConstructorSummary summary;
    summary.num_extracted_p_node_attrs = p_node_attrs;
    summary.num_extracted_p_link_attrs = p_link_attrs;
    summary.num_extracted_v_node_attrs = v_node_attrs;
    summary.num_extracted_v_link_attrs = v_link_attrs;
    summary.p_num_nodes = p_nodes;
    return summary;
}

void require_dimensions(
    const rl::PolicyFeatureDimensions& actual,
    const std::int64_t p_net_x,
    const std::int64_t p_net_edge,
    const std::int64_t v_net_x,
    const std::int64_t v_net_edge,
    const std::int64_t v_node_x,
    const std::int64_t p_nodes,
    const std::string& label) {
    require(actual.p_net_x_dim == p_net_x, label + " p_net_x_dim");
    require(actual.p_net_edge_dim == p_net_edge,
            label + " p_net_edge_dim");
    require(actual.v_net_x_dim == v_net_x, label + " v_net_x_dim");
    require(actual.v_net_edge_dim == v_net_edge,
            label + " v_net_edge_dim");
    require(actual.v_node_x_dim == v_node_x, label + " v_node_x_dim");
    require(actual.p_net_num_nodes == p_nodes,
            label + " p_net_num_nodes");
}

void test_multi_resource_defaults() {
    const auto actual = rl::derive_policy_feature_dimensions(
        make_summary(3U, 1U, 3U, 1U));
    require_dimensions(actual, 14, 1, 15, 1, 11, 500,
                       "multi-resource defaults");
}

void test_all_flag_combinations() {
    const auto summary = make_summary(2U, 3U, 5U, 2U, 17);
    for (unsigned mask = 0U; mask < 16U; ++mask) {
        rl::PolicyFeatureFlags flags;
        flags.use_node_status_flags = (mask & 1U) != 0U;
        flags.use_aggregated_link_attrs = (mask & 2U) != 0U;
        flags.use_degree_metric = (mask & 4U) != 0U;
        flags.use_more_topological_metrics = (mask & 8U) != 0U;

        const auto actual =
            rl::derive_policy_feature_dimensions(summary, flags);
        const std::int64_t p_net_x = 2 + 1
            + (flags.use_node_status_flags ? 2 : 0)
            + (flags.use_aggregated_link_attrs ? 3 * 4 : 0)
            + (flags.use_degree_metric ? 1 : 0)
            + (flags.use_more_topological_metrics ? 3 : 0);
        const std::int64_t v_node_x = 5 + 1
            + (flags.use_node_status_flags ? 3 : 0)
            + (flags.use_aggregated_link_attrs ? 2 * 4 : 0);
        const std::int64_t v_net_x = v_node_x
            + (flags.use_degree_metric ? 1 : 0)
            + (flags.use_more_topological_metrics ? 3 : 0);
        require_dimensions(
            actual, p_net_x, 3, v_net_x, 2, v_node_x, 17,
            "flag combination " + std::to_string(mask));
    }
}

void test_zero_attributes_and_general_config() {
    rl::PolicyFeatureFlags flags;
    flags.use_node_status_flags = false;
    flags.use_aggregated_link_attrs = false;
    flags.use_degree_metric = false;
    flags.use_more_topological_metrics = false;

    rl::PolicyGeneralNetworkConfig network;
    network.embedding_dim = 256;
    network.dropout_prob = 0.25;
    network.batch_norm = true;
    const auto config = rl::derive_policy_configuration(
        make_summary(0U, 0U, 0U, 0U, 1), flags, network);
    require_dimensions(
        config.feature_dimensions, 1, 0, 1, 0, 1, 1,
        "zero attributes");
    require(config.neural_network.embedding_dim == 256,
            "embedding dimension preserved");
    require(config.neural_network.dropout_prob == 0.25,
            "dropout probability preserved");
    require(config.neural_network.batch_norm,
            "batch norm preserved");
}

template <typename Function>
void require_error(
    Function&& function,
    const rl::PolicyConfigErrorCode expected,
    const std::string& label) {
    bool rejected = false;
    try {
        function();
    } catch (const rl::PolicyConfigException& error) {
        rejected = error.code() == expected;
    }
    require(rejected, label);
}

void test_invalid_counts_and_network_options() {
    require_error(
        [] {
            static_cast<void>(rl::derive_policy_feature_dimensions(
                make_summary(1U, 1U, 1U, 1U, 0)));
        },
        rl::PolicyConfigErrorCode::invalid_physical_node_count,
        "zero physical nodes rejected");
    require_error(
        [] {
            static_cast<void>(rl::derive_policy_feature_dimensions(
                make_summary(1U, 1U, 1U, 1U, -3)));
        },
        rl::PolicyConfigErrorCode::invalid_physical_node_count,
        "negative physical nodes rejected");
    require_error(
        [] {
            rl::PolicyGeneralNetworkConfig config;
            config.embedding_dim = -1;
            static_cast<void>(
                rl::validate_policy_general_network_config(config));
        },
        rl::PolicyConfigErrorCode::invalid_embedding_dimension,
        "negative embedding dimension rejected");
    require_error(
        [] {
            rl::PolicyGeneralNetworkConfig config;
            config.dropout_prob = -0.01;
            static_cast<void>(
                rl::validate_policy_general_network_config(config));
        },
        rl::PolicyConfigErrorCode::invalid_dropout_probability,
        "negative dropout rejected");
    require_error(
        [] {
            rl::PolicyGeneralNetworkConfig config;
            config.dropout_prob =
                std::numeric_limits<double>::quiet_NaN();
            static_cast<void>(
                rl::validate_policy_general_network_config(config));
        },
        rl::PolicyConfigErrorCode::invalid_dropout_probability,
        "non-finite dropout rejected");
}

void test_dimension_overflow() {
    constexpr auto maximum_size = std::numeric_limits<std::size_t>::max();
    require_error(
        [maximum_size] {
            static_cast<void>(rl::derive_policy_feature_dimensions(
                make_summary(maximum_size, 0U, 0U, 0U, 1)));
        },
        rl::PolicyConfigErrorCode::feature_dimension_overflow,
        "physical node feature overflow rejected");

    require_error(
        [maximum_size] {
            static_cast<void>(rl::derive_policy_feature_dimensions(
                make_summary(0U, maximum_size, 0U, 0U, 1)));
        },
        rl::PolicyConfigErrorCode::feature_dimension_overflow,
        "physical link aggregation overflow rejected");

    rl::PolicyFeatureFlags flags;
    flags.use_node_status_flags = false;
    flags.use_aggregated_link_attrs = false;
    flags.use_degree_metric = false;
    flags.use_more_topological_metrics = false;
    require_error(
        [maximum_size, flags] {
            static_cast<void>(rl::derive_policy_feature_dimensions(
                make_summary(0U, maximum_size, 0U, 0U, 1), flags));
        },
        rl::PolicyConfigErrorCode::feature_dimension_overflow,
        "edge dimension outside int64 rejected without aggregation");
}

}  // namespace

int main() {
    try {
        test_multi_resource_defaults();
        test_all_flag_combinations();
        test_zero_attributes_and_general_config();
        test_invalid_counts_and_network_options();
        test_dimension_overflow();
        std::cout << "policy_config_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "policy_config_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
