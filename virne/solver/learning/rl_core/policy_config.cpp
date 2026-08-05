#include "policy_config.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

[[noreturn]] void throw_dimension_overflow() {
    throw PolicyConfigException(
        PolicyConfigErrorCode::feature_dimension_overflow,
        "policy feature dimension exceeds the int64 LibTorch shape domain");
}

std::int64_t checked_count(const std::size_t value) {
    constexpr auto maximum =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
    if (static_cast<std::uint64_t>(value) > maximum) {
        throw_dimension_overflow();
    }
    return static_cast<std::int64_t>(value);
}

std::int64_t checked_add(
    const std::int64_t left,
    const std::int64_t right) {
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (right < 0 || left < 0 || left > maximum - right) {
        throw_dimension_overflow();
    }
    return left + right;
}

std::int64_t checked_multiply(
    const std::int64_t value,
    const std::int64_t multiplier) {
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (value < 0 || multiplier < 0
        || (multiplier != 0 && value > maximum / multiplier)) {
        throw_dimension_overflow();
    }
    return value * multiplier;
}

}  // namespace

PolicyConfigException::PolicyConfigException(
    const PolicyConfigErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

PolicyFeatureDimensions derive_policy_feature_dimensions(
    const utils::FeatureConstructorSummary& summary,
    const PolicyFeatureFlags flags) {
    if (summary.p_num_nodes <= 0) {
        throw PolicyConfigException(
            PolicyConfigErrorCode::invalid_physical_node_count,
            "policy physical node count must be positive");
    }

    const auto p_node_attrs =
        checked_count(summary.num_extracted_p_node_attrs);
    const auto p_link_attrs =
        checked_count(summary.num_extracted_p_link_attrs);
    const auto v_node_attrs =
        checked_count(summary.num_extracted_v_node_attrs);
    const auto v_link_attrs =
        checked_count(summary.num_extracted_v_link_attrs);

    PolicyFeatureDimensions dimensions;
    dimensions.p_net_x_dim = checked_add(p_node_attrs, 1);
    if (flags.use_node_status_flags) {
        dimensions.p_net_x_dim = checked_add(dimensions.p_net_x_dim, 2);
    }
    if (flags.use_aggregated_link_attrs) {
        dimensions.p_net_x_dim = checked_add(
            dimensions.p_net_x_dim,
            checked_multiply(p_link_attrs, 4));
    }
    if (flags.use_degree_metric) {
        dimensions.p_net_x_dim = checked_add(dimensions.p_net_x_dim, 1);
    }
    if (flags.use_more_topological_metrics) {
        dimensions.p_net_x_dim = checked_add(dimensions.p_net_x_dim, 3);
    }

    dimensions.p_net_edge_dim = p_link_attrs;

    dimensions.v_node_x_dim = checked_add(v_node_attrs, 1);
    if (flags.use_node_status_flags) {
        dimensions.v_node_x_dim = checked_add(dimensions.v_node_x_dim, 3);
    }
    if (flags.use_aggregated_link_attrs) {
        dimensions.v_node_x_dim = checked_add(
            dimensions.v_node_x_dim,
            checked_multiply(v_link_attrs, 4));
    }
    // Python reads both topology flags in get_v_node_x_dim but deliberately
    // adds zero for them.  Preserve that observable compatibility quirk.

    dimensions.v_net_x_dim = checked_add(v_node_attrs, 1);
    if (flags.use_node_status_flags) {
        dimensions.v_net_x_dim = checked_add(dimensions.v_net_x_dim, 3);
    }
    if (flags.use_aggregated_link_attrs) {
        dimensions.v_net_x_dim = checked_add(
            dimensions.v_net_x_dim,
            checked_multiply(v_link_attrs, 4));
    }
    if (flags.use_degree_metric) {
        dimensions.v_net_x_dim = checked_add(dimensions.v_net_x_dim, 1);
    }
    if (flags.use_more_topological_metrics) {
        dimensions.v_net_x_dim = checked_add(dimensions.v_net_x_dim, 3);
    }

    dimensions.v_net_edge_dim = v_link_attrs;
    dimensions.p_net_num_nodes = summary.p_num_nodes;
    return dimensions;
}

PolicyGeneralNetworkConfig validate_policy_general_network_config(
    PolicyGeneralNetworkConfig config) {
    if (config.embedding_dim <= 0) {
        throw PolicyConfigException(
            PolicyConfigErrorCode::invalid_embedding_dimension,
            "policy embedding dimension must be positive");
    }
    if (!std::isfinite(config.dropout_prob)
        || config.dropout_prob < 0.0
        || config.dropout_prob > 1.0) {
        throw PolicyConfigException(
            PolicyConfigErrorCode::invalid_dropout_probability,
            "policy dropout probability must be finite and within [0, 1]");
    }
    return config;
}

PolicyConfiguration derive_policy_configuration(
    const utils::FeatureConstructorSummary& summary,
    const PolicyFeatureFlags flags,
    PolicyGeneralNetworkConfig neural_network) {
    PolicyConfiguration config;
    config.feature_dimensions =
        derive_policy_feature_dimensions(summary, flags);
    config.neural_network =
        validate_policy_general_network_config(std::move(neural_network));
    return config;
}

}  // namespace virne::solver::learning::rl_core
