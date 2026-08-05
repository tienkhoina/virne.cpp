#pragma once

#include "../../../utils/feature_constructor_summary.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace virne::solver::learning::rl_core {

enum class PolicyConfigErrorCode : std::uint8_t {
    invalid_physical_node_count,
    invalid_embedding_dimension,
    invalid_dropout_probability,
    feature_dimension_overflow,
};

class PolicyConfigException final : public std::runtime_error {
public:
    PolicyConfigException(
        PolicyConfigErrorCode code,
        std::string message);

    PolicyConfigErrorCode code() const noexcept { return code_; }

private:
    PolicyConfigErrorCode code_;
};

// These are the four fixed switches read by Python's policy dimension helpers.
// They are resolved once from configuration; no name or mapping is retained.
struct PolicyFeatureFlags {
    bool use_node_status_flags = true;
    bool use_aggregated_link_attrs = true;
    bool use_degree_metric = true;
    bool use_more_topological_metrics = true;
};

// LibTorch shape values are signed int64.  Counts from the frozen config leaf
// are checked before conversion so no size_t-to-int64 truncation is possible.
struct PolicyFeatureDimensions {
    std::int64_t p_net_x_dim = 0;
    std::int64_t p_net_edge_dim = 0;
    std::int64_t v_net_x_dim = 0;
    std::int64_t v_net_edge_dim = 0;
    std::int64_t v_node_x_dim = 0;
    std::int64_t p_net_num_nodes = 0;
};

struct PolicyGeneralNetworkConfig {
    std::int64_t embedding_dim = 128;
    double dropout_prob = 0.0;
    bool batch_norm = false;
};

struct PolicyConfiguration {
    PolicyFeatureDimensions feature_dimensions;
    PolicyGeneralNetworkConfig neural_network;
};

// Mirrors policy_builder.py's five get_*_dim helpers and
// PolicyBuilder.get_feature_dim_config.  The function is allocation-free on
// the valid path and operates only on fixed fields.
[[nodiscard]] PolicyFeatureDimensions derive_policy_feature_dimensions(
    const utils::FeatureConstructorSummary& summary,
    PolicyFeatureFlags flags = {});

// General network fields are already typed at this boundary.  Validation is
// kept separate so policy construction can reject malformed cold config before
// allocating modules.
[[nodiscard]] PolicyGeneralNetworkConfig
validate_policy_general_network_config(PolicyGeneralNetworkConfig config);

[[nodiscard]] PolicyConfiguration derive_policy_configuration(
    const utils::FeatureConstructorSummary& summary,
    PolicyFeatureFlags flags = {},
    PolicyGeneralNetworkConfig neural_network = {});

}  // namespace virne::solver::learning::rl_core
