#pragma once

#include "rl_core/feature_types.h"

#include "../../core/solution.h"
#include "../../network/attribute/attribute_benchmark_manager.h"
#include "../../network/base_network.h"
#include "../../network/physical_network.h"
#include "../../network/topology/topological_metric_calculator.h"
#include "../../network/virtual_network.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::solver::learning::rl_core {

struct LinkResourceThreshold {
    // Cold boundary name: it is resolved to a graph-local AttrId once before
    // traversing an edge.
    std::string name;
    double minimum = 0.0;
};

struct TopologicalMetricSelection {
    bool degree = true;
    bool closeness = true;
    bool eigenvector = true;
    bool betweenness = true;
};

enum class FeatureConstructorErrorCode : std::uint8_t {
    invalid_kind,
    invalid_worker_count,
    invalid_current_virtual_node,
    missing_attribute,
    missing_benchmark,
    missing_metric,
    non_numeric_attribute,
    invalid_shape,
    disconnected_network,
    empty_incident_links,
    graph_identity_mismatch,
};

class FeatureConstructorException final : public std::runtime_error {
public:
    FeatureConstructorException(
        FeatureConstructorErrorCode code,
        std::string message);

    FeatureConstructorErrorCode code() const noexcept { return code_; }

private:
    FeatureConstructorErrorCode code_;
};

// Mirrors virne/solver/learning/obs_handler.py. These calls are stateless
// compatibility boundaries; FeatureConstructor uses the prepared ID-only
// internal layer so repeated construction performs no string lookup.
class ObservationHandler {
public:
    using AttributeKinds =
        std::vector<network::attribute::AttributeKind>;

    static DenseVectorI32 get_node_order_obs(
        const network::BaseNetwork& network);

    static DenseMatrixF32 get_node_degree_obs(
        const network::BaseNetwork& network,
        std::optional<double> degree_benchmark = std::nullopt,
        std::size_t workers = 1U);

    static DenseMatrixF32 get_node_topological_metrics(
        const network::BaseNetwork& network,
        const network::TopologicalMetrics& metrics,
        TopologicalMetricSelection selection = {});

    static DenseMatrixF32 get_node_attrs_obs(
        const network::BaseNetwork& network,
        const AttributeKinds& kinds,
        const network::attribute::AttributeBenchmarkMap* benchmarks = nullptr,
        std::size_t workers = 1U);

    static DenseMatrixF32 get_link_attrs_obs(
        const network::BaseNetwork& network,
        const AttributeKinds& kinds,
        const network::attribute::AttributeBenchmarkMap* benchmarks = nullptr,
        std::size_t workers = 1U);

    static DenseMatrixF32 get_link_aggr_attrs_obs(
        const network::BaseNetwork& network,
        const AttributeKinds& kinds,
        network::attribute::LinkAggregation aggregation,
        const network::attribute::AttributeBenchmarkMap* benchmarks = nullptr,
        std::size_t workers = 1U);

    static DenseVectorBool get_link_filters(
        const network::BaseNetwork& network,
        const std::vector<LinkResourceThreshold>& thresholds,
        std::size_t workers = 1U);

    static DenseMatrixI64 get_link_pair_obs(
        const network::BaseNetwork& network);

    static DenseMatrixI64 get_link_index_obs(
        const network::BaseNetwork& network);

    static DenseMatrixF32 get_average_distance(
        const network::BaseNetwork& network,
        const core::NodeSlots& node_slots,
        bool normalization = true,
        std::size_t workers = 1U);

    static DenseMatrixF32 get_p_net_nodes_status(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        const core::NodeSlots& node_slots,
        std::optional<std::size_t> v_node_id = std::nullopt);

    static DenseVectorF32 get_v_node_status(
        const network::VirtualNetwork& v_net,
        std::size_t v_node_id,
        std::size_t p_net_num_nodes);

    static DenseVectorF32 get_v_node_demand(
        const network::VirtualNetwork& v_net,
        std::size_t v_node_id,
        const AttributeKinds& kinds,
        const network::attribute::AttributeBenchmarkMap* benchmarks = nullptr);

    static DenseVectorF32 get_v_node_aggr_link_demands(
        const network::VirtualNetwork& v_net,
        std::size_t v_node_id,
        network::attribute::LinkAggregation aggregation,
        const AttributeKinds& kinds,
        const network::attribute::AttributeBenchmarkMap* benchmarks = nullptr);

    static DenseMatrixF32 get_v_net_nodes_status(
        const network::VirtualNetwork& v_net,
        const core::NodeSlots& node_slots,
        std::optional<std::size_t> v_node_id = std::nullopt,
        bool consist_decision = true,
        bool neighbor_flags = false);

    static DenseMatrixF32 get_v_node_neighbor_flags(
        const network::VirtualNetwork& v_net,
        const core::NodeSlots& node_slots,
        std::size_t v_node_id);
};

}  // namespace virne::solver::learning::rl_core
