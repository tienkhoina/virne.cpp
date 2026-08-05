#pragma once

#include "../obs_handler.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace virne::solver::learning::rl_core {

enum class FeatureConstructorKind : std::uint8_t {
    p_net,
    p_net_v_node,
    p_net_v_net,
};

// Dynamic configuration strings terminate here; repeated construction stores
// and switches on this compact fixed ID only.
FeatureConstructorKind feature_constructor_kind_from_string(
    std::string_view name);
std::string_view feature_constructor_kind_name(
    FeatureConstructorKind kind) noexcept;

struct FeatureConstructorOptions {
    FeatureConstructorKind kind = FeatureConstructorKind::p_net_v_node;
    ObservationHandler::AttributeKinds extracted_attr_types{
        network::attribute::AttributeKind::resource};
    bool use_node_status_flags = true;
    bool use_aggregated_link_attrs = true;
    bool use_degree_metric = true;
    bool use_more_topological_metrics = true;
    std::size_t workers = 1U;
};

struct FeatureConstructorPreparedData {
    std::optional<network::attribute::AttributeBenchmarks>
        p_net_attribute_benchmarks;
    std::optional<network::TopologicalMetrics> p_net_topological_metrics;
    std::optional<network::TopologicalMetrics> v_net_topological_metrics;
};

struct NetworkFeatures {
    DenseMatrixF32 x;
    DenseMatrixI64 edge_index;
    DenseMatrixF32 edge_attr;
};

struct FeatureObservation {
    NetworkFeatures p_net;
    std::optional<DenseVectorF32> v_node_x;
    std::optional<NetworkFeatures> v_net;
};

class FeatureConstructor {
public:
    FeatureConstructor(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        FeatureConstructorOptions options = {},
        FeatureConstructorPreparedData prepared = {});
    ~FeatureConstructor();

    FeatureConstructor(const FeatureConstructor&) = delete;
    FeatureConstructor& operator=(const FeatureConstructor&) = delete;
    FeatureConstructor(FeatureConstructor&&) noexcept;
    FeatureConstructor& operator=(FeatureConstructor&&) noexcept;

    const FeatureConstructorOptions& options() const noexcept;

    FeatureObservation construct(
        const core::Solution& solution,
        std::size_t curr_v_node_id) const;

    // Mirrors Python's protected leaf and preserves its terminal empty-vector
    // behavior without forcing construction of the physical observation.
    DenseVectorF32 construct_v_node_features(
        const core::Solution& solution,
        std::size_t curr_v_node_id) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace virne::solver::learning::rl_core
