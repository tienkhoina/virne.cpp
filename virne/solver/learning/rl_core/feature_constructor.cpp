#include "feature_constructor.h"

#include "../obs_handler_detail.h"

#include <memory>
#include <string>
#include <utility>

namespace virne::solver::learning::rl_core {
namespace {

[[noreturn]] void throw_feature(
    FeatureConstructorErrorCode code,
    std::string message) {
    throw FeatureConstructorException(code, std::move(message));
}

}  // namespace

using namespace detail;

FeatureConstructorKind feature_constructor_kind_from_string(
    std::string_view name) {
    if (name == "p_net") {
        return FeatureConstructorKind::p_net;
    }
    if (name == "p_net_v_node") {
        return FeatureConstructorKind::p_net_v_node;
    }
    if (name == "p_net_v_net") {
        return FeatureConstructorKind::p_net_v_net;
    }
    throw_feature(
        FeatureConstructorErrorCode::invalid_kind,
        "Unknown feature-constructor kind.");
}

std::string_view feature_constructor_kind_name(
    FeatureConstructorKind kind) noexcept {
    switch (kind) {
        case FeatureConstructorKind::p_net: return "p_net";
        case FeatureConstructorKind::p_net_v_node: return "p_net_v_node";
        case FeatureConstructorKind::p_net_v_net: return "p_net_v_net";
    }
    return "invalid";
}

class FeatureConstructor::Impl {
public:
    Impl(
        const network::PhysicalNetwork& physical,
        const network::VirtualNetwork& virtual_network,
        FeatureConstructorOptions input_options,
        FeatureConstructorPreparedData prepared)
        : p_net(physical),
          v_net(virtual_network),
          options_(std::move(input_options)) {
        require_workers(options_.workers);

        p_node_ids = select_node_ids(p_net, options_.extracted_attr_types);
        p_link_ids = select_link_ids(p_net, options_.extracted_attr_types);
        v_node_ids = select_node_ids(v_net, options_.extracted_attr_types);
        v_link_ids = select_link_ids(v_net, options_.extracted_attr_types);
        v_node_value_ids = resolve_node_value_ids(v_net, v_node_ids);
        v_link_value_ids = resolve_link_value_ids(v_net, v_link_ids);

        if (prepared.p_net_attribute_benchmarks.has_value()) {
            benchmarks = std::move(*prepared.p_net_attribute_benchmarks);
        } else {
            network::BaseNetworkBenchmarkSelection selection;
            selection.workers = options_.workers;
            benchmarks = network::get_attribute_benchmarks(p_net, selection);
        }
        if (!benchmarks.node_attr_benchmarks.has_value() ||
            !benchmarks.link_attr_benchmarks.has_value() ||
            !benchmarks.link_sum_attr_benchmarks.has_value()) {
            throw_feature(
                FeatureConstructorErrorCode::missing_benchmark,
                "Physical-network feature benchmarks are incomplete.");
        }
        p_node_benchmarks = resolve_node_benchmarks(
            p_net,
            p_node_ids,
            *benchmarks.node_attr_benchmarks);
        p_link_benchmarks = resolve_link_benchmarks(
            p_net,
            p_link_ids,
            *benchmarks.link_attr_benchmarks);
        p_link_sum_benchmarks = resolve_link_benchmarks(
            p_net,
            p_link_ids,
            *benchmarks.link_sum_attr_benchmarks);
        v_node_benchmarks = resolve_node_benchmarks(
            v_net,
            v_node_ids,
            *benchmarks.node_attr_benchmarks);
        v_link_benchmarks = resolve_link_benchmarks(
            v_net,
            v_link_ids,
            *benchmarks.link_attr_benchmarks);
        v_link_sum_benchmarks = resolve_link_benchmarks(
            v_net,
            v_link_ids,
            *benchmarks.link_sum_attr_benchmarks);

        network::TopologicalMetricOptions metric_options;
        metric_options.degree = options_.use_degree_metric;
        metric_options.closeness = options_.use_more_topological_metrics;
        metric_options.eigenvector = options_.use_more_topological_metrics;
        metric_options.betweenness = options_.use_more_topological_metrics;
        metric_options.worker_count = options_.workers;
        if (prepared.p_net_topological_metrics.has_value()) {
            p_metrics = std::move(*prepared.p_net_topological_metrics);
        } else {
            p_metrics = network::TopologicalMetricCalculator::calculate(
                p_net.graph(), metric_options);
        }
        if (prepared.v_net_topological_metrics.has_value()) {
            v_metrics = std::move(*prepared.v_net_topological_metrics);
        } else {
            v_metrics = network::TopologicalMetricCalculator::calculate(
                v_net.graph(), metric_options);
        }

        p_edge_index = ObservationHandler::get_link_index_obs(p_net);
        v_edge_index = ObservationHandler::get_link_index_obs(v_net);
    }

    const FeatureConstructorOptions& options() const noexcept {
        return options_;
    }

    DenseVectorF32 construct_v_node(
        const core::Solution& solution,
        std::size_t current) const {
        (void)solution;
        if (current >= v_net.num_nodes()) {
            return {};
        }
        const DenseVectorF32 demand = v_node_demand_by_id(
            v_net,
            current,
            v_node_ids,
            v_node_value_ids,
            &v_node_benchmarks);
        const DenseVectorF32 status = options_.use_node_status_flags
            ? ObservationHandler::get_v_node_status(
                  v_net, current, p_net.num_nodes())
            : DenseVectorF32{};
        DenseVectorF32 mean;
        DenseVectorF32 maximum;
        DenseVectorF32 minimum;
        DenseVectorF32 sum;
        if (options_.use_aggregated_link_attrs) {
            mean = v_node_aggr_by_id(
                v_net,
                current,
                LinkAggregation::mean,
                v_link_ids,
                v_link_value_ids,
                &v_link_benchmarks);
            maximum = v_node_aggr_by_id(
                v_net,
                current,
                LinkAggregation::maximum,
                v_link_ids,
                v_link_value_ids,
                &v_link_benchmarks);
            minimum = v_node_aggr_by_id(
                v_net,
                current,
                LinkAggregation::minimum,
                v_link_ids,
                v_link_value_ids,
                &v_link_benchmarks);
            sum = v_node_aggr_by_id(
                v_net,
                current,
                LinkAggregation::sum,
                v_link_ids,
                v_link_value_ids,
                &v_link_sum_benchmarks);
        }
        const DenseVectorF32 neighbors{{static_cast<float>(
            static_cast<double>(v_net.graph().degree(static_cast<Vertex>(current))) /
            static_cast<double>(v_net.num_nodes()))}};
        return concatenate_vectors({
            &demand, &status, &mean, &maximum, &minimum, &sum, &neighbors});
    }

    NetworkFeatures construct_p_net(
        const core::Solution& solution,
        std::size_t current) const {
        const std::size_t nodes = p_net.num_nodes();
        const DenseMatrixF32 node_attrs = node_attrs_obs_by_id(
            p_net,
            p_node_ids,
            &p_node_benchmarks,
            options_.workers);
        const DenseMatrixF32 status = options_.use_node_status_flags
            ? ObservationHandler::get_p_net_nodes_status(
                  p_net, v_net, solution.node_slots, current)
            : empty_matrix(nodes);
        DenseMatrixF32 minimum = empty_matrix(nodes);
        DenseMatrixF32 mean = empty_matrix(nodes);
        DenseMatrixF32 maximum = empty_matrix(nodes);
        DenseMatrixF32 sum = empty_matrix(nodes);
        if (options_.use_aggregated_link_attrs) {
            minimum = link_aggr_obs_by_id(
                p_net,
                p_link_ids,
                LinkAggregation::minimum,
                &p_link_benchmarks,
                options_.workers);
            // Preserve the Python call-site behavior: mean receives the
            // link-sum keyword, which this branch ignores, so it is raw.
            mean = link_aggr_obs_by_id(
                p_net,
                p_link_ids,
                LinkAggregation::mean,
                nullptr,
                options_.workers);
            maximum = link_aggr_obs_by_id(
                p_net,
                p_link_ids,
                LinkAggregation::maximum,
                &p_link_benchmarks,
                options_.workers);
            sum = link_aggr_obs_by_id(
                p_net,
                p_link_ids,
                LinkAggregation::sum,
                &p_link_sum_benchmarks,
                options_.workers);
        }
        const DenseMatrixF32 average_distance =
            ObservationHandler::get_average_distance(
                p_net, solution.node_slots, true, options_.workers);
        const DenseMatrixF32 degree = options_.use_degree_metric
            ? ObservationHandler::get_node_topological_metrics(
                  p_net,
                  p_metrics,
                  TopologicalMetricSelection{true, false, false, false})
            : empty_matrix(nodes);
        const DenseMatrixF32 more = options_.use_more_topological_metrics
            ? ObservationHandler::get_node_topological_metrics(
                  p_net,
                  p_metrics,
                  TopologicalMetricSelection{false, true, true, true})
            : empty_matrix(nodes);
        NetworkFeatures result;
        result.x = concatenate_columns(nodes, {
            &node_attrs,
            &status,
            &minimum,
            &mean,
            &maximum,
            &sum,
            &average_distance,
            &degree,
            &more,
        });
        result.edge_index = p_edge_index;
        result.edge_attr = link_attrs_obs_by_id(
            p_net,
            p_link_ids,
            &p_link_benchmarks,
            options_.workers);
        return result;
    }

    NetworkFeatures construct_v_net(
        const core::Solution& solution,
        std::size_t current) const {
        const std::size_t nodes = v_net.num_nodes();
        if (current >= nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_current_virtual_node,
                "Virtual-network features require a valid current node.");
        }
        const DenseMatrixF32 node_attrs = node_attrs_obs_by_id(
            v_net,
            v_node_ids,
            &v_node_benchmarks,
            options_.workers);
        const DenseMatrixF32 status = options_.use_node_status_flags
            ? ObservationHandler::get_v_net_nodes_status(
                  v_net,
                  solution.node_slots,
                  current,
                  true,
                  true)
            : empty_matrix(nodes);
        DenseMatrixF32 minimum = empty_matrix(nodes);
        DenseMatrixF32 maximum = empty_matrix(nodes);
        DenseMatrixF32 sum = empty_matrix(nodes);
        DenseMatrixF32 mean = empty_matrix(nodes);
        if (options_.use_aggregated_link_attrs) {
            minimum = link_aggr_obs_by_id(
                v_net,
                v_link_ids,
                LinkAggregation::minimum,
                &v_link_benchmarks,
                options_.workers);
            maximum = link_aggr_obs_by_id(
                v_net,
                v_link_ids,
                LinkAggregation::maximum,
                &v_link_benchmarks,
                options_.workers);
            sum = link_aggr_obs_by_id(
                v_net,
                v_link_ids,
                LinkAggregation::sum,
                &v_link_sum_benchmarks,
                options_.workers);
            // Same Python keyword mismatch as the original implementation.
            mean = link_aggr_obs_by_id(
                v_net,
                v_link_ids,
                LinkAggregation::mean,
                nullptr,
                options_.workers);
        }
        const float current_degree = static_cast<float>(
            static_cast<double>(v_net.graph().degree(static_cast<Vertex>(current))) /
            static_cast<double>(nodes));
        const DenseMatrixF32 neighbors = constant_column(nodes, current_degree);
        const DenseMatrixF32 degree = options_.use_degree_metric
            ? ObservationHandler::get_node_topological_metrics(
                  v_net,
                  v_metrics,
                  TopologicalMetricSelection{true, false, false, false})
            : empty_matrix(nodes);
        const DenseMatrixF32 more = options_.use_more_topological_metrics
            ? ObservationHandler::get_node_topological_metrics(
                  v_net,
                  v_metrics,
                  TopologicalMetricSelection{false, true, true, true})
            : empty_matrix(nodes);
        NetworkFeatures result;
        result.x = concatenate_columns(nodes, {
            &node_attrs,
            &status,
            &minimum,
            &maximum,
            &sum,
            &mean,
            &neighbors,
            &degree,
            &more,
        });
        result.edge_index = v_edge_index;
        result.edge_attr = link_attrs_obs_by_id(
            v_net,
            v_link_ids,
            &v_link_benchmarks,
            options_.workers);
        return result;
    }

    FeatureObservation construct(
        const core::Solution& solution,
        std::size_t current) const {
        FeatureObservation result;
        switch (options_.kind) {
            case FeatureConstructorKind::p_net:
                result.p_net = construct_p_net(solution, current);
                break;
            case FeatureConstructorKind::p_net_v_node:
                result.v_node_x = construct_v_node(solution, current);
                result.p_net = construct_p_net(solution, current);
                break;
            case FeatureConstructorKind::p_net_v_net:
                result.v_net = construct_v_net(solution, current);
                result.p_net = construct_p_net(solution, current);
                break;
            default:
                throw_feature(
                    FeatureConstructorErrorCode::invalid_kind,
                    "Invalid feature-constructor kind.");
        }
        return result;
    }

private:
    const network::PhysicalNetwork& p_net;
    const network::VirtualNetwork& v_net;
    FeatureConstructorOptions options_;
    network::attribute::AttributeBenchmarks benchmarks;
    network::TopologicalMetrics p_metrics;
    network::TopologicalMetrics v_metrics;
    std::vector<AttributeRegistryId> p_node_ids;
    std::vector<AttributeRegistryId> p_link_ids;
    std::vector<AttributeRegistryId> v_node_ids;
    std::vector<AttributeRegistryId> v_link_ids;
    std::vector<AttrId> v_node_value_ids;
    std::vector<AttrId> v_link_value_ids;
    std::vector<double> p_node_benchmarks;
    std::vector<double> p_link_benchmarks;
    std::vector<double> p_link_sum_benchmarks;
    std::vector<double> v_node_benchmarks;
    std::vector<double> v_link_benchmarks;
    std::vector<double> v_link_sum_benchmarks;
    DenseMatrixI64 p_edge_index;
    DenseMatrixI64 v_edge_index;
};

FeatureConstructor::FeatureConstructor(
    const network::PhysicalNetwork& p_net,
    const network::VirtualNetwork& v_net,
    FeatureConstructorOptions options,
    FeatureConstructorPreparedData prepared)
    : impl_(std::make_unique<Impl>(
          p_net, v_net, std::move(options), std::move(prepared))) {}

FeatureConstructor::~FeatureConstructor() = default;
FeatureConstructor::FeatureConstructor(FeatureConstructor&&) noexcept = default;
FeatureConstructor& FeatureConstructor::operator=(
    FeatureConstructor&&) noexcept = default;

const FeatureConstructorOptions& FeatureConstructor::options() const noexcept {
    return impl_->options();
}

FeatureObservation FeatureConstructor::construct(
    const core::Solution& solution,
    std::size_t curr_v_node_id) const {
    return impl_->construct(solution, curr_v_node_id);
}

DenseVectorF32 FeatureConstructor::construct_v_node_features(
    const core::Solution& solution,
    std::size_t curr_v_node_id) const {
    return impl_->construct_v_node(solution, curr_v_node_id);
}

}  // namespace virne::solver::learning::rl_core


