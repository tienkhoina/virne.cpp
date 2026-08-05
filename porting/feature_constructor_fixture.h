#pragma once

#include "../virne/solver/learning/rl_core/feature_constructor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace feature_constructor_fixture {

namespace attribute = virne::network::attribute;
namespace network = virne::network;
namespace feature = virne::solver::learning::rl_core;

inline attribute::AttributeFactorySpec resource_spec(
    std::string name,
    attribute::AttributeOwner owner) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    return result;
}

inline attribute::AttributeFactorySpec extrema_spec(
    std::string name,
    attribute::AttributeOwner owner,
    std::string originator) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::extrema;
    result.originator_name = std::move(originator);
    return result;
}

inline attribute::AttributeRegistryId node_id(
    const network::BaseNetwork& value,
    std::string_view name) {
    const auto binding = value.bind_node_attribute(name);
    if (!binding.has_value()) {
        throw std::runtime_error("feature fixture node binding is missing");
    }
    return binding->registry_id;
}

inline attribute::AttributeRegistryId link_id(
    const network::BaseNetwork& value,
    std::string_view name) {
    const auto binding = value.bind_link_attribute(name);
    if (!binding.has_value()) {
        throw std::runtime_error("feature fixture link binding is missing");
    }
    return binding->registry_id;
}

inline network::NodeAttributeDataUpdate dense_node(
    attribute::AttributeRegistryId id,
    const std::vector<std::int64_t>& source) {
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values.reserve(source.size());
    for (const std::int64_t value : source) {
        result.dense_values.emplace_back(value);
    }
    return result;
}

inline network::LinkAttributeDataUpdate sparse_link(
    attribute::AttributeRegistryId id,
    const std::vector<EdgeEndpoints>& edges,
    const std::vector<std::int64_t>& values) {
    if (edges.size() != values.size()) {
        throw std::runtime_error("feature fixture link shape mismatch");
    }
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values.reserve(edges.size());
    for (std::size_t index = 0U; index < edges.size(); ++index) {
        result.sparse_values.push_back(
            {edges[index].first, edges[index].second, values[index]});
    }
    return result;
}

inline network::BaseNetworkConstruction physical_construction() {
    const std::vector<EdgeEndpoints> edges{
        {0U, 1U}, {1U, 2U}, {2U, 3U}, {3U, 4U},
        {4U, 5U}, {5U, 0U}, {1U, 4U}, {2U, 5U},
    };
    network::BaseNetworkConstruction result;
    result.incoming_graph.emplace(6U, edges);
    result.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
        resource_spec("ram", attribute::AttributeOwner::node),
        extrema_spec("max_cpu", attribute::AttributeOwner::node, "cpu"),
        extrema_spec("max_gpu", attribute::AttributeOwner::node, "gpu"),
        extrema_spec("max_ram", attribute::AttributeOwner::node, "ram"),
    };
    result.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("latency", attribute::AttributeOwner::link),
        extrema_spec("max_bw", attribute::AttributeOwner::link, "bw"),
        extrema_spec(
            "max_latency", attribute::AttributeOwner::link, "latency"),
    };
    return result;
}

inline network::PhysicalNetwork physical_network() {
    network::PhysicalNetwork result(physical_construction());
    const std::vector<std::int64_t> cpu{100, 80, 60, 90, 70, 50};
    const std::vector<std::int64_t> gpu{40, 60, 20, 80, 50, 30};
    const std::vector<std::int64_t> ram{256, 128, 192, 160, 224, 96};
    result.set_node_attrs_data({
        dense_node(node_id(result, "cpu"), cpu),
        dense_node(node_id(result, "gpu"), gpu),
        dense_node(node_id(result, "ram"), ram),
        dense_node(node_id(result, "max_cpu"), cpu),
        dense_node(node_id(result, "max_gpu"), gpu),
        dense_node(node_id(result, "max_ram"), ram),
    });
    const std::vector<EdgeEndpoints> edges{
        {0U, 1U}, {1U, 2U}, {2U, 3U}, {3U, 4U},
        {4U, 5U}, {5U, 0U}, {1U, 4U}, {2U, 5U},
    };
    const std::vector<std::int64_t> bw{100, 80, 60, 90, 70, 50, 75, 65};
    const std::vector<std::int64_t> latency{8, 16, 24, 12, 20, 28, 10, 18};
    result.set_link_attrs_data({
        sparse_link(link_id(result, "bw"), edges, bw),
        sparse_link(link_id(result, "latency"), edges, latency),
        sparse_link(link_id(result, "max_bw"), edges, bw),
        sparse_link(link_id(result, "max_latency"), edges, latency),
    });
    return result;
}

inline network::BaseNetworkConstruction virtual_construction() {
    const std::vector<EdgeEndpoints> edges{
        {0U, 1U}, {0U, 2U}, {1U, 2U}, {2U, 3U},
    };
    network::BaseNetworkConstruction result;
    result.incoming_graph.emplace(4U, edges);
    result.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
        resource_spec("ram", attribute::AttributeOwner::node),
    };
    result.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("latency", attribute::AttributeOwner::link),
    };
    return result;
}

inline network::VirtualNetwork virtual_network() {
    network::VirtualNetwork result(virtual_construction());
    // Solution::from_v_net mirrors Python's request metadata contract.
    result.set_request_id(0);
    result.set_arrival_time(0.0);
    result.set_lifetime(100.0);
    result.set_node_attrs_data({
        dense_node(node_id(result, "cpu"), {20, 35, 15, 25}),
        dense_node(node_id(result, "gpu"), {5, 10, 15, 20}),
        dense_node(node_id(result, "ram"), {32, 48, 64, 40}),
    });
    const std::vector<EdgeEndpoints> edges{
        {0U, 1U}, {0U, 2U}, {1U, 2U}, {2U, 3U},
    };
    result.set_link_attrs_data({
        sparse_link(link_id(result, "bw"), edges, {12, 20, 16, 24}),
        sparse_link(link_id(result, "latency"), edges, {4, 6, 8, 10}),
    });
    return result;
}

inline network::TopologicalMetrics fixed_metrics(std::size_t nodes) {
    network::TopologicalMetrics result;
    result.node_degree_centrality.emplace();
    result.node_closeness_centrality.emplace();
    result.node_eigenvector_centrality.emplace();
    result.node_betweenness_centrality.emplace();
    for (std::size_t node = 0U; node < nodes; ++node) {
        const float value = static_cast<float>(node);
        const float reverse = static_cast<float>(nodes - 1U - node);
        result.node_degree_centrality->values.push_back(value / 8.0F);
        result.node_closeness_centrality->values.push_back(reverse / 8.0F);
        result.node_eigenvector_centrality->values.push_back(
            static_cast<float>((node * 3U) % 8U) / 8.0F);
        result.node_betweenness_centrality->values.push_back(
            static_cast<float>((node * node) % 8U) / 8.0F);
    }
    return result;
}

inline attribute::AttributeBenchmarks benchmarks(
    const network::PhysicalNetwork& p_net,
    std::size_t workers = 1U) {
    network::BaseNetworkBenchmarkSelection selection;
    selection.workers = workers;
    return network::get_attribute_benchmarks(p_net, selection);
}

inline feature::FeatureConstructorPreparedData prepared(
    const network::PhysicalNetwork& p_net,
    const network::VirtualNetwork& v_net,
    std::size_t workers = 1U) {
    feature::FeatureConstructorPreparedData result;
    result.p_net_attribute_benchmarks = benchmarks(p_net, workers);
    result.p_net_topological_metrics = fixed_metrics(p_net.num_nodes());
    result.v_net_topological_metrics = fixed_metrics(v_net.num_nodes());
    return result;
}

inline virne::core::Solution solution_with_slots(
    const network::VirtualNetwork& v_net,
    bool populated = true) {
    auto result = virne::core::Solution::from_v_net(v_net);
    if (populated) {
        result.node_slots.insert_or_assign(0, 2);
        result.node_slots.insert_or_assign(1, 4);
    }
    return result;
}

inline feature::FeatureConstructorOptions options(
    feature::FeatureConstructorKind kind,
    bool status = true,
    bool aggregate = true,
    bool degree = true,
    bool more_topology = true,
    std::size_t workers = 1U) {
    feature::FeatureConstructorOptions result;
    result.kind = kind;
    result.use_node_status_flags = status;
    result.use_aggregated_link_attrs = aggregate;
    result.use_degree_metric = degree;
    result.use_more_topological_metrics = more_topology;
    result.workers = workers;
    return result;
}

}  // namespace feature_constructor_fixture
