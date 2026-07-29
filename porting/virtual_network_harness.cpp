#include "virtual_network.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

attribute::AttributeFactorySpec make_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const attribute::AttributeKind kind) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = kind;
    return result;
}

network::VirtualNetwork make_fixture() {
    Graph graph;
    graph.add_edge(0U, 1U);
    graph.add_edge(1U, 2U);
    graph.add_edge(2U, 3U);
    network::BaseNetworkConstruction construction;
    construction.incoming_graph = std::move(graph);
    construction.config.node_attribute_specs = {
        make_spec(
            "cpu", attribute::AttributeOwner::node,
            attribute::AttributeKind::resource),
        make_spec(
            "peak", attribute::AttributeOwner::node,
            attribute::AttributeKind::resource),
    };
    construction.config.link_attribute_specs = {
        make_spec(
            "bw", attribute::AttributeOwner::link,
            attribute::AttributeKind::resource),
        make_spec(
            "peak_bw", attribute::AttributeOwner::link,
            attribute::AttributeKind::resource),
    };
    construction.extra_graph_attributes = {
        {"id", std::int64_t{7}},
        {"arrival_time", 2.5},
        {"lifetime", std::int64_t{9}},
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto peak = result.bind_node_attribute("peak");
    const auto bw = result.bind_link_attribute("bw");
    const auto peak_bw = result.bind_link_attribute("peak_bw");
    if (!cpu || !peak || !bw || !peak_bw) {
        throw std::runtime_error("fixture binding failed");
    }
    result.set_node_attrs_data({
        {cpu->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{1}, std::int64_t{2}, std::int64_t{3}, std::int64_t{4}}},
        {peak->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {0.5, 1.5, 2.5, 3.5}},
    });
    result.set_link_attrs_data({
        {bw->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{2}, std::int64_t{3}, std::int64_t{4}}},
        {peak_bw->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {0.25, 1.25, 2.25}},
    });
    return result;
}

std::string demand_payload(
    const network::VirtualNetwork& value,
    const std::size_t workers) {
    return "node=" + double_token(value.total_node_resource_demand(workers)) +
        ";link=" + double_token(value.total_link_resource_demand(workers)) +
        ";total=" + double_token(value.total_resource_demand(workers));
}

void emit(const std::string_view name, const std::string& payload) {
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

void emit_cases() {
    network::VirtualNetwork empty;
    emit(
        "empty",
        std::string("id=") + (empty.request_id() ? "1" : "0") +
            ";arrival=" + (empty.arrival_time() ? "1" : "0") +
            ";lifetime=" + (empty.lifetime() ? "1" : "0") +
            ";" + demand_payload(empty, 1U));

    auto fixed = make_fixture();
    emit(
        "fixed_metadata",
        "id=" + std::to_string(*fixed.request_id()) +
            ";arrival=" + double_token(*fixed.arrival_time()) +
            ";lifetime=" + double_token(*fixed.lifetime()));
    fixed.set_request_id(11);
    fixed.set_arrival_time(4.25);
    fixed.set_lifetime(12.5);
    emit(
        "fixed_assignment",
        "id=" + std::to_string(*fixed.request_id()) +
            ";arrival=" + double_token(*fixed.arrival_time()) +
            ";lifetime=" + double_token(*fixed.lifetime()));

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        const auto value = make_fixture();
        emit("demand_w" + std::to_string(workers), demand_payload(value, workers));
    }

    auto cached = make_fixture();
    const double before = cached.total_resource_demand();
    const auto cpu = cached.bind_node_attribute("cpu");
    if (!cpu) {
        throw std::runtime_error("cache cpu binding failed");
    }
    cached.set_node_attrs_data({
        {cpu->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{100}, std::int64_t{100},
          std::int64_t{100}, std::int64_t{100}}},
    });
    emit(
        "cached_stale",
        "before=" + double_token(before) +
            ";node=" + double_token(cached.total_node_resource_demand()) +
            ";total=" + double_token(cached.total_resource_demand()));
    cached.invalidate_cached_total_resource_demand();
    emit("native_cache_invalidate", demand_payload(cached, 8U));

    network::VirtualNetwork no_resources;
    emit("no_resources", demand_payload(no_resources, 1U));

    Graph lane_graph;
    lane_graph.add_edge(0U, 1U);
    network::BaseNetworkConstruction lane_construction;
    lane_construction.incoming_graph = std::move(lane_graph);
    lane_construction.config.node_attribute_specs = {
        make_spec(
            "flag", attribute::AttributeOwner::node,
            attribute::AttributeKind::resource),
    };
    lane_construction.config.link_attribute_specs = {
        make_spec(
            "units", attribute::AttributeOwner::link,
            attribute::AttributeKind::resource),
    };
    network::VirtualNetwork lanes(std::move(lane_construction));
    const auto flag = lanes.bind_node_attribute("flag");
    const auto units = lanes.bind_link_attribute("units");
    if (!flag || !units) {
        throw std::runtime_error("lane binding failed");
    }
    lanes.set_node_attrs_data({
        {flag->registry_id, network::AttributeDataLayout::dense, {}, {true, false}},
    });
    lanes.set_link_attrs_data({
        {units->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{-3}}},
    });
    emit("bool_integer", demand_payload(lanes, 2U));

    auto ragged = make_fixture();
    const auto ragged_cpu = ragged.bind_node_attribute("cpu");
    if (!ragged_cpu) {
        throw std::runtime_error("ragged binding failed");
    }
    ragged.graph().node_attrs(3U).erase(ragged_cpu->value_id);
    emit(
        "ragged_zero",
        "node=" + double_token(ragged.total_node_resource_demand()) +
            ";total=" + double_token(ragged.total_resource_demand()));

    auto nonnumeric = make_fixture();
    const auto bw = nonnumeric.bind_link_attribute("bw");
    if (!bw) {
        throw std::runtime_error("nonnumeric binding failed");
    }
    nonnumeric.graph().edge_attrs(nonnumeric.graph().edge(0U, 1U))
        .set(bw->value_id, std::string("bad"));
    emit(
        "nonnumeric_zero",
        "link=" + double_token(nonnumeric.total_link_resource_demand()) +
            ";total=" + double_token(nonnumeric.total_resource_demand()));

    auto source = make_fixture();
    source.set_request_id(42);
    const double source_cached = source.total_resource_demand();
    auto clone = source.clone();
    const auto clone_cpu = clone.bind_node_attribute("cpu");
    if (!clone_cpu) {
        throw std::runtime_error("clone binding failed");
    }
    clone.set_node_attrs_data({
        {clone_cpu->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{0}, std::int64_t{0}, std::int64_t{0}, std::int64_t{0}}},
    });
    emit(
        "clone",
        "id=" + std::to_string(*clone.request_id()) +
            ";clone_cached=" + double_token(clone.total_resource_demand()) +
            ";source_cached=" + double_token(source_cached) +
            ";source_node=" +
            double_token(source.total_node_resource_demand()));

    network::VirtualNetwork moved(std::move(clone));
    emit(
        "native_move",
        std::string("node_binding=") +
            (moved.bind_node_attribute("cpu") ? "1" : "0") +
            ";id=" + std::to_string(*moved.request_id()));
}

}  // namespace

int main() {
    try {
        emit_cases();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "virtual_network_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
