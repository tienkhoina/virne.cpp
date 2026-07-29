#include "virtual_network.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;

void expect(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void expect_near(
    const double actual,
    const double expected,
    const std::string_view message) {
    if (std::abs(actual - expected) > 1e-12) {
        throw std::runtime_error(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
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
        {"max_latency", std::int64_t{15}},
    };
    network::VirtualNetwork result(std::move(construction));

    const auto cpu = result.bind_node_attribute("cpu");
    const auto peak = result.bind_node_attribute("peak");
    const auto bw = result.bind_link_attribute("bw");
    const auto peak_bw = result.bind_link_attribute("peak_bw");
    expect(cpu.has_value() && peak.has_value(), "node resource binding");
    expect(bw.has_value() && peak_bw.has_value(), "link resource binding");
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

void test_fixed_fields_and_demand() {
    network::VirtualNetwork missing;
    expect(!missing.max_latency().has_value(), "missing max latency");

    auto value = make_fixture();
    expect(value.request_id() == std::optional<std::int64_t>{7}, "metadata id");
    expect(value.arrival_time() == std::optional<double>{2.5}, "metadata arrival");
    expect(value.lifetime() == std::optional<double>{9.0}, "metadata lifetime");
    expect(
        value.max_latency() == std::optional<double>{15.0},
        "metadata max latency");

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        expect_near(
            value.total_node_resource_demand(workers), 18.0,
            "node demand workers");
        expect_near(
            value.total_link_resource_demand(workers), 12.75,
            "link demand workers");
    }
    expect_near(value.total_resource_demand(2U), 30.75, "combined demand");

    value.set_request_id(11);
    value.set_arrival_time(4.25);
    value.set_lifetime(12.5);
    value.set_max_latency(21.75);
    expect(value.request_id() == std::optional<std::int64_t>{11}, "set id");
    expect(value.arrival_time() == std::optional<double>{4.25}, "set arrival");
    expect(value.lifetime() == std::optional<double>{12.5}, "set lifetime");
    expect(
        value.max_latency() == std::optional<double>{21.75},
        "set max latency");
}

void test_cache_and_invalidation() {
    auto value = make_fixture();
    expect_near(value.total_resource_demand(), 30.75, "cache first value");
    const auto cpu = value.bind_node_attribute("cpu");
    expect(cpu.has_value(), "cache cpu binding");
    value.set_node_attrs_data({
        {cpu->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{100}, std::int64_t{100},
          std::int64_t{100}, std::int64_t{100}}},
    });
    expect_near(value.total_node_resource_demand(), 408.0, "live node demand");
    expect_near(value.total_resource_demand(8U), 30.75, "stale cached total");
    value.invalidate_cached_total_resource_demand();
    expect_near(value.total_resource_demand(8U), 420.75, "invalidated total");
}

void test_error_to_zero() {
    network::VirtualNetwork empty;
    expect_near(empty.total_node_resource_demand(), 0.0, "no node resource");
    expect_near(empty.total_link_resource_demand(), 0.0, "no link resource");
    expect_near(empty.total_resource_demand(), 0.0, "no combined resource");

    auto ragged = make_fixture();
    const auto cpu = ragged.bind_node_attribute("cpu");
    expect(cpu.has_value(), "ragged binding");
    ragged.graph().node_attrs(3U).erase(cpu->value_id);
    expect_near(ragged.total_node_resource_demand(), 0.0, "ragged returns zero");

    auto nonnumeric = make_fixture();
    const auto bw = nonnumeric.bind_link_attribute("bw");
    expect(bw.has_value(), "nonnumeric binding");
    nonnumeric.graph().edge_attrs(nonnumeric.graph().edge(0U, 1U))
        .set(bw->value_id, std::string("bad"));
    expect_near(
        nonnumeric.total_link_resource_demand(), 0.0,
        "nonnumeric returns zero");
    expect_near(
        nonnumeric.total_resource_demand(), 0.0,
        "combined first error returns zero");
}

void test_bool_and_integer_lanes() {
    Graph graph;
    graph.add_edge(0U, 1U);
    network::BaseNetworkConstruction construction;
    construction.incoming_graph = std::move(graph);
    construction.config.node_attribute_specs = {
        make_spec(
            "flag", attribute::AttributeOwner::node,
            attribute::AttributeKind::resource),
    };
    construction.config.link_attribute_specs = {
        make_spec(
            "units", attribute::AttributeOwner::link,
            attribute::AttributeKind::resource),
    };
    network::VirtualNetwork value(std::move(construction));
    const auto flag = value.bind_node_attribute("flag");
    const auto units = value.bind_link_attribute("units");
    expect(flag.has_value() && units.has_value(), "integer lane bindings");
    value.set_node_attrs_data({
        {flag->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {true, false}},
    });
    value.set_link_attrs_data({
        {units->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{-3}}},
    });
    expect_near(value.total_node_resource_demand(), 1.0, "boolean sum");
    expect_near(value.total_link_resource_demand(), -3.0, "integer sum");
    expect_near(value.total_resource_demand(), -2.0, "mixed family sum");
}

void test_move_clone_and_gml() {
    auto source = make_fixture();
    source.set_request_id(42);
    source.set_max_latency(64.5);
    const double cached = source.total_resource_demand();
    auto cloned = source.clone();
    expect(cloned.request_id() == std::optional<std::int64_t>{42}, "clone id");
    expect(
        cloned.max_latency() == std::optional<double>{64.5},
        "clone max latency");
    expect_near(cloned.total_resource_demand(), cached, "clone cached demand");
    const auto cpu = cloned.bind_node_attribute("cpu");
    expect(cpu.has_value(), "clone binding");
    cloned.set_node_attrs_data({
        {cpu->registry_id,
         network::AttributeDataLayout::dense,
         {},
         {std::int64_t{0}, std::int64_t{0}, std::int64_t{0}, std::int64_t{0}}},
    });
    expect_near(source.total_node_resource_demand(), 18.0, "clone is deep");

    network::VirtualNetwork moved(std::move(cloned));
    expect(moved.bind_node_attribute("cpu").has_value(), "move rebind");
    expect(
        moved.max_latency() == std::optional<double>{64.5},
        "move max latency");
    network::VirtualNetwork assigned;
    assigned = std::move(moved);
    expect(assigned.bind_link_attribute("bw").has_value(), "move assignment rebind");
    expect(
        assigned.max_latency() == std::optional<double>{64.5},
        "move assignment max latency");

    const auto path = std::filesystem::temp_directory_path() /
        "virne_virtual_network_unit.gml";
    source.to_gml(path.string());
    std::ifstream input(path, std::ios::binary);
    const std::string bytes{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    expect(bytes.find("graph [") != std::string::npos, "GML graph output");
    std::error_code error;
    std::filesystem::remove(path, error);
}

void test_concurrent_independent_networks() {
    std::vector<std::future<double>> futures;
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        futures.push_back(std::async(std::launch::async, [workers] {
            auto value = make_fixture();
            return value.total_resource_demand(workers);
        }));
    }
    for (auto& future : futures) {
        expect_near(future.get(), 30.75, "concurrent demand");
    }
}

}  // namespace

int main() {
    try {
        test_fixed_fields_and_demand();
        test_cache_and_invalidation();
        test_error_to_zero();
        test_bool_and_integer_lanes();
        test_move_clone_and_gml();
        test_concurrent_independent_networks();
        std::cout << "virtual_network_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "virtual_network_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
