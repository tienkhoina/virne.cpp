#include "../virne/solver/exact/registry.h"
#include "../virne/solver/exact/exact_with_risk.h"

#include "../random/py_random.h"
#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace exact = virne::solver::exact;
namespace network = virne::network;
namespace solver = virne::solver;

using ResourceId = attribute::AttributeRegistryId;

constexpr std::size_t exact_solver_count = 3U;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::filesystem::path unique_temp_root() {
    static std::atomic<std::uint64_t> sequence{0U};
    const auto tick = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
    const auto suffix = sequence.fetch_add(1U, std::memory_order_relaxed);
    return std::filesystem::temp_directory_path() /
           ("virne-exact-solver-unit-" + std::to_string(tick) + "-" +
            std::to_string(suffix));
}

struct CleanupRoot {
    std::filesystem::path path;

    ~CleanupRoot() {
        std::error_code error;
        const auto temporary_root =
            std::filesystem::temp_directory_path(error);
        const std::string filename = path.filename().string();
        if (error || path.empty() || path.parent_path() != temporary_root ||
            filename.rfind("virne-exact-solver-unit-", 0U) != 0U) {
            return;
        }
        error.clear();
        std::filesystem::remove_all(path, error);
    }
};

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    attribute::AttributeOwner owner) {
    attribute::AttributeFactorySpec spec;
    spec.name = std::move(name);
    spec.owner = owner;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = owner == attribute::AttributeOwner::node
                              ? attribute::CheckingLevel::node
                              : attribute::CheckingLevel::link;
    return spec;
}

attribute::AttributeFactorySpec status_spec(
    std::string name,
    attribute::AttributeOwner owner) {
    attribute::AttributeFactorySpec spec;
    spec.name = std::move(name);
    spec.owner = owner;
    spec.kind = attribute::AttributeKind::status;
    return spec;
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name) {
    const auto binding = value.bind_node_attribute(name);
    require(binding.has_value(), "missing node attribute binding");
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name) {
    const auto binding = value.bind_link_attribute(name);
    require(binding.has_value(), "missing link attribute binding");
    return *binding;
}

std::vector<AttrValue> attribute_values(
    const std::vector<std::int64_t>& values) {
    std::vector<AttrValue> result;
    result.reserve(values.size());
    for (const std::int64_t value : values) {
        result.emplace_back(value);
    }
    return result;
}

network::NodeAttributeDataUpdate node_update(
    ResourceId id,
    const std::vector<std::int64_t>& values) {
    network::NodeAttributeDataUpdate update;
    update.registry_id = id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = attribute_values(values);
    return update;
}

network::LinkAttributeDataUpdate link_update(
    ResourceId id,
    const std::vector<std::int64_t>& values) {
    network::LinkAttributeDataUpdate update;
    update.registry_id = id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = attribute_values(values);
    return update;
}

network::LinkAttributeDataUpdate floating_link_update(
    ResourceId id,
    double value) {
    network::LinkAttributeDataUpdate update;
    update.registry_id = id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = {AttrValue{value}};
    return update;
}

enum class RequestShape : std::uint8_t {
    feasible,
    reject_gpu,
    reject_spectrum,
};

network::VirtualNetwork make_virtual_network(RequestShape shape) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    const std::int64_t second_gpu =
        shape == RequestShape::reject_gpu ? 11 : 4;
    const std::int64_t edge_spectrum =
        // The integral Python MIP permits split flow. Exceed the two-edge
        // source cut (2 * 10), rather than merely one physical edge.
        shape == RequestShape::reject_spectrum ? 21 : 3;
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {4, 5}),
        node_update(gpu.registry_id, {3, second_gpu}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {4}),
        link_update(spectrum.registry_id, {edge_spectrum}),
    });
    result.set_request_id(71);
    result.set_arrival_time(2.0);
    result.set_lifetime(8.0);
    return result;
}

network::PhysicalNetwork make_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 2U}, {2U, 3U}, {0U, 3U}});
    // Status fields deliberately shift the physical registry IDs. Exact
    // solvers must resolve each cold resource name once, then use numeric IDs.
    construction.config.node_attribute_specs = {
        status_spec("node_status", attribute::AttributeOwner::node),
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        status_spec("link_status", attribute::AttributeOwner::link),
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 10, 10, 10}),
        node_update(gpu.registry_id, {10, 10, 10, 10}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {10, 10, 10, 10}),
        link_update(spectrum.registry_id, {10, 10, 10, 10}),
    });
    return result;
}

network::VirtualNetwork make_python_compatible_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto bw = require_link_binding(result, "bw");
    result.set_node_attrs_data({node_update(cpu.registry_id, {4, 5})});
    result.set_link_attrs_data({link_update(bw.registry_id, {4})});
    result.set_request_id(72);
    result.set_arrival_time(2.0);
    result.set_lifetime(8.0);
    return result;
}

network::PhysicalNetwork make_python_compatible_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 2U}, {2U, 3U}, {0U, 3U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto bw = require_link_binding(result, "bw");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 10, 10, 10}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {10, 10, 10, 10}),
    });
    return result;
}

network::VirtualNetwork make_split_flow_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 10}),
        node_update(gpu.registry_id, {10, 10}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {4}),
        link_update(spectrum.registry_id, {8}),
    });
    result.set_request_id(73);
    result.set_arrival_time(3.0);
    result.set_lifetime(9.0);
    return result;
}

network::VirtualNetwork make_self_loop_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        1U, std::vector<EdgeEndpoints>{{0U, 0U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {1}),
        node_update(gpu.registry_id, {1}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {1}),
        link_update(spectrum.registry_id, {1}),
    });
    result.set_request_id(77);
    result.set_arrival_time(7.0);
    result.set_lifetime(13.0);
    return result;
}

network::VirtualNetwork make_tiny_positive_link_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {1, 1}),
        node_update(gpu.registry_id, {1, 1}),
    });
    result.set_link_attrs_data({
        floating_link_update(bw.registry_id, 5.0e-10),
        link_update(spectrum.registry_id, {1}),
    });
    result.set_request_id(78);
    result.set_arrival_time(8.0);
    result.set_lifetime(14.0);
    return result;
}

network::VirtualNetwork make_floating_link_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {1, 1}),
        node_update(gpu.registry_id, {1, 1}),
    });
    // With no positive integer link lane, q must be continuous. The old
    // all-integer path model incorrectly made conservation at 1.5 infeasible.
    result.set_link_attrs_data({
        floating_link_update(bw.registry_id, 1.5),
        floating_link_update(spectrum.registry_id, 2.5),
    });
    result.set_request_id(79);
    result.set_arrival_time(9.0);
    result.set_lifetime(15.0);
    return result;
}

network::PhysicalNetwork make_split_flow_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 3U}, {0U, 2U}, {2U, 3U}});
    construction.config.node_attribute_specs = {
        status_spec("node_status", attribute::AttributeOwner::node),
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        status_spec("link_status", attribute::AttributeOwner::link),
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 0, 0, 10}),
        node_update(gpu.registry_id, {10, 0, 0, 10}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {3, 3, 3, 3}),
        link_update(spectrum.registry_id, {6, 6, 6, 6}),
    });
    return result;
}

network::VirtualNetwork make_risk_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 10}),
        node_update(gpu.registry_id, {10, 10}),
    });
    result.set_link_attrs_data({
        link_update(bw.registry_id, {4}),
        link_update(spectrum.registry_id, {4}),
    });
    result.set_request_id(76);
    result.set_arrival_time(6.0);
    result.set_lifetime(12.0);
    return result;
}

network::PhysicalNetwork make_risk_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 3U}, {0U, 2U}, {2U, 3U}});
    construction.config.node_attribute_specs = {
        status_spec("node_status", attribute::AttributeOwner::node),
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        status_spec("link_status", attribute::AttributeOwner::link),
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 0, 0, 10}),
        node_update(gpu.registry_id, {10, 0, 0, 10}),
    });
    // Both routes have two hops. The upper route is a deliberately scarce,
    // skewed residual vector; risk must select the lower route while the
    // primary flow term remains tied.
    result.set_link_attrs_data({
        link_update(bw.registry_id, {5, 5, 20, 20}),
        link_update(spectrum.registry_id, {100, 100, 20, 20}),
    });
    return result;
}

network::PhysicalNetwork make_risk_longer_alternative_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        5U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 3U}, {0U, 2U}, {2U, 4U}, {4U, 3U}});
    construction.config.node_attribute_specs = {
        status_spec("node_status", attribute::AttributeOwner::node),
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        status_spec("link_status", attribute::AttributeOwner::link),
        resource_spec("bw", attribute::AttributeOwner::link),
        resource_spec("spectrum", attribute::AttributeOwner::link),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    const auto bw = require_link_binding(result, "bw");
    const auto spectrum = require_link_binding(result, "spectrum");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10, 0, 0, 10, 0}),
        node_update(gpu.registry_id, {10, 0, 0, 10, 0}),
    });
    // The two-hop path is scarce; the three-hop path is spacious. The route
    // term must still win because the secondary risk is normalized below one.
    result.set_link_attrs_data({
        link_update(bw.registry_id, {5, 5, 100, 100, 100}),
        link_update(spectrum.registry_id, {100, 100, 100, 100, 100}),
    });
    return result;
}

network::VirtualNetwork make_large_integer_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(1U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    result.set_node_attrs_data({node_update(
        cpu.registry_id,
        {(std::int64_t{1} << 53U) + 1})});
    result.set_request_id(74);
    result.set_arrival_time(4.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_large_integer_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(1U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    result.set_node_attrs_data({node_update(
        cpu.registry_id,
        {std::int64_t{1} << 53U})});
    return result;
}

network::VirtualNetwork make_aggregate_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(2U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {6, 6}),
        node_update(gpu.registry_id, {4, 4}),
    });
    result.set_request_id(75);
    result.set_arrival_time(5.0);
    result.set_lifetime(11.0);
    return result;
}

network::PhysicalNetwork make_aggregate_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(1U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node),
        resource_spec("gpu", attribute::AttributeOwner::node),
    };
    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto gpu = require_node_binding(result, "gpu");
    result.set_node_attrs_data({
        node_update(cpu.registry_id, {10}),
        node_update(gpu.registry_id, {10}),
    });
    return result;
}

controller::ControllerSelection python_compatible_selection(
    const network::VirtualNetwork& virtual_network) {
    const auto cpu = require_node_binding(virtual_network, "cpu");
    const auto bw = require_link_binding(virtual_network, "bw");
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {cpu.registry_id};
    selection.constraints.link_at_link = {bw.registry_id};
    selection.node_resources = {cpu.registry_id};
    selection.link_resources = {bw.registry_id};
    selection.hard_node_constraints = {cpu.registry_id};
    selection.hard_link_constraints = {bw.registry_id};
    selection.reusable = false;
    return selection;
}

controller::ControllerSelection node_only_selection(
    const network::VirtualNetwork& virtual_network) {
    const auto cpu = require_node_binding(virtual_network, "cpu");
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {cpu.registry_id};
    selection.node_resources = {cpu.registry_id};
    selection.hard_node_constraints = {cpu.registry_id};
    selection.reusable = false;
    return selection;
}

controller::ControllerSelection aggregate_selection(
    const network::VirtualNetwork& virtual_network) {
    const auto cpu = require_node_binding(virtual_network, "cpu");
    const auto gpu = require_node_binding(virtual_network, "gpu");
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {
        cpu.registry_id, gpu.registry_id};
    selection.node_resources = {cpu.registry_id, gpu.registry_id};
    selection.hard_node_constraints = {
        cpu.registry_id, gpu.registry_id};
    selection.reusable = true;
    return selection;
}

controller::ControllerSelection controller_selection(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network) {
    const auto cpu = require_node_binding(virtual_network, "cpu");
    const auto gpu = require_node_binding(virtual_network, "gpu");
    const auto bw = require_link_binding(virtual_network, "bw");
    const auto spectrum = require_link_binding(virtual_network, "spectrum");
    const auto physical_cpu = require_node_binding(physical_network, "cpu");
    const auto physical_gpu = require_node_binding(physical_network, "gpu");
    const auto physical_bw = require_link_binding(physical_network, "bw");
    const auto physical_spectrum =
        require_link_binding(physical_network, "spectrum");
    require(
        cpu.registry_id != physical_cpu.registry_id &&
            gpu.registry_id != physical_gpu.registry_id &&
            bw.registry_id != physical_bw.registry_id &&
            spectrum.registry_id != physical_spectrum.registry_id,
        "fixture did not shift physical resource IDs");
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {cpu.registry_id, gpu.registry_id};
    selection.constraints.link_at_link = {bw.registry_id, spectrum.registry_id};
    selection.node_resources = {cpu.registry_id, gpu.registry_id};
    selection.link_resources = {bw.registry_id, spectrum.registry_id};
    selection.hard_node_constraints = {cpu.registry_id, gpu.registry_id};
    selection.hard_link_constraints = {bw.registry_id, spectrum.registry_id};
    selection.reusable = false;
    return selection;
}

core::RecorderConfig recorder_config(const std::filesystem::path& root) {
    core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "exact-solver-unit";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

core::LoggerConfig logger_config(const std::filesystem::path& root) {
    core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "exact-solver-unit";
    config.run_id = "logger";
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = false;
    config.backends.file = false;
    config.level = core::LoggerLevel::critical;
    config.log_show_interval = 1U;
    return config;
}

struct RuntimeFixture {
    CleanupRoot cleanup;
    controller::Controller controller;
    core::Counter counter;
    core::Recorder recorder;
    core::Logger logger;

    explicit RuntimeFixture(controller::ControllerSelection selection)
        : cleanup{unique_temp_root()},
          controller{std::move(selection)},
          counter{core::CounterSelection{}},
          recorder{
              core::Counter(core::CounterSelection{}),
              recorder_config(cleanup.path)},
          logger{logger_config(cleanup.path)} {}

    solver::SolverDependencies dependencies() {
        return solver::SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger),
        };
    }
};

solver::SolverConfig solver_config(const std::filesystem::path& root) {
    solver::SolverConfig config;
    config.verbose = 0;
    config.save_dir = root;
    config.reusable = false;
    return config;
}

struct ResourceSnapshot {
    std::vector<std::vector<AttrValue>> nodes;
    std::vector<std::vector<AttrValue>> links;

    friend bool operator==(
        const ResourceSnapshot& left,
        const ResourceSnapshot& right) {
        return left.nodes == right.nodes && left.links == right.links;
    }

    friend bool operator!=(
        const ResourceSnapshot& left,
        const ResourceSnapshot& right) {
        return !(left == right);
    }
};

ResourceSnapshot resource_snapshot(network::PhysicalNetwork& value) {
    const auto cpu = require_node_binding(value, "cpu");
    const auto gpu = require_node_binding(value, "gpu");
    const auto bw = require_link_binding(value, "bw");
    const auto spectrum = require_link_binding(value, "spectrum");
    return ResourceSnapshot{
        network::get_node_attrs_data(
            value, {cpu.registry_id, gpu.registry_id}, 1U),
        network::get_link_attrs_data(
            value, {bw.registry_id, spectrum.registry_id}, 1U),
    };
}

struct ExpectedSolver {
    solver::SolverId id;
    std::string_view name;
    solver::SolverCategory category;
    exact::ExactAlgorithm algorithm;
};

std::array<ExpectedSolver, exact_solver_count> expected_solvers(
    const exact::ExactSolverIds& ids) {
    return {{
        {ids.mip, "mip", solver::SolverCategory::exact,
         exact::ExactAlgorithm::mixed_integer},
        {ids.d_round, "d_round", solver::SolverCategory::rounding,
         exact::ExactAlgorithm::deterministic_rounding},
        {ids.r_round, "r_round", solver::SolverCategory::rounding,
         exact::ExactAlgorithm::randomized_rounding},
    }};
}

void require_resource_journal(
    const core::Solution& solution,
    ResourceId cpu,
    ResourceId gpu,
    ResourceId bw,
    ResourceId spectrum) {
    require(
        solution.node_slots_info.size() == 2U,
        "exact node resource journal is incomplete");
    for (const auto& entry : solution.node_slots_info.entries()) {
        require(
            entry.value.find(cpu) != nullptr &&
                entry.value.find(gpu) != nullptr,
            "exact node journal omitted a resource lane");
    }
    require(
        !solution.link_paths_info.empty(),
        "exact link resource journal is empty");
    for (const auto& entry : solution.link_paths_info.entries()) {
        require(
            entry.value.find(bw) != nullptr &&
                entry.value.find(spectrum) != nullptr,
            "exact link journal omitted a resource lane");
    }
}

void require_feasible_solution(
    const core::Solution& solution,
    const network::VirtualNetwork& virtual_network,
    std::string_view context = "exact solver") {
    const auto cpu = require_node_binding(virtual_network, "cpu");
    const auto gpu = require_node_binding(virtual_network, "gpu");
    const auto bw = require_link_binding(virtual_network, "bw");
    const auto spectrum = require_link_binding(virtual_network, "spectrum");
    if (!solution.result || !solution.place_result ||
        !solution.route_result) {
        throw std::runtime_error(
            std::string(context) +
            " rejected the feasible multi-resource request: " +
            solution.description);
    }
    require(
        solution.node_slots.size() == 2U &&
            solution.link_paths.size() == 1U,
        "exact solver returned an incomplete mapping");
    const auto& slots = solution.node_slots.entries();
    require(
        slots[0U].value != slots[1U].value,
        "non-reusable exact solver reused a physical node");
    const auto& path = solution.link_paths.entries().front().value;
    require(!path.empty(), "exact solver returned an empty inter-node path");
    require_resource_journal(
        solution,
        cpu.registry_id,
        gpu.registry_id,
        bw.registry_id,
        spectrum.registry_id);
}

std::array<ExpectedSolver, exact_solver_count> register_and_check(
    solver::SolverRegistry& registry,
    PyRandom& random) {
    exact::ExactSolverParameters parameters;
    parameters.time_limit_ms = 5'000U;
    const auto expected = expected_solvers(
        exact::register_exact_solvers(registry, random, parameters));
    require(
        registry.size() == exact_solver_count,
        "exact registry size mismatch");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        require(
            expected[index].id ==
                solver::SolverId{static_cast<std::uint32_t>(index)},
            "exact registry ID/order mismatch");
        require(
            registry.resolve(expected[index].name) == expected[index].id,
            "exact registry cold-name resolution mismatch");
    }
    registry.freeze();
    for (const auto& entry : expected) {
        const auto& descriptor = registry.descriptor(entry.id);
        require(
            descriptor.name == entry.name &&
                descriptor.category == entry.category,
            "exact registry descriptor mismatch");
    }
    return expected;
}

void test_registry_const_and_multi_resource_rejection() {
    PyRandom random(0U);
    solver::SolverRegistry registry;
    const auto expected = register_and_check(registry, random);

    for (const auto& descriptor : expected) {
        auto virtual_network = make_virtual_network(RequestShape::feasible);
        auto physical_network = make_physical_network();
        RuntimeFixture runtime(controller_selection(
            virtual_network, physical_network));
        const ResourceSnapshot before = resource_snapshot(physical_network);
        auto instance = registry.create(
            descriptor.id,
            runtime.dependencies(),
            solver_config(runtime.cleanup.path));
        const auto* exact_instance =
            dynamic_cast<const exact::ExactSolver*>(instance.get());
        require(
            exact_instance != nullptr &&
                exact_instance->algorithm() == descriptor.algorithm,
            "exact factory algorithm mismatch");
        const core::Solution solution = instance->solve(
            solver::SolverInstance{virtual_network, physical_network});
        require_feasible_solution(
            solution, virtual_network, descriptor.name);
        require(
            resource_snapshot(physical_network) == before,
            "const exact solve mutated physical resources");

        for (const RequestShape rejected : {
                 RequestShape::reject_gpu,
                 RequestShape::reject_spectrum}) {
            auto rejected_virtual = make_virtual_network(rejected);
            auto rejected_physical = make_physical_network();
            RuntimeFixture rejected_runtime(controller_selection(
                rejected_virtual, rejected_physical));
            const ResourceSnapshot rejected_before =
                resource_snapshot(rejected_physical);
            auto rejected_instance = registry.create(
                descriptor.id,
                rejected_runtime.dependencies(),
                solver_config(rejected_runtime.cleanup.path));
            const core::Solution rejected_solution = rejected_instance->solve(
                solver::SolverInstance{
                    rejected_virtual, rejected_physical});
            require(
                !rejected_solution.result,
                rejected == RequestShape::reject_gpu
                    ? "exact solver ignored the GPU capacity lane"
                    : "exact solver ignored the spectrum capacity lane");
            require(
                resource_snapshot(rejected_physical) == rejected_before,
                "rejected const exact solve mutated physical resources");
        }
    }

    auto unsupported_virtual = make_virtual_network(RequestShape::feasible);
    auto unsupported_physical = make_physical_network();
    auto unsupported_selection = controller_selection(
        unsupported_virtual, unsupported_physical);
    const auto unsupported_id = static_cast<ResourceId>(999U);
    unsupported_selection.constraints.node_at_node.push_back(unsupported_id);
    unsupported_selection.hard_node_constraints.push_back(unsupported_id);
    RuntimeFixture unsupported_runtime(std::move(unsupported_selection));
    auto unsupported_instance = registry.create(
        expected.front().id,
        unsupported_runtime.dependencies(),
        solver_config(unsupported_runtime.cleanup.path));
    const core::Solution unsupported_solution = unsupported_instance->solve(
        solver::SolverInstance{
            unsupported_virtual, unsupported_physical});
    require(
        !unsupported_solution.result &&
            !unsupported_solution.description.empty(),
        "exact solver silently ignored a non-resource hard constraint");
}

void test_exact_with_risk_objective() {
    PyRandom random(0U);
    solver::SolverRegistry registry;
    exact::ExactSolverParameters parameters;
    parameters.time_limit_ms = 5'000U;
    static_cast<void>(exact::register_exact_solvers(
        registry, random, parameters));
    const auto risk_id = exact::register_exact_with_risk_solver(
        registry, random, parameters);
    require(
        registry.resolve("exact_with_risk") == risk_id &&
            risk_id == solver::SolverId{3U},
        "exact_with_risk registry entry mismatch");
    registry.freeze();

    auto virtual_network = make_risk_virtual_network();
    auto physical_network = make_risk_physical_network();
    RuntimeFixture runtime(controller_selection(
        virtual_network, physical_network));
    auto instance = registry.create(
        risk_id,
        runtime.dependencies(),
        solver_config(runtime.cleanup.path));
    const auto* risk_instance =
        dynamic_cast<const exact::ExactWithRiskSolver*>(instance.get());
    require(
        risk_instance != nullptr,
        "exact_with_risk factory returned the wrong type");
    const core::Solution solution = instance->solve(
        solver::SolverInstance{virtual_network, physical_network});
    require_feasible_solution(solution, virtual_network, "risk equal-hop");
    const auto& path = solution.link_paths.entries().front().value;
    require(
        path.size() == 2U,
        "exact_with_risk changed the primary shortest-route objective");
    for (const auto& link : path) {
        const auto low = std::min(link.source, link.target);
        const auto high = std::max(link.source, link.target);
        require(
            (low == 0U && high == 2U) ||
                (low == 2U && high == 3U),
            "exact_with_risk did not minimize the equal-hop fragmentation risk");
    }

    auto longer_physical = make_risk_longer_alternative_network();
    RuntimeFixture longer_runtime(controller_selection(
        virtual_network, longer_physical));
    auto longer_instance = registry.create(
        risk_id,
        longer_runtime.dependencies(),
        solver_config(longer_runtime.cleanup.path));
    const core::Solution shorter_solution = longer_instance->solve(
        solver::SolverInstance{virtual_network, longer_physical});
    require_feasible_solution(
        shorter_solution, virtual_network, "risk route dominance");
    const auto& shorter_path =
        shorter_solution.link_paths.entries().front().value;
    const auto is_short_edge = [](const core::SolutionLink& link) {
        const auto low = std::min(link.source, link.target);
        const auto high = std::max(link.source, link.target);
        return (low == 0U && high == 1U) ||
            (low == 1U && high == 3U);
    };
    require(
        shorter_path.size() == 2U &&
            is_short_edge(shorter_path[0U]) &&
            is_short_edge(shorter_path[1U]),
        "exact_with_risk allowed fragmentation risk to beat a shorter route");

    auto split_virtual = make_split_flow_virtual_network();
    auto split_physical = make_split_flow_physical_network();
    RuntimeFixture split_runtime(controller_selection(
        split_virtual, split_physical));
    auto split_instance = registry.create(
        risk_id,
        split_runtime.dependencies(),
        solver_config(split_runtime.cleanup.path));
    const core::Solution split_solution = split_instance->solve(
        solver::SolverInstance{split_virtual, split_physical});
    require_feasible_solution(
        split_solution, split_virtual, "risk split flow");
    require(
        split_solution.link_paths.entries().front().value.size() == 4U,
        "exact_with_risk lost split-flow feasibility");

    auto floating_virtual = make_floating_link_virtual_network();
    auto floating_physical = make_physical_network();
    RuntimeFixture floating_runtime(controller_selection(
        floating_virtual, floating_physical));
    auto floating_instance = registry.create(
        risk_id,
        floating_runtime.dependencies(),
        solver_config(floating_runtime.cleanup.path));
    const core::Solution floating_solution = floating_instance->solve(
        solver::SolverInstance{floating_virtual, floating_physical});
    require_feasible_solution(
        floating_solution, floating_virtual, "risk floating flow");

    auto self_loop_virtual = make_self_loop_virtual_network();
    auto self_loop_physical = make_physical_network();
    RuntimeFixture self_loop_runtime(controller_selection(
        self_loop_virtual, self_loop_physical));
    auto self_loop_instance = registry.create(
        risk_id,
        self_loop_runtime.dependencies(),
        solver_config(self_loop_runtime.cleanup.path));
    const core::Solution self_loop_solution = self_loop_instance->solve(
        solver::SolverInstance{self_loop_virtual, self_loop_physical});
    require(
        !self_loop_solution.result &&
            !self_loop_solution.description.empty(),
        "exact_with_risk did not inherit the self-loop guard");

    auto tiny_virtual = make_tiny_positive_link_virtual_network();
    auto tiny_physical = make_physical_network();
    RuntimeFixture tiny_runtime(controller_selection(
        tiny_virtual, tiny_physical));
    auto tiny_instance = registry.create(
        risk_id,
        tiny_runtime.dependencies(),
        solver_config(tiny_runtime.cleanup.path));
    const core::Solution tiny_solution = tiny_instance->solve(
        solver::SolverInstance{tiny_virtual, tiny_physical});
    require(
        !tiny_solution.result && !tiny_solution.description.empty(),
        "exact_with_risk did not inherit the tiny-demand guard");
}

void test_split_flow_and_fail_closed_guards() {
    PyRandom random(0U);
    solver::SolverRegistry registry;
    const auto expected = register_and_check(registry, random);

    auto split_virtual = make_split_flow_virtual_network();
    auto split_physical = make_split_flow_physical_network();
    RuntimeFixture split_runtime(controller_selection(
        split_virtual, split_physical));
    const ResourceSnapshot split_before = resource_snapshot(split_physical);
    auto split_instance = registry.create(
        expected.front().id,
        split_runtime.dependencies(),
        solver_config(split_runtime.cleanup.path));
    const core::Solution split_solution = split_instance->solve(
        solver::SolverInstance{split_virtual, split_physical});
    require(
        split_solution.result && split_solution.link_paths.size() == 1U,
        "MIP rejected a request feasible only through split flow");
    require(
        split_solution.link_paths.entries().front().value.size() == 4U,
        "MIP did not retain both branches of the split flow");
    require(
        resource_snapshot(split_physical) == split_before,
        "const split-flow solve mutated resources");

    core::SolutionNodeId physical_source = 0;
    bool source_found = false;
    for (const auto& entry : split_solution.node_slots.entries()) {
        if (entry.key == 0) {
            physical_source = entry.value;
            source_found = true;
            break;
        }
    }
    require(source_found, "split-flow solution omitted source placement");
    const auto bw = require_link_binding(split_virtual, "bw");
    const auto spectrum = require_link_binding(split_virtual, "spectrum");
    std::int64_t outgoing_bw = 0;
    std::int64_t outgoing_spectrum = 0;
    for (const auto& entry : split_solution.link_paths_info.entries()) {
        const auto* bw_amount = entry.value.find(bw.registry_id);
        const auto* spectrum_amount = entry.value.find(spectrum.registry_id);
        require(
            bw_amount != nullptr && spectrum_amount != nullptr,
            "split-flow journal omitted a resource lane");
        require(
            std::holds_alternative<std::int64_t>(*bw_amount) &&
                std::holds_alternative<std::int64_t>(*spectrum_amount),
            "split-flow integer journal lane lost its int64 representation");
        if (entry.key.physical_link.source != physical_source) {
            continue;
        }
        outgoing_bw += std::get<std::int64_t>(*bw_amount);
        outgoing_spectrum += std::get<std::int64_t>(*spectrum_amount);
    }
    require(
        outgoing_bw == 4 && outgoing_spectrum == 8,
        "split-flow journal does not conserve every resource lane");

    auto self_loop_virtual = make_self_loop_virtual_network();
    auto self_loop_physical = make_physical_network();
    RuntimeFixture self_loop_runtime(controller_selection(
        self_loop_virtual, self_loop_physical));
    auto self_loop_instance = registry.create(
        expected.front().id,
        self_loop_runtime.dependencies(),
        solver_config(self_loop_runtime.cleanup.path));
    const core::Solution self_loop_solution = self_loop_instance->solve(
        solver::SolverInstance{self_loop_virtual, self_loop_physical});
    require(
        !self_loop_solution.result &&
            !self_loop_solution.description.empty(),
        "exact solver did not fail closed for a virtual self-loop");

    auto tiny_virtual = make_tiny_positive_link_virtual_network();
    auto tiny_physical = make_physical_network();
    RuntimeFixture tiny_runtime(controller_selection(
        tiny_virtual, tiny_physical));
    auto tiny_instance = registry.create(
        expected.front().id,
        tiny_runtime.dependencies(),
        solver_config(tiny_runtime.cleanup.path));
    const core::Solution tiny_solution = tiny_instance->solve(
        solver::SolverInstance{tiny_virtual, tiny_physical});
    require(
        !tiny_solution.result && !tiny_solution.description.empty(),
        "exact solver did not fail closed for a tiny positive link demand");

    auto floating_virtual = make_floating_link_virtual_network();
    auto floating_physical = make_physical_network();
    RuntimeFixture floating_runtime(controller_selection(
        floating_virtual, floating_physical));
    auto floating_instance = registry.create(
        expected.front().id,
        floating_runtime.dependencies(),
        solver_config(floating_runtime.cleanup.path));
    const core::Solution floating_solution = floating_instance->solve(
        solver::SolverInstance{floating_virtual, floating_physical});
    require_feasible_solution(
        floating_solution, floating_virtual, "mip floating flow");
    const auto floating_bw = require_link_binding(floating_virtual, "bw");
    const auto* floating_amount =
        floating_solution.link_paths_info.entries().front().value.find(
            floating_bw.registry_id);
    require(
        floating_amount != nullptr &&
            std::holds_alternative<double>(*floating_amount),
        "MIP did not preserve a floating-only link resource lane");

    auto path_virtual = make_virtual_network(RequestShape::feasible);
    auto path_physical = make_physical_network();
    auto path_selection = controller_selection(
        path_virtual, path_physical);
    path_selection.constraints.link_at_path.push_back(
        require_link_binding(path_virtual, "bw").registry_id);
    RuntimeFixture path_runtime(std::move(path_selection));
    auto path_instance = registry.create(
        expected.front().id,
        path_runtime.dependencies(),
        solver_config(path_runtime.cleanup.path));
    const core::Solution path_solution = path_instance->solve(
        solver::SolverInstance{path_virtual, path_physical});
    require(
        !path_solution.result && !path_solution.description.empty(),
        "exact solver treated a path-level resource as an edge row");

    auto large_virtual = make_large_integer_virtual_network();
    auto large_physical = make_large_integer_physical_network();
    RuntimeFixture large_runtime(node_only_selection(large_virtual));
    auto large_instance = registry.create(
        expected.front().id,
        large_runtime.dependencies(),
        solver_config(large_runtime.cleanup.path));
    const core::Solution large_solution = large_instance->solve(
        solver::SolverInstance{large_virtual, large_physical});
    require(
        !large_solution.result && !large_solution.description.empty(),
        "exact solver rounded an int64 coefficient above 2^53");

    auto aggregate_virtual = make_aggregate_virtual_network();
    auto aggregate_physical = make_aggregate_physical_network();
    RuntimeFixture aggregate_runtime(aggregate_selection(aggregate_virtual));
    auto aggregate_instance = registry.create(
        expected.front().id,
        aggregate_runtime.dependencies(),
        solver_config(aggregate_runtime.cleanup.path));
    const core::Solution aggregate_solution = aggregate_instance->solve(
        solver::SolverInstance{aggregate_virtual, aggregate_physical});
    require(
        !aggregate_solution.result,
        "MIP omitted aggregate capacity for reusable node placement");
}

void test_mutable_commit_and_release() {
    PyRandom random(0U);
    solver::SolverRegistry registry;
    const auto expected = register_and_check(registry, random);
    auto virtual_network = make_virtual_network(RequestShape::feasible);
    auto physical_network = make_physical_network();
    RuntimeFixture runtime(controller_selection(
        virtual_network, physical_network));
    const ResourceSnapshot before = resource_snapshot(physical_network);
    auto mutation = runtime.controller.prepare_mutation(
        virtual_network, physical_network);
    mutation.begin_transaction();
    auto instance = registry.create(
        expected.front().id,
        runtime.dependencies(),
        solver_config(runtime.cleanup.path));
    const solver::MutableSolverResult solved = instance->solve_mutable(
        solver::MutableSolverInstance{
            virtual_network, physical_network, mutation});
    require(
        solved.mutation_state == solver::SolverMutationState::committed,
        "mutable exact solve did not report committed resources");
    require_feasible_solution(
        solved.solution, virtual_network, "mutable mip");
    require(
        resource_snapshot(physical_network) != before,
        "mutable exact solve did not consume resources");
    require(
        mutation.transaction_active(),
        "mutable exact solve lost the caller transaction");
    mutation.commit_transaction();
    require(
        mutation.release(solved.solution, {4U}),
        "exact solution release failed");
    require(
        resource_snapshot(physical_network) == before,
        "exact solution release did not restore every resource lane");

    auto owned_virtual = make_virtual_network(RequestShape::feasible);
    auto owned_physical = make_physical_network();
    RuntimeFixture owned_runtime(controller_selection(
        owned_virtual, owned_physical));
    const ResourceSnapshot owned_before = resource_snapshot(owned_physical);
    auto owned_mutation = owned_runtime.controller.prepare_mutation(
        owned_virtual, owned_physical);
    auto owned_instance = registry.create(
        expected.front().id,
        owned_runtime.dependencies(),
        solver_config(owned_runtime.cleanup.path));
    const solver::MutableSolverResult owned_solved =
        owned_instance->solve_mutable(solver::MutableSolverInstance{
            owned_virtual, owned_physical, owned_mutation});
    require(
        owned_solved.mutation_state ==
            solver::SolverMutationState::committed &&
            !owned_mutation.transaction_active(),
        "exact solver-owned transaction was not committed cleanly");
    require(
        owned_mutation.release(owned_solved.solution, {4U}),
        "solver-owned exact deployment release failed");
    require(
        resource_snapshot(owned_physical) == owned_before,
        "solver-owned exact transaction did not restore on release");
}

void benchmark_python_compatible_mip() {
    PyRandom random(0U);
    solver::SolverRegistry registry;
    const auto expected = register_and_check(registry, random);
    auto virtual_network = make_python_compatible_virtual_network();
    auto physical_network = make_python_compatible_physical_network();
    RuntimeFixture runtime(python_compatible_selection(virtual_network));
    auto instance = registry.create(
        expected.front().id,
        runtime.dependencies(),
        solver_config(runtime.cleanup.path));

    auto run_once = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        core::Solution solution = instance->solve(
            solver::SolverInstance{virtual_network, physical_network});
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count();
        require(solution.result, "compatibility MIP benchmark rejected");
        require(
            solution.node_slots.size() == 2U &&
                solution.link_paths.size() == 1U,
            "compatibility MIP benchmark mapping is incomplete");
        return std::pair<std::int64_t, core::Solution>{
            elapsed, std::move(solution)};
    };

    static_cast<void>(run_once());
    std::array<std::int64_t, 5U> samples{};
    core::Solution last = core::Solution::from_v_net(virtual_network);
    for (auto& sample : samples) {
        auto measured = run_once();
        sample = measured.first;
        last = std::move(measured.second);
    }
    std::sort(samples.begin(), samples.end());
    std::size_t path_edges = 0U;
    for (const auto& entry : last.link_paths.entries()) {
        path_edges += entry.value.size();
    }
    std::cout
        << "{\"solver\":\"mip\",\"resource_lanes\":2,"
        << "\"samples\":5,\"median_ns\":" << samples[2U]
        << ",\"min_ns\":" << samples.front()
        << ",\"accepted\":true,\"node_slots\":"
        << last.node_slots.size()
        << ",\"path_edges\":" << path_edges << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
            benchmark_python_compatible_mip();
            return 0;
        }
        test_registry_const_and_multi_resource_rejection();
        test_exact_with_risk_objective();
        test_split_flow_and_fail_closed_guards();
        test_mutable_commit_and_release();
        std::cout << "exact solver unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exact solver unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
