#include "controller.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using controller::ConstraintId;
using controller::ConstraintLink;
using controller::Controller;
using controller::ControllerErrorCode;
using controller::ControllerException;
using controller::ControllerFailurePhase;
using controller::ControllerMutationOptions;
using controller::ControllerOperation;
using controller::ControllerSelection;
using controller::LinkMapperErrorCode;
using controller::LinkMapperException;
using controller::LinkMapperOperation;
using controller::PlaceAndRouteOptions;
using controller::PlaceAndRouteResult;
using controller::PreparedController;
using controller::ResourceId;
using controller::ResourceUpdatorErrorCode;
using controller::ResourceUpdatorException;
using controller::ShortestPathMethod;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
ControllerException expect_controller_error(
    Callable&& callable,
    ControllerErrorCode code,
    ControllerOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const ControllerException& error)
    {
        expect(error.code() == code, "controller error code mismatch");
        expect(error.operation() == operation,
               "controller operation mismatch");
        return error;
    }
    throw std::runtime_error("expected ControllerException");
}

template <typename Callable>
ResourceUpdatorException expect_resource_error(
    Callable&& callable,
    ResourceUpdatorErrorCode code)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const ResourceUpdatorException& error)
    {
        expect(error.code() == code,
               "resource updater error code mismatch");
        return error;
    }
    throw std::runtime_error("expected ResourceUpdatorException");
}

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = owner == AttributeOwner::node
        ? CheckingLevel::node
        : CheckingLevel::link;
    return result;
}

AttributeFactorySpec status_spec(
    std::string name,
    AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::status;
    return result;
}

network::NodeAttributeDataUpdate dense_node_update(
    ResourceId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    ResourceId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    expect(binding.has_value(), "missing fixture node binding");
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing fixture link binding");
    return *binding;
}

struct PhysicalValues
{
    std::array<std::int64_t, 5U> cpu{{10, 10, 10, 10, 10}};
    std::int64_t first_bw = 10;
    std::int64_t second_bw = 10;
};

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{2U, 0U}, {2U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bw", AttributeOwner::link)};

    network::VirtualNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto bw = require_link_binding(result, "bw");
    result.set_node_attrs_data({dense_node_update(
        cpu.registry_id,
        {std::int64_t{2}, std::int64_t{3}, std::int64_t{4}})});
    result.set_link_attrs_data({sparse_link_update(
        bw.registry_id,
        {{2U, 0U, std::int64_t{3}},
         {2U, 1U, std::int64_t{4}}})});
    result.set_request_id(41);
    result.set_arrival_time(2.0);
    result.set_lifetime(8.0);
    return result;
}

network::PhysicalNetwork make_physical_network(
    const PhysicalValues& values)
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        5U, std::vector<EdgeEndpoints>{{2U, 0U}, {2U, 4U}});
    // The status definitions force physical registry IDs to differ from the
    // virtual IDs. Selected values are still bound once by their cold names.
    construction.config.node_attribute_specs = {
        status_spec("node_status", AttributeOwner::node),
        resource_spec("cpu", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        status_spec("link_status", AttributeOwner::link),
        resource_spec("bw", AttributeOwner::link)};

    network::PhysicalNetwork result(std::move(construction));
    const auto cpu = require_node_binding(result, "cpu");
    const auto bw = require_link_binding(result, "bw");
    result.set_node_attrs_data({dense_node_update(
        cpu.registry_id,
        {values.cpu[0U], values.cpu[1U], values.cpu[2U],
         values.cpu[3U], values.cpu[4U]})});
    result.set_link_attrs_data({sparse_link_update(
        bw.registry_id,
        {{2U, 0U, values.first_bw}, {2U, 4U, values.second_bw}})});
    return result;
}

ControllerSelection make_selection(
    ResourceId node_resource,
    ResourceId link_resource)
{
    ControllerSelection result;
    result.constraints.node_at_node = {node_resource};
    result.constraints.link_at_link = {link_resource};
    result.node_resources = {node_resource};
    result.link_resources = {link_resource};
    result.hard_node_constraints = {node_resource};
    result.hard_link_constraints = {link_resource};
    return result;
}

core::Solution make_solution(bool result = true)
{
    core::SolutionMetadata metadata;
    metadata.v_net_id = 41;
    metadata.v_net_lifetime = 8.0;
    metadata.v_net_arrival_time = 2.0;
    metadata.v_net_num_nodes = 3U;
    metadata.v_net_num_edges = 2U;
    core::Solution solution(metadata);
    solution.result = result;
    return solution;
}

struct Fixture
{
    network::VirtualNetwork virtual_network;
    network::PhysicalNetwork physical_network;
    network::NodeNetworkAttributeBinding virtual_cpu;
    network::LinkNetworkAttributeBinding virtual_bw;
    network::NodeNetworkAttributeBinding physical_cpu;
    network::LinkNetworkAttributeBinding physical_bw;
    Controller controller;
    PreparedController prepared;

    explicit Fixture(PhysicalValues values = {})
        : virtual_network(make_virtual_network()),
          physical_network(make_physical_network(values)),
          virtual_cpu(require_node_binding(virtual_network, "cpu")),
          virtual_bw(require_link_binding(virtual_network, "bw")),
          physical_cpu(require_node_binding(physical_network, "cpu")),
          physical_bw(require_link_binding(physical_network, "bw")),
          controller(make_selection(
              virtual_cpu.registry_id, virtual_bw.registry_id)),
          prepared(controller.prepare(virtual_network, physical_network))
    {
        expect(virtual_cpu.registry_id == virtual_bw.registry_id,
               "fixture must overlap independent virtual registry IDs");
        expect(virtual_cpu.registry_id != physical_cpu.registry_id &&
                   virtual_bw.registry_id != physical_bw.registry_id,
               "fixture must exercise independent network registries");
    }

    std::int64_t cpu(Vertex node) const
    {
        return std::get<std::int64_t>(
            physical_network.graph().node_attrs(node).at(
                physical_cpu.value_id));
    }

    std::int64_t bw(Vertex source, Vertex target) const
    {
        const Edge edge = physical_network.graph().edge(source, target);
        return std::get<std::int64_t>(
            physical_network.graph().edge_attrs(edge).at(
                physical_bw.value_id));
    }
};

double numeric_value(
    const core::SolutionAttributeValues& values,
    ConstraintId id,
    std::string_view message)
{
    const auto* value = values.find(id);
    expect(value != nullptr, message);
    if (const auto* integer = std::get_if<std::int64_t>(value))
    {
        return static_cast<double>(*integer);
    }
    if (const auto* floating = std::get_if<double>(value))
    {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(value))
    {
        return *boolean ? 1.0 : 0.0;
    }
    throw std::runtime_error("unexpected solution numeric lane");
}

const core::SolutionAttributeValues& require_node_values(
    const core::NodeConstraintTable& table,
    core::SolutionNodeId node,
    std::string_view message)
{
    const auto id = table.find_id(node);
    expect(id.has_value(), message);
    return table.at(*id);
}

const core::SolutionAttributeValues& require_link_values(
    const core::LinkConstraintTable& table,
    core::SolutionLink link,
    std::string_view message)
{
    const auto id = table.find_id(link);
    expect(id.has_value(), message);
    return table.at(*id);
}

bool slot_equals(
    const core::Solution& solution,
    core::SolutionNodeId virtual_node,
    core::SolutionNodeId physical_node)
{
    const auto id = solution.node_slots.find_id(virtual_node);
    return id.has_value() && solution.node_slots.at(*id) == physical_node;
}

struct PhysicalSnapshot
{
    std::array<std::int64_t, 5U> cpu{};
    std::array<std::int64_t, 2U> bw{};

    friend bool operator==(
        const PhysicalSnapshot& left,
        const PhysicalSnapshot& right)
    {
        return left.cpu == right.cpu && left.bw == right.bw;
    }
};

PhysicalSnapshot physical_snapshot(const Fixture& fixture)
{
    PhysicalSnapshot result;
    for (Vertex node = 0U; node < 5U; ++node)
    {
        result.cpu[node] = fixture.cpu(node);
    }
    result.bw = {{fixture.bw(2U, 0U), fixture.bw(2U, 4U)}};
    return result;
}

struct PlacementSnapshot
{
    bool succeeded = false;
    ControllerFailurePhase failure_phase = ControllerFailurePhase::none;
    bool placement = false;
    bool last_route_present = false;
    bool last_route_succeeded = false;
    std::size_t attempted_routes = 0U;
    bool result = false;
    bool place_result = false;
    bool route_result = false;
    core::NodeSlots node_slots;
    core::LinkPaths link_paths;
    core::NodeSlotsInfo node_info;
    core::LinkPathsInfo link_info;
    core::NodeConstraintTable node_offsets;
    core::LinkConstraintTable link_offsets;
    core::PathConstraintTable path_offsets;
    core::SolutionAttributeValues step_node;
    core::SolutionAttributeValues step_link;
    core::SolutionAttributeValues step_path;
    double single_hard = 0.0;
    double maximum_hard = 0.0;
    PhysicalSnapshot physical;

    friend bool operator==(
        const PlacementSnapshot& left,
        const PlacementSnapshot& right)
    {
        return left.succeeded == right.succeeded &&
            left.failure_phase == right.failure_phase &&
            left.placement == right.placement &&
            left.last_route_present == right.last_route_present &&
            left.last_route_succeeded == right.last_route_succeeded &&
            left.attempted_routes == right.attempted_routes &&
            left.result == right.result &&
            left.place_result == right.place_result &&
            left.route_result == right.route_result &&
            left.node_slots == right.node_slots &&
            left.link_paths == right.link_paths &&
            left.node_info == right.node_info &&
            left.link_info == right.link_info &&
            left.node_offsets == right.node_offsets &&
            left.link_offsets == right.link_offsets &&
            left.path_offsets == right.path_offsets &&
            left.step_node == right.step_node &&
            left.step_link == right.step_link &&
            left.step_path == right.step_path &&
            left.single_hard == right.single_hard &&
            left.maximum_hard == right.maximum_hard &&
            left.physical == right.physical;
    }
};

PlacementSnapshot placement_snapshot(
    const PlaceAndRouteResult& operation,
    const Fixture& fixture,
    const core::Solution& solution)
{
    PlacementSnapshot result;
    result.succeeded = operation.succeeded;
    result.failure_phase = operation.failure_phase;
    result.placement = operation.placement.placed;
    result.last_route_present = operation.last_route.has_value();
    result.last_route_succeeded = operation.last_route.has_value() &&
        operation.last_route->routed;
    result.attempted_routes = operation.attempted_routes;
    result.result = solution.result;
    result.place_result = solution.place_result;
    result.route_result = solution.route_result;
    result.node_slots = solution.node_slots;
    result.link_paths = solution.link_paths;
    result.node_info = solution.node_slots_info;
    result.link_info = solution.link_paths_info;
    result.node_offsets = solution.v_net_constraint_offsets.node_level;
    result.link_offsets = solution.v_net_constraint_offsets.link_level;
    result.path_offsets = solution.v_net_constraint_offsets.path_level;
    result.step_node = solution.v_net_single_step_constraint_offset.node_level;
    result.step_link = solution.v_net_single_step_constraint_offset.link_level;
    result.step_path = solution.v_net_single_step_constraint_offset.path_level;
    result.single_hard = solution.v_net_single_step_hard_constraint_offset;
    result.maximum_hard =
        solution.v_net_max_single_step_hard_constraint_violation;
    result.physical = physical_snapshot(fixture);
    return result;
}

PlaceAndRouteOptions placement_options(std::size_t workers)
{
    PlaceAndRouteOptions options;
    options.shortest_method = ShortestPathMethod::bfs_shortest;
    options.k = 1;
    options.workers.topology_constraint_workers = workers;
    options.workers.candidate_workers = workers;
    return options;
}

PlaceAndRouteResult place_three(
    Fixture& fixture,
    core::Solution& solution,
    std::size_t workers)
{
    const PlaceAndRouteOptions options = placement_options(workers);
    const PlaceAndRouteResult first = fixture.prepared.place_and_route(
        0U, 0U, solution, options);
    const PlaceAndRouteResult second = fixture.prepared.place_and_route(
        1U, 4U, solution, options);
    expect(first.succeeded && first.attempted_routes == 0U &&
               second.succeeded && second.attempted_routes == 0U,
           "fixture pre-placement failed");
    return fixture.prepared.place_and_route(2U, 2U, solution, options);
}

void test_place_without_routes_and_place_failure()
{
    Fixture fixture;
    core::Solution solution = make_solution();
    const PlaceAndRouteResult placed = fixture.prepared.place_and_route(
        0U, 0U, solution, placement_options(0U));
    expect(placed.succeeded &&
               placed.failure_phase == ControllerFailurePhase::none &&
               placed.placement.placed && !placed.last_route.has_value() &&
               placed.attempted_routes == 0U,
           "no-route placement result mismatch");
    expect(slot_equals(solution, 0, 0) &&
               solution.link_paths.empty() && fixture.cpu(0U) == 8 &&
               fixture.bw(2U, 0U) == 10 && fixture.bw(2U, 4U) == 10,
           "no-route placement mutation mismatch");
    const auto& raw = require_node_values(
        solution.v_net_constraint_offsets.node_level,
        0, "no-route placement omitted node offsets");
    expect(numeric_value(raw, fixture.virtual_cpu.registry_id,
                         "missing raw CPU offset") == -8.0 &&
               numeric_value(
                   solution.v_net_single_step_constraint_offset.node_level,
                   fixture.virtual_cpu.registry_id,
                   "missing node step offset") == -8.0 &&
               numeric_value(
                   solution.v_net_single_step_constraint_offset.link_level,
                   fixture.virtual_bw.registry_id,
                   "missing no-route link placeholder") == 0.0,
           "no-route step offset mismatch");
    expect(solution.v_net_single_step_hard_constraint_offset == 0.0 &&
               solution.v_net_max_single_step_hard_constraint_violation == 0.0,
           "no-route hard placeholder pooling mismatch");

    PhysicalValues insufficient;
    insufficient.cpu[0U] = 1;
    Fixture failure_fixture(insufficient);
    core::Solution failure = make_solution();
    const PlaceAndRouteResult rejected =
        failure_fixture.prepared.place_and_route(
            0U, 0U, failure, placement_options(1U));
    expect(!rejected.succeeded &&
               rejected.failure_phase == ControllerFailurePhase::place &&
               !rejected.placement.placed &&
               rejected.attempted_routes == 0U &&
               !rejected.last_route.has_value(),
           "placement failure result mismatch");
    expect(failure.node_slots.empty() &&
               failure_fixture.cpu(0U) == 1 &&
               !failure.place_result && !failure.result,
           "placement failure mutated resource/slot or retained flags");
    const auto& rejected_raw = require_node_values(
        failure.v_net_constraint_offsets.node_level,
        0, "placement failure omitted raw offsets");
    expect(numeric_value(
               rejected_raw,
               failure_fixture.virtual_cpu.registry_id,
               "missing rejected CPU offset") == 1.0 &&
               failure.v_net_single_step_constraint_offset.link_level.empty(),
           "placement failure step write order mismatch");
}

PlacementSnapshot run_multi_route(std::size_t workers)
{
    Fixture fixture;
    core::Solution solution = make_solution();
    const PlaceAndRouteResult result =
        place_three(fixture, solution, workers);
    return placement_snapshot(result, fixture, solution);
}

void test_multi_route_order_workers_and_skip()
{
    const PlacementSnapshot baseline = run_multi_route(0U);
    expect(baseline.succeeded &&
               baseline.failure_phase == ControllerFailurePhase::none &&
               baseline.placement && baseline.last_route_present &&
               baseline.last_route_succeeded &&
               baseline.attempted_routes == 2U,
           "multi-route result mismatch");
    expect(baseline.node_slots.size() == 3U &&
               baseline.link_paths.size() == 2U &&
               baseline.link_info.size() == 2U &&
               baseline.physical.cpu ==
                   std::array<std::int64_t, 5U>{{8, 10, 6, 10, 7}} &&
               baseline.physical.bw ==
                   std::array<std::int64_t, 2U>{{7, 6}},
           "multi-route physical/solution mutation mismatch");
    const auto& path_entries = baseline.link_paths.entries();
    expect(path_entries[0U].key == core::SolutionLink{2, 0} &&
               path_entries[1U].key == core::SolutionLink{2, 1},
           "incident routes did not follow adjacency order");
    expect(numeric_value(
               baseline.step_link,
               0U,
               "missing pooled link offset") == -6.0 &&
               baseline.single_hard == -6.0 &&
               baseline.maximum_hard == 0.0,
           "multi-route pooling/cumulative maximum mismatch");

    for (const std::size_t workers : {1U, 2U, 8U})
    {
        expect(run_multi_route(workers) == baseline,
               "place_and_route workers changed output/order");
    }

    Fixture skip_fixture;
    core::Solution skip_solution = make_solution();
    const PlaceAndRouteOptions options = placement_options(2U);
    expect(skip_fixture.prepared.place_and_route(
               0U, 0U, skip_solution, options).succeeded &&
               skip_fixture.prepared.place_and_route(
               1U, 4U, skip_solution, options).succeeded,
           "skip fixture pre-placement failed");
    skip_solution.link_paths.insert_or_assign(
        core::SolutionLink{0, 2}, std::vector<core::SolutionLink>{});
    const PlaceAndRouteResult skipped =
        skip_fixture.prepared.place_and_route(
            2U, 2U, skip_solution, options);
    expect(skipped.succeeded && skipped.attempted_routes == 1U &&
               skip_fixture.bw(2U, 0U) == 10 &&
               skip_fixture.bw(2U, 4U) == 6,
           "reverse-oriented stored route was not skipped");
}

void test_middle_route_failure_preserves_partial_state()
{
    PhysicalValues values;
    values.second_bw = 3;
    Fixture fixture(values);
    core::Solution solution = make_solution();
    const PlaceAndRouteResult result = place_three(fixture, solution, 8U);
    expect(!result.succeeded &&
               result.failure_phase == ControllerFailurePhase::route &&
               result.placement.placed && result.last_route.has_value() &&
               !result.last_route->routed && result.attempted_routes == 2U,
           "middle route failure result mismatch");
    expect(solution.node_slots.size() == 3U &&
               slot_equals(solution, 2, 2) &&
               fixture.cpu(2U) == 6 &&
               fixture.bw(2U, 0U) == 7 &&
               fixture.bw(2U, 4U) == 3,
           "middle route failure rolled back prior mutation");
    const auto& first_offsets = require_link_values(
        solution.v_net_constraint_offsets.link_level,
        {2, 0}, "missing successful first-route offsets");
    expect(numeric_value(
               first_offsets,
               fixture.virtual_bw.registry_id,
               "missing first-route bandwidth offset") == -7.0 &&
               numeric_value(
                   solution.v_net_single_step_constraint_offset.link_level,
                   fixture.virtual_bw.registry_id,
                   "missing failed pooled link offset") == 100.0 &&
               solution.v_net_single_step_hard_constraint_offset == 100.0 &&
               solution.v_net_max_single_step_hard_constraint_violation ==
                   100.0 &&
               !solution.route_result && !solution.result,
           "middle route failure pooling/flags mismatch");
}

void test_undo_success_missing_and_partial()
{
    Fixture fixture;
    core::Solution solution = make_solution();
    const PlaceAndRouteResult result = place_three(fixture, solution, 1U);
    expect(result.succeeded, "undo success fixture did not map");
    expect(fixture.prepared.undo_place_and_route(2U, solution),
           "undo_place_and_route returned false");
    expect(!solution.node_slots.contains(2) &&
               solution.node_slots.size() == 2U &&
               solution.link_paths.empty() &&
               solution.link_paths_info.empty() &&
               fixture.cpu(2U) == 10 &&
               fixture.bw(2U, 0U) == 10 && fixture.bw(2U, 4U) == 10 &&
               fixture.cpu(0U) == 8 && fixture.cpu(4U) == 7,
           "successful undo did not restore exact incident resources");
    expect_controller_error(
        [&]
        {
            static_cast<void>(
                fixture.prepared.undo_place_and_route(2U, solution));
        },
        ControllerErrorCode::missing_node_slot,
        ControllerOperation::undo_place_and_route);

    Fixture partial_fixture;
    core::Solution partial = make_solution();
    expect(place_three(partial_fixture, partial, 1U).succeeded,
           "partial undo fixture did not map");
    const auto& second = partial.link_paths.entries().at(1U);
    expect(!second.value.empty(), "partial undo fixture has empty path");
    const core::LinkPathInfoKey missing{
        second.key, second.value.front()};
    expect(partial.link_paths_info.erase(missing),
           "failed to remove partial undo route info");
    bool failed = false;
    try
    {
        static_cast<void>(
            partial_fixture.prepared.undo_place_and_route(2U, partial));
    }
    catch (const LinkMapperException& error)
    {
        expect(error.code() == LinkMapperErrorCode::route_info_not_found &&
                   error.operation() == LinkMapperOperation::undo_route,
               "partial undo dependency error mismatch");
        failed = true;
    }
    expect(failed, "partial undo unexpectedly succeeded");
    expect(!partial.node_slots.contains(2) &&
               partial_fixture.cpu(2U) == 10 &&
               partial_fixture.bw(2U, 0U) == 10 &&
               partial_fixture.bw(2U, 4U) == 6 &&
               partial.link_paths.size() == 1U,
           "partial undo did not retain Python restoration order");
}

core::SolutionAttributeValues resource_values(
    ResourceId id,
    std::int64_t value)
{
    core::SolutionAttributeValues result;
    result.set(id, value);
    return result;
}

core::Solution stored_solution(
    ResourceId node_resource,
    ResourceId link_resource,
    bool duplicate_targets)
{
    core::Solution result = make_solution();
    result.node_slots.insert_or_assign(0, 0);
    result.node_slots.insert_or_assign(1, duplicate_targets ? 0 : 4);
    result.node_slots_info.insert_or_assign(
        {0, 0}, resource_values(node_resource, 2));
    result.node_slots_info.insert_or_assign(
        {1, duplicate_targets ? 0 : 4},
        resource_values(node_resource, 3));

    result.link_paths.insert_or_assign(
        {2, 0}, {{2, 0}});
    result.link_paths.insert_or_assign(
        {2, 1}, duplicate_targets
            ? std::vector<core::SolutionLink>{{0, 2}}
            : std::vector<core::SolutionLink>{{2, 4}});
    result.link_paths_info.insert_or_assign(
        {{2, 0}, {2, 0}}, resource_values(link_resource, 3));
    result.link_paths_info.insert_or_assign(
        {{2, 1}, duplicate_targets
            ? core::SolutionLink{0, 2}
            : core::SolutionLink{2, 4}},
        resource_values(link_resource, 4));
    return result;
}

bool relevant_solution_equal(
    const core::Solution& left,
    const core::Solution& right)
{
    return left.result == right.result &&
        left.place_result == right.place_result &&
        left.route_result == right.route_result &&
        left.node_slots == right.node_slots &&
        left.link_paths == right.link_paths &&
        left.node_slots_info == right.node_slots_info &&
        left.link_paths_info == right.link_paths_info &&
        left.v_net_single_step_constraint_offset.node_level ==
            right.v_net_single_step_constraint_offset.node_level &&
        left.v_net_single_step_constraint_offset.link_level ==
            right.v_net_single_step_constraint_offset.link_level &&
        left.v_net_single_step_constraint_offset.path_level ==
            right.v_net_single_step_constraint_offset.path_level &&
        left.v_net_single_step_hard_constraint_offset ==
            right.v_net_single_step_hard_constraint_offset &&
        left.v_net_max_single_step_hard_constraint_violation ==
            right.v_net_max_single_step_hard_constraint_violation &&
        left.description == right.description;
}

void expect_deployed_disjoint(
    const Fixture& fixture,
    std::string_view message)
{
    expect(physical_snapshot(fixture) == PhysicalSnapshot{
               {{8, 10, 10, 10, 7}}, {{7, 6}}},
           message);
}

void expect_initial(const Fixture& fixture, std::string_view message)
{
    expect(physical_snapshot(fixture) == PhysicalSnapshot{
               {{10, 10, 10, 10, 10}}, {{10, 10}}},
           message);
}

void test_deploy_release_undo_deploy_and_workers()
{
    std::optional<PhysicalSnapshot> deployed_baseline;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        Fixture fixture;
        core::Solution solution = stored_solution(
            fixture.virtual_cpu.registry_id,
            fixture.virtual_bw.registry_id,
            false);
        expect(fixture.prepared.deploy(solution, {workers}),
               "disjoint deploy returned false");
        expect_deployed_disjoint(
            fixture, "disjoint deploy resource mismatch");
        if (!deployed_baseline.has_value())
        {
            deployed_baseline = physical_snapshot(fixture);
        }
        else
        {
            expect(physical_snapshot(fixture) == *deployed_baseline,
                   "deploy workers changed physical output");
        }
        expect(fixture.prepared.release(solution, {workers}),
               "disjoint release returned false");
        expect_initial(fixture, "release did not restore initial resources");
    }

    Fixture no_op_fixture;
    core::Solution no_op = stored_solution(
        no_op_fixture.virtual_cpu.registry_id,
        no_op_fixture.virtual_bw.registry_id,
        false);
    no_op.result = false;
    expect(!no_op_fixture.prepared.deploy(no_op, {8U}) &&
               !no_op_fixture.prepared.release(no_op, {8U}),
           "unsuccessful solution did not short-circuit lifecycle mutation");
    expect_initial(no_op_fixture, "false-result lifecycle mutated resources");

    Fixture undo_fixture;
    core::Solution undo_solution = stored_solution(
        undo_fixture.virtual_cpu.registry_id,
        undo_fixture.virtual_bw.registry_id,
        false);
    undo_solution.description = "must-survive-undo-deploy";
    expect(undo_fixture.prepared.deploy(undo_solution, {2U}),
           "undo_deploy fixture deployment failed");
    const core::Solution before = undo_solution;
    expect(undo_fixture.prepared.undo_deploy(undo_solution, {8U}),
           "undo_deploy returned false");
    expect_initial(undo_fixture, "undo_deploy did not release resources");
    expect(relevant_solution_equal(undo_solution, before),
           "undo_deploy reset or mutated the caller solution");
}

void test_duplicate_target_fallback_and_partial_replay()
{
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        Fixture fixture;
        core::Solution solution = stored_solution(
            fixture.virtual_cpu.registry_id,
            fixture.virtual_bw.registry_id,
            true);
        expect(fixture.prepared.deploy(solution, {workers}),
               "duplicate-target deploy returned false");
        expect(fixture.cpu(0U) == 5 && fixture.bw(2U, 0U) == 3 &&
                   fixture.bw(2U, 4U) == 10,
               "duplicate-target sequential fallback mismatch");
        expect(fixture.prepared.release(solution, {workers}),
               "duplicate-target release returned false");
        expect_initial(
            fixture, "duplicate-target release did not restore resources");
    }

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        PhysicalValues node_failure;
        node_failure.cpu[4U] = 1;
        Fixture fixture(node_failure);
        core::Solution solution = stored_solution(
            fixture.virtual_cpu.registry_id,
            fixture.virtual_bw.registry_id,
            false);
        expect_resource_error(
            [&]
            {
                static_cast<void>(fixture.prepared.deploy(
                    solution, ControllerMutationOptions{workers}));
            },
            ResourceUpdatorErrorCode::insufficient_resource);
        expect(fixture.cpu(0U) == 8 && fixture.cpu(4U) == 1 &&
                   fixture.bw(2U, 0U) == 10 &&
                   fixture.bw(2U, 4U) == 10,
               "node batch error did not replay scalar partial mutation");
    }

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        PhysicalValues link_failure;
        link_failure.second_bw = 3;
        Fixture fixture(link_failure);
        core::Solution solution = stored_solution(
            fixture.virtual_cpu.registry_id,
            fixture.virtual_bw.registry_id,
            false);
        expect_resource_error(
            [&]
            {
                static_cast<void>(fixture.prepared.deploy(
                    solution, ControllerMutationOptions{workers}));
            },
            ResourceUpdatorErrorCode::insufficient_resource);
        expect(fixture.cpu(0U) == 8 && fixture.cpu(4U) == 7 &&
                   fixture.bw(2U, 0U) == 7 &&
                   fixture.bw(2U, 4U) == 3,
               "link batch error did not preserve node/first-link mutation");
    }
}

void test_release_missing_info_partial_order()
{
    Fixture fixture;
    core::Solution solution = stored_solution(
        fixture.virtual_cpu.registry_id,
        fixture.virtual_bw.registry_id,
        false);
    expect(fixture.prepared.deploy(solution, {1U}),
           "release partial fixture deployment failed");
    expect(solution.link_paths_info.erase({{2, 1}, {2, 4}}),
           "failed to remove release link info");
    expect_controller_error(
        [&]
        {
            static_cast<void>(fixture.prepared.release(solution, {8U}));
        },
        ControllerErrorCode::missing_link_path_info,
        ControllerOperation::release);
    expect(fixture.cpu(0U) == 10 && fixture.cpu(4U) == 10 &&
               fixture.bw(2U, 0U) == 10 && fixture.bw(2U, 4U) == 6,
           "release missing-info error lost ordered partial restoration");
}

bool concurrent_lifecycle(std::size_t caller)
{
    Fixture fixture;
    core::Solution placement = make_solution();
    const std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    const std::size_t workers = widths[caller % widths.size()];
    if (!place_three(fixture, placement, workers).succeeded)
    {
        return false;
    }
    if (!fixture.prepared.undo_place_and_route(2U, placement) ||
        !fixture.prepared.undo_place_and_route(1U, placement) ||
        !fixture.prepared.undo_place_and_route(0U, placement))
    {
        return false;
    }
    if (!(physical_snapshot(fixture) == PhysicalSnapshot{
            {{10, 10, 10, 10, 10}}, {{10, 10}}}))
    {
        return false;
    }

    core::Solution stored = stored_solution(
        fixture.virtual_cpu.registry_id,
        fixture.virtual_bw.registry_id,
        caller % 2U != 0U);
    if (!fixture.prepared.deploy(stored, {workers}) ||
        !fixture.prepared.release(stored, {workers}))
    {
        return false;
    }
    return physical_snapshot(fixture) == PhysicalSnapshot{
        {{10, 10, 10, 10, 10}}, {{10, 10}}};
}

void test_concurrent_independent_networks()
{
    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.emplace_back(std::async(
            std::launch::async,
            [caller]
            {
                return concurrent_lifecycle(caller);
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(),
               "concurrent independent controller lifecycle drifted");
    }
}

} // namespace

int main()
{
    try
    {
        const auto run = [](std::string_view name, auto&& test)
        {
            try
            {
                test();
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(
                    std::string(name) + ": " + error.what());
            }
        };

        run("place/no-route/failure",
            test_place_without_routes_and_place_failure);
        run("multi-route/workers/skip",
            test_multi_route_order_workers_and_skip);
        run("middle-route partial",
            test_middle_route_failure_preserves_partial_state);
        run("undo", test_undo_success_missing_and_partial);
        run("deploy/release/undo/workers",
            test_deploy_release_undo_deploy_and_workers);
        run("duplicate/replay",
            test_duplicate_target_fallback_and_partial_replay);
        run("release partial", test_release_missing_info_partial_order);
        run("concurrent independent", test_concurrent_independent_networks);
        std::cout << "controller unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "controller unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
