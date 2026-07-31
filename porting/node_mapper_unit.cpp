#include "node_mapper.h"

#include <algorithm>
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
using controller::NodeMapper;
using controller::NodeMapperErrorCode;
using controller::NodeMapperException;
using controller::NodeMapperOperation;
using controller::NodeMapperSelection;
using controller::NodeMappingOptions;
using controller::NodeMatchingMethod;
using controller::NodePlacementOptions;
using controller::NodePlacementResult;
using controller::PreparedNodeMapper;
using controller::ResourceId;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
NodeMapperException expect_mapper_error(
    Callable&& callable,
    NodeMapperErrorCode code,
    NodeMapperOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const NodeMapperException& error)
    {
        expect(error.code() == code, "node mapper error code mismatch");
        expect(error.operation() == operation, "node mapper operation mismatch");
        return error;
    }
    throw std::runtime_error("expected NodeMapperException");
}

AttributeFactorySpec resource_spec(
    std::string name,
    ConstraintRestriction restriction)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::node;
    result.kind = AttributeKind::resource;
    result.restriction = restriction;
    result.checking_level = CheckingLevel::node;
    return result;
}

AttributeFactorySpec status_spec(std::string name)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::node;
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

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", ConstraintRestriction::hard),
        resource_spec("soft_limit", ConstraintRestriction::soft),
        resource_spec("memory", ConstraintRestriction::hard),
        status_spec("node_status")};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        status_spec("node_status"),
        resource_spec("memory", ConstraintRestriction::hard),
        resource_spec("soft_limit", ConstraintRestriction::soft),
        resource_spec("cpu", ConstraintRestriction::hard)};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    expect(binding.has_value(), "missing node fixture binding");
    return *binding;
}

core::Solution make_solution()
{
    core::SolutionMetadata metadata;
    metadata.v_net_id = 17;
    metadata.v_net_lifetime = 10.0;
    metadata.v_net_arrival_time = 2.0;
    metadata.v_net_num_nodes = 3U;
    metadata.v_net_num_edges = 0U;
    return core::Solution(metadata);
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();

    ResourceId virtual_cpu = 0U;
    ResourceId virtual_soft = 0U;
    ResourceId virtual_memory = 0U;
    ResourceId physical_cpu = 0U;
    ResourceId physical_soft = 0U;
    ResourceId physical_memory = 0U;
    AttrId physical_cpu_value = 0U;
    AttrId physical_memory_value = 0U;

    Fixture()
    {
        const auto v_cpu = require_node_binding(virtual_network, "cpu");
        const auto v_soft = require_node_binding(
            virtual_network, "soft_limit");
        const auto v_memory = require_node_binding(
            virtual_network, "memory");
        const auto p_cpu = require_node_binding(physical_network, "cpu");
        const auto p_soft = require_node_binding(
            physical_network, "soft_limit");
        const auto p_memory = require_node_binding(
            physical_network, "memory");

        virtual_cpu = v_cpu.registry_id;
        virtual_soft = v_soft.registry_id;
        virtual_memory = v_memory.registry_id;
        physical_cpu = p_cpu.registry_id;
        physical_soft = p_soft.registry_id;
        physical_memory = p_memory.registry_id;
        physical_cpu_value = p_cpu.value_id;
        physical_memory_value = p_memory.value_id;

        expect(
            virtual_cpu != physical_cpu &&
                virtual_soft != physical_soft &&
                virtual_memory != physical_memory,
            "fixture must exercise independent registry IDs");

        virtual_network.set_node_attrs_data({
            dense_node_update(
                virtual_cpu,
                {std::int64_t{5}, std::int64_t{2}, std::int64_t{1}}),
            dense_node_update(
                virtual_soft,
                {9.0, 1.0, 1.0}),
            dense_node_update(
                virtual_memory,
                {std::int64_t{2}, std::int64_t{2}, std::int64_t{1}})});

        physical_network.set_node_attrs_data({
            dense_node_update(
                physical_cpu,
                {std::int64_t{8}, std::int64_t{3},
                 std::int64_t{9}, std::int64_t{0}}),
            dense_node_update(
                physical_soft,
                {4.0, 10.0, 3.0, 0.0}),
            dense_node_update(
                physical_memory,
                {std::int64_t{6}, std::int64_t{5},
                 std::int64_t{4}, std::int64_t{0}})});
    }

    NodeMapperSelection default_selection() const
    {
        NodeMapperSelection selection;
        selection.node_constraints = {virtual_cpu, virtual_soft};
        selection.node_resources = {
            virtual_cpu, virtual_memory, virtual_cpu};
        selection.hard_constraints = {virtual_cpu};
        return selection;
    }

    PreparedNodeMapper prepare()
    {
        return NodeMapper(default_selection()).prepare(
            virtual_network, physical_network);
    }

    PreparedNodeMapper prepare(NodeMapperSelection selection)
    {
        return NodeMapper(std::move(selection)).prepare(
            virtual_network, physical_network);
    }

    std::int64_t physical_integer(Vertex node, AttrId value_id) const
    {
        return std::get<std::int64_t>(
            physical_network.graph().node_attrs(node).at(value_id));
    }

    std::int64_t cpu(Vertex node) const
    {
        return physical_integer(node, physical_cpu_value);
    }

    std::int64_t memory(Vertex node) const
    {
        return physical_integer(node, physical_memory_value);
    }
};

template <typename OrderedTable, typename Key>
const typename OrderedTable::Entry& require_entry(
    const OrderedTable& table,
    const Key& key,
    std::string_view message)
{
    const auto id = table.find_id(key);
    expect(id.has_value(), message);
    const auto& entries = table.entries();
    expect(
        static_cast<std::size_t>(id->value) < entries.size(), message);
    return entries[id->value];
}

const core::SolutionAttributeValues& require_node_values(
    const core::NodeConstraintTable& table,
    core::SolutionNodeId virtual_node,
    std::string_view message)
{
    return require_entry(table, virtual_node, message).value;
}

const core::SolutionAttributeValues& require_node_values(
    const core::NodeViolationTable& table,
    core::SolutionNodeId virtual_node,
    std::string_view message)
{
    return require_entry(table, virtual_node, message).value;
}

void expect_integer(
    const core::SolutionAttributeValues& values,
    ConstraintId id,
    std::int64_t expected,
    std::string_view message)
{
    const auto* value = values.find(id);
    expect(value != nullptr, message);
    const auto* integer = std::get_if<std::int64_t>(value);
    expect(integer != nullptr && *integer == expected, message);
}

void expect_double(
    const core::SolutionAttributeValues& values,
    ConstraintId id,
    double expected,
    std::string_view message)
{
    const auto* value = values.find(id);
    expect(value != nullptr, message);
    const auto* floating = std::get_if<double>(value);
    expect(floating != nullptr && *floating == expected, message);
}

core::SolutionNodeId mapped_node(
    const core::Solution& solution,
    core::SolutionNodeId virtual_node)
{
    const auto id = solution.node_slots.find_id(virtual_node);
    expect(id.has_value(), "missing node slot");
    return solution.node_slots.at(*id);
}

const core::SolutionAttributeValues& placement_info(
    const core::Solution& solution,
    core::SolutionNodeId virtual_node,
    core::SolutionNodeId physical_node)
{
    const core::NodeSlotInfoKey key{virtual_node, physical_node};
    const auto id = solution.node_slots_info.find_id(key);
    expect(id.has_value(), "missing node slot info");
    return solution.node_slots_info.at(*id);
}

void test_selection_and_safe_unsafe_place()
{
    Fixture fixture;
    const NodeMapperSelection selection = fixture.default_selection();
    const NodeMapper mapper(selection);
    expect(
        mapper.selection().node_constraints == selection.node_constraints &&
            mapper.selection().node_resources == selection.node_resources &&
            mapper.selection().hard_constraints == selection.hard_constraints,
        "mapper did not retain typed selection fields");

    PreparedNodeMapper prepared = fixture.prepare();
    core::Solution success = make_solution();
    NodePlacementOptions safe_options;
    safe_options.record_constraint_violation = true;
    const NodePlacementResult placed = prepared.place(
        0U, 0U, success, safe_options);

    expect(placed.placed && placed.check.feasible, "safe placement failed");
    expect(mapped_node(success, 0) == 0, "safe placement slot mismatch");
    expect(fixture.cpu(0U) == 3, "duplicate cpu resource was not collapsed");
    expect(fixture.memory(0U) == 4, "memory resource subtraction mismatch");
    const auto& info = placement_info(success, 0, 0);
    expect_integer(info, fixture.virtual_cpu, 5, "recorded cpu demand mismatch");
    expect_integer(
        info, fixture.virtual_memory, 2,
        "recorded memory demand mismatch");
    expect(
        info.find(fixture.virtual_soft) == nullptr,
        "constraint-only value leaked into resource info");
    expect_integer(
        placed.check.offsets, fixture.virtual_cpu, -3,
        "safe hard offset mismatch");
    expect_double(
        placed.check.offsets, fixture.virtual_soft, 5.0,
        "safe soft offset mismatch");

    Fixture failure_fixture;
    PreparedNodeMapper failure_mapper = failure_fixture.prepare();
    core::Solution failure = make_solution();
    NodePlacementOptions no_record;
    no_record.record_constraint_violation = false;
    const NodePlacementResult rejected = failure_mapper.place(
        0U, 1U, failure, no_record);
    expect(
        !rejected.placed && !rejected.check.feasible,
        "safe infeasible placement was accepted");
    expect(
        failure.node_slots.empty() && failure.node_slots_info.empty(),
        "safe failure changed solution mappings");
    expect(
        failure_fixture.cpu(1U) == 3 &&
            failure_fixture.memory(1U) == 5,
        "safe failure changed physical resources");

    Fixture unsafe_fixture;
    PreparedNodeMapper unsafe_mapper = unsafe_fixture.prepare();
    core::Solution unsafe = make_solution();
    NodePlacementOptions unsafe_options;
    unsafe_options.allow_constraint_violation = true;
    unsafe_options.record_constraint_violation = true;
    const NodePlacementResult forced = unsafe_mapper.place(
        0U, 1U, unsafe, unsafe_options);
    expect(
        forced.placed && !forced.check.feasible,
        "unsafe placement did not retain infeasible check result");
    expect(mapped_node(unsafe, 0) == 1, "unsafe placement slot mismatch");
    expect(
        unsafe_fixture.cpu(1U) == -2 &&
            unsafe_fixture.memory(1U) == 3,
        "unsafe placement did not subtract with safe=false");
    expect(
        unsafe.v_net_total_hard_constraint_violation == 2.0,
        "unsafe hard violation accumulation mismatch");
}

void test_record_violation_and_empty_hard_order()
{
    Fixture fixture;
    PreparedNodeMapper prepared = fixture.prepare();
    core::Solution solution = make_solution();

    core::SolutionAttributeValues first;
    first.set(fixture.virtual_cpu, std::int64_t{-3});
    first.set(fixture.virtual_soft, 5.0);
    prepared.record_place_constraint_violation(0U, first, solution);

    const auto& first_raw = require_node_values(
        solution.v_net_constraint_offsets.node_level,
        0, "missing raw node offsets");
    const auto& first_violation = require_node_values(
        solution.v_net_constraint_violations.node_level,
        0, "missing node violations");
    expect_integer(first_raw, fixture.virtual_cpu, -3, "raw hard offset drift");
    expect_double(first_raw, fixture.virtual_soft, 5.0, "raw soft offset drift");
    expect_integer(
        first_violation, fixture.virtual_cpu, 0,
        "negative hard offset was not clipped");
    expect_double(
        first_violation, fixture.virtual_soft, 5.0,
        "positive soft offset was changed");
    expect(
        solution.v_net_total_hard_constraint_violation == 0.0,
        "zero hard accumulation mismatch");

    core::SolutionAttributeValues second;
    second.set(fixture.virtual_cpu, std::int64_t{2});
    second.set(fixture.virtual_soft, -1.0);
    prepared.record_place_constraint_violation(0U, second, solution);
    expect(
        solution.v_net_total_hard_constraint_violation == 2.0,
        "repeated hard violation did not accumulate");
    const auto& second_raw = require_node_values(
        solution.v_net_constraint_offsets.node_level,
        0, "missing replaced raw node offsets");
    const auto& second_violation = require_node_values(
        solution.v_net_constraint_violations.node_level,
        0, "missing replaced node violations");
    expect_integer(second_raw, fixture.virtual_cpu, 2, "raw replacement mismatch");
    expect_integer(
        second_violation, fixture.virtual_cpu, 2,
        "hard violation replacement mismatch");
    expect_integer(
        second_violation, fixture.virtual_soft, 0,
        "negative soft offset was not clipped");

    core::SolutionAttributeValues third;
    third.set(fixture.virtual_cpu, std::int64_t{1});
    third.set(fixture.virtual_soft, 0.0);
    prepared.record_place_constraint_violation(1U, third, solution);
    expect(
        solution.v_net_total_hard_constraint_violation == 3.0,
        "hard violation accumulation across nodes mismatch");

    Fixture empty_fixture;
    NodeMapperSelection empty_selection;
    empty_selection.node_constraints = {empty_fixture.virtual_soft};
    PreparedNodeMapper empty_mapper = empty_fixture.prepare(
        std::move(empty_selection));
    core::Solution empty_solution = make_solution();
    core::SolutionAttributeValues soft_offset;
    soft_offset.set(empty_fixture.virtual_soft, 4.0);

    const NodeMapperException error = expect_mapper_error(
        [&]
        {
            empty_mapper.record_place_constraint_violation(
                0U, soft_offset, empty_solution);
        },
        NodeMapperErrorCode::empty_hard_constraint_offsets,
        NodeMapperOperation::record_violation);
    expect(
        error.virtual_node() == std::optional<Vertex>{0U},
        "empty-hard error virtual node mismatch");
    expect(
        empty_solution.v_net_constraint_offsets.node_level.size() == 1U &&
            empty_solution.v_net_constraint_violations.node_level.size() == 1U,
        "empty-hard error did not preserve first two writes");
    expect(
        empty_solution.v_net_total_hard_constraint_violation == 0.0,
        "empty-hard error changed accumulated total");
}

void test_undo_place()
{
    Fixture fixture;
    PreparedNodeMapper prepared = fixture.prepare();
    core::Solution solution = make_solution();
    NodePlacementOptions options;
    options.record_constraint_violation = false;

    expect(
        prepared.place(0U, 0U, solution, options).placed,
        "undo fixture placement failed");
    expect(fixture.cpu(0U) == 3 && fixture.memory(0U) == 4,
           "undo fixture resources were not subtracted");
    expect(prepared.undo_place(0U, solution), "undo returned false");
    expect(
        fixture.cpu(0U) == 8 && fixture.memory(0U) == 6,
        "undo did not restore resources in stored order");
    expect(
        solution.node_slots.empty() && solution.node_slots_info.empty(),
        "undo did not erase placement entries");
    expect_mapper_error(
        [&]
        {
            static_cast<void>(prepared.undo_place(0U, solution));
        },
        NodeMapperErrorCode::placement_not_found,
        NodeMapperOperation::undo_place);
}

void test_greedy_l2s2_reusable_and_clone()
{
    Fixture greedy_fixture;
    PreparedNodeMapper greedy_mapper = greedy_fixture.prepare();
    core::Solution greedy_solution = make_solution();
    greedy_solution.result = true;
    NodeMappingOptions greedy_options;
    greedy_options.method = NodeMatchingMethod::greedy;
    expect(
        greedy_mapper.node_mapping(
            {0U, 1U, 2U}, {1U, 0U, 2U},
            greedy_solution, greedy_options),
        "greedy mapping failed");
    expect(
        mapped_node(greedy_solution, 0) == 0 &&
            mapped_node(greedy_solution, 1) == 1 &&
            mapped_node(greedy_solution, 2) == 2,
        "greedy first-feasible order mismatch");

    Fixture l2s2_failure_fixture;
    PreparedNodeMapper l2s2_failure_mapper = l2s2_failure_fixture.prepare();
    core::Solution l2s2_failure = make_solution();
    l2s2_failure.result = true;
    NodeMappingOptions l2s2_options;
    l2s2_options.method = NodeMatchingMethod::l2s2;
    expect(
        !l2s2_failure_mapper.node_mapping(
            {0U, 1U, 2U}, {1U, 0U, 2U},
            l2s2_failure, l2s2_options),
        "l2s2 did not fail on first infeasible pair");
    expect(
        l2s2_failure.node_slots.empty(),
        "failed first l2s2 pair changed slots");

    Fixture l2s2_success_fixture;
    PreparedNodeMapper l2s2_success_mapper = l2s2_success_fixture.prepare();
    core::Solution l2s2_success = make_solution();
    expect(
        l2s2_success_mapper.node_mapping(
            {0U, 1U, 2U}, {0U, 1U, 2U},
            l2s2_success, l2s2_options),
        "l2s2 aligned mapping failed");

    Fixture reusable_fixture;
    PreparedNodeMapper reusable_mapper = reusable_fixture.prepare();
    core::Solution reusable_solution = make_solution();
    NodeMappingOptions reusable_options;
    reusable_options.reusable = true;
    expect(
        reusable_mapper.node_mapping(
            {0U, 1U, 2U}, {0U},
            reusable_solution, reusable_options),
        "reusable mapping failed");
    expect(
        mapped_node(reusable_solution, 0) == 0 &&
            mapped_node(reusable_solution, 1) == 0 &&
            mapped_node(reusable_solution, 2) == 0 &&
            reusable_fixture.cpu(0U) == 0 &&
            reusable_fixture.memory(0U) == 1,
        "reusable resource/order semantics mismatch");

    Fixture clone_fixture;
    PreparedNodeMapper clone_mapper = clone_fixture.prepare();
    core::Solution clone_solution = make_solution();
    NodeMappingOptions clone_options;
    clone_options.inplace = false;
    const std::array<std::int64_t, 3U> initial_cpu{
        clone_fixture.cpu(0U), clone_fixture.cpu(1U),
        clone_fixture.cpu(2U)};
    expect(
        clone_mapper.node_mapping(
            {0U, 1U, 2U}, {0U, 1U, 2U},
            clone_solution, clone_options),
        "inplace=false clone mapping failed");
    expect(
        clone_solution.node_slots.size() == 3U &&
            clone_fixture.cpu(0U) == initial_cpu[0] &&
            clone_fixture.cpu(1U) == initial_cpu[1] &&
            clone_fixture.cpu(2U) == initial_cpu[2],
        "inplace=false mutated original physical network");
}

void test_mapping_failure_and_boundary_errors()
{
    Fixture failure_fixture;
    PreparedNodeMapper failure_mapper = failure_fixture.prepare();
    core::Solution failure = make_solution();
    failure.result = true;
    failure.place_result = true;
    expect(
        !failure_mapper.node_mapping(
            {0U, 1U, 2U}, {3U}, failure),
        "all-infeasible greedy mapping unexpectedly succeeded");
    expect(
        !failure.result && !failure.place_result &&
            failure.node_slots.empty(),
        "mapping failure flags or slots mismatch");
    const auto& failure_offsets = require_node_values(
        failure.v_net_constraint_offsets.node_level,
        0, "mapping failure did not record last reachable offsets");
    expect_integer(
        failure_offsets,
        failure_fixture.virtual_cpu, 5,
        "mapping failure did not record last reachable offset");

    Fixture empty_fixture;
    PreparedNodeMapper empty_mapper = empty_fixture.prepare();
    core::Solution empty_solution = make_solution();
    empty_solution.node_slots.insert_or_assign(99, 99);
    expect_mapper_error(
        [&]
        {
            static_cast<void>(empty_mapper.node_mapping(
                {0U, 1U, 2U}, {}, empty_solution));
        },
        NodeMapperErrorCode::empty_physical_candidates,
        NodeMapperOperation::node_mapping);
    expect(
        empty_solution.node_slots.empty(),
        "empty-candidate error occurred before canonical clear");

    Fixture incomplete_fixture;
    PreparedNodeMapper incomplete_mapper = incomplete_fixture.prepare();
    core::Solution incomplete = make_solution();
    expect_mapper_error(
        [&]
        {
            static_cast<void>(incomplete_mapper.node_mapping(
                {0U, 1U}, {0U, 1U}, incomplete));
        },
        NodeMapperErrorCode::mapping_cardinality_mismatch,
        NodeMapperOperation::node_mapping);
    expect(
        incomplete.node_slots.size() == 2U,
        "incomplete mapping error did not preserve prior placements");

    Fixture unsupported_fixture;
    PreparedNodeMapper unsupported_mapper = unsupported_fixture.prepare();
    core::Solution unsupported = make_solution();
    unsupported.node_slots.insert_or_assign(77, 88);
    NodeMappingOptions unsupported_options;
    unsupported_options.allow_constraint_violation = true;
    expect_mapper_error(
        [&]
        {
            static_cast<void>(unsupported_mapper.node_mapping(
                {0U, 1U, 2U}, {0U, 1U, 2U},
                unsupported, unsupported_options));
        },
        NodeMapperErrorCode::unsupported_constraint_violation_mapping,
        NodeMapperOperation::node_mapping);
    expect(
        unsupported.node_slots.contains(77),
        "unsupported mode cleared solution before validation");
}

struct MappingSnapshot
{
    core::NodeSlots slots;
    core::NodeSlotsInfo info;
    core::NodeConstraintTable offsets;
    core::NodeViolationTable violations;
    std::vector<attribute::AttributeNumber> last_violations;
    double total_hard = 0.0;
    std::array<std::int64_t, 4U> cpu{};
    std::array<std::int64_t, 4U> memory{};

    friend bool operator==(
        const MappingSnapshot& left,
        const MappingSnapshot& right)
    {
        return left.slots == right.slots &&
            left.info == right.info &&
            left.offsets == right.offsets &&
            left.violations == right.violations &&
            left.last_violations == right.last_violations &&
            left.total_hard == right.total_hard &&
            left.cpu == right.cpu && left.memory == right.memory;
    }
};

MappingSnapshot run_mapping_snapshot(std::size_t workers)
{
    Fixture fixture;
    PreparedNodeMapper mapper = fixture.prepare();
    core::Solution solution = make_solution();
    solution.result = true;
    solution.place_result = true;
    NodeMappingOptions options;
    options.candidate_workers = workers;
    expect(
        mapper.node_mapping(
            {0U, 1U, 2U}, {1U, 0U, 2U}, solution, options),
        "worker snapshot mapping failed");

    MappingSnapshot snapshot;
    snapshot.slots = solution.node_slots;
    snapshot.info = solution.node_slots_info;
    snapshot.offsets = solution.v_net_constraint_offsets.node_level;
    snapshot.violations = solution.v_net_constraint_violations.node_level;
    snapshot.last_violations = solution.v_net_single_step_violation_list;
    snapshot.total_hard = solution.v_net_total_hard_constraint_violation;
    for (Vertex node = 0U; node < 4U; ++node)
    {
        snapshot.cpu[node] = fixture.cpu(node);
        snapshot.memory[node] = fixture.memory(node);
    }
    return snapshot;
}

void test_worker_equality_and_later_error_suppression()
{
    const MappingSnapshot reference = run_mapping_snapshot(1U);
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(
            run_mapping_snapshot(workers) == reference,
            "candidate worker output/order drift");
    }

    Fixture fixture;
    PreparedNodeMapper mapper = fixture.prepare();
    core::Solution solution = make_solution();
    NodeMappingOptions options;
    options.reusable = true;
    options.candidate_workers = 8U;
    expect(
        mapper.node_mapping(
            {0U, 1U, 2U}, {1U, 0U, 99U}, solution, options),
        "later invalid candidate was not suppressed");
    expect(
        mapped_node(solution, 0) == 0 &&
            mapped_node(solution, 1) == 1 &&
            mapped_node(solution, 2) == 1,
        "later-error suppression changed first-feasible order");

    Fixture window_fixture;
    PreparedNodeMapper window_mapper = window_fixture.prepare();
    core::Solution window_solution = make_solution();
    std::vector<Vertex> window_candidates(128U, 2U);
    std::fill_n(window_candidates.begin(), 16U, Vertex{1U});
    window_candidates[16U] = 0U;
    window_candidates[17U] = 99U;
    NodeMappingOptions window_options;
    window_options.reusable = true;
    window_options.candidate_workers = 8U;
    expect(
        window_mapper.node_mapping(
            {0U, 1U, 2U},
            window_candidates,
            window_solution,
            window_options),
        "ordered candidate window mapping failed");
    expect(
        mapped_node(window_solution, 0) == 0 &&
            mapped_node(window_solution, 1) == 1 &&
            mapped_node(window_solution, 2) == 1,
        "ordered candidate window changed first-feasible/error order");
}

void test_concurrent_independent_callers()
{
    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.emplace_back(std::async(
            std::launch::async,
            [caller]
            {
                const std::array<std::size_t, 4U> widths{0U, 1U, 2U, 8U};
                for (std::size_t iteration = 0U; iteration < 4U; ++iteration)
                {
                    Fixture fixture;
                    PreparedNodeMapper mapper = fixture.prepare();
                    core::Solution solution = make_solution();
                    NodeMappingOptions options;
                    options.candidate_workers =
                        widths[(caller + iteration) % widths.size()];
                    if (!mapper.node_mapping(
                            {0U, 1U, 2U}, {1U, 0U, 2U},
                            solution, options) ||
                        solution.node_slots.size() != 3U ||
                        mapped_node(solution, 0) != 0 ||
                        mapped_node(solution, 1) != 1 ||
                        mapped_node(solution, 2) != 2)
                    {
                        return false;
                    }
                }
                return true;
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(), "concurrent independent node mapper drift");
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

        run("selection/safe/unsafe", test_selection_and_safe_unsafe_place);
        run("record/empty-hard", test_record_violation_and_empty_hard_order);
        run("undo", test_undo_place);
        run("greedy/l2s2/reusable/clone",
            test_greedy_l2s2_reusable_and_clone);
        run("failure/boundaries", test_mapping_failure_and_boundary_errors);
        run("workers/error suppression",
            test_worker_equality_and_later_error_suppression);
        run("concurrent callers", test_concurrent_independent_callers);

        std::cout << "node mapper unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "node mapper unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
