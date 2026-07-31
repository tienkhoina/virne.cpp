#include "recorder.h"

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace core = virne::core;
namespace network = virne::network;
namespace utils = virne::utils;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using core::Counter;
using core::Recorder;
using core::RecorderConfig;
using core::RecorderErrorCode;
using core::RecorderException;
using core::RecorderInitialPhysicalState;
using core::RecorderOperation;
using core::RecorderOptions;
using core::RecorderRecord;
using core::RecorderState;
using core::Solution;
using core::SolutionMetadata;

void expect(const bool condition, const std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
RecorderException expect_recorder_error(
    Callable&& callable,
    const RecorderErrorCode code,
    const RecorderOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const RecorderException& error)
    {
        expect(error.code() == code, "recorder error code mismatch");
        expect(error.operation() == operation,
               "recorder operation mismatch");
        expect(!std::string_view(error.what()).empty(),
               "recorder diagnostic is empty");
        return error;
    }
    throw std::runtime_error("expected RecorderException");
}

struct ScopedDirectory
{
    explicit ScopedDirectory(std::string_view label)
    {
        static std::uint64_t sequence = 0U;
        const auto now = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("virne_recorder_unit_" + std::string(label) + "_" +
             std::to_string(now) + "_" + std::to_string(sequence++));
    }

    ~ScopedDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    ScopedDirectory(const ScopedDirectory&) = delete;
    ScopedDirectory& operator=(const ScopedDirectory&) = delete;

    std::filesystem::path path;
};

RecorderConfig recorder_config(
    const std::filesystem::path& root,
    std::string run_id,
    const bool temporary_records = false)
{
    RecorderConfig result;
    result.save_root_dir = root;
    result.solver_name = "recorder-unit";
    result.run_id = std::move(run_id);
    result.temporary_records = temporary_records;
    return result;
}

AttributeFactorySpec resource_spec(
    std::string name,
    const AttributeOwner owner)
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

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    const std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    expect(binding.has_value(), "missing fixture node resource binding");
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    const std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing fixture link resource binding");
    return *binding;
}

network::NodeAttributeDataUpdate dense_node_update(
    const attribute::AttributeRegistryId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    const attribute::AttributeRegistryId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

network::PhysicalNetwork make_physical_network(
    const std::array<double, 3U> node_values = {10.0, 20.0, 30.0},
    const std::array<double, 2U> link_values = {5.0, 7.0})
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bandwidth", AttributeOwner::link)};

    network::PhysicalNetwork result(std::move(construction));
    const auto node = require_node_binding(result, "cpu");
    const auto link = require_link_binding(result, "bandwidth");
    result.set_node_attrs_data({dense_node_update(
        node.registry_id,
        {node_values[0U], node_values[1U], node_values[2U]})});
    result.set_link_attrs_data({sparse_link_update(
        link.registry_id,
        {{0U, 1U, link_values[0U]}, {1U, 2U, link_values[1U]}})});
    return result;
}

network::VirtualNetwork make_virtual_network(
    const std::array<double, 2U> node_values = {2.0, 4.0},
    const double link_value = 3.0)
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bandwidth", AttributeOwner::link)};

    network::VirtualNetwork result(std::move(construction));
    const auto node = require_node_binding(result, "cpu");
    const auto link = require_link_binding(result, "bandwidth");
    result.set_node_attrs_data({dense_node_update(
        node.registry_id, {node_values[0U], node_values[1U]})});
    result.set_link_attrs_data({sparse_link_update(
        link.registry_id, {{0U, 1U, link_value}})});
    result.set_lifetime(1.0);
    return result;
}

enum class RouteShape : std::uint8_t
{
    normal,
    empty,
};

Solution make_success_solution(
    network::VirtualNetwork& virtual_network,
    const core::SolutionNodeId virtual_network_id,
    const core::SolutionNodeId first_physical_node,
    const core::SolutionNodeId second_physical_node,
    const RouteShape route_shape = RouteShape::normal)
{
    SolutionMetadata metadata;
    metadata.v_net_id = virtual_network_id;
    metadata.v_net_lifetime = 1.0;
    metadata.v_net_arrival_time =
        static_cast<double>(virtual_network_id + 1);
    metadata.v_net_num_nodes = 2U;
    metadata.v_net_num_edges = 1U;
    Solution result(metadata);
    result.result = true;
    result.description = "original";
    result.v_net_reward = static_cast<double>(virtual_network_id);
    result.node_slots.insert_or_assign(0, first_physical_node);
    result.node_slots.insert_or_assign(1, second_physical_node);

    const core::SolutionLink virtual_link{0, 1};
    if (route_shape == RouteShape::empty)
    {
        result.link_paths.insert_or_assign(virtual_link, {});
        return result;
    }

    const std::vector<core::SolutionLink> path{{0, 1}, {1, 2}};
    result.link_paths.insert_or_assign(virtual_link, path);
    const auto link_binding =
        require_link_binding(virtual_network, "bandwidth");
    core::SolutionAttributeValues route_info;
    route_info.set(link_binding.registry_id, 3.0);
    for (const core::SolutionLink physical_link : path)
    {
        result.link_paths_info.insert_or_assign(
            {virtual_link, physical_link}, route_info);
    }
    return result;
}

Solution make_failure_solution(const core::SolutionNodeId virtual_network_id)
{
    SolutionMetadata metadata;
    metadata.v_net_id = virtual_network_id;
    metadata.v_net_lifetime = 1.0;
    metadata.v_net_arrival_time =
        static_cast<double>(virtual_network_id + 1);
    metadata.v_net_num_nodes = 2U;
    metadata.v_net_num_edges = 1U;
    Solution result(metadata);
    result.result = false;
    return result;
}

bool same_event(
    const std::optional<core::RecorderEvent>& left,
    const std::optional<core::RecorderEvent>& right)
{
    if (left.has_value() != right.has_value())
    {
        return false;
    }
    return !left.has_value() ||
        (left->event_id == right->event_id && left->type == right->type);
}

bool same_state(const RecorderState& left, const RecorderState& right)
{
    return same_event(left.event, right.event) &&
        left.virtual_network_count == right.virtual_network_count &&
        left.success_count == right.success_count &&
        left.inservice_count == right.inservice_count &&
        left.total_revenue == right.total_revenue &&
        left.total_cost == right.total_cost &&
        left.total_time_revenue == right.total_time_revenue &&
        left.total_time_cost == right.total_time_cost &&
        left.long_term_r2c_ratio == right.long_term_r2c_ratio &&
        left.long_term_time_r2c_ratio ==
            right.long_term_time_r2c_ratio &&
        left.running_physical_node_count ==
            right.running_physical_node_count &&
        left.physical_available_resource ==
            right.physical_available_resource &&
        left.physical_node_available_resource ==
            right.physical_node_available_resource &&
        left.physical_link_available_resource ==
            right.physical_link_available_resource &&
        left.physical_node_resource_utilization ==
            right.physical_node_resource_utilization &&
        left.physical_link_resource_utilization ==
            right.physical_link_resource_utilization;
}

bool same_initial(
    const RecorderInitialPhysicalState& left,
    const RecorderInitialPhysicalState& right)
{
    return left.available_resource == right.available_resource &&
        left.node_available_resource == right.node_available_resource &&
        left.link_available_resource == right.link_available_resource;
}

void test_arrival_leave_membership_and_history()
{
    ScopedDirectory directory("transitions");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    Recorder recorder(
        Counter{}, recorder_config(directory.path, "transitions"));
    recorder.count_initial_physical_network(physical_network, {2U});

    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution first = make_success_solution(virtual_network, 10, 2, 2);
    RecorderRecord first_record = recorder.count(
        virtual_network, physical_network, first, {2U});
    expect(first_record.state.virtual_network_count == 1 &&
               first_record.state.success_count == 1 &&
               first_record.state.inservice_count == 1,
           "successful arrival counters mismatch");
    expect(first_record.state.running_physical_node_count == 1U &&
               recorder.running_physical_nodes() ==
                   std::vector<core::SolutionNodeId>{2},
           "duplicate node slots lost multiplicity semantics");
    recorder.add_record(std::move(first_record));

    recorder.set_event({1, network::VirtualEventType::arrival});
    Solution failure = make_failure_solution(20);
    RecorderRecord failure_record = recorder.count(
        virtual_network, physical_network, failure, {1U});
    expect(failure_record.state.virtual_network_count == 2 &&
               failure_record.state.success_count == 1 &&
               failure_record.state.inservice_count == 1 &&
               failure_record.state.running_physical_node_count == 1U,
           "failed arrival changed success/service state");
    recorder.add_record(std::move(failure_record));

    recorder.set_event({2, network::VirtualEventType::arrival});
    Solution second = make_success_solution(virtual_network, 30, 1, 2);
    recorder.add_record(recorder.count(
        virtual_network, physical_network, second, {8U}));
    expect(recorder.running_physical_nodes() ==
               (std::vector<core::SolutionNodeId>{2, 1}),
           "physical-node first-touch order mismatch");
    expect(recorder.state().running_physical_node_count == 2U &&
               recorder.state().inservice_count == 2,
           "second successful arrival membership mismatch");

    expect(recorder.record_by_event(-1).solution.v_net_id == 30 &&
               recorder.record_by_event(-3).solution.v_net_id == 10,
           "negative history index mismatch");
    expect(recorder.record_by_virtual_network(20).solution.result == false,
           "virtual-network arrival lookup mismatch");
    expect_recorder_error(
        [&recorder]
        {
            static_cast<void>(recorder.record_by_event(-4));
        },
        RecorderErrorCode::record_index_out_of_range,
        RecorderOperation::lookup_record);

    recorder.set_event({3, network::VirtualEventType::leave});
    Solution first_leave =
        make_success_solution(virtual_network, 10, 2, 2);
    first_leave.result = false;
    recorder.add_record(recorder.count(
        virtual_network, physical_network, first_leave, {0U}));
    expect(recorder.state().inservice_count == 1 &&
               recorder.state().running_physical_node_count == 2U &&
               recorder.running_physical_nodes() ==
                   (std::vector<core::SolutionNodeId>{2, 1}),
           "leave did not consume stored successful arrival/multiplicity");

    recorder.set_event({4, network::VirtualEventType::leave});
    Solution failed_leave = make_failure_solution(20);
    failed_leave.result = true;
    failed_leave.node_slots.insert_or_assign(0, 99);
    recorder.add_record(recorder.count(
        virtual_network, physical_network, failed_leave, {2U}));
    expect(recorder.state().inservice_count == 1 &&
               recorder.state().running_physical_node_count == 2U,
           "leave used current result instead of stored failed arrival");

    recorder.set_event({5, network::VirtualEventType::leave});
    Solution second_leave =
        make_success_solution(virtual_network, 30, 1, 2);
    second_leave.result = false;
    recorder.add_record(recorder.count(
        virtual_network, physical_network, second_leave, {8U}));
    expect(recorder.state().inservice_count == 0 &&
               recorder.state().running_physical_node_count == 0U &&
               recorder.running_physical_nodes().empty(),
           "final leave did not empty membership state");
    expect(recorder.memory().size() == 6U &&
               recorder.record_by_event(-1).state.event->event_id == 5,
           "history append order mismatch");
}

void test_precomputed_arrival_fast_path()
{
    ScopedDirectory directory("precomputed-arrival");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    const Counter counter{};

    Recorder regular(
        counter, recorder_config(directory.path, "precomputed-regular"));
    Recorder precomputed(
        counter, recorder_config(directory.path, "precomputed-fast"));
    regular.count_initial_physical_network(physical_network, {2U});
    precomputed.count_initial_physical_network(physical_network, {2U});
    regular.set_event({0, network::VirtualEventType::arrival});
    precomputed.set_event({0, network::VirtualEventType::arrival});

    Solution regular_solution =
        make_success_solution(virtual_network, 40, 0, 2);
    Solution precomputed_solution = regular_solution;
    const core::PreparedCounter prepared = counter.prepare(virtual_network);
    prepared.count_solution(precomputed_solution, {2U});

    const RecorderRecord regular_record = regular.count(
        virtual_network, physical_network, regular_solution, {2U});
    const RecorderRecord precomputed_record =
        precomputed.count_precomputed_arrival(
            physical_network, precomputed_solution, {2U});

    expect(same_state(regular_record.state, precomputed_record.state),
           "precomputed arrival recorder state mismatch");
    expect(regular_record.solution.v_net_demand ==
                   precomputed_record.solution.v_net_demand &&
               regular_record.solution.v_net_revenue ==
                   precomputed_record.solution.v_net_revenue &&
               regular_record.solution.v_net_cost ==
                   precomputed_record.solution.v_net_cost &&
               regular_record.solution.v_net_r2c_ratio ==
                   precomputed_record.solution.v_net_r2c_ratio,
           "precomputed arrival solution metrics mismatch");

    precomputed.set_event({1, network::VirtualEventType::leave});
    expect_recorder_error(
        [&]
        {
            static_cast<void>(precomputed.count_precomputed_arrival(
                physical_network, precomputed_solution, {1U}));
        },
        RecorderErrorCode::invalid_event_type,
        RecorderOperation::count);
}

struct WorkerFingerprint
{
    RecorderInitialPhysicalState initial;
    RecorderState state;
    double demand = 0.0;
    double revenue = 0.0;
    double cost = 0.0;
    double ratio = 0.0;
    std::vector<core::SolutionNodeId> running_nodes;
};

bool same_fingerprint(
    const WorkerFingerprint& left,
    const WorkerFingerprint& right)
{
    return same_initial(left.initial, right.initial) &&
        same_state(left.state, right.state) &&
        left.demand == right.demand &&
        left.revenue == right.revenue &&
        left.cost == right.cost &&
        left.ratio == right.ratio &&
        left.running_nodes == right.running_nodes;
}

void test_workers_and_reset_baseline()
{
    ScopedDirectory directory("workers");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    std::optional<WorkerFingerprint> reference;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        Recorder recorder(
            Counter{},
            recorder_config(
                directory.path, "workers-" + std::to_string(workers)));
        recorder.count_initial_physical_network(
            physical_network, {workers});
        recorder.set_event({0, network::VirtualEventType::arrival});
        Solution solution =
            make_success_solution(virtual_network, 40, 0, 2);
        const RecorderRecord record = recorder.count(
            virtual_network, physical_network, solution, {workers});
        expect(recorder.initial_physical_state().has_value(),
               "worker run lost initial physical state");
        WorkerFingerprint current{
            *recorder.initial_physical_state(),
            record.state,
            record.solution.v_net_demand,
            record.solution.v_net_revenue,
            record.solution.v_net_cost,
            record.solution.v_net_r2c_ratio,
            recorder.running_physical_nodes(),
        };
        if (!reference.has_value())
        {
            reference = current;
        }
        else
        {
            expect(same_fingerprint(*reference, current),
                   "workers 0/1/2/8 changed Recorder output");
        }
    }

    Recorder recorder(
        Counter{}, recorder_config(directory.path, "reset"));
    recorder.count_initial_physical_network(physical_network, {2U});
    const RecorderInitialPhysicalState baseline =
        *recorder.initial_physical_state();
    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution before_reset =
        make_success_solution(virtual_network, 41, 1, 2);
    recorder.add_record(recorder.count(
        virtual_network, physical_network, before_reset, {2U}));
    recorder.reset();
    expect(recorder.initial_physical_state().has_value() &&
               same_initial(*recorder.initial_physical_state(), baseline),
           "reset discarded the prepared initial baseline");
    expect(same_state(recorder.state(), RecorderState{}) &&
               recorder.memory().empty() &&
               recorder.running_physical_nodes().empty(),
           "reset did not clear event/history/membership state");

    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution after_reset =
        make_success_solution(virtual_network, 42, 1, 2);
    const RecorderRecord record = recorder.count(
        virtual_network, physical_network, after_reset, {8U});
    expect(record.state.success_count == 1 &&
               record.state.physical_node_resource_utilization == 0.0 &&
               record.state.physical_link_resource_utilization == 0.0,
           "retained reset baseline could not serve a new count");
}

void test_partial_error_order()
{
    ScopedDirectory directory("errors");
    auto virtual_network = make_virtual_network();

    auto physical_without_baseline = make_physical_network();
    Recorder missing(
        Counter{}, recorder_config(directory.path, "missing-baseline"));
    missing.set_event({0, network::VirtualEventType::arrival});
    Solution missing_solution = make_failure_solution(50);
    expect_recorder_error(
        [&]
        {
            static_cast<void>(missing.count(
                virtual_network,
                physical_without_baseline,
                missing_solution,
                {2U}));
        },
        RecorderErrorCode::missing_initial_physical_state,
        RecorderOperation::update_state);
    expect(missing.state().physical_available_resource == 72.0 &&
               missing.state().physical_node_available_resource == 60.0 &&
               missing.state().physical_link_available_resource == 12.0 &&
               missing.state().virtual_network_count == 0,
           "missing-baseline error lost preceding physical totals");

    auto zero_node_physical = make_physical_network(
        {0.0, 0.0, 0.0}, {5.0, 7.0});
    Recorder zero_node(
        Counter{}, recorder_config(directory.path, "zero-node"));
    zero_node.count_initial_physical_network(zero_node_physical);
    zero_node.set_event({0, network::VirtualEventType::arrival});
    Solution zero_node_solution = make_failure_solution(51);
    expect_recorder_error(
        [&]
        {
            static_cast<void>(zero_node.count(
                virtual_network,
                zero_node_physical,
                zero_node_solution));
        },
        RecorderErrorCode::zero_initial_node_resource,
        RecorderOperation::update_state);
    expect(zero_node.state().physical_available_resource == 12.0 &&
               zero_node.state().physical_node_available_resource == 0.0 &&
               zero_node.state().physical_link_available_resource == 12.0 &&
               zero_node.state().virtual_network_count == 0,
           "zero-node error wrote state out of contract order");

    auto zero_link_physical = make_physical_network(
        {10.0, 20.0, 30.0}, {0.0, 0.0});
    Recorder zero_link(
        Counter{}, recorder_config(directory.path, "zero-link"));
    zero_link.count_initial_physical_network(zero_link_physical);
    zero_link.set_event({0, network::VirtualEventType::arrival});
    Solution zero_link_solution = make_failure_solution(52);
    expect_recorder_error(
        [&]
        {
            static_cast<void>(zero_link.count(
                virtual_network,
                zero_link_physical,
                zero_link_solution));
        },
        RecorderErrorCode::zero_initial_link_resource,
        RecorderOperation::update_state);
    expect(zero_link.state().physical_node_resource_utilization == 0.0 &&
               zero_link.state().virtual_network_count == 0,
           "zero-link error lost preceding node utilization");

    auto physical_for_ratio = make_physical_network();
    Recorder excessive(
        Counter{}, recorder_config(directory.path, "excessive-ratio"));
    excessive.count_initial_physical_network(physical_for_ratio);
    excessive.set_event({0, network::VirtualEventType::arrival});
    Solution excessive_solution = make_success_solution(
        virtual_network, 53, 0, 1, RouteShape::empty);
    expect_recorder_error(
        [&]
        {
            static_cast<void>(excessive.count(
                virtual_network,
                physical_for_ratio,
                excessive_solution,
                {8U}));
        },
        RecorderErrorCode::invalid_time_ratio,
        RecorderOperation::update_state);
    expect(excessive.state().virtual_network_count == 1 &&
               excessive.state().success_count == 1 &&
               excessive.state().inservice_count == 1 &&
               excessive.state().long_term_time_r2c_ratio > 1.0 &&
               excessive.state().running_physical_node_count == 0U &&
               excessive.running_physical_nodes().empty(),
           "excessive ratio did not retain pre-membership partial state");

    auto nan_virtual_network = make_virtual_network(
        {std::numeric_limits<double>::quiet_NaN(), 4.0}, 3.0);
    Recorder nan_ratio(
        Counter{}, recorder_config(directory.path, "nan-ratio"));
    nan_ratio.count_initial_physical_network(physical_for_ratio);
    nan_ratio.set_event({0, network::VirtualEventType::arrival});
    Solution nan_solution =
        make_success_solution(nan_virtual_network, 54, 0, 1);
    expect_recorder_error(
        [&]
        {
            static_cast<void>(nan_ratio.count(
                nan_virtual_network,
                physical_for_ratio,
                nan_solution,
                {2U}));
        },
        RecorderErrorCode::invalid_time_ratio,
        RecorderOperation::update_state);
    expect(nan_ratio.state().virtual_network_count == 1 &&
               nan_ratio.state().success_count == 1 &&
               nan_ratio.state().inservice_count == 1 &&
               std::isnan(nan_ratio.state().long_term_time_r2c_ratio) &&
               nan_ratio.running_physical_nodes().empty(),
           "NaN ratio did not fail before membership mutation");

    Recorder invalid_node(
        Counter{}, recorder_config(directory.path, "invalid-node"));
    invalid_node.count_initial_physical_network(physical_for_ratio);
    invalid_node.set_event({0, network::VirtualEventType::arrival});
    Solution invalid_node_solution =
        make_success_solution(virtual_network, 55, 1, 99);
    const RecorderException invalid_node_error = expect_recorder_error(
        [&]
        {
            static_cast<void>(invalid_node.count(
                virtual_network,
                physical_for_ratio,
                invalid_node_solution));
        },
        RecorderErrorCode::invalid_physical_node,
        RecorderOperation::update_state);
    expect(invalid_node_error.physical_node_id() ==
               std::optional<core::SolutionNodeId>{99} &&
               invalid_node.state().success_count == 1 &&
               invalid_node.state().running_physical_node_count == 0U &&
               invalid_node.running_physical_nodes() ==
                   std::vector<core::SolutionNodeId>{1},
           "invalid-node error lost observable partial membership order");
}

void test_deep_record_snapshots()
{
    ScopedDirectory directory("snapshots");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    Recorder recorder(
        Counter{}, recorder_config(directory.path, "snapshots"));
    recorder.count_initial_physical_network(physical_network);
    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution solution =
        make_success_solution(virtual_network, 60, 0, 2);
    RecorderRecord counted = recorder.count(
        virtual_network, physical_network, solution);
    solution.description = "caller-mutated";
    solution.node_slots.insert_or_assign(0, 1);
    expect(counted.solution.description == "original" &&
               counted.solution.node_slots.entries().front().value == 0,
           "count result did not snapshot Solution by value");

    auto list = std::make_shared<utils::ClassAnyList>();
    list->push_back(std::int64_t{7});
    utils::ClassDict extra;
    extra.set_value("payload", list);
    RecorderRecord candidate(counted.state, counted.solution, extra);
    list->front() = std::int64_t{8};
    const auto candidate_id = candidate.extra.find_field_id("payload");
    expect(candidate_id.has_value(), "candidate extra field is missing");
    const auto& candidate_list = std::any_cast<const utils::ClassAnyListPtr&>(
        candidate.extra.at(*candidate_id));
    expect(std::any_cast<std::int64_t>(candidate_list->front()) == 7,
           "RecorderRecord construction shallow-copied ClassDict");

    const RecorderRecord& stored = recorder.add_record(candidate);
    candidate.solution.description = "candidate-mutated";
    auto& mutable_candidate_list =
        std::any_cast<utils::ClassAnyListPtr&>(
            candidate.extra.at(*candidate_id));
    mutable_candidate_list->front() = std::int64_t{9};
    const auto stored_id = stored.extra.find_field_id("payload");
    expect(stored_id.has_value(), "stored extra field is missing");
    const auto& stored_list = std::any_cast<const utils::ClassAnyListPtr&>(
        stored.extra.at(*stored_id));
    expect(stored.solution.description == "original" &&
               std::any_cast<std::int64_t>(stored_list->front()) == 7,
           "add_record shallow-copied Solution or ClassDict");
    expect(recorder.record_by_event(-1).solution.description == "original",
           "stored snapshot changed after caller mutation");

    expect_recorder_error(
        [&]
        {
            recorder.temporary_save_record(stored);
        },
        RecorderErrorCode::temporary_saving_disabled,
        RecorderOperation::temporary_save_record);
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    expect(stream.is_open(), "failed to open recorder CSV");
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

class FakeSummaryExtension final : public core::RecorderSummaryExtension
{
public:
    void append_columns(
        const Recorder& recorder,
        const core::CounterSummary& summary,
        std::vector<core::RecorderSummaryColumn>& columns) const override
    {
        ++calls;
        observed_recorder = &recorder;
        observed_success_count = summary.success_count;
        columns.push_back({"prepared_metric", "ready"});
    }

    mutable std::size_t calls = 0U;
    mutable const Recorder* observed_recorder = nullptr;
    mutable std::int64_t observed_success_count = -1;
};

void test_csv_paths_and_cold_summary_extension()
{
    ScopedDirectory directory("csv");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    Recorder recorder(
        Counter{}, recorder_config(directory.path, "csv", true));
    const auto extension = std::make_shared<FakeSummaryExtension>();
    recorder.set_summary_extension(extension);
    recorder.count_initial_physical_network(physical_network, {2U});

    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution success =
        make_success_solution(virtual_network, 70, 0, 2);
    RecorderRecord success_record = recorder.count(
        virtual_network, physical_network, success, {2U});
    success_record.extra.set_value("tag", std::string{"alpha"});
    recorder.add_record(std::move(success_record));

    recorder.set_event({1, network::VirtualEventType::arrival});
    Solution failure = make_failure_solution(71);
    RecorderRecord failure_record = recorder.count(
        virtual_network, physical_network, failure, {8U});
    failure_record.extra.set_value("tag", std::string{"beta"});
    recorder.add_record(std::move(failure_record));

    expect(extension->calls == 0U,
           "summary extension entered the count/history hot path");
    expect(recorder.temp_save_path().has_value() &&
               std::filesystem::exists(*recorder.temp_save_path()),
           "temporary record CSV was not created");
    const std::filesystem::path temp_path = *recorder.temp_save_path();
    const std::string temp_contents = read_file(temp_path);
    expect(temp_contents.find("tag") != std::string::npos &&
               temp_contents.find("alpha") != std::string::npos &&
               temp_contents.find("beta") != std::string::npos &&
               std::count(temp_contents.begin(), temp_contents.end(), '\n') ==
                   3,
           "temporary CSV header/ordered appends mismatch");

    expect_recorder_error(
        [&]
        {
            static_cast<void>(recorder.save_records("../escape.csv"));
        },
        RecorderErrorCode::invalid_filename,
        RecorderOperation::save_records);
    const std::filesystem::path records_path =
        recorder.save_records("records.csv", {8U});
    expect(std::filesystem::exists(records_path) &&
               !std::filesystem::exists(temp_path),
           "final record CSV or temporary cleanup mismatch");
    const std::string records_contents = read_file(records_path);
    expect(records_contents.find("alpha") != std::string::npos &&
               records_contents.find("beta") != std::string::npos &&
               std::count(
                   records_contents.begin(), records_contents.end(), '\n') ==
                   3,
           "parallel final CSV serialization mismatch");

    const core::CounterSummary summary = recorder.summary_records();
    expect(extension->calls == 0U,
           "summary extension entered summary calculation");
    expect_recorder_error(
        [&]
        {
            static_cast<void>(
                recorder.append_summary(summary, "..\\escape.csv"));
        },
        RecorderErrorCode::invalid_filename,
        RecorderOperation::append_summary);
    expect(extension->calls == 0U,
           "invalid summary path invoked extension before validation");

    const std::filesystem::path summary_path =
        recorder.append_summary(summary, "summary.csv");
    expect(extension->calls == 1U &&
               extension->observed_recorder == &recorder &&
               extension->observed_success_count == 1,
           "cold summary extension call/context mismatch");
    const std::string summary_contents = read_file(summary_path);
    expect(summary_contents.find("prepared_metric") != std::string::npos &&
               summary_contents.find("ready") != std::string::npos,
           "summary extension columns were not serialized");

    RecorderConfig bad = recorder_config(directory.path, "bad-constructor");
    bad.solver_name = "../escape";
    expect_recorder_error(
        [&]
        {
            Recorder rejected(Counter{}, bad);
            static_cast<void>(rejected);
        },
        RecorderErrorCode::invalid_filename,
        RecorderOperation::construct);
}

} // namespace

int main()
{
    try
    {
        const auto run = [](const std::string_view name, auto&& test)
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

        run("arrival/leave/history",
            test_arrival_leave_membership_and_history);
        run("precomputed arrival", test_precomputed_arrival_fast_path);
        run("workers/reset", test_workers_and_reset_baseline);
        run("partial errors", test_partial_error_order);
        run("CSV/paths/summary extension",
            test_csv_paths_and_cold_summary_extension);
        run("deep snapshots", test_deep_record_snapshots);
        std::cout << "recorder unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "recorder unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
