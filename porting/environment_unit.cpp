#include "environment.h"

#include "attribute/attribute_factory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
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
namespace core = virne::core;
namespace network = virne::network;

using core::EnvironmentErrorCode;
using core::EnvironmentException;
using core::EnvironmentFailureReason;
using core::EnvironmentOperation;
using core::EnvironmentPhase;
using core::EnvironmentStepResult;
using core::Solution;
using core::SolutionStepEnvironment;

void expect(const bool condition, const std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
void expect_environment_error(
    Callable&& callable,
    const EnvironmentErrorCode code,
    const EnvironmentOperation operation,
    const std::optional<std::size_t> schedule_index = std::nullopt,
    const std::optional<network::VirtualRequestId> request_id = std::nullopt)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const EnvironmentException& error)
    {
        expect(error.code() == code, "environment error code mismatch");
        expect(error.operation() == operation,
               "environment error operation mismatch");
        expect(error.schedule_index() == schedule_index,
               "environment error schedule index mismatch");
        expect(error.request_id() == request_id,
               "environment error request ID mismatch");
        expect(!std::string_view(error.what()).empty(),
               "environment error diagnostic is empty");
        return;
    }
    throw std::runtime_error("expected EnvironmentException");
}

struct ScopedDirectory
{
    explicit ScopedDirectory(const std::string_view label)
    {
        static std::atomic<std::uint64_t> sequence{0U};
        const auto now = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("virne_environment_unit_" + std::string(label) + "_" +
             std::to_string(now) + "_" +
             std::to_string(sequence.fetch_add(1U)));
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

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    const attribute::AttributeOwner owner)
{
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = owner == attribute::AttributeOwner::node
        ? attribute::CheckingLevel::node
        : attribute::CheckingLevel::link;
    return result;
}

network::BaseNetworkConstruction network_construction(Graph graph)
{
    network::BaseNetworkConstruction result;
    result.incoming_graph = std::move(graph);
    result.config.node_attribute_specs.push_back(
        resource_spec("cpu", attribute::AttributeOwner::node));
    result.config.link_attribute_specs.push_back(
        resource_spec("bandwidth", attribute::AttributeOwner::link));
    return result;
}

network::PhysicalNetwork make_physical_network()
{
    network::PhysicalNetwork result(network_construction(
        Graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    expect(cpu.has_value() && bandwidth.has_value(),
           "physical fixture binding failure");
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {100.0, 200.0, 300.0}}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        bandwidth->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {400.0, 500.0}}});
    return result;
}

struct RequestInput
{
    std::optional<network::VirtualRequestId> request_id;
    double arrival_time = 0.0;
};

network::VirtualNetwork make_virtual_network(const RequestInput input)
{
    network::VirtualNetwork result(network_construction(
        Graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    expect(cpu.has_value() && bandwidth.has_value(),
           "virtual fixture binding failure");
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {10.0, 20.0}}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        bandwidth->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {30.0}}});
    if (input.request_id.has_value())
    {
        result.set_request_id(*input.request_id);
    }
    result.set_arrival_time(input.arrival_time);
    result.set_lifetime(1.0);
    return result;
}

network::VirtualNetworkRequestSimulator make_simulator(
    const std::vector<RequestInput>& requests,
    const std::vector<network::VirtualNetworkEventInput>& event_inputs)
{
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = requests.size();
    config.virtual_network_size.value_kind =
        virne::utils::DatasetValueKind::integer;
    config.virtual_network_size.distribution.kind =
        virne::utils::DistributionKind::uniform;
    config.virtual_network_size.distribution.low = std::int64_t{1};
    config.virtual_network_size.distribution.high = std::int64_t{2};
    config.lifetime.value_kind = virne::utils::DatasetValueKind::floating;
    config.lifetime.distribution.kind =
        virne::utils::DistributionKind::uniform;
    config.lifetime.distribution.low = 1.0;
    config.lifetime.distribution.high = 2.0;
    config.arrival_rate.value_kind =
        virne::utils::DatasetValueKind::floating;
    config.arrival_rate.distribution.kind =
        virne::utils::DistributionKind::uniform;
    config.arrival_rate.distribution.low = 1.0;
    config.arrival_rate.distribution.high = 2.0;
    config.topology_type = network::TopologyType::Path;
    config.node_attribute_specs.push_back(
        resource_spec("cpu", attribute::AttributeOwner::node));
    config.link_attribute_specs.push_back(
        resource_spec("bandwidth", attribute::AttributeOwner::link));

    std::vector<network::VirtualNetwork> virtual_networks;
    virtual_networks.reserve(requests.size());
    for (const RequestInput request : requests)
    {
        virtual_networks.push_back(make_virtual_network(request));
    }

    std::vector<network::VirtualNetworkEvent> events;
    events.reserve(event_inputs.size());
    for (const auto& input : event_inputs)
    {
        events.emplace_back(input);
    }
    return network::VirtualNetworkRequestSimulator::from_state(
        std::move(config),
        std::move(virtual_networks),
        std::move(events));
}

std::vector<network::VirtualNetworkEventInput> one_request_events(
    const network::VirtualRequestId request_id,
    const network::VirtualEventId arrival_id = 0U,
    const network::VirtualEventId leave_id = 1U)
{
    return {
        {arrival_id, network::VirtualEventType::arrival, request_id, 0.0},
        {leave_id, network::VirtualEventType::leave, request_id, 1.0},
    };
}

core::EnvironmentConfig make_config(
    const std::filesystem::path& root,
    std::string case_name,
    const std::size_t workers)
{
    core::EnvironmentConfig result;
    result.controller.constraints.node_at_node = {0U};
    result.controller.constraints.link_at_link = {0U};
    result.controller.node_resources = {0U};
    result.controller.link_resources = {0U};
    result.controller.hard_node_constraints = {0U};
    result.controller.hard_link_constraints = {0U};
    result.counter.node_resources = std::vector<core::CounterResourceId>{0U};
    result.counter.link_resources = std::vector<core::CounterResourceId>{0U};
    result.recorder.save_root_dir = root;
    result.recorder.solver_name = std::move(case_name);
    result.recorder.run_id = "run";
    result.recorder.temporary_records = false;
    result.workers = {workers, workers, workers};
    return result;
}

std::unique_ptr<SolutionStepEnvironment> make_environment(
    const std::filesystem::path& root,
    std::string case_name,
    const std::vector<RequestInput>& requests,
    const std::vector<network::VirtualNetworkEventInput>& events,
    const std::size_t workers = 1U)
{
    return std::make_unique<SolutionStepEnvironment>(
        make_physical_network(),
        make_simulator(requests, events),
        make_config(root, std::move(case_name), workers));
}

double numeric_value(const AttrValue& value)
{
    if (const auto* floating = std::get_if<double>(&value))
    {
        return *floating;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return static_cast<double>(*integer);
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? 1.0 : 0.0;
    }
    throw std::runtime_error("fixture resource is not numeric");
}

struct PhysicalSnapshot
{
    std::array<double, 3U> cpu{};
    std::array<double, 2U> bandwidth{};

    friend bool operator==(
        const PhysicalSnapshot& left,
        const PhysicalSnapshot& right)
    {
        return left.cpu == right.cpu && left.bandwidth == right.bandwidth;
    }
};

PhysicalSnapshot physical_snapshot(const network::PhysicalNetwork& network)
{
    const auto cpu = network.bind_node_attribute("cpu");
    const auto bandwidth = network.bind_link_attribute("bandwidth");
    expect(cpu.has_value() && bandwidth.has_value(),
           "snapshot binding failure");
    const auto& graph = network.graph();
    PhysicalSnapshot result;
    for (std::size_t node = 0U; node < result.cpu.size(); ++node)
    {
        result.cpu[node] = numeric_value(
            graph.node_attrs(node).at(cpu->value_id));
    }
    result.bandwidth[0U] = numeric_value(
        graph.edge_attrs(graph.edge(0U, 1U)).at(bandwidth->value_id));
    result.bandwidth[1U] = numeric_value(
        graph.edge_attrs(graph.edge(1U, 2U)).at(bandwidth->value_id));
    return result;
}

core::SolutionAttributeValues one_value(
    const core::CounterResourceId resource_id,
    const double value)
{
    core::SolutionAttributeValues result;
    result.set(resource_id, value);
    return result;
}

Solution accepted_solution(SolutionStepEnvironment& environment)
{
    Solution result = environment.make_solution();
    result.result = true;
    result.node_slots.insert_or_assign(0, 0);
    result.node_slots.insert_or_assign(1, 2);
    result.node_slots_info.insert_or_assign(
        {0, 0}, one_value(0U, 10.0));
    result.node_slots_info.insert_or_assign(
        {1, 2}, one_value(0U, 20.0));

    const core::SolutionLink virtual_link{0, 1};
    const std::vector<core::SolutionLink> path{{0, 1}, {1, 2}};
    result.link_paths.insert_or_assign(virtual_link, path);
    for (const core::SolutionLink physical_link : path)
    {
        result.link_paths_info.insert_or_assign(
            {virtual_link, physical_link}, one_value(0U, 30.0));
    }
    return result;
}

void expect_event_ids(
    const SolutionStepEnvironment& environment,
    const std::vector<std::int64_t>& expected)
{
    const auto& records = environment.recorder().memory();
    expect(records.size() == expected.size(), "record count mismatch");
    for (std::size_t index = 0U; index < expected.size(); ++index)
    {
        expect(records[index].state.event.has_value(),
               "record omitted event");
        expect(records[index].state.event->event_id == expected[index],
               "record event order mismatch");
    }
}

void test_sparse_index_dense_schedule_and_restore(
    const std::filesystem::path& root)
{
    const std::vector<RequestInput> requests{{42, 2.0}, {7, 0.0}};
    const std::vector<network::VirtualNetworkEventInput> events{
        {0U, network::VirtualEventType::arrival, 7, 0.0},
        {1U, network::VirtualEventType::leave, 7, 1.0},
        {2U, network::VirtualEventType::arrival, 42, 2.0},
        {3U, network::VirtualEventType::leave, 42, 3.0},
    };
    auto environment = make_environment(
        root, "sparse-index", requests, events, 2U);
    environment->reset();

    const auto initial = physical_snapshot(environment->physical_network());
    const auto& first = environment->current_event();
    expect(environment->state().phase == EnvironmentPhase::active &&
               environment->state().num_processed_events == 1U &&
               first.schedule_index == 0U && first.event_id == 0U &&
               first.request_id == 7 && first.request_index == 1U &&
               environment->current_virtual_network().request_id() ==
                   std::optional<network::VirtualRequestId>{7},
           "sparse request was not resolved to its dense index");

    Solution accepted = accepted_solution(*environment);
    const EnvironmentStepResult first_step = environment->step(accepted);
    expect(first_step.accepted && !first_step.done &&
               first_step.failure_reason == EnvironmentFailureReason::none &&
               first_step.auto_released_events == 1U &&
               !first_step.summary.has_value() &&
               accepted.description == "Success",
           "accepted arrival transition mismatch");
    const auto& second = environment->current_event();
    expect(second.schedule_index == 2U && second.event_id == 2U &&
               second.request_id == 42 && second.request_index == 0U &&
               environment->state().num_processed_events == 3U,
           "dense schedule cursor did not skip the consumed leave");
    expect(physical_snapshot(environment->physical_network()) == initial,
           "accepted arrival/leave did not exactly restore resources");
    expect_event_ids(*environment, {0, 1});
    const auto& first_records = environment->recorder().memory();
    expect(first_records[0U].solution.description == "Success" &&
               first_records[1U].solution.description == "Leave Event",
           "arrival/leave descriptions mismatch");
    expect(first_records[0U].state.physical_node_available_resource ==
               570.0 &&
               first_records[0U].state.physical_link_available_resource ==
                   840.0 &&
               first_records[1U].state.physical_node_available_resource ==
                   600.0 &&
               first_records[1U].state.physical_link_available_resource ==
                   900.0,
           "arrival was not deployed before leave restored resources");

    Solution rejected = environment->make_solution();
    rejected.result = false;
    rejected.place_result = false;
    rejected.route_result = false;
    const EnvironmentStepResult second_step = environment->step(rejected);
    expect(!second_step.accepted && second_step.done &&
               second_step.failure_reason ==
                   EnvironmentFailureReason::placement &&
               second_step.auto_released_events == 1U &&
               second_step.summary.has_value() &&
               rejected.description == "Place Failure",
           "rejected arrival transition mismatch");
    expect(environment->state().phase == EnvironmentPhase::finished &&
               environment->state().num_processed_events == 4U &&
               environment->recorder().state().virtual_network_count == 2 &&
               environment->recorder().state().success_count == 1 &&
               environment->recorder().state().inservice_count == 0,
           "finished environment/Recorder state mismatch");
    expect_event_ids(*environment, {0, 1, 2, 3});
    expect(physical_snapshot(environment->physical_network()) == initial,
           "rejected lifecycle changed physical resources");
    expect(second_step.summary->success_count == 1 &&
               second_step.summary->place_failure_count == 1U,
           "finished summary mismatch");
}

void test_rejection_reasons_and_hard_gate(
    const std::filesystem::path& root)
{
    struct RejectionCase
    {
        EnvironmentFailureReason reason;
        std::string_view description;
    };
    const std::array<RejectionCase, 4U> cases{{
        {EnvironmentFailureReason::early_rejection, "Early Rejection"},
        {EnvironmentFailureReason::placement, "Place Failure"},
        {EnvironmentFailureReason::routing, "Route Failure"},
        {EnvironmentFailureReason::unknown, "Unknown Reason"},
    }};

    for (std::size_t index = 0U; index < cases.size(); ++index)
    {
        auto environment = make_environment(
            root,
            "reject-" + std::to_string(index),
            {{42, 0.0}},
            one_request_events(42));
        environment->reset();
        const auto initial = physical_snapshot(environment->physical_network());
        Solution solution = environment->make_solution();
        solution.result = false;
        switch (cases[index].reason)
        {
        case EnvironmentFailureReason::early_rejection:
            solution.early_rejection = true;
            solution.place_result = false;
            solution.route_result = false;
            break;
        case EnvironmentFailureReason::placement:
            solution.place_result = false;
            solution.route_result = false;
            break;
        case EnvironmentFailureReason::routing:
            solution.route_result = false;
            break;
        case EnvironmentFailureReason::unknown:
        case EnvironmentFailureReason::none:
            break;
        }
        const auto step = environment->step(solution);
        expect(!step.accepted && step.done &&
                   step.failure_reason == cases[index].reason &&
                   step.auto_released_events == 1U &&
                   solution.description == cases[index].description,
               "rejection priority/description mismatch");
        expect(physical_snapshot(environment->physical_network()) == initial,
               "rejection mutated physical resources");
    }

    auto hard = make_environment(
        root, "hard-gate", {{42, 0.0}}, one_request_events(42));
    hard->reset();
    const auto initial = physical_snapshot(hard->physical_network());
    Solution solution = accepted_solution(*hard);
    solution.v_net_total_hard_constraint_violation = 1.0;
    const auto step = hard->step(solution);
    expect(!step.accepted && step.done && !solution.result &&
               step.failure_reason == EnvironmentFailureReason::unknown &&
               solution.description == "Unknown Reason",
           "hard violation gate did not reject before deployment");
    expect(physical_snapshot(hard->physical_network()) == initial,
           "hard violation gate mutated physical resources");
}

struct LifecycleSnapshot
{
    bool accepted = false;
    bool done = false;
    EnvironmentFailureReason reason = EnvironmentFailureReason::none;
    std::size_t released = 0U;
    EnvironmentPhase phase = EnvironmentPhase::unready;
    std::size_t processed = 0U;
    std::int64_t success_count = 0;
    std::int64_t inservice_count = 0;
    std::vector<std::string> descriptions;
    PhysicalSnapshot physical;

    friend bool operator==(
        const LifecycleSnapshot& left,
        const LifecycleSnapshot& right)
    {
        return left.accepted == right.accepted &&
            left.done == right.done && left.reason == right.reason &&
            left.released == right.released && left.phase == right.phase &&
            left.processed == right.processed &&
            left.success_count == right.success_count &&
            left.inservice_count == right.inservice_count &&
            left.descriptions == right.descriptions &&
            left.physical == right.physical;
    }
};

LifecycleSnapshot run_accepted_lifecycle(
    const std::filesystem::path& root,
    std::string label,
    const std::size_t workers)
{
    auto environment = make_environment(
        root,
        std::move(label),
        {{42, 0.0}},
        one_request_events(42),
        workers);
    environment->reset();
    const auto initial = physical_snapshot(environment->physical_network());
    Solution solution = accepted_solution(*environment);
    const auto step = environment->step(solution);
    LifecycleSnapshot result;
    result.accepted = step.accepted;
    result.done = step.done;
    result.reason = step.failure_reason;
    result.released = step.auto_released_events;
    result.phase = environment->state().phase;
    result.processed = environment->state().num_processed_events;
    result.success_count = environment->recorder().state().success_count;
    result.inservice_count = environment->recorder().state().inservice_count;
    for (const auto& record : environment->recorder().memory())
    {
        result.descriptions.push_back(record.solution.description);
    }
    result.physical = physical_snapshot(environment->physical_network());
    expect(result.physical == initial, "worker lifecycle did not restore");
    return result;
}

void test_workers_and_independent_environments(
    const std::filesystem::path& root)
{
    const LifecycleSnapshot baseline =
        run_accepted_lifecycle(root, "workers-0", 0U);
    expect(baseline.accepted && baseline.done &&
               baseline.reason == EnvironmentFailureReason::none &&
               baseline.released == 1U &&
               baseline.phase == EnvironmentPhase::finished &&
               baseline.processed == 2U && baseline.success_count == 1 &&
               baseline.inservice_count == 0 &&
               baseline.descriptions ==
                   std::vector<std::string>{"Success", "Leave Event"},
           "sequential environment lifecycle mismatch");
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        expect(run_accepted_lifecycle(
                   root,
                   "workers-" + std::to_string(workers),
                   workers) == baseline,
               "environment worker output mismatch");
    }

    auto left = std::async(std::launch::async, [&root]
    {
        return run_accepted_lifecycle(root, "independent-left", 2U);
    });
    auto right = std::async(std::launch::async, [&root]
    {
        return run_accepted_lifecycle(root, "independent-right", 8U);
    });
    expect(left.get() == baseline && right.get() == baseline,
           "independent concurrent environments interfered");
}

void test_reset_reprepares(const std::filesystem::path& root)
{
    auto environment = make_environment(
        root, "reset-reprepare", {{42, 0.0}}, one_request_events(42), 8U);
    environment->reset();
    const auto initial = physical_snapshot(environment->physical_network());
    Solution first = accepted_solution(*environment);
    expect(environment->step(first).done,
           "first reset cycle did not finish");

    environment->reset();
    expect(environment->state().phase == EnvironmentPhase::active &&
               environment->state().num_processed_events == 1U &&
               environment->current_event().schedule_index == 0U &&
               environment->recorder().memory().empty() &&
               physical_snapshot(environment->physical_network()) == initial,
           "reset did not clear episode state and restore baseline");
    Solution second = accepted_solution(*environment);
    const auto step = environment->step(second);
    expect(step.done && step.accepted &&
               environment->recorder().memory().size() == 2U &&
               physical_snapshot(environment->physical_network()) == initial,
           "reset did not rebuild prepared request views");
}

void test_failed_auto_drain_and_reset_recovery(
    const std::filesystem::path& root)
{
    const std::vector<network::VirtualNetworkEventInput> events{
        {0U, network::VirtualEventType::arrival, 42, 0.0},
        {1U, network::VirtualEventType::leave, 7, 1.0},
    };
    auto environment = make_environment(
        root,
        "failed-auto-drain",
        {{42, 0.0}, {7, 1.0}},
        events,
        2U);
    environment->reset();
    const auto initial = physical_snapshot(environment->physical_network());

    Solution solution = accepted_solution(*environment);
    expect_environment_error(
        [&] { static_cast<void>(environment->step(solution)); },
        EnvironmentErrorCode::missing_arrival_record,
        EnvironmentOperation::release,
        1U,
        7);
    expect(environment->state().phase == EnvironmentPhase::active &&
               environment->state().num_processed_events == 2U &&
               environment->current_event().schedule_index == 1U &&
               environment->current_event().type ==
                   network::VirtualEventType::leave &&
               environment->recorder().memory().size() == 1U &&
               environment->recorder().memory().front().state.event.has_value() &&
               environment->recorder().memory().front().state.event->event_id ==
                   0 &&
               environment->recorder().memory().front().solution.description ==
                   "Success",
           "failed auto-drain lost its dense arrival history/retry cursor");
    expect(!(physical_snapshot(environment->physical_network()) == initial),
           "failed auto-drain unexpectedly rolled back committed deployment");

    expect_environment_error(
        [&] { static_cast<void>(environment->drain_leaves()); },
        EnvironmentErrorCode::missing_arrival_record,
        EnvironmentOperation::release,
        1U,
        7);
    expect(environment->current_event().schedule_index == 1U &&
               environment->recorder().memory().size() == 1U,
           "explicit drain changed the deterministic retry point");

    environment->reset();
    expect(environment->state().phase == EnvironmentPhase::active &&
               environment->state().num_processed_events == 1U &&
               environment->current_event().schedule_index == 0U &&
               environment->recorder().memory().empty() &&
               physical_snapshot(environment->physical_network()) == initial,
           "reset did not recover from a failed automatic leave");
    expect_environment_error(
        [&] { static_cast<void>(environment->drain_leaves()); },
        EnvironmentErrorCode::leave_event_required,
        EnvironmentOperation::transit,
        0U,
        42);
}

void test_schedule_errors(const std::filesystem::path& root)
{
    auto unready = make_environment(
        root, "unready", {{42, 0.0}}, one_request_events(42));
    expect_environment_error(
        [&] { static_cast<void>(unready->current_event()); },
        EnvironmentErrorCode::environment_not_ready,
        EnvironmentOperation::access);

    auto empty = make_environment(root, "empty", {{42, 0.0}}, {});
    expect_environment_error(
        [&] { empty->reset(); },
        EnvironmentErrorCode::empty_event_schedule,
        EnvironmentOperation::prepare_schedule);
    expect(empty->state().phase == EnvironmentPhase::unready &&
               empty->recorder().initial_physical_state().has_value(),
           "failed schedule reset order mismatch");

    auto missing = make_environment(
        root,
        "missing-request",
        {{std::nullopt, 0.0}},
        one_request_events(0));
    expect_environment_error(
        [&] { missing->reset(); },
        EnvironmentErrorCode::missing_request_id,
        EnvironmentOperation::prepare_schedule);

    auto duplicate = make_environment(
        root,
        "duplicate-request",
        {{42, 0.0}, {42, 2.0}},
        one_request_events(42));
    expect_environment_error(
        [&] { duplicate->reset(); },
        EnvironmentErrorCode::duplicate_request_id,
        EnvironmentOperation::prepare_schedule,
        std::nullopt,
        42);

    auto unknown = make_environment(
        root,
        "unknown-request",
        {{42, 0.0}},
        one_request_events(99));
    expect_environment_error(
        [&] { unknown->reset(); },
        EnvironmentErrorCode::unknown_event_request,
        EnvironmentOperation::prepare_schedule,
        0U,
        99);

    const auto overflow_id = static_cast<network::VirtualEventId>(
        static_cast<std::uintmax_t>(
            std::numeric_limits<std::int64_t>::max()) + 1U);
    auto overflow = make_environment(
        root,
        "event-overflow",
        {{42, 0.0}},
        one_request_events(42, overflow_id, 2U));
    expect_environment_error(
        [&] { overflow->reset(); },
        EnvironmentErrorCode::event_id_overflow,
        EnvironmentOperation::prepare_schedule,
        0U,
        42);

    const std::vector<network::VirtualNetworkEventInput> leave_first{
        {0U, network::VirtualEventType::leave, 42, 0.0},
    };
    auto wrong_type = make_environment(
        root, "leave-first", {{42, 0.0}}, leave_first);
    wrong_type->reset();
    Solution solution = wrong_type->make_solution();
    expect_environment_error(
        [&] { static_cast<void>(wrong_type->step(solution)); },
        EnvironmentErrorCode::arrival_event_required,
        EnvironmentOperation::step,
        0U,
        42);
}

void expect_clean_failed_step(
    const SolutionStepEnvironment& environment,
    const PhysicalSnapshot& initial)
{
    expect(environment.state().phase == EnvironmentPhase::active &&
               environment.current_event().schedule_index == 0U &&
               environment.recorder().memory().empty() &&
               physical_snapshot(environment.physical_network()) == initial,
           "failed step committed lifecycle or physical state");
}

void test_mapping_errors(const std::filesystem::path& root)
{
    auto environment = make_environment(
        root, "mapping-errors", {{42, 0.0}}, one_request_events(42));
    environment->reset();
    const auto initial = physical_snapshot(environment->physical_network());

    Solution mismatch = environment->make_solution();
    mismatch.v_net_id = 99;
    expect_environment_error(
        [&] { static_cast<void>(environment->step(mismatch)); },
        EnvironmentErrorCode::solution_request_mismatch,
        EnvironmentOperation::step,
        0U,
        42);
    expect_clean_failed_step(*environment, initial);

    environment->reset();
    Solution incomplete_node = accepted_solution(*environment);
    expect(incomplete_node.node_slots.erase(1),
           "failed to create incomplete node fixture");
    expect_environment_error(
        [&] { static_cast<void>(environment->step(incomplete_node)); },
        EnvironmentErrorCode::incomplete_node_mapping,
        EnvironmentOperation::step,
        0U,
        42);
    expect_clean_failed_step(*environment, initial);

    environment->reset();
    Solution incomplete_link = accepted_solution(*environment);
    expect(incomplete_link.link_paths.erase({0, 1}),
           "failed to create incomplete link fixture");
    expect_environment_error(
        [&] { static_cast<void>(environment->step(incomplete_link)); },
        EnvironmentErrorCode::incomplete_link_mapping,
        EnvironmentOperation::step,
        0U,
        42);
    expect_clean_failed_step(*environment, initial);

    environment->reset();
    Solution accepted = accepted_solution(*environment);
    expect(environment->step(accepted).done,
           "finished-state fixture did not finish");
    expect_environment_error(
        [&] { static_cast<void>(environment->step(accepted)); },
        EnvironmentErrorCode::environment_finished,
        EnvironmentOperation::step);
}

} // namespace

int main()
{
    try
    {
        ScopedDirectory directory("all");
        test_sparse_index_dense_schedule_and_restore(directory.path);
        test_rejection_reasons_and_hard_gate(directory.path);
        test_workers_and_independent_environments(directory.path);
        test_reset_reprepares(directory.path);
        test_failed_auto_drain_and_reset_recovery(directory.path);
        test_schedule_errors(directory.path);
        test_mapping_errors(directory.path);
        std::cout << "environment unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "environment unit: " << error.what() << '\n';
        return 1;
    }
}
