#include "../virne/system/changeable_system.h"
#include "../virne/system/offline_system.h"
#include "../virne/system/time_window_system.h"

#include "../random/random_context.h"
#include "../virne/core/controller/controller.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/attribute/attribute_factory.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace network = virne::network;
namespace solver = virne::solver;
namespace system = virne::system;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

class CapturedProgress final : public system::SystemProgressSink {
public:
    void begin_epoch(
        std::size_t epoch_index,
        std::size_t total) override {
        beginnings.emplace_back(epoch_index, total);
    }

    void update(const system::SystemProgressUpdate& value) override {
        updates.push_back(value);
    }

    void end_epoch(const system::SystemProgressUpdate& value) override {
        endings.push_back(value);
    }

    std::vector<std::pair<std::size_t, std::size_t>> beginnings;
    std::vector<system::SystemProgressUpdate> updates;
    std::vector<system::SystemProgressUpdate> endings;
};

template <typename Callable>
void require_system_error(
    Callable&& callable,
    system::SystemErrorCode code,
    system::SystemOperation operation) {
    try {
        std::forward<Callable>(callable)();
    } catch (const system::SystemException& error) {
        require(error.code() == code, "system error code mismatch");
        require(
            error.operation() == operation,
            "system error operation mismatch");
        require(
            !std::string_view(error.what()).empty(),
            "system error diagnostic is empty");
        return;
    }
    throw std::runtime_error("expected SystemException");
}

struct ScopedRoot {
    ScopedRoot() {
        static std::atomic<std::uint64_t> sequence{0U};
        const auto tick = std::chrono::steady_clock::now()
                              .time_since_epoch()
                              .count();
        path = std::filesystem::temp_directory_path() /
            ("virne-system-unit-" + std::to_string(tick) + "-" +
             std::to_string(
                 sequence.fetch_add(1U, std::memory_order_relaxed)));
    }

    ~ScopedRoot() {
        std::error_code error;
        const auto temporary_root = std::filesystem::temp_directory_path(error);
        const std::string name = path.filename().string();
        if (!error && path.parent_path() == temporary_root &&
            name.rfind("virne-system-unit-", 0U) == 0U) {
            std::filesystem::remove_all(path, error);
        }
    }

    ScopedRoot(const ScopedRoot&) = delete;
    ScopedRoot& operator=(const ScopedRoot&) = delete;

    std::filesystem::path path;
};

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    attribute::AttributeOwner owner) {
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

network::BaseNetworkConstruction network_construction(Graph graph) {
    network::BaseNetworkConstruction result;
    result.incoming_graph = std::move(graph);
    result.config.node_attribute_specs.push_back(
        resource_spec("cpu", attribute::AttributeOwner::node));
    result.config.link_attribute_specs.push_back(
        resource_spec("bandwidth", attribute::AttributeOwner::link));
    return result;
}

network::PhysicalNetwork make_physical_network() {
    network::PhysicalNetwork result(network_construction(
        Graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    require(
        cpu.has_value() && bandwidth.has_value(),
        "physical fixture binding failed");
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

network::VirtualNetwork make_request(
    network::VirtualRequestId request_id,
    double arrival_time,
    double lifetime = 20.0) {
    network::VirtualNetwork result(network_construction(
        Graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    require(
        cpu.has_value() && bandwidth.has_value(),
        "virtual fixture binding failed");
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
    result.set_request_id(request_id);
    result.set_arrival_time(arrival_time);
    result.set_lifetime(lifetime);
    return result;
}

network::VirtualNetworkSimulationConfig simulation_config(
    std::size_t request_count) {
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = request_count;
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
    return config;
}

network::VirtualNetworkRequestSimulator make_simulator(
    std::vector<network::VirtualNetwork> requests,
    const std::vector<network::VirtualNetworkEventInput>& inputs) {
    std::vector<network::VirtualNetworkEvent> events;
    events.reserve(inputs.size());
    for (const auto& input : inputs) {
        events.emplace_back(input);
    }
    return network::VirtualNetworkRequestSimulator::from_state(
        simulation_config(requests.size()),
        std::move(requests),
        std::move(events));
}

network::VirtualNetworkRequestSimulator make_two_request_simulator() {
    std::vector<network::VirtualNetwork> requests;
    requests.push_back(make_request(7, 0.0));
    requests.push_back(make_request(42, 12.0));
    return make_simulator(
        std::move(requests),
        {
            {0U, network::VirtualEventType::arrival, 7, 0.0},
            {1U, network::VirtualEventType::arrival, 42, 12.0},
            {2U, network::VirtualEventType::leave, 7, 20.0},
            {3U, network::VirtualEventType::leave, 42, 32.0},
        });
}

network::VirtualNetworkRequestSimulator make_one_request_simulator(
    network::VirtualRequestId request_id,
    double arrival_time) {
    std::vector<network::VirtualNetwork> requests;
    requests.push_back(make_request(request_id, arrival_time, 1.0));
    return make_simulator(
        std::move(requests),
        {
            {0U,
             network::VirtualEventType::arrival,
             request_id,
             arrival_time},
            {1U,
             network::VirtualEventType::leave,
             request_id,
             arrival_time + 1.0},
        });
}

core::EnvironmentConfig environment_config(
    const std::filesystem::path& root,
    std::string name) {
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
    result.recorder.solver_name = std::move(name);
    result.recorder.run_id = "run";
    result.recorder.temporary_records = false;
    result.workers = {1U, 1U, 1U};
    return result;
}

std::unique_ptr<core::SolutionStepEnvironment> make_environment(
    const std::filesystem::path& root,
    std::string name,
    network::VirtualNetworkRequestSimulator simulator) {
    return std::make_unique<core::SolutionStepEnvironment>(
        make_physical_network(),
        std::move(simulator),
        environment_config(root, std::move(name)));
}

controller::ControllerSelection solver_controller_selection() {
    controller::ControllerSelection result;
    result.constraints.node_at_node = {0U};
    result.constraints.link_at_link = {0U};
    result.node_resources = {0U};
    result.link_resources = {0U};
    result.hard_node_constraints = {0U};
    result.hard_link_constraints = {0U};
    return result;
}

core::CounterSelection solver_counter_selection() {
    core::CounterSelection result;
    result.node_resources = std::vector<core::CounterResourceId>{0U};
    result.link_resources = std::vector<core::CounterResourceId>{0U};
    return result;
}

core::SolutionAttributeValues one_value(double value) {
    core::SolutionAttributeValues result;
    result.set(0U, value);
    return result;
}

class FixtureSolver final : public solver::Solver {
public:
    FixtureSolver(
        solver::SolverDependencies dependencies,
        solver::SolverConfig config)
        : solver::Solver(dependencies, std::move(config)) {}

    void ready() override {
        ++ready_calls;
    }

    core::Solution solve(const solver::SolverInstance& instance) override {
        core::Solution result =
            core::Solution::from_v_net(instance.virtual_network);
        request_order.push_back(result.v_net_id);
        if (result.v_net_id == 42) {
            result.result = false;
            result.place_result = false;
            return result;
        }

        result.result = true;
        result.node_slots.insert_or_assign(0, 0);
        result.node_slots.insert_or_assign(1, 2);
        result.node_slots_info.insert_or_assign({0, 0}, one_value(10.0));
        result.node_slots_info.insert_or_assign({1, 2}, one_value(20.0));
        const core::SolutionLink virtual_link{0, 1};
        const std::vector<core::SolutionLink> path{{0, 1}, {1, 2}};
        result.link_paths.insert_or_assign(virtual_link, path);
        for (const auto physical_link : path) {
            result.link_paths_info.insert_or_assign(
                {virtual_link, physical_link},
                one_value(30.0));
        }
        return result;
    }

    std::size_t ready_calls = 0U;
    std::vector<network::VirtualRequestId> request_order;
};

class MutableOfflineFixtureSolver final : public solver::Solver {
public:
    MutableOfflineFixtureSolver(
        solver::SolverDependencies dependencies,
        solver::SolverConfig config,
        AttrId physical_cpu_value_id)
        : solver::Solver(dependencies, std::move(config)),
          physical_cpu_value_id_(physical_cpu_value_id) {}

    core::Solution solve(const solver::SolverInstance&) override {
        ++const_solve_calls;
        throw std::runtime_error(
            "offline mutable fixture unexpectedly used const solve");
    }

    solver::MutableSolverResult solve_mutable(
        const solver::MutableSolverInstance& instance) override {
        ++mutable_solve_calls;
        observed_cpu_before.push_back(std::get<double>(
            instance.physical_network.graph().node_attrs(0U).at(
                physical_cpu_value_id_)));

        core::Solution result =
            core::Solution::from_v_net(instance.virtual_network);
        result.result = true;
        result.node_slots.insert_or_assign(0, 0);
        result.node_slots.insert_or_assign(1, 2);
        result.node_slots_info.insert_or_assign({0, 0}, one_value(10.0));
        result.node_slots_info.insert_or_assign({1, 2}, one_value(20.0));
        const core::SolutionLink virtual_link{0, 1};
        const std::vector<core::SolutionLink> path{{0, 1}, {1, 2}};
        result.link_paths.insert_or_assign(virtual_link, path);
        for (const auto physical_link : path) {
            result.link_paths_info.insert_or_assign(
                {virtual_link, physical_link},
                one_value(30.0));
        }

        require(
            instance.mutation.deploy(result),
            "offline mutable fixture deploy failed");
        observed_cpu_after.push_back(std::get<double>(
            instance.physical_network.graph().node_attrs(0U).at(
                physical_cpu_value_id_)));
        return solver::MutableSolverResult{
            std::move(result), solver::SolverMutationState::committed};
    }

    std::size_t const_solve_calls = 0U;
    std::size_t mutable_solve_calls = 0U;
    std::vector<double> observed_cpu_before;
    std::vector<double> observed_cpu_after;

private:
    AttrId physical_cpu_value_id_ = 0U;
};

struct SolverRuntime {
    controller::Controller controller{solver_controller_selection()};
    core::Counter counter{solver_counter_selection()};
    core::Recorder recorder;
    core::Logger logger;
    FixtureSolver solver;

    explicit SolverRuntime(const std::filesystem::path& root)
        : recorder(
              core::Counter(solver_counter_selection()),
              core::RecorderConfig{
                  root,
                  "system-unit",
                  "solver",
                  "records",
                  false}),
          logger([&] {
              core::LoggerConfig config;
              config.save_root_dir = root;
              config.solver_name = "system-unit";
              config.run_id = "solver";
              config.backends.console = false;
              config.backends.file = false;
              config.level = core::LoggerLevel::critical;
              return config;
          }()),
          solver(
              solver::SolverDependencies{
                  std::cref(controller),
                  std::ref(recorder),
                  std::cref(counter),
                  std::ref(logger)},
              solver::SolverConfig{}) {}
};

struct MutableOfflineRuntime {
    controller::Controller controller{solver_controller_selection()};
    core::Counter counter{solver_counter_selection()};
    core::Recorder recorder;
    core::Logger logger;
    MutableOfflineFixtureSolver solver;

    MutableOfflineRuntime(
        const std::filesystem::path& root,
        AttrId physical_cpu_value_id)
        : recorder(
              core::Counter(solver_counter_selection()),
              core::RecorderConfig{
                  root,
                  "system-unit",
                  "offline-mutable",
                  "records",
                  false}),
          logger([&] {
              core::LoggerConfig config;
              config.save_root_dir = root;
              config.solver_name = "system-unit";
              config.run_id = "offline-mutable";
              config.backends.console = false;
              config.backends.file = false;
              config.level = core::LoggerLevel::critical;
              return config;
          }()),
          solver(
              solver::SolverDependencies{
                  std::cref(controller),
                  std::ref(recorder),
                  std::cref(counter),
                  std::ref(logger)},
              solver::SolverConfig{},
              physical_cpu_value_id) {}
};

void test_online(const std::filesystem::path& root, RandomContext& random) {
    SolverRuntime runtime(root);
    auto environment = make_environment(
        root, "online", make_two_request_simulator());
    system::OnlineSystem online(*environment, runtime.solver);
    system::SystemRunConfig config;
    config.num_simulations = 2U;
    config.capture_solutions = true;
    CapturedProgress progress;
    config.progress = &progress;
    const auto result = online.run(random, config);

    require(runtime.solver.ready_calls == 1U, "online ready count mismatch");
    require(result.epochs.size() == 2U, "online epoch count mismatch");
    require(result.steps.size() == 4U, "online trace count mismatch");
    require(
        runtime.solver.request_order ==
            std::vector<network::VirtualRequestId>{7, 42, 7, 42},
        "online request order mismatch");
    for (std::size_t epoch = 0U; epoch < result.epochs.size(); ++epoch) {
        const auto& summary = result.epochs[epoch];
        require(summary.epoch_index == epoch, "online epoch index mismatch");
        require(
            summary.arrival_steps == 2U && summary.accepted == 1U &&
                summary.rejected == 1U &&
                summary.auto_released_events == 2U,
            "online epoch counters mismatch");
        require(summary.summary.success_count == 1, "online summary mismatch");
    }
    require(
        result.steps[0U].request_id == 7 && result.steps[0U].accepted &&
            result.steps[0U].event_time == 0.0 &&
            result.steps[1U].request_id == 42 &&
            !result.steps[1U].accepted &&
            result.steps[1U].failure_reason ==
                core::EnvironmentFailureReason::placement &&
            result.steps[1U].event_time == 12.0,
        "online trace payload mismatch");
    require(
        progress.beginnings ==
                std::vector<std::pair<std::size_t, std::size_t>>{
                    {0U, 2U}, {1U, 2U}} &&
            progress.updates.size() == 4U &&
            progress.endings.size() == 2U &&
            progress.endings[0U].completed == 2U &&
            progress.endings[0U].success_count == 1 &&
            progress.endings[1U].inservice_count == 1,
        "online typed progress stream mismatch");
}

void test_offline(const std::filesystem::path& root) {
    SolverRuntime runtime(root);
    auto physical = make_physical_network();
    auto simulator = make_two_request_simulator();
    system::OfflineSystem offline(physical, simulator, runtime.solver);
    system::OfflineRunConfig config;
    config.num_simulations = 2U;
    config.counter_workers = 2U;
    CapturedProgress progress;
    config.progress = &progress;
    const auto result = offline.run(config);

    require(runtime.solver.ready_calls == 1U, "offline ready count mismatch");
    require(result.epochs.size() == 2U, "offline epoch count mismatch");
    require(result.steps.size() == 4U, "offline trace count mismatch");
    require(
        result.epochs[0U].accepted == 1U &&
            result.epochs[0U].rejected == 1U &&
            result.epochs[0U].auto_released_events == 0U &&
            result.epochs[0U].summary.success_count == 1 &&
            result.epochs[0U].summary.acceptance_rate == 0.5,
        "offline summary mismatch");
    require(
        result.steps[1U].failure_reason ==
            core::EnvironmentFailureReason::placement,
        "offline failure classification mismatch");
    require(
        result.steps[0U].solution.v_net_revenue > 0.0 &&
            result.epochs[0U].summary.total_revenue > 0.0,
        "offline counter metrics were not materialized");
    require(
        &offline.physical_network() == &physical &&
            &offline.simulator() == &simulator &&
            &offline.solver_instance() == &runtime.solver,
        "offline dependency identity mismatch");
    require(
        progress.beginnings ==
                std::vector<std::pair<std::size_t, std::size_t>>{
                    {0U, 2U}, {1U, 2U}} &&
            progress.updates.size() == 4U &&
            progress.endings.size() == 2U &&
            progress.updates[0U].inservice_count == 1 &&
            progress.updates[1U].inservice_count == 1 &&
            progress.endings[1U].success_count == 1,
        "offline Python-compatible progress stream mismatch");
}

void test_offline_mutable_transaction(const std::filesystem::path& root) {
    auto physical = make_physical_network();
    const auto cpu = physical.bind_node_attribute("cpu");
    require(cpu.has_value(), "offline mutable CPU binding failed");
    MutableOfflineRuntime runtime(root, cpu->value_id);
    auto simulator = make_two_request_simulator();
    system::OfflineSystem offline(physical, simulator, runtime.solver);
    system::OfflineRunConfig config;
    config.num_simulations = 2U;
    config.capture_solutions = false;
    const auto result = offline.run(config);

    require(
        runtime.solver.const_solve_calls == 0U &&
            runtime.solver.mutable_solve_calls == 4U,
        "offline did not use the mutable solver seam");
    require(
        runtime.solver.observed_cpu_before ==
            std::vector<double>(4U, 100.0),
        "offline request did not begin from the immutable snapshot");
    require(
        runtime.solver.observed_cpu_after ==
            std::vector<double>(4U, 90.0),
        "offline mutable fixture did not commit its request mutation");
    require(
        std::get<double>(physical.graph().node_attrs(0U).at(cpu->value_id)) ==
            100.0,
        "offline modified the caller physical network");
    require(
        result.steps.empty() && result.epochs.size() == 2U &&
            result.epochs[0U].accepted == 2U &&
            result.epochs[1U].accepted == 2U,
        "offline mutable result summary mismatch");
}

void test_changeable(
    const std::filesystem::path& root,
    RandomContext& random) {
    SolverRuntime runtime(root);
    auto first = make_environment(
        root, "changeable-first", make_one_request_simulator(7, 0.0));
    auto second = make_environment(
        root, "changeable-second", make_one_request_simulator(42, 12.0));

    system::SystemRunConfig first_config;
    first_config.num_simulations = 2U;
    system::SystemRunConfig second_config;
    second_config.num_simulations = 1U;
    CapturedProgress progress;
    first_config.progress = &progress;
    second_config.progress = &progress;
    std::vector<system::ChangeableStage> stages;
    stages.emplace_back(*first, first_config);
    stages.emplace_back(*second, second_config);
    system::ChangeableSystem changeable(std::move(stages), runtime.solver);
    const auto result = changeable.run(random);

    require(changeable.stages().size() == 2U, "stage accessor mismatch");
    require(
        &changeable.solver_instance() == &runtime.solver,
        "changeable solver identity mismatch");
    require(runtime.solver.ready_calls == 2U, "stage ready count mismatch");
    require(result.stages.size() == 2U, "stage summary count mismatch");
    require(
        result.stages[0U].first_epoch_index == 0U &&
            result.stages[0U].num_epochs == 2U &&
            result.stages[0U].first_step_index == 0U &&
            result.stages[0U].num_steps == 2U &&
            result.stages[1U].first_epoch_index == 2U &&
            result.stages[1U].num_epochs == 1U &&
            result.stages[1U].first_step_index == 2U &&
            result.stages[1U].num_steps == 1U,
        "stage boundary summary mismatch");
    require(
        result.run.epochs.size() == 3U && result.run.steps.size() == 3U &&
            result.run.steps[0U].stage_index == 0U &&
            result.run.steps[1U].stage_index == 0U &&
            result.run.steps[2U].stage_index == 1U &&
            result.run.steps[0U].epoch_index == 0U &&
            result.run.steps[1U].epoch_index == 1U &&
            result.run.steps[2U].epoch_index == 2U,
        "changeable flattened indexes mismatch");
    require(
        progress.beginnings ==
                std::vector<std::pair<std::size_t, std::size_t>>{
                    {0U, 1U}, {1U, 1U}, {2U, 1U}} &&
            progress.endings.size() == 3U &&
            progress.endings[0U].epoch_index == 0U &&
            progress.endings[1U].epoch_index == 1U &&
            progress.endings[2U].epoch_index == 2U,
        "changeable progress epoch offset mismatch");

    require_system_error(
        [&] {
            system::ChangeableSystem invalid({}, runtime.solver);
            static_cast<void>(invalid);
        },
        system::SystemErrorCode::empty_changeable_stages,
        system::SystemOperation::construct_changeable);
}

void test_time_window(
    const std::filesystem::path& root,
    RandomContext& random) {
    SolverRuntime runtime(root);
    auto environment = make_environment(
        root, "time-window", make_two_request_simulator());
    system::TimeWindowSystem windows(*environment, runtime.solver);
    system::TimeWindowRunConfig config;
    config.system.num_simulations = 2U;
    config.window_size = 10.0;
    CapturedProgress progress;
    config.system.progress = &progress;
    const auto result = windows.run(random, config);

    require(result.run.epochs.size() == 2U, "window epoch count mismatch");
    require(result.run.steps.size() == 4U, "window trace count mismatch");
    require(result.windows.size() == 4U, "window summary count mismatch");
    require(
        result.run.steps[0U].window_index == 0U &&
            result.run.steps[1U].window_index == 1U &&
            result.run.steps[2U].window_index == 0U &&
            result.run.steps[3U].window_index == 1U,
        "trace window index mismatch");
    for (std::size_t index = 0U; index < result.windows.size(); ++index) {
        const auto& window = result.windows[index];
        require(
            window.epoch_index == index / 2U &&
                window.window_index == index % 2U &&
                window.arrival_steps == 1U,
            "window summary identity mismatch");
    }
    require(
        result.windows[0U].accepted == 1U &&
            result.windows[1U].rejected == 1U &&
            result.windows[1U].auto_released_events == 2U,
        "window acceptance counters mismatch");
    require(
        progress.beginnings ==
                std::vector<std::pair<std::size_t, std::size_t>>{
                    {0U, 2U}, {1U, 2U}} &&
            progress.updates.size() == 4U &&
            progress.endings.size() == 2U &&
            progress.endings[0U].completed == 2U,
        "time-window typed progress stream mismatch");

    system::TimeWindowRunConfig invalid = config;
    invalid.window_size = 0.0;
    require_system_error(
        [&] { static_cast<void>(windows.run(random, invalid)); },
        system::SystemErrorCode::invalid_time_window,
        system::SystemOperation::run_time_window);
}

} // namespace

int main() {
    try {
        ScopedRoot root;
        RandomContext random(23U);
        test_online(root.path, random);
        test_offline(root.path);
        test_offline_mutable_transaction(root.path);
        test_changeable(root.path, random);
        test_time_window(root.path, random);
        std::cout << "system unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "system unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
