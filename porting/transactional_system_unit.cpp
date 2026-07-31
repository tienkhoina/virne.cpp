#include "../virne/system/online_system.h"

#include "../random/numpy_random_state.h"
#include "../random/random_context.h"
#include "../virne/core/controller/controller.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/solver/heuristic/custom_rank_variants.h"
#include "../virne/solver/heuristic/node_rank.h"
#include "../virne/solver/heuristic/random_rank.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
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
namespace heuristic = virne::solver::heuristic;
namespace network = virne::network;
namespace solver = virne::solver;
namespace system = virne::system;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

struct ScopedRoot {
    ScopedRoot() {
        static std::atomic<std::uint64_t> sequence{0U};
        const auto tick = std::chrono::steady_clock::now()
                              .time_since_epoch()
                              .count();
        path = std::filesystem::temp_directory_path() /
            ("virne-transaction-unit-" + std::to_string(tick) + "-" +
             std::to_string(
                 sequence.fetch_add(1U, std::memory_order_relaxed)));
    }

    ~ScopedRoot() {
        std::error_code error;
        const auto temporary_root = std::filesystem::temp_directory_path(error);
        if (!error && path.parent_path() == temporary_root &&
            path.filename().string().rfind("virne-transaction-unit-", 0U) ==
                0U) {
            std::filesystem::remove_all(path, error);
        }
    }

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

network::BaseNetworkConstruction construction(Graph graph) {
    network::BaseNetworkConstruction result;
    result.incoming_graph = std::move(graph);
    result.config.node_attribute_specs.push_back(
        resource_spec("cpu", attribute::AttributeOwner::node));
    result.config.link_attribute_specs.push_back(
        resource_spec("bandwidth", attribute::AttributeOwner::link));
    return result;
}

network::PhysicalNetwork make_physical_network() {
    network::PhysicalNetwork result(construction(
        Graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    require(cpu.has_value() && bandwidth.has_value(), "physical bind failed");
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
    bool force_placement_failure = false,
    bool use_rounding_probe = false) {
    network::VirtualNetwork result(construction(
        Graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    require(cpu.has_value() && bandwidth.has_value(), "virtual bind failed");
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        force_placement_failure
            ? std::vector<AttrValue>{
                  use_rounding_probe ? 16.712847250013507 : 10.0,
                  1000.0}
            : std::vector<AttrValue>{
                  use_rounding_probe ? 16.712847250013507 : 10.0,
                  20.0}}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        bandwidth->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {30.0}}});
    result.set_request_id(7);
    result.set_arrival_time(0.0);
    result.set_lifetime(1.0);
    return result;
}

network::VirtualNetwork make_route_failure_request() {
    network::VirtualNetwork result(construction(
        Graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}})));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    require(cpu.has_value() && bandwidth.has_value(), "route bind failed");
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {10.0, 10.0, 10.0}}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        bandwidth->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {16.712847250013507, 1000.0}}});
    result.set_request_id(7);
    result.set_arrival_time(0.0);
    result.set_lifetime(1.0);
    return result;
}

network::VirtualNetworkSimulationConfig simulation_config() {
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = 1U;
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
    config.arrival_rate.value_kind = virne::utils::DatasetValueKind::floating;
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
    bool use_rounding_probe = false) {
    std::vector<network::VirtualNetwork> requests;
    requests.push_back(make_request(false, use_rounding_probe));
    std::vector<network::VirtualNetworkEvent> events;
    events.emplace_back(network::VirtualNetworkEventInput{
        0U, network::VirtualEventType::arrival, 7, 0.0});
    events.emplace_back(network::VirtualNetworkEventInput{
        1U, network::VirtualEventType::leave, 7, 1.0});
    return network::VirtualNetworkRequestSimulator::from_state(
        simulation_config(), std::move(requests), std::move(events));
}

controller::ControllerSelection controller_selection() {
    controller::ControllerSelection result;
    result.constraints.node_at_node = {0U};
    result.constraints.link_at_link = {0U};
    result.node_resources = {0U};
    result.link_resources = {0U};
    result.hard_node_constraints = {0U};
    result.hard_link_constraints = {0U};
    return result;
}

core::CounterSelection counter_selection() {
    core::CounterSelection result;
    result.node_resources = std::vector<core::CounterResourceId>{0U};
    result.link_resources = std::vector<core::CounterResourceId>{0U};
    return result;
}

core::EnvironmentConfig environment_config(
    const std::filesystem::path& root,
    bool reject_after_commit = false,
    std::size_t mutation_workers = 1U) {
    core::EnvironmentConfig result;
    result.controller = controller_selection();
    result.counter = counter_selection();
    result.recorder.save_root_dir = root;
    result.recorder.solver_name = "transaction-unit";
    result.recorder.run_id = "run";
    result.recorder.temporary_records = false;
    result.workers = {1U, 1U, mutation_workers};
    if (reject_after_commit) {
        result.admission.r2c_ratio_threshold = 1.0;
        result.admission.virtual_network_size_threshold = 0U;
    }
    return result;
}

struct ResourceSnapshot {
    std::vector<std::vector<AttrValue>> nodes;
    std::vector<std::vector<AttrValue>> links;

    friend bool operator==(
        const ResourceSnapshot& left,
        const ResourceSnapshot& right) {
        return left.nodes == right.nodes && left.links == right.links;
    }
};

ResourceSnapshot snapshot(const network::PhysicalNetwork& physical) {
    return ResourceSnapshot{
        network::get_node_attrs_data(physical, {0U}, 1U),
        network::get_link_attrs_data(physical, {0U}, 1U)};
}

struct Runtime {
    explicit Runtime(const std::filesystem::path& root)
        : recorder(
              core::Counter(counter_selection()),
              core::RecorderConfig{
                  root,
                  "transaction-unit",
                  "solver",
                  "records",
                  false}),
          logger([&] {
              core::LoggerConfig config;
              config.save_root_dir = root;
              config.solver_name = "transaction-unit";
              config.run_id = "solver";
              config.backends.console = false;
              config.backends.file = false;
              config.level = core::LoggerLevel::critical;
              return config;
          }()) {}

    solver::SolverDependencies dependencies() {
        return solver::SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger)};
    }

    controller::Controller controller{controller_selection()};
    core::Counter counter{counter_selection()};
    core::Recorder recorder;
    core::Logger logger;
};

solver::SolverConfig solver_config() {
    solver::SolverConfig result;
    result.shortest_method = controller::ShortestPathMethod::k_shortest;
    result.k_shortest = 10;
    return result;
}

template <typename SolverType>
void require_mutable_success(
    SolverType& instance,
    Runtime& runtime,
    std::string_view message) {
    auto virtual_network = make_request();
    auto physical_network = make_physical_network();
    const auto before = snapshot(physical_network);
    auto mutation = runtime.controller.prepare_mutation(
        virtual_network, physical_network);
    mutation.begin_transaction();
    solver::Solver& base = instance;
    solver::MutableSolverResult result = base.solve_mutable(
        solver::MutableSolverInstance{
            virtual_network, physical_network, mutation});
    require(
        result.mutation_state == solver::SolverMutationState::committed &&
            result.solution.result,
        message);
    mutation.rollback(result.solution);
    require(snapshot(physical_network) == before, "mutable rollback mismatch");
}

void test_all_node_rank_overrides(const std::filesystem::path& root) {
    Runtime runtime(root);
    heuristic::OrderRankSolver order(
        runtime.dependencies(), solver_config());
    require_mutable_success(order, runtime, "base node-rank used fallback");

    NumpyRandomState random_state(17U);
    heuristic::RandomRankSolver random(
        runtime.dependencies(), solver_config(), random_state);
    require_mutable_success(random, runtime, "random rank used fallback");

    heuristic::PLRankSolver proximity(
        runtime.dependencies(), solver_config());
    require_mutable_success(proximity, runtime, "PL rank used fallback");

    heuristic::NEARankSolver essentiality(
        runtime.dependencies(), solver_config());
    require_mutable_success(essentiality, runtime, "NEA rank used fallback");
}

void test_partial_failure_rollback(const std::filesystem::path& root) {
    Runtime runtime(root);
    heuristic::OrderRankSolver order(
        runtime.dependencies(), solver_config());
    auto virtual_network = make_request(true, true);
    auto physical_network = make_physical_network();
    const auto cpu = physical_network.bind_node_attribute("cpu");
    require(cpu.has_value(), "rounding probe bind failed");
    physical_network.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {57.56510141648885, 200.0, 300.0}}});
    const auto before = snapshot(physical_network);
    auto mutation = runtime.controller.prepare_mutation(
        virtual_network, physical_network);
    mutation.begin_transaction();
    const solver::MutableSolverResult result = order.solve_mutable(
        solver::MutableSolverInstance{
            virtual_network, physical_network, mutation});
    require(
        result.mutation_state == solver::SolverMutationState::detached &&
            !result.solution.result && !result.solution.place_result &&
            result.solution.node_slots.size() == 1U,
        "partial placement output changed");
    require(snapshot(physical_network) == before, "partial failure leaked");
}

void test_route_failure_rollback(const std::filesystem::path& root) {
    Runtime runtime(root);
    heuristic::OrderRankSolver order(
        runtime.dependencies(), solver_config());
    auto virtual_network = make_route_failure_request();
    auto physical_network = make_physical_network();
    const auto bandwidth = physical_network.bind_link_attribute("bandwidth");
    require(bandwidth.has_value(), "route probe bind failed");
    physical_network.set_link_attrs_data({network::LinkAttributeDataUpdate{
        bandwidth->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {57.56510141648885, 500.0}}});
    const auto before = snapshot(physical_network);
    auto mutation = runtime.controller.prepare_mutation(
        virtual_network, physical_network);
    mutation.begin_transaction();
    const solver::MutableSolverResult result = order.solve_mutable(
        solver::MutableSolverInstance{
            virtual_network, physical_network, mutation});
    require(
        result.mutation_state == solver::SolverMutationState::detached &&
            !result.solution.result && !result.solution.route_result,
        "partial route failure state changed");
    require(
        result.solution.link_paths.size() == 2U &&
            !result.solution.link_paths_info.empty(),
        "partial route output changed");
    require(snapshot(physical_network) == before, "route failure leaked");
}

void test_system_commit_and_admission_rollback(
    const std::filesystem::path& root,
    RandomContext& random) {
    for (const bool reject : {false, true}) {
        Runtime runtime(root);
        heuristic::OrderRankSolver order(
            runtime.dependencies(), solver_config());
        auto physical_network = make_physical_network();
        if (reject) {
            const auto cpu = physical_network.bind_node_attribute("cpu");
            require(cpu.has_value(), "admission probe bind failed");
            physical_network.set_node_attrs_data(
                {network::NodeAttributeDataUpdate{
                    cpu->registry_id,
                    network::AttributeDataLayout::dense,
                    {},
                    {57.56510141648885, 200.0, 300.0}}});
        }
        const auto expected = snapshot(physical_network);
        core::SolutionStepEnvironment environment(
            std::move(physical_network),
            make_simulator(reject),
            environment_config(root, reject));
        system::OnlineSystem online(environment, order);
        system::SystemRunConfig config;
        config.capture_solutions = true;
        const auto result = online.run(random, config);
        require(
            result.epochs.size() == 1U &&
                result.epochs[0U].arrival_steps == 1U &&
                result.epochs[0U].accepted == (reject ? 0U : 1U),
            "transactional online admission mismatch");
        require(
            snapshot(environment.physical_network()) == expected,
            reject ? "admission rejection leaked committed resources"
                   : "success was deployed twice or released incorrectly");
    }
}

void test_success_worker_parity(
    const std::filesystem::path& root,
    RandomContext& random) {
    std::optional<double> expected_r2c;
    for (const std::size_t width : {1U, 8U}) {
        Runtime runtime(root);
        const heuristic::NodeRankSolverWorkers workers{
            width, width, width, width};
        heuristic::OrderRankSolver order(
            runtime.dependencies(), solver_config(), workers);
        auto physical_network = make_physical_network();
        const auto expected_resources = snapshot(physical_network);
        core::SolutionStepEnvironment environment(
            std::move(physical_network),
            make_simulator(),
            environment_config(root, false, width));
        system::OnlineSystem online(environment, order);
        system::SystemRunConfig config;
        config.capture_solutions = false;
        const auto result = online.run(random, config);
        require(
            result.epochs.size() == 1U &&
                result.epochs[0U].accepted == 1U &&
                result.epochs[0U].summary.success_count == 1,
            "success worker result mismatch");
        if (!expected_r2c.has_value()) {
            expected_r2c =
                result.epochs[0U].summary.long_term_r2c_ratio;
        } else {
            require(
                result.epochs[0U].summary.long_term_r2c_ratio ==
                    *expected_r2c,
                "worker width changed exact summary");
        }
        require(
            snapshot(environment.physical_network()) == expected_resources,
            "worker width changed final resources");
    }
}

void test_validation_exception_rollback(const std::filesystem::path& root) {
    core::SolutionStepEnvironment environment(
        make_physical_network(),
        make_simulator(),
        environment_config(root));
    environment.reset();
    const auto before = snapshot(environment.physical_network());
    auto transaction = environment.solver_transaction();
    core::Solution solution = environment.make_solution();
    solution.result = true;
    solution.node_slots.insert_or_assign(0, 0);
    core::SolutionAttributeValues value;
    value.set(0U, 10.0);
    solution.node_slots_info.insert_or_assign({0, 0}, value);
    require(transaction.mutation.deploy(solution), "test commit failed");

    try {
        static_cast<void>(environment.step(
            solution, core::EnvironmentSolutionState::committed));
    } catch (const core::EnvironmentException& error) {
        require(
            error.code() == core::EnvironmentErrorCode::incomplete_node_mapping,
            "validation error code mismatch");
        require(
            snapshot(environment.physical_network()) == before,
            "validation exception leaked committed resources");
        return;
    }
    throw std::runtime_error("expected validation exception");
}

} // namespace

int main() {
    try {
        ScopedRoot root;
        RandomContext random(23U);
        test_all_node_rank_overrides(root.path);
        test_partial_failure_rollback(root.path);
        test_route_failure_rollback(root.path);
        test_system_commit_and_admission_rollback(root.path, random);
        test_success_worker_parity(root.path, random);
        test_validation_exception_rollback(root.path);
        std::cout << "transactional system unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "transactional system unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
