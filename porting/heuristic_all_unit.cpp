#include "../virne/solver/heuristic/registry.h"

#include "../random/numpy_random_state.h"
#include "../random/py_random.h"
#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <array>
#include <atomic>
#include <chrono>
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
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace heuristic = virne::solver::heuristic;
namespace network = virne::network;
namespace solver = virne::solver;

constexpr attribute::AttributeRegistryId cpu_id{0U};
constexpr attribute::AttributeRegistryId bandwidth_id{0U};
constexpr std::size_t solver_count = 14U;

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
           ("virne-heuristic-all-unit-" + std::to_string(tick) + "-" +
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
            filename.rfind("virne-heuristic-all-unit-", 0U) != 0U) {
            return;
        }
        error.clear();
        std::filesystem::remove_all(path, error);
    }
};

core::RecorderConfig recorder_config(const std::filesystem::path& root) {
    core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-all-unit";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

core::LoggerConfig logger_config(const std::filesystem::path& root) {
    core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-all-unit";
    config.run_id = "logger";
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = false;
    config.backends.file = false;
    config.level = core::LoggerLevel::critical;
    config.log_show_interval = 1U;
    return config;
}

controller::ControllerSelection controller_selection() {
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {cpu_id};
    selection.constraints.link_at_link = {bandwidth_id};
    selection.node_resources = {cpu_id};
    selection.link_resources = {bandwidth_id};
    selection.hard_node_constraints = {cpu_id};
    selection.hard_link_constraints = {bandwidth_id};
    selection.reusable = false;
    return selection;
}

struct RuntimeFixture {
    CleanupRoot cleanup;
    controller::Controller controller;
    core::Counter counter;
    core::Recorder recorder;
    core::Logger logger;

    RuntimeFixture()
        : cleanup{unique_temp_root()},
          controller{controller_selection()},
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

attribute::AttributeFactorySpec node_resource_spec() {
    attribute::AttributeFactorySpec spec;
    spec.name = "cpu";
    spec.owner = attribute::AttributeOwner::node;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::node;
    return spec;
}

attribute::AttributeFactorySpec link_resource_spec() {
    attribute::AttributeFactorySpec spec;
    spec.name = "bandwidth";
    spec.owner = attribute::AttributeOwner::link;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::link;
    return spec;
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
    const std::vector<std::int64_t>& values) {
    network::NodeAttributeDataUpdate update;
    update.registry_id = cpu_id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = attribute_values(values);
    return update;
}

network::LinkAttributeDataUpdate link_update(
    const std::vector<std::int64_t>& values) {
    network::LinkAttributeDataUpdate update;
    update.registry_id = bandwidth_id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = attribute_values(values);
    return update;
}

network::VirtualNetwork make_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update({3, 2})});
    result.set_link_attrs_data({link_update({2})});
    result.set_request_id(47);
    result.set_arrival_time(0.0);
    result.set_lifetime(10.0);
    return result;
}

network::VirtualNetwork make_partial_failure_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));
    // Node 0 commits first; node 1 cannot fit anywhere. This leaves a real
    // partial journal for both BFS and joint place-route rollback paths.
    result.set_node_attrs_data({node_update({3, 200})});
    result.set_link_attrs_data({link_update({2})});
    result.set_request_id(53);
    result.set_arrival_time(0.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U},
            {0U, 2U},
            {0U, 3U},
            {1U, 2U},
            {1U, 3U},
            {2U, 3U},
        });
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update({20, 18, 16, 14})});
    result.set_link_attrs_data({link_update({20, 20, 20, 20, 20, 20})});
    return result;
}

solver::SolverConfig solver_config(const std::filesystem::path& root) {
    solver::SolverConfig config;
    config.verbose = 0;
    config.save_dir = root;
    config.matching_method = controller::NodeMatchingMethod::greedy;
    config.shortest_method = controller::ShortestPathMethod::bfs_shortest;
    config.k_shortest = 10;
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
};

ResourceSnapshot resource_snapshot(const network::PhysicalNetwork& network) {
    return ResourceSnapshot{
        network::get_node_attrs_data(network, {cpu_id}, 1U),
        network::get_link_attrs_data(network, {bandwidth_id}, 1U),
    };
}

std::vector<std::int64_t> solution_snapshot(
    const core::Solution& solution) {
    std::vector<std::int64_t> result;
    result.reserve(
        4U + solution.node_slots.size() * 2U +
        solution.link_paths.size() * 8U);
    result.push_back(solution.result ? 1 : 0);
    result.push_back(solution.place_result ? 1 : 0);
    result.push_back(solution.route_result ? 1 : 0);
    result.push_back(static_cast<std::int64_t>(solution.node_slots.size()));
    for (const auto& entry : solution.node_slots.entries()) {
        result.push_back(entry.key);
        result.push_back(entry.value);
    }
    result.push_back(static_cast<std::int64_t>(solution.link_paths.size()));
    for (const auto& entry : solution.link_paths.entries()) {
        result.push_back(entry.key.source);
        result.push_back(entry.key.target);
        result.push_back(static_cast<std::int64_t>(entry.value.size()));
        for (const auto& link : entry.value) {
            result.push_back(link.source);
            result.push_back(link.target);
        }
    }
    return result;
}

struct ExpectedSolver {
    solver::SolverId id;
    std::string_view name;
    solver::SolverCategory category;
};

std::array<ExpectedSolver, solver_count> expected_solvers(
    const heuristic::HeuristicSolverIds& ids) {
    // This array deliberately enumerates every fixed field. It prevents a
    // grouped node-rank implementation from silently disappearing behind the
    // aggregate registry API.
    return {{
        {ids.order_rank, "order_rank", solver::SolverCategory::node_ranking},
        {ids.random_rank, "random_rank", solver::SolverCategory::node_ranking},
        {ids.grc_rank, "grc_rank", solver::SolverCategory::node_ranking},
        {ids.ffd_rank, "ffd_rank", solver::SolverCategory::node_ranking},
        {ids.nrm_rank, "nrm_rank", solver::SolverCategory::node_ranking},
        {ids.pl_rank, "pl_rank", solver::SolverCategory::node_ranking},
        {ids.nea_rank, "nea_rank", solver::SolverCategory::node_ranking},
        {ids.rw_rank, "rw_rank", solver::SolverCategory::node_ranking},
        {ids.order_rank_bfs, "order_rank_bfs", solver::SolverCategory::heuristic},
        {ids.random_rank_bfs, "random_rank_bfs", solver::SolverCategory::heuristic},
        {ids.rw_rank_bfs, "rw_rank_bfs", solver::SolverCategory::heuristic},
        {ids.random_joint_pr, "random_joint_pr", solver::SolverCategory::heuristic},
        {ids.order_joint_pr, "order_joint_pr", solver::SolverCategory::heuristic},
        {ids.ffd_joint_pr, "ffd_joint_pr", solver::SolverCategory::heuristic},
    }};
}

struct CatalogRun {
    std::array<std::vector<std::int64_t>, solver_count> solutions;
    std::uint32_t numpy_continuation = 0U;
    std::uint32_t python_continuation = 0U;
    double milliseconds = 0.0;
};

CatalogRun run_catalog(std::size_t worker_width) {
    RuntimeFixture runtime;
    NumpyRandomState numpy_random(0U);
    PyRandom python_random(0U);
    solver::SolverRegistry registry;

    heuristic::HeuristicSolverRegistryOptions options;
    options.workers = heuristic::NodeRankSolverWorkers{
        worker_width, worker_width, worker_width, worker_width};
    options.bfs.max_visit = 50;
    options.bfs.max_depth = 5;
    options.bfs.shortest_method =
        controller::ShortestPathMethod::bfs_shortest;
    options.bfs.k_shortest = 10;
    const heuristic::HeuristicSolverIds ids =
        heuristic::register_heuristic_solvers(
            registry, numpy_random, python_random, options);
    const auto expected = expected_solvers(ids);

    require(registry.size() == solver_count, "heuristic registry size mismatch");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        require(
            expected[index].id == solver::SolverId{
                static_cast<std::uint32_t>(index)},
            "heuristic compact ID/order mismatch");
        require(
            registry.resolve(expected[index].name) == expected[index].id,
            "heuristic cold name resolution mismatch");
    }
    registry.freeze();

    const auto descriptors = registry.list_registered();
    require(descriptors.size() == expected.size(), "descriptor count mismatch");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        const auto& descriptor = registry.descriptor(expected[index].id);
        require(
            descriptor.id == expected[index].id &&
                descriptor.name == expected[index].name &&
                descriptor.category == expected[index].category,
            "heuristic descriptor mismatch");
        require(
            descriptors[index].id == descriptor.id &&
                descriptors[index].name == descriptor.name &&
                descriptors[index].category == descriptor.category,
            "heuristic descriptor listing order mismatch");
    }

    CatalogRun run;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        auto virtual_network = make_virtual_network();
        auto physical_network = make_physical_network();
        const auto before = resource_snapshot(physical_network);
        std::unique_ptr<solver::Solver> instance = registry.create(
            expected[index].id,
            runtime.dependencies(),
            solver_config(runtime.cleanup.path));
        require(instance != nullptr, "heuristic factory returned null");

        const core::Solution solution = instance->solve(
            solver::SolverInstance{virtual_network, physical_network});
        require(
            solution.result && solution.place_result && solution.route_result,
            "heuristic failed the shared feasible request");
        require(
            solution.node_slots.size() == 2U &&
                solution.link_paths.size() == 1U,
            "heuristic produced an incomplete successful solution");
        require(
            resource_snapshot(physical_network) == before,
            "const heuristic solve mutated the input physical network");
        run.solutions[index] = solution_snapshot(solution);
    }
    run.milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    run.numpy_continuation = numpy_random.next_uint32();
    run.python_continuation = python_random.genrand_uint32();
    return run;
}

void test_complete_heuristic_registry_and_workers() {
    const CatalogRun sequential = run_catalog(1U);
    const CatalogRun parallel = run_catalog(4U);
    for (std::size_t index = 0U; index < solver_count; ++index) {
        require(
            parallel.solutions[index] == sequential.solutions[index],
            "worker width changed exact heuristic output");
    }
    require(
        parallel.numpy_continuation == sequential.numpy_continuation &&
            parallel.python_continuation == sequential.python_continuation,
        "worker width changed a caller-owned random stream");
    std::cout << "heuristic catalog timing: workers=1 "
              << sequential.milliseconds << " ms, workers=4 "
              << parallel.milliseconds << " ms\n";
}

void test_partial_failure_rollback() {
    RuntimeFixture runtime;
    NumpyRandomState numpy_random(0U);
    PyRandom python_random(0U);
    solver::SolverRegistry registry;
    heuristic::HeuristicSolverRegistryOptions options;
    options.workers = heuristic::NodeRankSolverWorkers{4U, 4U, 4U, 4U};
    const auto ids = heuristic::register_heuristic_solvers(
        registry, numpy_random, python_random, options);
    registry.freeze();

    const std::array<solver::SolverId, 2U> rollback_solvers{{
        ids.order_rank_bfs,
        ids.order_joint_pr,
    }};
    for (const solver::SolverId id : rollback_solvers) {
        auto virtual_network = make_partial_failure_virtual_network();
        auto physical_network = make_physical_network();
        const auto before = resource_snapshot(physical_network);
        auto mutation = runtime.controller.prepare_mutation(
            virtual_network, physical_network);
        mutation.begin_transaction();
        auto instance = registry.create(
            id,
            runtime.dependencies(),
            solver_config(runtime.cleanup.path));
        const auto result = instance->solve_mutable(
            solver::MutableSolverInstance{
                virtual_network, physical_network, mutation});
        require(
            result.mutation_state == solver::SolverMutationState::detached &&
                !result.solution.result &&
                result.solution.node_slots.size() == 1U,
            "partial heuristic failure state mismatch");
        require(
            resource_snapshot(physical_network) == before,
            "partial heuristic failure leaked physical resources");
    }
}

} // namespace

int main() {
    try {
        test_complete_heuristic_registry_and_workers();
        test_partial_failure_rollback();
        std::cout << "heuristic all unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "heuristic all unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
