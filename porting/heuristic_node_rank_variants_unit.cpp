#include "../virne/solver/heuristic/ffd_rank.h"
#include "../virne/solver/heuristic/custom_rank_variants.h"
#include "../virne/solver/heuristic/random_rank.h"
#include "../virne/solver/heuristic/standard_rank_variants.h"
#include "../virne/solver/heuristic/python_int_set_order.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/core/solution.h"
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
namespace rank = virne::solver::rank;
namespace solver = virne::solver;

constexpr attribute::AttributeRegistryId cpu_id{0U};
constexpr attribute::AttributeRegistryId auxiliary_id{1U};
constexpr attribute::AttributeRegistryId bandwidth_id{0U};

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
           ("virne-heuristic-node-rank-variants-unit-" + std::to_string(tick) + "-" +
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
            filename.rfind("virne-heuristic-node-rank-variants-unit-", 0U) != 0U) {
            return;
        }
        error.clear();
        std::filesystem::remove_all(path, error);
    }
};

core::RecorderConfig recorder_config(const std::filesystem::path& root) {
    core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-node-rank-variants-unit";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

core::LoggerConfig logger_config(const std::filesystem::path& root) {
    core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-node-rank-variants-unit";
    config.run_id = "logger";
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = false;
    config.backends.file = false;
    config.level = core::LoggerLevel::critical;
    config.log_show_interval = 1U;
    return config;
}

controller::ControllerSelection controller_selection(bool with_links = false) {
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {cpu_id};
    selection.node_resources = {cpu_id};
    selection.hard_node_constraints = {cpu_id};
    if (with_links) {
        selection.constraints.link_at_link = {bandwidth_id};
        selection.link_resources = {bandwidth_id};
        selection.hard_link_constraints = {bandwidth_id};
    }
    selection.reusable = false;
    return selection;
}

struct RuntimeFixture {
    CleanupRoot cleanup;
    controller::Controller controller;
    core::Counter counter;
    core::Recorder recorder;
    core::Logger logger;

    explicit RuntimeFixture(bool with_links = false)
        : cleanup{unique_temp_root()},
          controller{controller_selection(with_links)},
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

attribute::AttributeFactorySpec node_resource_spec(
    std::string name,
    attribute::ConstraintRestriction restriction) {
    attribute::AttributeFactorySpec spec;
    spec.name = std::move(name);
    spec.owner = attribute::AttributeOwner::node;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = restriction;
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
    attribute::AttributeRegistryId id,
    const std::vector<std::int64_t>& values) {
    network::NodeAttributeDataUpdate update;
    update.registry_id = id;
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

network::VirtualNetwork make_virtual_network(
    const std::vector<std::int64_t>& cpu,
    const std::vector<std::int64_t>& auxiliary) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(cpu.size(), std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        node_resource_spec(
            "cpu", attribute::ConstraintRestriction::hard),
        node_resource_spec(
            "auxiliary", attribute::ConstraintRestriction::soft),
    };
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({
        node_update(cpu_id, cpu),
        node_update(auxiliary_id, auxiliary),
    });
    result.set_request_id(29);
    result.set_arrival_time(0.0);
    result.set_lifetime(1.0);
    return result;
}

network::PhysicalNetwork make_physical_network(
    const std::vector<std::int64_t>& cpu,
    const std::vector<std::int64_t>& auxiliary) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(cpu.size(), std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        node_resource_spec(
            "cpu", attribute::ConstraintRestriction::hard),
        node_resource_spec(
            "auxiliary", attribute::ConstraintRestriction::soft),
    };
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({
        node_update(cpu_id, cpu),
        node_update(auxiliary_id, auxiliary),
    });
    return result;
}

network::VirtualNetwork make_linked_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        node_resource_spec(
            "cpu", attribute::ConstraintRestriction::hard),
        node_resource_spec(
            "auxiliary", attribute::ConstraintRestriction::soft),
    };
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({
        node_update(cpu_id, {3, 2}),
        node_update(auxiliary_id, {1, 4}),
    });
    result.set_link_attrs_data({link_update({2})});
    result.set_request_id(31);
    result.set_arrival_time(2.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_linked_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        node_resource_spec(
            "cpu", attribute::ConstraintRestriction::hard),
        node_resource_spec(
            "auxiliary", attribute::ConstraintRestriction::soft),
    };
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({
        node_update(cpu_id, {10, 12, 9}),
        node_update(auxiliary_id, {5, 2, 8}),
    });
    result.set_link_attrs_data({link_update({10, 10})});
    return result;
}

network::VirtualNetwork make_sparse_tie_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        1U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        node_resource_spec(
            "cpu", attribute::ConstraintRestriction::hard),
    };
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update(cpu_id, {10})});
    result.set_link_attrs_data({link_update({})});
    result.set_request_id(37);
    result.set_arrival_time(0.0);
    result.set_lifetime(1.0);
    return result;
}

network::PhysicalNetwork make_sparse_tie_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        9U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        node_resource_spec(
            "cpu", attribute::ConstraintRestriction::hard),
    };
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update(
        cpu_id, {0, 10, 0, 0, 0, 0, 0, 0, 10})});
    result.set_link_attrs_data({link_update({})});
    return result;
}

solver::SolverConfig solver_config(const std::filesystem::path& root) {
    solver::SolverConfig config;
    config.verbose = 0;
    config.save_dir = root;
    config.node_ranking_method = rank::NodeRankMethod::random;
    config.matching_method = controller::NodeMatchingMethod::greedy;
    config.shortest_method = controller::ShortestPathMethod::bfs_shortest;
    return config;
}

using NodeRows = std::vector<std::vector<AttrValue>>;

NodeRows resource_snapshot(const network::PhysicalNetwork& network) {
    return network::get_node_attrs_data(
        network, {cpu_id, auxiliary_id}, 1U);
}

struct LinkedResourceSnapshot {
    NodeRows nodes;
    std::vector<std::vector<AttrValue>> links;

    friend bool operator==(
        const LinkedResourceSnapshot& left,
        const LinkedResourceSnapshot& right) {
        return left.nodes == right.nodes && left.links == right.links;
    }
};

LinkedResourceSnapshot linked_resource_snapshot(
    const network::PhysicalNetwork& network) {
    return LinkedResourceSnapshot{
        network::get_node_attrs_data(
            network, {cpu_id, auxiliary_id}, 1U),
        network::get_link_attrs_data(network, {bandwidth_id}, 1U),
    };
}

using ExpectedSlot = std::pair<std::int64_t, std::int64_t>;

void require_slots(
    const core::Solution& solution,
    const std::vector<ExpectedSlot>& expected,
    std::string_view message) {
    const auto& entries = solution.node_slots.entries();
    require(entries.size() == expected.size(), message);
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        require(
            entries[index].key == expected[index].first &&
                entries[index].value == expected[index].second,
            message);
    }
}

std::vector<std::int64_t> solution_snapshot(const core::Solution& solution) {
    std::vector<std::int64_t> result;
    result.reserve(
        4U + solution.node_slots.size() * 2U +
        solution.link_paths.size() * 6U);
    result.push_back(solution.result ? 1 : 0);
    result.push_back(solution.place_result ? 1 : 0);
    result.push_back(solution.route_result ? 1 : 0);
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

core::Solution solve_ffd(
    RuntimeFixture& runtime,
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    heuristic::NodeRankSolverWorkers workers = {}) {
    heuristic::FFDRankSolver solver(
        runtime.dependencies(), solver_config(runtime.cleanup.path), workers);
    return solver.solve(solver::SolverInstance{
        virtual_network,
        physical_network,
    });
}

enum class Variant : std::uint8_t {
    grc,
    nrm,
    random_walk,
    proximity,
    essentiality,
    random,
};

core::Solution solve_variant(
    Variant variant,
    RuntimeFixture& runtime,
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    heuristic::NodeRankSolverWorkers workers,
    NumpyRandomState* random_state = nullptr) {
    std::unique_ptr<solver::Solver> instance;
    switch (variant) {
    case Variant::grc:
        instance = std::make_unique<heuristic::GRCRankSolver>(
            runtime.dependencies(),
            solver_config(runtime.cleanup.path),
            heuristic::GRCRankSolverParameters{},
            workers);
        break;
    case Variant::nrm:
        instance = std::make_unique<heuristic::NRMRankSolver>(
            runtime.dependencies(),
            solver_config(runtime.cleanup.path),
            workers);
        break;
    case Variant::random_walk:
        instance = std::make_unique<heuristic::RandomWalkRankSolver>(
            runtime.dependencies(),
            solver_config(runtime.cleanup.path),
            heuristic::RandomWalkRankSolverParameters{},
            workers);
        break;
    case Variant::proximity:
        instance = std::make_unique<heuristic::PLRankSolver>(
            runtime.dependencies(),
            solver_config(runtime.cleanup.path),
            workers);
        break;
    case Variant::essentiality:
        instance = std::make_unique<heuristic::NEARankSolver>(
            runtime.dependencies(),
            solver_config(runtime.cleanup.path),
            workers);
        break;
    case Variant::random:
        if (random_state == nullptr) {
            throw std::invalid_argument("random state is required");
        }
        instance = std::make_unique<heuristic::RandomRankSolver>(
            runtime.dependencies(),
            solver_config(runtime.cleanup.path),
            *random_state,
            workers);
        break;
    }
    return instance->solve(solver::SolverInstance{
        virtual_network,
        physical_network,
    });
}

void test_api_and_registration() {
    RuntimeFixture runtime;
    const heuristic::NodeRankSolverWorkers workers{0U, 2U, 8U, 1U};
    heuristic::FFDRankSolver direct(
        runtime.dependencies(), solver_config(runtime.cleanup.path), workers);
    require(
        direct.node_rank_method() == rank::NodeRankMethod::ffd,
        "FFD method accessor mismatch");
    require(
        direct.workers().rank_workers == 0U &&
            direct.workers().node_candidate_workers == 2U &&
            direct.workers().link_topology_constraint_workers == 8U &&
            direct.workers().link_candidate_workers == 1U,
        "worker snapshot mismatch");

    solver::SolverRegistry registry;
    const solver::SolverId id =
        heuristic::register_ffd_rank_solver(registry, workers);
    require(id == solver::SolverId{0U}, "compact solver ID mismatch");
    require(registry.resolve("ffd_rank") == id, "cold resolution mismatch");
    registry.freeze();
    const auto& descriptor = registry.descriptor(id);
    require(
        descriptor.name == "ffd_rank" &&
            descriptor.category == solver::SolverCategory::node_ranking,
        "registry descriptor mismatch");
    const std::unique_ptr<solver::Solver> created = registry.create(
        id, runtime.dependencies(), solver_config(runtime.cleanup.path));
    const auto* typed =
        dynamic_cast<const heuristic::FFDRankSolver*>(created.get());
    require(typed != nullptr, "registry factory type mismatch");
    require(
        typed->node_rank_method() == rank::NodeRankMethod::ffd &&
            typed->workers().link_topology_constraint_workers == 8U,
        "registry factory state mismatch");
}

void test_remaining_api_and_registration() {
    RuntimeFixture runtime;
    const heuristic::NodeRankSolverWorkers workers{0U, 2U, 8U, 1U};
    const heuristic::GRCRankSolverParameters grc_parameters{2.5e-5, 0.75};
    const heuristic::RandomWalkRankSolverParameters rw_parameters{
        2.5e-4, 0.2, 0.8};
    NumpyRandomState random_state(123U);

    heuristic::GRCRankSolver grc(
        runtime.dependencies(),
        solver_config(runtime.cleanup.path),
        grc_parameters,
        workers);
    heuristic::NRMRankSolver nrm(
        runtime.dependencies(), solver_config(runtime.cleanup.path), workers);
    heuristic::RandomWalkRankSolver random_walk(
        runtime.dependencies(),
        solver_config(runtime.cleanup.path),
        rw_parameters,
        workers);
    heuristic::PLRankSolver proximity(
        runtime.dependencies(), solver_config(runtime.cleanup.path), workers);
    heuristic::NEARankSolver essentiality(
        runtime.dependencies(), solver_config(runtime.cleanup.path), workers);
    heuristic::RandomRankSolver random(
        runtime.dependencies(),
        solver_config(runtime.cleanup.path),
        random_state,
        workers);

    require(
        grc.node_rank_method() == rank::NodeRankMethod::grc &&
            grc.parameters().sigma == grc_parameters.sigma &&
            grc.parameters().damping == grc_parameters.damping,
        "GRC typed parameters mismatch");
    require(
        nrm.node_rank_method() == rank::NodeRankMethod::nrm,
        "NRM method mismatch");
    require(
        random_walk.node_rank_method() == rank::NodeRankMethod::rw &&
            random_walk.parameters().sigma == rw_parameters.sigma &&
            random_walk.parameters().jump_probability ==
                rw_parameters.jump_probability &&
            random_walk.parameters().forwarding_probability ==
                rw_parameters.forwarding_probability,
        "random-walk typed parameters mismatch");
    require(
        proximity.strategy() == heuristic::CandidateRankStrategy::proximity &&
            proximity.node_rank_method() == rank::NodeRankMethod::nrm,
        "PL strategy mismatch");
    require(
        essentiality.strategy() ==
                heuristic::CandidateRankStrategy::essentiality &&
            essentiality.node_rank_method() == rank::NodeRankMethod::nea,
        "NEA strategy mismatch");
    require(
        &random.random_state() == &random_state &&
            random.workers().rank_workers == 0U,
        "random solver ownership/worker mismatch");

    solver::SolverRegistry registry;
    const std::array<solver::SolverId, 7U> ids{{
        heuristic::register_grc_rank_solver(
            registry, grc_parameters, workers),
        heuristic::register_nrm_rank_solver(registry, workers),
        heuristic::register_pl_rank_solver(registry, workers),
        heuristic::register_nea_rank_solver(registry, workers),
        heuristic::register_random_walk_rank_solver(
            registry, rw_parameters, workers),
        heuristic::register_random_rank_solver(
            registry, random_state, workers),
        heuristic::register_ffd_rank_solver(registry, workers),
    }};
    const std::array<std::string_view, 7U> names{{
        "grc_rank",
        "nrm_rank",
        "pl_rank",
        "nea_rank",
        "rw_rank",
        "random_rank",
        "ffd_rank",
    }};
    registry.freeze();
    for (std::size_t index = 0U; index < ids.size(); ++index) {
        require(
            ids[index] == solver::SolverId{
                static_cast<std::uint32_t>(index)},
            "variant registry ID order mismatch");
        require(
            registry.resolve(names[index]) == ids[index] &&
                registry.descriptor(ids[index]).category ==
                    solver::SolverCategory::node_ranking,
            "variant registry descriptor mismatch");
    }
    const auto created = registry.create(
        ids[5U], runtime.dependencies(), solver_config(runtime.cleanup.path));
    require(
        dynamic_cast<const heuristic::RandomRankSolver*>(created.get()) !=
            nullptr,
        "random registry factory type mismatch");
}

void test_ffd_permutation_and_stable_tie() {
    RuntimeFixture runtime;
    auto virtual_network = make_virtual_network({2, 9, 5}, {0, 0, 0});
    auto physical_network = make_physical_network({6, 3, 12}, {0, 0, 0});
    const NodeRows before = resource_snapshot(physical_network);
    const core::Solution solution = solve_ffd(
        runtime, virtual_network, physical_network);
    require(
        solution.result && solution.place_result && solution.route_result,
        "FFD permutation solve failed");
    require_slots(
        solution, {{1, 2}, {2, 0}, {0, 1}},
        "FFD permutation mismatch");
    require(
        resource_snapshot(physical_network) == before,
        "FFD solve mutated input physical resources");

    RuntimeFixture tie_runtime;
    auto tie_virtual = make_virtual_network({8, 7, 1}, {0, 0, 0});
    auto tie_physical = make_physical_network({10, 3, 10}, {0, 0, 0});
    const core::Solution tie_solution = solve_ffd(
        tie_runtime, tie_virtual, tie_physical);
    require_slots(
        tie_solution, {{0, 0}, {1, 2}, {2, 1}},
        "FFD stable tie order mismatch");
}

void test_partial_failure_retains_ffd_prefix() {
    RuntimeFixture runtime;
    auto virtual_network = make_virtual_network({1, 100}, {100, 0});
    auto physical_network = make_physical_network({10, 10}, {200, 100});
    const NodeRows before = resource_snapshot(physical_network);
    const core::Solution solution = solve_ffd(
        runtime, virtual_network, physical_network);
    require(
        !solution.result && !solution.place_result && solution.route_result,
        "partial placement flags mismatch");
    require_slots(solution, {{0, 0}}, "partial FFD prefix was not retained");
    require(
        resource_snapshot(physical_network) == before,
        "partial solve mutated input physical resources");
}

void test_worker_exact_equality() {
    constexpr std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    std::vector<std::int64_t> expected;
    for (const std::size_t width : widths) {
        RuntimeFixture runtime;
        auto virtual_network = make_virtual_network({2, 9, 5}, {0, 0, 0});
        auto physical_network = make_physical_network({6, 3, 12}, {0, 0, 0});
        const NodeRows before = resource_snapshot(physical_network);
        const heuristic::NodeRankSolverWorkers workers{
            width, width, width, width};
        const auto snapshot = solution_snapshot(solve_ffd(
            runtime, virtual_network, physical_network, workers));
        if (expected.empty()) {
            expected = snapshot;
        } else {
            require(snapshot == expected, "worker width changed exact output");
        }
        require(
            resource_snapshot(physical_network) == before,
            "worker solve mutated input physical resources");
    }
}

void test_deterministic_variants_and_workers() {
    constexpr std::array<Variant, 5U> variants{{
        Variant::grc,
        Variant::nrm,
        Variant::random_walk,
        Variant::proximity,
        Variant::essentiality,
    }};
    constexpr std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    for (const Variant variant : variants) {
        std::vector<std::int64_t> expected;
        for (const std::size_t width : widths) {
            RuntimeFixture runtime(true);
            auto virtual_network = make_linked_virtual_network();
            auto physical_network = make_linked_physical_network();
            const auto before = linked_resource_snapshot(physical_network);
            const heuristic::NodeRankSolverWorkers workers{
                width, width, width, width};
            const core::Solution solution = solve_variant(
                variant,
                runtime,
                virtual_network,
                physical_network,
                workers);
            require(
                solution.result && solution.place_result &&
                    solution.route_result,
                "deterministic variant solve failed");
            const auto snapshot = solution_snapshot(solution);
            if (expected.empty()) {
                expected = snapshot;
            } else {
                require(
                    snapshot == expected,
                    "variant worker width changed exact output");
            }
            require(
                linked_resource_snapshot(physical_network) == before,
                "variant solve mutated input physical resources");
        }
    }
}

void test_sparse_candidate_set_order_and_tie() {
    namespace detail = virne::solver::heuristic::detail;
    const std::vector<Vertex> sparse{1U, 8U};
    const auto first = detail::cpython310_int_set_difference_order(
        sparse, {});
    require(
        first == std::vector<Vertex>({8U, 1U}),
        "CPython 3.10 sparse set order anchor mismatch");
    std::vector<Vertex> all_nodes(9U);
    for (Vertex node = 0U; node < all_nodes.size(); ++node) {
        all_nodes[node] = node;
    }
    const auto new_filter =
        detail::cpython310_int_set_difference_order(all_nodes, first);
    const auto second =
        detail::cpython310_int_set_difference_order(first, new_filter);
    require(
        second == std::vector<Vertex>({8U, 1U}),
        "CPython 3.10 link-filter set round trip mismatch");

    constexpr std::array<Variant, 2U> variants{{
        Variant::proximity,
        Variant::essentiality,
    }};
    constexpr std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    for (const Variant variant : variants) {
        for (const std::size_t width : widths) {
            RuntimeFixture runtime(true);
            auto virtual_network = make_sparse_tie_virtual_network();
            auto physical_network = make_sparse_tie_physical_network();
            const auto before = network::get_node_attrs_data(
                physical_network, {cpu_id}, 1U);
            const heuristic::NodeRankSolverWorkers workers{
                width, width, width, width};
            const auto solution = solve_variant(
                variant,
                runtime,
                virtual_network,
                physical_network,
                workers);
            require(
                solution.result && solution.place_result &&
                    solution.route_result,
                "sparse candidate tie solve failed");
            require_slots(
                solution, {{0, 8}},
                "sparse candidate tie did not retain CPython set order");
            require(
                network::get_node_attrs_data(
                    physical_network, {cpu_id}, 1U) == before,
                "sparse candidate tie mutated input physical resources");
        }
    }
}

void test_random_variant_stream_and_workers() {
    constexpr std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    std::vector<std::int64_t> expected;
    std::uint32_t expected_continuation = 0U;
    for (const std::size_t width : widths) {
        RuntimeFixture runtime(true);
        auto virtual_network = make_linked_virtual_network();
        auto physical_network = make_linked_physical_network();
        const auto before = linked_resource_snapshot(physical_network);
        NumpyRandomState random_state(77U);
        const heuristic::NodeRankSolverWorkers workers{
            width, width, width, width};
        const core::Solution solution = solve_variant(
            Variant::random,
            runtime,
            virtual_network,
            physical_network,
            workers,
            &random_state);
        require(
            solution.result && solution.place_result &&
                solution.route_result,
            "random variant solve failed");
        const auto snapshot = solution_snapshot(solution);
        const std::uint32_t continuation = random_state.next_uint32();
        if (expected.empty()) {
            expected = snapshot;
            expected_continuation = continuation;
        } else {
            require(
                snapshot == expected &&
                    continuation == expected_continuation,
                "random worker width changed output or RNG continuation");
        }
        require(
            linked_resource_snapshot(physical_network) == before,
            "random solve mutated input physical resources");
    }
}

} // namespace

int main() {
    try {
        test_api_and_registration();
        test_remaining_api_and_registration();
        test_ffd_permutation_and_stable_tie();
        test_partial_failure_retains_ffd_prefix();
        test_worker_exact_equality();
        test_deterministic_variants_and_workers();
        test_sparse_candidate_set_order_and_tie();
        test_random_variant_stream_and_workers();
        std::cout << "heuristic node-rank variants unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "heuristic node-rank variants unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
