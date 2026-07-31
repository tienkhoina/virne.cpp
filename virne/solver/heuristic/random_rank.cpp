#include "random_rank.h"

#include "../../core/controller/controller.h"
#include "../../network/physical_network.h"

#include <memory>
#include <utility>
#include <vector>

namespace virne::solver::heuristic {
namespace {

using RankedNodeId = decltype(rank::NodeRankEntry{}.node_id);

std::vector<RankedNodeId> ordered_node_ids(
    const rank::NodeRanking& ranking) {
    std::vector<RankedNodeId> nodes;
    nodes.reserve(ranking.size());
    for (const auto& entry : ranking) {
        nodes.push_back(entry.node_id);
    }
    return nodes;
}

core::controller::NodeMapperSelection make_node_mapper_selection(
    const core::controller::ControllerSelection& selection) {
    core::controller::NodeMapperSelection result;
    result.node_constraints = selection.constraints.node_at_node;
    result.node_resources = selection.node_resources;
    result.hard_constraints = selection.hard_node_constraints;
    return result;
}

core::controller::LinkMapperSelection make_link_mapper_selection(
    const core::controller::ControllerSelection& selection) {
    core::controller::LinkMapperSelection result;
    result.link_constraints = selection.constraints.link_at_link;
    result.path_constraints = selection.constraints.link_at_path;
    result.link_resources = selection.link_resources;
    result.hard_constraints = selection.hard_link_constraints;
    result.reusable = selection.reusable;
    return result;
}

std::vector<core::controller::ConstraintLink> ordered_virtual_links(
    const network::VirtualNetwork& virtual_network) {
    const auto& graph = virtual_network.graph();
    std::vector<core::controller::ConstraintLink> links;
    links.reserve(graph.num_edges());
    for (const auto [source, target] : graph.edge_view()) {
        links.push_back(core::controller::ConstraintLink{source, target});
    }
    return links;
}

} // namespace

RandomRankSolver::RandomRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NumpyRandomState& random_state,
    NodeRankSolverWorkers workers)
    : Solver(dependencies, std::move(config)),
      node_ranker_(rank::NodeRankSelection{}, rank::NodeRankParameters{}),
      node_mapper_(make_node_mapper_selection(
          dependencies.controller.get().selection())),
      link_mapper_(make_link_mapper_selection(
          dependencies.controller.get().selection())),
      random_state_(random_state),
      workers_(workers) {}

NumpyRandomState& RandomRankSolver::random_state() const noexcept {
    return random_state_.get();
}

const NodeRankSolverWorkers& RandomRankSolver::workers() const noexcept {
    return workers_;
}

core::Solution RandomRankSolver::solve(const SolverInstance& instance) {
    const auto& virtual_network = instance.virtual_network;
    const auto& physical_network = instance.physical_network;

    core::Solution solution = core::Solution::from_v_net(virtual_network);

    rank::NodeRankOptions rank_options;
    rank_options.sort = true;
    rank_options.workers = workers_.rank_workers;
    rank_options.max_iterations = std::nullopt;

    // One caller-owned stream is advanced in Python's exact virtual-then-
    // physical order. Random draws are intentionally never parallelized.
    const auto virtual_ranking =
        node_ranker_.prepare(virtual_network).rank_random(
            random_state_.get(), rank_options);
    const auto physical_ranking =
        node_ranker_.prepare(physical_network).rank_random(
            random_state_.get(), rank_options);

    const auto virtual_nodes = ordered_node_ids(virtual_ranking);
    const auto physical_nodes = ordered_node_ids(physical_ranking);
    network::PhysicalNetwork working_physical = physical_network.clone();

    bool node_mapping_succeeded = false;
    {
        auto mapper = node_mapper_.prepare(
            virtual_network, working_physical);
        core::controller::NodeMappingOptions options;
        options.reusable = false;
        options.inplace = true;
        options.method = config().matching_method;
        options.allow_constraint_violation = false;
        options.candidate_workers = workers_.node_candidate_workers;
        node_mapping_succeeded = mapper.node_mapping(
            virtual_nodes, physical_nodes, solution, options);
    }

    if (!node_mapping_succeeded) {
        solution.place_result = false;
        solution.result = false;
        return solution;
    }

    const auto virtual_links = ordered_virtual_links(virtual_network);
    bool link_mapping_succeeded = false;
    {
        auto mapper = link_mapper_.prepare(
            virtual_network, working_physical);
        core::controller::LinkMappingOptions options;
        options.shortest_method = config().shortest_method;
        options.k = config().k_shortest;
        options.max_path_nodes = 1.0e6;
        options.topology_constraint_workers =
            workers_.link_topology_constraint_workers;
        options.candidate_workers = workers_.link_candidate_workers;
        options.inplace = true;
        options.allow_constraint_violation = false;
        link_mapping_succeeded = mapper.link_mapping(
            virtual_links, solution, options);
    }

    if (!link_mapping_succeeded) {
        solution.route_result = false;
        solution.result = false;
        return solution;
    }

    solution.result = true;
    return solution;
}

MutableSolverResult RandomRankSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    const auto& virtual_network = instance.virtual_network;
    auto& physical_network = instance.physical_network;
    core::Solution solution = core::Solution::from_v_net(virtual_network);

    rank::NodeRankOptions rank_options;
    rank_options.sort = true;
    rank_options.workers = workers_.rank_workers;
    rank_options.max_iterations = std::nullopt;

    // Preserve the single Python-compatible RNG stream order exactly.
    const auto virtual_ranking =
        node_ranker_.prepare(virtual_network).rank_random(
            random_state_.get(), rank_options);
    const auto physical_ranking =
        node_ranker_.prepare(physical_network).rank_random(
            random_state_.get(), rank_options);
    const auto virtual_nodes = ordered_node_ids(virtual_ranking);
    const auto physical_nodes = ordered_node_ids(physical_ranking);

    bool rollback_started = false;
    const auto rollback = [&]() {
        rollback_started = true;
        instance.mutation.rollback(solution);
    };
    try {
        auto node_mapper = node_mapper_.prepare(
            virtual_network, physical_network);
        core::controller::NodeMappingOptions node_options;
        node_options.reusable = false;
        node_options.inplace = true;
        node_options.method = config().matching_method;
        node_options.allow_constraint_violation = false;
        node_options.candidate_workers = workers_.node_candidate_workers;
        if (!node_mapper.node_mapping(
                virtual_nodes, physical_nodes, solution, node_options)) {
            solution.place_result = false;
            solution.result = false;
            rollback();
            return MutableSolverResult{
                std::move(solution), SolverMutationState::detached};
        }

        const auto virtual_links = ordered_virtual_links(virtual_network);
        auto link_mapper = link_mapper_.prepare(
            virtual_network, physical_network);
        core::controller::LinkMappingOptions link_options;
        link_options.shortest_method = config().shortest_method;
        link_options.k = config().k_shortest;
        link_options.max_path_nodes = 1.0e6;
        link_options.topology_constraint_workers =
            workers_.link_topology_constraint_workers;
        link_options.candidate_workers = workers_.link_candidate_workers;
        link_options.inplace = true;
        link_options.allow_constraint_violation = false;
        if (!link_mapper.link_mapping(
                virtual_links, solution, link_options)) {
            solution.route_result = false;
            solution.result = false;
            rollback();
            return MutableSolverResult{
                std::move(solution), SolverMutationState::detached};
        }
    } catch (...) {
        if (!rollback_started) {
            rollback();
        }
        throw;
    }

    solution.result = true;
    return MutableSolverResult{
        std::move(solution), SolverMutationState::committed};
}

SolverId register_random_rank_solver(
    SolverRegistry& registry,
    NumpyRandomState& random_state,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "random_rank",
        SolverCategory::node_ranking,
        [&random_state, workers](
            SolverDependencies dependencies,
            SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<RandomRankSolver>(
                std::move(dependencies),
                std::move(config),
                random_state,
                workers);
        });
}

} // namespace virne::solver::heuristic
