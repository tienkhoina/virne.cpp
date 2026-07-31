#include "node_rank.h"

#include "../../core/controller/controller.h"
#include "../../core/controller/link_mapper.h"
#include "../../core/controller/node_mapper.h"
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
    core::controller::NodeMapperSelection mapper_selection;
    mapper_selection.node_constraints = selection.constraints.node_at_node;
    mapper_selection.node_resources = selection.node_resources;
    mapper_selection.hard_constraints = selection.hard_node_constraints;
    return mapper_selection;
}

core::controller::LinkMapperSelection make_link_mapper_selection(
    const core::controller::ControllerSelection& selection) {
    core::controller::LinkMapperSelection mapper_selection;
    mapper_selection.link_constraints = selection.constraints.link_at_link;
    mapper_selection.path_constraints = selection.constraints.link_at_path;
    mapper_selection.link_resources = selection.link_resources;
    mapper_selection.hard_constraints = selection.hard_link_constraints;
    mapper_selection.reusable = selection.reusable;
    return mapper_selection;
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

BaseNodeRankSolver::BaseNodeRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    rank::NodeRankMethod node_rank_method,
    rank::NodeRankParameters rank_parameters,
    NodeRankSolverWorkers workers)
    : Solver(dependencies, std::move(config)),
      node_rank_method_(node_rank_method),
      node_ranker_(rank::NodeRankSelection{}, std::move(rank_parameters)),
      node_mapper_(make_node_mapper_selection(
          dependencies.controller.get().selection())),
      link_mapper_(make_link_mapper_selection(
          dependencies.controller.get().selection())),
      workers_(workers) {}

rank::NodeRankMethod BaseNodeRankSolver::node_rank_method() const noexcept {
    return node_rank_method_;
}

const NodeRankSolverWorkers& BaseNodeRankSolver::workers() const noexcept {
    return workers_;
}

core::Solution BaseNodeRankSolver::solve(const SolverInstance& instance) {
    const auto& virtual_network = instance.virtual_network;
    const auto& physical_network = instance.physical_network;

    // Preserve Python's observable failure order: Solution construction, then
    // virtual ranking, then physical ranking. The physical clone is delayed
    // until every read-only ranking step has succeeded.
    core::Solution solution = core::Solution::from_v_net(virtual_network);

    rank::NodeRankOptions rank_options;
    rank_options.sort = true;
    rank_options.workers = workers_.rank_workers;
    rank_options.max_iterations = std::nullopt;

    const auto virtual_ranking =
        node_ranker_.prepare(virtual_network).rank(
            node_rank_method_, rank_options);
    const auto physical_ranking =
        node_ranker_.prepare(physical_network).rank(
            node_rank_method_, rank_options);

    // Python materializes both ordered key lists only after both rank calls.
    const auto virtual_nodes = ordered_node_ids(virtual_ranking);
    const auto physical_nodes = ordered_node_ids(physical_ranking);

    network::PhysicalNetwork working_physical = physical_network.clone();

    bool node_mapping_succeeded = false;
    {
        auto prepared_node_mapper =
            node_mapper_.prepare(virtual_network, working_physical);

        core::controller::NodeMappingOptions options;
        options.reusable = false;
        options.inplace = true;
        options.method = config().matching_method;
        options.allow_constraint_violation = false;
        options.candidate_workers = workers_.node_candidate_workers;

        node_mapping_succeeded = prepared_node_mapper.node_mapping(
            virtual_nodes, physical_nodes, solution, options);
    }

    if (!node_mapping_succeeded) {
        solution.place_result = false;
        solution.result = false;
        return solution;
    }

    // Python reads v_net.links only after successful node placement. Graph's
    // public edge view retains the exact NetworkX endpoint and insertion order.
    const auto virtual_links = ordered_virtual_links(virtual_network);

    bool link_mapping_succeeded = false;
    {
        auto prepared_link_mapper =
            link_mapper_.prepare(virtual_network, working_physical);

        core::controller::LinkMappingOptions options;
        options.shortest_method = config().shortest_method;
        options.k = config().k_shortest;
        options.max_path_nodes = 1.0e6;
        options.topology_constraint_workers =
            workers_.link_topology_constraint_workers;
        options.candidate_workers = workers_.link_candidate_workers;
        options.inplace = true;
        options.allow_constraint_violation = false;

        link_mapping_succeeded = prepared_link_mapper.link_mapping(
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

MutableSolverResult BaseNodeRankSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    const auto& virtual_network = instance.virtual_network;
    auto& physical_network = instance.physical_network;
    core::Solution solution = core::Solution::from_v_net(virtual_network);

    rank::NodeRankOptions rank_options;
    rank_options.sort = true;
    rank_options.workers = workers_.rank_workers;
    rank_options.max_iterations = std::nullopt;

    // Keep the frozen const solve failure/RNG order: both rankings and both
    // ordered ID vectors are complete before the first physical mutation.
    const auto virtual_ranking =
        node_ranker_.prepare(virtual_network).rank(
            node_rank_method_, rank_options);
    const auto physical_ranking =
        node_ranker_.prepare(physical_network).rank(
            node_rank_method_, rank_options);
    const auto virtual_nodes = ordered_node_ids(virtual_ranking);
    const auto physical_nodes = ordered_node_ids(physical_ranking);

    bool rollback_started = false;
    const auto rollback = [&]() {
        rollback_started = true;
        instance.mutation.rollback(solution);
    };
    try {
        auto prepared_node_mapper =
            node_mapper_.prepare(virtual_network, physical_network);
        core::controller::NodeMappingOptions node_options;
        node_options.reusable = false;
        node_options.inplace = true;
        node_options.method = config().matching_method;
        node_options.allow_constraint_violation = false;
        node_options.candidate_workers = workers_.node_candidate_workers;

        if (!prepared_node_mapper.node_mapping(
                virtual_nodes, physical_nodes, solution, node_options)) {
            solution.place_result = false;
            solution.result = false;
            rollback();
            return MutableSolverResult{
                std::move(solution), SolverMutationState::detached};
        }

        const auto virtual_links = ordered_virtual_links(virtual_network);
        auto prepared_link_mapper =
            link_mapper_.prepare(virtual_network, physical_network);
        core::controller::LinkMappingOptions link_options;
        link_options.shortest_method = config().shortest_method;
        link_options.k = config().k_shortest;
        link_options.max_path_nodes = 1.0e6;
        link_options.topology_constraint_workers =
            workers_.link_topology_constraint_workers;
        link_options.candidate_workers = workers_.link_candidate_workers;
        link_options.inplace = true;
        link_options.allow_constraint_violation = false;

        if (!prepared_link_mapper.link_mapping(
                virtual_links, solution, link_options)) {
            solution.route_result = false;
            solution.result = false;
            rollback();
            return MutableSolverResult{
                std::move(solution), SolverMutationState::detached};
        }
    } catch (...) {
        // Mapping commits are journaled directly into fixed Solution tables.
        // Restore only recorded mutations; the caller still observes the
        // original dependency exception after a successful rollback.
        if (!rollback_started) {
            rollback();
        }
        throw;
    }

    solution.result = true;
    return MutableSolverResult{
        std::move(solution), SolverMutationState::committed};
}

OrderRankSolver::OrderRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : BaseNodeRankSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::order,
          rank::NodeRankParameters{},
          workers) {}

SolverId register_order_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "order_rank",
        SolverCategory::node_ranking,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<OrderRankSolver>(
                std::move(dependencies), std::move(config), workers);
        });
}

} // namespace virne::solver::heuristic
