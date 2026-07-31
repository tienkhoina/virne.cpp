#include "bfs_trials.h"

#include "../../core/controller/controller.h"

#include "../rank/python310_generic_timsort.h"

#include "../../../graph/graph.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace virne::solver::heuristic {
namespace {

using core::controller::PlaceAndRouteOptions;

std::vector<Vertex> ranking_ids(
    const rank::NodeRanking& ranking) {
    std::vector<Vertex> result;
    result.reserve(ranking.size());
    for (const auto& entry : ranking) {
        result.push_back(entry.node_id);
    }
    return result;
}

struct RwLevelEntry {
    Vertex node = 0U;
    std::size_t level = 0U;
    double rank = 0.0;
};

std::vector<Vertex> rw_virtual_order(
    const network::VirtualNetwork& virtual_network,
    const rank::NodeRanking& ranking) {
    if (ranking.empty()) {
        return {};
    }

    const Vertex root = ranking.front().node_id;
    const Graph& graph = virtual_network.graph();
    std::vector<std::optional<std::size_t>> levels(graph.num_nodes());
    std::vector<Vertex> discovery;
    discovery.reserve(graph.num_nodes());
    std::vector<Vertex> queue;
    queue.reserve(graph.num_nodes());
    std::size_t queue_head = 0U;
    levels[root] = 0U;
    queue.push_back(root);
    while (queue_head < queue.size()) {
        const Vertex current = queue[queue_head++];
        discovery.push_back(current);
        const std::size_t next_level = *levels[current] + 1U;
        for (const auto [source, target] : graph.edges(current)) {
            const Vertex neighbor = source == current ? target : source;
            if (!levels[neighbor].has_value()) {
                levels[neighbor] = next_level;
                queue.push_back(neighbor);
            }
        }
    }

    std::vector<double> rank_values(graph.num_nodes(), 0.0);
    for (const auto& entry : ranking) {
        rank_values[entry.node_id] = entry.value;
    }

    std::vector<RwLevelEntry> entries;
    entries.reserve(discovery.size());
    for (const Vertex node : discovery) {
        entries.push_back(RwLevelEntry{
            node,
            *levels[node],
            rank_values[node]});
    }
    rank::detail::python310_timsort(
        entries,
        [](const RwLevelEntry& left, const RwLevelEntry& right) {
            if (left.level != right.level) {
                return left.level < right.level;
            }
            // Python's key is (level, -rank), so use the same IEEE '<'
            // relation instead of an STL comparator that imposes a new NaN
            // ordering.
            return right.rank < left.rank;
        });

    std::vector<Vertex> result;
    result.reserve(entries.size());
    for (const auto& entry : entries) {
        result.push_back(entry.node);
    }
    return result;
}

} // namespace

BfsSolverException::BfsSolverException(
    BfsSolverErrorCode code,
    BfsSolverOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

BfsSolverErrorCode BfsSolverException::code() const noexcept {
    return code_;
}

BfsSolverOperation BfsSolverException::operation() const noexcept {
    return operation_;
}

BfsSolver::BfsSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    rank::NodeRankMethod method,
    BfsSolverParameters parameters,
    NodeRankSolverWorkers workers,
    NumpyRandomState* random_state)
    : Solver(std::move(dependencies), std::move(config)),
      node_ranker_(),
      method_(method),
      parameters_(parameters),
      workers_(workers),
      random_state_(random_state) {}

rank::NodeRanking BfsSolver::rank_network(
    const network::BaseNetwork& network) const {
    rank::NodeRankOptions options;
    options.sort = true;
    options.workers = workers_.rank_workers;
    options.max_iterations = std::nullopt;
    const auto prepared = node_ranker_.prepare(network);
    if (method_ == rank::NodeRankMethod::random) {
        if (random_state_ == nullptr) {
            throw BfsSolverException(
                BfsSolverErrorCode::empty_physical_ranking,
                BfsSolverOperation::rank,
                "random BFS rank requires a caller-owned random state");
        }
        return prepared.rank_random(*random_state_, options);
    }
    return prepared.rank(method_, options);
}

std::vector<Vertex> BfsSolver::ranking_ids(
    const rank::NodeRanking& ranking) const {
    return ::virne::solver::heuristic::ranking_ids(ranking);
}

const BfsSolverParameters& BfsSolver::parameters() const noexcept {
    return parameters_;
}

const NodeRankSolverWorkers& BfsSolver::workers() const noexcept {
    return workers_;
}

void BfsSolver::bfs_deploy(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes,
    Vertex initial_physical_node,
    std::int64_t max_visit,
    std::int64_t max_depth,
    BfsTraversalPolicy policy,
    core::Solution& solution) const {
    if (max_depth == 0) {
        throw BfsSolverException(
            BfsSolverErrorCode::invalid_max_depth,
            BfsSolverOperation::traverse,
            "BFS max_depth must not be zero");
    }

    const double branch_power = std::pow(
        static_cast<double>(max_visit),
        1.0 / static_cast<double>(max_depth));
    if (!std::isfinite(branch_power) || branch_power < 0.0 ||
        branch_power > static_cast<double>(
            std::numeric_limits<std::int64_t>::max())) {
        throw BfsSolverException(
            BfsSolverErrorCode::invalid_max_visit_power,
            BfsSolverOperation::traverse,
            "BFS max_visit/max_depth cannot form a finite branch limit");
    }
    const std::int64_t max_visit_at_every_depth =
        static_cast<std::int64_t>(branch_power);

    auto prepared = controller().prepare(virtual_network, physical_network);
    PlaceAndRouteOptions options;
    options.shortest_method = parameters_.shortest_method;
    options.k = policy == BfsTraversalPolicy::order_defaults
        ? 10
        : parameters_.k_shortest;
    options.max_path_nodes = 1.0e6;
    options.workers.topology_constraint_workers =
        workers_.link_topology_constraint_workers;
    options.workers.candidate_workers = workers_.link_candidate_workers;
    options.allow_constraint_violation = false;

    const Graph& physical_graph = physical_network.graph();
    std::vector<std::uint8_t> visited(
        physical_graph.num_nodes(), std::uint8_t{0});
    if (initial_physical_node >= visited.size()) {
        throw BfsSolverException(
            BfsSolverErrorCode::invalid_initial_physical_node,
            BfsSolverOperation::traverse,
            "BFS initial physical node is outside the graph");
    }
    std::vector<std::pair<Vertex, std::int64_t>> queue;
    queue.reserve(visited.size());
    std::size_t queue_head = 0U;
    queue.emplace_back(initial_physical_node, 0);
    visited[initial_physical_node] = std::uint8_t{1};
    if (virtual_nodes.empty()) {
        throw BfsSolverException(
            BfsSolverErrorCode::empty_virtual_ranking,
            BfsSolverOperation::traverse,
            "BFS virtual ranking is empty");
    }
    std::size_t num_placed_nodes = 0U;
    Vertex virtual_node = virtual_nodes[0U];
    std::size_t num_attempt_times = 0U;
    // Keep one adjacency scratch buffer for the whole traversal. Python's
    // incident-edge prefix is still materialized before the visited check,
    // but revisiting high-degree nodes no longer allocates a fresh vector.
    std::vector<Vertex> incident_neighbors;

    while (queue_head < queue.size()) {
        const auto [physical_node, depth] = queue[queue_head++];
        if (depth > max_depth) {
            break;
        }

        const auto placement = prepared.place_and_route(
            virtual_node,
            physical_node,
            solution,
            options);
        if (placement.succeeded) {
            ++num_placed_nodes;
            if (num_placed_nodes >= virtual_nodes.size()) {
                solution.result = true;
                solution.num_attempt_times = num_attempt_times;
                return;
            }
            virtual_node = virtual_nodes[num_placed_nodes];
        } else {
            ++num_attempt_times;
            if (solution.node_slots.contains(virtual_node)) {
                static_cast<void>(prepared.undo_place_and_route(
                    virtual_node,
                    solution));
            }
        }

        if (depth == max_depth) {
            continue;
        }
        incident_neighbors.clear();
        incident_neighbors.reserve(
            physical_graph.degree(physical_node));
        for (const auto [source, target] :
             physical_graph.edges(physical_node)) {
            const Vertex neighbor = source == physical_node
                ? target
                : source;
            incident_neighbors.push_back(neighbor);
        }
        const bool truncate = max_visit < 0 ||
            incident_neighbors.size() > static_cast<std::size_t>(max_visit);
        const std::size_t prefix_size = truncate
            ? std::min(
                  incident_neighbors.size(),
                  static_cast<std::size_t>(max_visit_at_every_depth))
            : incident_neighbors.size();
        // Python slices the incident edge list before checking visited state.
        // Counting only newly pushed nodes would scan beyond that prefix.
        for (std::size_t index = 0U; index < prefix_size; ++index) {
            const Vertex neighbor = incident_neighbors[index];
            if (visited[neighbor] == std::uint8_t{0}) {
                queue.emplace_back(neighbor, depth + 1);
                visited[neighbor] = std::uint8_t{1};
            }
        }
    }

    solution.num_attempt_times = num_attempt_times;
}

core::Solution BfsSolver::solve_ranked(
    const SolverInstance& instance,
    const std::vector<Vertex>& virtual_nodes,
    const std::vector<Vertex>& physical_nodes,
    std::int64_t max_visit,
    std::int64_t max_depth,
    BfsTraversalPolicy policy) const {
    if (physical_nodes.empty()) {
        throw BfsSolverException(
            BfsSolverErrorCode::empty_physical_ranking,
            BfsSolverOperation::traverse,
            "BFS physical ranking is empty");
    }
    auto working_physical = instance.physical_network.clone();
    core::Solution solution = core::Solution::from_v_net(
        instance.virtual_network);
    bfs_deploy(
        instance.virtual_network,
        working_physical,
        virtual_nodes,
        physical_nodes[0U],
        max_visit,
        max_depth,
        policy,
        solution);
    return solution;
}

MutableSolverResult BfsSolver::solve_ranked_mutable(
    const MutableSolverInstance& instance,
    const std::vector<Vertex>& virtual_nodes,
    const std::vector<Vertex>& physical_nodes,
    std::int64_t max_visit,
    std::int64_t max_depth,
    BfsTraversalPolicy policy) const {
    if (physical_nodes.empty()) {
        throw BfsSolverException(
            BfsSolverErrorCode::empty_physical_ranking,
            BfsSolverOperation::traverse,
            "BFS physical ranking is empty");
    }
    core::Solution solution = core::Solution::from_v_net(
        instance.virtual_network);
    try {
        bfs_deploy(
            instance.virtual_network,
            instance.physical_network,
            virtual_nodes,
            physical_nodes[0U],
            max_visit,
            max_depth,
            policy,
            solution);
    } catch (...) {
        instance.mutation.rollback(solution);
        throw;
    }
    const bool committed = solution.result;
    if (!committed) {
        instance.mutation.rollback(solution);
    }
    return MutableSolverResult{
        std::move(solution),
        committed ? SolverMutationState::committed
                  : SolverMutationState::detached};
}

OrderRankBfsSolver::OrderRankBfsSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers,
    BfsSolverParameters parameters)
    : BfsSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::order,
          parameters,
          workers) {}

core::Solution OrderRankBfsSolver::solve(
    const SolverInstance& instance) {
    const auto virtual_nodes = ranking_ids(
        rank_network(instance.virtual_network));
    const auto physical_nodes = ranking_ids(
        rank_network(instance.physical_network));
    // Python's OrderRankBfsSolver omits max_visit/max_depth/k, selecting the
    // legacy Controller.bfs_deploy defaults 100/10/10.
    return solve_ranked(
        instance,
        virtual_nodes,
        physical_nodes,
        100,
        10,
        BfsTraversalPolicy::order_defaults);
}

MutableSolverResult OrderRankBfsSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    const auto virtual_nodes = ranking_ids(
        rank_network(instance.virtual_network));
    const auto physical_nodes = ranking_ids(
        rank_network(instance.physical_network));
    return solve_ranked_mutable(
        instance,
        virtual_nodes,
        physical_nodes,
        100,
        10,
        BfsTraversalPolicy::order_defaults);
}

RandomRankBfsSolver::RandomRankBfsSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NumpyRandomState& random_state,
    BfsSolverParameters parameters,
    NodeRankSolverWorkers workers)
    : BfsSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::random,
          parameters,
          workers,
          &random_state) {}

core::Solution RandomRankBfsSolver::solve(
    const SolverInstance& instance) {
    const auto virtual_nodes = ranking_ids(
        rank_network(instance.virtual_network));
    const auto physical_nodes = ranking_ids(
        rank_network(instance.physical_network));
    return solve_ranked(
        instance,
        virtual_nodes,
        physical_nodes,
        parameters().max_visit,
        parameters().max_depth,
        BfsTraversalPolicy::configured);
}

MutableSolverResult RandomRankBfsSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    const auto virtual_nodes = ranking_ids(
        rank_network(instance.virtual_network));
    const auto physical_nodes = ranking_ids(
        rank_network(instance.physical_network));
    return solve_ranked_mutable(
        instance,
        virtual_nodes,
        physical_nodes,
        parameters().max_visit,
        parameters().max_depth,
        BfsTraversalPolicy::configured);
}

RandomWalkRankBfsSolver::RandomWalkRankBfsSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    BfsSolverParameters parameters,
    NodeRankSolverWorkers workers)
    : BfsSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::rw,
          parameters,
          workers) {}

core::Solution RandomWalkRankBfsSolver::solve(
    const SolverInstance& instance) {
    const auto virtual_ranking = rank_network(instance.virtual_network);
    const auto physical_nodes = ranking_ids(
        rank_network(instance.physical_network));
    const auto virtual_nodes = rw_virtual_order(
        instance.virtual_network,
        virtual_ranking);
    return solve_ranked(
        instance,
        virtual_nodes,
        physical_nodes,
        parameters().max_visit,
        parameters().max_depth,
        BfsTraversalPolicy::configured);
}

MutableSolverResult RandomWalkRankBfsSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    const auto virtual_ranking = rank_network(instance.virtual_network);
    const auto physical_nodes = ranking_ids(
        rank_network(instance.physical_network));
    const auto virtual_nodes = rw_virtual_order(
        instance.virtual_network,
        virtual_ranking);
    return solve_ranked_mutable(
        instance,
        virtual_nodes,
        physical_nodes,
        parameters().max_visit,
        parameters().max_depth,
        BfsTraversalPolicy::configured);
}

SolverId register_order_rank_bfs_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers,
    BfsSolverParameters parameters) {
    return registry.register_solver(
        "order_rank_bfs",
        SolverCategory::heuristic,
        [workers, parameters](SolverDependencies dependencies,
                              SolverConfig config)
            -> std::unique_ptr<Solver> {
            return std::make_unique<OrderRankBfsSolver>(
                std::move(dependencies),
                std::move(config),
                workers,
                parameters);
        });
}

SolverId register_random_rank_bfs_solver(
    SolverRegistry& registry,
    NumpyRandomState& random_state,
    BfsSolverParameters parameters,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "random_rank_bfs",
        SolverCategory::heuristic,
        [&random_state, parameters, workers](
            SolverDependencies dependencies,
            SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<RandomRankBfsSolver>(
                std::move(dependencies),
                std::move(config),
                random_state,
                parameters,
                workers);
        });
}

SolverId register_random_walk_rank_bfs_solver(
    SolverRegistry& registry,
    BfsSolverParameters parameters,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "rw_rank_bfs",
        SolverCategory::heuristic,
        [parameters, workers](SolverDependencies dependencies,
                              SolverConfig config)
            -> std::unique_ptr<Solver> {
            return std::make_unique<RandomWalkRankBfsSolver>(
                std::move(dependencies), std::move(config), parameters, workers);
        });
}

} // namespace virne::solver::heuristic
