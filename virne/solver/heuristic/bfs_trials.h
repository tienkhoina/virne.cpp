#pragma once

#include "node_rank.h"

#include <cstdint>
#include <functional>
#include <stdexcept>

namespace virne::solver::heuristic {

struct BfsSolverParameters {
    // These signed fields retain the Python cold-boundary validation order;
    // normal configurations use the documented positive defaults.
    std::int64_t max_visit = 50;
    std::int64_t max_depth = 5;
    core::controller::ShortestPathMethod shortest_method =
        core::controller::ShortestPathMethod::bfs_shortest;
    std::int64_t k_shortest = 10;
};

enum class BfsSolverErrorCode : std::uint8_t {
    empty_virtual_ranking,
    empty_physical_ranking,
    invalid_initial_physical_node,
    invalid_max_depth,
    invalid_max_visit_power,
};

enum class BfsSolverOperation : std::uint8_t {
    rank,
    prepare,
    traverse,
};

enum class BfsTraversalPolicy : std::uint8_t {
    order_defaults,
    configured,
};

class BfsSolverException final : public std::runtime_error {
public:
    BfsSolverException(
        BfsSolverErrorCode code,
        BfsSolverOperation operation,
        std::string message);

    BfsSolverErrorCode code() const noexcept;
    BfsSolverOperation operation() const noexcept;

private:
    BfsSolverErrorCode code_;
    BfsSolverOperation operation_;
};

class BfsSolver : public Solver {
protected:
    BfsSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        rank::NodeRankMethod method,
        BfsSolverParameters parameters = {},
        NodeRankSolverWorkers workers = {},
        NumpyRandomState* random_state = nullptr);

    rank::NodeRanking rank_network(
        const network::BaseNetwork& network) const;

    std::vector<Vertex> ranking_ids(
        const rank::NodeRanking& ranking) const;

    core::Solution solve_ranked(
        const SolverInstance& instance,
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<Vertex>& physical_nodes,
        std::int64_t max_visit,
        std::int64_t max_depth,
        BfsTraversalPolicy policy) const;

    MutableSolverResult solve_ranked_mutable(
        const MutableSolverInstance& instance,
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<Vertex>& physical_nodes,
        std::int64_t max_visit,
        std::int64_t max_depth,
        BfsTraversalPolicy policy) const;

    const BfsSolverParameters& parameters() const noexcept;
    const NodeRankSolverWorkers& workers() const noexcept;

private:
    void bfs_deploy(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes,
        Vertex initial_physical_node,
        std::int64_t max_visit,
        std::int64_t max_depth,
        BfsTraversalPolicy policy,
        core::Solution& solution) const;

    rank::NodeRanker node_ranker_;
    rank::NodeRankMethod method_;
    BfsSolverParameters parameters_;
    NodeRankSolverWorkers workers_;
    NumpyRandomState* random_state_ = nullptr;
};

class OrderRankBfsSolver final : public BfsSolver {
public:
    OrderRankBfsSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {},
        BfsSolverParameters parameters = {});

    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
};

class RandomRankBfsSolver final : public BfsSolver {
public:
    RandomRankBfsSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NumpyRandomState& random_state,
        BfsSolverParameters parameters = {},
        NodeRankSolverWorkers workers = {});

    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
};

class RandomWalkRankBfsSolver final : public BfsSolver {
public:
    RandomWalkRankBfsSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        BfsSolverParameters parameters = {},
        NodeRankSolverWorkers workers = {});

    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
};

SolverId register_order_rank_bfs_solver(
    SolverRegistry&,
    NodeRankSolverWorkers workers = {},
    BfsSolverParameters parameters = {});

SolverId register_random_rank_bfs_solver(
    SolverRegistry&,
    NumpyRandomState&,
    BfsSolverParameters parameters = {},
    NodeRankSolverWorkers workers = {});

SolverId register_random_walk_rank_bfs_solver(
    SolverRegistry&,
    BfsSolverParameters parameters = {},
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
