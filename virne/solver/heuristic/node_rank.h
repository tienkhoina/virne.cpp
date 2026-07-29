#pragma once

#include "../base_solver.h"
#include "../rank/node_rank.h"
#include "../../core/controller/link_mapper.h"
#include "../../core/controller/node_mapper.h"

#include <cstddef>

namespace virne::solver::heuristic {

// Worker widths are explicit caller policy. Zero and one retain the completed
// dependencies' canonical sequential paths; no value is inferred from the
// host. Commit order remains sequential for every width.
struct NodeRankSolverWorkers {
    std::size_t rank_workers = 1U;
    std::size_t node_candidate_workers = 1U;
    std::size_t link_topology_constraint_workers = 1U;
    std::size_t link_candidate_workers = 1U;
};

// Shared implementation for deterministic node-ranking solvers. The concrete
// solver supplies a fixed rank method; SolverConfig::node_ranking_method and
// link_ranking_method do not dynamically replace that algorithm.
class BaseNodeRankSolver : public Solver {
public:
    core::Solution solve(const SolverInstance& instance) override;

    rank::NodeRankMethod node_rank_method() const noexcept;
    const NodeRankSolverWorkers& workers() const noexcept;

protected:
    BaseNodeRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        rank::NodeRankMethod node_rank_method,
        rank::NodeRankParameters rank_parameters = {},
        NodeRankSolverWorkers workers = {});

private:
    rank::NodeRankMethod node_rank_method_;
    rank::NodeRanker node_ranker_;
    core::controller::NodeMapper node_mapper_;
    core::controller::LinkMapper link_mapper_;
    NodeRankSolverWorkers workers_;
};

class OrderRankSolver final : public BaseNodeRankSolver {
public:
    OrderRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

// Registers one immutable factory configuration. The dynamic name is confined
// to this cold startup boundary; callers retain the returned compact SolverId.
SolverId register_order_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
