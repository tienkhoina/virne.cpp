#pragma once

#include "node_rank.h"

#include "../../../random/numpy_random_state.h"

#include <functional>

namespace virne::solver::heuristic {

// RandomRankSolver deliberately owns no RNG. The caller controls stream
// lifetime, seeding, continuation, and synchronization explicitly.
class RandomRankSolver final : public Solver {
public:
    RandomRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NumpyRandomState& random_state,
        NodeRankSolverWorkers workers = {});

    core::Solution solve(const SolverInstance& instance) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance) override;

    NumpyRandomState& random_state() const noexcept;
    const NodeRankSolverWorkers& workers() const noexcept;

private:
    rank::NodeRanker node_ranker_;
    core::controller::NodeMapper node_mapper_;
    core::controller::LinkMapper link_mapper_;
    std::reference_wrapper<NumpyRandomState> random_state_;
    NodeRankSolverWorkers workers_;
};

SolverId register_random_rank_solver(
    SolverRegistry& registry,
    NumpyRandomState& random_state,
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
