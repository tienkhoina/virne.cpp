#pragma once

#include "node_rank.h"

namespace virne::solver::heuristic {

class FFDRankSolver final : public BaseNodeRankSolver {
public:
    FFDRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

// The dynamic compatibility name is confined to this cold registration
// boundary. Repeated factory lookup and construction use the returned ID.
SolverId register_ffd_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
