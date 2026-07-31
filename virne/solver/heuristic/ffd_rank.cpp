#include "ffd_rank.h"

#include <memory>
#include <utility>

namespace virne::solver::heuristic {

FFDRankSolver::FFDRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : BaseNodeRankSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::ffd,
          rank::NodeRankParameters{},
          workers) {}

SolverId register_ffd_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "ffd_rank",
        SolverCategory::node_ranking,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<FFDRankSolver>(
                std::move(dependencies), std::move(config), workers);
        });
}

} // namespace virne::solver::heuristic
