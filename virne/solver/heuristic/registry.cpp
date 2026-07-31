#include "registry.h"

#include "custom_rank_variants.h"
#include "ffd_rank.h"
#include "joint_pr.h"
#include "random_rank.h"

namespace virne::solver::heuristic {

HeuristicSolverIds register_heuristic_solvers(
    SolverRegistry& registry,
    NumpyRandomState& numpy_random,
    PyRandom& python_random,
    HeuristicSolverRegistryOptions options) {
    HeuristicSolverIds ids{};
    ids.order_rank = register_order_rank_solver(registry, options.workers);
    ids.random_rank = register_random_rank_solver(
        registry, numpy_random, options.workers);
    ids.grc_rank = register_grc_rank_solver(
        registry, options.grc, options.workers);
    ids.ffd_rank = register_ffd_rank_solver(registry, options.workers);
    ids.nrm_rank = register_nrm_rank_solver(registry, options.workers);
    ids.pl_rank = register_pl_rank_solver(registry, options.workers);
    ids.nea_rank = register_nea_rank_solver(registry, options.workers);
    ids.rw_rank = register_random_walk_rank_solver(
        registry, options.random_walk, options.workers);
    ids.order_rank_bfs = register_order_rank_bfs_solver(
        registry, options.workers, options.bfs);
    ids.random_rank_bfs = register_random_rank_bfs_solver(
        registry, numpy_random, options.bfs, options.workers);
    ids.rw_rank_bfs = register_random_walk_rank_bfs_solver(
        registry, options.bfs, options.workers);
    ids.random_joint_pr = register_random_joint_pr_solver(
        registry, python_random, options.workers);
    ids.order_joint_pr = register_order_joint_pr_solver(
        registry, options.workers);
    ids.ffd_joint_pr = register_ffd_joint_pr_solver(
        registry, options.workers);
    return ids;
}

} // namespace virne::solver::heuristic
