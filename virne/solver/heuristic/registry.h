#pragma once

#include "bfs_trials.h"
#include "standard_rank_variants.h"

class PyRandom;

namespace virne::solver::heuristic {

struct HeuristicSolverRegistryOptions {
    NodeRankSolverWorkers workers;
    GRCRankSolverParameters grc;
    RandomWalkRankSolverParameters random_walk;
    BfsSolverParameters bfs;
};

// Every canonical solver ID is a fixed direct field. Dynamic compatibility
// names remain confined to the individual cold registration functions.
struct HeuristicSolverIds {
    SolverId order_rank;
    SolverId random_rank;
    SolverId grc_rank;
    SolverId ffd_rank;
    SolverId nrm_rank;
    SolverId pl_rank;
    SolverId nea_rank;
    SolverId rw_rank;
    SolverId order_rank_bfs;
    SolverId random_rank_bfs;
    SolverId rw_rank_bfs;
    SolverId random_joint_pr;
    SolverId order_joint_pr;
    SolverId ffd_joint_pr;
};

HeuristicSolverIds register_heuristic_solvers(
    SolverRegistry& registry,
    NumpyRandomState& numpy_random,
    PyRandom& python_random,
    HeuristicSolverRegistryOptions options = {});

} // namespace virne::solver::heuristic
