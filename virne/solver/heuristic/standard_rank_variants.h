#pragma once

#include "node_rank.h"

namespace virne::solver::heuristic {

struct GRCRankSolverParameters {
    double sigma = 1.0e-5;
    double damping = 0.85;
};

class GRCRankSolver final : public BaseNodeRankSolver {
public:
    GRCRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        GRCRankSolverParameters parameters = {},
        NodeRankSolverWorkers workers = {});

    const GRCRankSolverParameters& parameters() const noexcept;

private:
    GRCRankSolverParameters parameters_;
};

class NRMRankSolver final : public BaseNodeRankSolver {
public:
    NRMRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

struct RandomWalkRankSolverParameters {
    double sigma = 1.0e-4;
    double jump_probability = 0.15;
    double forwarding_probability = 0.85;
};

class RandomWalkRankSolver final : public BaseNodeRankSolver {
public:
    RandomWalkRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        RandomWalkRankSolverParameters parameters = {},
        NodeRankSolverWorkers workers = {});

    const RandomWalkRankSolverParameters& parameters() const noexcept;

private:
    RandomWalkRankSolverParameters parameters_;
};

SolverId register_grc_rank_solver(
    SolverRegistry& registry,
    GRCRankSolverParameters parameters = {},
    NodeRankSolverWorkers workers = {});

SolverId register_nrm_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

SolverId register_random_walk_rank_solver(
    SolverRegistry& registry,
    RandomWalkRankSolverParameters parameters = {},
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
