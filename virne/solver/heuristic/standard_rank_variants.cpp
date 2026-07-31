#include "standard_rank_variants.h"

#include <memory>
#include <utility>

namespace virne::solver::heuristic {
namespace {

rank::NodeRankParameters grc_parameters(
    const GRCRankSolverParameters& parameters) {
    rank::NodeRankParameters result;
    result.grc.sigma = parameters.sigma;
    result.grc.damping = parameters.damping;
    return result;
}

rank::NodeRankParameters random_walk_parameters(
    const RandomWalkRankSolverParameters& parameters) {
    rank::NodeRankParameters result;
    result.rw.sigma = parameters.sigma;
    result.rw.jump_probability = parameters.jump_probability;
    result.rw.forwarding_probability = parameters.forwarding_probability;
    return result;
}

} // namespace

GRCRankSolver::GRCRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    GRCRankSolverParameters parameters,
    NodeRankSolverWorkers workers)
    : BaseNodeRankSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::grc,
          grc_parameters(parameters),
          workers),
      parameters_(parameters) {}

const GRCRankSolverParameters& GRCRankSolver::parameters() const noexcept {
    return parameters_;
}

NRMRankSolver::NRMRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : BaseNodeRankSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::nrm,
          rank::NodeRankParameters{},
          workers) {}

RandomWalkRankSolver::RandomWalkRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    RandomWalkRankSolverParameters parameters,
    NodeRankSolverWorkers workers)
    : BaseNodeRankSolver(
          std::move(dependencies),
          std::move(config),
          rank::NodeRankMethod::rw,
          random_walk_parameters(parameters),
          workers),
      parameters_(parameters) {}

const RandomWalkRankSolverParameters&
RandomWalkRankSolver::parameters() const noexcept {
    return parameters_;
}

SolverId register_grc_rank_solver(
    SolverRegistry& registry,
    GRCRankSolverParameters parameters,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "grc_rank",
        SolverCategory::node_ranking,
        [parameters, workers](
            SolverDependencies dependencies,
            SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<GRCRankSolver>(
                std::move(dependencies),
                std::move(config),
                parameters,
                workers);
        });
}

SolverId register_nrm_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "nrm_rank",
        SolverCategory::node_ranking,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<NRMRankSolver>(
                std::move(dependencies), std::move(config), workers);
        });
}

SolverId register_random_walk_rank_solver(
    SolverRegistry& registry,
    RandomWalkRankSolverParameters parameters,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "rw_rank",
        SolverCategory::node_ranking,
        [parameters, workers](
            SolverDependencies dependencies,
            SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<RandomWalkRankSolver>(
                std::move(dependencies),
                std::move(config),
                parameters,
                workers);
        });
}

} // namespace virne::solver::heuristic
