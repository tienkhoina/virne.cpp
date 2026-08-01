#pragma once

#include "../base_solver.h"
#include "exact_solver.h"

#include <cstdint>

class PyRandom;

namespace virne::solver::exact {

// Fixed objective controls. Resource names remain dynamic only at the model
// boundary; all risk evaluation uses the dense numeric buffers copied from
// the exact MIP implementation.
struct ExactRiskParameters {
    double scarcity_weight = 1.0;
    double balance_weight = 1.0;
    double criticality_weight = 1.0;
};

class ExactWithRiskSolver final : public Solver {
public:
    ExactWithRiskSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        PyRandom& random,
        ExactSolverParameters parameters = {},
        ExactRiskParameters risk_parameters = {});

    const ExactSolverParameters& parameters() const noexcept;
    const ExactRiskParameters& risk_parameters() const noexcept;

    core::Solution solve(const SolverInstance& instance) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance) override;

private:
    ExactSolverParameters parameters_;
    ExactRiskParameters risk_parameters_;
    PyRandom* random_ = nullptr;
};

SolverId register_exact_with_risk_solver(
    SolverRegistry& registry,
    PyRandom& random,
    ExactSolverParameters parameters = {},
    ExactRiskParameters risk_parameters = {});

}  // namespace virne::solver::exact
