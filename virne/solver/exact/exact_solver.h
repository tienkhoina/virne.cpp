#pragma once

#include "../base_solver.h"

#include <cstddef>
#include <cstdint>

class PyRandom;

namespace virne::solver::exact {

enum class ExactAlgorithm : std::uint8_t {
    mixed_integer,
    deterministic_rounding,
    randomized_rounding,
};

// Fixed native controls. Dynamic configuration strings are resolved by
// MainConfig once; the request/model loops only read these direct fields.
struct ExactSolverParameters {
    std::uint64_t time_limit_ms = 10'000U;
    std::uint64_t search_node_limit = 5'000'000U;
    std::size_t workers = 1U;
};

class ExactSolver final : public Solver {
public:
    ExactSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        ExactAlgorithm algorithm,
        PyRandom& random,
        ExactSolverParameters parameters = {});

    ExactAlgorithm algorithm() const noexcept;
    const ExactSolverParameters& parameters() const noexcept;

    core::Solution solve(const SolverInstance& instance) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance) override;

private:
    ExactAlgorithm algorithm_ = ExactAlgorithm::mixed_integer;
    ExactSolverParameters parameters_;
    PyRandom* random_ = nullptr;
};

struct ExactSolverIds {
    SolverId mip;
    SolverId d_round;
    SolverId r_round;
};

ExactSolverIds register_exact_solvers(
    SolverRegistry& registry,
    PyRandom& random,
    ExactSolverParameters parameters = {});

}  // namespace virne::solver::exact
