#pragma once

#include "node_rank.h"

#include "../../core/controller/constraint_checker.h"
#include "../../core/controller/controller.h"

#include <cstdint>
#include <vector>

namespace virne::solver::heuristic {

enum class CandidateRankStrategy : std::uint8_t {
    proximity,
    essentiality,
};

class CandidateRankSolver : public Solver {
public:
    core::Solution solve(const SolverInstance& instance) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance) override;

    CandidateRankStrategy strategy() const noexcept;
    rank::NodeRankMethod node_rank_method() const noexcept;
    const NodeRankSolverWorkers& workers() const noexcept;

protected:
    CandidateRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        CandidateRankStrategy strategy,
        rank::NodeRankMethod node_rank_method,
        NodeRankSolverWorkers workers = {});

private:
    MutableSolverResult solve_on_physical(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        core::Solution solution,
        const std::vector<Vertex>& virtual_nodes,
        core::controller::PreparedControllerMutation* mutation);

    CandidateRankStrategy strategy_;
    rank::NodeRankMethod node_rank_method_;
    rank::NodeRanker node_ranker_;
    core::controller::NodeMapper node_mapper_;
    core::controller::LinkMapper link_mapper_;
    core::controller::ConstraintChecker constraint_checker_;
    core::controller::ControllerSelection selection_;
    NodeRankSolverWorkers workers_;
};

class PLRankSolver final : public CandidateRankSolver {
public:
    PLRankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

class NEARankSolver final : public CandidateRankSolver {
public:
    NEARankSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

SolverId register_pl_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

SolverId register_nea_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
