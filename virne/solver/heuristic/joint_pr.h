#pragma once

#include "node_rank.h"

#include "../../core/controller/constraint_checker.h"
#include "../../core/controller/controller.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class PyRandom;

namespace virne::solver::heuristic {

enum class JointPRStrategy : std::uint8_t {
    order,
    random,
    ffd,
};

enum class JointPRErrorCode : std::uint8_t {
    random_stream_required,
    invalid_selected_physical_node,
    missing_physical_link_resource,
    aggregation_shape_mismatch,
    constraint_batch_shape_mismatch,
    invalid_candidate_node,
    ffd_candidate_not_ranked,
};

enum class JointPROperation : std::uint8_t {
    prepare,
    collect_candidates,
    select_candidate,
    rank,
};

class JointPRException final : public std::runtime_error {
public:
    JointPRException(
        JointPRErrorCode code,
        JointPROperation operation,
        std::string message,
        std::optional<Vertex> virtual_node = std::nullopt,
        std::optional<Vertex> physical_node = std::nullopt,
        std::optional<network::attribute::AttributeRegistryId>
            resource_id = std::nullopt);

    JointPRErrorCode code() const noexcept;
    JointPROperation operation() const noexcept;
    const std::optional<Vertex>& virtual_node() const noexcept;
    const std::optional<Vertex>& physical_node() const noexcept;
    const std::optional<network::attribute::AttributeRegistryId>&
    resource_id() const noexcept;

private:
    JointPRErrorCode code_;
    JointPROperation operation_;
    std::optional<Vertex> virtual_node_;
    std::optional<Vertex> physical_node_;
    std::optional<network::attribute::AttributeRegistryId> resource_id_;
};

class BaseJointPRSolver : public Solver {
public:
    core::Solution solve(const SolverInstance& instance) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance) override;

    JointPRStrategy strategy() const noexcept;
    const NodeRankSolverWorkers& workers() const noexcept;

protected:
    BaseJointPRSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        JointPRStrategy strategy,
        NodeRankSolverWorkers workers = {},
        PyRandom* random = nullptr);

private:
    void solve_on(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        core::Solution& solution);

    Vertex select_physical_node(
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& candidates,
        std::optional<rank::PreparedNodeRanker>& prepared_ranker);

    JointPRStrategy strategy_;
    NodeRankSolverWorkers workers_;
    PyRandom* random_ = nullptr;
    rank::NodeRanker node_ranker_;
    core::controller::ConstraintChecker constraint_checker_;
    core::controller::ControllerSelection controller_selection_;
    // FFD candidate membership is generation-stamped so each virtual-node
    // selection avoids allocating and zeroing a physical-node-sized mask.
    std::vector<std::uint32_t> ffd_candidate_marks_;
    std::uint32_t ffd_candidate_generation_ = 0U;
};

class RandomJointPRSolver final : public BaseJointPRSolver {
public:
    RandomJointPRSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        PyRandom& random,
        NodeRankSolverWorkers workers = {});
};

class OrderJointPRSolver final : public BaseJointPRSolver {
public:
    OrderJointPRSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

class FFDJointPRSolver final : public BaseJointPRSolver {
public:
    FFDJointPRSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        NodeRankSolverWorkers workers = {});
};

// Registration confines dynamic compatibility names to the cold startup
// boundary. The returned IDs are used for all later factory access.
SolverId register_order_joint_pr_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

SolverId register_random_joint_pr_solver(
    SolverRegistry& registry,
    PyRandom& random,
    NodeRankSolverWorkers workers = {});

SolverId register_ffd_joint_pr_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers = {});

} // namespace virne::solver::heuristic
