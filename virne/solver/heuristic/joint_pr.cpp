#include "joint_pr.h"

#include "python_int_set_order.h"

#include "../../../random/py_random.h"
#include "../../network/attribute/link_attribute.h"
#include "../../network/physical_network.h"
#include "../../network/virtual_network.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace virne::solver::heuristic {
namespace {

using ResourceId = network::attribute::AttributeRegistryId;
using AggregationMatrix = std::vector<std::vector<double>>;

struct CandidateScratch {
    std::vector<core::controller::NodeConstraintRequest> requests;
    std::vector<Vertex> suitable;
};

std::vector<Vertex> ordered_nodes(const Graph& graph) {
    std::vector<Vertex> result;
    result.reserve(graph.num_nodes());
    for (const Vertex node : graph.node_view()) {
        result.push_back(node);
    }
    return result;
}

std::vector<Vertex> selected_physical_nodes(
    const core::Solution& solution,
    std::size_t physical_node_count) {
    std::vector<Vertex> result;
    result.reserve(solution.node_slots.size());
    for (const auto& entry : solution.node_slots.entries()) {
        if (entry.value < 0 ||
            static_cast<std::uint64_t>(entry.value) >=
                physical_node_count) {
            throw JointPRException(
                JointPRErrorCode::invalid_selected_physical_node,
                JointPROperation::collect_candidates,
                "joint PR solution contains an invalid physical node",
                static_cast<Vertex>(entry.key));
        }
        result.push_back(static_cast<Vertex>(entry.value));
    }
    return result;
}

std::vector<ResourceId> resolve_physical_link_resources(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<ResourceId>& virtual_ids) {
    const auto& virtual_registry = virtual_network.link_attributes();
    const auto& physical_registry = physical_network.link_attributes();
    std::vector<ResourceId> result;
    result.reserve(virtual_ids.size());
    for (const ResourceId virtual_id : virtual_ids) {
        const auto& entry = virtual_registry.entries().at(virtual_id);
        const auto physical_id = physical_registry.bind(entry.name);
        if (!physical_id.has_value()) {
            throw JointPRException(
                JointPRErrorCode::missing_physical_link_resource,
                JointPROperation::prepare,
                "joint PR physical network is missing a link resource",
                std::nullopt,
                std::nullopt,
                virtual_id);
        }
        result.push_back(*physical_id);
    }
    return result;
}

void validate_aggregation(
    const AggregationMatrix& values,
    std::size_t expected_rows,
    std::size_t expected_columns,
    std::optional<Vertex> virtual_node = std::nullopt) {
    if (values.size() != expected_rows) {
        throw JointPRException(
            JointPRErrorCode::aggregation_shape_mismatch,
            JointPROperation::collect_candidates,
            "joint PR link aggregation row count does not match resources",
            virtual_node);
    }
    for (const auto& row : values) {
        if (row.size() != expected_columns) {
            throw JointPRException(
                JointPRErrorCode::aggregation_shape_mismatch,
                JointPROperation::collect_candidates,
                "joint PR link aggregation column count does not match nodes",
                virtual_node);
        }
    }
}

std::vector<Vertex> collect_candidates(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    Vertex virtual_node,
    const std::vector<Vertex>& all_physical_nodes,
    const std::vector<Vertex>& selected,
    const core::controller::PreparedConstraintChecker& checker,
    const std::vector<ResourceId>& virtual_link_resources,
    const std::vector<ResourceId>& physical_link_resources,
    const AggregationMatrix& virtual_link_maximum,
    const NodeRankSolverWorkers& workers,
    CandidateScratch& scratch) {
    if (scratch.requests.size() != all_physical_nodes.size()) {
        scratch.requests.clear();
        scratch.requests.reserve(all_physical_nodes.size());
        for (const Vertex physical_node : all_physical_nodes) {
            scratch.requests.push_back(
                core::controller::NodeConstraintRequest{
                    virtual_node, physical_node});
        }
    } else {
        for (auto& request : scratch.requests) {
            request.virtual_node = virtual_node;
        }
    }

    const auto checks = checker.check_node_level_constraints_batch(
        scratch.requests, workers.node_candidate_workers);
    if (checks.size() != scratch.requests.size()) {
        throw JointPRException(
            JointPRErrorCode::constraint_batch_shape_mismatch,
            JointPROperation::collect_candidates,
            "joint PR node constraint batch returned an invalid size",
            virtual_node);
    }

    scratch.suitable.clear();
    scratch.suitable.reserve(all_physical_nodes.size());
    for (std::size_t index = 0U; index < checks.size(); ++index) {
        if (checks[index].feasible) {
            scratch.suitable.push_back(all_physical_nodes[index]);
        }
    }

    auto candidates = detail::cpython310_int_set_difference_order(
        scratch.suitable, selected);

    // Python computes the max-aggregated link-resource and degree filter, but
    // then accidentally ignores `suitable_nodes`. Preserve aggregation and
    // shape/error evaluation. Once both dense matrices are validated, the
    // degree/resource comparison is pure, non-throwing and cannot affect the
    // returned candidates, so do not materialize its dead result vector.
    const auto physical_link_maximum =
        network::get_aggregation_attrs_data(
            physical_network,
            physical_link_resources,
            network::attribute::LinkAggregation::maximum,
            false,
            workers.rank_workers);
    validate_aggregation(
        virtual_link_maximum,
        virtual_link_resources.size(),
        virtual_network.graph().num_nodes(),
        virtual_node);
    validate_aggregation(
        physical_link_maximum,
        physical_link_resources.size(),
        physical_network.graph().num_nodes(),
        virtual_node);

    const auto new_filter = detail::cpython310_int_set_difference_order(
        all_physical_nodes, candidates);
    return detail::cpython310_int_set_difference_order(
        candidates, new_filter);
}

} // namespace

JointPRException::JointPRException(
    JointPRErrorCode code,
    JointPROperation operation,
    std::string message,
    std::optional<Vertex> virtual_node,
    std::optional<Vertex> physical_node,
    std::optional<ResourceId> resource_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      virtual_node_(virtual_node),
      physical_node_(physical_node),
      resource_id_(resource_id) {}

JointPRErrorCode JointPRException::code() const noexcept {
    return code_;
}

JointPROperation JointPRException::operation() const noexcept {
    return operation_;
}

const std::optional<Vertex>& JointPRException::virtual_node() const noexcept {
    return virtual_node_;
}

const std::optional<Vertex>& JointPRException::physical_node() const noexcept {
    return physical_node_;
}

const std::optional<ResourceId>& JointPRException::resource_id() const noexcept {
    return resource_id_;
}

BaseJointPRSolver::BaseJointPRSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    JointPRStrategy strategy,
    NodeRankSolverWorkers workers,
    PyRandom* random)
    : Solver(dependencies, std::move(config)),
      strategy_(strategy),
      workers_(workers),
      random_(random),
      node_ranker_(rank::NodeRankSelection{}, rank::NodeRankParameters{}),
      constraint_checker_(
          dependencies.controller.get().selection().constraints),
      controller_selection_(dependencies.controller.get().selection()) {}

JointPRStrategy BaseJointPRSolver::strategy() const noexcept {
    return strategy_;
}

const NodeRankSolverWorkers& BaseJointPRSolver::workers() const noexcept {
    return workers_;
}

core::Solution BaseJointPRSolver::solve(const SolverInstance& instance) {
    core::Solution solution =
        core::Solution::from_v_net(instance.virtual_network);
    network::PhysicalNetwork physical_network =
        instance.physical_network.clone();
    solve_on(
        instance.virtual_network, physical_network, solution);
    return solution;
}

MutableSolverResult BaseJointPRSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    core::Solution solution =
        core::Solution::from_v_net(instance.virtual_network);
    try {
        solve_on(
            instance.virtual_network,
            instance.physical_network,
            solution);
    } catch (...) {
        instance.mutation.rollback(solution);
        throw;
    }

    const bool committed = solution.result;
    if (!committed) {
        instance.mutation.rollback(solution);
    }
    return MutableSolverResult{
        std::move(solution),
        committed ? SolverMutationState::committed
                  : SolverMutationState::detached};
}

Vertex BaseJointPRSolver::select_physical_node(
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& candidates,
    std::optional<rank::PreparedNodeRanker>& prepared_ranker) {
    if (candidates.empty()) {
        throw JointPRException(
            JointPRErrorCode::invalid_candidate_node,
            JointPROperation::select_candidate,
            "joint PR cannot select from an empty candidate list");
    }

    switch (strategy_) {
    case JointPRStrategy::order:
        return candidates.front();
    case JointPRStrategy::random:
        if (random_ == nullptr) {
            throw JointPRException(
                JointPRErrorCode::random_stream_required,
                JointPROperation::select_candidate,
                "random joint PR requires a caller-owned PyRandom");
        }
        return random_->choice(candidates);
    case JointPRStrategy::ffd: {
        rank::NodeRankOptions options;
        options.sort = true;
        options.workers = workers_.rank_workers;
        options.max_iterations = std::nullopt;
        if (!prepared_ranker.has_value()) {
            // Bind graph-local resource IDs at the original first-use point.
            // The prepared ranker observes every later resource mutation.
            prepared_ranker.emplace(node_ranker_.prepare(physical_network));
        }
        const auto ranking = prepared_ranker->rank_ffd(options);

        const std::size_t physical_node_count =
            physical_network.graph().num_nodes();
        if (ffd_candidate_marks_.size() != physical_node_count) {
            ffd_candidate_marks_.assign(physical_node_count, 0U);
            ffd_candidate_generation_ = 0U;
        }
        ++ffd_candidate_generation_;
        if (ffd_candidate_generation_ == 0U) {
            std::fill(
                ffd_candidate_marks_.begin(),
                ffd_candidate_marks_.end(),
                0U);
            ffd_candidate_generation_ = 1U;
        }
        for (const Vertex candidate : candidates) {
            if (candidate >= ffd_candidate_marks_.size()) {
                throw JointPRException(
                    JointPRErrorCode::invalid_candidate_node,
                    JointPROperation::select_candidate,
                    "joint PR candidate is outside the physical graph",
                    std::nullopt,
                    candidate);
            }
            ffd_candidate_marks_[candidate] = ffd_candidate_generation_;
        }
        for (const auto& entry : ranking) {
            if (entry.node_id < ffd_candidate_marks_.size() &&
                ffd_candidate_marks_[entry.node_id] ==
                    ffd_candidate_generation_) {
                return entry.node_id;
            }
        }
        throw JointPRException(
            JointPRErrorCode::ffd_candidate_not_ranked,
            JointPROperation::rank,
            "FFD ranking did not contain any feasible joint PR candidate");
    }
    }
    throw JointPRException(
        JointPRErrorCode::invalid_candidate_node,
        JointPROperation::select_candidate,
        "joint PR strategy is invalid");
}

void BaseJointPRSolver::solve_on(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    core::Solution& solution) {
    const auto virtual_nodes = ordered_nodes(virtual_network.graph());
    const auto physical_nodes = ordered_nodes(physical_network.graph());

    std::vector<ResourceId> physical_link_resources;
    AggregationMatrix virtual_link_maximum;
    if (!virtual_nodes.empty()) {
        physical_link_resources = resolve_physical_link_resources(
            virtual_network,
            physical_network,
            controller_selection_.link_resources);
        virtual_link_maximum = network::get_aggregation_attrs_data(
            virtual_network,
            controller_selection_.link_resources,
            network::attribute::LinkAggregation::maximum,
            false,
            workers_.rank_workers);
    }

    auto checker = constraint_checker_.prepare(
        virtual_network, physical_network);
    auto prepared_controller = controller().prepare(
        virtual_network, physical_network);

    core::controller::PlaceAndRouteOptions options;
    options.shortest_method = config().shortest_method;
    options.k = 1;
    options.max_path_nodes = 1.0e6;
    options.workers.topology_constraint_workers =
        workers_.link_topology_constraint_workers;
    options.workers.candidate_workers = workers_.link_candidate_workers;
    options.allow_constraint_violation = false;

    // A successful Controller call appends exactly this chosen physical node
    // to node_slots. Keep that insertion-order view incrementally instead of
    // rebuilding it from the Solution for every virtual node.
    std::vector<Vertex> selected = selected_physical_nodes(
        solution, physical_network.graph().num_nodes());
    selected.reserve(selected.size() + virtual_nodes.size());
    CandidateScratch candidate_scratch;
    candidate_scratch.requests.reserve(physical_nodes.size());
    candidate_scratch.suitable.reserve(physical_nodes.size());
    std::optional<rank::PreparedNodeRanker> prepared_physical_ranker;

    for (const Vertex virtual_node : virtual_nodes) {
        const auto candidates = collect_candidates(
            virtual_network,
            physical_network,
            virtual_node,
            physical_nodes,
            selected,
            checker,
            controller_selection_.link_resources,
            physical_link_resources,
            virtual_link_maximum,
            workers_,
            candidate_scratch);
        if (candidates.empty()) {
            solution.place_result = false;
            return;
        }

        const Vertex physical_node = select_physical_node(
            physical_network,
            candidates,
            prepared_physical_ranker);
        const auto result = prepared_controller.place_and_route(
            virtual_node, physical_node, solution, options);
        if (!result.succeeded) {
            // Python labels every joint place-and-route failure as a route
            // failure, even when Controller reports the placement phase.
            solution.route_result = false;
            return;
        }
        selected.push_back(physical_node);
    }

    solution.result = true;
}

RandomJointPRSolver::RandomJointPRSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    PyRandom& random,
    NodeRankSolverWorkers workers)
    : BaseJointPRSolver(
          std::move(dependencies),
          std::move(config),
          JointPRStrategy::random,
          workers,
          &random) {}

OrderJointPRSolver::OrderJointPRSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : BaseJointPRSolver(
          std::move(dependencies),
          std::move(config),
          JointPRStrategy::order,
          workers) {}

FFDJointPRSolver::FFDJointPRSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : BaseJointPRSolver(
          std::move(dependencies),
          std::move(config),
          JointPRStrategy::ffd,
          workers) {}

SolverId register_random_joint_pr_solver(
    SolverRegistry& registry,
    PyRandom& random,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "random_joint_pr",
        SolverCategory::heuristic,
        [random_ptr = &random, workers](
            SolverDependencies dependencies,
            SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<RandomJointPRSolver>(
                std::move(dependencies),
                std::move(config),
                *random_ptr,
                workers);
        });
}

SolverId register_order_joint_pr_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "order_joint_pr",
        SolverCategory::heuristic,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<OrderJointPRSolver>(
                std::move(dependencies),
                std::move(config),
                workers);
        });
}

SolverId register_ffd_joint_pr_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "ffd_joint_pr",
        SolverCategory::heuristic,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<FFDJointPRSolver>(
                std::move(dependencies),
                std::move(config),
                workers);
        });
}

} // namespace virne::solver::heuristic
