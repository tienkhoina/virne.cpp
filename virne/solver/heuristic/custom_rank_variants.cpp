#include "custom_rank_variants.h"
#include "python_int_set_order.h"
#include "../rank/python310_generic_timsort.h"

#include "../../core/controller/controller.h"
#include "../../network/physical_network.h"
#include "../../utils/deterministic_executor.h"
#include "../../../graph/nx/shortest_paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace virne::solver::heuristic {
namespace {

using RankedNodeId = decltype(rank::NodeRankEntry{}.node_id);
using ResourceId = network::attribute::AttributeRegistryId;
using DijkstraDistances = std::vector<std::optional<double>>;

struct DijkstraQueueEntry {
    double distance = 0.0;
    std::uint64_t order = 0U;
    Vertex vertex = 0U;
};

// CPython heapq compares the `(distance, insertion_order, vertex)` tuples
// lexicographically. Keep its exact sift schedule because IEEE NaN is not a
// strict weak ordering and a standard priority_queue may choose another item.
bool python_queue_entry_less(
    const DijkstraQueueEntry& left,
    const DijkstraQueueEntry& right) noexcept {
    if (left.distance == right.distance) {
        if (left.order == right.order) {
            return left.vertex < right.vertex;
        }
        return left.order < right.order;
    }
    return left.distance < right.distance;
}

void python_heap_siftdown(
    std::vector<DijkstraQueueEntry>& heap,
    std::size_t start,
    std::size_t position) {
    const DijkstraQueueEntry item = heap[position];
    while (position > start) {
        const std::size_t parent = (position - 1U) >> 1U;
        if (!python_queue_entry_less(item, heap[parent])) {
            break;
        }
        heap[position] = heap[parent];
        position = parent;
    }
    heap[position] = item;
}

void python_heap_push(
    std::vector<DijkstraQueueEntry>& heap,
    DijkstraQueueEntry item) {
    heap.push_back(std::move(item));
    python_heap_siftdown(heap, 0U, heap.size() - 1U);
}

DijkstraQueueEntry python_heap_pop(
    std::vector<DijkstraQueueEntry>& heap) {
    DijkstraQueueEntry last = std::move(heap.back());
    heap.pop_back();
    if (heap.empty()) {
        return last;
    }

    DijkstraQueueEntry result = std::move(heap.front());
    heap.front() = std::move(last);
    const std::size_t end = heap.size();
    std::size_t position = 0U;
    const std::size_t start = position;
    DijkstraQueueEntry item = heap[position];
    std::size_t child = 2U * position + 1U;
    while (child < end) {
        const std::size_t right = child + 1U;
        if (right < end &&
            !python_queue_entry_less(heap[child], heap[right])) {
            child = right;
        }
        heap[position] = heap[child];
        position = child;
        child = 2U * position + 1U;
    }
    heap[position] = std::move(item);
    python_heap_siftdown(heap, start, position);
    return result;
}

DijkstraDistances networkx_dijkstra_distances(
    const Graph& graph,
    Vertex source,
    AttrId weight_id) {
    const std::size_t node_count = graph.num_nodes();
    if (source >= node_count) {
        throw std::out_of_range("Dijkstra source is outside the graph");
    }

    DijkstraDistances distances(node_count);
    DijkstraDistances seen(node_count);
    std::vector<DijkstraQueueEntry> heap;
    heap.reserve(node_count);
    std::uint64_t next_order = 0U;
    seen[source] = 0.0;
    python_heap_push(heap, DijkstraQueueEntry{0.0, next_order++, source});

    while (!heap.empty()) {
        const DijkstraQueueEntry current = python_heap_pop(heap);
        const Vertex vertex = current.vertex;
        if (distances[vertex].has_value()) {
            continue;
        }
        distances[vertex] = current.distance;

        for (const auto& edge : graph.neighbors_fast(vertex)) {
            const Vertex neighbor = edge.get_target();
            const AttrValue* raw_weight =
                edge.get_property().attrs.find(weight_id);
            const double weight = raw_weight == nullptr
                ? 1.0
                : attr_to_double(*raw_weight);
            const double candidate = current.distance + weight;

            if (distances[neighbor].has_value()) {
                if (candidate < *distances[neighbor]) {
                    throw std::invalid_argument(
                        "Contradictory paths found: negative weights?");
                }
                continue;
            }
            if (!seen[neighbor].has_value() ||
                candidate < *seen[neighbor]) {
                seen[neighbor] = candidate;
                python_heap_push(
                    heap,
                    DijkstraQueueEntry{
                        candidate, next_order++, neighbor});
            }
        }
    }
    return distances;
}

std::vector<RankedNodeId> ordered_node_ids(
    const rank::NodeRanking& ranking) {
    std::vector<RankedNodeId> nodes;
    nodes.reserve(ranking.size());
    for (const auto& entry : ranking) {
        nodes.push_back(entry.node_id);
    }
    return nodes;
}

core::controller::NodeMapperSelection make_node_mapper_selection(
    const core::controller::ControllerSelection& selection) {
    core::controller::NodeMapperSelection result;
    result.node_constraints = selection.constraints.node_at_node;
    result.node_resources = selection.node_resources;
    result.hard_constraints = selection.hard_node_constraints;
    return result;
}

core::controller::LinkMapperSelection make_link_mapper_selection(
    const core::controller::ControllerSelection& selection) {
    core::controller::LinkMapperSelection result;
    result.link_constraints = selection.constraints.link_at_link;
    result.path_constraints = selection.constraints.link_at_path;
    result.link_resources = selection.link_resources;
    result.hard_constraints = selection.hard_link_constraints;
    result.reusable = selection.reusable;
    return result;
}

std::vector<core::controller::ConstraintLink> ordered_virtual_links(
    const network::VirtualNetwork& virtual_network) {
    const auto& graph = virtual_network.graph();
    std::vector<core::controller::ConstraintLink> links;
    links.reserve(graph.num_edges());
    for (const auto [source, target] : graph.edge_view()) {
        links.push_back(core::controller::ConstraintLink{source, target});
    }
    return links;
}

std::vector<Vertex> selected_physical_nodes(
    const core::Solution& solution) {
    std::vector<Vertex> result;
    result.reserve(solution.node_slots.size());
    for (const auto& entry : solution.node_slots.entries()) {
        result.push_back(static_cast<Vertex>(entry.value));
    }
    return result;
}

struct CandidateScratch {
    std::vector<core::controller::NodeConstraintRequest> requests;
    std::vector<Vertex> suitable;
    std::vector<std::exception_ptr> errors;
    std::vector<double> candidate_scores;
    std::vector<double> physical_scores;
};

template <typename Function>
void parallel_indexed(
    std::size_t count,
    std::size_t workers,
    std::vector<std::exception_ptr>& errors,
    Function&& function) {
    if (count == 0U) {
        return;
    }
    if (workers <= 1U || count == 1U) {
        for (std::size_t index = 0U; index < count; ++index) {
            function(index);
        }
        return;
    }

    errors.assign(count, std::exception_ptr{});
    virne::utils::deterministic_parallel_blocks(
        count,
        workers,
        1U,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                try {
                    function(index);
                } catch (...) {
                    errors[index] = std::current_exception();
                }
            }
        });
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

std::vector<Vertex> node_candidates(
    const network::PhysicalNetwork& physical_network,
    Vertex virtual_node,
    const std::vector<Vertex>& selected,
    const core::controller::PreparedConstraintChecker& checker,
    std::size_t workers,
    const std::vector<Vertex>* link_filter_all_nodes,
    CandidateScratch& scratch) {
    const std::size_t physical_count = physical_network.graph().num_nodes();
    if (scratch.requests.size() != physical_count) {
        scratch.requests.clear();
        scratch.requests.reserve(physical_count);
        for (Vertex physical_node = 0U;
             physical_node < physical_count;
             ++physical_node) {
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
        scratch.requests, workers);

    scratch.suitable.clear();
    scratch.suitable.reserve(physical_count);
    for (Vertex physical_node = 0U;
         physical_node < physical_count;
         ++physical_node) {
        if (checks[physical_node].feasible) {
            scratch.suitable.push_back(physical_node);
        }
    }

    // The pinned Python Controller materializes candidates through set
    // difference. Its check_link_constraint branch computes a suitable set
    // but accidentally ignores it, then performs a second set-difference
    // round trip. Preserve both observable integer-set iteration orders.
    auto candidates = detail::cpython310_int_set_difference_order(
        scratch.suitable, selected);
    if (link_filter_all_nodes != nullptr) {
        const auto new_filter = detail::cpython310_int_set_difference_order(
            *link_filter_all_nodes, candidates);
        candidates = detail::cpython310_int_set_difference_order(
            candidates, new_filter);
    }
    return candidates;
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
            throw std::invalid_argument(
                "physical link resource selection is missing");
        }
        result.push_back(*physical_id);
    }
    return result;
}

void assign_ranking_scores(
    const rank::NodeRanking& ranking,
    std::size_t node_count,
    std::vector<double>& scores) {
    // NRM's validated ranking covers every dense node exactly once, so every
    // retained slot is overwritten and needs no per-round zero fill.
    scores.resize(node_count);
    for (const auto& entry : ranking) {
        scores[entry.node_id] = entry.value;
    }
}

Vertex first_max_candidate(
    const std::vector<Vertex>& candidates,
    const std::vector<double>& scores) {
    if (candidates.empty()) {
        throw std::invalid_argument("candidate list is empty");
    }
    bool has_nan = false;
    for (const double score : scores) {
        has_nan = has_nan || std::isnan(score);
    }
    if (has_nan) {
        std::vector<std::size_t> order(candidates.size());
        for (std::size_t index = 0U; index < order.size(); ++index) {
            order[index] = index;
        }
        rank::detail::python310_timsort_reverse(
            order,
            [&scores](std::size_t left, std::size_t right) {
                return scores[left] < scores[right];
            });
        return candidates[order.front()];
    }

    Vertex selected = candidates.front();
    double best = scores.front();
    for (std::size_t index = 1U; index < candidates.size(); ++index) {
        if (scores[index] > best) {
            selected = candidates[index];
            best = scores[index];
        }
    }
    return selected;
}

} // namespace

CandidateRankSolver::CandidateRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    CandidateRankStrategy strategy,
    rank::NodeRankMethod node_rank_method,
    NodeRankSolverWorkers workers)
    : Solver(dependencies, std::move(config)),
      strategy_(strategy),
      node_rank_method_(node_rank_method),
      node_ranker_(rank::NodeRankSelection{}, rank::NodeRankParameters{}),
      node_mapper_(make_node_mapper_selection(
          dependencies.controller.get().selection())),
      link_mapper_(make_link_mapper_selection(
          dependencies.controller.get().selection())),
      constraint_checker_(
          dependencies.controller.get().selection().constraints),
      selection_(dependencies.controller.get().selection()),
      workers_(workers) {}

CandidateRankStrategy CandidateRankSolver::strategy() const noexcept {
    return strategy_;
}

rank::NodeRankMethod CandidateRankSolver::node_rank_method() const noexcept {
    return node_rank_method_;
}

const NodeRankSolverWorkers& CandidateRankSolver::workers() const noexcept {
    return workers_;
}

core::Solution CandidateRankSolver::solve(const SolverInstance& instance) {
    const auto& virtual_network = instance.virtual_network;
    const auto& physical_network = instance.physical_network;
    core::Solution solution = core::Solution::from_v_net(virtual_network);

    rank::NodeRankOptions rank_options;
    rank_options.sort = true;
    rank_options.workers = workers_.rank_workers;
    rank_options.max_iterations = std::nullopt;
    const auto virtual_ranking = node_ranker_.prepare(virtual_network).rank(
        node_rank_method_, rank_options);
    const auto virtual_nodes = ordered_node_ids(virtual_ranking);

    network::PhysicalNetwork working_physical = physical_network.clone();
    MutableSolverResult result = solve_on_physical(
        virtual_network,
        working_physical,
        std::move(solution),
        virtual_nodes,
        nullptr);
    return std::move(result.solution);
}

MutableSolverResult CandidateRankSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    const auto& virtual_network = instance.virtual_network;
    core::Solution solution = core::Solution::from_v_net(virtual_network);

    rank::NodeRankOptions rank_options;
    rank_options.sort = true;
    rank_options.workers = workers_.rank_workers;
    rank_options.max_iterations = std::nullopt;
    const auto virtual_ranking = node_ranker_.prepare(virtual_network).rank(
        node_rank_method_, rank_options);
    const auto virtual_nodes = ordered_node_ids(virtual_ranking);

    return solve_on_physical(
        virtual_network,
        instance.physical_network,
        std::move(solution),
        virtual_nodes,
        &instance.mutation);
}

MutableSolverResult CandidateRankSolver::solve_on_physical(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    core::Solution solution,
    const std::vector<Vertex>& virtual_nodes,
    core::controller::PreparedControllerMutation* mutation) {
    bool rollback_started = false;
    const auto rollback = [&]() {
        if (mutation != nullptr) {
            rollback_started = true;
            mutation->rollback(solution);
        }
    };

    try {
      rank::NodeRankOptions rank_options;
      // PL immediately reindexes every NRM value by physical node ID before
      // applying its own candidate-order stable maximum. The intermediate
      // ranking order is therefore unobservable; skip its O(P log P) sort.
      rank_options.sort = false;
      rank_options.workers = workers_.rank_workers;
      rank_options.max_iterations = std::nullopt;

      auto prepared_checker = constraint_checker_.prepare(
          virtual_network, physical_network);
      auto prepared_node_mapper = node_mapper_.prepare(
          virtual_network, physical_network);

    const std::size_t physical_node_count =
        physical_network.graph().num_nodes();
    std::vector<ResourceId> physical_link_resources;
    std::vector<DistanceMatrix> physical_link_adjacency;
    std::optional<nx::AllPairsPathLengths> all_path_lengths;
    std::optional<nx::AllPairsPaths> all_paths;
    std::optional<rank::PreparedNodeRanker> prepared_physical_ranker;
    std::optional<AttrId> weight_id;
    std::vector<std::optional<DijkstraDistances>> weighted_distances;
    std::vector<Vertex> link_filter_all_nodes;

    if (strategy_ == CandidateRankStrategy::essentiality) {
        physical_link_resources = resolve_physical_link_resources(
            virtual_network,
            physical_network,
            selection_.link_resources);
        link_filter_all_nodes.resize(physical_node_count);
        for (Vertex physical_node = 0U;
             physical_node < physical_node_count;
             ++physical_node) {
            link_filter_all_nodes[physical_node] = physical_node;
        }
    }

    bool node_mapping_succeeded = true;
    // NodeMapper appends the chosen physical node to node_slots on each
    // successful placement. Retain the same insertion-order list directly
    // instead of rebuilding and reallocating it from Solution every round.
    std::vector<Vertex> selected = selected_physical_nodes(solution);
    selected.reserve(selected.size() + virtual_nodes.size());
    CandidateScratch candidate_scratch;
    candidate_scratch.requests.reserve(physical_node_count);
    candidate_scratch.suitable.reserve(physical_node_count);

    for (const Vertex virtual_node : virtual_nodes) {
        const auto candidates = node_candidates(
            physical_network,
            virtual_node,
            selected,
            prepared_checker,
            workers_.node_candidate_workers,
            strategy_ == CandidateRankStrategy::essentiality
                ? &link_filter_all_nodes
                : nullptr,
            candidate_scratch);
        if (candidates.empty()) {
            solution.place_result = false;
            node_mapping_succeeded = false;
            break;
        }

        auto& candidate_scores = candidate_scratch.candidate_scores;
        candidate_scores.resize(candidates.size());
        if (strategy_ == CandidateRankStrategy::proximity) {
            if (!prepared_physical_ranker.has_value()) {
                // Preparation binds only registry/AttrId identities; ordinary
                // resource mutations are deliberately observed by each rank.
                // Bind at the original first-use point, then reuse it.
                prepared_physical_ranker.emplace(
                    node_ranker_.prepare(physical_network));
            }
            const auto physical_ranking =
                prepared_physical_ranker->rank_nrm(rank_options);
            assign_ranking_scores(
                physical_ranking,
                physical_node_count,
                candidate_scratch.physical_scores);
            const auto& physical_scores = candidate_scratch.physical_scores;
            if (!selected.empty() && !weight_id.has_value()) {
                // Python uses the conventional dynamic `weight` key. Resolve
                // it once; every relaxation below reads the numeric AttrId.
                weight_id = physical_network.graph().attr_id("weight");
                weighted_distances.resize(physical_node_count);
            }
            parallel_indexed(
                candidates.size(),
                workers_.node_candidate_workers,
                candidate_scratch.errors,
                 [&](std::size_t index) {
                    const Vertex candidate = candidates[index];
                    double distance_sum = 0.0;
                    if (!selected.empty()) {
                        auto& cached = weighted_distances[candidate];
                        if (!cached.has_value()) {
                            cached = networkx_dijkstra_distances(
                                physical_network.graph(),
                                candidate,
                                *weight_id);
                        }
                        for (const Vertex placed : selected) {
                            const auto& distance = cached->at(placed);
                            if (!distance.has_value()) {
                                throw std::out_of_range(
                                    "selected physical node is unreachable");
                            }
                            distance_sum += *distance;
                        }
                    }
                    candidate_scores[index] =
                        physical_scores[candidate] /
                        (distance_sum + 1.0e-6);
                 });
        } else {
            if (!all_path_lengths.has_value()) {
                all_path_lengths.emplace(
                    nx::shortest_path_length(physical_network.graph()));
                all_paths.emplace(nx::shortest_path(physical_network.graph()));
                physical_link_adjacency =
                    network::get_adjacency_attrs_data(
                        physical_network,
                        physical_link_resources,
                        false,
                        workers_.rank_workers);
            }
            parallel_indexed(
                candidates.size(),
                workers_.node_candidate_workers,
                candidate_scratch.errors,
                [&](std::size_t index) {
                    const Vertex candidate = candidates[index];
                    double hop_sum = 0.0;
                    double path_capacity_sum = 0.0;
                    for (const Vertex placed : selected) {
                        const auto hop_count =
                            all_path_lengths->at(candidate).at(placed);
                        hop_sum += static_cast<double>(hop_count);
                        const auto& path =
                            all_paths->at(candidate).at(placed);
                        double free_resource = 0.0;
                        for (const auto& matrix : physical_link_adjacency) {
                            for (std::size_t path_index = 1U;
                                 path_index < path.size();
                                 ++path_index) {
                                free_resource += matrix(
                                    path[path_index - 1U], path[path_index]);
                            }
                        }
                        path_capacity_sum += free_resource /
                            (static_cast<double>(hop_count) + 1.0e-6);
                    }
                    const double degree = static_cast<double>(
                        physical_network.graph().degree(candidate));
                    candidate_scores[index] =
                        degree / (1.0 + hop_sum) *
                        (2.0 + path_capacity_sum);
                });
        }

        const Vertex physical_node =
            first_max_candidate(candidates, candidate_scores);
        const auto placement = prepared_node_mapper.place(
            virtual_node, physical_node, solution);
        if (!placement.placed) {
            node_mapping_succeeded = false;
            break;
        }
        selected.push_back(physical_node);
    }

    if (!node_mapping_succeeded) {
        solution.place_result = false;
        solution.result = false;
        rollback();
        return MutableSolverResult{
            std::move(solution), SolverMutationState::detached};
    }

    const auto virtual_links = ordered_virtual_links(virtual_network);
    bool link_mapping_succeeded = false;
    {
        auto mapper = link_mapper_.prepare(
            virtual_network, physical_network);
        core::controller::LinkMappingOptions options;
        options.shortest_method = config().shortest_method;
        options.k = config().k_shortest;
        options.max_path_nodes = 1.0e6;
        options.topology_constraint_workers =
            workers_.link_topology_constraint_workers;
        options.candidate_workers = workers_.link_candidate_workers;
        options.inplace = true;
        options.allow_constraint_violation = false;
        link_mapping_succeeded = mapper.link_mapping(
            virtual_links, solution, options);
    }
    if (!link_mapping_succeeded) {
        solution.route_result = false;
        solution.result = false;
        rollback();
        return MutableSolverResult{
            std::move(solution), SolverMutationState::detached};
    }
    solution.result = true;
    return MutableSolverResult{
        std::move(solution), SolverMutationState::committed};
    } catch (...) {
        if (mutation != nullptr && !rollback_started) {
            rollback();
        }
        throw;
    }
}

PLRankSolver::PLRankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : CandidateRankSolver(
          std::move(dependencies),
          std::move(config),
          CandidateRankStrategy::proximity,
          rank::NodeRankMethod::nrm,
          workers) {}

NEARankSolver::NEARankSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    NodeRankSolverWorkers workers)
    : CandidateRankSolver(
          std::move(dependencies),
          std::move(config),
          CandidateRankStrategy::essentiality,
          rank::NodeRankMethod::nea,
          workers) {}

SolverId register_pl_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "pl_rank",
        SolverCategory::node_ranking,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<PLRankSolver>(
                std::move(dependencies), std::move(config), workers);
        });
}

SolverId register_nea_rank_solver(
    SolverRegistry& registry,
    NodeRankSolverWorkers workers) {
    return registry.register_solver(
        "nea_rank",
        SolverCategory::node_ranking,
        [workers](SolverDependencies dependencies,
                  SolverConfig config) -> std::unique_ptr<Solver> {
            return std::make_unique<NEARankSolver>(
                std::move(dependencies), std::move(config), workers);
        });
}

} // namespace virne::solver::heuristic
