#include "node_mapper.h"

#include "../../utils/deterministic_executor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace virne::core::controller
{
namespace
{

using AttributeNumber = network::attribute::AttributeNumber;
using ResourceUpdateOperation = network::attribute::ResourceUpdateOperation;

constexpr std::size_t minimum_parallel_candidate_count = 128U;
constexpr std::size_t maximum_sequential_probe_count = 8U;
constexpr std::size_t minimum_candidate_window = 32U;
constexpr std::size_t candidate_window_per_worker = 8U;

SolutionNodeId solution_node_id(Vertex node) noexcept
{
    return static_cast<SolutionNodeId>(node);
}

AttributeNumber non_negative_python_max(const AttributeNumber& value)
{
    return std::visit(
        [](const auto& number) -> AttributeNumber
        {
            using Number = std::decay_t<decltype(number)>;
            if constexpr (std::is_same_v<Number, bool>)
            {
                return number;
            }
            else if constexpr (std::is_same_v<Number, std::int64_t>)
            {
                return number < 0 ? AttributeNumber{std::int64_t{0}}
                                  : AttributeNumber{number};
            }
            else
            {
                // Python max(value, 0) retains value on equality and when the
                // first value is NaN. This also preserves negative zero.
                return number < 0.0 ? AttributeNumber{std::int64_t{0}}
                                    : AttributeNumber{number};
            }
        },
        value);
}

long double number_as_long_double(const AttributeNumber& value)
{
    return std::visit(
        [](const auto& number) -> long double
        {
            return static_cast<long double>(number);
        },
        value);
}

bool python_max_replaces(
    const AttributeNumber& current,
    const AttributeNumber& candidate)
{
    // Python max keeps the earlier element on equality. Comparisons with NaN
    // are false in either direction, so the earlier value is retained too.
    return number_as_long_double(candidate) > number_as_long_double(current);
}

double number_as_double(const AttributeNumber& value)
{
    return std::visit(
        [](const auto& number) -> double
        {
            return static_cast<double>(number);
        },
        value);
}

bool contains_id(
    const std::vector<ConstraintId>& values,
    ConstraintId id)
{
    return std::find(values.begin(), values.end(), id) != values.end();
}

} // namespace

NodeMapperException::NodeMapperException(
    NodeMapperErrorCode code,
    NodeMapperOperation operation,
    std::string message,
    std::optional<Vertex> virtual_node,
    std::optional<Vertex> physical_node,
    std::optional<ResourceId> resource_id,
    std::optional<ConstraintId> constraint_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      virtual_node_(virtual_node),
      physical_node_(physical_node),
      resource_id_(resource_id),
      constraint_id_(constraint_id)
{
}

NodeMapperErrorCode NodeMapperException::code() const noexcept
{
    return code_;
}

NodeMapperOperation NodeMapperException::operation() const noexcept
{
    return operation_;
}

const std::optional<Vertex>& NodeMapperException::virtual_node() const noexcept
{
    return virtual_node_;
}

const std::optional<Vertex>& NodeMapperException::physical_node() const noexcept
{
    return physical_node_;
}

const std::optional<ResourceId>& NodeMapperException::resource_id() const noexcept
{
    return resource_id_;
}

const std::optional<ConstraintId>& NodeMapperException::constraint_id() const noexcept
{
    return constraint_id_;
}

NodeMapper::NodeMapper(NodeMapperSelection selection)
    : selection_(std::move(selection))
{
}

const NodeMapperSelection& NodeMapper::selection() const noexcept
{
    return selection_;
}

PreparedNodeMapper NodeMapper::prepare(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network) const
{
    const auto& registry = virtual_network.node_attributes();
    const auto& entries = registry.entries();

    std::vector<PreparedNodeMapper::PreparedNodeResource> node_resources;
    node_resources.reserve(selection_.node_resources.size());
    std::vector<ResourceId> effective_resource_ids;
    effective_resource_ids.reserve(selection_.node_resources.size());

    for (const ResourceId id : selection_.node_resources)
    {
        if (static_cast<std::size_t>(id) >= entries.size())
        {
            throw NodeMapperException(
                NodeMapperErrorCode::invalid_node_resource_selection,
                NodeMapperOperation::prepare,
                "node resource selection is outside the virtual registry",
                std::nullopt,
                std::nullopt,
                id);
        }
        const bool duplicate = std::find_if(
            node_resources.begin(),
            node_resources.end(),
            [id](const PreparedNodeMapper::PreparedNodeResource& value)
            {
                return value.resource_id == id;
            }) != node_resources.end();
        if (duplicate)
        {
            continue;
        }

        const auto binding = virtual_network.bind_node_attribute(
            entries[static_cast<std::size_t>(id)].name);
        if (!binding.has_value())
        {
            throw NodeMapperException(
                NodeMapperErrorCode::invalid_node_resource_selection,
                NodeMapperOperation::prepare,
                "node resource has no virtual graph binding",
                std::nullopt,
                std::nullopt,
                id);
        }
        node_resources.push_back(
            PreparedNodeMapper::PreparedNodeResource{id, binding->value_id});
        effective_resource_ids.push_back(id);
    }

    ConstraintCheckerSelection checker_selection;
    checker_selection.node_at_node = selection_.node_constraints;
    ConstraintChecker checker(std::move(checker_selection));
    PreparedConstraintChecker prepared_checker =
        checker.prepare(virtual_network, physical_network);

    ResourceUpdatorSelection updator_selection;
    updator_selection.node_resources = std::move(effective_resource_ids);
    ResourceUpdator updator(std::move(updator_selection));
    PreparedResourceUpdator prepared_updator =
        updator.prepare(virtual_network, physical_network);

    std::vector<ConstraintId> constraint_order;
    constraint_order.reserve(selection_.node_constraints.size());
    for (const ConstraintId id : selection_.node_constraints)
    {
        if (!contains_id(constraint_order, id))
        {
            constraint_order.push_back(id);
        }
    }

    std::vector<std::uint8_t> hard_constraint_mask(
        registry.size(), std::uint8_t{0});
    for (const ConstraintId id : selection_.hard_constraints)
    {
        if (static_cast<std::size_t>(id) < hard_constraint_mask.size())
        {
            hard_constraint_mask[static_cast<std::size_t>(id)] =
                std::uint8_t{1};
        }
    }

    return PreparedNodeMapper(
        selection_,
        virtual_network,
        physical_network,
        std::move(prepared_checker),
        std::move(prepared_updator),
        std::move(node_resources),
        std::move(constraint_order),
        std::move(hard_constraint_mask));
}

PreparedNodeMapper::PreparedNodeMapper(
    NodeMapperSelection selection,
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    PreparedConstraintChecker constraint_checker,
    PreparedResourceUpdator resource_updator,
    std::vector<PreparedNodeResource> node_resources,
    std::vector<ConstraintId> constraint_order,
    std::vector<std::uint8_t> hard_constraint_mask)
    : selection_(std::move(selection)),
      virtual_network_(&virtual_network),
      physical_network_(&physical_network),
      constraint_checker_(std::move(constraint_checker)),
      resource_updator_(std::move(resource_updator)),
      node_resources_(std::move(node_resources)),
      constraint_order_(std::move(constraint_order)),
      hard_constraint_mask_(std::move(hard_constraint_mask))
{
}

const std::vector<ResourceAmount>& PreparedNodeMapper::gather_node_resources(
    Vertex virtual_node,
    SolutionAttributeValues* recorded_values)
{
    const AttrMap& node = virtual_network_->graph().node_attrs(virtual_node);
    auto& resources = resource_scratch_;
    resources.clear();
    resources.reserve(node_resources_.size());
    for (const PreparedNodeResource& prepared : node_resources_)
    {
        const AttrValue* value = node.find(prepared.virtual_value_id);
        if (value == nullptr)
        {
            throw NodeMapperException(
                NodeMapperErrorCode::missing_node_resource_value,
                NodeMapperOperation::place,
                "virtual node resource value is missing",
                virtual_node,
                std::nullopt,
                prepared.resource_id);
        }

        AttributeNumber number;
        if (const auto* integer = std::get_if<std::int64_t>(value))
        {
            number = *integer;
        }
        else if (const auto* floating = std::get_if<double>(value))
        {
            number = *floating;
        }
        else if (const auto* boolean = std::get_if<bool>(value))
        {
            number = *boolean;
        }
        else
        {
            throw NodeMapperException(
                NodeMapperErrorCode::non_numeric_node_resource,
                NodeMapperOperation::place,
                "virtual node resource value is not numeric",
                virtual_node,
                std::nullopt,
                prepared.resource_id);
        }
        resources.push_back(ResourceAmount{prepared.resource_id, number});
        if (recorded_values != nullptr)
        {
            recorded_values->set(prepared.resource_id, std::move(number));
        }
    }
    return resources;
}

NodePlacementResult PreparedNodeMapper::commit_place_after_check(
    Vertex virtual_node,
    Vertex physical_node,
    Solution& solution,
    ConstraintCheckResult check,
    bool allow_constraint_violation)
{
    if (!allow_constraint_violation && !check.feasible)
    {
        return NodePlacementResult{false, std::move(check)};
    }

    SolutionAttributeValues recorded_values;
    const std::vector<ResourceAmount>& resources =
        gather_node_resources(virtual_node, &recorded_values);
    resource_updator_.update_node_resources(
        physical_node,
        resources,
        ResourceUpdateOperation::subtract,
        !allow_constraint_violation);

    const SolutionNodeId virtual_id = solution_node_id(virtual_node);
    const SolutionNodeId physical_id = solution_node_id(physical_node);
    solution.node_slots.insert_or_assign(virtual_id, physical_id);
    solution.node_slots_info.insert_or_assign(
        NodeSlotInfoKey{virtual_id, physical_id},
        std::move(recorded_values));
    return NodePlacementResult{true, std::move(check)};
}

NodePlacementResult PreparedNodeMapper::place(
    Vertex virtual_node,
    Vertex physical_node,
    Solution& solution,
    NodePlacementOptions options)
{
    ConstraintCheckResult check =
        constraint_checker_.check_node_level_constraints(
            virtual_node, physical_node);
    NodePlacementResult result = commit_place_after_check(
        virtual_node,
        physical_node,
        solution,
        std::move(check),
        options.allow_constraint_violation);
    if (options.record_constraint_violation)
    {
        record_place_constraint_violation(
            virtual_node, result.check.offsets, solution);
    }
    return result;
}

void PreparedNodeMapper::record_place_constraint_violation(
    Vertex virtual_node,
    const SolutionAttributeValues& offsets,
    Solution& solution) const
{
    const SolutionNodeId virtual_id = solution_node_id(virtual_node);
    solution.v_net_constraint_offsets.node_level.insert_or_assign(
        virtual_id, offsets);

    SolutionAttributeValues violations;
    const auto& slots = offsets.slots();
    for (std::size_t index = 0U; index < slots.size(); ++index)
    {
        if (slots[index].has_value())
        {
            violations.set(
                static_cast<ConstraintId>(index),
                non_negative_python_max(*slots[index]));
        }
    }
    solution.v_net_constraint_violations.node_level.insert_or_assign(
        virtual_id, std::move(violations));

    std::optional<AttributeNumber> maximum;
    for (const ConstraintId id : constraint_order_)
    {
        if (static_cast<std::size_t>(id) >= hard_constraint_mask_.size() ||
            hard_constraint_mask_[static_cast<std::size_t>(id)] ==
                std::uint8_t{0})
        {
            continue;
        }
        const AttributeNumber* value = offsets.find(id);
        if (value == nullptr)
        {
            continue;
        }
        if (!maximum.has_value() || python_max_replaces(*maximum, *value))
        {
            maximum = *value;
        }
    }
    if (!maximum.has_value())
    {
        throw NodeMapperException(
            NodeMapperErrorCode::empty_hard_constraint_offsets,
            NodeMapperOperation::record_violation,
            "no checked offset matches a selected hard constraint",
            virtual_node);
    }

    const AttributeNumber maximum_violation =
        non_negative_python_max(*maximum);
    solution.v_net_total_hard_constraint_violation +=
        number_as_double(maximum_violation);
}

bool PreparedNodeMapper::undo_place(
    Vertex virtual_node,
    Solution& solution)
{
    const SolutionNodeId virtual_id = solution_node_id(virtual_node);
    const auto slot_id = solution.node_slots.find_id(virtual_id);
    if (!slot_id.has_value())
    {
        throw NodeMapperException(
            NodeMapperErrorCode::placement_not_found,
            NodeMapperOperation::undo_place,
            "virtual node has no placement",
            virtual_node);
    }

    const SolutionNodeId stored_physical = solution.node_slots.at(*slot_id);
    if (stored_physical < 0 ||
        static_cast<std::uint64_t>(stored_physical) >
            static_cast<std::uint64_t>(std::numeric_limits<Vertex>::max()))
    {
        throw NodeMapperException(
            NodeMapperErrorCode::placement_not_found,
            NodeMapperOperation::undo_place,
            "stored physical node is outside the native vertex range",
            virtual_node);
    }
    const Vertex physical_node = static_cast<Vertex>(stored_physical);
    const NodeSlotInfoKey info_key{virtual_id, stored_physical};
    const auto info_id = solution.node_slots_info.find_id(info_key);
    if (!info_id.has_value())
    {
        throw NodeMapperException(
            NodeMapperErrorCode::placement_info_not_found,
            NodeMapperOperation::undo_place,
            "placement resource information is missing",
            virtual_node,
            physical_node);
    }

    const SolutionAttributeValues& recorded =
        solution.node_slots_info.at(*info_id);
    auto& resources = resource_scratch_;
    resources.clear();
    resources.reserve(node_resources_.size());
    for (const PreparedNodeResource& prepared : node_resources_)
    {
        const AttributeNumber* value = recorded.find(prepared.resource_id);
        if (value == nullptr)
        {
            throw NodeMapperException(
                NodeMapperErrorCode::placement_info_not_found,
                NodeMapperOperation::undo_place,
                "placement resource value is missing",
                virtual_node,
                physical_node,
                prepared.resource_id);
        }
        resources.push_back(ResourceAmount{prepared.resource_id, *value});
    }

    resource_updator_.update_node_resources(
        physical_node,
        resources,
        ResourceUpdateOperation::add,
        true);
    solution.node_slots.erase(virtual_id);
    solution.node_slots_info.erase(info_key);
    return true;
}

std::vector<PreparedNodeMapper::CandidateCheckOutcome>&
PreparedNodeMapper::check_candidates_ordered(
    Vertex virtual_node,
    const std::vector<Vertex>& physical_nodes,
    std::size_t begin_index,
    std::size_t end_index,
    std::size_t workers)
{
    const std::size_t count = end_index - begin_index;
    auto& outcomes = candidate_check_scratch_;
    outcomes.assign(count, CandidateCheckOutcome{});
    const auto check_range =
        [this,
         virtual_node,
         &physical_nodes,
         &outcomes,
         begin_index](
            std::size_t begin,
            std::size_t end)
        {
            for (std::size_t local_index = begin;
                 local_index < end;
                 ++local_index)
            {
                const std::size_t candidate_index =
                    begin_index + local_index;
                try
                {
                    outcomes[local_index].result.emplace(
                        constraint_checker_.check_node_level_constraints(
                            virtual_node,
                            physical_nodes[candidate_index]));
                }
                catch (...)
                {
                    outcomes[local_index].error = std::current_exception();
                }
            }
        };

    virne::utils::deterministic_parallel_blocks(
        count,
        workers,
        1U,
        check_range);
    return outcomes;
}

bool PreparedNodeMapper::node_mapping(
    const std::vector<Vertex>& virtual_nodes,
    const std::vector<Vertex>& physical_nodes,
    Solution& solution,
    NodeMappingOptions options)
{
    if (options.allow_constraint_violation)
    {
        throw NodeMapperException(
            NodeMapperErrorCode::unsupported_constraint_violation_mapping,
            NodeMapperOperation::node_mapping,
            "constraint-violation node mapping is not implemented");
    }
    if (options.method != NodeMatchingMethod::greedy &&
        options.method != NodeMatchingMethod::l2s2)
    {
        throw NodeMapperException(
            NodeMapperErrorCode::invalid_matching_method,
            NodeMapperOperation::node_mapping,
            "invalid node matching method");
    }

    if (!options.inplace)
    {
        solution.node_slots.clear();
        solution.node_slots_info.clear();
        network::PhysicalNetwork physical_copy = physical_network_->clone();
        NodeMapper mapper(selection_);
        PreparedNodeMapper prepared =
            mapper.prepare(*virtual_network_, physical_copy);
        options.inplace = true;
        return prepared.node_mapping_inplace(
            virtual_nodes,
            physical_nodes,
            solution,
            options,
            false);
    }

    return node_mapping_inplace(
        virtual_nodes,
        physical_nodes,
        solution,
        options,
        true);
}

bool PreparedNodeMapper::node_mapping_inplace(
    const std::vector<Vertex>& virtual_nodes,
    const std::vector<Vertex>& physical_nodes,
    Solution& solution,
    const NodeMappingOptions& options,
    bool clear_solution)
{
    if (clear_solution)
    {
        solution.node_slots.clear();
        solution.node_slots_info.clear();
    }
    if (!virtual_nodes.empty() && physical_nodes.empty())
    {
        throw NodeMapperException(
            NodeMapperErrorCode::empty_physical_candidates,
            NodeMapperOperation::node_mapping,
            "physical candidate list is empty",
            virtual_nodes.front());
    }

    std::vector<Vertex> candidates = physical_nodes;
    for (const Vertex virtual_node : virtual_nodes)
    {
        if (candidates.empty())
        {
            throw NodeMapperException(
                NodeMapperErrorCode::empty_physical_candidates,
                NodeMapperOperation::node_mapping,
                "physical candidate list was exhausted",
                virtual_node);
        }

        if (options.method == NodeMatchingMethod::l2s2)
        {
            NodePlacementOptions placement_options;
            placement_options.record_constraint_violation = false;
            NodePlacementResult placement = place(
                virtual_node,
                candidates.front(),
                solution,
                placement_options);
            record_place_constraint_violation(
                virtual_node, placement.check.offsets, solution);
            if (!placement.placed)
            {
                solution.place_result = false;
                solution.result = false;
                return false;
            }
            if (!options.reusable)
            {
                candidates.erase(candidates.begin());
            }
            continue;
        }

        bool placed = false;
        ConstraintCheckResult last_check;
        bool has_last_check = false;
        const auto commit_candidate =
            [&](std::size_t index, ConstraintCheckResult check)
            {
                NodePlacementResult placement = commit_place_after_check(
                    virtual_node,
                    candidates[index],
                    solution,
                    std::move(check),
                    false);
                record_place_constraint_violation(
                    virtual_node, placement.check.offsets, solution);
                if (!options.reusable)
                {
                    candidates.erase(
                        candidates.begin() +
                        static_cast<std::ptrdiff_t>(index));
                }
                placed = true;
            };

        const bool use_parallel_windows =
            options.candidate_workers > 1U &&
            candidates.size() >= minimum_parallel_candidate_count;
        if (!use_parallel_windows)
        {
            for (std::size_t index = 0U; index < candidates.size(); ++index)
            {
                last_check =
                    constraint_checker_.check_node_level_constraints(
                        virtual_node, candidates[index]);
                has_last_check = true;
                if (!last_check.feasible)
                {
                    continue;
                }
                commit_candidate(index, std::move(last_check));
                break;
            }
        }
        else
        {
            const std::size_t bounded_workers = std::min(
                options.candidate_workers,
                candidates.size());
            const std::size_t probe_count = std::min(
                {candidates.size(),
                 bounded_workers,
                 maximum_sequential_probe_count});

            // The high-capacity prefix is normally successful.  Preserve
            // Python's zero-fan-out early return before opening a worker
            // window; only a difficult search pays parallel dispatch cost.
            for (std::size_t index = 0U; index < probe_count; ++index)
            {
                last_check =
                    constraint_checker_.check_node_level_constraints(
                        virtual_node, candidates[index]);
                has_last_check = true;
                if (!last_check.feasible)
                {
                    continue;
                }
                commit_candidate(index, std::move(last_check));
                break;
            }

            std::size_t scaled_window = candidates.size();
            if (bounded_workers <=
                std::numeric_limits<std::size_t>::max() /
                    candidate_window_per_worker)
            {
                scaled_window =
                    bounded_workers * candidate_window_per_worker;
            }
            const std::size_t window_width = std::max(
                minimum_candidate_window,
                scaled_window);

            for (std::size_t window_begin = probe_count;
                 !placed && window_begin < candidates.size();)
            {
                const std::size_t remaining =
                    candidates.size() - window_begin;
                const std::size_t window_end = window_begin +
                    std::min(window_width, remaining);
                std::vector<CandidateCheckOutcome>& outcomes =
                    check_candidates_ordered(
                        virtual_node,
                        candidates,
                        window_begin,
                        window_end,
                        options.candidate_workers);
                for (std::size_t local_index = 0U;
                     local_index < outcomes.size();
                     ++local_index)
                {
                    CandidateCheckOutcome& outcome = outcomes[local_index];
                    if (outcome.error)
                    {
                        std::rethrow_exception(outcome.error);
                    }
                    if (!outcome.result.has_value())
                    {
                        throw std::logic_error(
                            "candidate check produced no result");
                    }
                    last_check = std::move(*outcome.result);
                    has_last_check = true;
                    if (!last_check.feasible)
                    {
                        continue;
                    }
                    commit_candidate(
                        window_begin + local_index,
                        std::move(last_check));
                    break;
                }
                window_begin = window_end;
            }
        }
        if (!placed)
        {
            if (!has_last_check)
            {
                throw NodeMapperException(
                    NodeMapperErrorCode::empty_physical_candidates,
                    NodeMapperOperation::node_mapping,
                    "physical candidate list is empty",
                    virtual_node);
            }
            record_place_constraint_violation(
                virtual_node, last_check.offsets, solution);
            solution.place_result = false;
            solution.result = false;
            return false;
        }
    }

    if (solution.node_slots.size() != virtual_network_->num_nodes())
    {
        throw NodeMapperException(
            NodeMapperErrorCode::mapping_cardinality_mismatch,
            NodeMapperOperation::node_mapping,
            "mapped node count does not match the virtual network");
    }
    return true;
}

} // namespace virne::core::controller
