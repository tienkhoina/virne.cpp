#include "controller.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <thread>
#include <unordered_set>
#include <utility>

namespace virne::core::controller
{
namespace
{

template <typename Id>
std::vector<Id> deduplicate_first(const std::vector<Id>& values)
{
    std::vector<Id> result;
    result.reserve(values.size());
    std::unordered_set<Id> seen;
    seen.reserve(values.size());
    for (const Id value : values)
    {
        if (seen.insert(value).second)
        {
            result.push_back(value);
        }
    }
    return result;
}

std::vector<std::uint8_t> make_hard_mask(
    const std::vector<ConstraintId>& selected,
    const std::vector<ConstraintId>& hard)
{
    if (selected.empty())
    {
        return {};
    }

    const ConstraintId maximum =
        *std::max_element(selected.begin(), selected.end());
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(maximum) + 1U,
        std::uint8_t{0U});
    std::unordered_set<ConstraintId> hard_ids(hard.begin(), hard.end());
    for (const ConstraintId id : selected)
    {
        if (hard_ids.find(id) != hard_ids.end())
        {
            mask[id] = std::uint8_t{1U};
        }
    }
    return mask;
}

std::vector<ConstraintId> combined_constraints(
    const std::vector<ConstraintId>& first,
    const std::vector<ConstraintId>& second)
{
    std::vector<ConstraintId> result;
    result.reserve(first.size() + second.size());
    result.insert(result.end(), first.begin(), first.end());
    result.insert(result.end(), second.begin(), second.end());
    return deduplicate_first(result);
}

SolutionAttributeValues zero_values(
    const std::vector<ConstraintId>& constraints)
{
    SolutionAttributeValues result;
    for (const ConstraintId id : constraints)
    {
        result.set(id, 0.0);
    }
    return result;
}

std::vector<std::optional<AttrId>> bind_node_resource_values(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<ResourceId>& resources)
{
    std::vector<std::optional<AttrId>> result(
        virtual_network.node_attributes().size());
    const auto& entries = virtual_network.node_attributes().entries();
    for (const ResourceId id : resources)
    {
        const auto binding =
            physical_network.bind_node_attribute(entries.at(id).name);
        if (!binding.has_value())
        {
            throw ControllerException(
                ControllerErrorCode::missing_node_slot_info,
                ControllerOperation::prepare,
                "prepared node resource has no physical value binding",
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                id);
        }
        result[id] = binding->value_id;
    }
    return result;
}

std::vector<std::optional<AttrId>> bind_link_resource_values(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<ResourceId>& resources)
{
    std::vector<std::optional<AttrId>> result(
        virtual_network.link_attributes().size());
    const auto& entries = virtual_network.link_attributes().entries();
    for (const ResourceId id : resources)
    {
        const auto binding =
            physical_network.bind_link_attribute(entries.at(id).name);
        if (!binding.has_value())
        {
            throw ControllerException(
                ControllerErrorCode::missing_link_path_info,
                ControllerOperation::prepare,
                "prepared link resource has no physical value binding",
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                id);
        }
        result[id] = binding->value_id;
    }
    return result;
}

struct DirectMutation
{
    AttrValue* target = nullptr;
    AttrValue value = std::int64_t{0};
};

bool plan_numeric_mutation(
    AttrValue& current,
    const network::attribute::AttributeNumber& requested,
    network::attribute::ResourceUpdateOperation operation,
    AttrValue& result)
{
    network::attribute::AttributeNumber physical;
    if (const auto* integer_value = std::get_if<std::int64_t>(&current))
    {
        physical = *integer_value;
    }
    else if (const auto* floating_value = std::get_if<double>(&current))
    {
        physical = *floating_value;
    }
    else if (const auto* boolean_value = std::get_if<bool>(&current))
    {
        physical = *boolean_value;
    }
    else
    {
        return false;
    }

    try
    {
        network::attribute::update_resource_value(
            requested,
            physical,
            operation,
            true);
    }
    catch (const network::attribute::AttributeMethodException&)
    {
        return false;
    }

    if (const auto* integer_result = std::get_if<std::int64_t>(&physical))
    {
        result = *integer_result;
    }
    else if (const auto* floating_result = std::get_if<double>(&physical))
    {
        result = *floating_result;
    }
    else
    {
        result = std::get<bool>(physical);
    }
    return true;
}

void commit_direct_mutations(
    std::vector<DirectMutation>& mutations,
    std::size_t requested_workers)
{
    const std::size_t worker_count = std::min(
        requested_workers,
        mutations.size());
    if (worker_count <= 1U)
    {
        for (auto& mutation : mutations)
        {
            *mutation.target = std::move(mutation.value);
        }
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    try
    {
        for (std::size_t worker = 0U; worker < worker_count; ++worker)
        {
            threads.emplace_back(
                [&, worker]
                {
                    const std::size_t begin =
                        mutations.size() * worker / worker_count;
                    const std::size_t end =
                        mutations.size() * (worker + 1U) / worker_count;
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        *mutations[index].target =
                            std::move(mutations[index].value);
                    }
                });
        }
    }
    catch (...)
    {
        for (std::thread& thread : threads)
        {
            thread.join();
        }
        throw;
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
}

SolutionLink solution_link(ConstraintLink link) noexcept
{
    return SolutionLink{
        static_cast<SolutionNodeId>(link.source),
        static_cast<SolutionNodeId>(link.target)};
}

double number_as_double(
    const network::attribute::AttributeNumber& value) noexcept
{
    return std::visit(
        [](const auto number) noexcept
        {
            return static_cast<double>(number);
        },
        value);
}

Vertex checked_vertex(
    SolutionNodeId value,
    ControllerOperation operation,
    std::optional<Vertex> virtual_node = std::nullopt)
{
    if (value < 0 ||
        static_cast<std::uint64_t>(value) >
            static_cast<std::uint64_t>(std::numeric_limits<Vertex>::max()))
    {
        throw ControllerException(
            ControllerErrorCode::invalid_stored_node,
            operation,
            "stored solution node is outside the native vertex range",
            virtual_node);
    }
    return static_cast<Vertex>(value);
}

ConstraintLink checked_constraint_link(
    SolutionLink link,
    ControllerOperation operation,
    std::optional<ConstraintLink> virtual_link = std::nullopt)
{
    const Vertex source = checked_vertex(link.source, operation);
    const Vertex target = checked_vertex(link.target, operation);
    const ConstraintLink result{source, target};
    if (virtual_link.has_value())
    {
        (void)virtual_link;
    }
    return result;
}

bool is_hard(
    const std::vector<std::uint8_t>& mask,
    ConstraintId id) noexcept
{
    return id < mask.size() && mask[id] != std::uint8_t{0U};
}

SolutionAttributeValues pool_max_values(
    const std::vector<const SolutionAttributeValues*>& values,
    const SolutionAttributeValues& placeholder,
    ControllerOperation operation)
{
    if (values.empty())
    {
        return placeholder;
    }

    SolutionAttributeValues pooled;
    const auto& first_slots = values.front()->slots();
    for (std::size_t raw_id = 0U; raw_id < first_slots.size(); ++raw_id)
    {
        if (!first_slots[raw_id].has_value())
        {
            continue;
        }

        const ConstraintId id = static_cast<ConstraintId>(raw_id);
        network::attribute::AttributeNumber best = *first_slots[raw_id];
        double best_number = number_as_double(best);
        for (std::size_t index = 1U; index < values.size(); ++index)
        {
            const auto* candidate = values[index]->find(id);
            if (candidate == nullptr)
            {
                throw ControllerException(
                    ControllerErrorCode::missing_constraint_offset,
                    operation,
                    "a routed link is missing a pooled constraint offset",
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    id);
            }
            const double candidate_number = number_as_double(*candidate);
            if (candidate_number > best_number)
            {
                best = *candidate;
                best_number = candidate_number;
            }
        }
        pooled.set(id, std::move(best));
    }
    return pooled;
}

} // namespace

ControllerException::ControllerException(
    ControllerErrorCode code,
    ControllerOperation operation,
    std::string message,
    std::optional<Vertex> virtual_node,
    std::optional<Vertex> physical_node,
    std::optional<ConstraintLink> virtual_link,
    std::optional<ConstraintLink> physical_link,
    std::optional<ConstraintId> constraint_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      virtual_node_(virtual_node),
      physical_node_(physical_node),
      virtual_link_(virtual_link),
      physical_link_(physical_link),
      constraint_id_(constraint_id)
{
}

ControllerErrorCode ControllerException::code() const noexcept
{
    return code_;
}

ControllerOperation ControllerException::operation() const noexcept
{
    return operation_;
}

const std::optional<Vertex>& ControllerException::virtual_node() const noexcept
{
    return virtual_node_;
}

const std::optional<Vertex>& ControllerException::physical_node() const noexcept
{
    return physical_node_;
}

const std::optional<ConstraintLink>&
ControllerException::virtual_link() const noexcept
{
    return virtual_link_;
}

const std::optional<ConstraintLink>&
ControllerException::physical_link() const noexcept
{
    return physical_link_;
}

const std::optional<ConstraintId>&
ControllerException::constraint_id() const noexcept
{
    return constraint_id_;
}

Controller::Controller(ControllerSelection selection)
    : selection_(std::move(selection)),
      resource_updator_(ResourceUpdatorSelection{
          selection_.node_resources,
          selection_.link_resources}),
      node_mapper_(NodeMapperSelection{
          selection_.constraints.node_at_node,
          selection_.node_resources,
          selection_.hard_node_constraints}),
      link_mapper_(LinkMapperSelection{
          selection_.constraints.link_at_link,
          selection_.constraints.link_at_path,
          selection_.link_resources,
          selection_.hard_link_constraints,
          selection_.reusable})
{
}

const ControllerSelection& Controller::selection() const noexcept
{
    return selection_;
}

PreparedController Controller::prepare(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network) const
{
    auto prepared_updator =
        resource_updator_.prepare(virtual_network, physical_network);
    auto prepared_node_mapper =
        node_mapper_.prepare(virtual_network, physical_network);
    auto prepared_link_mapper =
        link_mapper_.prepare(virtual_network, physical_network);

    auto node_constraints =
        deduplicate_first(selection_.constraints.node_at_node);
    auto link_constraints =
        deduplicate_first(selection_.constraints.link_at_link);
    auto path_constraints =
        deduplicate_first(selection_.constraints.link_at_path);
    const auto all_link_constraints =
        combined_constraints(link_constraints, path_constraints);
    const auto node_resource_order =
        deduplicate_first(selection_.node_resources);
    const auto link_resource_order =
        deduplicate_first(selection_.link_resources);
    auto node_physical_value_ids = bind_node_resource_values(
        virtual_network, physical_network, node_resource_order);
    auto link_physical_value_ids = bind_link_resource_values(
        virtual_network, physical_network, link_resource_order);

    return PreparedController(
        virtual_network,
        physical_network,
        std::move(prepared_updator),
        std::move(prepared_node_mapper),
        std::move(prepared_link_mapper),
        node_resource_order,
        link_resource_order,
        std::move(node_constraints),
        std::move(link_constraints),
        std::move(path_constraints),
        make_hard_mask(
            selection_.constraints.node_at_node,
            selection_.hard_node_constraints),
        make_hard_mask(
            all_link_constraints,
            selection_.hard_link_constraints),
        std::move(node_physical_value_ids),
        std::move(link_physical_value_ids),
        zero_values(selection_.constraints.link_at_link),
        zero_values(selection_.constraints.link_at_path));
}

PreparedController::PreparedController(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    PreparedResourceUpdator resource_updator,
    PreparedNodeMapper node_mapper,
    PreparedLinkMapper link_mapper,
    std::vector<ResourceId> node_resource_order,
    std::vector<ResourceId> link_resource_order,
    std::vector<ConstraintId> node_constraint_order,
    std::vector<ConstraintId> link_constraint_order,
    std::vector<ConstraintId> path_constraint_order,
    std::vector<std::uint8_t> hard_node_mask,
    std::vector<std::uint8_t> hard_link_mask,
    std::vector<std::optional<AttrId>> node_physical_value_ids,
    std::vector<std::optional<AttrId>> link_physical_value_ids,
    SolutionAttributeValues link_step_placeholder,
    SolutionAttributeValues path_step_placeholder)
    : virtual_network_(&virtual_network),
      physical_network_(&physical_network),
      resource_updator_(std::move(resource_updator)),
      node_mapper_(std::move(node_mapper)),
      link_mapper_(std::move(link_mapper)),
      node_resource_order_(std::move(node_resource_order)),
      link_resource_order_(std::move(link_resource_order)),
      node_constraint_order_(std::move(node_constraint_order)),
      link_constraint_order_(std::move(link_constraint_order)),
      path_constraint_order_(std::move(path_constraint_order)),
      hard_node_mask_(std::move(hard_node_mask)),
      hard_link_mask_(std::move(hard_link_mask)),
      node_physical_value_ids_(std::move(node_physical_value_ids)),
      link_physical_value_ids_(std::move(link_physical_value_ids)),
      link_step_placeholder_(std::move(link_step_placeholder)),
      path_step_placeholder_(std::move(path_step_placeholder))
{
}

PlaceAndRouteResult PreparedController::place_and_route(
    Vertex virtual_node,
    Vertex physical_node,
    Solution& solution,
    PlaceAndRouteOptions options)
{
    if (options.allow_constraint_violation)
    {
        return unsafely_place_and_route(
            virtual_node,
            physical_node,
            solution,
            options);
    }
    return safely_place_and_route(
        virtual_node,
        physical_node,
        solution,
        options);
}

std::vector<ConstraintLink> PreparedController::links_to_route(
    Vertex virtual_node,
    const Solution& solution) const
{
    std::vector<ConstraintLink> result;
    const auto& adjacency =
        virtual_network_->graph().neighbors_fast(virtual_node);
    result.reserve(adjacency.size());
    for (const auto& edge : adjacency)
    {
        const Vertex neighbor = edge.get_target();
        const auto slot_id = solution.node_slots.find_id(
            static_cast<SolutionNodeId>(neighbor));
        if (!slot_id.has_value() ||
            solution.node_slots.at(*slot_id) == SolutionNodeId{-1})
        {
            continue;
        }

        const SolutionLink forward{
            static_cast<SolutionNodeId>(virtual_node),
            static_cast<SolutionNodeId>(neighbor)};
        const SolutionLink reverse{forward.target, forward.source};
        if (!solution.link_paths.contains(forward) &&
            !solution.link_paths.contains(reverse))
        {
            result.push_back(ConstraintLink{virtual_node, neighbor});
        }
    }
    return result;
}

PlaceAndRouteResult PreparedController::safely_place_and_route(
    Vertex virtual_node,
    Vertex physical_node,
    Solution& solution,
    const PlaceAndRouteOptions& options)
{
    PlaceAndRouteResult result;
    result.placement = node_mapper_.place(
        virtual_node,
        physical_node,
        solution,
        NodePlacementOptions{false, true});

    solution.v_net_single_step_constraint_offset.node_level =
        result.placement.check.offsets;
    solution.v_net_single_step_constraint_offset.link_level.clear();
    solution.v_net_single_step_constraint_offset.path_level.clear();

    if (!result.placement.placed)
    {
        solution.place_result = false;
        solution.result = false;
        result.failure_phase = ControllerFailurePhase::place;
        return result;
    }

    const auto to_route = links_to_route(virtual_node, solution);
    std::vector<ConstraintLink> attempted;
    attempted.reserve(to_route.size());
    for (const ConstraintLink virtual_link : to_route)
    {
        const auto neighbor_slot = solution.node_slots.find_id(
            static_cast<SolutionNodeId>(virtual_link.target));
        if (!neighbor_slot.has_value())
        {
            throw ControllerException(
                ControllerErrorCode::missing_node_slot,
                ControllerOperation::place_and_route,
                "placed neighbor has no stored physical node",
                virtual_link.target,
                std::nullopt,
                virtual_link);
        }
        const Vertex neighbor_physical = checked_vertex(
            solution.node_slots.at(*neighbor_slot),
            ControllerOperation::place_and_route,
            virtual_link.target);

        LinkRouteOptions route_options;
        route_options.shortest_method = options.shortest_method;
        route_options.k = options.k;
        route_options.max_path_nodes = options.max_path_nodes;
        route_options.topology_constraint_workers =
            options.workers.topology_constraint_workers;
        route_options.candidate_workers = options.workers.candidate_workers;
        route_options.allow_constraint_violation = false;
        route_options.record_constraint_violation = true;

        LinkRouteResult route = link_mapper_.route(
            virtual_link,
            ConstraintLink{physical_node, neighbor_physical},
            solution,
            route_options);
        attempted.push_back(virtual_link);
        result.last_route = route;
        result.attempted_routes = attempted.size();

        if (!route.routed)
        {
            pool_step_offsets(attempted, solution);
            calculate_max_single_step_constraint_violation(solution);
            solution.route_result = false;
            solution.result = false;
            result.failure_phase = ControllerFailurePhase::route;
            return result;
        }
    }

    pool_step_offsets(attempted, solution);
    calculate_max_single_step_constraint_violation(solution);
    result.succeeded = true;
    return result;
}

PlaceAndRouteResult PreparedController::unsafely_place_and_route(
    Vertex virtual_node,
    Vertex physical_node,
    Solution& solution,
    const PlaceAndRouteOptions& options)
{
    PlaceAndRouteResult result;
    result.placement = node_mapper_.place(
        virtual_node,
        physical_node,
        solution,
        NodePlacementOptions{true, true});

    solution.v_net_single_step_constraint_offset.node_level =
        result.placement.check.offsets;
    solution.v_net_single_step_constraint_offset.link_level.clear();
    solution.v_net_single_step_constraint_offset.path_level.clear();

    const auto to_route = links_to_route(virtual_node, solution);
    std::vector<ConstraintLink> attempted;
    attempted.reserve(to_route.size());
    for (const ConstraintLink virtual_link : to_route)
    {
        const auto neighbor_slot = solution.node_slots.find_id(
            static_cast<SolutionNodeId>(virtual_link.target));
        if (!neighbor_slot.has_value())
        {
            throw ControllerException(
                ControllerErrorCode::missing_node_slot,
                ControllerOperation::place_and_route,
                "placed neighbor has no stored physical node",
                virtual_link.target,
                std::nullopt,
                virtual_link);
        }
        const Vertex neighbor_physical = checked_vertex(
            solution.node_slots.at(*neighbor_slot),
            ControllerOperation::place_and_route,
            virtual_link.target);

        LinkRouteOptions route_options;
        route_options.shortest_method = options.shortest_method;
        route_options.k = options.k;
        route_options.max_path_nodes = options.max_path_nodes;
        route_options.topology_constraint_workers =
            options.workers.topology_constraint_workers;
        route_options.candidate_workers = options.workers.candidate_workers;
        route_options.allow_constraint_violation = true;
        route_options.record_constraint_violation = true;

        result.last_route = link_mapper_.route(
            virtual_link,
            ConstraintLink{physical_node, neighbor_physical},
            solution,
            route_options);
        attempted.push_back(virtual_link);
        result.attempted_routes = attempted.size();
    }

    pool_step_offsets(attempted, solution);
    calculate_max_single_step_constraint_violation(solution);
    result.succeeded = true;
    return result;
}

void PreparedController::pool_step_offsets(
    const std::vector<ConstraintLink>& attempted_links,
    Solution& solution) const
{
    std::vector<const SolutionAttributeValues*> link_values;
    std::vector<const SolutionAttributeValues*> path_values;
    link_values.reserve(attempted_links.size());
    path_values.reserve(attempted_links.size());

    for (const ConstraintLink link : attempted_links)
    {
        const SolutionLink key = solution_link(link);
        const auto link_id =
            solution.v_net_constraint_offsets.link_level.find_id(key);
        const auto path_id =
            solution.v_net_constraint_offsets.path_level.find_id(key);
        if (!link_id.has_value() || !path_id.has_value())
        {
            throw ControllerException(
                ControllerErrorCode::missing_constraint_offset,
                ControllerOperation::pool_step_offsets,
                "routed virtual link has no stored constraint offsets",
                std::nullopt,
                std::nullopt,
                link);
        }
        link_values.push_back(
            &solution.v_net_constraint_offsets.link_level.at(*link_id));
        path_values.push_back(
            &solution.v_net_constraint_offsets.path_level.at(*path_id));
    }

    SolutionAttributeValues pooled_link = pool_max_values(
        link_values,
        link_step_placeholder_,
        ControllerOperation::pool_step_offsets);
    SolutionAttributeValues pooled_path = pool_max_values(
        path_values,
        path_step_placeholder_,
        ControllerOperation::pool_step_offsets);

    solution.v_net_single_step_constraint_offset.link_level =
        std::move(pooled_link);
    solution.v_net_single_step_constraint_offset.path_level =
        std::move(pooled_path);
}

void PreparedController::calculate_max_single_step_constraint_violation(
    Solution& solution) const
{
    bool has_value = false;
    double maximum = 0.0;

    const auto consider =
        [&has_value, &maximum](
            const SolutionAttributeValues& values,
            const std::vector<ConstraintId>& order,
            const std::vector<std::uint8_t>& hard_mask)
        {
            for (const ConstraintId id : order)
            {
                if (!is_hard(hard_mask, id))
                {
                    continue;
                }
                const auto* value = values.find(id);
                if (value == nullptr)
                {
                    continue;
                }
                const double numeric = number_as_double(*value);
                if (!has_value)
                {
                    maximum = numeric;
                    has_value = true;
                }
                else if (numeric > maximum)
                {
                    maximum = numeric;
                }
            }
        };

    consider(
        solution.v_net_single_step_constraint_offset.node_level,
        node_constraint_order_,
        hard_node_mask_);
    consider(
        solution.v_net_single_step_constraint_offset.link_level,
        link_constraint_order_,
        hard_link_mask_);
    consider(
        solution.v_net_single_step_constraint_offset.path_level,
        path_constraint_order_,
        hard_link_mask_);

    solution.v_net_single_step_hard_constraint_offset = maximum;
    solution.v_net_max_single_step_hard_constraint_violation = std::max(
        solution.v_net_max_single_step_hard_constraint_violation,
        maximum);
}

bool PreparedController::undo_place_and_route(
    Vertex virtual_node,
    Solution& solution)
{
    if (!solution.node_slots.contains(
            static_cast<SolutionNodeId>(virtual_node)))
    {
        throw ControllerException(
            ControllerErrorCode::missing_node_slot,
            ControllerOperation::undo_place_and_route,
            "virtual node has not been placed",
            virtual_node);
    }

    (void)node_mapper_.undo_place(virtual_node, solution);

    std::vector<SolutionLink> original_links;
    original_links.reserve(solution.link_paths.size());
    for (const auto& entry : solution.link_paths.entries())
    {
        original_links.push_back(entry.key);
    }
    const SolutionNodeId selected = static_cast<SolutionNodeId>(virtual_node);
    for (const SolutionLink link : original_links)
    {
        if (link.source == selected || link.target == selected)
        {
            (void)link_mapper_.undo_route(
                checked_constraint_link(
                    link,
                    ControllerOperation::undo_place_and_route),
                solution);
        }
    }
    return true;
}

std::vector<ResourceAmount> PreparedController::gather_resources(
    const SolutionAttributeValues& values,
    const std::vector<ResourceId>& resource_order) const
{
    std::vector<ResourceAmount> resources;
    resources.reserve(resource_order.size());
    for (const ResourceId id : resource_order)
    {
        const auto* value = values.find(id);
        if (value != nullptr)
        {
            resources.push_back(ResourceAmount{id, *value});
        }
    }
    return resources;
}

void PreparedController::apply_node_values(
    Vertex physical_node,
    const SolutionAttributeValues& values,
    network::attribute::ResourceUpdateOperation operation)
{
    for (const ResourceId id : node_resource_order_)
    {
        const auto* value = values.find(id);
        if (value != nullptr)
        {
            resource_updator_.update_node_resource(
                physical_node,
                ResourceAmount{id, *value},
                operation,
                true);
        }
    }
}

void PreparedController::apply_link_values(
    ConstraintLink physical_link,
    const SolutionAttributeValues& values,
    network::attribute::ResourceUpdateOperation operation)
{
    for (const ResourceId id : link_resource_order_)
    {
        const auto* value = values.find(id);
        if (value != nullptr)
        {
            resource_updator_.update_link_resource(
                physical_link,
                ResourceAmount{id, *value},
                operation,
                true);
        }
    }
}

bool PreparedController::node_targets_are_disjoint(
    const std::vector<NodeResourceUpdateRequest>& requests) const
{
    std::vector<std::uint8_t> seen(
        physical_network_->graph().num_nodes(),
        std::uint8_t{0U});
    for (const auto& request : requests)
    {
        if (request.physical_node >= seen.size() ||
            seen[request.physical_node] != std::uint8_t{0U})
        {
            return false;
        }
        seen[request.physical_node] = std::uint8_t{1U};
    }
    return true;
}

bool PreparedController::link_targets_are_disjoint(
    const std::vector<LinkResourceUpdateRequest>& requests) const
{
    const auto& graph = physical_network_->graph();
    std::vector<std::uint8_t> seen(
        graph.edge_id_capacity(),
        std::uint8_t{0U});
    for (const auto& request : requests)
    {
        if (!graph.has_edge(
                request.physical_link.source,
                request.physical_link.target))
        {
            return false;
        }
        const auto edge = graph.edge(
            request.physical_link.source,
            request.physical_link.target);
        const std::size_t id = graph.edge_id(edge);
        if (id >= seen.size() || seen[id] != std::uint8_t{0U})
        {
            return false;
        }
        seen[id] = std::uint8_t{1U};
    }
    return true;
}

void PreparedController::apply_node_requests(
    const std::vector<NodeResourceUpdateRequest>& requests,
    network::attribute::ResourceUpdateOperation operation,
    std::size_t workers)
{
    const auto scalar = [this, &requests, operation]()
    {
        for (const auto& request : requests)
        {
            resource_updator_.update_node_resources(
                request.physical_node,
                request.resources,
                operation,
                true);
        }
    };

    if (workers <= 1U || requests.size() <= 1U ||
        !node_targets_are_disjoint(requests))
    {
        scalar();
        return;
    }

    try
    {
        resource_updator_.update_node_resources_batch(
            requests,
            operation,
            true,
            workers);
    }
    catch (const ResourceUpdatorException&)
    {
        const std::exception_ptr batch_error = std::current_exception();
        scalar();
        std::rethrow_exception(batch_error);
    }
}

void PreparedController::apply_link_requests(
    const std::vector<LinkResourceUpdateRequest>& requests,
    network::attribute::ResourceUpdateOperation operation,
    std::size_t workers)
{
    const auto scalar = [this, &requests, operation]()
    {
        for (const auto& request : requests)
        {
            resource_updator_.update_link_resources(
                request.physical_link,
                request.resources,
                operation,
                true);
        }
    };

    if (workers <= 1U || requests.size() <= 1U ||
        !link_targets_are_disjoint(requests))
    {
        scalar();
        return;
    }

    try
    {
        resource_updator_.update_link_resources_batch(
            requests,
            operation,
            true,
            workers);
    }
    catch (const ResourceUpdatorException&)
    {
        const std::exception_ptr batch_error = std::current_exception();
        scalar();
        std::rethrow_exception(batch_error);
    }
}

bool PreparedController::try_deploy_parallel(
    const Solution& solution,
    std::size_t workers)
{
    Graph& graph = physical_network_->graph();
    std::vector<std::uint8_t> seen_nodes(
        graph.num_nodes(),
        std::uint8_t{0U});
    std::vector<std::uint8_t> seen_edges(
        graph.edge_id_capacity(),
        std::uint8_t{0U});
    std::vector<DirectMutation> node_mutations;
    std::vector<DirectMutation> link_mutations;
    node_mutations.reserve(
        solution.node_slots_info.size() * node_resource_order_.size());
    link_mutations.reserve(
        solution.link_paths_info.size() * link_resource_order_.size());

    try
    {
        for (const auto& entry : solution.node_slots_info.entries())
        {
            const Vertex physical_node = checked_vertex(
                entry.key.physical_node,
                ControllerOperation::deploy);
            if (physical_node >= seen_nodes.size() ||
                seen_nodes[physical_node] != std::uint8_t{0U})
            {
                return false;
            }
            seen_nodes[physical_node] = std::uint8_t{1U};
            AttrMap& physical_values = graph.node_attrs(physical_node);
            for (const ResourceId id : node_resource_order_)
            {
                const auto* requested = entry.value.find(id);
                if (requested == nullptr)
                {
                    continue;
                }
                const auto& value_id = node_physical_value_ids_.at(id);
                if (!value_id.has_value())
                {
                    return false;
                }
                AttrValue* current = physical_values.find(*value_id);
                if (current == nullptr)
                {
                    return false;
                }
                DirectMutation mutation;
                mutation.target = current;
                if (!plan_numeric_mutation(
                        *current,
                        *requested,
                        network::attribute::ResourceUpdateOperation::subtract,
                        mutation.value))
                {
                    return false;
                }
                node_mutations.push_back(std::move(mutation));
            }
        }

        for (const auto& entry : solution.link_paths_info.entries())
        {
            const ConstraintLink physical_link = checked_constraint_link(
                entry.key.physical_link,
                ControllerOperation::deploy);
            if (!graph.has_edge(
                    physical_link.source,
                    physical_link.target))
            {
                return false;
            }
            const auto edge = graph.edge(
                physical_link.source,
                physical_link.target);
            const std::size_t edge_id = graph.edge_id(edge);
            if (edge_id >= seen_edges.size() ||
                seen_edges[edge_id] != std::uint8_t{0U})
            {
                return false;
            }
            seen_edges[edge_id] = std::uint8_t{1U};
            AttrMap& physical_values = graph.edge_attrs(edge);
            for (const ResourceId id : link_resource_order_)
            {
                const auto* requested = entry.value.find(id);
                if (requested == nullptr)
                {
                    continue;
                }
                const auto& value_id = link_physical_value_ids_.at(id);
                if (!value_id.has_value())
                {
                    return false;
                }
                AttrValue* current = physical_values.find(*value_id);
                if (current == nullptr)
                {
                    return false;
                }
                DirectMutation mutation;
                mutation.target = current;
                if (!plan_numeric_mutation(
                        *current,
                        *requested,
                        network::attribute::ResourceUpdateOperation::subtract,
                        mutation.value))
                {
                    return false;
                }
                link_mutations.push_back(std::move(mutation));
            }
        }
    }
    catch (const ControllerException&)
    {
        return false;
    }

    commit_direct_mutations(node_mutations, workers);
    commit_direct_mutations(link_mutations, workers);
    return true;
}

bool PreparedController::try_release_parallel(
    const Solution& solution,
    std::size_t workers)
{
    Graph& graph = physical_network_->graph();
    std::vector<std::uint8_t> seen_nodes(
        graph.num_nodes(),
        std::uint8_t{0U});
    std::vector<std::uint8_t> seen_edges(
        graph.edge_id_capacity(),
        std::uint8_t{0U});
    std::vector<DirectMutation> node_mutations;
    std::vector<DirectMutation> link_mutations;
    node_mutations.reserve(
        solution.node_slots.size() * node_resource_order_.size());
    link_mutations.reserve(
        solution.link_paths_info.size() * link_resource_order_.size());

    try
    {
        for (const auto& entry : solution.node_slots.entries())
        {
            const NodeSlotInfoKey info_key{entry.key, entry.value};
            const auto info_id = solution.node_slots_info.find_id(info_key);
            if (!info_id.has_value())
            {
                return false;
            }
            const Vertex physical_node = checked_vertex(
                entry.value,
                ControllerOperation::release);
            if (physical_node >= seen_nodes.size() ||
                seen_nodes[physical_node] != std::uint8_t{0U})
            {
                return false;
            }
            seen_nodes[physical_node] = std::uint8_t{1U};
            AttrMap& physical_values = graph.node_attrs(physical_node);
            const auto& recorded = solution.node_slots_info.at(*info_id);
            for (const ResourceId id : node_resource_order_)
            {
                const auto* requested = recorded.find(id);
                if (requested == nullptr)
                {
                    continue;
                }
                const auto& value_id = node_physical_value_ids_.at(id);
                if (!value_id.has_value())
                {
                    return false;
                }
                AttrValue* current = physical_values.find(*value_id);
                if (current == nullptr)
                {
                    return false;
                }
                DirectMutation mutation;
                mutation.target = current;
                if (!plan_numeric_mutation(
                        *current,
                        *requested,
                        network::attribute::ResourceUpdateOperation::add,
                        mutation.value))
                {
                    return false;
                }
                node_mutations.push_back(std::move(mutation));
            }
        }

        for (const auto& path_entry : solution.link_paths.entries())
        {
            for (const SolutionLink physical_link_value : path_entry.value)
            {
                const LinkPathInfoKey info_key{
                    path_entry.key,
                    physical_link_value};
                const auto info_id =
                    solution.link_paths_info.find_id(info_key);
                if (!info_id.has_value())
                {
                    return false;
                }
                const ConstraintLink physical_link = checked_constraint_link(
                    physical_link_value,
                    ControllerOperation::release);
                if (!graph.has_edge(
                        physical_link.source,
                        physical_link.target))
                {
                    return false;
                }
                const auto edge = graph.edge(
                    physical_link.source,
                    physical_link.target);
                const std::size_t edge_id = graph.edge_id(edge);
                if (edge_id >= seen_edges.size() ||
                    seen_edges[edge_id] != std::uint8_t{0U})
                {
                    return false;
                }
                seen_edges[edge_id] = std::uint8_t{1U};
                AttrMap& physical_values = graph.edge_attrs(edge);
                const auto& recorded = solution.link_paths_info.at(*info_id);
                for (const ResourceId id : link_resource_order_)
                {
                    const auto* requested = recorded.find(id);
                    if (requested == nullptr)
                    {
                        continue;
                    }
                    const auto& value_id = link_physical_value_ids_.at(id);
                    if (!value_id.has_value())
                    {
                        return false;
                    }
                    AttrValue* current = physical_values.find(*value_id);
                    if (current == nullptr)
                    {
                        return false;
                    }
                    DirectMutation mutation;
                    mutation.target = current;
                    if (!plan_numeric_mutation(
                            *current,
                            *requested,
                            network::attribute::ResourceUpdateOperation::add,
                            mutation.value))
                    {
                        return false;
                    }
                    link_mutations.push_back(std::move(mutation));
                }
            }
        }
    }
    catch (const ControllerException&)
    {
        return false;
    }

    commit_direct_mutations(node_mutations, workers);
    commit_direct_mutations(link_mutations, workers);
    return true;
}

bool PreparedController::deploy(
    Solution& solution,
    ControllerMutationOptions options)
{
    if (!solution.result)
    {
        return false;
    }

    if (options.workers <= 1U)
    {
        for (const auto& entry : solution.node_slots_info.entries())
        {
            apply_node_values(
                checked_vertex(
                    entry.key.physical_node,
                    ControllerOperation::deploy),
                entry.value,
                network::attribute::ResourceUpdateOperation::subtract);
        }
        for (const auto& entry : solution.link_paths_info.entries())
        {
            apply_link_values(
                checked_constraint_link(
                    entry.key.physical_link,
                    ControllerOperation::deploy),
                entry.value,
                network::attribute::ResourceUpdateOperation::subtract);
        }
        return true;
    }
    if (try_deploy_parallel(solution, options.workers))
    {
        return true;
    }

    std::vector<NodeResourceUpdateRequest> node_requests;
    node_requests.reserve(solution.node_slots_info.size());
    for (const auto& entry : solution.node_slots_info.entries())
    {
        try
        {
            node_requests.push_back(NodeResourceUpdateRequest{
                checked_vertex(
                    entry.key.physical_node,
                    ControllerOperation::deploy),
                gather_resources(entry.value, node_resource_order_)});
        }
        catch (...)
        {
            apply_node_requests(
                node_requests,
                network::attribute::ResourceUpdateOperation::subtract,
                1U);
            throw;
        }
    }
    apply_node_requests(
        node_requests,
        network::attribute::ResourceUpdateOperation::subtract,
        options.workers);

    std::vector<LinkResourceUpdateRequest> link_requests;
    link_requests.reserve(solution.link_paths_info.size());
    for (const auto& entry : solution.link_paths_info.entries())
    {
        try
        {
            link_requests.push_back(LinkResourceUpdateRequest{
                checked_constraint_link(
                    entry.key.physical_link,
                    ControllerOperation::deploy),
                gather_resources(entry.value, link_resource_order_)});
        }
        catch (...)
        {
            apply_link_requests(
                link_requests,
                network::attribute::ResourceUpdateOperation::subtract,
                1U);
            throw;
        }
    }
    apply_link_requests(
        link_requests,
        network::attribute::ResourceUpdateOperation::subtract,
        options.workers);
    return true;
}

bool PreparedController::release(
    const Solution& solution,
    ControllerMutationOptions options)
{
    if (!solution.result)
    {
        return false;
    }

    if (options.workers <= 1U)
    {
        for (const auto& entry : solution.node_slots.entries())
        {
            const NodeSlotInfoKey info_key{entry.key, entry.value};
            const auto info_id = solution.node_slots_info.find_id(info_key);
            if (!info_id.has_value())
            {
                throw ControllerException(
                    ControllerErrorCode::missing_node_slot_info,
                    ControllerOperation::release,
                    "placed node has no stored resource information",
                    checked_vertex(
                        entry.key,
                        ControllerOperation::release),
                    checked_vertex(
                        entry.value,
                        ControllerOperation::release));
            }
            apply_node_values(
                checked_vertex(
                    entry.value,
                    ControllerOperation::release),
                solution.node_slots_info.at(*info_id),
                network::attribute::ResourceUpdateOperation::add);
        }
        for (const auto& path_entry : solution.link_paths.entries())
        {
            for (const SolutionLink physical_link : path_entry.value)
            {
                const LinkPathInfoKey info_key{
                    path_entry.key,
                    physical_link};
                const auto info_id =
                    solution.link_paths_info.find_id(info_key);
                if (!info_id.has_value())
                {
                    throw ControllerException(
                        ControllerErrorCode::missing_link_path_info,
                        ControllerOperation::release,
                        "routed physical link has no stored resource information",
                        std::nullopt,
                        std::nullopt,
                        checked_constraint_link(
                            path_entry.key,
                            ControllerOperation::release),
                        checked_constraint_link(
                            physical_link,
                            ControllerOperation::release));
                }
                apply_link_values(
                    checked_constraint_link(
                        physical_link,
                        ControllerOperation::release),
                    solution.link_paths_info.at(*info_id),
                    network::attribute::ResourceUpdateOperation::add);
            }
        }
        return true;
    }
    if (try_release_parallel(solution, options.workers))
    {
        return true;
    }

    std::vector<NodeResourceUpdateRequest> node_requests;
    node_requests.reserve(solution.node_slots.size());
    for (const auto& entry : solution.node_slots.entries())
    {
        try
        {
            const NodeSlotInfoKey info_key{entry.key, entry.value};
            const auto info_id = solution.node_slots_info.find_id(info_key);
            if (!info_id.has_value())
            {
                throw ControllerException(
                    ControllerErrorCode::missing_node_slot_info,
                    ControllerOperation::release,
                    "placed node has no stored resource information",
                    checked_vertex(
                        entry.key,
                        ControllerOperation::release),
                    checked_vertex(
                        entry.value,
                        ControllerOperation::release));
            }
            node_requests.push_back(NodeResourceUpdateRequest{
                checked_vertex(
                    entry.value,
                    ControllerOperation::release),
                gather_resources(
                    solution.node_slots_info.at(*info_id),
                    node_resource_order_)});
        }
        catch (...)
        {
            apply_node_requests(
                node_requests,
                network::attribute::ResourceUpdateOperation::add,
                1U);
            throw;
        }
    }
    apply_node_requests(
        node_requests,
        network::attribute::ResourceUpdateOperation::add,
        options.workers);

    std::vector<LinkResourceUpdateRequest> link_requests;
    for (const auto& path_entry : solution.link_paths.entries())
    {
        link_requests.reserve(
            link_requests.size() + path_entry.value.size());
        for (const SolutionLink physical_link : path_entry.value)
        {
            try
            {
                const LinkPathInfoKey info_key{
                    path_entry.key,
                    physical_link};
                const auto info_id =
                    solution.link_paths_info.find_id(info_key);
                if (!info_id.has_value())
                {
                    throw ControllerException(
                        ControllerErrorCode::missing_link_path_info,
                        ControllerOperation::release,
                        "routed physical link has no stored resource information",
                        std::nullopt,
                        std::nullopt,
                        checked_constraint_link(
                            path_entry.key,
                            ControllerOperation::release),
                        checked_constraint_link(
                            physical_link,
                            ControllerOperation::release));
                }
                link_requests.push_back(LinkResourceUpdateRequest{
                    checked_constraint_link(
                        physical_link,
                        ControllerOperation::release),
                    gather_resources(
                        solution.link_paths_info.at(*info_id),
                        link_resource_order_)});
            }
            catch (...)
            {
                apply_link_requests(
                    link_requests,
                    network::attribute::ResourceUpdateOperation::add,
                    1U);
                throw;
            }
        }
    }
    apply_link_requests(
        link_requests,
        network::attribute::ResourceUpdateOperation::add,
        options.workers);
    return true;
}

bool PreparedController::undo_deploy(
    const Solution& solution,
    ControllerMutationOptions options)
{
    (void)release(solution, options);
    (void)Solution::from_v_net(*virtual_network_);
    return true;
}

} // namespace virne::core::controller
