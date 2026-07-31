#include "link_mapper.h"

#include "../../utils/deterministic_executor.h"
#include "../../utils/network.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace virne::core::controller
{
namespace
{

using AttributeNumber = network::attribute::AttributeNumber;
using ResourceUpdateOperation = network::attribute::ResourceUpdateOperation;

constexpr std::size_t minimum_parallel_path_count = 64U;
constexpr std::size_t maximum_sequential_probe_count = 8U;
constexpr std::size_t minimum_path_window = 32U;
constexpr std::size_t path_window_per_worker = 128U;

SolutionNodeId solution_node_id(Vertex node) noexcept
{
    return static_cast<SolutionNodeId>(node);
}

SolutionLink solution_link(ConstraintLink link) noexcept
{
    return SolutionLink{
        solution_node_id(link.source),
        solution_node_id(link.target)};
}

ConstraintLink constraint_link(
    const SolutionLink& link,
    LinkMapperOperation operation,
    ConstraintLink virtual_link)
{
    const auto valid = [](SolutionNodeId node) noexcept
    {
        return node >= 0 &&
            static_cast<std::uint64_t>(node) <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<Vertex>::max());
    };
    if (!valid(link.source) || !valid(link.target))
    {
        throw LinkMapperException(
            LinkMapperErrorCode::invalid_stored_node,
            operation,
            "stored physical link is outside the native vertex range",
            virtual_link);
    }
    return ConstraintLink{
        static_cast<Vertex>(link.source),
        static_cast<Vertex>(link.target)};
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

double number_as_double(const AttributeNumber& value)
{
    return std::visit(
        [](const auto& number) -> double
        {
            return static_cast<double>(number);
        },
        value);
}

bool python_max_replaces(
    const AttributeNumber& current,
    const AttributeNumber& candidate)
{
    return number_as_long_double(candidate) > number_as_long_double(current);
}

bool python_min_replaces(long double current, long double candidate)
{
    return candidate < current;
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
                return number < 0.0 ? AttributeNumber{std::int64_t{0}}
                                    : AttributeNumber{number};
            }
        },
        value);
}

AttributeNumber add_non_negative(
    const AttributeNumber& left,
    const AttributeNumber& right)
{
    if (std::holds_alternative<double>(left) ||
        std::holds_alternative<double>(right))
    {
        return number_as_double(left) + number_as_double(right);
    }
    const auto integer_value = [](const AttributeNumber& value)
    {
        if (const auto* boolean = std::get_if<bool>(&value))
        {
            return *boolean ? std::int64_t{1} : std::int64_t{0};
        }
        return std::get<std::int64_t>(value);
    };
    const std::int64_t first = integer_value(left);
    const std::int64_t second = integer_value(right);
    if (second > 0 &&
        first > std::numeric_limits<std::int64_t>::max() - second)
    {
        throw std::overflow_error("link constraint violation sum overflow");
    }
    return first + second;
}

AttributeNumber maximum_value(
    const std::vector<const AttributeNumber*>& values)
{
    AttributeNumber result = *values.front();
    for (std::size_t index = 1U; index < values.size(); ++index)
    {
        if (python_max_replaces(result, *values[index]))
        {
            result = *values[index];
        }
    }
    return result;
}

AttributeNumber sum_positive(
    const std::vector<const AttributeNumber*>& values)
{
    AttributeNumber result = std::int64_t{0};
    for (const AttributeNumber* value : values)
    {
        if (number_as_long_double(*value) > 0.0L)
        {
            result = add_non_negative(result, *value);
        }
    }
    return result;
}

bool contains_constraint_id(
    const std::vector<ConstraintId>& values,
    ConstraintId id)
{
    return std::find(values.begin(), values.end(), id) != values.end();
}

} // namespace

LinkMapperException::LinkMapperException(
    LinkMapperErrorCode code,
    LinkMapperOperation operation,
    std::string message,
    std::optional<ConstraintLink> virtual_link,
    std::optional<ConstraintLink> physical_link,
    std::optional<ResourceId> resource_id,
    std::optional<ConstraintId> constraint_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      virtual_link_(virtual_link),
      physical_link_(physical_link),
      resource_id_(resource_id),
      constraint_id_(constraint_id)
{
}

LinkMapperErrorCode LinkMapperException::code() const noexcept
{
    return code_;
}

LinkMapperOperation LinkMapperException::operation() const noexcept
{
    return operation_;
}

const std::optional<ConstraintLink>&
LinkMapperException::virtual_link() const noexcept
{
    return virtual_link_;
}

const std::optional<ConstraintLink>&
LinkMapperException::physical_link() const noexcept
{
    return physical_link_;
}

const std::optional<ResourceId>& LinkMapperException::resource_id() const noexcept
{
    return resource_id_;
}

const std::optional<ConstraintId>&
LinkMapperException::constraint_id() const noexcept
{
    return constraint_id_;
}

LinkMapper::LinkMapper(LinkMapperSelection selection)
    : selection_(std::move(selection))
{
}

const LinkMapperSelection& LinkMapper::selection() const noexcept
{
    return selection_;
}

PreparedLinkMapper LinkMapper::prepare(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network) const
{
    const auto& registry = virtual_network.link_attributes();
    const auto& entries = registry.entries();
    std::vector<PreparedLinkMapper::PreparedLinkResource> resources;
    resources.reserve(selection_.link_resources.size());
    std::vector<ResourceId> effective_resource_ids;
    effective_resource_ids.reserve(selection_.link_resources.size());

    for (const ResourceId id : selection_.link_resources)
    {
        if (static_cast<std::size_t>(id) >= entries.size())
        {
            throw LinkMapperException(
                LinkMapperErrorCode::invalid_link_resource_selection,
                LinkMapperOperation::prepare,
                "link resource selection is outside the virtual registry",
                std::nullopt,
                std::nullopt,
                id);
        }
        const auto& entry = entries[static_cast<std::size_t>(id)];
        if (entry.attribute == nullptr ||
            entry.attribute->spec().kind !=
                network::attribute::AttributeKind::resource)
        {
            throw LinkMapperException(
                LinkMapperErrorCode::invalid_link_resource_selection,
                LinkMapperOperation::prepare,
                "selected link attribute is not a resource",
                std::nullopt,
                std::nullopt,
                id);
        }
        const bool duplicate = std::find_if(
            resources.begin(),
            resources.end(),
            [id](const PreparedLinkMapper::PreparedLinkResource& value)
            {
                return value.resource_id == id;
            }) != resources.end();
        if (duplicate)
        {
            continue;
        }
        const auto binding = virtual_network.bind_link_attribute(
            entry.name);
        if (!binding.has_value())
        {
            throw LinkMapperException(
                LinkMapperErrorCode::invalid_link_resource_selection,
                LinkMapperOperation::prepare,
                "link resource has no virtual graph binding",
                std::nullopt,
                std::nullopt,
                id);
        }
        resources.push_back(
            PreparedLinkMapper::PreparedLinkResource{id, binding->value_id});
        effective_resource_ids.push_back(id);
    }

    ConstraintCheckerSelection checker_selection;
    checker_selection.link_at_link = selection_.link_constraints;
    checker_selection.link_at_path = selection_.path_constraints;
    ConstraintChecker checker(checker_selection);
    PreparedConstraintChecker prepared_checker =
        checker.prepare(virtual_network, physical_network);

    ResourceUpdatorSelection updator_selection;
    updator_selection.link_resources = std::move(effective_resource_ids);
    ResourceUpdator updator(std::move(updator_selection));
    PreparedResourceUpdator prepared_updator =
        updator.prepare(virtual_network, physical_network);

    TopologyAnalyzerSelection topology_selection;
    topology_selection.constraints = checker_selection;
    topology_selection.link_resources = selection_.link_resources;
    TopologyAnalyzer analyzer(std::move(topology_selection));
    PreparedTopologyAnalyzer prepared_analyzer =
        analyzer.prepare(virtual_network, physical_network);

    std::vector<ConstraintId> link_order;
    link_order.reserve(selection_.link_constraints.size());
    for (const ConstraintId id : selection_.link_constraints)
    {
        if (!contains_constraint_id(link_order, id))
        {
            link_order.push_back(id);
        }
    }
    std::vector<ConstraintId> path_order;
    path_order.reserve(selection_.path_constraints.size());
    for (const ConstraintId id : selection_.path_constraints)
    {
        if (!contains_constraint_id(path_order, id))
        {
            path_order.push_back(id);
        }
    }
    std::vector<ConstraintId> combined_order = link_order;
    for (const ConstraintId id : path_order)
    {
        if (!contains_constraint_id(combined_order, id))
        {
            combined_order.push_back(id);
        }
    }

    std::vector<std::uint8_t> hard_mask(
        registry.size(), std::uint8_t{0});
    for (const ConstraintId id : selection_.hard_constraints)
    {
        if (static_cast<std::size_t>(id) < hard_mask.size())
        {
            hard_mask[static_cast<std::size_t>(id)] = std::uint8_t{1};
        }
    }

    return PreparedLinkMapper(
        selection_,
        virtual_network,
        physical_network,
        std::move(prepared_checker),
        std::move(prepared_updator),
        std::move(prepared_analyzer),
        std::move(resources),
        std::move(link_order),
        std::move(path_order),
        std::move(combined_order),
        std::move(hard_mask));
}

PreparedLinkMapper::PreparedLinkMapper(
    LinkMapperSelection selection,
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    PreparedConstraintChecker constraint_checker,
    PreparedResourceUpdator resource_updator,
    PreparedTopologyAnalyzer topology_analyzer,
    std::vector<PreparedLinkResource> link_resources,
    std::vector<ConstraintId> link_constraint_order,
    std::vector<ConstraintId> path_constraint_order,
    std::vector<ConstraintId> combined_constraint_order,
    std::vector<std::uint8_t> hard_constraint_mask)
    : selection_(std::move(selection)),
      virtual_network_(&virtual_network),
      physical_network_(&physical_network),
      constraint_checker_(std::move(constraint_checker)),
      resource_updator_(std::move(resource_updator)),
      topology_analyzer_(std::move(topology_analyzer)),
      link_resources_(std::move(link_resources)),
      link_constraint_order_(std::move(link_constraint_order)),
      path_constraint_order_(std::move(path_constraint_order)),
      combined_constraint_order_(std::move(combined_constraint_order)),
      hard_constraint_mask_(std::move(hard_constraint_mask))
{
}

LinkRouteCheckInfo PreparedLinkMapper::placeholder_check() const
{
    LinkRouteCheckInfo result;
    result.placeholder = true;
    for (const ConstraintId id : path_constraint_order_)
    {
        result.constraints.path_level.set(id, 0.0);
    }
    return result;
}

void PreparedLinkMapper::clear_existing_route(
    ConstraintLink virtual_link,
    Solution& solution) const
{
    const SolutionLink key = solution_link(virtual_link);
    const auto existing = solution.link_paths.find_id(key);
    if (existing.has_value())
    {
        const std::vector<SolutionLink>& old_path =
            solution.link_paths.at(*existing);
        for (const SolutionLink& physical_link : old_path)
        {
            solution.link_paths_info.erase(
                LinkPathInfoKey{key, physical_link});
        }
    }
    solution.link_paths.insert_or_assign(key, {});
}

const std::vector<ResourceAmount>& PreparedLinkMapper::gather_link_resources(
    ConstraintLink virtual_link,
    SolutionAttributeValues* recorded_values)
{
    const auto edge = virtual_network_->graph().edge(
        virtual_link.source, virtual_link.target);
    const AttrMap& values = virtual_network_->graph().edge_attrs(edge);
    auto& result = resource_scratch_;
    result.clear();
    result.reserve(link_resources_.size());
    for (const PreparedLinkResource& prepared : link_resources_)
    {
        const AttrValue* value = values.find(prepared.virtual_value_id);
        if (value == nullptr)
        {
            throw LinkMapperException(
                LinkMapperErrorCode::missing_link_resource_value,
                LinkMapperOperation::route,
                "virtual link resource value is missing",
                virtual_link,
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
            throw LinkMapperException(
                LinkMapperErrorCode::non_numeric_link_resource,
                LinkMapperOperation::route,
                "virtual link resource value is not numeric",
                virtual_link,
                std::nullopt,
                prepared.resource_id);
        }
        result.push_back(ResourceAmount{prepared.resource_id, number});
        if (recorded_values != nullptr)
        {
            recorded_values->set(prepared.resource_id, std::move(number));
        }
    }
    return result;
}

LinkRouteResult PreparedLinkMapper::commit_path(
    ConstraintLink virtual_link,
    const PhysicalPath& path,
    LinkRouteCheckInfo check,
    Solution& solution,
    bool safe,
    std::size_t path_workers)
{
    const virne::utils::PathLinks links =
        virne::utils::path_to_links(path, path_workers);
    std::vector<SolutionLink> stored_links;
    stored_links.reserve(links.size());
    for (const auto& link : links)
    {
        stored_links.push_back(SolutionLink{
            solution_node_id(link.first),
            solution_node_id(link.second)});
    }

    const SolutionLink virtual_key = solution_link(virtual_link);
    solution.link_paths.insert_or_assign(virtual_key, stored_links);
    if (links.empty())
    {
        return LinkRouteResult{true, std::move(check)};
    }

    SolutionAttributeValues recorded_values;
    const std::vector<ResourceAmount>& resources =
        gather_link_resources(virtual_link, &recorded_values);
    for (const auto& link : links)
    {
        const ConstraintLink physical_link{link.first, link.second};
        resource_updator_.update_link_resources(
            physical_link,
            resources,
            ResourceUpdateOperation::subtract,
            safe);
        solution.link_paths_info.insert_or_assign(
            LinkPathInfoKey{
                virtual_key,
                solution_link(physical_link)},
            recorded_values);
    }
    return LinkRouteResult{true, std::move(check)};
}

std::vector<PreparedLinkMapper::PathCheckOutcome>&
PreparedLinkMapper::check_paths_ordered(
    ConstraintLink virtual_link,
    const PhysicalPaths& paths,
    std::size_t begin_index,
    std::size_t end_index,
    std::size_t workers)
{
    const std::size_t count = end_index - begin_index;
    auto& outcomes = path_check_scratch_;
    outcomes.assign(count, PathCheckOutcome{});
    const auto check_range =
        [this, virtual_link, &paths, &outcomes, begin_index](
            std::size_t begin,
            std::size_t end)
        {
            for (std::size_t local_index = begin;
                 local_index < end;
                 ++local_index)
            {
                const std::size_t path_index = begin_index + local_index;
                try
                {
                    outcomes[local_index].result.emplace(
                        constraint_checker_.check_path_level_constraints(
                            virtual_link, paths[path_index]));
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

LinkRouteResult PreparedLinkMapper::safely_route(
    ConstraintLink virtual_link,
    ConstraintLink physical_pair,
    Solution& solution,
    const LinkRouteOptions& options)
{
    clear_existing_route(virtual_link, solution);
    TopologyPathRequest request;
    request.virtual_link = virtual_link;
    request.physical_pair = physical_pair;
    request.options.method = options.shortest_method;
    request.options.k = options.k;
    request.options.max_path_nodes = options.max_path_nodes;
    request.options.constraint_workers =
        options.topology_constraint_workers;
    PhysicalPaths paths = topology_analyzer_.find_shortest_paths(request);
    if (options.ranker != nullptr)
    {
        (*options.ranker)(paths);
    }
    if (paths.empty())
    {
        return LinkRouteResult{false, placeholder_check()};
    }

    const bool use_parallel_windows =
        options.candidate_workers > 1U &&
        paths.size() >= minimum_parallel_path_count;
    if (!use_parallel_windows)
    {
        PathConstraintCheckResult last;
        for (const PhysicalPath& path : paths)
        {
            last = constraint_checker_.check_path_level_constraints(
                virtual_link, path);
            if (last.feasible)
            {
                LinkRouteCheckInfo check{false, std::move(last)};
                return commit_path(
                    virtual_link,
                    path,
                    std::move(check),
                    solution,
                    true,
                    options.candidate_workers);
            }
        }
        return LinkRouteResult{
            false, LinkRouteCheckInfo{false, std::move(last)}};
    }

    PathConstraintCheckResult last;
    const std::size_t bounded_workers = std::min(
        options.candidate_workers,
        paths.size());
    const std::size_t probe_count = std::min(
        {paths.size(), bounded_workers, maximum_sequential_probe_count});
    for (std::size_t index = 0U; index < probe_count; ++index)
    {
        last = constraint_checker_.check_path_level_constraints(
            virtual_link, paths[index]);
        if (last.feasible)
        {
            LinkRouteCheckInfo check{false, std::move(last)};
            return commit_path(
                virtual_link,
                paths[index],
                std::move(check),
                solution,
                true,
                options.candidate_workers);
        }
    }

    std::size_t scaled_window = paths.size();
    if (bounded_workers <=
        std::numeric_limits<std::size_t>::max() / path_window_per_worker)
    {
        scaled_window = bounded_workers * path_window_per_worker;
    }
    const std::size_t window_width = std::max(
        minimum_path_window,
        scaled_window);
    for (std::size_t window_begin = probe_count;
         window_begin < paths.size();)
    {
        const std::size_t remaining = paths.size() - window_begin;
        const std::size_t window_end = window_begin +
            std::min(window_width, remaining);
        std::vector<PathCheckOutcome>& outcomes = check_paths_ordered(
            virtual_link,
            paths,
            window_begin,
            window_end,
            options.candidate_workers);
        for (std::size_t local_index = 0U;
             local_index < outcomes.size();
             ++local_index)
        {
            PathCheckOutcome& outcome = outcomes[local_index];
            if (outcome.error)
            {
                std::rethrow_exception(outcome.error);
            }
            if (!outcome.result.has_value())
            {
                throw std::logic_error("path check produced no result");
            }
            last = std::move(*outcome.result);
            if (last.feasible)
            {
                LinkRouteCheckInfo check{false, std::move(last)};
                return commit_path(
                    virtual_link,
                    paths[window_begin + local_index],
                    std::move(check),
                    solution,
                    true,
                    options.candidate_workers);
            }
        }
        window_begin = window_end;
    }
    return LinkRouteResult{
        false, LinkRouteCheckInfo{false, std::move(last)}};
}

PreparedLinkMapper::PooledConstraintValues
PreparedLinkMapper::pool_constraints(const LinkRouteCheckInfo& check) const
{
    PooledConstraintValues result;
    if (check.placeholder)
    {
        for (const ConstraintId id : link_constraint_order_)
        {
            result.link_offsets.set(id, 100.0);
            result.link_violations.set(id, 100.0);
        }
        for (const ConstraintId id : path_constraint_order_)
        {
            const AttributeNumber* value =
                check.constraints.path_level.find(id);
            const AttributeNumber placeholder =
                value == nullptr ? AttributeNumber{0.0} : *value;
            result.path_offsets.set(id, placeholder);
            result.path_violations.set(
                id, non_negative_python_max(placeholder));
        }
        return result;
    }

    for (const ConstraintId id : link_constraint_order_)
    {
        std::vector<const AttributeNumber*> values;
        values.reserve(check.constraints.link_level.size());
        for (const PhysicalLinkConstraintResult& physical :
             check.constraints.link_level)
        {
            if (const AttributeNumber* value = physical.offsets.find(id))
            {
                values.push_back(value);
            }
        }
        if (values.empty())
        {
            throw LinkMapperException(
                LinkMapperErrorCode::empty_link_constraint_offsets,
                LinkMapperOperation::record_violation,
                "link constraint has no physical-link offsets",
                std::nullopt,
                std::nullopt,
                std::nullopt,
                id);
        }
        const AttributeNumber maximum = maximum_value(values);
        result.link_offsets.set(id, maximum);
        result.link_violations.set(
            id,
            number_as_long_double(maximum) <= 0.0L
                ? AttributeNumber{std::int64_t{0}}
                : sum_positive(values));
    }
    for (const ConstraintId id : path_constraint_order_)
    {
        const AttributeNumber* value = check.constraints.path_level.find(id);
        if (value == nullptr)
        {
            throw LinkMapperException(
                LinkMapperErrorCode::empty_link_constraint_offsets,
                LinkMapperOperation::record_violation,
                "path constraint offset is missing",
                std::nullopt,
                std::nullopt,
                std::nullopt,
                id);
        }
        result.path_offsets.set(id, *value);
        result.path_violations.set(id, non_negative_python_max(*value));
    }
    return result;
}

long double PreparedLinkMapper::violation_score(
    const LinkRouteCheckInfo& check) const
{
    const PooledConstraintValues pooled = pool_constraints(check);
    AttributeNumber total = std::int64_t{0};
    for (const ConstraintId id : combined_constraint_order_)
    {
        const AttributeNumber* value = pooled.path_violations.find(id);
        if (value == nullptr)
        {
            value = pooled.link_violations.find(id);
        }
        if (value != nullptr)
        {
            total = add_non_negative(total, *value);
        }
    }
    return number_as_long_double(total);
}

LinkRouteResult PreparedLinkMapper::unsafely_route(
    ConstraintLink virtual_link,
    ConstraintLink physical_pair,
    Solution& solution,
    const LinkRouteOptions& options)
{
    if (options.shortest_method != ShortestPathMethod::k_shortest &&
        options.shortest_method != ShortestPathMethod::all_shortest &&
        options.shortest_method != ShortestPathMethod::k_shortest_length)
    {
        throw LinkMapperException(
            LinkMapperErrorCode::unsupported_unsafe_shortest_method,
            LinkMapperOperation::route,
            "unsafe route does not support this shortest-path method",
            virtual_link,
            physical_pair);
    }

    clear_existing_route(virtual_link, solution);
    TopologyPathRequest request;
    request.virtual_link = virtual_link;
    request.physical_pair = physical_pair;
    request.options.method = options.shortest_method;
    request.options.k = options.k;
    request.options.max_path_nodes = options.max_path_nodes;
    request.options.constraint_workers =
        options.topology_constraint_workers;
    const PhysicalPaths paths = topology_analyzer_.find_shortest_paths(request);
    if (paths.empty())
    {
        return LinkRouteResult{false, placeholder_check()};
    }

    std::vector<PathConstraintCheckResult> checks;
    checks.reserve(paths.size());
    const bool use_parallel_windows =
        options.candidate_workers > 1U &&
        paths.size() >= minimum_parallel_path_count;
    if (!use_parallel_windows)
    {
        for (const PhysicalPath& path : paths)
        {
            PathConstraintCheckResult current =
                constraint_checker_.check_path_level_constraints(
                    virtual_link, path);
            if (current.feasible)
            {
                LinkRouteCheckInfo check{false, std::move(current)};
                return commit_path(
                    virtual_link,
                    path,
                    std::move(check),
                    solution,
                    true,
                    options.candidate_workers);
            }
            checks.push_back(std::move(current));
        }
    }
    else
    {
        const std::size_t bounded_workers = std::min(
            options.candidate_workers,
            paths.size());
        const std::size_t probe_count = std::min(
            {paths.size(), bounded_workers, maximum_sequential_probe_count});
        for (std::size_t index = 0U; index < probe_count; ++index)
        {
            PathConstraintCheckResult current =
                constraint_checker_.check_path_level_constraints(
                    virtual_link, paths[index]);
            if (current.feasible)
            {
                LinkRouteCheckInfo check{false, std::move(current)};
                return commit_path(
                    virtual_link,
                    paths[index],
                    std::move(check),
                    solution,
                    true,
                    options.candidate_workers);
            }
            checks.push_back(std::move(current));
        }

        std::size_t scaled_window = paths.size();
        if (bounded_workers <=
            std::numeric_limits<std::size_t>::max() /
                path_window_per_worker)
        {
            scaled_window = bounded_workers * path_window_per_worker;
        }
        const std::size_t window_width = std::max(
            minimum_path_window,
            scaled_window);
        for (std::size_t window_begin = probe_count;
             window_begin < paths.size();)
        {
            const std::size_t remaining = paths.size() - window_begin;
            const std::size_t window_end = window_begin +
                std::min(window_width, remaining);
            std::vector<PathCheckOutcome>& outcomes = check_paths_ordered(
                virtual_link,
                paths,
                window_begin,
                window_end,
                options.candidate_workers);
            for (std::size_t local_index = 0U;
                 local_index < outcomes.size();
                 ++local_index)
            {
                PathCheckOutcome& outcome = outcomes[local_index];
                if (outcome.error)
                {
                    std::rethrow_exception(outcome.error);
                }
                if (!outcome.result.has_value())
                {
                    throw std::logic_error(
                        "path check produced no result");
                }
                PathConstraintCheckResult current =
                    std::move(*outcome.result);
                if (current.feasible)
                {
                    LinkRouteCheckInfo check{false, std::move(current)};
                    return commit_path(
                        virtual_link,
                        paths[window_begin + local_index],
                        std::move(check),
                        solution,
                        true,
                        options.candidate_workers);
                }
                checks.push_back(std::move(current));
            }
            window_begin = window_end;
        }
    }

    std::size_t best_index = 0U;
    long double best_score = violation_score(
        LinkRouteCheckInfo{false, checks.front()});
    for (std::size_t index = 1U; index < checks.size(); ++index)
    {
        const long double score = violation_score(
            LinkRouteCheckInfo{false, checks[index]});
        if (python_min_replaces(best_score, score))
        {
            best_score = score;
            best_index = index;
        }
    }
    LinkRouteCheckInfo selected{false, std::move(checks[best_index])};
    return commit_path(
        virtual_link,
        paths[best_index],
        std::move(selected),
        solution,
        false,
        options.candidate_workers);
}

LinkRouteResult PreparedLinkMapper::route(
    ConstraintLink virtual_link,
    ConstraintLink physical_pair,
    Solution& solution,
    LinkRouteOptions options)
{
    if (physical_pair.source == physical_pair.target)
    {
        if (selection_.reusable)
        {
            return LinkRouteResult{true, placeholder_check()};
        }
        throw LinkMapperException(
            LinkMapperErrorCode::same_physical_node,
            LinkMapperOperation::route,
            "virtual link cannot use identical physical endpoints",
            virtual_link,
            physical_pair);
    }

    LinkRouteResult result = options.allow_constraint_violation
        ? unsafely_route(virtual_link, physical_pair, solution, options)
        : safely_route(virtual_link, physical_pair, solution, options);
    if (options.record_constraint_violation)
    {
        record_route_constraint_violation(
            virtual_link, result.check, solution);
    }
    return result;
}

void PreparedLinkMapper::record_route_constraint_violation(
    ConstraintLink virtual_link,
    const LinkRouteCheckInfo& check,
    Solution& solution) const
{
    const SolutionLink key = solution_link(virtual_link);
    const auto link_violation_id =
        solution.v_net_constraint_violations.link_level.insert_or_assign(
            key, {});
    const auto path_violation_id =
        solution.v_net_constraint_violations.path_level.insert_or_assign(
            key, {});
    const auto link_offset_id =
        solution.v_net_constraint_offsets.link_level.insert_or_assign(
            key, {});
    const auto path_offset_id =
        solution.v_net_constraint_offsets.path_level.insert_or_assign(
            key, {});

    SolutionAttributeValues& link_violations =
        solution.v_net_constraint_violations.link_level.at(link_violation_id);
    SolutionAttributeValues& path_violations =
        solution.v_net_constraint_violations.path_level.at(path_violation_id);
    SolutionAttributeValues& link_offsets =
        solution.v_net_constraint_offsets.link_level.at(link_offset_id);
    SolutionAttributeValues& path_offsets =
        solution.v_net_constraint_offsets.path_level.at(path_offset_id);

    if (check.placeholder)
    {
        for (const ConstraintId id : link_constraint_order_)
        {
            link_violations.set(id, 100.0);
            link_offsets.set(id, 100.0);
        }
    }
    else
    {
        for (const ConstraintId id : link_constraint_order_)
        {
            std::vector<const AttributeNumber*> values;
            values.reserve(check.constraints.link_level.size());
            for (const PhysicalLinkConstraintResult& physical :
                 check.constraints.link_level)
            {
                if (const AttributeNumber* value = physical.offsets.find(id))
                {
                    values.push_back(value);
                }
            }
            if (values.empty())
            {
                throw LinkMapperException(
                    LinkMapperErrorCode::empty_link_constraint_offsets,
                    LinkMapperOperation::record_violation,
                    "link constraint has no physical-link offsets",
                    virtual_link,
                    std::nullopt,
                    std::nullopt,
                    id);
            }
            const AttributeNumber maximum = maximum_value(values);
            link_offsets.set(id, maximum);
            link_violations.set(
                id,
                number_as_long_double(maximum) <= 0.0L
                    ? AttributeNumber{std::int64_t{0}}
                    : sum_positive(values));
        }
    }

    for (const ConstraintId id : path_constraint_order_)
    {
        const AttributeNumber* value = check.constraints.path_level.find(id);
        const AttributeNumber placeholder =
            check.placeholder && value == nullptr
            ? AttributeNumber{0.0}
            : value == nullptr
                ? throw LinkMapperException(
                    LinkMapperErrorCode::empty_link_constraint_offsets,
                    LinkMapperOperation::record_violation,
                    "path constraint offset is missing",
                    virtual_link,
                    std::nullopt,
                    std::nullopt,
                    id)
                : *value;
        path_offsets.set(id, placeholder);
        path_violations.set(id, non_negative_python_max(placeholder));
    }

    std::optional<AttributeNumber> maximum_hard;
    for (const ConstraintId id : combined_constraint_order_)
    {
        if (static_cast<std::size_t>(id) >= hard_constraint_mask_.size() ||
            hard_constraint_mask_[static_cast<std::size_t>(id)] ==
                std::uint8_t{0})
        {
            continue;
        }
        const AttributeNumber* value = path_violations.find(id);
        if (value == nullptr)
        {
            value = link_violations.find(id);
        }
        if (value != nullptr &&
            (!maximum_hard.has_value() ||
             python_max_replaces(*maximum_hard, *value)))
        {
            maximum_hard = *value;
        }
    }
    if (!maximum_hard.has_value())
    {
        throw LinkMapperException(
            LinkMapperErrorCode::empty_hard_constraint_violations,
            LinkMapperOperation::record_violation,
            "no pooled violation matches a selected hard constraint",
            virtual_link);
    }
    solution.v_net_total_hard_constraint_violation +=
        number_as_double(*maximum_hard);
}

bool PreparedLinkMapper::undo_route(
    ConstraintLink virtual_link,
    Solution& solution)
{
    const SolutionLink key = solution_link(virtual_link);
    const auto route_id = solution.link_paths.find_id(key);
    if (!route_id.has_value())
    {
        throw LinkMapperException(
            LinkMapperErrorCode::route_not_found,
            LinkMapperOperation::undo_route,
            "virtual link has no routed path",
            virtual_link);
    }
    const std::vector<SolutionLink>& path =
        solution.link_paths.at(*route_id);
    for (const SolutionLink& stored_physical : path)
    {
        const ConstraintLink physical_link = constraint_link(
            stored_physical,
            LinkMapperOperation::undo_route,
            virtual_link);
        const LinkPathInfoKey info_key{key, stored_physical};
        const auto info_id = solution.link_paths_info.find_id(info_key);
        if (!info_id.has_value())
        {
            throw LinkMapperException(
                LinkMapperErrorCode::route_info_not_found,
                LinkMapperOperation::undo_route,
                "routed physical link has no resource information",
                virtual_link,
                physical_link);
        }
        const SolutionAttributeValues& recorded =
            solution.link_paths_info.at(*info_id);
        auto& resources = resource_scratch_;
        resources.clear();
        resources.reserve(link_resources_.size());
        for (const PreparedLinkResource& prepared : link_resources_)
        {
            const AttributeNumber* value = recorded.find(prepared.resource_id);
            if (value == nullptr)
            {
                throw LinkMapperException(
                    LinkMapperErrorCode::route_info_not_found,
                    LinkMapperOperation::undo_route,
                    "routed physical link resource value is missing",
                    virtual_link,
                    physical_link,
                    prepared.resource_id);
            }
            resources.push_back(ResourceAmount{prepared.resource_id, *value});
        }
        resource_updator_.update_link_resources(
            physical_link,
            resources,
            ResourceUpdateOperation::add,
            true);
        solution.link_paths_info.erase(info_key);
    }
    solution.link_paths.erase(key);
    return true;
}

std::vector<ConstraintLink> PreparedLinkMapper::all_virtual_links() const
{
    std::vector<ConstraintLink> result;
    result.reserve(virtual_network_->num_links());
    const Graph& graph = virtual_network_->graph();
    auto range = graph.edges();
    for (auto iterator = range.first; iterator != range.second; ++iterator)
    {
        const auto edge = *iterator;
        result.push_back(ConstraintLink{
            graph.source(edge), graph.target(edge)});
    }
    return result;
}

ConstraintLink PreparedLinkMapper::mapped_physical_pair(
    ConstraintLink virtual_link,
    const Solution& solution) const
{
    const auto read_node = [&solution, virtual_link](Vertex virtual_node)
    {
        const auto id = solution.node_slots.find_id(
            solution_node_id(virtual_node));
        if (!id.has_value())
        {
            throw LinkMapperException(
                LinkMapperErrorCode::missing_node_slot,
                LinkMapperOperation::link_mapping,
                "virtual link endpoint has no node placement",
                virtual_link);
        }
        const SolutionNodeId stored = solution.node_slots.at(*id);
        if (stored < 0 ||
            static_cast<std::uint64_t>(stored) >
                static_cast<std::uint64_t>(
                    std::numeric_limits<Vertex>::max()))
        {
            throw LinkMapperException(
                LinkMapperErrorCode::invalid_stored_node,
                LinkMapperOperation::link_mapping,
                "stored physical node is outside the native vertex range",
                virtual_link);
        }
        return static_cast<Vertex>(stored);
    };
    return ConstraintLink{
        read_node(virtual_link.source),
        read_node(virtual_link.target)};
}

bool PreparedLinkMapper::link_mapping(
    Solution& solution,
    LinkMappingOptions options)
{
    return link_mapping(all_virtual_links(), solution, options);
}

bool PreparedLinkMapper::link_mapping(
    const std::vector<ConstraintLink>& virtual_links,
    Solution& solution,
    LinkMappingOptions options)
{
    if (options.allow_constraint_violation)
    {
        throw LinkMapperException(
            LinkMapperErrorCode::unsupported_unsafe_link_mapping,
            LinkMapperOperation::link_mapping,
            "unsafe whole-link mapping is not a valid Python behavior surface");
    }
    if (!options.inplace)
    {
        solution.link_paths.clear();
        solution.link_paths_info.clear();
        network::PhysicalNetwork physical_copy = physical_network_->clone();
        LinkMapper mapper(selection_);
        PreparedLinkMapper prepared =
            mapper.prepare(*virtual_network_, physical_copy);
        options.inplace = true;
        return prepared.link_mapping_impl(
            virtual_links, solution, options, false);
    }
    return link_mapping_impl(
        virtual_links, solution, options, true);
}

bool PreparedLinkMapper::link_mapping_impl(
    const std::vector<ConstraintLink>& virtual_links,
    Solution& solution,
    const LinkMappingOptions& options,
    bool clear_solution)
{
    if (clear_solution)
    {
        solution.link_paths.clear();
        solution.link_paths_info.clear();
    }
    for (const ConstraintLink virtual_link : virtual_links)
    {
        LinkRouteOptions route_options;
        route_options.shortest_method = options.shortest_method;
        route_options.k = options.k;
        route_options.max_path_nodes = options.max_path_nodes;
        route_options.topology_constraint_workers =
            options.topology_constraint_workers;
        route_options.candidate_workers = options.candidate_workers;
        const ConstraintLink physical_pair = mapped_physical_pair(
            virtual_link, solution);
        const LinkRouteResult routed = route(
            virtual_link,
            physical_pair,
            solution,
            route_options);
        if (!routed.routed)
        {
            solution.route_result = false;
            solution.result = false;
            return false;
        }
    }
    if (solution.link_paths.size() != virtual_network_->num_links())
    {
        throw LinkMapperException(
            LinkMapperErrorCode::mapping_cardinality_mismatch,
            LinkMapperOperation::link_mapping,
            "routed link count does not match the virtual network");
    }
    return true;
}

} // namespace virne::core::controller
