#include "constraint_checker.h"

#include "network.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <thread>
#include <utility>

namespace virne::core::controller
{
namespace
{

using network::attribute::AttributeRegistryId;
using network::attribute::CheckingLevel;
using network::attribute::GraphResourceAttribute;
using network::attribute::LinkLatencyAttribute;
using network::attribute::LinkResourceAttribute;
using network::attribute::NodeResourceAttribute;

[[noreturn]] void throw_selection_error(
    ConstraintCheckerErrorCode code,
    std::size_t item_index,
    ConstraintId id,
    const char* message)
{
    throw ConstraintCheckerException(
        code,
        ConstraintCheckerOperation::prepare,
        message,
        std::nullopt,
        item_index,
        id);
}

template <typename Function>
void parallel_indexed(
    std::size_t count,
    std::size_t requested_workers,
    Function&& function)
{
    const std::size_t worker_count = requested_workers <= 1U
        ? 1U
        : std::min(requested_workers, count);
    if (worker_count <= 1U)
    {
        for (std::size_t index = 0U; index < count; ++index)
        {
            function(index);
        }
        return;
    }

    struct Failure
    {
        std::size_t index = std::numeric_limits<std::size_t>::max();
        std::exception_ptr error;
    };

    std::vector<Failure> failures(worker_count);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    try
    {
        for (std::size_t worker = 0U; worker < worker_count; ++worker)
        {
            threads.emplace_back(
                [&, worker]
                {
                    const std::size_t begin = count * worker / worker_count;
                    const std::size_t end =
                        count * (worker + 1U) / worker_count;
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        try
                        {
                            function(index);
                        }
                        catch (...)
                        {
                            failures[worker] =
                                {index, std::current_exception()};
                            break;
                        }
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

    const Failure* first = nullptr;
    for (const Failure& failure : failures)
    {
        if (failure.error && (first == nullptr || failure.index < first->index))
        {
            first = &failure;
        }
    }
    if (first != nullptr)
    {
        std::rethrow_exception(first->error);
    }
}

template <typename Result, typename Request, typename Function>
std::vector<Result> transform_batch(
    const std::vector<Request>& requests,
    std::size_t workers,
    Function&& function)
{
    std::vector<Result> results(requests.size());
    parallel_indexed(
        requests.size(),
        workers,
        [&](std::size_t index)
        {
            try
            {
                results[index] = function(requests[index]);
            }
            catch (const ConstraintCheckerException& error)
            {
                throw ConstraintCheckerException(
                    error.code(),
                    error.operation(),
                    error.what(),
                    index,
                    error.item_index(),
                    error.constraint_id());
            }
        });
    return results;
}

const AttrMap& checked_virtual_edge(
    const network::VirtualNetwork& network_value,
    ConstraintLink link,
    ConstraintCheckerOperation operation,
    std::optional<std::size_t> item_index = std::nullopt)
{
    try
    {
        const Graph& graph = network_value.graph();
        return graph.edge_attrs(graph.edge(link.source, link.target));
    }
    catch (const std::out_of_range&)
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::virtual_link_not_found,
            operation,
            "virtual link was not found",
            std::nullopt,
            item_index);
    }
    catch (const std::runtime_error&)
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::virtual_link_not_found,
            operation,
            "virtual link was not found",
            std::nullopt,
            item_index);
    }
}

const AttrMap& checked_physical_edge(
    const network::PhysicalNetwork& network_value,
    ConstraintLink link,
    ConstraintCheckerOperation operation,
    std::optional<std::size_t> item_index = std::nullopt)
{
    try
    {
        const Graph& graph = network_value.graph();
        return graph.edge_attrs(graph.edge(link.source, link.target));
    }
    catch (const std::out_of_range&)
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::physical_link_not_found,
            operation,
            "physical link was not found",
            std::nullopt,
            item_index);
    }
    catch (const std::runtime_error&)
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::physical_link_not_found,
            operation,
            "physical link was not found",
            std::nullopt,
            item_index);
    }
}

} // namespace

ConstraintCheckerException::ConstraintCheckerException(
    ConstraintCheckerErrorCode code,
    ConstraintCheckerOperation operation,
    std::string message,
    std::optional<std::size_t> request_index,
    std::optional<std::size_t> item_index,
    std::optional<ConstraintId> constraint_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      request_index_(request_index),
      item_index_(item_index),
      constraint_id_(constraint_id)
{
}

ConstraintCheckerErrorCode ConstraintCheckerException::code() const noexcept
{
    return code_;
}

ConstraintCheckerOperation ConstraintCheckerException::operation() const noexcept
{
    return operation_;
}

const std::optional<std::size_t>&
ConstraintCheckerException::request_index() const noexcept
{
    return request_index_;
}

const std::optional<std::size_t>&
ConstraintCheckerException::item_index() const noexcept
{
    return item_index_;
}

const std::optional<ConstraintId>&
ConstraintCheckerException::constraint_id() const noexcept
{
    return constraint_id_;
}

ConstraintChecker::ConstraintChecker(ConstraintCheckerSelection selection)
    : selection_(std::move(selection))
{
}

const ConstraintCheckerSelection& ConstraintChecker::selection() const noexcept
{
    return selection_;
}

PreparedConstraintChecker ConstraintChecker::prepare(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network) const
{
    std::vector<PreparedConstraintChecker::PreparedNodeConstraint>
        node_constraints;
    node_constraints.reserve(selection_.node_at_node.size());
    // Selection IDs belong to the virtual definition registry. The selected
    // dynamic name is resolved once by bind() against each independent graph;
    // registry IDs are never copied across networks.
    const auto& node_entries = virtual_network.node_attributes().entries();
    for (std::size_t index = 0U; index < selection_.node_at_node.size(); ++index)
    {
        const ConstraintId id = selection_.node_at_node[index];
        if (id >= node_entries.size())
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_node_selection,
                index,
                id,
                "node constraint selection is out of range");
        }
        const auto* attribute = dynamic_cast<const NodeResourceAttribute*>(
            node_entries[id].attribute.get());
        if (attribute == nullptr ||
            attribute->checking_level() != CheckingLevel::node)
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_node_selection,
                index,
                id,
                "node constraint is not a node-level resource");
        }
        const auto virtual_binding = attribute->bind(virtual_network.graph());
        const auto physical_binding = attribute->bind(physical_network.graph());
        node_constraints.push_back({
            id,
            attribute,
            virtual_binding.value_id,
            physical_binding.value_id});
    }

    std::vector<PreparedConstraintChecker::PreparedLinkConstraint>
        link_constraints;
    link_constraints.reserve(selection_.link_at_link.size());
    const auto& link_entries = virtual_network.link_attributes().entries();
    for (std::size_t index = 0U; index < selection_.link_at_link.size(); ++index)
    {
        const ConstraintId id = selection_.link_at_link[index];
        if (id >= link_entries.size())
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_link_selection,
                index,
                id,
                "link constraint selection is out of range");
        }
        const auto* attribute = dynamic_cast<const LinkResourceAttribute*>(
            link_entries[id].attribute.get());
        if (attribute == nullptr ||
            attribute->checking_level() != CheckingLevel::link)
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_link_selection,
                index,
                id,
                "link constraint is not a link-level resource");
        }
        const auto virtual_binding = attribute->bind(virtual_network.graph());
        const auto physical_binding = attribute->bind(physical_network.graph());
        link_constraints.push_back({
            id,
            attribute,
            virtual_binding.value_id,
            physical_binding.value_id});
    }

    std::vector<PreparedConstraintChecker::PreparedPathConstraint>
        path_constraints;
    path_constraints.reserve(selection_.link_at_path.size());
    for (std::size_t index = 0U; index < selection_.link_at_path.size(); ++index)
    {
        const ConstraintId id = selection_.link_at_path[index];
        if (id >= link_entries.size())
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_path_selection,
                index,
                id,
                "path constraint selection is out of range");
        }
        const auto* attribute = dynamic_cast<const LinkLatencyAttribute*>(
            link_entries[id].attribute.get());
        if (attribute == nullptr ||
            attribute->checking_level() != CheckingLevel::path)
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_path_selection,
                index,
                id,
                "path constraint is not a path-level latency");
        }
        const auto virtual_binding = attribute->bind(virtual_network.graph());
        const auto physical_binding = attribute->bind(physical_network.graph());
        path_constraints.push_back({
            id,
            attribute,
            virtual_binding.value_id,
            physical_binding.value_id});
    }

    std::vector<PreparedConstraintChecker::PreparedGraphConstraint>
        graph_constraints;
    graph_constraints.reserve(selection_.graph.size());
    for (std::size_t index = 0U; index < selection_.graph.size(); ++index)
    {
        const GraphConstraintSelection& selected = selection_.graph[index];
        if (selected.attribute == nullptr)
        {
            throw ConstraintCheckerException(
                ConstraintCheckerErrorCode::null_graph_attribute,
                ConstraintCheckerOperation::prepare,
                "graph constraint attribute is null",
                std::nullopt,
                index,
                selected.output_id);
        }
        if (selected.output_id == invalid_constraint_id ||
            selected.attribute->checking_level() != CheckingLevel::graph)
        {
            throw_selection_error(
                ConstraintCheckerErrorCode::invalid_graph_selection,
                index,
                selected.output_id,
                "graph constraint selection is invalid");
        }
        graph_constraints.push_back({
            selected.output_id,
            selected.attribute,
            selected.attribute->bind(virtual_network.graph()),
            selected.attribute->bind(physical_network.graph())});
    }

    return PreparedConstraintChecker(
        virtual_network,
        physical_network,
        std::move(node_constraints),
        std::move(link_constraints),
        std::move(path_constraints),
        std::move(graph_constraints));
}

PreparedConstraintChecker::PreparedConstraintChecker(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    std::vector<PreparedNodeConstraint> node_constraints,
    std::vector<PreparedLinkConstraint> link_constraints,
    std::vector<PreparedPathConstraint> path_constraints,
    std::vector<PreparedGraphConstraint> graph_constraints)
    : virtual_network_(&virtual_network),
      physical_network_(&physical_network),
      node_constraints_(std::move(node_constraints)),
      link_constraints_(std::move(link_constraints)),
      path_constraints_(std::move(path_constraints)),
      graph_constraints_(std::move(graph_constraints))
{
}

ConstraintCheckResult PreparedConstraintChecker::check_graph_constraints() const
{
    ConstraintCheckResult result;
    const AttrMap& virtual_values = virtual_network_->graph_attributes();
    const AttrMap& physical_values = physical_network_->graph_attributes();
    for (const PreparedGraphConstraint& constraint : graph_constraints_)
    {
        const auto checked = constraint.attribute->check_constraint_satisfiability(
            virtual_values,
            constraint.virtual_binding.value_id,
            physical_values,
            constraint.physical_binding.value_id);
        if (!checked.flag)
        {
            result.feasible = false;
        }
        result.offsets.set(constraint.output_id, checked.offset);
    }
    return result;
}

ConstraintCheckResult PreparedConstraintChecker::check_node_level_constraints(
    Vertex virtual_node,
    Vertex physical_node) const
{
    const Graph& physical_graph = physical_network_->graph();
    if (physical_node >= physical_graph.num_nodes())
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::physical_node_out_of_range,
            ConstraintCheckerOperation::check_node,
            "physical node is out of range");
    }
    const Graph& virtual_graph = virtual_network_->graph();
    if (virtual_node >= virtual_graph.num_nodes())
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::virtual_node_out_of_range,
            ConstraintCheckerOperation::check_node,
            "virtual node is out of range");
    }

    const AttrMap& virtual_values = virtual_graph.node_attrs(virtual_node);
    const AttrMap& physical_values = physical_graph.node_attrs(physical_node);
    ConstraintCheckResult result;
    for (const PreparedNodeConstraint& constraint : node_constraints_)
    {
        const auto checked = constraint.attribute->check_constraint_satisfiability(
            virtual_values,
            constraint.virtual_value_id,
            physical_values,
            constraint.physical_value_id);
        if (!checked.flag)
        {
            result.feasible = false;
        }
        result.offsets.set(constraint.output_id, checked.offset);
    }
    return result;
}

ConstraintCheckResult PreparedConstraintChecker::check_link_values(
    const AttrMap& virtual_link,
    const AttrMap& physical_link) const
{
    ConstraintCheckResult result;
    for (const PreparedLinkConstraint& constraint : link_constraints_)
    {
        const auto checked = constraint.attribute->check_constraint_satisfiability(
            virtual_link,
            constraint.virtual_value_id,
            physical_link,
            constraint.physical_value_id);
        if (!checked.flag)
        {
            result.feasible = false;
        }
        result.offsets.set(constraint.output_id, checked.offset);
    }
    return result;
}

ConstraintCheckResult PreparedConstraintChecker::check_link_level_constraints(
    ConstraintLink virtual_link,
    ConstraintLink physical_link) const
{
    const AttrMap& virtual_values = checked_virtual_edge(
        *virtual_network_, virtual_link, ConstraintCheckerOperation::check_link);
    const AttrMap& physical_values = checked_physical_edge(
        *physical_network_, physical_link, ConstraintCheckerOperation::check_link);
    return check_link_values(virtual_values, physical_values);
}

PathConstraintCheckResult
PreparedConstraintChecker::check_path_level_constraints(
    ConstraintLink virtual_link,
    const std::vector<Vertex>& physical_path) const
{
    virne::utils::PathLinks physical_links;
    try
    {
        physical_links = virne::utils::path_to_links(physical_path, 0U);
    }
    catch (const std::invalid_argument&)
    {
        throw ConstraintCheckerException(
            ConstraintCheckerErrorCode::invalid_path,
            ConstraintCheckerOperation::check_path,
            "physical path must contain at least two nodes");
    }

    // The virtual edge is immutable for the prepared check. Resolve it once
    // before the first physical edge, preserving Python's first-access order
    // without repeating graph lookup for every hop.
    const AttrMap& virtual_values = checked_virtual_edge(
        *virtual_network_,
        virtual_link,
        ConstraintCheckerOperation::check_path,
        0U);

    PathConstraintCheckResult result;
    result.link_level.resize(physical_links.size());
    std::vector<const AttrMap*> physical_values(physical_links.size());
    bool link_feasible = true;
    for (std::size_t index = 0U; index < physical_links.size(); ++index)
    {
        const ConstraintLink physical_link{
            physical_links[index].first,
            physical_links[index].second};
        const AttrMap& physical_map = checked_physical_edge(
            *physical_network_,
            physical_link,
            ConstraintCheckerOperation::check_path,
            index);
        ConstraintCheckResult link_result =
            check_link_values(virtual_values, physical_map);
        if (!link_result.feasible)
        {
            link_feasible = false;
        }
        result.link_level[index] = {
            physical_link,
            std::move(link_result.offsets)};
        physical_values[index] = &physical_map;
    }

    bool path_feasible = true;
    for (const PreparedPathConstraint& constraint : path_constraints_)
    {
        const auto checked = constraint.attribute->check_constraint_satisfiability(
            virtual_values,
            constraint.virtual_value_id,
            physical_values,
            constraint.physical_value_id);
        if (!checked.flag)
        {
            path_feasible = false;
        }
        result.path_level.set(constraint.output_id, checked.offset);
    }
    result.feasible = link_feasible && path_feasible;
    return result;
}

std::vector<ConstraintCheckResult>
PreparedConstraintChecker::check_node_level_constraints_batch(
    const std::vector<NodeConstraintRequest>& requests,
    std::size_t workers) const
{
    return transform_batch<ConstraintCheckResult>(
        requests,
        workers,
        [&](const NodeConstraintRequest& request)
        {
            return check_node_level_constraints(
                request.virtual_node, request.physical_node);
        });
}

std::vector<ConstraintCheckResult>
PreparedConstraintChecker::check_link_level_constraints_batch(
    const std::vector<LinkConstraintRequest>& requests,
    std::size_t workers) const
{
    return transform_batch<ConstraintCheckResult>(
        requests,
        workers,
        [&](const LinkConstraintRequest& request)
        {
            return check_link_level_constraints(
                request.virtual_link, request.physical_link);
        });
}

std::vector<PathConstraintCheckResult>
PreparedConstraintChecker::check_path_level_constraints_batch(
    const std::vector<PathConstraintRequest>& requests,
    std::size_t workers) const
{
    return transform_batch<PathConstraintCheckResult>(
        requests,
        workers,
        [&](const PathConstraintRequest& request)
        {
            return check_path_level_constraints(
                request.virtual_link, request.physical_path);
        });
}

} // namespace virne::core::controller
