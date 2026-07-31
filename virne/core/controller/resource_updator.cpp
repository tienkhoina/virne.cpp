#include "resource_updator.h"

#include "../../utils/deterministic_executor.h"
#include "network.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

namespace virne::core::controller
{
namespace
{

using network::attribute::AttributeMethodErrorCode;
using network::attribute::AttributeMethodException;
using network::attribute::AttributeNumber;
using network::attribute::LinkResourceAttribute;
using network::attribute::NodeResourceAttribute;
using network::attribute::ResourceUpdateOperation;

[[noreturn]] void throw_selection_error(
    ResourceUpdatorErrorCode code,
    std::size_t item_index,
    ResourceId id,
    const char* message)
{
    throw ResourceUpdatorException(
        code,
        ResourceUpdatorOperation::prepare,
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
    if (requested_workers <= 1U || count <= 1U)
    {
        for (std::size_t index = 0U; index < count; ++index)
        {
            function(index);
        }
        return;
    }

    std::vector<std::exception_ptr> failures(count);
    virne::utils::deterministic_parallel_blocks(
        count,
        requested_workers,
        1U,
        [&](std::size_t begin, std::size_t end)
        {
            for (std::size_t index = begin; index < end; ++index)
            {
                try
                {
                    function(index);
                }
                catch (...)
                {
                    failures[index] = std::current_exception();
                    break;
                }
            }
        });

    for (const std::exception_ptr& failure : failures)
    {
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }
}

ResourceUpdatorException with_request_index(
    const ResourceUpdatorException& error,
    std::size_t request_index,
    ResourceUpdatorOperation operation)
{
    return ResourceUpdatorException(
        error.code(),
        operation,
        error.what(),
        request_index,
        error.item_index(),
        error.resource_id());
}

AttributeNumber to_number(
    const AttrValue& value,
    ResourceUpdatorOperation operation,
    ResourceId resource_id,
    std::optional<std::size_t> item_index)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return *integer;
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean;
    }
    throw ResourceUpdatorException(
        ResourceUpdatorErrorCode::non_numeric_resource,
        operation,
        "resource value is not numeric",
        std::nullopt,
        item_index,
        resource_id);
}

AttrValue to_attr_value(const AttributeNumber& value)
{
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return *integer;
    }
    return std::get<double>(value);
}

void update_value(
    AttrMap& values,
    AttrId value_id,
    const ResourceAmount& resource,
    ResourceUpdateOperation update_operation,
    bool safe,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> item_index)
{
    AttrValue* current = values.find(value_id);
    if (current == nullptr)
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::missing_resource_value,
            operation,
            "resource value is missing",
            std::nullopt,
            item_index,
            resource.resource_id);
    }

    AttributeNumber physical =
        to_number(*current, operation, resource.resource_id, item_index);
    try
    {
        network::attribute::update_resource_value(
            resource.value,
            physical,
            update_operation,
            safe);
    }
    catch (const AttributeMethodException& error)
    {
        const ResourceUpdatorErrorCode code =
            error.code() == AttributeMethodErrorCode::insufficient_resource
            ? ResourceUpdatorErrorCode::insufficient_resource
            : ResourceUpdatorErrorCode::numeric_update_failure;
        throw ResourceUpdatorException(
            code,
            operation,
            error.what(),
            std::nullopt,
            item_index,
            resource.resource_id);
    }
    *current = to_attr_value(physical);
}

const AttrMap& checked_virtual_edge(
    const network::VirtualNetwork& network_value,
    ConstraintLink link,
    std::optional<std::size_t> item_index = std::nullopt)
{
    try
    {
        const Graph& graph = network_value.graph();
        return graph.edge_attrs(graph.edge(link.source, link.target));
    }
    catch (const std::out_of_range&)
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::virtual_link_not_found,
            ResourceUpdatorOperation::update_path,
            "virtual link was not found",
            std::nullopt,
            item_index);
    }
    catch (const std::runtime_error&)
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::virtual_link_not_found,
            ResourceUpdatorOperation::update_path,
            "virtual link was not found",
            std::nullopt,
            item_index);
    }
}

AttrMap& checked_physical_edge(
    network::PhysicalNetwork& network_value,
    ConstraintLink link,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> item_index = std::nullopt)
{
    try
    {
        Graph& graph = network_value.graph();
        return graph.edge_attrs(graph.edge(link.source, link.target));
    }
    catch (const std::out_of_range&)
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::physical_link_not_found,
            operation,
            "physical link was not found",
            std::nullopt,
            item_index);
    }
    catch (const std::runtime_error&)
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::physical_link_not_found,
            operation,
            "physical link was not found",
            std::nullopt,
            item_index);
    }
}

bool duplicate_nodes(const std::vector<NodeResourceUpdateRequest>& requests)
{
    std::vector<Vertex> targets;
    targets.reserve(requests.size());
    for (const NodeResourceUpdateRequest& request : requests)
    {
        if (!request.resources.empty())
        {
            targets.push_back(request.physical_node);
        }
    }
    std::sort(targets.begin(), targets.end());
    return std::adjacent_find(targets.begin(), targets.end()) != targets.end();
}

std::pair<Vertex, Vertex> canonical_link(ConstraintLink link) noexcept
{
    return link.source <= link.target
        ? std::pair<Vertex, Vertex>{link.source, link.target}
        : std::pair<Vertex, Vertex>{link.target, link.source};
}

bool duplicate_links(const std::vector<LinkResourceUpdateRequest>& requests)
{
    std::vector<std::pair<Vertex, Vertex>> targets;
    targets.reserve(requests.size());
    for (const LinkResourceUpdateRequest& request : requests)
    {
        if (!request.resources.empty())
        {
            targets.push_back(canonical_link(request.physical_link));
        }
    }
    std::sort(targets.begin(), targets.end());
    return std::adjacent_find(targets.begin(), targets.end()) != targets.end();
}

} // namespace

ResourceUpdatorException::ResourceUpdatorException(
    ResourceUpdatorErrorCode code,
    ResourceUpdatorOperation operation,
    std::string message,
    std::optional<std::size_t> request_index,
    std::optional<std::size_t> item_index,
    std::optional<ResourceId> resource_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      request_index_(request_index),
      item_index_(item_index),
      resource_id_(resource_id)
{
}

ResourceUpdatorErrorCode ResourceUpdatorException::code() const noexcept
{
    return code_;
}

ResourceUpdatorOperation ResourceUpdatorException::operation() const noexcept
{
    return operation_;
}

const std::optional<std::size_t>&
ResourceUpdatorException::request_index() const noexcept
{
    return request_index_;
}

const std::optional<std::size_t>&
ResourceUpdatorException::item_index() const noexcept
{
    return item_index_;
}

const std::optional<ResourceId>&
ResourceUpdatorException::resource_id() const noexcept
{
    return resource_id_;
}

ResourceUpdator::ResourceUpdator(ResourceUpdatorSelection selection)
    : selection_(std::move(selection))
{
}

const ResourceUpdatorSelection& ResourceUpdator::selection() const noexcept
{
    return selection_;
}

PreparedResourceUpdator ResourceUpdator::prepare(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network) const
{
    const auto& node_entries = virtual_network.node_attributes().entries();
    std::vector<std::optional<PreparedResourceUpdator::PreparedNodeResource>>
        node_resources(node_entries.size());
    for (std::size_t index = 0U; index < selection_.node_resources.size(); ++index)
    {
        const ResourceId id = selection_.node_resources[index];
        if (id >= node_entries.size())
        {
            throw_selection_error(
                ResourceUpdatorErrorCode::invalid_node_selection,
                index,
                id,
                "node resource selection is out of range");
        }
        const auto* attribute = dynamic_cast<const NodeResourceAttribute*>(
            node_entries[id].attribute.get());
        if (attribute == nullptr)
        {
            throw_selection_error(
                ResourceUpdatorErrorCode::invalid_node_selection,
                index,
                id,
                "node resource selection has the wrong family");
        }
        const auto physical_binding = attribute->bind(physical_network.graph());
        node_resources[id] = PreparedResourceUpdator::PreparedNodeResource{
            id, physical_binding.value_id};
    }

    const auto& link_entries = virtual_network.link_attributes().entries();
    std::vector<std::optional<PreparedResourceUpdator::PreparedLinkResource>>
        link_resources(link_entries.size());
    std::vector<PreparedResourceUpdator::PreparedLinkResource> path_resources;
    path_resources.reserve(selection_.link_resources.size());
    for (std::size_t index = 0U; index < selection_.link_resources.size(); ++index)
    {
        const ResourceId id = selection_.link_resources[index];
        if (id >= link_entries.size())
        {
            throw_selection_error(
                ResourceUpdatorErrorCode::invalid_link_selection,
                index,
                id,
                "link resource selection is out of range");
        }
        const auto* attribute = dynamic_cast<const LinkResourceAttribute*>(
            link_entries[id].attribute.get());
        if (attribute == nullptr)
        {
            throw_selection_error(
                ResourceUpdatorErrorCode::invalid_link_selection,
                index,
                id,
                "link resource selection has the wrong family");
        }
        const auto virtual_binding = attribute->bind(virtual_network.graph());
        const auto physical_binding = attribute->bind(physical_network.graph());
        const PreparedResourceUpdator::PreparedLinkResource prepared{
            id,
            attribute,
            virtual_binding.value_id,
            physical_binding.value_id};
        link_resources[id] = prepared;
        path_resources.push_back(prepared);
    }

    return PreparedResourceUpdator(
        virtual_network,
        physical_network,
        std::move(node_resources),
        std::move(link_resources),
        std::move(path_resources));
}

PreparedResourceUpdator::PreparedResourceUpdator(
    const network::VirtualNetwork& virtual_network,
    network::PhysicalNetwork& physical_network,
    std::vector<std::optional<PreparedNodeResource>> node_resources,
    std::vector<std::optional<PreparedLinkResource>> link_resources,
    std::vector<PreparedLinkResource> path_resources)
    : virtual_network_(&virtual_network),
      physical_network_(&physical_network),
      node_resources_(std::move(node_resources)),
      link_resources_(std::move(link_resources)),
      path_resources_(std::move(path_resources))
{
}

const PreparedResourceUpdator::PreparedNodeResource&
PreparedResourceUpdator::node_resource(
    ResourceId id,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> item_index) const
{
    if (id >= node_resources_.size() || !node_resources_[id])
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::unprepared_resource_id,
            operation,
            "node resource ID was not prepared",
            std::nullopt,
            item_index,
            id);
    }
    return *node_resources_[id];
}

const PreparedResourceUpdator::PreparedLinkResource&
PreparedResourceUpdator::link_resource(
    ResourceId id,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> item_index) const
{
    if (id >= link_resources_.size() || !link_resources_[id])
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::unprepared_resource_id,
            operation,
            "link resource ID was not prepared",
            std::nullopt,
            item_index,
            id);
    }
    return *link_resources_[id];
}

void PreparedResourceUpdator::update_node_resource_impl(
    Vertex physical_node,
    const ResourceAmount& resource,
    ResourceUpdateOperation update_operation,
    bool safe,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> item_index)
{
    Graph& graph = physical_network_->graph();
    if (physical_node >= graph.num_nodes())
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::physical_node_out_of_range,
            operation,
            "physical node is out of range",
            std::nullopt,
            item_index,
            resource.resource_id);
    }
    const PreparedNodeResource& prepared =
        node_resource(resource.resource_id, operation, item_index);
    update_value(
        graph.node_attrs(physical_node),
        prepared.physical_value_id,
        resource,
        update_operation,
        safe,
        operation,
        item_index);
}

void PreparedResourceUpdator::update_link_resource_impl(
    ConstraintLink physical_link,
    const ResourceAmount& resource,
    ResourceUpdateOperation update_operation,
    bool safe,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> item_index)
{
    AttrMap& values = checked_physical_edge(
        *physical_network_, physical_link, operation, item_index);
    const PreparedLinkResource& prepared =
        link_resource(resource.resource_id, operation, item_index);
    update_value(
        values,
        prepared.physical_value_id,
        resource,
        update_operation,
        safe,
        operation,
        item_index);
}

void PreparedResourceUpdator::update_node_resource(
    Vertex physical_node,
    ResourceAmount resource,
    ResourceUpdateOperation operation,
    bool safe)
{
    update_node_resource_impl(
        physical_node,
        resource,
        operation,
        safe,
        ResourceUpdatorOperation::update_node,
        std::nullopt);
}

void PreparedResourceUpdator::update_node_resources(
    Vertex physical_node,
    const std::vector<ResourceAmount>& resources,
    ResourceUpdateOperation operation,
    bool safe)
{
    for (std::size_t index = 0U; index < resources.size(); ++index)
    {
        update_node_resource_impl(
            physical_node,
            resources[index],
            operation,
            safe,
            ResourceUpdatorOperation::update_node,
            index);
    }
}

void PreparedResourceUpdator::update_link_resource(
    ConstraintLink physical_link,
    ResourceAmount resource,
    ResourceUpdateOperation operation,
    bool safe)
{
    update_link_resource_impl(
        physical_link,
        resource,
        operation,
        safe,
        ResourceUpdatorOperation::update_link,
        std::nullopt);
}

void PreparedResourceUpdator::update_link_resources(
    ConstraintLink physical_link,
    const std::vector<ResourceAmount>& resources,
    ResourceUpdateOperation operation,
    bool safe)
{
    for (std::size_t index = 0U; index < resources.size(); ++index)
    {
        update_link_resource_impl(
            physical_link,
            resources[index],
            operation,
            safe,
            ResourceUpdatorOperation::update_link,
            index);
    }
}

void PreparedResourceUpdator::update_path_resources(
    ConstraintLink virtual_link,
    const std::vector<Vertex>& physical_path,
    ResourceUpdateOperation update_operation,
    bool safe)
{
    if (path_resources_.empty())
    {
        return;
    }

    const AttrMap& virtual_values =
        checked_virtual_edge(*virtual_network_, virtual_link);
    virne::utils::PathLinks physical_links;
    try
    {
        physical_links = virne::utils::path_to_links(physical_path, 0U);
    }
    catch (const std::invalid_argument&)
    {
        throw ResourceUpdatorException(
            ResourceUpdatorErrorCode::invalid_path,
            ResourceUpdatorOperation::update_path,
            "physical path must contain at least two nodes");
    }

    for (const PreparedLinkResource& prepared : path_resources_)
    {
        for (std::size_t index = 0U; index < physical_links.size(); ++index)
        {
            const ConstraintLink physical_link{
                physical_links[index].first,
                physical_links[index].second};
            AttrMap& physical_values = checked_physical_edge(
                *physical_network_,
                physical_link,
                ResourceUpdatorOperation::update_path,
                index);
            const AttrValue* virtual_value =
                virtual_values.find(prepared.virtual_value_id);
            if (virtual_value == nullptr)
            {
                throw ResourceUpdatorException(
                    ResourceUpdatorErrorCode::missing_resource_value,
                    ResourceUpdatorOperation::update_path,
                    "virtual link resource value is missing",
                    std::nullopt,
                    index,
                    prepared.resource_id);
            }
            const ResourceAmount resource{
                prepared.resource_id,
                to_number(
                    *virtual_value,
                    ResourceUpdatorOperation::update_path,
                    prepared.resource_id,
                    index)};
            update_value(
                physical_values,
                prepared.physical_value_id,
                resource,
                update_operation,
                safe,
                ResourceUpdatorOperation::update_path,
                index);
        }
    }
}

void PreparedResourceUpdator::update_node_resources_batch(
    const std::vector<NodeResourceUpdateRequest>& requests,
    ResourceUpdateOperation update_operation,
    bool safe,
    std::size_t workers)
{
    if (workers <= 1U || requests.size() <= 1U || duplicate_nodes(requests))
    {
        for (std::size_t request_index = 0U;
             request_index < requests.size();
             ++request_index)
        {
            const NodeResourceUpdateRequest& request = requests[request_index];
            try
            {
                for (std::size_t item = 0U;
                     item < request.resources.size();
                     ++item)
                {
                    update_node_resource_impl(
                        request.physical_node,
                        request.resources[item],
                        update_operation,
                        safe,
                        ResourceUpdatorOperation::update_node_batch,
                        item);
                }
            }
            catch (const ResourceUpdatorException& error)
            {
                throw with_request_index(
                    error,
                    request_index,
                    ResourceUpdatorOperation::update_node_batch);
            }
        }
        return;
    }

    std::vector<std::optional<AttrMap>> plans(requests.size());
    parallel_indexed(
        requests.size(),
        workers,
        [&](std::size_t request_index)
        {
            const NodeResourceUpdateRequest& request = requests[request_index];
            if (request.resources.empty())
            {
                return;
            }
            try
            {
                Graph& graph = physical_network_->graph();
                if (request.physical_node >= graph.num_nodes())
                {
                    throw ResourceUpdatorException(
                        ResourceUpdatorErrorCode::physical_node_out_of_range,
                        ResourceUpdatorOperation::update_node_batch,
                        "physical node is out of range");
                }
                plans[request_index].emplace(
                    graph.node_attrs(request.physical_node));
                for (std::size_t item = 0U;
                     item < request.resources.size();
                     ++item)
                {
                    const ResourceAmount& resource = request.resources[item];
                    const PreparedNodeResource& prepared = node_resource(
                        resource.resource_id,
                        ResourceUpdatorOperation::update_node_batch,
                        item);
                    update_value(
                        *plans[request_index],
                        prepared.physical_value_id,
                        resource,
                        update_operation,
                        safe,
                        ResourceUpdatorOperation::update_node_batch,
                        item);
                }
            }
            catch (const ResourceUpdatorException& error)
            {
                throw with_request_index(
                    error,
                    request_index,
                    ResourceUpdatorOperation::update_node_batch);
            }
        });

    parallel_indexed(
        requests.size(),
        workers,
        [&](std::size_t request_index)
        {
            if (plans[request_index])
            {
                physical_network_->graph().node_attrs(
                    requests[request_index].physical_node) =
                    std::move(*plans[request_index]);
            }
        });
}

void PreparedResourceUpdator::update_link_resources_batch(
    const std::vector<LinkResourceUpdateRequest>& requests,
    ResourceUpdateOperation update_operation,
    bool safe,
    std::size_t workers)
{
    if (workers <= 1U || requests.size() <= 1U || duplicate_links(requests))
    {
        for (std::size_t request_index = 0U;
             request_index < requests.size();
             ++request_index)
        {
            const LinkResourceUpdateRequest& request = requests[request_index];
            try
            {
                for (std::size_t item = 0U;
                     item < request.resources.size();
                     ++item)
                {
                    update_link_resource_impl(
                        request.physical_link,
                        request.resources[item],
                        update_operation,
                        safe,
                        ResourceUpdatorOperation::update_link_batch,
                        item);
                }
            }
            catch (const ResourceUpdatorException& error)
            {
                throw with_request_index(
                    error,
                    request_index,
                    ResourceUpdatorOperation::update_link_batch);
            }
        }
        return;
    }

    std::vector<std::optional<AttrMap>> plans(requests.size());
    parallel_indexed(
        requests.size(),
        workers,
        [&](std::size_t request_index)
        {
            const LinkResourceUpdateRequest& request = requests[request_index];
            if (request.resources.empty())
            {
                return;
            }
            try
            {
                AttrMap& current = checked_physical_edge(
                    *physical_network_,
                    request.physical_link,
                    ResourceUpdatorOperation::update_link_batch);
                plans[request_index].emplace(current);
                for (std::size_t item = 0U;
                     item < request.resources.size();
                     ++item)
                {
                    const ResourceAmount& resource = request.resources[item];
                    const PreparedLinkResource& prepared = link_resource(
                        resource.resource_id,
                        ResourceUpdatorOperation::update_link_batch,
                        item);
                    update_value(
                        *plans[request_index],
                        prepared.physical_value_id,
                        resource,
                        update_operation,
                        safe,
                        ResourceUpdatorOperation::update_link_batch,
                        item);
                }
            }
            catch (const ResourceUpdatorException& error)
            {
                throw with_request_index(
                    error,
                    request_index,
                    ResourceUpdatorOperation::update_link_batch);
            }
        });

    parallel_indexed(
        requests.size(),
        workers,
        [&](std::size_t request_index)
        {
            if (plans[request_index])
            {
                AttrMap& target = checked_physical_edge(
                    *physical_network_,
                    requests[request_index].physical_link,
                    ResourceUpdatorOperation::update_link_batch);
                target = std::move(*plans[request_index]);
            }
        });
}

} // namespace virne::core::controller
