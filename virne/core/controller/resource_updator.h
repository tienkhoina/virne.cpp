#pragma once

#include "constraint_checker.h"

#include "attribute/attribute_method.h"
#include "physical_network.h"
#include "virtual_network.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::core::controller
{

using ResourceId = network::attribute::AttributeRegistryId;

inline constexpr ResourceId invalid_resource_id =
    network::attribute::invalid_attribute_registry_id;

enum class ResourceUpdatorErrorCode : std::uint8_t
{
    invalid_node_selection,
    invalid_link_selection,
    unprepared_resource_id,
    physical_node_out_of_range,
    virtual_link_not_found,
    physical_link_not_found,
    invalid_path,
    missing_resource_value,
    non_numeric_resource,
    insufficient_resource,
    numeric_update_failure,
};

enum class ResourceUpdatorOperation : std::uint8_t
{
    prepare,
    update_node,
    update_link,
    update_path,
    update_node_batch,
    update_link_batch,
};

class ResourceUpdatorException : public std::runtime_error
{
public:
    ResourceUpdatorException(
        ResourceUpdatorErrorCode code,
        ResourceUpdatorOperation operation,
        std::string message,
        std::optional<std::size_t> request_index = std::nullopt,
        std::optional<std::size_t> item_index = std::nullopt,
        std::optional<ResourceId> resource_id = std::nullopt);

    ResourceUpdatorErrorCode code() const noexcept;
    ResourceUpdatorOperation operation() const noexcept;
    const std::optional<std::size_t>& request_index() const noexcept;
    const std::optional<std::size_t>& item_index() const noexcept;
    const std::optional<ResourceId>& resource_id() const noexcept;

private:
    ResourceUpdatorErrorCode code_;
    ResourceUpdatorOperation operation_;
    std::optional<std::size_t> request_index_;
    std::optional<std::size_t> item_index_;
    std::optional<ResourceId> resource_id_;
};

struct ResourceAmount
{
    ResourceId resource_id = 0U;
    network::attribute::AttributeNumber value = std::int64_t{0};
};

struct ResourceUpdatorSelection
{
    std::vector<ResourceId> node_resources;
    std::vector<ResourceId> link_resources;
};

struct NodeResourceUpdateRequest
{
    Vertex physical_node = 0U;
    std::vector<ResourceAmount> resources;
};

struct LinkResourceUpdateRequest
{
    ConstraintLink physical_link;
    std::vector<ResourceAmount> resources;
};

class PreparedResourceUpdator;

class ResourceUpdator
{
public:
    explicit ResourceUpdator(ResourceUpdatorSelection selection);

    const ResourceUpdatorSelection& selection() const noexcept;

    PreparedResourceUpdator prepare(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network) const;

private:
    ResourceUpdatorSelection selection_;
};

class PreparedResourceUpdator
{
public:
    using ResourceUpdateOperation =
        network::attribute::ResourceUpdateOperation;

    void update_node_resource(
        Vertex physical_node,
        ResourceAmount resource,
        ResourceUpdateOperation operation,
        bool safe = true);

    void update_node_resources(
        Vertex physical_node,
        const std::vector<ResourceAmount>& resources,
        ResourceUpdateOperation operation,
        bool safe = true);

    void update_link_resource(
        ConstraintLink physical_link,
        ResourceAmount resource,
        ResourceUpdateOperation operation,
        bool safe = true);

    void update_link_resources(
        ConstraintLink physical_link,
        const std::vector<ResourceAmount>& resources,
        ResourceUpdateOperation operation,
        bool safe = true);

    void update_path_resources(
        ConstraintLink virtual_link,
        const std::vector<Vertex>& physical_path,
        ResourceUpdateOperation operation,
        bool safe = true);

    void update_node_resources_batch(
        const std::vector<NodeResourceUpdateRequest>& requests,
        ResourceUpdateOperation operation,
        bool safe = true,
        std::size_t workers = 1U);

    void update_link_resources_batch(
        const std::vector<LinkResourceUpdateRequest>& requests,
        ResourceUpdateOperation operation,
        bool safe = true,
        std::size_t workers = 1U);

private:
    struct PreparedNodeResource
    {
        ResourceId resource_id = 0U;
        AttrId physical_value_id = 0U;
    };

    struct PreparedLinkResource
    {
        ResourceId resource_id = 0U;
        const network::attribute::LinkResourceAttribute* attribute = nullptr;
        AttrId virtual_value_id = 0U;
        AttrId physical_value_id = 0U;
    };

    PreparedResourceUpdator(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        std::vector<std::optional<PreparedNodeResource>> node_resources,
        std::vector<std::optional<PreparedLinkResource>> link_resources,
        std::vector<PreparedLinkResource> path_resources);

    const PreparedNodeResource& node_resource(
        ResourceId id,
        ResourceUpdatorOperation operation,
        std::optional<std::size_t> item_index = std::nullopt) const;

    const PreparedLinkResource& link_resource(
        ResourceId id,
        ResourceUpdatorOperation operation,
        std::optional<std::size_t> item_index = std::nullopt) const;

    void update_node_resource_impl(
        Vertex physical_node,
        const ResourceAmount& resource,
        ResourceUpdateOperation operation,
        bool safe,
        ResourceUpdatorOperation context,
        std::optional<std::size_t> item_index);

    void update_link_resource_impl(
        ConstraintLink physical_link,
        const ResourceAmount& resource,
        ResourceUpdateOperation operation,
        bool safe,
        ResourceUpdatorOperation context,
        std::optional<std::size_t> item_index);

    const network::VirtualNetwork* virtual_network_ = nullptr;
    network::PhysicalNetwork* physical_network_ = nullptr;
    std::vector<std::optional<PreparedNodeResource>> node_resources_;
    std::vector<std::optional<PreparedLinkResource>> link_resources_;
    std::vector<PreparedLinkResource> path_resources_;

    friend class ResourceUpdator;
};

} // namespace virne::core::controller
