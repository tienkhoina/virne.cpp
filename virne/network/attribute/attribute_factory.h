#pragma once

#include "graph_attribute.h"
#include "../../utils/setting.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace virne::network::attribute {

struct AttributeFactorySpec {
    std::string name;
    AttributeOwner owner = AttributeOwner::node;
    AttributeKind kind = AttributeKind::status;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    std::optional<std::string> originator_name;
    std::optional<AttributeDefinitionId> originator_id;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    std::optional<CheckingLevel> checking_level;
    double minimum_radius = 0.0;
    double maximum_radius = 1.0;
    LatencyGenerationKind latency_generation =
        LatencyGenerationKind::configured;
    AttributeNumber minimum = 0.0;
    AttributeNumber maximum = 1.0;
};

using AttributeRegistryId = AttributeDefinitionId;

inline constexpr AttributeRegistryId invalid_attribute_registry_id =
    std::numeric_limits<AttributeRegistryId>::max();

enum class AttributeFactoryErrorCode : std::uint8_t {
    missing_name,
    invalid_name,
    missing_owner,
    invalid_owner,
    missing_kind,
    invalid_kind,
    unsupported_pair,
    invalid_setting_item,
    invalid_setting_value,
    invalid_distribution,
    invalid_dtype,
    invalid_restriction,
    invalid_checking_level,
    invalid_numeric_field,
    family_mismatch,
    registry_id_range,
    invalid_registry_id,
};

enum class AttributeFactoryOperation : std::uint8_t {
    decode_name,
    decode_owner,
    decode_kind,
    validate_pair,
    decode_fields,
    construct_attribute,
    validate_family,
    decode_setting_list,
    build_registry,
    access_registry,
};

class AttributeFactoryException : public std::runtime_error {
public:
    AttributeFactoryException(
        AttributeFactoryErrorCode code,
        AttributeFactoryOperation operation,
        std::string message,
        std::optional<std::size_t> input_index = std::nullopt);

    AttributeFactoryErrorCode code() const noexcept;
    AttributeFactoryOperation operation() const noexcept;
    const std::optional<std::size_t>& input_index() const noexcept;

private:
    AttributeFactoryErrorCode code_;
    AttributeFactoryOperation operation_;
    std::optional<std::size_t> input_index_;
};

AttributeFactorySpec attribute_factory_spec_from_setting(
    const virne::utils::SettingObject& setting);

std::unique_ptr<BaseAttribute> create_attribute(AttributeFactorySpec spec);
std::unique_ptr<NodeAttribute> create_node_attribute(AttributeFactorySpec spec);
std::unique_ptr<LinkAttribute> create_link_attribute(AttributeFactorySpec spec);
std::unique_ptr<GraphAttribute> create_graph_attribute(AttributeFactorySpec spec);

struct AttributeRegistryEntry {
    std::string name;
    std::unique_ptr<BaseAttribute> attribute;
};

struct NodeAttributeRegistryEntry {
    std::string name;
    std::unique_ptr<NodeAttribute> attribute;
};

struct LinkAttributeRegistryEntry {
    std::string name;
    std::unique_ptr<LinkAttribute> attribute;
};

class AttributeRegistry {
public:
    AttributeRegistry() = default;
    AttributeRegistry(const AttributeRegistry&) = delete;
    AttributeRegistry& operator=(const AttributeRegistry&) = delete;
    AttributeRegistry(AttributeRegistry&&) noexcept = default;
    AttributeRegistry& operator=(AttributeRegistry&&) noexcept = default;

    std::optional<AttributeRegistryId> bind(std::string_view name) const;
    const BaseAttribute& at(AttributeRegistryId id) const;
    const BaseAttribute* find(std::string_view name) const;
    const std::vector<AttributeRegistryEntry>& entries() const noexcept;
    std::size_t size() const noexcept;

private:
    void reserve(std::size_t size);
    void insert_or_assign(
        std::string_view name,
        std::unique_ptr<BaseAttribute> attribute);

    std::vector<AttributeRegistryEntry> entries_;
    std::unordered_map<std::string_view, AttributeRegistryId> ids_;

    friend AttributeRegistry create_attributes_from_specs(
        const std::vector<AttributeFactorySpec>& specs,
        std::size_t workers);
};

class NodeAttributeRegistry {
public:
    NodeAttributeRegistry() = default;
    NodeAttributeRegistry(const NodeAttributeRegistry&) = delete;
    NodeAttributeRegistry& operator=(const NodeAttributeRegistry&) = delete;
    NodeAttributeRegistry(NodeAttributeRegistry&&) noexcept = default;
    NodeAttributeRegistry& operator=(NodeAttributeRegistry&&) noexcept = default;

    std::optional<AttributeRegistryId> bind(std::string_view name) const;
    const NodeAttribute& at(AttributeRegistryId id) const;
    const NodeAttribute* find(std::string_view name) const;
    const std::vector<NodeAttributeRegistryEntry>& entries() const noexcept;
    std::size_t size() const noexcept;

private:
    void reserve(std::size_t size);
    void insert_or_assign(
        std::string_view name,
        std::unique_ptr<NodeAttribute> attribute);

    std::vector<NodeAttributeRegistryEntry> entries_;
    std::unordered_map<std::string_view, AttributeRegistryId> ids_;

    friend NodeAttributeRegistry create_node_attributes_from_specs(
        const std::vector<AttributeFactorySpec>& specs,
        std::size_t workers);
};

class LinkAttributeRegistry {
public:
    LinkAttributeRegistry() = default;
    LinkAttributeRegistry(const LinkAttributeRegistry&) = delete;
    LinkAttributeRegistry& operator=(const LinkAttributeRegistry&) = delete;
    LinkAttributeRegistry(LinkAttributeRegistry&&) noexcept = default;
    LinkAttributeRegistry& operator=(LinkAttributeRegistry&&) noexcept = default;

    std::optional<AttributeRegistryId> bind(std::string_view name) const;
    const LinkAttribute& at(AttributeRegistryId id) const;
    const LinkAttribute* find(std::string_view name) const;
    const std::vector<LinkAttributeRegistryEntry>& entries() const noexcept;
    std::size_t size() const noexcept;

private:
    void reserve(std::size_t size);
    void insert_or_assign(
        std::string_view name,
        std::unique_ptr<LinkAttribute> attribute);

    std::vector<LinkAttributeRegistryEntry> entries_;
    std::unordered_map<std::string_view, AttributeRegistryId> ids_;

    friend LinkAttributeRegistry create_link_attributes_from_specs(
        const std::vector<AttributeFactorySpec>& specs,
        std::size_t workers);
};

AttributeRegistry create_attributes_from_specs(
    const std::vector<AttributeFactorySpec>& specs,
    std::size_t workers = 1U);

NodeAttributeRegistry create_node_attributes_from_specs(
    const std::vector<AttributeFactorySpec>& specs,
    std::size_t workers = 1U);

LinkAttributeRegistry create_link_attributes_from_specs(
    const std::vector<AttributeFactorySpec>& specs,
    std::size_t workers = 1U);

AttributeRegistry create_attributes_from_setting(
    const virne::utils::SettingList& settings,
    std::size_t workers = 1U);

NodeAttributeRegistry create_node_attributes_from_setting(
    const virne::utils::SettingList& settings,
    std::size_t workers = 1U);

LinkAttributeRegistry create_link_attributes_from_setting(
    const virne::utils::SettingList& settings,
    std::size_t workers = 1U);

}  // namespace virne::network::attribute
