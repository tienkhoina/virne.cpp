#pragma once

#include "attribute_method.h"
#include "../../utils/dataset.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

class NumpyRandomState;

namespace virne::network::attribute {

struct NetworkCardinality {
    std::size_t num_nodes = 0U;
    std::size_t num_links = 0U;
};

struct BaseAttributeSpec {
    std::string name;
    AttributeOwner owner = AttributeOwner::node;
    AttributeKind kind = AttributeKind::status;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    std::optional<std::string> originator;
    std::optional<bool> is_constraint;
};

struct BaseAttributeSnapshotEntry {
    std::string name;
    virne::utils::DatasetScalar value;
};

using BaseAttributeSnapshot = std::vector<BaseAttributeSnapshotEntry>;

enum class BaseAttributeErrorCode : std::uint8_t {
    invalid_owner,
    invalid_kind,
    not_implemented,
    not_generative,
    unsupported_distribution,
    invalid_custom_range,
};

enum class BaseAttributeOperation : std::uint8_t {
    resolve_owner,
    resolve_kind,
    generate_data,
    update_data,
    generate_configured_data,
};

class BaseAttributeException : public std::runtime_error {
public:
    BaseAttributeException(
        BaseAttributeErrorCode code,
        BaseAttributeOperation operation,
        std::string message);

    BaseAttributeErrorCode code() const noexcept;
    BaseAttributeOperation operation() const noexcept;

private:
    BaseAttributeErrorCode code_;
    BaseAttributeOperation operation_;
};

AttributeOwner attribute_owner_from_string(std::string_view value);
std::string_view attribute_owner_name(AttributeOwner value) noexcept;

AttributeKind attribute_kind_from_string(std::string_view value);
std::string_view attribute_kind_name(AttributeKind value) noexcept;

class BaseAttribute {
public:
    explicit BaseAttribute(BaseAttributeSpec spec);
    virtual ~BaseAttribute() = default;

    const BaseAttributeSpec& spec() const noexcept;

    virtual virne::utils::GeneratedData generate_data() const;
    virtual void update_data();

    virne::utils::GeneratedData generate_configured_data(
        const NetworkCardinality& network,
        NumpyRandomState& rng,
        std::size_t workers = 1U) const;

    BaseAttributeSnapshot to_dict() const;
    std::string repr(std::string_view class_name = "BaseAttribute") const;

private:
    BaseAttributeSpec spec_;
};

}  // namespace virne::network::attribute
