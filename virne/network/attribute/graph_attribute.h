#pragma once

#include "link_attribute.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace virne::network::attribute {

class GraphAttribute;

struct GraphAttributeBinding {
    AttrId value_id = 0U;
    const AttrMap* graph_identity = nullptr;
    const GraphAttribute* definition_identity = nullptr;
};

struct GraphAttributePairBinding {
    GraphAttributeBinding virtual_graph;
    GraphAttributeBinding physical_graph;
};

struct GraphAttributeConstSlot {
    const AttrMap* graph_attributes = nullptr;
    GraphAttributeBinding binding;
};

struct GraphAttributeMutableSlot {
    AttrMap* graph_attributes = nullptr;
    GraphAttributeBinding binding;
};

enum class GraphAttributeErrorCode : std::uint8_t {
    invalid_graph_spec,
    missing_attribute,
    invalid_binding,
    null_batch_slot,
    invalid_batch_shape,
    missing_originator,
    missing_resource_value,
    non_numeric_resource,
};

enum class GraphAttributeOperation : std::uint8_t {
    construct,
    bind,
    get,
    get_data,
    set_data,
    get_data_batch,
    set_data_batch,
    generate_extrema,
    update_resource,
    check_resource,
};

class GraphAttributeException : public std::runtime_error {
public:
    GraphAttributeException(
        GraphAttributeErrorCode code,
        GraphAttributeOperation operation,
        std::string message);

    GraphAttributeErrorCode code() const noexcept;
    GraphAttributeOperation operation() const noexcept;

private:
    GraphAttributeErrorCode code_;
    GraphAttributeOperation operation_;
};

class GraphAttribute : public BaseAttribute {
public:
    explicit GraphAttribute(BaseAttributeSpec spec);

    GraphAttributeBinding bind(const Graph& graph) const;
    GraphAttributeBinding bind(const DiGraph& graph) const;

    const AttrValue& get(
        const Graph& graph,
        GraphAttributeBinding binding) const;
    const AttrValue& get(
        const DiGraph& graph,
        GraphAttributeBinding binding) const;

    const AttrValue& get_data(
        const Graph& graph,
        GraphAttributeBinding binding) const;
    const AttrValue& get_data(
        const DiGraph& graph,
        GraphAttributeBinding binding) const;

    void set_data(
        Graph& graph,
        const AttrValue& value,
        GraphAttributeBinding binding) const;
    void set_data(
        DiGraph& graph,
        const AttrValue& value,
        GraphAttributeBinding binding) const;

    std::vector<AttrValue> get_data_batch(
        const std::vector<GraphAttributeConstSlot>& slots,
        std::size_t workers = 1U) const;

    void set_data_batch(
        const std::vector<GraphAttributeMutableSlot>& slots,
        const std::vector<AttrValue>& values,
        std::size_t workers = 1U) const;
};

struct GraphStatusSpec {
    std::string name = "status";
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
};

class GraphStatusAttribute final : public GraphAttribute {
public:
    explicit GraphStatusAttribute(GraphStatusSpec spec = {});
};

struct GraphExtremaSpec {
    std::string name;
    std::optional<std::string> originator_name;
    AttributeDefinitionId originator_id = 0U;
};

class GraphExtremaAttribute final : public GraphAttribute {
public:
    explicit GraphExtremaAttribute(GraphExtremaSpec spec);

    std::string_view originator_name() const noexcept;
    AttributeDefinitionId originator_id() const noexcept;

    std::vector<AttrValue> generate_from_resolved_originator(
        const Graph& graph,
        const LinkAttribute& originator,
        LinkAttributeBinding originator_binding,
        std::size_t workers = 1U) const;
    std::vector<AttrValue> generate_from_resolved_originator(
        const DiGraph& graph,
        const LinkAttribute& originator,
        LinkAttributeBinding originator_binding,
        std::size_t workers = 1U) const;

private:
    std::string originator_name_;
    AttributeDefinitionId originator_id_ = 0U;
};

struct GraphResourceSpec {
    std::string name;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    CheckingLevel checking_level = CheckingLevel::graph;
};

class GraphResourceAttribute final : public GraphAttribute {
public:
    explicit GraphResourceAttribute(GraphResourceSpec spec);

    ConstraintRestriction restriction() const noexcept;
    CheckingLevel checking_level() const noexcept;

    bool update(
        AttrMap& target,
        AttrId target_id,
        const AttrMap& operand,
        AttrId operand_id,
        ResourceUpdateOperation operation,
        bool safe = true) const;

    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_graph,
        AttrId virtual_id,
        const AttrMap& physical_graph,
        AttrId physical_id,
        ComparisonOperation method = ComparisonOperation::less_equal) const;

private:
    ConstraintRestriction restriction_ = ConstraintRestriction::hard;
    CheckingLevel checking_level_ = CheckingLevel::graph;
};

}  // namespace virne::network::attribute
