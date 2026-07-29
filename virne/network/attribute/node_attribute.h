#pragma once

#include "base_attribute.h"
#include "graph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::network::attribute {

struct NodeAttributeBinding {
    AttrId value_id = 0U;
};

struct NodeAttributePairBinding {
    NodeAttributeBinding virtual_graph;
    NodeAttributeBinding physical_graph;
};

struct NodeAttributeAssignment {
    Vertex node = 0U;
    AttrValue value = std::int64_t{0};
};

enum class CheckingLevel : std::uint8_t {
    node,
    link,
    path,
    graph,
};

enum class NodeAttributeErrorCode : std::uint8_t {
    invalid_node_spec,
    node_out_of_range,
    missing_attribute,
    dense_data_too_short,
    missing_originator,
    missing_resource_value,
    non_numeric_resource,
    empty_position_network,
    missing_position_data,
};

enum class NodeAttributeOperation : std::uint8_t {
    construct,
    bind,
    get,
    set_data,
    get_data,
    generate_extrema,
    update_resource,
    check_resource,
    generate_position,
    get_existing_position,
};

class NodeAttributeException : public std::runtime_error {
public:
    NodeAttributeException(
        NodeAttributeErrorCode code,
        NodeAttributeOperation operation,
        std::string message);

    NodeAttributeErrorCode code() const noexcept;
    NodeAttributeOperation operation() const noexcept;

private:
    NodeAttributeErrorCode code_;
    NodeAttributeOperation operation_;
};

class NodeAttribute : public BaseAttribute {
public:
    explicit NodeAttribute(BaseAttributeSpec spec);

    NodeAttributeBinding bind(const Graph& graph) const;
    NodeAttributeBinding bind(const DiGraph& graph) const;

    const AttrValue& get(
        const Graph& graph,
        Vertex node,
        NodeAttributeBinding binding) const;
    const AttrValue& get(
        const DiGraph& graph,
        Vertex node,
        NodeAttributeBinding binding) const;

    void set_data(
        Graph& graph,
        const std::vector<NodeAttributeAssignment>& assignments,
        NodeAttributeBinding binding) const;
    void set_data(
        DiGraph& graph,
        const std::vector<NodeAttributeAssignment>& assignments,
        NodeAttributeBinding binding) const;

    void set_data_dense(
        Graph& graph,
        const std::vector<AttrValue>& values,
        NodeAttributeBinding binding,
        std::size_t workers = 1U) const;
    void set_data_dense(
        DiGraph& graph,
        const std::vector<AttrValue>& values,
        NodeAttributeBinding binding,
        std::size_t workers = 1U) const;

    std::vector<AttrValue> get_data(
        const Graph& graph,
        NodeAttributeBinding binding,
        std::size_t workers = 1U) const;
    std::vector<AttrValue> get_data(
        const DiGraph& graph,
        NodeAttributeBinding binding,
        std::size_t workers = 1U) const;
};

struct NodeStatusSpec {
    std::string name = "status";
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
};

class NodeStatusAttribute final : public NodeAttribute {
public:
    explicit NodeStatusAttribute(NodeStatusSpec spec = {});
};

struct NodeExtremaSpec {
    std::string name;
    std::optional<std::string> originator_name;
    AttributeDefinitionId originator_id = 0U;
};

class NodeExtremaAttribute final : public NodeAttribute {
public:
    explicit NodeExtremaAttribute(NodeExtremaSpec spec);

    std::string_view originator_name() const noexcept;
    AttributeDefinitionId originator_id() const noexcept;

    std::vector<AttrValue> generate_from_resolved_originator(
        const Graph& graph,
        const NodeAttribute& originator,
        NodeAttributeBinding originator_binding,
        std::size_t workers = 1U) const;
    std::vector<AttrValue> generate_from_resolved_originator(
        const DiGraph& graph,
        const NodeAttribute& originator,
        NodeAttributeBinding originator_binding,
        std::size_t workers = 1U) const;

private:
    std::string originator_name_;
    AttributeDefinitionId originator_id_ = 0U;
};

struct NodeResourceSpec {
    std::string name;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    CheckingLevel checking_level = CheckingLevel::node;
};

class NodeResourceAttribute final : public NodeAttribute {
public:
    explicit NodeResourceAttribute(NodeResourceSpec spec);

    ConstraintRestriction restriction() const noexcept;
    CheckingLevel checking_level() const noexcept;

    bool update(
        const AttrMap& virtual_node,
        AttrId virtual_id,
        AttrMap& physical_node,
        AttrId physical_id,
        ResourceUpdateOperation operation,
        bool safe = true) const;

    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_node,
        AttrId virtual_id,
        const AttrMap& physical_node,
        AttrId physical_id,
        ComparisonOperation method = ComparisonOperation::less_equal) const;

private:
    ConstraintRestriction restriction_ = ConstraintRestriction::hard;
    CheckingLevel checking_level_ = CheckingLevel::node;
};

struct NodePositionSpec {
    std::string name = "pos";
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    double minimum_radius = 0.0;
    double maximum_radius = 1.0;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
};

struct NodePositionValue {
    AttributeNumber x = std::int64_t{0};
    AttributeNumber y = std::int64_t{0};
    double radius = 0.0;
};

class NodePositionAttribute final : public NodeAttribute {
public:
    explicit NodePositionAttribute(NodePositionSpec spec = {});

    double minimum_radius() const noexcept;
    double maximum_radius() const noexcept;
    ConstraintRestriction restriction() const noexcept;

    std::vector<NodePositionValue> generate_positions(
        const NetworkCardinality& network,
        NumpyRandomState& rng,
        std::size_t workers = 1U) const;

    NodeAttributeBinding bind_existing_pos(const Graph& graph) const;
    NodeAttributeBinding bind_existing_pos(const DiGraph& graph) const;
    std::vector<AttrValue> get_existing_pos_data(
        const Graph& graph,
        NodeAttributeBinding literal_pos_binding,
        std::size_t workers = 1U) const;
    std::vector<AttrValue> get_existing_pos_data(
        const DiGraph& graph,
        NodeAttributeBinding literal_pos_binding,
        std::size_t workers = 1U) const;

private:
    double minimum_radius_ = 0.0;
    double maximum_radius_ = 1.0;
    ConstraintRestriction restriction_ = ConstraintRestriction::hard;
};

}  // namespace virne::network::attribute
