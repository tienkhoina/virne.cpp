#pragma once

#include "node_attribute.h"
#include "distance_matrix.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace virne::network::attribute {

struct LinkAttributeBinding {
    AttrId value_id = 0U;
};

struct LinkAttributePairBinding {
    LinkAttributeBinding virtual_graph;
    LinkAttributeBinding physical_graph;
};

struct LinkAttributeAssignment {
    Vertex source = 0U;
    Vertex target = 0U;
    AttrValue value = std::int64_t{0};
};

enum class LinkAggregation : std::uint8_t {
    sum,
    mean,
    maximum,
    minimum,
};

enum class LatencyGenerationKind : std::uint8_t {
    configured,
    position,
};

enum class LinkAttributeErrorCode : std::uint8_t {
    invalid_link_spec,
    edge_not_found,
    missing_attribute,
    dense_data_too_short,
    unsupported_aggregation,
    update_path_not_implemented,
    missing_originator,
    missing_resource_value,
    non_numeric_resource,
    path_too_short,
    empty_position_network,
    missing_position_data,
    invalid_position_data,
    non_generative_latency,
    invalid_latency_generation,
    empty_aggregation,
};

enum class LinkAttributeOperation : std::uint8_t {
    construct,
    resolve_aggregation,
    bind,
    get,
    set_data,
    get_data,
    adjacency,
    aggregation,
    generate_extrema,
    update_resource,
    update_path,
    check_resource,
    check_latency,
    resolve_position,
    generate_latency,
};

class LinkAttributeException : public std::runtime_error {
public:
    LinkAttributeException(
        LinkAttributeErrorCode code,
        LinkAttributeOperation operation,
        std::string message);

    LinkAttributeErrorCode code() const noexcept;
    LinkAttributeOperation operation() const noexcept;

private:
    LinkAttributeErrorCode code_;
    LinkAttributeOperation operation_;
};

LinkAggregation link_aggregation_from_string(std::string_view value);
std::string_view link_aggregation_name(LinkAggregation value) noexcept;

class LinkAttribute : public BaseAttribute {
public:
    explicit LinkAttribute(BaseAttributeSpec spec);

    LinkAttributeBinding bind(const Graph& graph) const;
    LinkAttributeBinding bind(const DiGraph& graph) const;

    const AttrValue& get(
        const Graph& graph,
        Vertex source,
        Vertex target,
        LinkAttributeBinding binding) const;
    const AttrValue& get(
        const DiGraph& graph,
        Vertex source,
        Vertex target,
        LinkAttributeBinding binding) const;

    void set_data(
        Graph& graph,
        const std::vector<LinkAttributeAssignment>& assignments,
        LinkAttributeBinding binding) const;
    void set_data(
        DiGraph& graph,
        const std::vector<LinkAttributeAssignment>& assignments,
        LinkAttributeBinding binding) const;

    void set_data_dense(
        Graph& graph,
        const std::vector<AttrValue>& values,
        LinkAttributeBinding binding,
        std::size_t workers = 1U) const;
    void set_data_dense(
        DiGraph& graph,
        const std::vector<AttrValue>& values,
        LinkAttributeBinding binding,
        std::size_t workers = 1U) const;

    std::vector<AttrValue> get_data(
        const Graph& graph,
        LinkAttributeBinding binding,
        std::size_t workers = 1U) const;
    std::vector<AttrValue> get_data(
        const DiGraph& graph,
        LinkAttributeBinding binding,
        std::size_t workers = 1U) const;

    DistanceMatrix get_adjacency_data(
        const Graph& graph,
        LinkAttributeBinding binding,
        bool normalized = false,
        std::size_t workers = 1U) const;
    DistanceMatrix get_adjacency_data(
        const DiGraph& graph,
        LinkAttributeBinding binding,
        bool normalized = false,
        std::size_t workers = 1U) const;

    std::vector<double> get_aggregation_data(
        const Graph& graph,
        LinkAttributeBinding binding,
        LinkAggregation aggregation = LinkAggregation::sum,
        bool normalized = false,
        std::size_t workers = 1U) const;
    std::vector<double> get_aggregation_data(
        const DiGraph& graph,
        LinkAttributeBinding binding,
        LinkAggregation aggregation = LinkAggregation::sum,
        bool normalized = false,
        std::size_t workers = 1U) const;

    virtual bool update_path() const;
};

struct LinkStatusSpec {
    std::string name = "status";
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
};

class LinkStatusAttribute final : public LinkAttribute {
public:
    explicit LinkStatusAttribute(LinkStatusSpec spec = {});
};

struct LinkExtremaSpec {
    std::string name;
    std::optional<std::string> originator_name;
    AttributeDefinitionId originator_id = 0U;
};

class LinkExtremaAttribute final : public LinkAttribute {
public:
    explicit LinkExtremaAttribute(LinkExtremaSpec spec);

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

struct LinkResourceSpec {
    std::string name;
    bool generative = false;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    CheckingLevel checking_level = CheckingLevel::link;
};

class LinkResourceAttribute final : public LinkAttribute {
public:
    explicit LinkResourceAttribute(LinkResourceSpec spec);

    ConstraintRestriction restriction() const noexcept;
    CheckingLevel checking_level() const noexcept;

    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_link,
        AttrId virtual_id,
        const AttrMap& physical_link,
        AttrId physical_id,
        ComparisonOperation method = ComparisonOperation::less_equal) const;

    bool update_path(
        const AttrMap& virtual_link,
        AttrId virtual_id,
        Graph& physical_graph,
        const std::vector<Vertex>& path,
        LinkAttributeBinding physical_binding,
        ResourceUpdateOperation operation,
        bool safe = true) const;
    bool update_path(
        const AttrMap& virtual_link,
        AttrId virtual_id,
        DiGraph& physical_graph,
        const std::vector<Vertex>& path,
        LinkAttributeBinding physical_binding,
        ResourceUpdateOperation operation,
        bool safe = true) const;

private:
    ConstraintRestriction restriction_ = ConstraintRestriction::hard;
    CheckingLevel checking_level_ = CheckingLevel::link;
};

struct LinkLatencySpec {
    std::string name = "latency";
    bool generative = false;
    LatencyGenerationKind generation = LatencyGenerationKind::configured;
    virne::utils::DistributionSpec distribution;
    std::optional<virne::utils::DatasetValueKind> dtype;
    AttributeNumber minimum = 0.0;
    AttributeNumber maximum = 1.0;
    ConstraintRestriction restriction = ConstraintRestriction::hard;
    CheckingLevel checking_level = CheckingLevel::path;
};

class LinkLatencyAttribute final : public LinkAttribute {
public:
    explicit LinkLatencyAttribute(LinkLatencySpec spec = {});

    LatencyGenerationKind generation_kind() const noexcept;
    const AttributeNumber& minimum() const noexcept;
    const AttributeNumber& maximum() const noexcept;
    ConstraintRestriction restriction() const noexcept;
    CheckingLevel checking_level() const noexcept;

    SatisfiabilityResult check_constraint_satisfiability(
        const AttrMap& virtual_link,
        AttrId virtual_id,
        const std::vector<const AttrMap*>& physical_path,
        AttrId physical_id,
        ComparisonOperation method = ComparisonOperation::greater_equal) const;

    NodeAttributeBinding resolve_position_binding(const Graph& graph) const;
    NodeAttributeBinding resolve_position_binding(const DiGraph& graph) const;
    std::vector<double> generate_from_position(
        const Graph& graph,
        NodeAttributeBinding position_binding,
        std::size_t workers = 1U) const;
    std::vector<double> generate_from_position(
        const DiGraph& graph,
        NodeAttributeBinding position_binding,
        std::size_t workers = 1U) const;

private:
    LatencyGenerationKind generation_ = LatencyGenerationKind::configured;
    AttributeNumber minimum_ = 0.0;
    AttributeNumber maximum_ = 1.0;
    ConstraintRestriction restriction_ = ConstraintRestriction::hard;
    CheckingLevel checking_level_ = CheckingLevel::path;
};

}  // namespace virne::network::attribute
