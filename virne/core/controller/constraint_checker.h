#pragma once

#include "../solution.h"

#include "attribute/graph_attribute.h"
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

using ConstraintId = network::attribute::AttributeRegistryId;

inline constexpr ConstraintId invalid_constraint_id =
    network::attribute::invalid_attribute_registry_id;

enum class ConstraintCheckerErrorCode : std::uint8_t
{
    invalid_node_selection,
    invalid_link_selection,
    invalid_path_selection,
    invalid_graph_selection,
    null_graph_attribute,
    physical_node_out_of_range,
    virtual_node_out_of_range,
    virtual_link_not_found,
    physical_link_not_found,
    invalid_path,
};

enum class ConstraintCheckerOperation : std::uint8_t
{
    prepare,
    check_graph,
    check_node,
    check_link,
    check_path,
};

class ConstraintCheckerException : public std::runtime_error
{
public:
    ConstraintCheckerException(
        ConstraintCheckerErrorCode code,
        ConstraintCheckerOperation operation,
        std::string message,
        std::optional<std::size_t> request_index = std::nullopt,
        std::optional<std::size_t> item_index = std::nullopt,
        std::optional<ConstraintId> constraint_id = std::nullopt);

    ConstraintCheckerErrorCode code() const noexcept;
    ConstraintCheckerOperation operation() const noexcept;
    const std::optional<std::size_t>& request_index() const noexcept;
    const std::optional<std::size_t>& item_index() const noexcept;
    const std::optional<ConstraintId>& constraint_id() const noexcept;

private:
    ConstraintCheckerErrorCode code_;
    ConstraintCheckerOperation operation_;
    std::optional<std::size_t> request_index_;
    std::optional<std::size_t> item_index_;
    std::optional<ConstraintId> constraint_id_;
};

struct ConstraintLink
{
    Vertex source = 0U;
    Vertex target = 0U;

    friend bool operator==(
        const ConstraintLink& left,
        const ConstraintLink& right) noexcept
    {
        return left.source == right.source && left.target == right.target;
    }
};

struct GraphConstraintSelection
{
    ConstraintId output_id = 0U;
    const network::attribute::GraphResourceAttribute* attribute = nullptr;
};

struct ConstraintCheckerSelection
{
    std::vector<ConstraintId> node_at_node;
    std::vector<ConstraintId> link_at_link;
    std::vector<ConstraintId> link_at_path;
    std::vector<GraphConstraintSelection> graph;
};

struct ConstraintCheckResult
{
    bool feasible = true;
    SolutionAttributeValues offsets;
};

struct PhysicalLinkConstraintResult
{
    ConstraintLink physical_link;
    SolutionAttributeValues offsets;
};

struct PathConstraintCheckResult
{
    bool feasible = true;
    std::vector<PhysicalLinkConstraintResult> link_level;
    SolutionAttributeValues path_level;
};

struct NodeConstraintRequest
{
    Vertex virtual_node = 0U;
    Vertex physical_node = 0U;
};

struct LinkConstraintRequest
{
    ConstraintLink virtual_link;
    ConstraintLink physical_link;
};

struct PathConstraintRequest
{
    ConstraintLink virtual_link;
    std::vector<Vertex> physical_path;
};

class PreparedConstraintChecker;

class ConstraintChecker
{
public:
    explicit ConstraintChecker(ConstraintCheckerSelection selection);

    const ConstraintCheckerSelection& selection() const noexcept;

    PreparedConstraintChecker prepare(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network) const;

private:
    ConstraintCheckerSelection selection_;
};

class PreparedConstraintChecker
{
public:
    ConstraintCheckResult check_graph_constraints() const;

    ConstraintCheckResult check_node_level_constraints(
        Vertex virtual_node,
        Vertex physical_node) const;

    ConstraintCheckResult check_link_level_constraints(
        ConstraintLink virtual_link,
        ConstraintLink physical_link) const;

    PathConstraintCheckResult check_path_level_constraints(
        ConstraintLink virtual_link,
        const std::vector<Vertex>& physical_path) const;

    std::vector<ConstraintCheckResult>
    check_node_level_constraints_batch(
        const std::vector<NodeConstraintRequest>& requests,
        std::size_t workers = 1U) const;

    std::vector<ConstraintCheckResult>
    check_link_level_constraints_batch(
        const std::vector<LinkConstraintRequest>& requests,
        std::size_t workers = 1U) const;

    std::vector<PathConstraintCheckResult>
    check_path_level_constraints_batch(
        const std::vector<PathConstraintRequest>& requests,
        std::size_t workers = 1U) const;

private:
    struct PreparedNodeConstraint
    {
        ConstraintId output_id = 0U;
        const network::attribute::NodeResourceAttribute* attribute = nullptr;
        AttrId virtual_value_id = 0U;
        AttrId physical_value_id = 0U;
    };

    struct PreparedLinkConstraint
    {
        ConstraintId output_id = 0U;
        const network::attribute::LinkResourceAttribute* attribute = nullptr;
        AttrId virtual_value_id = 0U;
        AttrId physical_value_id = 0U;
    };

    struct PreparedPathConstraint
    {
        ConstraintId output_id = 0U;
        const network::attribute::LinkLatencyAttribute* attribute = nullptr;
        AttrId virtual_value_id = 0U;
        AttrId physical_value_id = 0U;
    };

    struct PreparedGraphConstraint
    {
        ConstraintId output_id = 0U;
        const network::attribute::GraphResourceAttribute* attribute = nullptr;
        network::attribute::GraphAttributeBinding virtual_binding;
        network::attribute::GraphAttributeBinding physical_binding;
    };

    PreparedConstraintChecker(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        std::vector<PreparedNodeConstraint> node_constraints,
        std::vector<PreparedLinkConstraint> link_constraints,
        std::vector<PreparedPathConstraint> path_constraints,
        std::vector<PreparedGraphConstraint> graph_constraints);

    ConstraintCheckResult check_link_values(
        const AttrMap& virtual_link,
        const AttrMap& physical_link) const;

    const network::VirtualNetwork* virtual_network_ = nullptr;
    const network::PhysicalNetwork* physical_network_ = nullptr;
    std::vector<PreparedNodeConstraint> node_constraints_;
    std::vector<PreparedLinkConstraint> link_constraints_;
    std::vector<PreparedPathConstraint> path_constraints_;
    std::vector<PreparedGraphConstraint> graph_constraints_;

    friend class ConstraintChecker;
};

} // namespace virne::core::controller
