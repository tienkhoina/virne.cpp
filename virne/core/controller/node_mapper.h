#pragma once

#include "resource_updator.h"

#include "physical_network.h"
#include "virtual_network.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::core::controller
{

enum class NodeMatchingMethod : std::uint8_t
{
    greedy,
    l2s2,
};

enum class NodeMapperErrorCode : std::uint8_t
{
    invalid_matching_method,
    invalid_node_resource_selection,
    missing_node_resource_value,
    non_numeric_node_resource,
    empty_hard_constraint_offsets,
    placement_not_found,
    placement_info_not_found,
    unsupported_constraint_violation_mapping,
    empty_physical_candidates,
    mapping_cardinality_mismatch,
};

enum class NodeMapperOperation : std::uint8_t
{
    prepare,
    place,
    record_violation,
    undo_place,
    node_mapping,
};

class NodeMapperException final : public std::runtime_error
{
public:
    NodeMapperException(
        NodeMapperErrorCode code,
        NodeMapperOperation operation,
        std::string message,
        std::optional<Vertex> virtual_node = std::nullopt,
        std::optional<Vertex> physical_node = std::nullopt,
        std::optional<ResourceId> resource_id = std::nullopt,
        std::optional<ConstraintId> constraint_id = std::nullopt);

    NodeMapperErrorCode code() const noexcept;
    NodeMapperOperation operation() const noexcept;
    const std::optional<Vertex>& virtual_node() const noexcept;
    const std::optional<Vertex>& physical_node() const noexcept;
    const std::optional<ResourceId>& resource_id() const noexcept;
    const std::optional<ConstraintId>& constraint_id() const noexcept;

private:
    NodeMapperErrorCode code_;
    NodeMapperOperation operation_;
    std::optional<Vertex> virtual_node_;
    std::optional<Vertex> physical_node_;
    std::optional<ResourceId> resource_id_;
    std::optional<ConstraintId> constraint_id_;
};

struct NodeMapperSelection
{
    std::vector<ConstraintId> node_constraints;
    std::vector<ResourceId> node_resources;
    std::vector<ConstraintId> hard_constraints;
};

struct NodePlacementOptions
{
    bool allow_constraint_violation = false;
    bool record_constraint_violation = true;
};

struct NodeMappingOptions
{
    bool reusable = false;
    bool inplace = true;
    NodeMatchingMethod method = NodeMatchingMethod::greedy;
    bool allow_constraint_violation = false;
    std::size_t candidate_workers = 1U;
};

struct NodePlacementResult
{
    bool placed = false;
    ConstraintCheckResult check;
};

class PreparedNodeMapper;

class NodeMapper
{
public:
    explicit NodeMapper(NodeMapperSelection selection);

    const NodeMapperSelection& selection() const noexcept;

    PreparedNodeMapper prepare(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network) const;

private:
    NodeMapperSelection selection_;
};

class PreparedNodeMapper
{
public:
    NodePlacementResult place(
        Vertex virtual_node,
        Vertex physical_node,
        Solution& solution,
        NodePlacementOptions options = {});

    void record_place_constraint_violation(
        Vertex virtual_node,
        const SolutionAttributeValues& offsets,
        Solution& solution) const;

    bool undo_place(
        Vertex virtual_node,
        Solution& solution);

    bool node_mapping(
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<Vertex>& physical_nodes,
        Solution& solution,
        NodeMappingOptions options = {});

private:
    struct PreparedNodeResource
    {
        ResourceId resource_id = 0U;
        AttrId virtual_value_id = 0U;
    };

    struct CandidateCheckOutcome
    {
        std::optional<ConstraintCheckResult> result;
        std::exception_ptr error;
    };

    PreparedNodeMapper(
        NodeMapperSelection selection,
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        PreparedConstraintChecker constraint_checker,
        PreparedResourceUpdator resource_updator,
        std::vector<PreparedNodeResource> node_resources,
        std::vector<ConstraintId> constraint_order,
        std::vector<std::uint8_t> hard_constraint_mask);

    const std::vector<ResourceAmount>& gather_node_resources(
        Vertex virtual_node,
        SolutionAttributeValues* recorded_values);

    NodePlacementResult commit_place_after_check(
        Vertex virtual_node,
        Vertex physical_node,
        Solution& solution,
        ConstraintCheckResult check,
        bool allow_constraint_violation);

    std::vector<CandidateCheckOutcome>& check_candidates_ordered(
        Vertex virtual_node,
        const std::vector<Vertex>& physical_nodes,
        std::size_t begin_index,
        std::size_t end_index,
        std::size_t workers);

    bool node_mapping_inplace(
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<Vertex>& physical_nodes,
        Solution& solution,
        const NodeMappingOptions& options,
        bool clear_solution);

    NodeMapperSelection selection_;
    const network::VirtualNetwork* virtual_network_ = nullptr;
    network::PhysicalNetwork* physical_network_ = nullptr;
    PreparedConstraintChecker constraint_checker_;
    PreparedResourceUpdator resource_updator_;
    std::vector<PreparedNodeResource> node_resources_;
    std::vector<ConstraintId> constraint_order_;
    std::vector<std::uint8_t> hard_constraint_mask_;
    std::vector<ResourceAmount> resource_scratch_;
    std::vector<CandidateCheckOutcome> candidate_check_scratch_;

    friend class NodeMapper;
};

} // namespace virne::core::controller
