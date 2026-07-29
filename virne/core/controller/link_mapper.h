#pragma once

#include "resource_updator.h"
#include "topology_analyzer.h"

#include "physical_network.h"
#include "virtual_network.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::core::controller
{

using PhysicalPath = std::vector<Vertex>;
using PhysicalPaths = std::vector<PhysicalPath>;
using LinkPathRanker = std::function<void(PhysicalPaths&)>;

enum class LinkMapperErrorCode : std::uint8_t
{
    invalid_link_resource_selection,
    missing_link_resource_value,
    non_numeric_link_resource,
    same_physical_node,
    unsupported_unsafe_shortest_method,
    empty_link_constraint_offsets,
    empty_hard_constraint_violations,
    route_not_found,
    route_info_not_found,
    missing_node_slot,
    invalid_stored_node,
    unsupported_unsafe_link_mapping,
    mapping_cardinality_mismatch,
};

enum class LinkMapperOperation : std::uint8_t
{
    prepare,
    route,
    record_violation,
    undo_route,
    link_mapping,
};

class LinkMapperException final : public std::runtime_error
{
public:
    LinkMapperException(
        LinkMapperErrorCode code,
        LinkMapperOperation operation,
        std::string message,
        std::optional<ConstraintLink> virtual_link = std::nullopt,
        std::optional<ConstraintLink> physical_link = std::nullopt,
        std::optional<ResourceId> resource_id = std::nullopt,
        std::optional<ConstraintId> constraint_id = std::nullopt);

    LinkMapperErrorCode code() const noexcept;
    LinkMapperOperation operation() const noexcept;
    const std::optional<ConstraintLink>& virtual_link() const noexcept;
    const std::optional<ConstraintLink>& physical_link() const noexcept;
    const std::optional<ResourceId>& resource_id() const noexcept;
    const std::optional<ConstraintId>& constraint_id() const noexcept;

private:
    LinkMapperErrorCode code_;
    LinkMapperOperation operation_;
    std::optional<ConstraintLink> virtual_link_;
    std::optional<ConstraintLink> physical_link_;
    std::optional<ResourceId> resource_id_;
    std::optional<ConstraintId> constraint_id_;
};

struct LinkMapperSelection
{
    std::vector<ConstraintId> link_constraints;
    std::vector<ConstraintId> path_constraints;
    std::vector<ResourceId> link_resources;
    std::vector<ConstraintId> hard_constraints;
    bool reusable = false;
};

struct LinkRouteOptions
{
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 1;
    double max_path_nodes = 1.0e6;
    std::size_t topology_constraint_workers = 1U;
    std::size_t candidate_workers = 1U;
    const LinkPathRanker* ranker = nullptr;
    bool allow_constraint_violation = false;
    bool record_constraint_violation = true;
};

struct LinkMappingOptions
{
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 10;
    double max_path_nodes = 1.0e6;
    std::size_t topology_constraint_workers = 1U;
    std::size_t candidate_workers = 1U;
    bool inplace = true;
    bool allow_constraint_violation = false;
};

struct LinkRouteCheckInfo
{
    bool placeholder = false;
    PathConstraintCheckResult constraints;
};

struct LinkRouteResult
{
    bool routed = false;
    LinkRouteCheckInfo check;
};

class PreparedLinkMapper;

class LinkMapper
{
public:
    explicit LinkMapper(LinkMapperSelection selection);

    const LinkMapperSelection& selection() const noexcept;

    PreparedLinkMapper prepare(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network) const;

private:
    LinkMapperSelection selection_;
};

class PreparedLinkMapper
{
public:
    LinkRouteResult route(
        ConstraintLink virtual_link,
        ConstraintLink physical_pair,
        Solution& solution,
        LinkRouteOptions options = {});

    void record_route_constraint_violation(
        ConstraintLink virtual_link,
        const LinkRouteCheckInfo& check,
        Solution& solution) const;

    bool undo_route(
        ConstraintLink virtual_link,
        Solution& solution);

    bool link_mapping(
        Solution& solution,
        LinkMappingOptions options = {});

    bool link_mapping(
        const std::vector<ConstraintLink>& virtual_links,
        Solution& solution,
        LinkMappingOptions options = {});

private:
    struct PreparedLinkResource
    {
        ResourceId resource_id = 0U;
        AttrId virtual_value_id = 0U;
    };

    struct PathCheckOutcome
    {
        std::optional<PathConstraintCheckResult> result;
        std::exception_ptr error;
    };

    struct PooledConstraintValues
    {
        SolutionAttributeValues link_offsets;
        SolutionAttributeValues link_violations;
        SolutionAttributeValues path_offsets;
        SolutionAttributeValues path_violations;
    };

    PreparedLinkMapper(
        LinkMapperSelection selection,
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        PreparedConstraintChecker constraint_checker,
        PreparedResourceUpdator resource_updator,
        PreparedTopologyAnalyzer topology_analyzer,
        std::vector<PreparedLinkResource> link_resources,
        std::vector<ConstraintId> link_constraint_order,
        std::vector<ConstraintId> path_constraint_order,
        std::vector<ConstraintId> combined_constraint_order,
        std::vector<std::uint8_t> hard_constraint_mask);

    LinkRouteCheckInfo placeholder_check() const;
    void clear_existing_route(
        ConstraintLink virtual_link,
        Solution& solution) const;
    std::vector<ResourceAmount> gather_link_resources(
        ConstraintLink virtual_link,
        SolutionAttributeValues* recorded_values) const;
    LinkRouteResult commit_path(
        ConstraintLink virtual_link,
        const PhysicalPath& path,
        LinkRouteCheckInfo check,
        Solution& solution,
        bool safe,
        std::size_t path_workers);
    LinkRouteResult safely_route(
        ConstraintLink virtual_link,
        ConstraintLink physical_pair,
        Solution& solution,
        const LinkRouteOptions& options);
    LinkRouteResult unsafely_route(
        ConstraintLink virtual_link,
        ConstraintLink physical_pair,
        Solution& solution,
        const LinkRouteOptions& options);
    std::vector<PathCheckOutcome> check_paths_ordered(
        ConstraintLink virtual_link,
        const PhysicalPaths& paths,
        std::size_t workers) const;
    PooledConstraintValues pool_constraints(
        const LinkRouteCheckInfo& check) const;
    long double violation_score(
        const LinkRouteCheckInfo& check) const;
    bool link_mapping_impl(
        const std::vector<ConstraintLink>& virtual_links,
        Solution& solution,
        const LinkMappingOptions& options,
        bool clear_solution);
    std::vector<ConstraintLink> all_virtual_links() const;
    ConstraintLink mapped_physical_pair(
        ConstraintLink virtual_link,
        const Solution& solution) const;

    LinkMapperSelection selection_;
    const network::VirtualNetwork* virtual_network_ = nullptr;
    network::PhysicalNetwork* physical_network_ = nullptr;
    PreparedConstraintChecker constraint_checker_;
    PreparedResourceUpdator resource_updator_;
    PreparedTopologyAnalyzer topology_analyzer_;
    std::vector<PreparedLinkResource> link_resources_;
    std::vector<ConstraintId> link_constraint_order_;
    std::vector<ConstraintId> path_constraint_order_;
    std::vector<ConstraintId> combined_constraint_order_;
    std::vector<std::uint8_t> hard_constraint_mask_;

    friend class LinkMapper;
};

} // namespace virne::core::controller
