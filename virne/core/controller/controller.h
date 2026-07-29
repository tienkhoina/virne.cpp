#pragma once

#include "link_mapper.h"
#include "node_mapper.h"

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

enum class ControllerErrorCode : std::uint8_t
{
    missing_node_slot,
    invalid_stored_node,
    missing_constraint_offset,
    missing_node_slot_info,
    missing_link_path_info,
};

enum class ControllerOperation : std::uint8_t
{
    prepare,
    place_and_route,
    pool_step_offsets,
    undo_place_and_route,
    deploy,
    release,
    undo_deploy,
};

class ControllerException final : public std::runtime_error
{
public:
    ControllerException(
        ControllerErrorCode code,
        ControllerOperation operation,
        std::string message,
        std::optional<Vertex> virtual_node = std::nullopt,
        std::optional<Vertex> physical_node = std::nullopt,
        std::optional<ConstraintLink> virtual_link = std::nullopt,
        std::optional<ConstraintLink> physical_link = std::nullopt,
        std::optional<ConstraintId> constraint_id = std::nullopt);

    ControllerErrorCode code() const noexcept;
    ControllerOperation operation() const noexcept;
    const std::optional<Vertex>& virtual_node() const noexcept;
    const std::optional<Vertex>& physical_node() const noexcept;
    const std::optional<ConstraintLink>& virtual_link() const noexcept;
    const std::optional<ConstraintLink>& physical_link() const noexcept;
    const std::optional<ConstraintId>& constraint_id() const noexcept;

private:
    ControllerErrorCode code_;
    ControllerOperation operation_;
    std::optional<Vertex> virtual_node_;
    std::optional<Vertex> physical_node_;
    std::optional<ConstraintLink> virtual_link_;
    std::optional<ConstraintLink> physical_link_;
    std::optional<ConstraintId> constraint_id_;
};

struct ControllerSelection
{
    ConstraintCheckerSelection constraints;
    std::vector<ResourceId> node_resources;
    std::vector<ResourceId> link_resources;
    std::vector<ConstraintId> hard_node_constraints;
    std::vector<ConstraintId> hard_link_constraints;
    bool reusable = false;
};

struct ControllerWorkers
{
    std::size_t topology_constraint_workers = 1U;
    std::size_t candidate_workers = 1U;
};

struct PlaceAndRouteOptions
{
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 1;
    double max_path_nodes = 1.0e6;
    ControllerWorkers workers;
    bool allow_constraint_violation = false;
};

enum class ControllerFailurePhase : std::uint8_t
{
    none,
    place,
    route,
};

struct PlaceAndRouteResult
{
    bool succeeded = false;
    ControllerFailurePhase failure_phase = ControllerFailurePhase::none;
    NodePlacementResult placement;
    std::optional<LinkRouteResult> last_route;
    std::size_t attempted_routes = 0U;
};

struct ControllerMutationOptions
{
    std::size_t workers = 1U;
};

class PreparedController;

class Controller
{
public:
    explicit Controller(ControllerSelection selection);

    const ControllerSelection& selection() const noexcept;

    PreparedController prepare(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network) const;

private:
    ControllerSelection selection_;
    ResourceUpdator resource_updator_;
    NodeMapper node_mapper_;
    LinkMapper link_mapper_;
};

class PreparedController
{
public:
    PlaceAndRouteResult place_and_route(
        Vertex virtual_node,
        Vertex physical_node,
        Solution& solution,
        PlaceAndRouteOptions options = {});

    bool undo_place_and_route(
        Vertex virtual_node,
        Solution& solution);

    bool deploy(
        Solution& solution,
        ControllerMutationOptions options = {});

    bool release(
        const Solution& solution,
        ControllerMutationOptions options = {});

    bool undo_deploy(
        const Solution& solution,
        ControllerMutationOptions options = {});

private:
    PreparedController(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network,
        PreparedResourceUpdator resource_updator,
        PreparedNodeMapper node_mapper,
        PreparedLinkMapper link_mapper,
        std::vector<ResourceId> node_resource_order,
        std::vector<ResourceId> link_resource_order,
        std::vector<ConstraintId> node_constraint_order,
        std::vector<ConstraintId> link_constraint_order,
        std::vector<ConstraintId> path_constraint_order,
        std::vector<std::uint8_t> hard_node_mask,
        std::vector<std::uint8_t> hard_link_mask,
        std::vector<std::optional<AttrId>> node_physical_value_ids,
        std::vector<std::optional<AttrId>> link_physical_value_ids,
        SolutionAttributeValues link_step_placeholder,
        SolutionAttributeValues path_step_placeholder);

    PlaceAndRouteResult safely_place_and_route(
        Vertex virtual_node,
        Vertex physical_node,
        Solution& solution,
        const PlaceAndRouteOptions& options);

    PlaceAndRouteResult unsafely_place_and_route(
        Vertex virtual_node,
        Vertex physical_node,
        Solution& solution,
        const PlaceAndRouteOptions& options);

    std::vector<ConstraintLink> links_to_route(
        Vertex virtual_node,
        const Solution& solution) const;

    void pool_step_offsets(
        const std::vector<ConstraintLink>& attempted_links,
        Solution& solution) const;

    void calculate_max_single_step_constraint_violation(
        Solution& solution) const;

    std::vector<ResourceAmount> gather_resources(
        const SolutionAttributeValues& values,
        const std::vector<ResourceId>& resource_order) const;

    void apply_node_values(
        Vertex physical_node,
        const SolutionAttributeValues& values,
        network::attribute::ResourceUpdateOperation operation);

    void apply_link_values(
        ConstraintLink physical_link,
        const SolutionAttributeValues& values,
        network::attribute::ResourceUpdateOperation operation);

    void apply_node_requests(
        const std::vector<NodeResourceUpdateRequest>& requests,
        network::attribute::ResourceUpdateOperation operation,
        std::size_t workers);

    void apply_link_requests(
        const std::vector<LinkResourceUpdateRequest>& requests,
        network::attribute::ResourceUpdateOperation operation,
        std::size_t workers);

    bool node_targets_are_disjoint(
        const std::vector<NodeResourceUpdateRequest>& requests) const;

    bool link_targets_are_disjoint(
        const std::vector<LinkResourceUpdateRequest>& requests) const;

    bool try_deploy_parallel(
        const Solution& solution,
        std::size_t workers);

    bool try_release_parallel(
        const Solution& solution,
        std::size_t workers);

    const network::VirtualNetwork* virtual_network_ = nullptr;
    network::PhysicalNetwork* physical_network_ = nullptr;
    PreparedResourceUpdator resource_updator_;
    PreparedNodeMapper node_mapper_;
    PreparedLinkMapper link_mapper_;
    std::vector<ResourceId> node_resource_order_;
    std::vector<ResourceId> link_resource_order_;
    std::vector<ConstraintId> node_constraint_order_;
    std::vector<ConstraintId> link_constraint_order_;
    std::vector<ConstraintId> path_constraint_order_;
    std::vector<std::uint8_t> hard_node_mask_;
    std::vector<std::uint8_t> hard_link_mask_;
    std::vector<std::optional<AttrId>> node_physical_value_ids_;
    std::vector<std::optional<AttrId>> link_physical_value_ids_;
    SolutionAttributeValues link_step_placeholder_;
    SolutionAttributeValues path_step_placeholder_;

    friend class Controller;
};

} // namespace virne::core::controller
