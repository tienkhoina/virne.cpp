#pragma once

#include "link_mapper.h"
#include "node_mapper.h"

#include "physical_network.h"
#include "virtual_network.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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
    transaction_already_active,
};

enum class ControllerOperation : std::uint8_t
{
    prepare,
    place_and_route,
    pool_step_offsets,
    undo_place_and_route,
    begin_transaction,
    commit_transaction,
    deploy,
    release,
    rollback,
    undo_deploy,
    deploy_with_node_slots,
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

// Safe whole-request evaluation for a caller-supplied, insertion-ordered
// virtual-to-physical node assignment. The prepared controller is already
// bound to its mutable physical network, so both mapping phases are inplace.
struct DeployWithNodeSlotsOptions
{
    ShortestPathMethod shortest_method = ShortestPathMethod::bfs_shortest;
    std::int64_t k = 10;
    double max_path_nodes = 1.0e6;
    ControllerWorkers workers;
};

class PreparedController;
class PreparedControllerMutation;

class Controller
{
public:
    explicit Controller(ControllerSelection selection);

    const ControllerSelection& selection() const noexcept;

    PreparedController prepare(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network) const;

    // Binds only numeric resource IDs and the resource updater. Environment
    // deploy/release and solver rollback do not need constraint checkers,
    // node/link mappers, or topology analysis.
    PreparedControllerMutation prepare_mutation(
        const network::VirtualNetwork& virtual_network,
        network::PhysicalNetwork& physical_network) const;

private:
    ControllerSelection selection_;
    ResourceUpdator resource_updator_;
    NodeMapper node_mapper_;
    LinkMapper link_mapper_;
    // Opaque shared pool for exact transaction checkpoint storage. The
    // concrete type belongs to PreparedControllerMutation below.
    std::shared_ptr<void> mutation_checkpoint_workspace_;
};

// Lightweight non-owning resource-mutation view. Dynamic resource names are
// resolved once by prepare_mutation(); every hot mutation loop uses IDs.
class PreparedControllerMutation
{
public:
    bool deploy(
        const Solution& solution,
        ControllerMutationOptions options = {});

    bool release(
        const Solution& solution,
        ControllerMutationOptions options = {});

    // Captures only selected physical resource slots. The checkpoint is exact
    // (AttrValue copies), not an inverse floating-point arithmetic journal.
    void begin_transaction();
    void commit_transaction() noexcept;
    void rollback_transaction();
    bool transaction_active() const noexcept;

    // Restores every mutation recorded in a complete or partial Solution,
    // regardless of Solution::result. It does not alter the Solution journal.
    void rollback(
        const Solution& solution,
        ControllerMutationOptions options = {});

private:
    PreparedControllerMutation(
        network::PhysicalNetwork& physical_network,
        PreparedResourceUpdator resource_updator,
        std::vector<ResourceId> node_resource_order,
        std::vector<ResourceId> link_resource_order,
        std::vector<std::optional<AttrId>> node_physical_value_ids,
        std::vector<std::optional<AttrId>> link_physical_value_ids,
        std::shared_ptr<void> checkpoint_workspace);

    std::vector<ResourceAmount> gather_resources(
        const SolutionAttributeValues& values,
        const std::vector<ResourceId>& resource_order) const;

    void gather_resources_into(
        const SolutionAttributeValues& values,
        const std::vector<ResourceId>& resource_order,
        std::vector<ResourceAmount>& output) const;

    void apply_recorded_values(
        const Solution& solution,
        network::attribute::ResourceUpdateOperation operation,
        std::size_t workers);

    void apply_mapped_values(
        const Solution& solution,
        network::attribute::ResourceUpdateOperation operation,
        std::size_t workers);

    PreparedResourceUpdator resource_updator_;
    network::PhysicalNetwork* physical_network_ = nullptr;
    std::vector<ResourceId> node_resource_order_;
    std::vector<ResourceId> link_resource_order_;
    std::vector<std::optional<AttrId>> node_physical_value_ids_;
    std::vector<std::optional<AttrId>> link_physical_value_ids_;
    std::vector<ResourceAmount> resource_scratch_;

    enum class CheckpointTarget : std::uint8_t
    {
        node,
        edge,
    };

    struct CheckpointValue
    {
        CheckpointTarget target = CheckpointTarget::node;
        std::uint32_t target_id = 0U;
        AttrId attr_id = 0U;
        AttrValue value = std::int64_t{0};
    };

    struct CheckpointWorkspace
    {
        std::mutex mutex;
        std::vector<
            std::unique_ptr<std::vector<CheckpointValue>>> free_buffers;
    };

    void return_checkpoint_buffer() noexcept;

    std::shared_ptr<CheckpointWorkspace> checkpoint_workspace_;
    std::unique_ptr<std::vector<CheckpointValue>> checkpoint_values_;
    bool checkpoint_active_ = false;

    friend class Controller;
};

class PreparedController
{
public:
    // Safe-only counterpart of Python Controller._safely_deploy_with_node_slots.
    // Returns false for incomplete/-1 assignments or mapper infeasibility;
    // dependency exceptions retain the partial state produced before failure.
    bool deploy_with_node_slots(
        const NodeSlots& node_slots,
        Solution& solution,
        DeployWithNodeSlotsOptions options = {});

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

    const std::vector<ConstraintLink>& links_to_route(
        Vertex virtual_node,
        const Solution& solution);

    void pool_step_offsets(
        const std::vector<ConstraintLink>& attempted_links,
        Solution& solution);

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
    // A prepared controller is a mutable transaction bound to one physical
    // network. Reuse its per-step buffers across virtual nodes instead of
    // allocating the same small route/offset vectors in every hot call.
    std::vector<ConstraintLink> links_to_route_scratch_;
    std::vector<ConstraintLink> attempted_routes_scratch_;
    std::vector<const SolutionAttributeValues*> link_values_scratch_;
    std::vector<const SolutionAttributeValues*> path_values_scratch_;
    std::vector<Vertex> node_slot_virtual_nodes_scratch_;
    std::vector<Vertex> node_slot_physical_nodes_scratch_;

    friend class Controller;
};

} // namespace virne::core::controller
