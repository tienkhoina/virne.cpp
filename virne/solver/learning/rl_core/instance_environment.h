#pragma once

#include "feature_types.h"
#include "../../../core/controller/route_method.h"
#include <memory>
#include <utility>

namespace virne::network { class PhysicalNetwork; class VirtualNetwork; }
namespace virne::core { class Solution; }

namespace virne::solver::learning::rl_core {

enum class InstanceStepMode : std::uint8_t { placement, joint, node_pair, node_slots, solution };
enum class EpisodeStatus : std::uint8_t { running, success, rejected, placement_failed, routing_failed, incomplete };
struct InstanceEnvironmentOptions {
    InstanceStepMode mode = InstanceStepMode::joint;
    bool allow_rejection = false, allow_revocable = false, reusable = false, allow_constraint_violation = false;
    bool check_link_candidates = true;
    core::controller::ShortestPathMethod shortest_method = core::controller::ShortestPathMethod::bfs_shortest;
    std::int64_t k_shortest = 10;
    FeatureConstructorKind features = FeatureConstructorKind::p_net_v_node;
    bool node_status = true, aggregated_links = true, degree = true, other_metrics = true;
    double intermediate_reward = 0.1;
    // The fixed reward IDs match RewardCalculatorKind; resolved before stepping.
    enum class Reward : std::uint8_t { gradual, adaptive, fixed, vanilla } reward = Reward::fixed;
    std::vector<std::int64_t> node_order;
    std::size_t workers = 1;
};
struct NodeAssignment { std::int64_t virtual_node = 0, physical_node = 0; };
struct RouteEdge { std::int64_t source = 0, target = 0; };
struct RouteAssignment { RouteEdge virtual_edge; std::vector<RouteEdge> physical_edges; };
struct EpisodeInfo {
    EpisodeStatus status = EpisodeStatus::running;
    bool result = false, place_result = true, route_result = true, early_rejection = false;
    std::int64_t request_id = 0, num_interactions = 0, revoke_times = 0;
    double cumulative_reward = 0.0, revenue = 0.0, cost = 0.0, revenue_cost_ratio = 0.0, hard_constraint_violation = 0.0;
    std::vector<NodeAssignment> node_slots;
    std::vector<RouteAssignment> link_paths;
    std::vector<std::int64_t> selected_actions;
};
struct InstanceObservation {
    FeatureObservation features;
    std::vector<std::uint8_t> action_mask;
    std::int64_t curr_v_node_id = 0, v_net_size = 0, num_actions = 0;
};
struct InstanceStep { InstanceObservation observation; double reward = 0; bool done = false; EpisodeInfo info; };

// Graph ownership stays behind this implementation. Public observations contain
// only numeric vectors, avoiding std::string-bearing graph objects in Torch's ABI.
class InstanceEnvironment final {
public:
    InstanceEnvironment(network::PhysicalNetwork&& physical, network::VirtualNetwork&& virtual_network,
        InstanceEnvironmentOptions options = {});
    ~InstanceEnvironment();
    InstanceEnvironment(InstanceEnvironment&&) noexcept;
    InstanceEnvironment& operator=(InstanceEnvironment&&) noexcept;
    InstanceEnvironment(const InstanceEnvironment&) = delete;
    InstanceEnvironment& operator=(const InstanceEnvironment&) = delete;
    InstanceObservation reset();
    InstanceObservation observation() const;
    std::vector<std::uint8_t> action_masks() const;
    InstanceStep step(std::int64_t action);
    InstanceStep step_node_slots(const std::vector<NodeAssignment>& assignments);
    InstanceStep step_solution(const core::Solution& solution);
    InstanceStep reject();
    InstanceStep revoke();
    EpisodeInfo info() const;
    // Graph-ABI bridge for submitting a detached journal to BaseEnvironment.
    core::Solution solution_snapshot() const;
    const network::PhysicalNetwork& physical_network() const noexcept;
    std::int64_t num_actions() const;
    std::int64_t current_virtual_node() const;
    bool done() const noexcept;
    double placement_progress() const;
    double node_load_balance(std::int64_t physical_node) const;
    std::unique_ptr<InstanceEnvironment> clone() const;
private:
    class Impl;
    explicit InstanceEnvironment(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace virne::solver::learning::rl_core
