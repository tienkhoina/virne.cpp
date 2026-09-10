#pragma once
#include "instance_environment.h"
#include "../../../core/environment.h"

namespace virne::solver::learning::rl_core {
struct OnlineObservation {
    std::optional<InstanceObservation> instance;
    std::optional<core::EnvironmentEventState> event;
    bool done = false;
};
struct OnlineStep {
    OnlineObservation observation;
    EpisodeInfo episode;
    double reward = 0., cumulative_reward = 0.;
    bool request_done = false, done = false;
    std::optional<core::EnvironmentStepResult> admission;
};
// Owns the event lifecycle. An instance works on a private physical snapshot;
// only its detached journal is committed to the lifecycle at request completion.
class OnlineEnvironment final {
public:
    OnlineEnvironment(network::PhysicalNetwork physical, network::VirtualNetworkRequestSimulator simulator,
        core::EnvironmentConfig lifecycle, InstanceEnvironmentOptions options = {});
    OnlineObservation reset();
    OnlineObservation observation() const;
    OnlineStep step(std::int64_t action);
    OnlineStep step_node_slots(const std::vector<NodeAssignment>& assignments);
    OnlineStep step_solution(const core::Solution& solution);
    OnlineStep reject();
    OnlineStep revoke();
    OnlineStep finish_instance();
    InstanceEnvironment& instance();
    const network::PhysicalNetwork& physical_network() const noexcept;
    const core::SolutionStepEnvironment& lifecycle() const noexcept;
    double cumulative_reward() const noexcept;
    bool done() const noexcept;
private:
    void prepare_instance();
    OnlineStep transition(InstanceStep step);
    core::SolutionStepEnvironment lifecycle_;
    InstanceEnvironmentOptions options_;
    std::unique_ptr<InstanceEnvironment> instance_;
    double cumulative_reward_ = 0.;
    double credited_instance_reward_ = 0.;
};
}  // namespace virne::solver::learning::rl_core
