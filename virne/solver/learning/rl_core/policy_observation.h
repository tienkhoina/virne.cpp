#pragma once
#include "instance_environment.h"
#include "../rl_policy/policy_registry.h"

namespace virne::solver::learning::rl_core {
rl_policy::PolicyObservation prepare_policy_observation(const InstanceObservation& observation,
    rl_policy::ObservationKind kind, const torch::Device& device = torch::kCPU);
rl_policy::PolicyObservation gather_policy_observations(
    const std::vector<rl_policy::PolicyObservation>& observations, const torch::Tensor& indices);
}  // namespace virne::solver::learning::rl_core
