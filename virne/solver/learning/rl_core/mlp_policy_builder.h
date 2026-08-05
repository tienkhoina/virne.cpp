#pragma once

#include "policy_config.h"
#include "../rl_policy/mlp_policy.h"

#include <torch/types.h>

namespace virne::solver::learning::rl_core {

// Resolve Python PolicyBuilder.build_mlp_policy's fixed constructor arguments.
// This is a cold construction boundary: feature/config names have already been
// resolved into direct fields by PolicyConfiguration.
[[nodiscard]] rl_policy::MlpActorCriticOptions
make_mlp_actor_critic_options(const PolicyConfiguration& configuration);

// Construct the policy and move all registered parameters/buffers to device,
// exactly matching MlpActorCritic(...).to(agent.device). Optimizer ownership
// and all training state intentionally remain outside this adapter.
[[nodiscard]] rl_policy::MlpActorCritic build_mlp_policy(
    const PolicyConfiguration& configuration,
    const torch::Device& device);

}  // namespace virne::solver::learning::rl_core
