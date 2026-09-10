#pragma once

#include "adam.h"
#include "training_types.h"
#include "policy_config.h"
#include "../rl_policy/policy_registry.h"

namespace virne::solver::learning::rl_core {

rl_policy::PolicyOptions policy_options_from_configuration(
    rl_policy::PolicyKind kind, const PolicyConfiguration& configuration);
std::unique_ptr<rl_policy::Policy> build_policy(rl_policy::PolicyKind kind,
    const PolicyConfiguration& configuration, const torch::Device& device = torch::kCPU);
std::unique_ptr<Adam> build_policy_optimizer(rl_policy::Policy& policy,
    const PolicyLearningRates& learning_rates, double weight_decay = 0.0);

}  // namespace virne::solver::learning::rl_core
