#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace virne::solver::learning::rl_core {

enum class PolicyActionMode : std::uint8_t {
    sample,
    greedy,
};

enum class PolicyActionErrorCode : std::uint8_t {
    undefined_logits,
    invalid_logits,
    invalid_temperature,
    invalid_actions,
    shape_mismatch,
};

class PolicyActionException final : public std::runtime_error {
public:
    PolicyActionException(PolicyActionErrorCode code, std::string message);

    PolicyActionErrorCode code() const noexcept { return code_; }

private:
    PolicyActionErrorCode code_;
};

struct PolicyActionOptions {
    double softmax_temperature = 1.0;
    bool mask_actions = true;
    bool maskable_policy = true;
};

struct PolicyActionDistribution {
    torch::Tensor probabilities;
    torch::Tensor candidate_logits;
};

struct PolicyActionSelection {
    torch::Tensor actions;
    torch::Tensor log_probabilities;
};

struct PolicyActionEvaluation {
    std::optional<torch::Tensor> values;
    torch::Tensor log_probabilities;
    torch::Tensor entropy;
    std::optional<torch::Tensor> masked_action_probability;
    std::optional<torch::Tensor> prediction_loss;
};

PolicyActionDistribution policy_action_distribution(
    const torch::Tensor& action_logits,
    const std::optional<torch::Tensor>& action_mask = std::nullopt,
    const PolicyActionOptions& options = {});

PolicyActionSelection select_policy_action(
    const torch::Tensor& action_logits,
    const std::optional<torch::Tensor>& action_mask = std::nullopt,
    PolicyActionMode mode = PolicyActionMode::sample,
    const PolicyActionOptions& options = {});

PolicyActionEvaluation evaluate_policy_actions(
    const torch::Tensor& action_logits,
    const torch::Tensor& actions,
    const std::optional<torch::Tensor>& values = std::nullopt,
    const std::optional<torch::Tensor>& action_mask = std::nullopt,
    const std::optional<torch::Tensor>& predicted_mask = std::nullopt,
    bool return_others = false,
    const PolicyActionOptions& options = {});

}  // namespace virne::solver::learning::rl_core
