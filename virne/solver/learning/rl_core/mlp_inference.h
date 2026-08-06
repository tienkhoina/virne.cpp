#pragma once

#include "mlp_policy_builder.h"
#include "policy_action.h"
#include "tensor_convertor.h"

#include <torch/types.h>

#include <optional>

namespace virne::solver::learning::rl_core {

struct MlpInferenceForward {
    torch::Tensor action_logits;
    torch::Tensor values;
};

// Reusable typed inference boundary for the Python MLP solver path. The
// policy is constructed once; observations are converted through fixed fields
// and the existing action helpers are reused without string dispatch.
class MlpInferenceSession final {
public:
    MlpInferenceSession(
        const PolicyConfiguration& configuration,
        PolicyActionOptions action_options = {},
        const torch::Device& device = torch::kCPU);

    MlpInferenceSession(
        rl_policy::MlpActorCritic policy,
        PolicyActionOptions action_options = {},
        const torch::Device& device = torch::kCPU);

    [[nodiscard]] MlpInferenceForward forward(
        const DenseObservation& observation) const;
    [[nodiscard]] MlpInferenceForward forward(
        const DenseObservationBatch& observations) const;

    [[nodiscard]] torch::Tensor action_logits(
        const DenseObservation& observation) const;
    [[nodiscard]] torch::Tensor action_logits(
        const DenseObservationBatch& observations) const;

    [[nodiscard]] torch::Tensor values(
        const DenseObservation& observation) const;
    [[nodiscard]] torch::Tensor values(
        const DenseObservationBatch& observations) const;

    [[nodiscard]] PolicyActionDistribution get_action_prob_dist(
        const DenseObservation& observation) const;
    [[nodiscard]] PolicyActionDistribution get_action_prob_dist(
        const DenseObservationBatch& observations) const;

    [[nodiscard]] PolicyActionSelection select_action(
        const DenseObservation& observation,
        PolicyActionMode mode = PolicyActionMode::sample) const;
    [[nodiscard]] PolicyActionSelection select_action(
        const DenseObservationBatch& observations,
        PolicyActionMode mode = PolicyActionMode::sample) const;

    [[nodiscard]] PolicyActionEvaluation evaluate_actions(
        const DenseObservation& observation,
        const torch::Tensor& actions,
        bool return_others = false,
        const std::optional<torch::Tensor>& predicted_mask = std::nullopt)
        const;
    [[nodiscard]] PolicyActionEvaluation evaluate_actions(
        const DenseObservationBatch& observations,
        const torch::Tensor& actions,
        bool return_others = false,
        const std::optional<torch::Tensor>& predicted_mask = std::nullopt)
        const;

    [[nodiscard]] double estimate_value(
        const DenseObservation& observation) const;
    [[nodiscard]] torch::Tensor estimate_values(
        const DenseObservationBatch& observations) const;

    [[nodiscard]] rl_policy::MlpActorCritic& policy() noexcept {
        return policy_;
    }
    [[nodiscard]] const rl_policy::MlpActorCritic& policy() const noexcept {
        return policy_;
    }
    [[nodiscard]] const PolicyActionOptions& action_options() const noexcept {
        return action_options_;
    }
    [[nodiscard]] const torch::Device& device() const noexcept {
        return device_;
    }

private:
    MlpInferenceForward forward_prepared(
        const MlpTensorObservation& observation) const;
    torch::Tensor values_prepared(
        const MlpTensorObservation& observation) const;
    PolicyActionDistribution get_action_prob_dist_prepared(
        const MlpTensorObservation& observation) const;
    PolicyActionSelection select_action_prepared(
        const MlpTensorObservation& observation,
        PolicyActionMode mode) const;
    PolicyActionEvaluation evaluate_actions_prepared(
        const MlpTensorObservation& observation,
        const torch::Tensor& actions,
        bool return_others,
        const std::optional<torch::Tensor>& predicted_mask) const;

    mutable rl_policy::MlpActorCritic policy_{nullptr};
    PolicyActionOptions action_options_{};
    torch::Device device_{torch::kCPU};
};

}  // namespace virne::solver::learning::rl_core
