#include "mlp_inference.h"

#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace virne::solver::learning::rl_core {

MlpInferenceSession::MlpInferenceSession(
    const PolicyConfiguration& configuration,
    const PolicyActionOptions action_options,
    const torch::Device& device)
    : policy_(build_mlp_policy(configuration, device)),
      action_options_(action_options),
      device_(device) {
    policy_->eval();
}

MlpInferenceSession::MlpInferenceSession(
    rl_policy::MlpActorCritic policy,
    const PolicyActionOptions action_options,
    const torch::Device& device)
    : policy_(std::move(policy)),
      action_options_(action_options),
      device_(device) {
    if (!policy_) {
        throw std::invalid_argument("MLP inference policy is undefined");
    }
    policy_->to(device_);
    policy_->eval();
}

MlpInferenceForward MlpInferenceSession::forward_prepared(
    const MlpTensorObservation& observation) const {
    if (!policy_) {
        throw std::logic_error("MLP inference policy is undefined");
    }
    torch::NoGradGuard no_grad;
    return MlpInferenceForward{
        policy_->act(observation.p_net_x),
        policy_->evaluate(observation.p_net_x),
    };
}

torch::Tensor MlpInferenceSession::values_prepared(
    const MlpTensorObservation& observation) const {
    if (!policy_) {
        throw std::logic_error("MLP inference policy is undefined");
    }
    torch::NoGradGuard no_grad;
    return policy_->evaluate(observation.p_net_x);
}

MlpInferenceForward MlpInferenceSession::forward(
    const DenseObservation& observation) const {
    return forward_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observation, device_));
}

MlpInferenceForward MlpInferenceSession::forward(
    const DenseObservationBatch& observations) const {
    return forward_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observations, device_));
}

torch::Tensor MlpInferenceSession::action_logits(
    const DenseObservation& observation) const {
    return forward(observation).action_logits;
}

torch::Tensor MlpInferenceSession::action_logits(
    const DenseObservationBatch& observations) const {
    return forward(observations).action_logits;
}

torch::Tensor MlpInferenceSession::values(
    const DenseObservation& observation) const {
    return values_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observation, device_));
}

torch::Tensor MlpInferenceSession::values(
    const DenseObservationBatch& observations) const {
    return values_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observations, device_));
}

PolicyActionDistribution MlpInferenceSession::get_action_prob_dist_prepared(
    const MlpTensorObservation& observation) const {
    return policy_action_distribution(
        forward_prepared(observation).action_logits,
        observation.general.action_mask,
        action_options_);
}

PolicyActionDistribution MlpInferenceSession::get_action_prob_dist(
    const DenseObservation& observation) const {
    return get_action_prob_dist_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observation, device_));
}

PolicyActionDistribution MlpInferenceSession::get_action_prob_dist(
    const DenseObservationBatch& observations) const {
    return get_action_prob_dist_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observations, device_));
}

PolicyActionSelection MlpInferenceSession::select_action_prepared(
    const MlpTensorObservation& observation,
    const PolicyActionMode mode) const {
    return select_policy_action(
        forward_prepared(observation).action_logits,
        observation.general.action_mask,
        mode,
        action_options_);
}

PolicyActionSelection MlpInferenceSession::select_action(
    const DenseObservation& observation,
    const PolicyActionMode mode) const {
    return select_action_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observation, device_), mode);
}

PolicyActionSelection MlpInferenceSession::select_action(
    const DenseObservationBatch& observations,
    const PolicyActionMode mode) const {
    return select_action_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observations, device_), mode);
}

PolicyActionEvaluation MlpInferenceSession::evaluate_actions_prepared(
    const MlpTensorObservation& observation,
    const torch::Tensor& actions,
    const bool return_others,
    const std::optional<torch::Tensor>& predicted_mask) const {
    const auto output = forward_prepared(observation);
    return evaluate_policy_actions(
        output.action_logits,
        actions,
        output.values,
        observation.general.action_mask,
        predicted_mask,
        return_others,
        action_options_);
}

PolicyActionEvaluation MlpInferenceSession::evaluate_actions(
    const DenseObservation& observation,
    const torch::Tensor& actions,
    const bool return_others,
    const std::optional<torch::Tensor>& predicted_mask) const {
    return evaluate_actions_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observation, device_),
        actions,
        return_others,
        predicted_mask);
}

PolicyActionEvaluation MlpInferenceSession::evaluate_actions(
    const DenseObservationBatch& observations,
    const torch::Tensor& actions,
    const bool return_others,
    const std::optional<torch::Tensor>& predicted_mask) const {
    return evaluate_actions_prepared(
        TensorConvertor::obs_as_tensor_for_mlp(observations, device_),
        actions,
        return_others,
        predicted_mask);
}

double MlpInferenceSession::estimate_value(
    const DenseObservation& observation) const {
    return values(observation).item<double>();
}

torch::Tensor MlpInferenceSession::estimate_values(
    const DenseObservationBatch& observations) const {
    return values(observations);
}

}  // namespace virne::solver::learning::rl_core
