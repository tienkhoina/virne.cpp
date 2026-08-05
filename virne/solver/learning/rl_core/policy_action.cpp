#include "policy_action.h"

#include "action_mask.h"

#include <torch/nn/functional/loss.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

struct CategoricalState {
    torch::Tensor probabilities;
    torch::Tensor normalized_logits;
};

void validate_logits(
    const torch::Tensor& logits,
    const PolicyActionOptions& options) {
    if (!logits.defined()) {
        throw PolicyActionException(
            PolicyActionErrorCode::undefined_logits,
            "policy action logits are undefined");
    }
    if (logits.dim() < 1 || logits.size(-1) <= 0
        || (logits.scalar_type() != torch::kFloat16
            && logits.scalar_type() != torch::kBFloat16
            && logits.scalar_type() != torch::kFloat32
            && logits.scalar_type() != torch::kFloat64)) {
        throw PolicyActionException(
            PolicyActionErrorCode::invalid_logits,
            "policy action logits must be non-empty floating point tensors");
    }
    if (!std::isfinite(options.softmax_temperature)
        || options.softmax_temperature <= 0.0) {
        throw PolicyActionException(
            PolicyActionErrorCode::invalid_temperature,
            "softmax temperature must be finite and positive");
    }
}

torch::Tensor normalized_mask(
    const torch::Tensor& logits,
    const torch::Tensor& mask) {
    if (!mask.defined() || mask.numel() != logits.numel()) {
        throw PolicyActionException(
            PolicyActionErrorCode::shape_mismatch,
            "action mask must contain one entry per logit");
    }
    return mask.to(logits.device(), torch::kBool, false, false)
        .reshape(logits.sizes());
}

torch::Tensor candidate_logits(
    const torch::Tensor& logits,
    const std::optional<torch::Tensor>& mask,
    const bool apply_mask) {
    if (!mask.has_value() || !apply_mask) return logits;
    return apply_mask_to_logit(logits, *mask);
}

torch::Tensor masked_logits(
    const torch::Tensor& logits,
    const torch::Tensor& normalized) {
    const double fill_value = logits.scalar_type() == torch::kFloat16
        ? -std::numeric_limits<double>::infinity()
        : -1e8;
    const auto fill = torch::full(
        {}, fill_value, logits.options().requires_grad(false));
    return torch::where(normalized, logits, fill);
}

torch::Tensor categorical_logits(const torch::Tensor& logits) {
    return logits - logits.logsumexp(-1, true);
}

double dtype_epsilon(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat16:
        return 0.0009765625;
    case torch::kBFloat16:
        return 0.0078125;
    case torch::kFloat64:
        return std::numeric_limits<double>::epsilon();
    default:
        return static_cast<double>(std::numeric_limits<float>::epsilon());
    }
}

double dtype_lowest(const torch::ScalarType type) {
    switch (type) {
    case torch::kFloat16:
        return -65504.0;
    case torch::kBFloat16:
        return -3.3895313892515355e38;
    case torch::kFloat64:
        return std::numeric_limits<double>::lowest();
    default:
        return static_cast<double>(std::numeric_limits<float>::lowest());
    }
}

CategoricalState categorical_from_probabilities(
    const torch::Tensor& probabilities) {
    CategoricalState result;
    result.probabilities = probabilities
        / probabilities.sum(-1, true);
    const double epsilon = dtype_epsilon(result.probabilities.scalar_type());
    result.normalized_logits = result.probabilities
        .clamp(epsilon, 1.0 - epsilon)
        .log();
    return result;
}

void validate_actions(
    const torch::Tensor& logits,
    const torch::Tensor& actions) {
    bool shape_matches = actions.defined()
        && actions.dim() == logits.dim() - 1;
    if (shape_matches) {
        for (std::int64_t axis = 0; axis < actions.dim(); ++axis) {
            if (actions.size(axis) != logits.size(axis)) {
                shape_matches = false;
                break;
            }
        }
    }
    if (!shape_matches || actions.device() != logits.device()) {
        throw PolicyActionException(
            PolicyActionErrorCode::invalid_actions,
            "actions must match the logits batch shape and device");
    }
}

torch::Tensor categorical_log_prob(
    const torch::Tensor& normalized_logits,
    const torch::Tensor& actions) {
    return normalized_logits.gather(
        -1, actions.to(torch::kInt64).unsqueeze(-1)).squeeze(-1);
}

torch::Tensor categorical_sample(const torch::Tensor& normalized_logits) {
    const auto probabilities = torch::softmax(normalized_logits, -1);
    const auto events = probabilities.size(-1);
    const auto shape = probabilities.sizes().slice(
        0, static_cast<std::size_t>(probabilities.dim() - 1));
    return torch::multinomial(
               probabilities.reshape({-1, events}),
               1,
               true)
        .t()
        .reshape(shape);
}

torch::Tensor categorical_entropy(const CategoricalState& distribution) {
    const auto logits = distribution.normalized_logits.clamp_min(
        dtype_lowest(distribution.normalized_logits.scalar_type()));
    return -(logits * distribution.probabilities).sum(-1);
}

}  // namespace

PolicyActionException::PolicyActionException(
    const PolicyActionErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

PolicyActionDistribution policy_action_distribution(
    const torch::Tensor& action_logits,
    const std::optional<torch::Tensor>& action_mask,
    const PolicyActionOptions& options) {
    validate_logits(action_logits, options);
    torch::NoGradGuard no_grad;
    const auto candidates = candidate_logits(
        action_logits, action_mask, options.mask_actions);
    return PolicyActionDistribution{
        torch::softmax(candidates / options.softmax_temperature, -1),
        candidates.detach()};
}

PolicyActionSelection select_policy_action(
    const torch::Tensor& action_logits,
    const std::optional<torch::Tensor>& action_mask,
    const PolicyActionMode mode,
    const PolicyActionOptions& options) {
    validate_logits(action_logits, options);
    torch::NoGradGuard no_grad;
    const auto candidates = candidate_logits(
        action_logits, action_mask, options.mask_actions);
    const auto candidate_normalized_logits = categorical_logits(
        candidates / options.softmax_temperature);
    const auto actions = mode == PolicyActionMode::sample
        ? categorical_sample(candidate_normalized_logits)
        : candidates.argmax(-1);
    const auto log_normalized_logits =
        options.mask_actions && options.maskable_policy
        ? candidate_normalized_logits
        : categorical_logits(action_logits / options.softmax_temperature);
    return PolicyActionSelection{
        actions,
        categorical_log_prob(log_normalized_logits, actions)};
}

PolicyActionEvaluation evaluate_policy_actions(
    const torch::Tensor& action_logits,
    const torch::Tensor& actions,
    const std::optional<torch::Tensor>& values,
    const std::optional<torch::Tensor>& action_mask,
    const std::optional<torch::Tensor>& predicted_mask,
    const bool return_others,
    const PolicyActionOptions& options) {
    validate_logits(action_logits, options);
    validate_actions(action_logits, actions);

    const bool use_candidates = options.mask_actions
        && options.maskable_policy;
    const bool need_raw_probabilities = !use_candidates
        || (return_others && action_mask.has_value());
    const auto mask = action_mask.has_value()
        ? std::optional<torch::Tensor>(normalized_mask(
              action_logits, *action_mask))
        : std::nullopt;
    torch::Tensor raw_probabilities;
    if (need_raw_probabilities) {
        raw_probabilities = torch::softmax(
            action_logits / options.softmax_temperature, -1);
    }
    // Preserve Python's evaluate_actions quirk: a present mask is applied
    // regardless of mask_actions, and candidate probabilities omit temperature.
    CategoricalState policy_distribution;
    if (use_candidates) {
        const auto candidates = mask.has_value()
            ? masked_logits(action_logits, *mask)
            : action_logits;
        policy_distribution = categorical_from_probabilities(
            torch::softmax(candidates, -1));
    } else {
        policy_distribution = categorical_from_probabilities(
            raw_probabilities);
    }

    PolicyActionEvaluation result;
    if (values.has_value()) result.values = values->squeeze(-1);
    result.log_probabilities = categorical_log_prob(
        policy_distribution.normalized_logits, actions);
    result.entropy = categorical_entropy(policy_distribution);

    if (return_others && action_mask.has_value()) {
        result.masked_action_probability = (
            raw_probabilities * mask->logical_not())
                .sum(-1)
                .mean();
        if (predicted_mask.has_value()) {
            if (!predicted_mask->defined()
                || predicted_mask->device() != action_logits.device()
                || predicted_mask->sizes() != action_logits.sizes()) {
                throw PolicyActionException(
                    PolicyActionErrorCode::shape_mismatch,
                    "predicted mask must match action logits shape and device");
            }
            result.prediction_loss =
                torch::nn::functional::binary_cross_entropy(
                    *predicted_mask,
                    mask->to(torch::kFloat32));
        }
    }
    return result;
}

}  // namespace virne::solver::learning::rl_core
