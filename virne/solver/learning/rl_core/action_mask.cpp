#include "action_mask.h"

#include <torch/torch.h>

#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

void validate_values(const torch::Tensor& values) {
    if (!values.defined()) {
        throw ActionMaskException(
            ActionMaskErrorCode::undefined_values,
            "action-mask values tensor is undefined");
    }
}

void validate_mask(const torch::Tensor& mask) {
    if (!mask.defined()) {
        throw ActionMaskException(
            ActionMaskErrorCode::undefined_mask,
            "action-mask tensor is undefined");
    }
}

torch::Tensor normalize_mask(
    const torch::Tensor& values,
    const torch::Tensor& mask) {
    validate_values(values);
    validate_mask(mask);
    if (mask.numel() != values.numel()) {
        throw ActionMaskException(
            ActionMaskErrorCode::shape_mismatch,
            "action-mask tensor must have the same number of elements as values");
    }
    // Conversion happens once at the typed boundary.  reshape preserves
    // Python's element-count check and supports arbitrary action shapes.
    return mask.to(values.device(), torch::kBool, false, false)
        .reshape(values.sizes());
}

torch::Tensor vector_mask_tensor(
    const torch::Tensor& values,
    const std::vector<std::uint8_t>& mask) {
    validate_values(values);
    auto options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    if (mask.empty()) {
        return torch::empty({0}, options);
    }
    return torch::from_blob(
               const_cast<std::uint8_t*>(mask.data()),
               {static_cast<std::int64_t>(mask.size())},
               options)
        .clone();
}

torch::Tensor apply_mask(
    const torch::Tensor& values,
    const torch::Tensor& mask,
    const double fill_value) {
    const torch::Tensor normalized = normalize_mask(values, mask);
    // Python constructs a scalar fill tensor and lets where broadcast it.  A
    // scalar avoids materializing a second values-sized buffer on every call.
    const torch::Tensor fill = torch::full(
        {}, fill_value, values.options().requires_grad(false));
    return torch::where(normalized, values, fill);
}

}  // namespace

ActionMaskException::ActionMaskException(
    const ActionMaskErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

torch::Tensor apply_mask_to_logit(const torch::Tensor& logit) {
    validate_values(logit);
    return logit;
}

torch::Tensor apply_mask_to_logit(
    const torch::Tensor& logit,
    const torch::Tensor& mask) {
    return apply_mask(logit, mask, -1e8);
}

torch::Tensor apply_mask_to_logit(
    const torch::Tensor& logit,
    const std::vector<std::uint8_t>& mask) {
    return apply_mask_to_logit(logit, vector_mask_tensor(logit, mask));
}

torch::Tensor apply_mask_to_prob(const torch::Tensor& prob) {
    validate_values(prob);
    return prob;
}

torch::Tensor apply_mask_to_prob(
    const torch::Tensor& prob,
    const torch::Tensor& mask) {
    return apply_mask(prob, mask, 0.0);
}

torch::Tensor apply_mask_to_prob(
    const torch::Tensor& prob,
    const std::vector<std::uint8_t>& mask) {
    return apply_mask_to_prob(prob, vector_mask_tensor(prob, mask));
}

}  // namespace virne::solver::learning::rl_core
