#pragma once

#include <torch/types.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::solver::learning::rl_core {

enum class ActionMaskErrorCode : std::uint8_t {
    undefined_values,
    undefined_mask,
    shape_mismatch,
};

class ActionMaskException final : public std::runtime_error {
public:
    ActionMaskException(ActionMaskErrorCode code, std::string message);

    ActionMaskErrorCode code() const noexcept { return code_; }

private:
    ActionMaskErrorCode code_;
};

// The no-mask overload deliberately returns the same Tensor handle, matching
// Python's early return and avoiding an allocation on the common fast path.
torch::Tensor apply_mask_to_logit(const torch::Tensor& logit);
torch::Tensor apply_mask_to_logit(
    const torch::Tensor& logit,
    const torch::Tensor& mask);
torch::Tensor apply_mask_to_logit(
    const torch::Tensor& logit,
    const std::vector<std::uint8_t>& mask);

torch::Tensor apply_mask_to_prob(const torch::Tensor& prob);
torch::Tensor apply_mask_to_prob(
    const torch::Tensor& prob,
    const torch::Tensor& mask);
torch::Tensor apply_mask_to_prob(
    const torch::Tensor& prob,
    const std::vector<std::uint8_t>& mask);

}  // namespace virne::solver::learning::rl_core
