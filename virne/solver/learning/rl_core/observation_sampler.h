#pragma once

#include "tensor_convertor.h"
#include "../pyg_lite/pyg_lite.h"

#include <torch/types.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace virne::solver::learning::rl_core {

enum class ObservationSampleErrorCode : std::uint8_t {
    undefined_tensor,
    invalid_tensor_rank,
    invalid_indices,
    device_mismatch,
    batch_size_mismatch,
};

class ObservationSampleException final : public std::runtime_error {
public:
    ObservationSampleException(
        ObservationSampleErrorCode code,
        std::string message);

    ObservationSampleErrorCode code() const noexcept { return code_; }

private:
    ObservationSampleErrorCode code_;
};

torch::Tensor sample_observations(
    const torch::Tensor& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

DenseObservationBatch sample_observations(
    const DenseObservationBatch& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

GeneralTensorObservation sample_observations(
    const GeneralTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

MlpTensorObservation sample_observations(
    const MlpTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

MlpFcTensorObservation sample_observations(
    const MlpFcTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

AttentionTensorObservation sample_observations(
    const AttentionTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

pyg_lite::Batch sample_observations(
    const pyg_lite::Batch& observations,
    const torch::Tensor& indices,
    const torch::Device& device = torch::kCPU);

}  // namespace virne::solver::learning::rl_core
