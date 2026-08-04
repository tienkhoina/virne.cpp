#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>

namespace virne::solver::learning::rl_core {

// Fixed observation fields are direct members. The Python string-keyed mapping
// is resolved once before this boundary and never reaches tensor hot paths.
struct DenseObservation {
    torch::Tensor p_net_x;
    torch::Tensor v_node_x;
    std::optional<std::int64_t> curr_v_node_id;
    std::optional<torch::Tensor> action_mask;
    std::optional<double> v_net_size;
};

struct DenseObservationBatch {
    torch::Tensor p_net_x;
    torch::Tensor v_node_x;
    std::optional<torch::Tensor> curr_v_node_id;
    std::optional<torch::Tensor> action_mask;
    std::optional<torch::Tensor> v_net_size;
};

struct GeneralTensorObservation {
    std::optional<torch::Tensor> curr_v_node_id;
    std::optional<torch::Tensor> action_mask;
    std::optional<torch::Tensor> v_net_size;
};

struct MlpTensorObservation {
    torch::Tensor p_net_x;
    GeneralTensorObservation general;
};

struct MlpFcTensorObservation {
    torch::Tensor pv_nodes_x;
    GeneralTensorObservation general;
};

struct AttentionTensorObservation {
    torch::Tensor p_net_x;
    torch::Tensor v_node_x;
    GeneralTensorObservation general;
};

class TensorConvertor final {
public:
    TensorConvertor() = delete;

    static torch::Tensor p_net_x_obs_as_tensor(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static torch::Tensor p_net_x_obs_as_tensor(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);

    static torch::Tensor v_node_obs_as_tensor(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static torch::Tensor v_node_obs_as_tensor(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);

    static GeneralTensorObservation general_obs_as_tensor(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static GeneralTensorObservation general_obs_as_tensor(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);

    static MlpTensorObservation obs_as_tensor_for_mlp(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static MlpTensorObservation obs_as_tensor_for_mlp(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);

    static MlpTensorObservation obs_as_tensor_for_cnn(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static MlpTensorObservation obs_as_tensor_for_cnn(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);

    static MlpFcTensorObservation obs_as_tensor_for_mlp_fc(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static MlpFcTensorObservation obs_as_tensor_for_mlp_fc(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);

    static AttentionTensorObservation obs_as_tensor_for_att(
        const DenseObservation& observation,
        const torch::Device& device = torch::kCPU);
    static AttentionTensorObservation obs_as_tensor_for_att(
        const DenseObservationBatch& observations,
        const torch::Device& device = torch::kCPU);
};

} // namespace virne::solver::learning::rl_core
