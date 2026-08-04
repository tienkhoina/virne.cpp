#include "tensor_convertor.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace virne::solver::learning::rl_core {
namespace {

torch::Tensor to_float32(
    const torch::Tensor& input,
    const torch::Device& device) {
    if (!input.defined()) {
        throw std::invalid_argument("dense observation tensor is undefined");
    }
    return input.to(device, torch::kFloat32, false, false);
}

torch::Tensor to_int64(
    const torch::Tensor& input,
    const torch::Device& device) {
    if (!input.defined()) {
        throw std::invalid_argument("dense observation tensor is undefined");
    }
    return input.to(device, torch::kInt64, false, false);
}

void require_rank(
    const torch::Tensor& tensor,
    const std::int64_t expected,
    const char* field) {
    if (tensor.dim() != expected) {
        throw std::invalid_argument(
            std::string(field) + " has an invalid rank");
    }
}

void require_batch_size(
    const torch::Tensor& tensor,
    const std::int64_t expected,
    const char* field) {
    if (tensor.size(0) != expected) {
        throw std::invalid_argument(
            std::string(field) + " has an incompatible batch size");
    }
}

MlpTensorObservation combine_mlp(
    torch::Tensor p_net_x,
    torch::Tensor v_node_x,
    GeneralTensorObservation general) {
    require_rank(p_net_x, 3, "p_net_x");
    require_rank(v_node_x, 2, "v_node_x");
    require_batch_size(v_node_x, p_net_x.size(0), "v_node_x");
    const auto expanded_v_node = v_node_x.unsqueeze(1).expand(
        {-1, p_net_x.size(1), -1});

    MlpTensorObservation result{};
    result.p_net_x = torch::cat({p_net_x, expanded_v_node}, -1);
    result.general = std::move(general);
    return result;
}

MlpFcTensorObservation combine_mlp_fc(
    torch::Tensor p_net_x,
    torch::Tensor v_node_x,
    GeneralTensorObservation general) {
    require_rank(p_net_x, 3, "p_net_x");
    require_rank(v_node_x, 2, "v_node_x");
    require_batch_size(v_node_x, p_net_x.size(0), "v_node_x");

    MlpFcTensorObservation result{};
    result.pv_nodes_x = torch::cat({p_net_x.flatten(1), v_node_x}, -1);
    result.general = std::move(general);
    return result;
}

AttentionTensorObservation combine_attention(
    torch::Tensor p_net_x,
    torch::Tensor v_node_x,
    GeneralTensorObservation general) {
    require_rank(p_net_x, 3, "p_net_x");
    require_rank(v_node_x, 2, "v_node_x");
    require_batch_size(v_node_x, p_net_x.size(0), "v_node_x");

    AttentionTensorObservation result{};
    result.p_net_x = std::move(p_net_x);
    result.v_node_x = std::move(v_node_x);
    result.general = std::move(general);
    return result;
}

} // namespace

torch::Tensor TensorConvertor::p_net_x_obs_as_tensor(
    const DenseObservation& observation,
    const torch::Device& device) {
    auto result = to_float32(observation.p_net_x, device);
    require_rank(result, 2, "p_net_x");
    return result.unsqueeze(0);
}

torch::Tensor TensorConvertor::p_net_x_obs_as_tensor(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    auto result = to_float32(observations.p_net_x, device);
    require_rank(result, 3, "p_net_x");
    return result;
}

torch::Tensor TensorConvertor::v_node_obs_as_tensor(
    const DenseObservation& observation,
    const torch::Device& device) {
    auto result = to_float32(observation.v_node_x, device);
    require_rank(result, 1, "v_node_x");
    return result.unsqueeze(0);
}

torch::Tensor TensorConvertor::v_node_obs_as_tensor(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    auto result = to_float32(observations.v_node_x, device);
    require_rank(result, 2, "v_node_x");
    return result;
}

GeneralTensorObservation TensorConvertor::general_obs_as_tensor(
    const DenseObservation& observation,
    const torch::Device& device) {
    GeneralTensorObservation result{};
    if (observation.curr_v_node_id.has_value()) {
        result.curr_v_node_id = torch::tensor(
            {*observation.curr_v_node_id},
            torch::TensorOptions().dtype(torch::kInt64).device(device));
    }
    if (observation.action_mask.has_value()) {
        result.action_mask =
            to_float32(*observation.action_mask, device).unsqueeze(0);
    }
    if (observation.v_net_size.has_value()) {
        result.v_net_size = torch::tensor(
            {*observation.v_net_size},
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
    return result;
}

GeneralTensorObservation TensorConvertor::general_obs_as_tensor(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    GeneralTensorObservation result{};
    const auto batch_size = observations.p_net_x.size(0);
    if (observations.curr_v_node_id.has_value()) {
        auto value = to_int64(*observations.curr_v_node_id, device);
        require_rank(value, 1, "curr_v_node_id");
        require_batch_size(value, batch_size, "curr_v_node_id");
        result.curr_v_node_id = std::move(value);
    }
    if (observations.action_mask.has_value()) {
        auto value = to_float32(*observations.action_mask, device);
        require_rank(value, 2, "action_mask");
        require_batch_size(value, batch_size, "action_mask");
        result.action_mask = std::move(value);
    }
    if (observations.v_net_size.has_value()) {
        auto value = to_float32(*observations.v_net_size, device);
        require_rank(value, 1, "v_net_size");
        require_batch_size(value, batch_size, "v_net_size");
        result.v_net_size = std::move(value);
    }
    return result;
}

MlpTensorObservation TensorConvertor::obs_as_tensor_for_mlp(
    const DenseObservation& observation,
    const torch::Device& device) {
    return combine_mlp(
        p_net_x_obs_as_tensor(observation, device),
        v_node_obs_as_tensor(observation, device),
        general_obs_as_tensor(observation, device));
}

MlpTensorObservation TensorConvertor::obs_as_tensor_for_mlp(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    return combine_mlp(
        p_net_x_obs_as_tensor(observations, device),
        v_node_obs_as_tensor(observations, device),
        general_obs_as_tensor(observations, device));
}

MlpTensorObservation TensorConvertor::obs_as_tensor_for_cnn(
    const DenseObservation& observation,
    const torch::Device& device) {
    auto result = obs_as_tensor_for_mlp(observation, device);
    result.p_net_x = result.p_net_x.unsqueeze(1);
    return result;
}

MlpTensorObservation TensorConvertor::obs_as_tensor_for_cnn(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    auto result = obs_as_tensor_for_mlp(observations, device);
    result.p_net_x = result.p_net_x.unsqueeze(1);
    return result;
}

MlpFcTensorObservation TensorConvertor::obs_as_tensor_for_mlp_fc(
    const DenseObservation& observation,
    const torch::Device& device) {
    return combine_mlp_fc(
        p_net_x_obs_as_tensor(observation, device),
        v_node_obs_as_tensor(observation, device),
        general_obs_as_tensor(observation, device));
}

MlpFcTensorObservation TensorConvertor::obs_as_tensor_for_mlp_fc(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    return combine_mlp_fc(
        p_net_x_obs_as_tensor(observations, device),
        v_node_obs_as_tensor(observations, device),
        general_obs_as_tensor(observations, device));
}

AttentionTensorObservation TensorConvertor::obs_as_tensor_for_att(
    const DenseObservation& observation,
    const torch::Device& device) {
    return combine_attention(
        p_net_x_obs_as_tensor(observation, device),
        v_node_obs_as_tensor(observation, device),
        general_obs_as_tensor(observation, device));
}

AttentionTensorObservation TensorConvertor::obs_as_tensor_for_att(
    const DenseObservationBatch& observations,
    const torch::Device& device) {
    return combine_attention(
        p_net_x_obs_as_tensor(observations, device),
        v_node_obs_as_tensor(observations, device),
        general_obs_as_tensor(observations, device));
}

} // namespace virne::solver::learning::rl_core
