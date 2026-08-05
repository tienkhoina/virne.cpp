#include "observation_sampler.h"

#include <torch/torch.h>

#include <optional>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

void validate_indices(const torch::Tensor& indices) {
    if (!indices.defined()) {
        throw ObservationSampleException(
            ObservationSampleErrorCode::invalid_indices,
            "observation sample indices are undefined");
    }
    if (indices.dim() != 1 || indices.scalar_type() != torch::kInt64) {
        throw ObservationSampleException(
            ObservationSampleErrorCode::invalid_indices,
            "observation sample indices must be a one-dimensional int64 tensor");
    }
}

void validate_source(const torch::Tensor& source) {
    if (!source.defined()) {
        throw ObservationSampleException(
            ObservationSampleErrorCode::undefined_tensor,
            "observation source tensor is undefined");
    }
    if (source.dim() < 1) {
        throw ObservationSampleException(
            ObservationSampleErrorCode::invalid_tensor_rank,
            "observation source tensor requires a leading batch dimension");
    }
}

torch::Tensor indices_on(
    const torch::Tensor& indices,
    const torch::Device& source_device,
    const std::int64_t batch_size) {
    validate_indices(indices);
    const torch::Tensor prepared = indices.to(
        source_device, torch::kInt64, false, false);
    // Python permits negative leading-dimension indices whereas index_select
    // does not. Normalize once per fixed observation, then use index_select
    // for every field; positive and excessively negative indices still fail.
    return torch::where(
        prepared.ge(0), prepared, prepared + batch_size);
}

void validate_device(
    const torch::Tensor& source,
    const torch::Device& expected,
    const std::int64_t expected_batch_size) {
    validate_source(source);
    if (source.device() != expected) {
        throw ObservationSampleException(
            ObservationSampleErrorCode::device_mismatch,
            "fixed observation fields must share one source device");
    }
    if (source.size(0) != expected_batch_size) {
        throw ObservationSampleException(
            ObservationSampleErrorCode::batch_size_mismatch,
            "fixed observation fields must share one leading batch size");
    }
}

torch::Tensor sample_prepared(
    const torch::Tensor& source,
    const torch::Tensor& prepared_indices,
    const torch::Device& target_device,
    const torch::Device& source_device,
    const std::int64_t batch_size) {
    validate_device(source, source_device, batch_size);
    return source.index_select(0, prepared_indices).to(
        target_device, source.scalar_type(), false, false);
}

std::optional<torch::Tensor> sample_optional(
    const std::optional<torch::Tensor>& source,
    const torch::Tensor& prepared_indices,
    const torch::Device& target_device,
    const torch::Device& source_device,
    const std::int64_t batch_size) {
    if (!source.has_value()) {
        return std::nullopt;
    }
    return sample_prepared(
        *source, prepared_indices, target_device, source_device, batch_size);
}

GeneralTensorObservation sample_general_prepared(
    const GeneralTensorObservation& source,
    const torch::Tensor& prepared_indices,
    const torch::Device& target_device,
    const torch::Device& source_device,
    const std::int64_t batch_size) {
    GeneralTensorObservation result;
    result.curr_v_node_id = sample_optional(
        source.curr_v_node_id, prepared_indices, target_device, source_device,
        batch_size);
    result.action_mask = sample_optional(
        source.action_mask, prepared_indices, target_device, source_device,
        batch_size);
    result.v_net_size = sample_optional(
        source.v_net_size, prepared_indices, target_device, source_device,
        batch_size);
    return result;
}

const torch::Tensor* first_general_tensor(
    const GeneralTensorObservation& source) noexcept {
    if (source.curr_v_node_id.has_value()) {
        return &*source.curr_v_node_id;
    }
    if (source.action_mask.has_value()) {
        return &*source.action_mask;
    }
    if (source.v_net_size.has_value()) {
        return &*source.v_net_size;
    }
    return nullptr;
}

}  // namespace

ObservationSampleException::ObservationSampleException(
    const ObservationSampleErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

torch::Tensor sample_observations(
    const torch::Tensor& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    validate_source(observations);
    const torch::Tensor prepared = indices_on(
        indices, observations.device(), observations.size(0));
    return sample_prepared(
        observations, prepared, device, observations.device(),
        observations.size(0));
}

DenseObservationBatch sample_observations(
    const DenseObservationBatch& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    validate_source(observations.p_net_x);
    const torch::Device source_device = observations.p_net_x.device();
    const torch::Tensor prepared = indices_on(
        indices, source_device, observations.p_net_x.size(0));
    DenseObservationBatch result;
    result.p_net_x = sample_prepared(
        observations.p_net_x, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.v_node_x = sample_prepared(
        observations.v_node_x, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.curr_v_node_id = sample_optional(
        observations.curr_v_node_id, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.action_mask = sample_optional(
        observations.action_mask, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.v_net_size = sample_optional(
        observations.v_net_size, prepared, device, source_device,
        observations.p_net_x.size(0));
    return result;
}

GeneralTensorObservation sample_observations(
    const GeneralTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    const torch::Tensor* source = first_general_tensor(observations);
    if (source == nullptr) {
        validate_indices(indices);
        return {};
    }
    validate_source(*source);
    const torch::Tensor prepared = indices_on(
        indices, source->device(), source->size(0));
    return sample_general_prepared(
        observations, prepared, device, source->device(), source->size(0));
}

MlpTensorObservation sample_observations(
    const MlpTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    validate_source(observations.p_net_x);
    const torch::Device source_device = observations.p_net_x.device();
    const torch::Tensor prepared = indices_on(
        indices, source_device, observations.p_net_x.size(0));
    MlpTensorObservation result;
    result.p_net_x = sample_prepared(
        observations.p_net_x, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.general = sample_general_prepared(
        observations.general, prepared, device, source_device,
        observations.p_net_x.size(0));
    return result;
}

MlpFcTensorObservation sample_observations(
    const MlpFcTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    validate_source(observations.pv_nodes_x);
    const torch::Device source_device = observations.pv_nodes_x.device();
    const torch::Tensor prepared = indices_on(
        indices, source_device, observations.pv_nodes_x.size(0));
    MlpFcTensorObservation result;
    result.pv_nodes_x = sample_prepared(
        observations.pv_nodes_x, prepared, device, source_device,
        observations.pv_nodes_x.size(0));
    result.general = sample_general_prepared(
        observations.general, prepared, device, source_device,
        observations.pv_nodes_x.size(0));
    return result;
}

AttentionTensorObservation sample_observations(
    const AttentionTensorObservation& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    validate_source(observations.p_net_x);
    const torch::Device source_device = observations.p_net_x.device();
    const torch::Tensor prepared = indices_on(
        indices, source_device, observations.p_net_x.size(0));
    AttentionTensorObservation result;
    result.p_net_x = sample_prepared(
        observations.p_net_x, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.v_node_x = sample_prepared(
        observations.v_node_x, prepared, device, source_device,
        observations.p_net_x.size(0));
    result.general = sample_general_prepared(
        observations.general, prepared, device, source_device,
        observations.p_net_x.size(0));
    return result;
}

pyg_lite::Batch sample_observations(
    const pyg_lite::Batch& observations,
    const torch::Tensor& indices,
    const torch::Device& device) {
    validate_indices(indices);
    return pyg_lite::Batch::from_data_list(
               observations.index_select(indices))
        .to(device, false, false);
}

}  // namespace virne::solver::learning::rl_core
