#include "policy_observation.h"
#include "tensor_convertor.h"
#include <torch/torch.h>
#include <torch/nn/utils/rnn.h>
#include <limits>
#include <stdexcept>

namespace virne::solver::learning::rl_core {
namespace p = rl_policy;
namespace g = pyg_lite;
namespace {
template <class T>
torch::Tensor matrix(const std::vector<T>& values, const std::size_t rows, const std::size_t columns,
    const torch::ScalarType dtype, const torch::Device& device) {
    if (columns && rows > std::numeric_limits<std::size_t>::max() / columns) throw std::overflow_error("observation shape overflow");
    if (values.size() != rows * columns || rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
        || columns > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) throw std::invalid_argument("observation data size mismatch");
    const std::vector<std::int64_t> shape{static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)};
    if (values.empty()) return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device));
    return torch::from_blob(const_cast<T*>(values.data()), shape, torch::TensorOptions().dtype(dtype)).clone().to(device);
}
torch::Tensor features(const DenseMatrixF32& value, const torch::Device& device) {
    return matrix(value.values, value.rows, value.columns, torch::kFloat32, device);
}
g::Batch graph(const NetworkFeatures& value, const torch::Device& device) {
    g::Data data;
    data.x = features(value.x, device);
    data.edge_index = matrix(value.edge_index.values, value.edge_index.rows, value.edge_index.columns, torch::kInt64, device);
    data.edge_attr = features(value.edge_attr, device);
    return g::Batch::from_data_list({data});
}
g::Data single_graph(const g::Batch& batch) {
    if (batch.num_graphs != 1) throw std::invalid_argument("rollout graph record must contain one graph");
    g::Data data; data.x = batch.x; data.edge_index = batch.edge_index; data.edge_attr = batch.edge_attr; return data;
}
}

p::PolicyObservation prepare_policy_observation(const InstanceObservation& obs,
    const p::ObservationKind kind, const torch::Device& device) {
    p::PolicyObservation result;
    if (obs.num_actions < 0 || obs.action_mask.size() != static_cast<std::size_t>(obs.num_actions)) throw std::invalid_argument("action-mask size mismatch");
    result.action_mask = matrix(obs.action_mask, 1, obs.action_mask.size(), torch::kUInt8, device).to(torch::kFloat32);
    result.current_virtual_node = torch::tensor({obs.curr_v_node_id}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    result.virtual_net_size = torch::tensor({static_cast<float>(obs.v_net_size)}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    if (obs.features.v_node_x) result.v_node_x = matrix(obs.features.v_node_x->values, 1, obs.features.v_node_x->values.size(), torch::kFloat32, device);
    switch (kind) {
    case p::ObservationKind::dense:
    case p::ObservationKind::cnn: {
        if (!result.v_node_x.defined()) throw std::invalid_argument("dense policy requires virtual-node features");
        DenseObservation dense;
        dense.p_net_x = features(obs.features.p_net.x, device);
        dense.v_node_x = result.v_node_x.squeeze(0);
        result.p_net_x = (kind == p::ObservationKind::cnn
            ? TensorConvertor::obs_as_tensor_for_cnn(dense, device)
            : TensorConvertor::obs_as_tensor_for_mlp(dense, device)).p_net_x;
        result.v_node_x = torch::Tensor{};
        break;
    }
    case p::ObservationKind::attention:
        if (!result.v_node_x.defined()) throw std::invalid_argument("attention policy requires virtual-node features");
        result.p_net_x = features(obs.features.p_net.x, device).unsqueeze(0); break;
    case p::ObservationKind::gnn_mlp:
        if (!result.v_node_x.defined()) throw std::invalid_argument("GNN/MLP policy requires virtual-node features");
        result.p_net = graph(obs.features.p_net, device); break;
    case p::ObservationKind::dual:
    case p::ObservationKind::recurrent:
        if (!obs.features.v_net) throw std::invalid_argument("policy requires complete virtual-network features");
        result.p_net = graph(obs.features.p_net, device);
        if (kind == p::ObservationKind::dual) result.v_net = graph(*obs.features.v_net, device);
        else result.v_net_x = features(obs.features.v_net->x, device).unsqueeze(0);
        break;
    }
    return result;
}

p::PolicyObservation gather_policy_observations(const std::vector<p::PolicyObservation>& records,
    const torch::Tensor& indices) {
    if (indices.dim() != 1 || indices.scalar_type() != torch::kInt64 || indices.device().type() != torch::kCPU) throw std::invalid_argument("sample indices must be a CPU int64 vector");
    const auto index = indices.contiguous();
    const auto access = index.accessor<std::int64_t, 1>();
    std::vector<torch::Tensor> physical, vnode, virtual_x, current, sizes, hidden, node_id, encoded, masks, actions;
    std::vector<g::Data> p_graphs, v_graphs;
    for (std::int64_t i = 0; i < index.numel(); ++i) {
        const auto id = access[i];
        if (id < 0 || id >= static_cast<std::int64_t>(records.size())) throw std::out_of_range("observation sample index");
        const auto& record = records[static_cast<std::size_t>(id)];
        if (record.p_net_x.defined()) physical.push_back(record.p_net_x);
        if (record.v_node_x.defined()) vnode.push_back(record.v_node_x);
        if (record.v_net_x.defined()) virtual_x.push_back(record.v_net_x.squeeze(0));
        if (record.current_virtual_node.defined()) current.push_back(record.current_virtual_node);
        if (record.virtual_net_size.defined()) sizes.push_back(record.virtual_net_size);
        if (record.p_net.x.defined()) p_graphs.push_back(single_graph(record.p_net));
        if (record.v_net.x.defined()) v_graphs.push_back(single_graph(record.v_net));
        if (record.hidden_state.defined()) hidden.push_back(record.hidden_state);
        if (record.p_node_id.defined()) node_id.push_back(record.p_node_id);
        if (record.encoder_outputs.defined()) encoded.push_back(record.encoder_outputs.squeeze(0));
        if (record.recurrent_mask) masks.push_back(record.recurrent_mask->squeeze(0));
        if (record.action_mask) actions.push_back(*record.action_mask);
    }
    if (index.numel() == 0) throw std::invalid_argument("empty observation sample");
    const auto check_count = [&](const auto& values) {
        if (!values.empty() && values.size() != static_cast<std::size_t>(index.numel()))
            throw std::invalid_argument("mixed observation schemas in sample");
    };
    for (const auto* values : {&physical, &vnode, &virtual_x, &current, &sizes, &hidden, &node_id, &encoded, &masks, &actions}) check_count(*values);
    check_count(p_graphs); check_count(v_graphs);
    p::PolicyObservation result;
    if (!physical.empty()) result.p_net_x = torch::cat(physical, 0);
    if (!vnode.empty()) result.v_node_x = torch::cat(vnode, 0);
    if (!virtual_x.empty()) result.v_net_x = torch::nn::utils::rnn::pad_sequence(virtual_x, true);
    if (!current.empty()) result.current_virtual_node = torch::cat(current, 0);
    if (!sizes.empty()) result.virtual_net_size = torch::cat(sizes, 0);
    if (!p_graphs.empty()) result.p_net = g::Batch::from_data_list(p_graphs);
    if (!v_graphs.empty()) result.v_net = g::Batch::from_data_list(v_graphs);
    if (!hidden.empty()) result.hidden_state = torch::cat(hidden, 0);
    if (!node_id.empty()) result.p_node_id = torch::cat(node_id, 0);
    if (!encoded.empty()) result.encoder_outputs = torch::nn::utils::rnn::pad_sequence(encoded, true);
    if (!masks.empty()) result.recurrent_mask = torch::nn::utils::rnn::pad_sequence(masks, true);
    if (!actions.empty()) result.action_mask = torch::cat(actions, 0);
    return result;
}

}  // namespace virne::solver::learning::rl_core
