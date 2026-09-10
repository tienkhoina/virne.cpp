#include "att_policy.h"
#include <ATen/ops/_native_multi_head_attention.h>
#include <torch/torch.h>
#include <cmath>

namespace virne::solver::learning::rl_policy {

AttEncoderImpl::AttEncoderImpl(AttPolicyOptions o, const std::int64_t num_heads) : heads_(num_heads) {
    p_mlp = register_module("p_mlp", neural_network::MLPNet(neural_network::MlpOptions{
        o.p_net_x_dim, o.embedding_dim, o.num_layers, std::monostate{}, o.batch_norm, o.dropout_prob}));
    v_mlp = register_module("v_mlp", neural_network::MLPNet(neural_network::MlpOptions{
        o.v_node_feature_dim, o.embedding_dim, o.num_layers, std::monostate{}, o.batch_norm, o.dropout_prob}));
    att = register_module("att", torch::nn::MultiheadAttention(o.embedding_dim, heads_));
}

torch::Tensor AttEncoderImpl::forward(const AttObservation& obs) {
    const auto p = p_mlp->forward(obs.p_net_x);
    const auto v = v_mlp->forward(obs.v_node_x);
    const auto fusion = p + v.unsqueeze(1).repeat({1, p.size(1), 1});
    torch::Tensor attended;
    const bool tracks_grad = torch::GradMode::is_enabled() && (fusion.requires_grad()
        || att->in_proj_weight.requires_grad() || att->in_proj_bias.requires_grad()
        || att->out_proj->weight.requires_grad() || att->out_proj->bias.requires_grad());
    if (!att->is_training() && heads_ % 2 == 0 && !tracks_grad) {
        // Python MultiheadAttention's batch-first inference path uses this ATen operator.
        attended = std::get<0>(at::_native_multi_head_attention(fusion, fusion, fusion,
            att->options.embed_dim(), heads_, att->in_proj_weight, att->in_proj_bias,
            att->out_proj->weight, att->out_proj->bias, std::nullopt, true, true, std::nullopt));
    } else {
        const auto sequence = fusion.transpose(0, 1);
        const auto length = sequence.size(0), batch = sequence.size(1), embedding = sequence.size(2);
        const auto head_dim = embedding / heads_;
        // Torch 2.6 Python uses packed contiguous projections and a 2D output
        // linear. The older C++ frontend's 3D linear changes bias gradients.
        const auto packed = torch::linear(sequence, att->in_proj_weight, att->in_proj_bias)
            .unflatten(-1, {3, embedding}).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous();
        const auto q = packed.select(0, 0).view({length, batch * heads_, head_dim}).transpose(0, 1);
        const auto k = packed.select(0, 1).view({-1, batch * heads_, head_dim}).transpose(0, 1);
        const auto v = packed.select(0, 2).view({-1, batch * heads_, head_dim}).transpose(0, 1);
        const auto scaled_q = q * std::sqrt(1.0 / static_cast<double>(head_dim));
        const auto weights = torch::softmax(torch::bmm(scaled_q, k.transpose(-2, -1)), -1);
        const auto combined = torch::bmm(weights, v).transpose(0, 1).contiguous().view({length * batch, embedding});
        attended = torch::linear(combined, att->out_proj->weight, att->out_proj->bias)
            .view({length, batch, embedding}).transpose(0, 1);
    }
    return attended + fusion;
}

AttActorImpl::AttActorImpl(AttPolicyOptions o) {
    encoder = register_module("encoder", AttEncoder(o));
    lin_fusion = register_module("lin_fusion", torch::nn::Sequential());
    fusion_in_ = torch::nn::Linear(o.embedding_dim, o.embedding_dim);
    lin_fusion->push_back(fusion_in_);
    lin_fusion->push_back(torch::nn::ReLU());
    fusion_out_ = torch::nn::Linear(o.embedding_dim, 1);
    lin_fusion->push_back(fusion_out_);
}
torch::Tensor AttActorImpl::forward(const AttObservation& obs) {
    const auto encoded = encoder->forward(obs);
    const auto hidden = torch::relu(torch::linear(encoded, fusion_in_->weight, fusion_in_->bias));
    return torch::linear(hidden, fusion_out_->weight, fusion_out_->bias).squeeze(-1);
}
AttCriticImpl::AttCriticImpl(AttPolicyOptions o) {
    encoder = register_module("encoder", AttEncoder(o));
}
torch::Tensor AttCriticImpl::forward(const AttObservation& obs) {
    return encoder->forward(obs).mean(1).mean(1).unsqueeze(-1);
}
AttActorCriticImpl::AttActorCriticImpl(AttPolicyOptions o) {
    actor = register_module("actor", AttActor(o));
    critic = register_module("critic", AttCritic(o));
}
torch::Tensor AttActorCriticImpl::act(const AttObservation& obs) { return actor->forward(obs); }
torch::Tensor AttActorCriticImpl::evaluate(const AttObservation& obs) { return critic->forward(obs); }

}  // namespace virne::solver::learning::rl_policy
