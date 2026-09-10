#include "seq2seq_policy.h"
#include <torch/torch.h>
#include <algorithm>

namespace virne::solver::learning::rl_policy {

Seq2SeqEncoderImpl::Seq2SeqEncoderImpl(const std::int64_t features, const std::int64_t embedding) {
    emb = register_module("emb", torch::nn::Linear(features, embedding));
    gru = register_module("gru", torch::nn::GRU(embedding, embedding));
}
RecurrentOutput Seq2SeqEncoderImpl::forward(const torch::Tensor& input) {
    const auto sequence = input.permute({1, 0, 2});
    const auto embeddings = torch::relu(torch::linear(sequence, emb->weight, emb->bias));
    const auto result = gru->forward(embeddings);
    return {std::get<0>(result), std::get<1>(result)};
}
AdditiveAttentionImpl::AdditiveAttentionImpl(const std::int64_t hidden) {
    attn = register_module("attn", torch::nn::Linear(hidden * 2, hidden));
    v = register_module("v", torch::nn::Linear(torch::nn::LinearOptions(hidden, 1).bias(false)));
}
AdditiveAttentionOutput AdditiveAttentionImpl::forward(const torch::Tensor& hidden,
    const torch::Tensor& encoder_outputs, const std::optional<torch::Tensor>& mask) {
    const auto repeated = hidden.transpose(0, 1).repeat({1, encoder_outputs.size(1), 1});
    const auto energy = torch::tanh(torch::linear(torch::cat({repeated, encoder_outputs}, 2), attn->weight, attn->bias));
    auto scores = torch::linear(energy, v->weight, v->bias).squeeze(2);
    if (mask) scores = scores.masked_fill(*mask == 0, -1e10);
    const auto weights = torch::softmax(scores, 1);
    return {torch::bmm(weights.unsqueeze(1), encoder_outputs), weights};
}
Seq2SeqDecoderImpl::Seq2SeqDecoderImpl(Seq2SeqOptions o) : options_(o) {
    emb = register_module("emb", torch::nn::Embedding(o.p_net_num_nodes + 1, o.embedding_dim));
    att = register_module("att", AdditiveAttention(o.embedding_dim));
    const bool legacy = o.flavor == Seq2SeqFlavor::legacy_gcn;
    const auto kind = legacy ? neural_network::GnnKind::gcn : o.kind;
    const auto layers = kind == neural_network::GnnKind::deep_edge_gat ? 5 : 3;
    gcn = register_graph_module(*this, "gcn", kind, neural_network::GraphConvNetOptions{
        o.p_net_x_dim, o.embedding_dim, layers, o.embedding_dim, 1,
        legacy ? std::nullopt : o.p_net_edge_dim, legacy ? true : o.batch_norm,
        legacy ? 0.0 : o.dropout_prob, true, neural_network::GraphPoolingKind::none});
    mlp = register_module("mlp", torch::nn::Sequential());
    output_linear_ = torch::nn::Linear(o.embedding_dim, 1);
    mlp->push_back(output_linear_);
    mlp->push_back(torch::nn::Flatten());
    gru = register_module("gru", torch::nn::GRU(o.embedding_dim, o.embedding_dim));
}
DecoderOutput Seq2SeqDecoderImpl::forward(const Seq2SeqObservation& obs) {
    const auto sparse_embeddings = graph_forward(gcn, obs.p_net);
    const auto dense = sparse_embeddings.reshape({obs.p_net.num_graphs, -1, sparse_embeddings.size(-1)});
    const auto logits = torch::linear(dense + obs.hidden_state, output_linear_->weight, output_linear_->bias).flatten(1);
    const auto hidden = obs.hidden_state.permute({1, 0, 2});
    const auto node_id = options_.flavor == Seq2SeqFlavor::configurable_gnn
        ? torch::clamp_max(obs.p_node_id, options_.p_net_num_nodes) : obs.p_node_id;
    const auto embedded = emb->forward(node_id).unsqueeze(0);
    // The reference computes context but feeds only the node embedding into GRU.
    const auto context = att->forward(hidden, obs.encoder_outputs, obs.mask);
    static_cast<void>(context);
    const auto output = gru->forward(embedded, hidden);
    return {logits, std::get<0>(output), std::get<1>(output)};
}
Seq2SeqActorImpl::Seq2SeqActorImpl(Seq2SeqOptions o) {
    decoder = register_module("decoder", Seq2SeqDecoder(o));
}
DecoderOutput Seq2SeqActorImpl::forward(const Seq2SeqObservation& obs) { return decoder->forward(obs); }
Seq2SeqCriticImpl::Seq2SeqCriticImpl(Seq2SeqOptions o) {
    decoder = register_module("decoder", Seq2SeqDecoder(o));
}
torch::Tensor Seq2SeqCriticImpl::forward(const Seq2SeqObservation& obs) { return decoder->forward(obs).logits.mean(-1, true); }
GnnSeq2SeqActorCriticImpl::GnnSeq2SeqActorCriticImpl(Seq2SeqOptions o) {
    encoder = register_module("encoder", Seq2SeqEncoder(o.v_net_x_dim, o.embedding_dim));
    actor = register_module("actor", Seq2SeqActor(o));
    critic = register_module("critic", Seq2SeqCritic(o));
}
torch::Tensor GnnSeq2SeqActorCriticImpl::encode(const Seq2SeqObservation& obs) {
    const auto result = encoder->forward(obs.v_net_x);
    last_hidden_ = result.hidden_state;
    return result.outputs;
}
torch::Tensor GnnSeq2SeqActorCriticImpl::act(const Seq2SeqObservation& obs) {
    const auto result = actor->forward(obs);
    last_hidden_ = result.hidden_state;
    return result.logits;
}
torch::Tensor GnnSeq2SeqActorCriticImpl::evaluate(const Seq2SeqObservation& obs) { return critic->forward(obs); }

}  // namespace virne::solver::learning::rl_policy
