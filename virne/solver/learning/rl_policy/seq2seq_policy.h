#pragma once

#include "graph_module.h"
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/rnn.h>
#include <torch/nn/modules/container/sequential.h>

namespace virne::solver::learning::rl_policy {

enum class Seq2SeqFlavor : std::uint8_t { legacy_gcn, configurable_gnn };
struct Seq2SeqOptions {
    std::int64_t p_net_num_nodes = 0, p_net_x_dim = 0;
    std::optional<std::int64_t> p_net_edge_dim;
    std::int64_t v_net_x_dim = 0, embedding_dim = 128;
    double dropout_prob = 0.0;
    bool batch_norm = false;
    neural_network::GnnKind kind = neural_network::GnnKind::gcn;
    Seq2SeqFlavor flavor = Seq2SeqFlavor::legacy_gcn;
};
struct Seq2SeqObservation {
    pyg_lite::Batch p_net;
    torch::Tensor v_net_x, hidden_state, p_node_id, encoder_outputs;
    std::optional<torch::Tensor> mask;
};
struct RecurrentOutput { torch::Tensor outputs, hidden_state; };
struct DecoderOutput { torch::Tensor logits, outputs, hidden_state; };
struct AdditiveAttentionOutput { torch::Tensor context, weights; };

class Seq2SeqEncoderImpl final : public torch::nn::Module {
public:
    Seq2SeqEncoderImpl(std::int64_t input_features, std::int64_t embedding = 128);
    RecurrentOutput forward(const torch::Tensor& input);
    torch::nn::Linear emb{nullptr};
    torch::nn::GRU gru{nullptr};
};
TORCH_MODULE(Seq2SeqEncoder);

class AdditiveAttentionImpl final : public torch::nn::Module {
public:
    explicit AdditiveAttentionImpl(std::int64_t hidden_dim);
    AdditiveAttentionOutput forward(const torch::Tensor& hidden,
        const torch::Tensor& encoder_outputs, const std::optional<torch::Tensor>& mask = std::nullopt);
    torch::nn::Linear attn{nullptr}, v{nullptr};
};
TORCH_MODULE(AdditiveAttention);

class Seq2SeqDecoderImpl final : public torch::nn::Module {
public:
    explicit Seq2SeqDecoderImpl(Seq2SeqOptions options);
    DecoderOutput forward(const Seq2SeqObservation& observation);
    torch::nn::Embedding emb{nullptr};
    AdditiveAttention att{nullptr};
    GraphModule gcn{neural_network::GCNConvNet{nullptr}};
    torch::nn::Sequential mlp{nullptr};
    torch::nn::GRU gru{nullptr};
private:
    Seq2SeqOptions options_;
    torch::nn::Linear output_linear_{nullptr};
};
TORCH_MODULE(Seq2SeqDecoder);

class Seq2SeqActorImpl final : public torch::nn::Module {
public:
    explicit Seq2SeqActorImpl(Seq2SeqOptions options);
    DecoderOutput forward(const Seq2SeqObservation& observation);
    Seq2SeqDecoder decoder{nullptr};
};
TORCH_MODULE(Seq2SeqActor);
class Seq2SeqCriticImpl final : public torch::nn::Module {
public:
    explicit Seq2SeqCriticImpl(Seq2SeqOptions options);
    torch::Tensor forward(const Seq2SeqObservation& observation);
    Seq2SeqDecoder decoder{nullptr};
};
TORCH_MODULE(Seq2SeqCritic);

class GnnSeq2SeqActorCriticImpl final : public torch::nn::Module {
public:
    explicit GnnSeq2SeqActorCriticImpl(Seq2SeqOptions options);
    torch::Tensor encode(const Seq2SeqObservation& observation);
    torch::Tensor act(const Seq2SeqObservation& observation);
    torch::Tensor evaluate(const Seq2SeqObservation& observation);
    const torch::Tensor& get_last_rnn_state() const noexcept { return last_hidden_; }
    void set_last_rnn_hidden(torch::Tensor hidden) { last_hidden_ = std::move(hidden); }
    Seq2SeqEncoder encoder{nullptr};
    Seq2SeqActor actor{nullptr};
    Seq2SeqCritic critic{nullptr};
private:
    torch::Tensor last_hidden_;
};
TORCH_MODULE(GnnSeq2SeqActorCritic);

}  // namespace virne::solver::learning::rl_policy
