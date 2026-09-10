#pragma once

#include "../neural_network/mlp.h"
#include <torch/nn/modules/activation.h>

namespace virne::solver::learning::rl_policy {

struct AttPolicyOptions {
    std::int64_t p_net_num_nodes = 0;
    std::int64_t p_net_x_dim = 0;
    std::int64_t v_node_feature_dim = 0;
    std::int64_t num_layers = 3;
    std::int64_t embedding_dim = 128;
    double dropout_prob = 0.0;
    bool batch_norm = false;
};

struct AttObservation { torch::Tensor p_net_x, v_node_x; };

class AttEncoderImpl final : public torch::nn::Module {
public:
    explicit AttEncoderImpl(AttPolicyOptions options, std::int64_t num_heads = 1);
    torch::Tensor forward(const AttObservation& observation);
    neural_network::MLPNet p_mlp{nullptr}, v_mlp{nullptr};
    torch::nn::MultiheadAttention att{nullptr};
private:
    std::int64_t heads_;
};
TORCH_MODULE(AttEncoder);

class AttActorImpl final : public torch::nn::Module {
public:
    explicit AttActorImpl(AttPolicyOptions options);
    torch::Tensor forward(const AttObservation& observation);
    AttEncoder encoder{nullptr};
    torch::nn::Sequential lin_fusion{nullptr};
private:
    torch::nn::Linear fusion_in_{nullptr}, fusion_out_{nullptr};
};
TORCH_MODULE(AttActor);

class AttCriticImpl final : public torch::nn::Module {
public:
    explicit AttCriticImpl(AttPolicyOptions options);
    torch::Tensor forward(const AttObservation& observation);
    AttEncoder encoder{nullptr};
};
TORCH_MODULE(AttCritic);

class AttActorCriticImpl final : public torch::nn::Module {
public:
    explicit AttActorCriticImpl(AttPolicyOptions options);
    torch::Tensor act(const AttObservation& observation);
    torch::Tensor evaluate(const AttObservation& observation);
    AttActor actor{nullptr};
    AttCritic critic{nullptr};
};
TORCH_MODULE(AttActorCritic);

}  // namespace virne::solver::learning::rl_policy
