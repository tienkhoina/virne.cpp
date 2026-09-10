#pragma once

#include "../neural_network/gnn.h"
#include "../neural_network/mlp.h"

namespace virne::solver::learning::rl_policy {

struct GnnMlpPolicyOptions {
    std::int64_t p_net_num_nodes = 0;
    std::int64_t p_net_x_dim = 0;
    std::int64_t v_node_feature_dim = 0;
    std::int64_t num_layers = 3;
    std::int64_t embedding_dim = 128;
    std::optional<std::int64_t> p_net_edge_dim;
    double dropout_prob = 0.0;
    bool batch_norm = false;
};

struct GnnMlpObservation {
    pyg_lite::Batch p_net;
    torch::Tensor v_node_x;
};

class GnnMlpEncoderImpl final : public torch::nn::Module {
public:
    GnnMlpEncoderImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options);
    torch::Tensor forward(const pyg_lite::Batch& p_net, const torch::Tensor& v_node_x);
    neural_network::MLPNet mlp{nullptr};

private:
    neural_network::GnnKind kind_;
    neural_network::GCNConvNet gcn_{nullptr};
    neural_network::GATConvNet gat_{nullptr};
    neural_network::DeepEdgeFeatureGAT deep_{nullptr};
};
TORCH_MODULE(GnnMlpEncoder);

class GnnMlpActorImpl final : public torch::nn::Module {
public:
    GnnMlpActorImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options);
    torch::Tensor forward(const GnnMlpObservation& observation);
    GnnMlpEncoder encoder{nullptr};
    neural_network::MLPNet head{nullptr};
};
TORCH_MODULE(GnnMlpActor);

class GnnMlpCriticImpl final : public torch::nn::Module {
public:
    GnnMlpCriticImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options);
    torch::Tensor forward(const GnnMlpObservation& observation);
    GnnMlpEncoder encoder{nullptr};
    neural_network::MLPNet head{nullptr};
};
TORCH_MODULE(GnnMlpCritic);

class GnnMlpActorCriticImpl : public torch::nn::Module {
public:
    GnnMlpActorCriticImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options);
    torch::Tensor act(const GnnMlpObservation& observation);
    torch::Tensor evaluate(const GnnMlpObservation& observation);
    GnnMlpActor actor{nullptr};
    GnnMlpCritic critic{nullptr};
};
TORCH_MODULE(GnnMlpActorCritic);

class GcnMlpActorCriticImpl final : public GnnMlpActorCriticImpl {
public:
    explicit GcnMlpActorCriticImpl(GnnMlpPolicyOptions options);
};
TORCH_MODULE(GcnMlpActorCritic);

class GatMlpActorCriticImpl final : public GnnMlpActorCriticImpl {
public:
    explicit GatMlpActorCriticImpl(GnnMlpPolicyOptions options);
};
TORCH_MODULE(GatMlpActorCritic);

class DeepEdgeFeatureGATActorCriticImpl final : public GnnMlpActorCriticImpl {
public:
    explicit DeepEdgeFeatureGATActorCriticImpl(GnnMlpPolicyOptions options);
};
TORCH_MODULE(DeepEdgeFeatureGATActorCritic);

}  // namespace virne::solver::learning::rl_policy
