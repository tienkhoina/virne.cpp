#pragma once

#include "graph_module.h"
#include <torch/nn/modules/container/sequential.h>

namespace virne::solver::learning::rl_policy {

struct DualPolicyOptions {
    std::int64_t p_net_num_nodes = 0, p_net_x_dim = 0;
    std::optional<std::int64_t> p_net_edge_dim;
    std::int64_t v_net_x_dim = 0;
    std::optional<std::int64_t> v_net_edge_dim;
    std::int64_t embedding_dim = 128;
    double dropout_prob = 0.0;
    bool batch_norm = false;
    std::int64_t num_layers = 3;
    neural_network::GnnKind kind = neural_network::GnnKind::gcn;
    std::optional<neural_network::GnnKind> v_kind, p_kind;
    bool cost_critic = false, lambda_net = false;
};

struct DualObservation {
    pyg_lite::Batch v_net, p_net;
    torch::Tensor curr_v_node_id;
};

struct NetEncoding { torch::Tensor nodes, graph, dense, initial_dense; };

class NetEncoderImpl final : public torch::nn::Module {
public:
    NetEncoderImpl(std::int64_t features, std::optional<std::int64_t> edges,
        std::int64_t embedding = 128, double dropout = 0.0, bool norm = false,
        std::int64_t layers = 3, neural_network::GnnKind kind = neural_network::GnnKind::gcn,
        bool initialize_residual_weights = true);
    NetEncoding forward(const pyg_lite::Batch& input);
    torch::nn::Linear init_lin{nullptr};
    GraphModule net_gnn{neural_network::GCNConvNet{nullptr}};
    neural_network::GraphPoolingModule att_pool{nullptr};
};
TORCH_MODULE(NetEncoder);

class BiGnnBaseModelImpl final : public torch::nn::Module {
public:
    explicit BiGnnBaseModelImpl(DualPolicyOptions options);
    torch::Tensor forward(const DualObservation& observation);
    void init_parameters();
    NetEncoder v_net_encoder{nullptr}, p_net_encoder{nullptr};
    torch::nn::Sequential lin{nullptr};
private:
    torch::nn::Linear linear_in_{nullptr}, linear_out_{nullptr};
};
TORCH_MODULE(BiGnnBaseModel);

class DualActorImpl final : public torch::nn::Module {
public:
    explicit DualActorImpl(DualPolicyOptions options);
    torch::Tensor forward(const DualObservation& observation);
    BiGnnBaseModel encoder{nullptr};
};
TORCH_MODULE(DualActor);
class DualCriticImpl final : public torch::nn::Module {
public:
    explicit DualCriticImpl(DualPolicyOptions options);
    torch::Tensor forward(const DualObservation& observation);
    BiGnnBaseModel encoder{nullptr};
};
TORCH_MODULE(DualCritic);

class DualActorCriticImpl final : public torch::nn::Module {
public:
    explicit DualActorCriticImpl(DualPolicyOptions options);
    torch::Tensor act(const DualObservation& observation);
    torch::Tensor evaluate(const DualObservation& observation);
    DualActor actor{nullptr};
    DualCritic critic{nullptr}, cost_critic{nullptr}, lambda_net{nullptr};
};
TORCH_MODULE(DualActorCritic);

}  // namespace virne::solver::learning::rl_policy
