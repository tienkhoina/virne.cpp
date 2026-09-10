#include "gnn_mlp_policy.h"

#include <stdexcept>

namespace virne::solver::learning::rl_policy {
namespace {

neural_network::MLPNet make_head(const GnnMlpPolicyOptions& options) {
    return neural_network::MLPNet(neural_network::MlpOptions{
        options.embedding_dim, 1, 2, options.embedding_dim / 2,
        options.batch_norm, options.dropout_prob});
}

}  // namespace

GnnMlpEncoderImpl::GnnMlpEncoderImpl(
    const neural_network::GnnKind kind, GnnMlpPolicyOptions options) : kind_(kind) {
    using namespace neural_network;
    switch (kind_) {
    case GnnKind::gcn:
        gcn_ = register_module("gnn", GCNConvNet(GCNConvNetOptions{
            options.p_net_x_dim, options.embedding_dim, options.embedding_dim,
            options.num_layers, options.batch_norm, options.dropout_prob, true,
            GraphPoolingKind::none}));
        break;
    case GnnKind::gat:
        gat_ = register_module("gnn", GATConvNet(GraphConvNetOptions{
            options.p_net_x_dim, options.embedding_dim, options.num_layers,
            options.embedding_dim, 1, options.p_net_edge_dim,
            options.batch_norm, options.dropout_prob, true, GraphPoolingKind::none}));
        break;
    case GnnKind::deep_edge_gat:
        deep_ = register_module("gnn", DeepEdgeFeatureGAT(DeepEdgeFeatureGATOptions{
            options.p_net_x_dim, options.embedding_dim, options.p_net_edge_dim,
            options.num_layers, 0.2, 0.2, options.embedding_dim, 1,
            options.batch_norm, options.dropout_prob, true, GraphPoolingKind::none}));
        break;
    default:
        throw std::invalid_argument("GNN/MLP policy supports GCN, GAT and DeepEdgeFeatureGAT");
    }
    mlp = register_module("mlp", MLPNet(MlpOptions{
        options.v_node_feature_dim, options.embedding_dim, 2, std::monostate{},
        options.batch_norm, options.dropout_prob}));
}

torch::Tensor GnnMlpEncoderImpl::forward(
    const pyg_lite::Batch& p_net, const torch::Tensor& v_node_x) {
    torch::Tensor p_embedding;
    switch (kind_) {
    case neural_network::GnnKind::gcn: p_embedding = gcn_->forward(p_net); break;
    case neural_network::GnnKind::gat: p_embedding = gat_->forward(p_net); break;
    case neural_network::GnnKind::deep_edge_gat: p_embedding = deep_->forward(p_net); break;
    default: throw std::logic_error("invalid GNN/MLP encoder kind");
    }
    const auto v_embedding = mlp->forward(v_node_x);
    return p_embedding + v_embedding.unsqueeze(1).expand({-1, p_embedding.size(1), -1});
}

GnnMlpActorImpl::GnnMlpActorImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options) {
    encoder = register_module("encoder", GnnMlpEncoder(kind, options));
    head = register_module("head", make_head(options));
}

torch::Tensor GnnMlpActorImpl::forward(const GnnMlpObservation& observation) {
    return head->forward(encoder->forward(observation.p_net, observation.v_node_x)).squeeze(-1);
}

GnnMlpCriticImpl::GnnMlpCriticImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options) {
    encoder = register_module("encoder", GnnMlpEncoder(kind, options));
    head = register_module("head", make_head(options));
}

torch::Tensor GnnMlpCriticImpl::forward(const GnnMlpObservation& observation) {
    return head->forward(encoder->forward(observation.p_net, observation.v_node_x)).mean(1);
}

GnnMlpActorCriticImpl::GnnMlpActorCriticImpl(neural_network::GnnKind kind, GnnMlpPolicyOptions options) {
    actor = register_module("actor", GnnMlpActor(kind, options));
    critic = register_module("critic", GnnMlpCritic(kind, options));
}

torch::Tensor GnnMlpActorCriticImpl::act(const GnnMlpObservation& observation) {
    return actor->forward(observation);
}

torch::Tensor GnnMlpActorCriticImpl::evaluate(const GnnMlpObservation& observation) {
    return critic->forward(observation);
}

GcnMlpActorCriticImpl::GcnMlpActorCriticImpl(GnnMlpPolicyOptions options)
    : GnnMlpActorCriticImpl(neural_network::GnnKind::gcn, options) {}
GatMlpActorCriticImpl::GatMlpActorCriticImpl(GnnMlpPolicyOptions options)
    : GnnMlpActorCriticImpl(neural_network::GnnKind::gat, options) {}
DeepEdgeFeatureGATActorCriticImpl::DeepEdgeFeatureGATActorCriticImpl(GnnMlpPolicyOptions options)
    : GnnMlpActorCriticImpl(neural_network::GnnKind::deep_edge_gat, options) {}

}  // namespace virne::solver::learning::rl_policy
