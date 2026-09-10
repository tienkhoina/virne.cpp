#include "graph_module.h"
#include <stdexcept>

namespace virne::solver::learning::rl_policy {

GraphModule register_graph_module(torch::nn::Module& owner, const std::string& name,
    const neural_network::GnnKind kind, const neural_network::GraphConvNetOptions& o,
    const bool initialize_residual_weights) {
    using namespace neural_network;
    switch (kind) {
    case GnnKind::gcn:
        return GCNConvNet(owner.register_module(name, GCNConvNet(GCNConvNetOptions{
            o.input_dim, o.output_dim, o.embedding_dim, o.num_layers, o.batch_norm,
            o.dropout_prob, o.return_batch, o.pooling})));
    case GnnKind::gat:
        return GATConvNet(owner.register_module(name, GATConvNet(o)));
    case GnnKind::edge_fusion_gat:
        return EdgeFusionGATConvNet(owner.register_module(name, EdgeFusionGATConvNet(o)));
    case GnnKind::deep_edge_gat:
        return DeepEdgeFeatureGAT(owner.register_module(name, DeepEdgeFeatureGAT(DeepEdgeFeatureGATOptions{
            o.input_dim, o.output_dim, o.edge_dim, o.num_layers, 0.2, 0.2, o.embedding_dim,
            o.num_heads, o.batch_norm, o.dropout_prob, o.return_batch, o.pooling, initialize_residual_weights})));
    }
    throw std::invalid_argument("unsupported graph module kind");
}

torch::Tensor graph_forward(GraphModule& module, const pyg_lite::Batch& input) {
    return std::visit([&](auto& layer) { return layer->forward(input); }, module);
}

}  // namespace virne::solver::learning::rl_policy
