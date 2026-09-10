#pragma once

#include "../neural_network/gnn.h"
#include <variant>

namespace virne::solver::learning::rl_policy {

using GraphModule = std::variant<neural_network::GCNConvNet,
    neural_network::GATConvNet, neural_network::EdgeFusionGATConvNet,
    neural_network::DeepEdgeFeatureGAT>;

GraphModule register_graph_module(torch::nn::Module& owner, const std::string& name,
    neural_network::GnnKind kind, const neural_network::GraphConvNetOptions& options,
    bool initialize_residual_weights = true);
torch::Tensor graph_forward(GraphModule& module, const pyg_lite::Batch& input);

}  // namespace virne::solver::learning::rl_policy
