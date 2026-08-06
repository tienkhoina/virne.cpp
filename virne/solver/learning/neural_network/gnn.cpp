#include "gnn.h"

#include <torch/nn/init.h>
#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

GCNConvNetOptions validate_options(GCNConvNetOptions options) {
    if (options.input_dim <= 0 || options.output_dim <= 0
        || options.embedding_dim <= 0) {
        throw std::invalid_argument(
            "GCNConvNet channel dimensions must be positive");
    }
    if (options.num_layers <= 0) {
        throw std::invalid_argument("GCNConvNet num_layers must be positive");
    }
    if (!std::isfinite(options.dropout_prob)
        || options.dropout_prob < 0.0
        || options.dropout_prob > 1.0) {
        throw std::invalid_argument("GCNConvNet dropout_prob must be in [0, 1]");
    }
    if (options.pooling == GraphPoolingKind::none
        || options.pooling == GraphPoolingKind::add
        || options.pooling == GraphPoolingKind::mean
        || options.pooling == GraphPoolingKind::max) {
        return options;
    }
    throw std::invalid_argument("GCNConvNet pooling kind is unsupported");
}

void require_graph_input(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    const std::int64_t input_dim) {
    if (!x.defined() || x.dim() != 2 || x.size(1) != input_dim) {
        throw std::invalid_argument(
            "GCNConvNet x must have shape [nodes, input_dim]");
    }
    if (!edge_index.defined() || edge_index.dim() != 2
        || edge_index.size(0) != 2
        || edge_index.scalar_type() != torch::kInt64) {
        throw std::invalid_argument(
            "GCNConvNet edge_index must have int64 shape [2, edges]");
    }
    if (x.device() != edge_index.device()) {
        throw std::invalid_argument(
            "GCNConvNet x and edge_index must share a device");
    }
}

}  // namespace

GraphPoolingKind graph_pooling_kind_from_name(const std::string_view name) {
    if (name == "add" || name == "sum") return GraphPoolingKind::add;
    if (name == "mean") return GraphPoolingKind::mean;
    if (name == "max") return GraphPoolingKind::max;
    if (name.empty() || name == "none") return GraphPoolingKind::none;
    throw std::invalid_argument("unsupported graph pooling name");
}

GraphPooling::GraphPooling(const GraphPoolingKind kind) : kind_(kind) {
    if (kind_ == GraphPoolingKind::none) {
        throw std::invalid_argument("GraphPooling requires add, mean or max");
    }
}

torch::Tensor GraphPooling::forward(
    const torch::Tensor& node_embeddings,
    const std::optional<torch::Tensor>& batch,
    const std::optional<std::int64_t> size) const {
    switch (kind_) {
    case GraphPoolingKind::add:
        return pyg_lite::global_add_pool(node_embeddings, batch, size);
    case GraphPoolingKind::mean:
        return pyg_lite::global_mean_pool(node_embeddings, batch, size);
    case GraphPoolingKind::max:
        return pyg_lite::global_max_pool(node_embeddings, batch, size);
    case GraphPoolingKind::none:
        break;
    }
    throw std::logic_error("invalid graph pooling kind");
}

GCNConvNetImpl::GCNConvNetImpl(GCNConvNetOptions options)
    : options_(validate_options(std::move(options))) {
    layers_.reserve(static_cast<std::size_t>(options_.num_layers));
    for (std::int64_t layer_id = 0;
         layer_id < options_.num_layers;
         ++layer_id) {
        const bool final_layer = layer_id == options_.num_layers - 1;
        const std::int64_t input_dim = layer_id == 0
            ? options_.input_dim
            : options_.embedding_dim;
        const std::int64_t output_dim = final_layer
            ? options_.output_dim
            : options_.embedding_dim;
        const auto norm_dim = output_dim;
        Layer layer;
        const auto layer_name = std::to_string(layer_id);
        layer.conv = register_module(
            "conv_" + layer_name,
            pyg_lite::GCNConv(pyg_lite::GCNConvOptions{
                input_dim,
                output_dim,
                false,
                false,
                true,
                true,
                true,
            }));
        if (options_.batch_norm) {
            layer.has_norm = true;
            layer.norm = register_module(
                "norm_" + layer_name,
                torch::nn::BatchNorm1d(
                    torch::nn::BatchNorm1dOptions(norm_dim)));
        } else {
            register_module("norm_" + layer_name, torch::nn::Identity());
        }
        if (options_.dropout_prob < 1.0) {
            layer.has_dropout = true;
            layer.dropout = register_module(
                "dout_" + layer_name,
                torch::nn::Dropout(torch::nn::DropoutOptions(
                    options_.dropout_prob)));
        } else {
            register_module("dout_" + layer_name, torch::nn::Identity());
        }
        layers_.push_back(std::move(layer));
    }

    // Python constructs every layer first, then orthogonally initializes each
    // GCN linear weight in layer order.
    for (auto& layer : layers_) {
        torch::nn::init::orthogonal_(layer.conv->lin->weight);
    }
    if (options_.pooling != GraphPoolingKind::none) {
        graph_pooling_.emplace(options_.pooling);
    }
}

torch::Tensor GCNConvNetImpl::forward(const pyg_lite::Data& input) {
    require_graph_input(input.x, input.edge_index, options_.input_dim);
    return forward_impl(input.x, input.edge_index, input.batch,
                        std::nullopt, std::nullopt);
}

torch::Tensor GCNConvNetImpl::forward(const pyg_lite::Batch& input) {
    require_graph_input(input.x, input.edge_index, options_.input_dim);
    return forward_impl(input.x, input.edge_index, input.batch,
                        input.max_num_nodes, input.num_graphs);
}

torch::Tensor GCNConvNetImpl::forward_impl(
    const torch::Tensor& input,
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& batch,
    const std::optional<std::int64_t> max_num_nodes,
    const std::optional<std::int64_t> batch_size) {
    auto x = input;
    for (std::size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
        auto& layer = layers_[layer_id];
        x = layer.conv->forward(x, edge_index);
        if (layer.has_norm) x = layer.norm->forward(x);
        if (layer.has_dropout) x = layer.dropout->forward(x);
        if (layer_id + 1U != layers_.size()) {
            x = torch::leaky_relu(x, 0.01);
        }
    }

    if (options_.return_batch) {
        const auto dense = pyg_lite::to_dense_batch(
            x, batch, 0.0, max_num_nodes, batch_size);
        x = dense.values;
    }
    if (graph_pooling_.has_value()) {
        x = graph_pooling_->forward(x, batch, batch_size);
    }
    return x;
}

}  // namespace virne::solver::learning::neural_network
