#include "gnn.h"

#include <torch/nn/init.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
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
        || options.pooling == GraphPoolingKind::max
        || options.pooling == GraphPoolingKind::attention) {
        return options;
    }
    throw std::invalid_argument("GCNConvNet pooling kind is unsupported");
}

GraphConvNetOptions validate_options(GraphConvNetOptions options) {
    if (options.input_dim <= 0 || options.output_dim <= 0
        || options.embedding_dim <= 0) {
        throw std::invalid_argument(
            "GraphConvNet channel dimensions must be positive");
    }
    if (options.num_layers <= 0) {
        throw std::invalid_argument("GraphConvNet num_layers must be positive");
    }
    if (options.num_heads <= 0) {
        throw std::invalid_argument("GraphConvNet num_heads must be positive");
    }
    if (options.edge_dim.has_value() && *options.edge_dim <= 0) {
        throw std::invalid_argument("GraphConvNet edge_dim must be positive");
    }
    if (!std::isfinite(options.dropout_prob)
        || options.dropout_prob < 0.0
        || options.dropout_prob > 1.0) {
        throw std::invalid_argument("GraphConvNet dropout_prob must be in [0, 1]");
    }
    return options;
}

DeepEdgeFeatureGATOptions validate_options(DeepEdgeFeatureGATOptions options) {
    if (options.input_dim <= 0 || options.output_dim <= 0
        || options.embedding_dim <= 0
        || (options.edge_dim.has_value() && *options.edge_dim <= 0)) {
        throw std::invalid_argument(
            "DeepEdgeFeatureGAT dimensions must be positive");
    }
    if (options.num_layers < 2) {
        throw std::invalid_argument(
            "DeepEdgeFeatureGAT num_layers must be at least two");
    }
    // With concatenation enabled the Python network is dimensionally valid
    // only for one head. Reject that reference error before entering a hot
    // forward loop.
    if (options.num_heads != 1) {
        throw std::invalid_argument(
            "DeepEdgeFeatureGAT reference path supports num_heads=1");
    }
    if (!std::isfinite(options.alpha) || !std::isfinite(options.theta)
        || (options.num_layers > 2 && options.theta <= -1.0)) {
        throw std::invalid_argument(
            "DeepEdgeFeatureGAT alpha/theta must be finite and residual theta > -1");
    }
    if (!std::isfinite(options.dropout_prob)
        || options.dropout_prob < 0.0
        || options.dropout_prob > 1.0) {
        throw std::invalid_argument(
            "DeepEdgeFeatureGAT dropout_prob must be in [0, 1]");
    }
    return options;
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

void require_edge_attributes(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& edge_attr,
    const std::optional<std::int64_t> edge_dim,
    const std::string_view owner,
    const bool required = false) {
    if (!edge_dim.has_value()) {
        if (edge_attr.has_value() && edge_attr->defined()
            && edge_attr->size(0) != edge_index.size(1)) {
            throw std::invalid_argument(
                std::string(owner) + " edge_attr row count must match edges");
        }
        return;
    }
    if (!edge_attr.has_value() || !edge_attr->defined()) {
        if (!required) return;
        throw std::invalid_argument(
            std::string(owner) + " edge_dim requires edge_attr");
    }
    const bool scalar_edges = edge_attr->dim() == 1 && *edge_dim == 1;
    if ((!scalar_edges && edge_attr->dim() != 2)
        || edge_attr->size(0) != edge_index.size(1)
        || (!scalar_edges && edge_attr->size(1) != *edge_dim)) {
        throw std::invalid_argument(
            std::string(owner) + " edge_attr must have shape [edges, edge_dim]");
    }
    if (edge_attr->device() != x.device()) {
        throw std::invalid_argument(
            std::string(owner) + " edge_attr must share the x device");
    }
}

std::int64_t resolve_pool_size(
    const torch::Tensor& x,
    const torch::Tensor& batch,
    const std::optional<std::int64_t> size) {
    if (size.has_value()) {
        if (*size < 0) throw std::invalid_argument("pool size must be non-negative");
        return *size;
    }
    if (x.size(0) == 0 || batch.numel() == 0) {
        throw std::invalid_argument(
            "attention pooling needs explicit size for an empty batch");
    }
    // This fallback mirrors Python's batch[-1].item()+1. Batch callers pass
    // num_graphs so the main path remains synchronization-free on CUDA.
    return batch.select(0, batch.size(0) - 1).item<std::int64_t>() + 1;
}

}  // namespace

GnnKind gnn_kind_from_name(const std::string_view name) {
    if (name == "gcn") return GnnKind::gcn;
    if (name == "gat") return GnnKind::gat;
    if (name == "edge_fusion_gat") return GnnKind::edge_fusion_gat;
    if (name == "deep_edge_gat") return GnnKind::deep_edge_gat;
    throw std::invalid_argument("unsupported GNN type");
}

torch::Tensor to_sparse_batch(
    const torch::Tensor& node_dense_embeddings,
    const torch::Tensor& mask) {
    if (!node_dense_embeddings.defined() || node_dense_embeddings.dim() != 3) {
        throw std::invalid_argument(
            "to_sparse_batch embeddings must have shape [B, N, D]");
    }
    if (!mask.defined() || mask.dim() != 2
        || mask.size(0) != node_dense_embeddings.size(0)
        || mask.size(1) != node_dense_embeddings.size(1)
        || mask.scalar_type() != torch::kBool) {
        throw std::invalid_argument(
            "to_sparse_batch mask must have bool shape [B, N]");
    }
    if (mask.device() != node_dense_embeddings.device()) {
        throw std::invalid_argument(
            "to_sparse_batch embeddings and mask must share a device");
    }
    const std::int64_t embedding_dim = node_dense_embeddings.size(2);
    return node_dense_embeddings.masked_select(
        mask.unsqueeze(-1).expand_as(node_dense_embeddings))
        .reshape({-1, embedding_dim});
}

DenseToSparseOutput DenseToSparseImpl::forward(
    const torch::Tensor& input,
    const torch::Tensor& adjacency,
    const torch::Tensor& weight) const {
    if (!input.defined() || input.dim() != 3) {
        throw std::invalid_argument("DenseToSparse x must have shape [B, N, D]");
    }
    const std::int64_t batch_count = input.size(0);
    const std::int64_t node_count = input.size(1);
    if (!adjacency.defined() || adjacency.dim() != 3
        || adjacency.size(0) != batch_count
        || adjacency.size(1) != node_count
        || adjacency.size(2) != node_count) {
        throw std::invalid_argument(
            "DenseToSparse adjacency must have shape [B, N, N]");
    }
    if (!weight.defined() || weight.sizes() != adjacency.sizes()) {
        throw std::invalid_argument(
            "DenseToSparse weight must match adjacency shape");
    }
    if (input.device() != adjacency.device() || input.device() != weight.device()) {
        throw std::invalid_argument(
            "DenseToSparse tensors must share a device");
    }

    const torch::Tensor indices = torch::nonzero(adjacency > 0);
    const torch::Tensor offsets = indices.select(1, 0);
    const torch::Tensor rows = indices.select(1, 1);
    const torch::Tensor columns = indices.select(1, 2);
    const torch::Tensor edge_weight = weight.index({offsets, rows, columns})
        .to(torch::kFloat32);
    const torch::Tensor flat_rows = rows + offsets * node_count;
    const torch::Tensor flat_columns = columns + offsets * node_count;
    const torch::Tensor edge_index = torch::stack(
        {flat_rows, flat_columns}, 0).to(torch::kInt64);
    const torch::Tensor flattened = input.view(
        {batch_count * node_count, input.size(2)});
    const torch::Tensor batch_index = torch::arange(
        batch_count,
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()))
        .view({-1, 1})
        .repeat({1, node_count})
        .view({-1});
    return {flattened, edge_index, edge_weight, batch_index};
}

GraphPoolingKind graph_pooling_kind_from_name(const std::string_view name) {
    if (name == "add" || name == "sum") return GraphPoolingKind::add;
    if (name == "mean") return GraphPoolingKind::mean;
    if (name == "max") return GraphPoolingKind::max;
    if (name == "att" || name == "attention") {
        return GraphPoolingKind::attention;
    }
    if (name.empty() || name == "none") return GraphPoolingKind::none;
    throw std::invalid_argument("unsupported graph pooling name");
}

GraphPooling::GraphPooling(const GraphPoolingKind kind) : kind_(kind) {
    if (kind_ == GraphPoolingKind::none
        || kind_ == GraphPoolingKind::attention) {
        throw std::invalid_argument(
            "GraphPooling requires stateless add, mean or max");
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
    case GraphPoolingKind::attention:
    case GraphPoolingKind::none:
        break;
    }
    throw std::logic_error("invalid graph pooling kind");
}

GraphAttentionPoolingImpl::GraphAttentionPoolingImpl(
    const std::int64_t input_dim)
    : input_dim_(input_dim) {
    if (input_dim_ <= 0) {
        throw std::invalid_argument(
            "GraphAttentionPooling input_dim must be positive");
    }
    weight = register_parameter(
        "weight", torch::empty({input_dim_, input_dim_}));
    torch::nn::init::orthogonal_(weight);
}

torch::Tensor GraphAttentionPoolingImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& batch,
    const std::optional<std::int64_t> size) const {
    if (!x.defined() || x.dim() != 2 || x.size(1) != input_dim_) {
        throw std::invalid_argument(
            "GraphAttentionPooling x must have shape [nodes, input_dim]");
    }
    if (!batch.defined() || batch.dim() != 1 || batch.size(0) != x.size(0)
        || batch.scalar_type() != torch::kInt64 || batch.device() != x.device()) {
        throw std::invalid_argument(
            "GraphAttentionPooling batch must have int64 shape [nodes]");
    }
    const std::int64_t graph_count = resolve_pool_size(x, batch, size);
    const torch::Tensor mean = pyg_lite::scatter_dim0(
        x, batch, graph_count, pyg_lite::ScatterReduce::mean);
    const torch::Tensor transformed_global = torch::tanh(torch::mm(mean, weight));
    const torch::Tensor coefficients = torch::sigmoid(
        (x * transformed_global.index_select(0, batch) * 10.0).sum(1));
    const torch::Tensor weighted = coefficients.unsqueeze(-1) * x;
    return pyg_lite::scatter_dim0(
        weighted, batch, graph_count, pyg_lite::ScatterReduce::sum);
}

torch::Tensor GraphAttentionPoolingImpl::get_coefs(
    const torch::Tensor& x) const {
    if (!x.defined() || x.dim() != 2 || x.size(1) != input_dim_) {
        throw std::invalid_argument(
            "GraphAttentionPooling x must have shape [nodes, input_dim]");
    }
    const torch::Tensor mean = x.mean(0);
    const torch::Tensor transformed_global = torch::tanh(
        torch::matmul(mean, weight));
    return torch::sigmoid(torch::matmul(x, transformed_global));
}

GraphPoolingModuleImpl::GraphPoolingModuleImpl(
    const GraphPoolingKind kind,
    const std::int64_t input_dim)
    : kind_(kind) {
    if (kind_ == GraphPoolingKind::none) {
        throw std::invalid_argument("GraphPoolingModule requires a pooling kind");
    }
    if (kind_ == GraphPoolingKind::attention) {
        pooling = register_module(
            "pooling", GraphAttentionPooling(input_dim));
    }
}

torch::Tensor GraphPoolingModuleImpl::forward(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& batch,
    const std::optional<std::int64_t> size) const {
    if (kind_ == GraphPoolingKind::attention) {
        if (!batch.has_value()) {
            throw std::invalid_argument(
                "attention graph pooling requires a batch tensor");
        }
        return pooling->forward(x, *batch, size);
    }
    return GraphPooling(kind_).forward(x, batch, size);
}

GCNConvNetImpl::GCNConvNetImpl(GCNConvNetOptions options)
    : options_(validate_options(std::move(options))) {
    if (options_.pooling != GraphPoolingKind::none) {
        learned_graph_pooling_ = register_module(
            "graph_pooling",
            GraphPoolingModule(options_.pooling, options_.output_dim));
    }
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
    if (learned_graph_pooling_) {
        x = learned_graph_pooling_->forward(x, batch, batch_size);
    }
    return x;
}

GraphConvNetImpl::GraphConvNetImpl(
    GraphConvNetOptions options,
    const GraphConvKind kind)
    : options_(validate_options(std::move(options))), kind_(kind) {
    if (options_.pooling != GraphPoolingKind::none) {
        graph_pooling_ = register_module(
            "graph_pooling",
            GraphPoolingModule(options_.pooling, options_.output_dim));
    }
    layers_.reserve(static_cast<std::size_t>(options_.num_layers));
    for (std::int64_t layer_id = 0;
         layer_id < options_.num_layers;
         ++layer_id) {
        const bool final_layer = layer_id + 1 == options_.num_layers;
        const std::int64_t input_dim = layer_id == 0
            ? options_.input_dim
            : options_.embedding_dim;
        const std::int64_t output_dim = final_layer
            ? options_.output_dim
            : options_.embedding_dim;
        const std::string layer_name = std::to_string(layer_id);
        Layer layer;
        if (kind_ == GraphConvKind::gat) {
            layer.gat = register_module(
                "conv_" + layer_name,
                pyg_lite::GATConv(pyg_lite::GATConvOptions{
                    input_dim,
                    output_dim,
                    1,
                    true,
                    0.2,
                    0.0,
                    true,
                    options_.edge_dim,
                    pyg_lite::SelfLoopFill::reduce(
                        pyg_lite::SelfLoopFillKind::maximum),
                    true,
                    false,
                }));
        } else {
            layer.edge_fusion = register_module(
                "conv_" + layer_name,
                pyg_lite::EdgeFusionGATConv(
                    pyg_lite::EdgeFusionGATConvOptions{
                        input_dim,
                        output_dim,
                        1,
                        true,
                        0.2,
                        0.0,
                        true,
                        options_.edge_dim,
                        pyg_lite::SelfLoopFill::reduce(
                            pyg_lite::SelfLoopFillKind::maximum),
                        true,
                        pyg_lite::ScatterReduce::sum,
                    }));
        }
        if (options_.batch_norm) {
            layer.has_norm = true;
            layer.norm = register_module(
                "norm_" + layer_name,
                torch::nn::BatchNorm1d(output_dim));
        } else {
            register_module("norm_" + layer_name, torch::nn::Identity());
        }
        if (options_.dropout_prob < 1.0) {
            layer.has_dropout = true;
            layer.dropout = register_module(
                "dout_" + layer_name,
                torch::nn::Dropout(options_.dropout_prob));
        } else {
            register_module("dout_" + layer_name, torch::nn::Identity());
        }
        layers_.push_back(std::move(layer));
    }
}

torch::Tensor GraphConvNetImpl::forward(const pyg_lite::Data& input) {
    require_graph_input(input.x, input.edge_index, options_.input_dim);
    require_edge_attributes(
        input.x,
        input.edge_index,
        input.edge_attr,
        options_.edge_dim,
        "GraphConvNet", kind_ == GraphConvKind::edge_fusion_gat);
    return forward_impl(
        input.x,
        input.edge_index,
        input.edge_attr,
        input.batch,
        std::nullopt,
        std::nullopt);
}

torch::Tensor GraphConvNetImpl::forward(const pyg_lite::Batch& input) {
    require_graph_input(input.x, input.edge_index, options_.input_dim);
    require_edge_attributes(
        input.x,
        input.edge_index,
        input.edge_attr,
        options_.edge_dim,
        "GraphConvNet", kind_ == GraphConvKind::edge_fusion_gat);
    return forward_impl(
        input.x,
        input.edge_index,
        input.edge_attr,
        input.batch,
        input.max_num_nodes,
        input.num_graphs);
}

torch::Tensor GraphConvNetImpl::forward_impl(
    const torch::Tensor& input,
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& edge_attr,
    const std::optional<torch::Tensor>& batch,
    const std::optional<std::int64_t> max_num_nodes,
    const std::optional<std::int64_t> batch_size) {
    auto x = input;
    for (std::size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
        auto& layer = layers_[layer_id];
        if (kind_ == GraphConvKind::gat) {
            x = layer.gat->forward(x, edge_index, edge_attr);
        } else {
            x = layer.edge_fusion->forward(x, edge_index, edge_attr);
        }
        if (layer.has_norm) x = layer.norm->forward(x);
        if (layer.has_dropout) x = layer.dropout->forward(x);
        if (layer_id + 1 != layers_.size()) {
            x = torch::leaky_relu(x, 0.01);
        }
    }
    if (options_.return_batch) {
        x = pyg_lite::to_dense_batch(
                x, batch, 0.0, max_num_nodes, batch_size)
                .values;
    } else if (graph_pooling_) {
        x = graph_pooling_->forward(x, batch, batch_size);
    }
    return x;
}

GATConvNetImpl::GATConvNetImpl(GraphConvNetOptions options)
    : GraphConvNetImpl(std::move(options), GraphConvKind::gat) {}

EdgeFusionGATConvNetImpl::EdgeFusionGATConvNetImpl(
    GraphConvNetOptions options)
    : GraphConvNetImpl(std::move(options), GraphConvKind::edge_fusion_gat) {}

DeepEdgeFeatureGATImpl::DeepEdgeFeatureGATImpl(
    DeepEdgeFeatureGATOptions options)
    : options_(validate_options(std::move(options))) {
    if (options_.pooling != GraphPoolingKind::none) {
        graph_pooling_ = register_module(
            "graph_pooling",
            GraphPoolingModule(options_.pooling, options_.output_dim));
    }
    conv_s = register_module(
        "conv_s",
        pyg_lite::GATConv(pyg_lite::GATConvOptions{
            options_.input_dim,
            options_.embedding_dim,
            1,
            true,
            0.2,
            0.0,
            true,
            options_.edge_dim,
            pyg_lite::SelfLoopFill::reduce(
                pyg_lite::SelfLoopFillKind::mean),
            true,
            false,
        }));
    mid_layers_.reserve(
        static_cast<std::size_t>(options_.num_layers - 2));
    for (std::int64_t layer_id = 0;
         layer_id < options_.num_layers - 2;
         ++layer_id) {
        const std::string layer_name = std::to_string(layer_id);
        MidLayer layer;
        layer.conv = register_module(
            "conv_" + layer_name,
            pyg_lite::GATConv(pyg_lite::GATConvOptions{
                options_.embedding_dim,
                options_.embedding_dim,
                1,
                true,
                0.2,
                0.0,
                true,
                options_.edge_dim,
                pyg_lite::SelfLoopFill::reduce(
                    pyg_lite::SelfLoopFillKind::mean),
                true,
                false,
            }));
        if (options_.batch_norm) {
            layer.has_norm = true;
            layer.norm = register_module(
                "norm_" + layer_name,
                torch::nn::BatchNorm1d(options_.embedding_dim));
        } else {
            register_module("norm_" + layer_name, torch::nn::Identity());
        }
        if (options_.dropout_prob < 1.0) {
            layer.has_dropout = true;
            layer.dropout = register_module(
                "dout_" + layer_name,
                torch::nn::Dropout(options_.dropout_prob));
        } else {
            register_module("dout_" + layer_name, torch::nn::Identity());
        }
        layer.weight = register_parameter(
            "weight_" + layer_name,
            torch::empty({options_.embedding_dim, options_.embedding_dim}));
        mid_layers_.push_back(std::move(layer));
    }
    conv_e = register_module(
        "conv_e",
        pyg_lite::GATConv(pyg_lite::GATConvOptions{
            options_.embedding_dim,
            options_.output_dim,
            1,
            true,
            0.2,
            0.0,
            true,
            options_.edge_dim,
            pyg_lite::SelfLoopFill::reduce(
                pyg_lite::SelfLoopFillKind::mean),
            true,
            false,
        }));
    if (options_.initialize_residual_weights) {
        for (auto& layer : mid_layers_) torch::nn::init::orthogonal_(layer.weight);
    }
}

torch::Tensor DeepEdgeFeatureGATImpl::forward(
    const pyg_lite::Data& input) {
    require_graph_input(input.x, input.edge_index, options_.input_dim);
    require_edge_attributes(
        input.x,
        input.edge_index,
        input.edge_attr,
        options_.edge_dim,
        "DeepEdgeFeatureGAT");
    return forward_impl(
        input.x,
        input.edge_index,
        input.edge_attr,
        input.batch,
        std::nullopt,
        std::nullopt);
}

torch::Tensor DeepEdgeFeatureGATImpl::forward(
    const pyg_lite::Batch& input) {
    require_graph_input(input.x, input.edge_index, options_.input_dim);
    require_edge_attributes(
        input.x,
        input.edge_index,
        input.edge_attr,
        options_.edge_dim,
        "DeepEdgeFeatureGAT");
    return forward_impl(
        input.x,
        input.edge_index,
        input.edge_attr,
        input.batch,
        input.max_num_nodes,
        input.num_graphs);
}

torch::Tensor DeepEdgeFeatureGATImpl::forward_impl(
    const torch::Tensor& input,
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& edge_attr,
    const std::optional<torch::Tensor>& batch,
    const std::optional<std::int64_t> max_num_nodes,
    const std::optional<std::int64_t> batch_size) {
    const auto initial = conv_s->forward(input, edge_index, edge_attr);
    auto x = initial;
    for (std::size_t layer_id = 0;
         layer_id < mid_layers_.size();
         ++layer_id) {
        auto& layer = mid_layers_[layer_id];
        auto convolved = layer.conv->forward(x, edge_index, edge_attr);
        const double beta = std::log(
            options_.theta / static_cast<double>(layer_id + 1) + 1.0);
        convolved.mul_(1.0 - options_.alpha);
        x = convolved.add(options_.alpha * initial);
        x = torch::addmm(x, x, layer.weight, 1.0 - beta, beta);
        if (layer.has_norm) x = layer.norm->forward(x);
        if (layer.has_dropout) x = layer.dropout->forward(x);
        x = torch::leaky_relu(x, 0.01);
    }
    x = conv_e->forward(x, edge_index, edge_attr);
    if (options_.return_batch) {
        x = pyg_lite::to_dense_batch(
                x, batch, 0.0, max_num_nodes, batch_size)
                .values;
    }
    if (graph_pooling_) {
        x = graph_pooling_->forward(x, batch, batch_size);
    }
    return x;
}

}  // namespace virne::solver::learning::neural_network
