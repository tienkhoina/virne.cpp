#pragma once

#include "../pyg_lite/message_passing.h"
#include "../pyg_lite/pyg_lite.h"

#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace virne::solver::learning::neural_network {

enum class GnnKind : std::uint8_t {
    gcn,
    gat,
    edge_fusion_gat,
    deep_edge_gat,
};

// Resolve a dynamic configuration name once at the API boundary. The enum is
// used by all subsequent dispatch and hot loops.
GnnKind gnn_kind_from_name(std::string_view name);

torch::Tensor to_sparse_batch(
    const torch::Tensor& node_dense_embeddings,
    const torch::Tensor& mask);

struct DenseToSparseOutput {
    torch::Tensor x;
    torch::Tensor edge_index;
    torch::Tensor edge_weight;
    torch::Tensor batch;
};

class DenseToSparseImpl final : public torch::nn::Module {
public:
    DenseToSparseOutput forward(
        const torch::Tensor& x,
        const torch::Tensor& adjacency,
        const torch::Tensor& weight) const;
};
TORCH_MODULE(DenseToSparse);

enum class GraphPoolingKind : std::uint8_t {
    none,
    add,
    mean,
    max,
    attention,
};

GraphPoolingKind graph_pooling_kind_from_name(std::string_view name);

class GraphPooling final {
public:
    explicit GraphPooling(GraphPoolingKind kind);

    torch::Tensor forward(
        const torch::Tensor& node_embeddings,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> size = std::nullopt) const;

    [[nodiscard]] GraphPoolingKind kind() const noexcept { return kind_; }

private:
    GraphPoolingKind kind_ = GraphPoolingKind::none;
};

class GraphAttentionPoolingImpl final : public torch::nn::Module {
public:
    explicit GraphAttentionPoolingImpl(std::int64_t input_dim);

    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& batch,
        std::optional<std::int64_t> size = std::nullopt) const;
    torch::Tensor get_coefs(const torch::Tensor& x) const;

    [[nodiscard]] std::int64_t input_dim() const noexcept { return input_dim_; }

    torch::Tensor weight;

private:
    std::int64_t input_dim_ = 0;
};
TORCH_MODULE(GraphAttentionPooling);

// Mirrors Python GraphPooling's nested `pooling` module, preserving the
// `graph_pooling.pooling.weight` state-dict path for attention pooling.
class GraphPoolingModuleImpl final : public torch::nn::Module {
public:
    GraphPoolingModuleImpl(GraphPoolingKind kind, std::int64_t input_dim);

    torch::Tensor forward(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> size = std::nullopt) const;

private:
    GraphPoolingKind kind_ = GraphPoolingKind::none;
    GraphAttentionPooling pooling{nullptr};
};
TORCH_MODULE(GraphPoolingModule);

struct GCNConvNetOptions {
    std::int64_t input_dim = 0;
    std::int64_t output_dim = 0;
    std::int64_t embedding_dim = 128;
    std::int64_t num_layers = 3;
    bool batch_norm = true;
    double dropout_prob = 1.0;
    bool return_batch = false;
    GraphPoolingKind pooling = GraphPoolingKind::none;
};

class GCNConvNetImpl final : public torch::nn::Module {
public:
    explicit GCNConvNetImpl(GCNConvNetOptions options);

    torch::Tensor forward(const pyg_lite::Data& input);
    torch::Tensor forward(const pyg_lite::Batch& input);

    [[nodiscard]] const GCNConvNetOptions& options() const noexcept {
        return options_;
    }

private:
    struct Layer {
        pyg_lite::GCNConv conv{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::Dropout dropout{nullptr};
        bool has_norm = false;
        bool has_dropout = false;
    };

    torch::Tensor forward_impl(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> max_num_nodes,
        std::optional<std::int64_t> batch_size);

    GCNConvNetOptions options_{};
    std::vector<Layer> layers_;
    std::optional<GraphPooling> graph_pooling_;
    GraphPoolingModule learned_graph_pooling_{nullptr};
};

TORCH_MODULE(GCNConvNet);

struct GraphConvNetOptions {
    std::int64_t input_dim = 0;
    std::int64_t output_dim = 0;
    std::int64_t num_layers = 3;
    std::int64_t embedding_dim = 128;
    // Python's GraphConvNet passes `heads=` but its get_conv reads
    // `num_heads`; consequently the viable reference path always uses one
    // head. Keep the requested value for config parity and use one internally.
    std::int64_t num_heads = 1;
    std::optional<std::int64_t> edge_dim;
    bool batch_norm = false;
    double dropout_prob = 1.0;
    bool return_batch = false;
    GraphPoolingKind pooling = GraphPoolingKind::none;
};

enum class GraphConvKind : std::uint8_t {
    gat,
    edge_fusion_gat,
};

class GraphConvNetImpl : public torch::nn::Module {
public:
    GraphConvNetImpl(GraphConvNetOptions options, GraphConvKind kind);

    torch::Tensor forward(const pyg_lite::Data& input);
    torch::Tensor forward(const pyg_lite::Batch& input);

    [[nodiscard]] const GraphConvNetOptions& options() const noexcept {
        return options_;
    }
    [[nodiscard]] GraphConvKind kind() const noexcept { return kind_; }

protected:
    struct Layer {
        pyg_lite::GATConv gat{nullptr};
        pyg_lite::EdgeFusionGATConv edge_fusion{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::Dropout dropout{nullptr};
        bool has_norm = false;
        bool has_dropout = false;
    };

private:
    torch::Tensor forward_impl(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        const std::optional<torch::Tensor>& edge_attr,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> max_num_nodes,
        std::optional<std::int64_t> batch_size);

    GraphConvNetOptions options_{};
    GraphConvKind kind_ = GraphConvKind::gat;
    std::vector<Layer> layers_;
    GraphPoolingModule graph_pooling_{nullptr};
};

class GATConvNetImpl final : public GraphConvNetImpl {
public:
    explicit GATConvNetImpl(GraphConvNetOptions options);
};
TORCH_MODULE(GATConvNet);

class EdgeFusionGATConvNetImpl final : public GraphConvNetImpl {
public:
    explicit EdgeFusionGATConvNetImpl(GraphConvNetOptions options);
};
TORCH_MODULE(EdgeFusionGATConvNet);

struct DeepEdgeFeatureGATOptions {
    std::int64_t input_dim = 0;
    std::int64_t output_dim = 0;
    std::optional<std::int64_t> edge_dim;
    std::int64_t num_layers = 5;
    double alpha = 0.2;
    double theta = 0.2;
    std::int64_t embedding_dim = 128;
    std::int64_t num_heads = 1;
    bool batch_norm = false;
    double dropout_prob = 1.0;
    bool return_batch = false;
    GraphPoolingKind pooling = GraphPoolingKind::none;
    // A parent which initializes every parameter itself may defer this reset.
    bool initialize_residual_weights = true;
};

class DeepEdgeFeatureGATImpl final : public torch::nn::Module {
public:
    explicit DeepEdgeFeatureGATImpl(DeepEdgeFeatureGATOptions options);

    torch::Tensor forward(const pyg_lite::Data& input);
    torch::Tensor forward(const pyg_lite::Batch& input);

    [[nodiscard]] const DeepEdgeFeatureGATOptions& options() const noexcept {
        return options_;
    }

    pyg_lite::GATConv conv_s{nullptr};
    pyg_lite::GATConv conv_e{nullptr};

private:
    struct MidLayer {
        pyg_lite::GATConv conv{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::Tensor weight;
        bool has_norm = false;
        bool has_dropout = false;
    };

    torch::Tensor forward_impl(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        const std::optional<torch::Tensor>& edge_attr,
        const std::optional<torch::Tensor>& batch,
        std::optional<std::int64_t> max_num_nodes,
        std::optional<std::int64_t> batch_size);

    DeepEdgeFeatureGATOptions options_{};
    std::vector<MidLayer> mid_layers_;
    GraphPoolingModule graph_pooling_{nullptr};
};
TORCH_MODULE(DeepEdgeFeatureGAT);

}  // namespace virne::solver::learning::neural_network
