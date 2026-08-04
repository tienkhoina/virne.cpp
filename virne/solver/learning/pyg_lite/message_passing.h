#pragma once

#include "pyg_lite.h"

#include <torch/nn/module.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace virne::solver::learning::pyg_lite {

// The generic surface is deliberately limited to homogeneous tensor graphs.
// Dynamic PyG kwargs are resolved by the caller into this typed options object.
struct PropagationOptions {
    ScatterReduce aggregation = ScatterReduce::sum;
    std::optional<std::int64_t> target_num_nodes;
};

struct PropagationBatch {
    torch::Tensor source_index;
    torch::Tensor target_index;
    torch::Tensor x_j;
    torch::Tensor x_i;
    std::optional<torch::Tensor> edge_attr;
    std::int64_t target_num_nodes = 0;
};

// Cold boundary validation plus device-native index gathers. Callers should
// pass target_num_nodes for static graphs so no device synchronization is
// needed; the target tensor shape is otherwise used directly.
PropagationBatch prepare_homogeneous_propagation(
    const torch::Tensor& x_source,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr = std::nullopt,
    PropagationOptions options = {},
    std::optional<torch::Tensor> x_target = std::nullopt);

// MessageFn receives [E,...] target/source features and optional edge attrs.
// It must return an [E,...] tensor. The templated callback is inlined at the
// call site, while gathers and reduction remain fused ATen kernels.
template <typename MessageFn>
torch::Tensor propagate_homogeneous(
    const torch::Tensor& x_source,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr,
    PropagationOptions options,
    MessageFn&& message_fn,
    std::optional<torch::Tensor> x_target = std::nullopt) {
    const PropagationBatch prepared = prepare_homogeneous_propagation(
        x_source,
        edge_index,
        std::move(edge_attr),
        options,
        std::move(x_target));
    torch::Tensor messages = message_fn(
        prepared.x_i,
        prepared.x_j,
        prepared.edge_attr);
    if (!messages.defined() || messages.dim() < 1
        || messages.size(0) != prepared.target_index.size(0)
        || messages.device() != prepared.x_j.device()) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "message callback must return an [E,...] tensor on the input device");
    }
    return scatter_dim0(
        messages,
        prepared.target_index,
        prepared.target_num_nodes,
        options.aggregation);
}

struct GraphViNEConvOptions {
    std::int64_t in_channels = 0;
    std::int64_t out_channels = 0;
    bool normalize = false;
    bool bias = true;
};

class GraphViNEConvImpl final : public torch::nn::Module {
public:
    explicit GraphViNEConvImpl(GraphViNEConvOptions options);

    void reset_parameters();
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_weight = std::nullopt);

    const GraphViNEConvOptions& options() const noexcept { return options_; }

    // Names mirror the Python module for straightforward state transfer.
    torch::nn::Linear lin_rel{nullptr};
    torch::nn::Linear lin_root{nullptr};

private:
    GraphViNEConvOptions options_;
};
TORCH_MODULE(GraphViNEConv);

struct EdgeFusionGATConvOptions {
    std::int64_t in_channels = 0;
    std::int64_t out_channels = 0;
    std::int64_t heads = 1;
    bool concat = true;
    double negative_slope = 0.2;  // Kept for Python API compatibility.
    double dropout = 0.0;         // Attention is disabled by the Python layer.
    bool add_self_loops = true;
    std::optional<std::int64_t> edge_dim;
    SelfLoopFill fill_value = SelfLoopFill::reduce(SelfLoopFillKind::mean);
    bool bias = true;
    ScatterReduce aggregation = ScatterReduce::sum;
};

class EdgeFusionGATConvImpl final : public torch::nn::Module {
public:
    explicit EdgeFusionGATConvImpl(EdgeFusionGATConvOptions options);

    void reset_parameters();
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& edge_index,
        std::optional<torch::Tensor> edge_attr = std::nullopt);

    const EdgeFusionGATConvOptions& options() const noexcept {
        return options_;
    }

    torch::nn::Linear lin_src{nullptr};
    // For homogeneous graphs this is the same module as lin_src. It is
    // registered under both fixed names to preserve Python state-dict paths.
    torch::nn::Linear lin_dst{nullptr};
    torch::nn::Linear lin_edge{nullptr};
    torch::nn::ModuleList pre_nns{nullptr};
    torch::Tensor bias;

private:
    EdgeFusionGATConvOptions options_;
};
TORCH_MODULE(EdgeFusionGATConv);

}  // namespace virne::solver::learning::pyg_lite
