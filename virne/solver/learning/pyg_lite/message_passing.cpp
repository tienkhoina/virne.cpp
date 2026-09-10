#include "message_passing.h"

#include <torch/nn/init.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/linear.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace virne::solver::learning::pyg_lite {
namespace {

void require_defined(const torch::Tensor& value, const char* name) {
    if (!value.defined()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            std::string(name) + " must be a defined tensor");
    }
}

void validate_edge_index_local(const torch::Tensor& edge_index) {
    require_defined(edge_index, "edge_index");
    if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_index must have shape [2, E]");
    }
    if (edge_index.scalar_type() != torch::kInt64) {
        throw PygLiteException(
            ErrorCode::invalid_dtype,
            "edge_index must use int64 indices");
    }
}

void validate_edge_attr_local(
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& edge_attr) {
    if (!edge_attr.has_value()) return;
    require_defined(*edge_attr, "edge_attr");
    if (edge_attr->dim() < 1 || edge_attr->size(0) != edge_index.size(1)) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_attr leading dimension must equal the edge count");
    }
    if (edge_attr->device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "edge_index and edge_attr must be on the same device");
    }
}

void validate_node_features(
    const torch::Tensor& value,
    const char* name,
    const std::int64_t channels) {
    require_defined(value, name);
    if (value.dim() != 2 || value.size(1) != channels) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            std::string(name) + " must have shape [N, channels]");
    }
}

void validate_channels(
    const std::int64_t in_channels,
    const std::int64_t out_channels,
    const char* name) {
    if (in_channels <= 0 || out_channels <= 0) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            std::string(name) + " channel counts must be positive");
    }
}

void glorot_last_two(torch::Tensor value) {
    if (!value.defined()) return;
    if (value.dim() < 2) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "glorot initialization requires at least two dimensions");
    }
    const double bound = std::sqrt(
        6.0 / static_cast<double>(value.size(-2) + value.size(-1)));
    torch::NoGradGuard guard;
    value.uniform_(-bound, bound);
}

void zero_parameter(torch::Tensor value) {
    if (!value.defined()) return;
    torch::NoGradGuard guard;
    value.zero_();
}

torch::Tensor normalize_rows(const torch::Tensor& value) {
    // torch.nn.functional.normalize defaults to p=2, dim=-1, eps=1e-12.
    return value / value.norm(2, -1, true).clamp_min(1e-12);
}

}  // namespace

PropagationBatch prepare_homogeneous_propagation(
    const torch::Tensor& x_source,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr,
    PropagationOptions options,
    std::optional<torch::Tensor> x_target) {
    require_defined(x_source, "x_source");
    if (x_source.dim() < 1) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "x_source must have a leading node dimension");
    }
    validate_edge_index_local(edge_index);
    validate_edge_attr_local(edge_index, edge_attr);
    if (x_source.device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "x_source and edge_index must be on the same device");
    }

    const torch::Tensor target = x_target.has_value()
        ? *x_target
        : x_source;
    require_defined(target, "x_target");
    if (target.dim() < 1) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "x_target must have a leading node dimension");
    }
    if (target.device() != x_source.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "source and target node tensors must share a device");
    }
    if (target.scalar_type() != x_source.scalar_type()) {
        throw PygLiteException(
            ErrorCode::invalid_dtype,
            "source and target node tensors must share a dtype");
    }
    if (options.target_num_nodes.has_value()
        && (*options.target_num_nodes < target.size(0))) {
        throw PygLiteException(
            ErrorCode::invalid_index,
            "target_num_nodes cannot be smaller than x_target rows");
    }

    const std::int64_t target_num_nodes = options.target_num_nodes.value_or(
        target.size(0));
    const torch::Tensor source_index = edge_index.select(0, 0);
    const torch::Tensor target_index = edge_index.select(0, 1);
    return {
        source_index,
        target_index,
        x_source.index_select(0, source_index),
        target.index_select(0, target_index),
        std::move(edge_attr),
        target_num_nodes};
}

GraphViNEConvImpl::GraphViNEConvImpl(GraphViNEConvOptions options)
    : options_(std::move(options)) {
    validate_channels(
        options_.in_channels,
        options_.out_channels,
        "GraphViNEConv");
    lin_rel = register_module(
        "lin_rel",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.in_channels,
            options_.out_channels).bias(options_.bias)));
    lin_root = register_module(
        "lin_root",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.in_channels,
            options_.out_channels).bias(false)));
    reset_parameters();
}

void GraphViNEConvImpl::reset_parameters() {
    lin_rel->reset_parameters();
    lin_root->reset_parameters();
}

torch::Tensor GraphViNEConvImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_weight) {
    validate_node_features(x, "GraphViNEConv x", options_.in_channels);
    validate_edge_index_local(edge_index);
    if (x.device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "GraphViNEConv x and edge_index must share a device");
    }
    if (edge_weight.has_value()) {
        require_defined(*edge_weight, "edge_weight");
        if (edge_weight->device() != x.device()
            || edge_weight->numel() != edge_index.size(1)) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "GraphViNEConv edge_weight must contain one value per edge");
        }
    }

    // GraphViNE's message() only consumes x_j and an optional scalar weight.
    // Avoid the generic propagation boundary's unused x_i gather in this hot
    // homogeneous path; custom layers that need both endpoints use the typed
    // callback API above.
    const torch::Tensor source_index = edge_index.select(0, 0);
    const torch::Tensor target_index = edge_index.select(0, 1);
    torch::Tensor messages = x.index_select(0, source_index);
    if (edge_weight.has_value()) {
        messages = messages * edge_weight->reshape({-1, 1});
    }
    const torch::Tensor aggregated = scatter_dim0(
        messages, target_index, x.size(0), ScatterReduce::mean);
    torch::Tensor output = lin_rel->forward(aggregated)
        + lin_root->forward(x);
    if (options_.normalize) output = normalize_rows(output);
    return output;
}

EdgeFusionGATConvImpl::EdgeFusionGATConvImpl(
    EdgeFusionGATConvOptions options)
    : options_(std::move(options)) {
    validate_channels(
        options_.in_channels,
        options_.out_channels,
        "EdgeFusionGATConv");
    if (options_.heads != 1) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "EdgeFusionGATConv currently supports the Virne homogeneous heads=1 path");
    }
    if (options_.dropout < 0.0 || options_.dropout > 1.0) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "EdgeFusionGATConv dropout must lie in [0, 1]");
    }
    if (options_.edge_dim.has_value() && *options_.edge_dim <= 0) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "EdgeFusionGATConv edge_dim must be positive");
    }
    const std::int64_t projected =
        options_.heads * options_.out_channels;
    // Python constructs the pre-network first; registration order also controls
    // parameter traversal and optimizer state ordering.
    pre_nns = register_module("pre_nns", torch::nn::ModuleList());
    const std::int64_t message_input =
        (options_.edge_dim.has_value() ? 3 : 2) * options_.out_channels;
    auto sequence = torch::nn::Sequential();
    sequence->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(message_input, options_.out_channels)
            .bias(true)));
    sequence->push_back(torch::nn::ReLU());
    sequence->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(
            options_.out_channels,
            options_.out_channels).bias(true)));
    pre_nns->push_back(sequence);

    lin_src = register_module(
        "lin_src",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.in_channels, projected).bias(false)));
    // PyG Linear's constructor uses Glorot here, with one RNG draw per weight.
    // LibTorch Linear has already consumed the same number of uniform draws;
    // the values are overwritten by the final reset below.
    lin_dst = register_module("lin_dst", lin_src);
    if (options_.edge_dim.has_value()) {
        lin_edge = register_module(
            "lin_edge",
            torch::nn::Linear(torch::nn::LinearOptions(
                *options_.edge_dim, projected).bias(false)));
    }

    const std::int64_t total = options_.concat
        ? projected
        : options_.out_channels;
    if (options_.bias) {
        bias = register_parameter("bias", torch::empty({total}));
    }
    reset_parameters();
}

void EdgeFusionGATConvImpl::reset_parameters() {
    glorot_last_two(lin_src->weight);
    // Python resets both names even though they refer to the same parameter.
    glorot_last_two(lin_dst->weight);
    if (lin_edge) glorot_last_two(lin_edge->weight);
    auto& sequence = pre_nns->at<torch::nn::SequentialImpl>(0);
    // The fixed Python pre-network is Linear -> ReLU -> Linear. Resolve the
    // typed positions once; do not reflect over AnyModule in a hot path.
    auto& first = sequence.at<torch::nn::LinearImpl>(0);
    auto& second = sequence.at<torch::nn::LinearImpl>(2);
    // The two PyG Linear layers have no explicit weight initializer: they use
    // the same Kaiming-uniform weight and uniform bias defaults as nn.Linear.
    first.reset_parameters();
    second.reset_parameters();
    zero_parameter(bias);
}

torch::Tensor EdgeFusionGATConvImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr) {
    validate_node_features(x, "EdgeFusionGATConv x", options_.in_channels);
    validate_edge_index_local(edge_index);
    validate_edge_attr_local(edge_index, edge_attr);
    if (x.device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "EdgeFusionGATConv x and edge_index must share a device");
    }
    if (options_.edge_dim.has_value()) {
        if (!edge_attr.has_value()) {
            throw PygLiteException(
                ErrorCode::invalid_tensor,
                "EdgeFusionGATConv edge_dim requires edge_attr");
        }
        if (edge_attr->dim() != 2 || edge_attr->size(1) != *options_.edge_dim) {
            throw PygLiteException(
                ErrorCode::invalid_shape,
                "EdgeFusionGATConv edge_attr must have shape [E, edge_dim]");
        }
    } else if (edge_attr.has_value()) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "EdgeFusionGATConv edge_attr requires edge_dim");
    }

    torch::Tensor propagated_edge_index = edge_index;
    std::optional<torch::Tensor> propagated_edge_attr = std::move(edge_attr);
    if (options_.add_self_loops) {
        const LoopResult without_loops = remove_self_loops(
            propagated_edge_index,
            propagated_edge_attr);
        const LoopResult with_loops = add_self_loops(
            without_loops.edge_index,
            without_loops.edge_attr,
            options_.fill_value,
            x.size(0));
        propagated_edge_index = with_loops.edge_index;
        propagated_edge_attr = with_loops.edge_attr;
    }

    const std::int64_t heads = options_.heads;
    const std::int64_t channels = options_.out_channels;
    // The bounded Virne path is homogeneous: Python assigns x_src = x_dst
    // from one lin_src(x) result. Reuse that projection instead of executing
    // the same registered module twice; bipartite channels are deferred to a
    // separate typed component.
    const torch::Tensor source = lin_src->forward(x).view({-1, heads, channels});
    PropagationOptions propagation{};
    propagation.aggregation = options_.aggregation;
    propagation.target_num_nodes = x.size(0);
    const torch::Tensor aggregated = propagate_homogeneous(
        source,
        propagated_edge_index,
        std::move(propagated_edge_attr),
        propagation,
        [this](const torch::Tensor& x_i,
               const torch::Tensor& x_j,
               const std::optional<torch::Tensor>& edge_features) {
            torch::Tensor message_input;
            if (edge_features.has_value()) {
                const torch::Tensor edge_embedding = lin_edge->forward(
                    *edge_features).view({-1, 1, options_.out_channels});
                message_input = torch::cat(
                    {x_i, x_j, edge_embedding}, -1);
            } else {
                message_input = torch::cat({x_i, x_j}, -1);
            }
            auto& sequence = pre_nns->at<torch::nn::SequentialImpl>(0);
            // The Python implementation has one pre-network and heads=1 in
            // Virne. Keep the explicit head loop so the tensor contract stays
            // well-defined if this bounded layer is reused with more heads.
            std::vector<torch::Tensor> outputs;
            outputs.reserve(static_cast<std::size_t>(options_.heads));
            for (std::int64_t head = 0; head < options_.heads; ++head) {
                outputs.push_back(sequence.forward(message_input.select(1, head)));
            }
            return torch::stack(outputs, 1);
        });

    torch::Tensor output = options_.concat
        ? aggregated.view({-1, heads * channels})
        : aggregated.mean(1);
    if (bias.defined()) output = output + bias;
    return output;
}

}  // namespace virne::solver::learning::pyg_lite
