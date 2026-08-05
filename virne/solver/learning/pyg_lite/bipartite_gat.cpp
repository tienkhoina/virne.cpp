#include "bipartite_gat.h"

#include <cmath>
#include <string>

namespace virne::solver::learning::pyg_lite {

namespace {

void require_defined(const torch::Tensor& value, const char* name) {
    if (!value.defined()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            std::string(name) + " must be a defined tensor");
    }
}

void validate_edge_index(const torch::Tensor& edge_index) {
    require_defined(edge_index, "edge_index");
    if (edge_index.dim() != 2 || edge_index.size(0) != 2
        || edge_index.scalar_type() != torch::kInt64) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_index must have shape [2, E] and int64 dtype");
    }
}

void validate_edge_attr(
    const torch::Tensor& edge_index,
    const std::optional<torch::Tensor>& edge_attr,
    const std::optional<std::int64_t> edge_dim) {
    if (!edge_attr.has_value()) {
        if (edge_dim.has_value()) return;
        return;
    }
    require_defined(*edge_attr, "edge_attr");
    if (edge_attr->dim() < 1 || edge_attr->size(0) != edge_index.size(1)
        || edge_attr->device() != edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_attr leading dimension/device mismatch");
    }
    if (edge_dim.has_value()
        && (edge_attr->dim() != 2 || edge_attr->size(1) != *edge_dim)) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "edge_attr must have shape [E, edge_dim]");
    }
}

void validate_features(
    const torch::Tensor& source,
    const torch::Tensor& target,
    const BipartiteGATConvOptions& options) {
    require_defined(source, "x_source");
    require_defined(target, "x_target");
    if (source.dim() != 2 || target.dim() != 2
        || source.size(1) != options.source_in_channels
        || target.size(1) != options.target_in_channels) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "bipartite GAT inputs must have the configured [N, F] shapes");
    }
}

void glorot_last_two(torch::Tensor value) {
    if (!value.defined()) return;
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

}  // namespace

BipartiteGATConvImpl::BipartiteGATConvImpl(
    BipartiteGATConvOptions options)
    : options_(std::move(options)) {
    if (options_.source_in_channels <= 0
        || options_.target_in_channels <= 0
        || options_.out_channels <= 0
        || options_.heads <= 0) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "bipartite GAT channel/head counts must be positive");
    }
    if (options_.dropout < 0.0 || options_.dropout > 1.0) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "bipartite GAT dropout must lie in [0, 1]");
    }
    if (options_.edge_dim.has_value() && *options_.edge_dim <= 0) {
        throw PygLiteException(
            ErrorCode::invalid_shape,
            "bipartite GAT edge_dim must be positive");
    }
    const auto projected = options_.heads * options_.out_channels;
    lin_src = register_module(
        "lin_src",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.source_in_channels, projected).bias(false)));
    lin_dst = register_module(
        "lin_dst",
        torch::nn::Linear(torch::nn::LinearOptions(
            options_.target_in_channels, projected).bias(false)));
    att_src = register_parameter(
        "att_src", torch::empty({1, options_.heads, options_.out_channels}));
    att_dst = register_parameter(
        "att_dst", torch::empty({1, options_.heads, options_.out_channels}));
    if (options_.edge_dim.has_value()) {
        lin_edge = register_module(
            "lin_edge",
            torch::nn::Linear(torch::nn::LinearOptions(
                *options_.edge_dim, projected).bias(false)));
        att_edge = register_parameter(
            "att_edge", torch::empty({1, options_.heads, options_.out_channels}));
    }
    const auto total = options_.concat ? projected : options_.out_channels;
    if (options_.residual) {
        if (options_.source_in_channels != options_.target_in_channels) {
            throw PygLiteException(
                ErrorCode::unsupported_option,
                "bipartite GAT residual requires equal source/target channels");
        }
        res = register_module(
            "res",
            torch::nn::Linear(torch::nn::LinearOptions(
                options_.target_in_channels, total).bias(false)));
    }
    if (options_.bias) bias = register_parameter("bias", torch::empty({total}));
    reset_parameters();
}

void BipartiteGATConvImpl::reset_parameters() {
    torch::nn::init::xavier_uniform_(lin_src->weight);
    torch::nn::init::xavier_uniform_(lin_dst->weight);
    if (lin_edge) torch::nn::init::xavier_uniform_(lin_edge->weight);
    if (res) torch::nn::init::xavier_uniform_(res->weight);
    glorot_last_two(att_src);
    glorot_last_two(att_dst);
    glorot_last_two(att_edge);
    zero_parameter(bias);
}

void BipartiteGATConvImpl::to(
    torch::Device device,
    torch::Dtype dtype,
    const bool non_blocking) {
    torch::nn::Module::to(device, dtype, non_blocking);
    if (options_.fill_value.kind == SelfLoopFillKind::tensor
        && options_.fill_value.value.has_value()) {
        options_.fill_value.value = options_.fill_value.value->to(
            device, dtype, non_blocking, false);
    }
}

void BipartiteGATConvImpl::to(
    torch::Dtype dtype,
    const bool non_blocking) {
    torch::nn::Module::to(dtype, non_blocking);
    if (options_.fill_value.kind == SelfLoopFillKind::tensor
        && options_.fill_value.value.has_value()) {
        options_.fill_value.value = options_.fill_value.value->to(
            dtype, non_blocking, false);
    }
}

void BipartiteGATConvImpl::to(
    torch::Device device,
    const bool non_blocking) {
    torch::nn::Module::to(device, non_blocking);
    if (options_.fill_value.kind == SelfLoopFillKind::tensor
        && options_.fill_value.value.has_value()) {
        options_.fill_value.value = options_.fill_value.value->to(
            device,
            options_.fill_value.value->scalar_type(),
            non_blocking,
            false);
    }
}

GATForwardResult BipartiteGATConvImpl::forward_with_attention(
    const torch::Tensor& x_source,
    const torch::Tensor& x_target,
    const torch::Tensor& original_edge_index,
    std::optional<torch::Tensor> edge_attr) {
    validate_features(x_source, x_target, options_);
    validate_edge_index(original_edge_index);
    validate_edge_attr(original_edge_index, edge_attr, options_.edge_dim);
    if (x_source.device() != x_target.device()
        || x_source.device() != original_edge_index.device()) {
        throw PygLiteException(
            ErrorCode::invalid_tensor,
            "bipartite GAT inputs must share a device");
    }
    if (options_.add_self_loops
        && (x_source.size(0) != x_target.size(0))) {
        throw PygLiteException(
            ErrorCode::unsupported_option,
            "bipartite GAT self-loops require equal source/target node counts");
    }

    const auto source_projected = lin_src->forward(x_source).view({
        -1, options_.heads, options_.out_channels});
    const auto target_projected = lin_dst->forward(x_target).view({
        -1, options_.heads, options_.out_channels});
    torch::Tensor edge_index = original_edge_index;
    if (options_.add_self_loops) {
        LoopResult without = remove_self_loops(edge_index, std::move(edge_attr));
        LoopResult with = add_self_loops(
            without.edge_index,
            std::move(without.edge_attr),
            options_.fill_value,
            x_target.size(0));
        edge_index = std::move(with.edge_index);
        edge_attr = std::move(with.edge_attr);
    }

    const auto source_index = edge_index.select(0, 0);
    const auto target_index = edge_index.select(0, 1);
    torch::Tensor alpha =
        (source_projected.index_select(0, source_index) * att_src).sum(-1)
        + (target_projected.index_select(0, target_index) * att_dst).sum(-1);
    if (edge_attr.has_value() && lin_edge) {
        torch::Tensor attributes = *edge_attr;
        if (attributes.dim() == 1) attributes = attributes.view({-1, 1});
        const auto projected_edges = lin_edge->forward(attributes).view({
            -1, options_.heads, options_.out_channels});
        alpha = alpha + (projected_edges * att_edge).sum(-1);
    }
    alpha = torch::leaky_relu(alpha, options_.negative_slope);
    alpha = segment_softmax(alpha, target_index, x_target.size(0));
    alpha = torch::dropout(alpha, options_.dropout, is_training());
    const auto messages = alpha.unsqueeze(-1)
        * source_projected.index_select(0, source_index);
    torch::Tensor output = scatter_dim0(
        messages, target_index, x_target.size(0), ScatterReduce::sum);
    output = options_.concat
        ? output.view({-1, options_.heads * options_.out_channels})
        : output.mean(1);
    if (res) output = output + res->forward(x_target);
    if (bias.defined()) output = output + bias;
    return {output, edge_index, alpha};
}

torch::Tensor BipartiteGATConvImpl::forward(
    const torch::Tensor& x_source,
    const torch::Tensor& x_target,
    const torch::Tensor& edge_index,
    std::optional<torch::Tensor> edge_attr) {
    return forward_with_attention(
        x_source, x_target, edge_index, std::move(edge_attr)).output;
}

}  // namespace virne::solver::learning::pyg_lite
