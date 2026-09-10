#include "mlp.h"

#include <torch/nn/init.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/linear.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

std::vector<std::int64_t> resolve_hidden_dims(const MlpOptions& options) {
    if (options.input_dim <= 0) {
        throw std::invalid_argument("MLPNet input_dim must be positive");
    }
    if (options.output_dim <= 0) {
        throw std::invalid_argument("MLPNet output_dim must be positive");
    }
    if (options.num_layers < 1) {
        throw std::invalid_argument("MLPNet num_layers must be positive");
    }

    const auto layer_count_u64 =
        static_cast<std::uint64_t>(options.num_layers);
    if (layer_count_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("MLPNet num_layers is too large");
    }
    const auto hidden_count_u64 = layer_count_u64 - 1U;
    const auto hidden_count = static_cast<std::size_t>(hidden_count_u64);

    if (std::holds_alternative<std::monostate>(options.embedding_dims)) {
        if (options.input_dim >
            std::numeric_limits<std::int64_t>::max() / 2) {
            throw std::invalid_argument(
                "MLPNet default hidden dimension overflows int64");
        }
        return std::vector<std::int64_t>(
            hidden_count,
            options.input_dim * 2);
    }

    if (const auto* repeated =
            std::get_if<std::int64_t>(&options.embedding_dims)) {
        if (*repeated <= 0) {
            throw std::invalid_argument(
                "MLPNet repeated hidden dimension must be positive");
        }
        return std::vector<std::int64_t>(hidden_count, *repeated);
    }

    const auto& explicit_dims =
        std::get<std::vector<std::int64_t>>(options.embedding_dims);
    if (explicit_dims.size() != hidden_count) {
        throw std::invalid_argument(
            "MLPNet explicit hidden dimensions must match num_layers - 1");
    }
    for (const std::int64_t dimension : explicit_dims) {
        if (dimension <= 0) {
            throw std::invalid_argument(
                "MLPNet explicit hidden dimensions must be positive");
        }
    }
    return explicit_dims;
}

} // namespace

MLPNetImpl::MLPNetImpl(MlpOptions options)
    : options_(std::move(options)),
      resolved_hidden_dims_(resolve_hidden_dims(options_)) {
    lins = register_module("lins", torch::nn::Sequential());

    // Retain handles only until initialization. Sequential owns the same
    // module implementations, without registering a second set of names.
    std::vector<torch::nn::Linear> linear_layers;
    linear_layers.reserve(static_cast<std::size_t>(options_.num_layers));

    for (std::int64_t layer_id = 0;
         layer_id < options_.num_layers;
         ++layer_id) {
        const bool final_layer = layer_id == options_.num_layers - 1;
        const auto layer_index = static_cast<std::size_t>(layer_id);
        const std::int64_t input_dimension = layer_id == 0
            ? options_.input_dim
            : resolved_hidden_dims_[layer_index - 1U];
        const std::int64_t output_dimension = final_layer
            ? options_.output_dim
            : resolved_hidden_dims_[layer_index];

        linear_layers.emplace_back(
            torch::nn::LinearOptions(input_dimension, output_dimension));
        lins->push_back(linear_layers.back());
        Layer layer;
        layer.linear = linear_layers.back();
        layer.activation = !final_layer;

        if (final_layer) {
            lins->push_back(torch::nn::Identity());
            layers_.push_back(std::move(layer));
            continue;
        }

        if (options_.batch_norm) {
            layer.norm = torch::nn::BatchNorm1d(output_dimension);
            lins->push_back(layer.norm);
        } else {
            lins->push_back(torch::nn::Identity());
        }
        lins->push_back(torch::nn::LeakyReLU(
            torch::nn::LeakyReLUOptions()
                .negative_slope(0.01)
                .inplace(false)));
        layers_.push_back(std::move(layer));
    }

    // Python constructs every Linear (including its default bias) first, then
    // traverses the complete Sequential and replaces only Linear weights with
    // orthogonal initialization. Keeping that order preserves RNG consumption.
    for (auto& linear : linear_layers) {
        torch::nn::init::orthogonal_(linear->weight);
    }
}

torch::Tensor MLPNetImpl::forward(const torch::Tensor& input) {
    auto x = input;
    auto output_shape = input.sizes().vec();
    const bool flatten_features = options_.batch_norm && input.dim() > 2;
    if (flatten_features) x = x.reshape({-1, options_.input_dim});
    for (auto& layer : layers_) {
        // ATen linear matches Python's fused rank-3 projection/backward path.
        x = torch::linear(x, layer.linear->weight, layer.linear->bias);
        if (layer.norm) x = layer.norm->forward(x);
        if (layer.activation) x = torch::leaky_relu(x, 0.01);
    }
    if (flatten_features) {
        output_shape.back() = options_.output_dim;
        x = x.reshape(output_shape);
    }
    return x;
}

const MlpOptions& MLPNetImpl::options() const noexcept {
    return options_;
}

const std::vector<std::int64_t>&
MLPNetImpl::resolved_hidden_dims() const noexcept {
    return resolved_hidden_dims_;
}

} // namespace virne::solver::learning::neural_network
