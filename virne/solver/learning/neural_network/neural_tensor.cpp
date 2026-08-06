#include "neural_tensor.h"

#include <torch/nn/init.h>
#include <torch/torch.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

NeuralTensorNetworkOptions validate_options(
    NeuralTensorNetworkOptions options) {
    if (options.input_features <= 0 || options.tensor_neurons <= 0) {
        throw std::invalid_argument(
            "NeuralTensorNetwork input_features and tensor_neurons must be positive");
    }
    return options;
}

void require_embedding(
    const torch::Tensor& embedding,
    const std::int64_t input_features,
    const char* name) {
    if (!embedding.defined() || embedding.dim() != 2
        || embedding.size(1) != input_features) {
        throw std::invalid_argument(
            std::string("NeuralTensorNetwork ") + name
            + " must have rank [batch, input_features]");
    }
}

}  // namespace

NeuralTensorNetworkImpl::NeuralTensorNetworkImpl(
    NeuralTensorNetworkOptions options)
    : options_(validate_options(std::move(options))) {
    weight_matrix = register_parameter(
        "weight_matrix",
        torch::empty({
            options_.input_features,
            options_.input_features,
            options_.tensor_neurons,
        }));
    weight_matrix_block = register_parameter(
        "weight_matrix_block",
        torch::empty({
            options_.tensor_neurons,
            2 * options_.input_features,
        }));
    bias = register_parameter(
        "bias", torch::empty({options_.tensor_neurons, 1}));

    // Python calls orthogonal_ in parameter order. The LibTorch initializer
    // uses the same generator and flattening rules for rank >= 2 tensors.
    torch::nn::init::orthogonal_(weight_matrix);
    torch::nn::init::orthogonal_(weight_matrix_block);
    torch::nn::init::orthogonal_(bias);
}

torch::Tensor NeuralTensorNetworkImpl::forward(
    const torch::Tensor& embedding_1,
    const torch::Tensor& embedding_2) {
    require_embedding(embedding_1, options_.input_features, "embedding_1");
    require_embedding(embedding_2, options_.input_features, "embedding_2");
    if (embedding_1.size(0) != embedding_2.size(0)) {
        throw std::invalid_argument(
            "NeuralTensorNetwork embedding batch sizes differ");
    }

    const auto batch_size = embedding_1.size(0);
    auto scoring = torch::matmul(
        embedding_1,
        weight_matrix.view({options_.input_features, -1}));
    scoring = scoring.view({batch_size, options_.input_features, -1})
                 .permute({0, 2, 1});
    scoring = torch::matmul(
        scoring,
        embedding_2.view({batch_size, options_.input_features, 1}))
                 .view({batch_size, -1});

    const auto combined = torch::cat({embedding_1, embedding_2}, 1);
    const auto block_scoring = torch::mm(
        weight_matrix_block, combined.transpose(0, 1)).transpose(0, 1);
    return torch::relu(scoring + block_scoring + bias.view({-1}));
}

}  // namespace virne::solver::learning::neural_network
