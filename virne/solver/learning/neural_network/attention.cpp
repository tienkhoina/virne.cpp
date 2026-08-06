#include "attention.h"

#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

PositionalEncoderOptions validate_options(PositionalEncoderOptions options) {
    if (options.embedding_dim <= 0 || options.embedding_dim % 2 != 0) {
        throw std::invalid_argument(
            "PositionalEncoder embedding_dim must be a positive even value");
    }
    if (options.max_len <= 0) {
        throw std::invalid_argument(
            "PositionalEncoder max_len must be positive");
    }
    if (!std::isfinite(options.dropout_prob)
        || options.dropout_prob < 0.0
        || options.dropout_prob > 1.0) {
        throw std::invalid_argument(
            "PositionalEncoder dropout_prob must be in [0, 1]");
    }
    return options;
}

torch::Tensor make_positional_buffer(
    const PositionalEncoderOptions& options) {
    const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto pe = torch::zeros(
        {options.max_len, options.embedding_dim}, tensor_options);
    const auto position = torch::arange(
        options.max_len,
        torch::TensorOptions().dtype(torch::kInt64))
        .unsqueeze(1)
        .to(torch::kFloat32);
    const auto even = torch::arange(
        0,
        options.embedding_dim,
        2,
        tensor_options);
    const auto div_term = torch::exp(
        even * (-std::log(10000.0) / options.embedding_dim));
    const auto angles = position * div_term;
    using torch::indexing::Slice;
    pe.index({Slice(), Slice(0, torch::indexing::None, 2)})
        .copy_(torch::sin(angles));
    pe.index({Slice(), Slice(1, torch::indexing::None, 2)})
        .copy_(torch::cos(angles));
    return pe.unsqueeze(0);
}

}  // namespace

PositionalEncoderImpl::PositionalEncoderImpl(PositionalEncoderOptions options)
    : options_(validate_options(std::move(options))) {
    pe = register_buffer("pe", make_positional_buffer(options_));
    if (options_.dropout_prob < 1.0) {
        has_dropout_ = true;
        dropout = register_module(
            "dropout",
            torch::nn::Dropout(torch::nn::DropoutOptions(
                options_.dropout_prob)));
    }
}

torch::Tensor PositionalEncoderImpl::forward(const torch::Tensor& input) {
    if (!input.defined() || input.dim() != 3) {
        throw std::invalid_argument(
            "PositionalEncoder input must be a defined rank-3 tensor");
    }
    const auto sequence_length = input.size(1);
    if (sequence_length <= 0 || sequence_length > options_.max_len) {
        throw std::invalid_argument(
            "PositionalEncoder input sequence exceeds max_len");
    }
    const auto embeddings = pe.narrow(1, 0, sequence_length).repeat(
        {input.size(0), 1, 1});
    torch::Tensor output;
    if (options_.method == PositionalEncodingMethod::add) {
        if (input.size(2) != options_.embedding_dim) {
            throw std::invalid_argument(
                "PositionalEncoder add input width differs from embedding_dim");
        }
        output = input + embeddings;
    } else {
        output = torch::cat({input, embeddings}, -1);
    }
    return has_dropout_ ? dropout->forward(output) : output;
}

}  // namespace virne::solver::learning::neural_network
