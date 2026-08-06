#include "attention.h"

#include <torch/torch.h>

#include <cmath>
#include <string>
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

namespace {

MultiHeadSelfAttentionOptions validate_attention_options(
    MultiHeadSelfAttentionOptions options) {
    if (options.input_dim <= 0 || options.num_heads <= 0) {
        throw std::invalid_argument(
            "MultiHeadSelfAttention input_dim and num_heads must be positive");
    }
    if (options.val_dim <= 0) {
        if (options.embed_dim <= 0
            || options.embed_dim % options.num_heads != 0) {
            throw std::invalid_argument(
                "MultiHeadSelfAttention embed_dim must divide by num_heads");
        }
        options.val_dim = options.embed_dim / options.num_heads;
    }
    if (options.key_dim <= 0) options.key_dim = options.val_dim;
    if (options.key_dim <= 0 || options.val_dim <= 0) {
        throw std::invalid_argument(
            "MultiHeadSelfAttention key_dim and val_dim must be positive");
    }
    return options;
}

void require_attention_input(
    const torch::Tensor& value,
    const std::int64_t input_dim,
    const char* name) {
    if (!value.defined() || value.dim() != 3
        || value.size(2) != input_dim) {
        throw std::invalid_argument(
            std::string("MultiHeadSelfAttention ") + name
            + " must have rank [batch, sequence, input_dim]");
    }
}

}  // namespace

MultiHeadSelfAttentionImpl::MultiHeadSelfAttentionImpl(
    MultiHeadSelfAttentionOptions options)
    : options_(validate_attention_options(std::move(options))),
      norm_factor_(1.0 / std::sqrt(static_cast<double>(options_.key_dim))) {
    W_query = register_parameter(
        "W_query",
        torch::empty({
            options_.num_heads,
            options_.input_dim,
            options_.key_dim,
        }));
    W_key = register_parameter(
        "W_key",
        torch::empty({
            options_.num_heads,
            options_.input_dim,
            options_.key_dim,
        }));
    torch::NoGradGuard no_grad;
    const auto query_std = 1.0 / std::sqrt(
        static_cast<double>(W_query.size(-1)));
    W_query.uniform_(-query_std, query_std);
    const auto key_std = 1.0 / std::sqrt(
        static_cast<double>(W_key.size(-1)));
    W_key.uniform_(-key_std, key_std);
}

torch::Tensor MultiHeadSelfAttentionImpl::forward(
    const torch::Tensor& query,
    const std::optional<torch::Tensor>& key,
    const std::optional<torch::Tensor>& value,
    const std::optional<torch::Tensor>& attention_mask) {
    (void)value;
    (void)attention_mask;
    const auto& actual_key = key.has_value() ? *key : query;
    require_attention_input(query, options_.input_dim, "query");
    require_attention_input(actual_key, options_.input_dim, "key");
    if (query.size(0) != actual_key.size(0)) {
        throw std::invalid_argument(
            "MultiHeadSelfAttention query/key batch sizes differ");
    }
    const auto batch_size = actual_key.size(0);
    const auto graph_size = actual_key.size(1);
    const auto num_query = query.size(1);
    const auto kflat = actual_key.contiguous().view({-1, options_.input_dim});
    const auto qflat = query.contiguous().view({-1, options_.input_dim});
    const auto projected_query = torch::matmul(qflat, W_query).view({
        options_.num_heads,
        batch_size,
        num_query,
        options_.key_dim,
    });
    const auto projected_key = torch::matmul(kflat, W_key).view({
        options_.num_heads,
        batch_size,
        graph_size,
        options_.key_dim,
    });
    const auto compatibility_raw = norm_factor_ * torch::matmul(
        projected_query,
        projected_key.transpose(2, 3));
    return torch::tanh(compatibility_raw.mean(0)) * 10.0;
}

namespace {

MultiHeadAttentionWithOutputOptions validate_output_attention_options(
    MultiHeadAttentionWithOutputOptions options) {
    if (options.embedding_dim <= 0 || options.num_heads <= 0) {
        throw std::invalid_argument(
            "MultiHeadAttentionWithOutput embedding_dim and num_heads must be positive");
    }
    if (options.d_k < 0 || options.d_v < 0) {
        throw std::invalid_argument(
            "MultiHeadAttentionWithOutput d_k and d_v cannot be negative");
    }
    if (options.d_k == 0) options.d_k = options.embedding_dim;
    if (options.d_v == 0) options.d_v = options.embedding_dim;
    if (!std::isfinite(options.dropout)
        || options.dropout < 0.0
        || options.dropout > 1.0) {
        throw std::invalid_argument(
            "MultiHeadAttentionWithOutput dropout must be in [0, 1]");
    }
    return options;
}

void require_output_attention_input(
    const torch::Tensor& input,
    const std::int64_t embedding_dim,
    const char* name) {
    if (!input.defined() || input.dim() != 3
        || input.size(2) != embedding_dim) {
        throw std::invalid_argument(
            std::string("MultiHeadAttentionWithOutput ") + name
            + " must have rank [batch, sequence, embedding_dim]");
    }
}

}  // namespace

MultiHeadAttentionWithOutputImpl::MultiHeadAttentionWithOutputImpl(
    MultiHeadAttentionWithOutputOptions options)
    : options_(validate_output_attention_options(std::move(options))),
      scale_(std::sqrt(static_cast<float>(options_.d_k))) {
    const auto qk_width = options_.num_heads * options_.d_k;
    const auto v_width = options_.num_heads * options_.d_v;
    W_q = register_module(
        "W_q",
        torch::nn::Linear(torch::nn::LinearOptions(
                              options_.embedding_dim, qk_width)
                              .bias(false)));
    W_k = register_module(
        "W_k",
        torch::nn::Linear(torch::nn::LinearOptions(
                              options_.embedding_dim, qk_width)
                              .bias(false)));
    W_v = register_module(
        "W_v",
        torch::nn::Linear(torch::nn::LinearOptions(
                              options_.embedding_dim, v_width)
                              .bias(false)));
    dropout = register_module(
        "dropout", torch::nn::Dropout(torch::nn::DropoutOptions(
                          options_.dropout)));
    fc = register_module(
        "fc",
        torch::nn::Linear(torch::nn::LinearOptions(
            v_width, options_.embedding_dim)));
}

MultiHeadAttentionOutput MultiHeadAttentionWithOutputImpl::forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& attention_mask) {
    require_output_attention_input(query, options_.embedding_dim, "query");
    require_output_attention_input(key, options_.embedding_dim, "key");
    require_output_attention_input(value, options_.embedding_dim, "value");
    if (query.size(0) != key.size(0) || query.size(0) != value.size(0)) {
        throw std::invalid_argument(
            "MultiHeadAttentionWithOutput query/key/value batch sizes differ");
    }

    const auto batch_size = query.size(0);
    const auto query_length = query.size(1);
    const auto key_length = key.size(1);
    const auto value_length = value.size(1);
    const auto query_projected = W_q->forward(query).view({
        batch_size, query_length, options_.num_heads, options_.d_k});
    const auto key_projected = W_k->forward(key).view({
        batch_size, key_length, options_.num_heads, options_.d_k});
    const auto value_projected = W_v->forward(value).view({
        batch_size, value_length, options_.num_heads, options_.d_v});
    const auto queries = query_projected.transpose(1, 2);
    const auto keys = key_projected.transpose(1, 2);
    const auto values = value_projected.transpose(1, 2);

    auto scores = torch::matmul(queries, keys.transpose(-2, -1)) / scale_;
    if (attention_mask.has_value()) {
        scores = scores.masked_fill(attention_mask->eq(0), -1e9);
    }
    auto weights = torch::softmax(scores, -1);
    weights = dropout->forward(weights);
    const auto attention_output = torch::matmul(weights, values)
                                      .transpose(1, 2)
                                      .contiguous()
                                      .view({
                                          batch_size,
                                          query_length,
                                          options_.num_heads * options_.d_v});
    return MultiHeadAttentionOutput{fc->forward(attention_output), weights};
}

}  // namespace virne::solver::learning::neural_network
