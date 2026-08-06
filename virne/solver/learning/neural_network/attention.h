#pragma once

#include <torch/nn/modules/linear.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>

namespace virne::solver::learning::neural_network {

enum class PositionalEncodingMethod : std::uint8_t {
    add,
    concat,
};

struct PositionalEncoderOptions {
    std::int64_t embedding_dim = 0;
    double dropout_prob = 1.0;
    std::int64_t max_len = 50;
    PositionalEncodingMethod method = PositionalEncodingMethod::add;
};

class PositionalEncoderImpl final : public torch::nn::Module {
public:
    explicit PositionalEncoderImpl(PositionalEncoderOptions options);

    torch::Tensor forward(const torch::Tensor& input);

    [[nodiscard]] const PositionalEncoderOptions& options() const noexcept {
        return options_;
    }
    [[nodiscard]] const torch::Tensor& positional_buffer() const noexcept {
        return pe;
    }

    // Registered as `pe` to preserve Python state-dict names.
    torch::Tensor pe;
    torch::nn::Dropout dropout{nullptr};

private:
    PositionalEncoderOptions options_{};
    bool has_dropout_ = false;
};

TORCH_MODULE(PositionalEncoder);

struct MultiHeadSelfAttentionOptions {
    std::int64_t input_dim = 0;
    std::int64_t embed_dim = 0;
    std::int64_t val_dim = 0;
    std::int64_t key_dim = 0;
    std::int64_t num_heads = 1;
};

class MultiHeadSelfAttentionImpl final : public torch::nn::Module {
public:
    explicit MultiHeadSelfAttentionImpl(
        MultiHeadSelfAttentionOptions options);

    torch::Tensor forward(
        const torch::Tensor& query,
        const std::optional<torch::Tensor>& key = std::nullopt,
        const std::optional<torch::Tensor>& value = std::nullopt,
        const std::optional<torch::Tensor>& attention_mask = std::nullopt);

    [[nodiscard]] const MultiHeadSelfAttentionOptions& options() const noexcept {
        return options_;
    }
    [[nodiscard]] double norm_factor() const noexcept {
        return norm_factor_;
    }

    // Names and parameter order match Python's state dictionary.
    torch::Tensor W_query;
    torch::Tensor W_key;

private:
    MultiHeadSelfAttentionOptions options_{};
    double norm_factor_ = 1.0;
};

TORCH_MODULE(MultiHeadSelfAttention);

struct MultiHeadAttentionWithOutputOptions {
    std::int64_t embedding_dim = 0;
    std::int64_t num_heads = 1;
    // Zero selects Python's None/default behavior (embedding_dim).
    std::int64_t d_k = 0;
    std::int64_t d_v = 0;
    double dropout = 0.1;
};

struct MultiHeadAttentionOutput {
    torch::Tensor output;
    torch::Tensor weights;
};

class MultiHeadAttentionWithOutputImpl final : public torch::nn::Module {
public:
    explicit MultiHeadAttentionWithOutputImpl(
        MultiHeadAttentionWithOutputOptions options);

    MultiHeadAttentionOutput forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const std::optional<torch::Tensor>& attention_mask = std::nullopt);

    [[nodiscard]] const MultiHeadAttentionWithOutputOptions& options() const
        noexcept {
        return options_;
    }

    // Public handles make fixed parameters inspectable without dynamic lookup.
    torch::nn::Linear W_q{nullptr};
    torch::nn::Linear W_k{nullptr};
    torch::nn::Linear W_v{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear fc{nullptr};

private:
    MultiHeadAttentionWithOutputOptions options_{};
    float scale_ = 1.0F;
};

TORCH_MODULE(MultiHeadAttentionWithOutput);

}  // namespace virne::solver::learning::neural_network
