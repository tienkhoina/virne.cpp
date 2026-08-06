#pragma once

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

}  // namespace virne::solver::learning::neural_network
