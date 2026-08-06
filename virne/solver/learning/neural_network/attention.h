#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

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

}  // namespace virne::solver::learning::neural_network
