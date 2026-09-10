#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace virne::solver::learning::neural_network {

struct ResNetBlockOptions {
    std::int64_t input_channels = 0;
};

struct ResLinearNetOptions {
    std::int64_t num_in_feats = 0;
    std::int64_t num_out_feats = 0;
    std::int64_t num_layers = 3;
    bool batch_norm = true;
};

class ResNetBlockImpl final : public torch::nn::Module {
public:
    explicit ResNetBlockImpl(ResNetBlockOptions options);

    torch::Tensor forward(const torch::Tensor& input);

    [[nodiscard]] const ResNetBlockOptions& options() const noexcept {
        return options_;
    }

    torch::nn::Conv2d conv_1{nullptr};
    torch::nn::Conv2d conv_2{nullptr};

private:
    ResNetBlockOptions options_{};
};

TORCH_MODULE(ResNetBlock);

class ResLinearNetImpl final : public torch::nn::Module {
public:
    explicit ResLinearNetImpl(ResLinearNetOptions options);

    torch::Tensor forward(const torch::Tensor& input);

    [[nodiscard]] const ResLinearNetOptions& options() const noexcept {
        return options_;
    }

private:
    ResLinearNetOptions options_{};
};

TORCH_MODULE(ResLinearNet);

}  // namespace virne::solver::learning::neural_network
