#include "res.h"

#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace virne::solver::learning::neural_network {

ResNetBlockImpl::ResNetBlockImpl(ResNetBlockOptions options)
    : options_(std::move(options)) {
    if (options_.input_channels <= 0) {
        throw std::invalid_argument(
            "ResNetBlock input_channels must be positive");
    }
    const auto convolution_options = [&] {
        return torch::nn::Conv2dOptions(
                   options_.input_channels,
                   options_.input_channels,
                   {1, 1})
            .stride({1, 1});
    };
    conv_1 = register_module(
        "conv_1", torch::nn::Conv2d(convolution_options()));
    conv_2 = register_module(
        "conv_2", torch::nn::Conv2d(convolution_options()));
}

torch::Tensor ResNetBlockImpl::forward(const torch::Tensor& input) {
    if (!input.defined() || (input.dim() != 3 && input.dim() != 4)
        || input.size(input.dim() - 3) != options_.input_channels) {
        throw std::invalid_argument(
            "ResNetBlock input must have shape [channels, height, width] "
            "or [batch, channels, height, width]");
    }
    auto output = torch::leaky_relu(conv_1->forward(input), 0.01);
    output = torch::leaky_relu(conv_2->forward(output), 0.01);
    return output + input;
}

ResLinearNetImpl::ResLinearNetImpl(ResLinearNetOptions options)
    : options_(std::move(options)) {
    throw std::runtime_error(
        "Python ResLinearNet is unavailable: its constructor calls "
        "super(ResNetBlock, self) with an unrelated instance");
}

torch::Tensor ResLinearNetImpl::forward(const torch::Tensor&) {
    throw std::logic_error(
        "ResLinearNet cannot be forwarded because construction must fail");
}

}  // namespace virne::solver::learning::neural_network
