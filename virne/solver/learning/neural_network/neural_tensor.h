#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace virne::solver::learning::neural_network {

struct NeuralTensorNetworkOptions {
    std::int64_t input_features = 0;
    std::int64_t tensor_neurons = 0;
};

class NeuralTensorNetworkImpl final : public torch::nn::Module {
public:
    explicit NeuralTensorNetworkImpl(NeuralTensorNetworkOptions options);

    torch::Tensor forward(
        const torch::Tensor& embedding_1,
        const torch::Tensor& embedding_2);

    [[nodiscard]] const NeuralTensorNetworkOptions& options() const noexcept {
        return options_;
    }

    // Fixed parameter handles preserve Python state-dict names.
    torch::Tensor weight_matrix;
    torch::Tensor weight_matrix_block;
    torch::Tensor bias;

private:
    NeuralTensorNetworkOptions options_{};
};

TORCH_MODULE(NeuralTensorNetwork);

}  // namespace virne::solver::learning::neural_network
