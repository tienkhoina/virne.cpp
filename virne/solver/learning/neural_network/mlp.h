#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>
#include <variant>
#include <vector>

namespace virne::solver::learning::neural_network {

using MlpHiddenDims = std::variant<
    std::monostate,
    std::int64_t,
    std::vector<std::int64_t>>;

struct MlpOptions {
    std::int64_t input_dim = 0;
    std::int64_t output_dim = 0;
    std::int64_t num_layers = 2;
    MlpHiddenDims embedding_dims = std::monostate{};
    bool batch_norm = false;

    // Kept for exact API compatibility with Python MLPNet. The original
    // implementation accepts this value but does not add a Dropout module.
    double dropout_prob = 1.0;
};

class MLPNetImpl final : public torch::nn::Module {
public:
    explicit MLPNetImpl(MlpOptions options);

    torch::Tensor forward(const torch::Tensor& input);

    [[nodiscard]] const MlpOptions& options() const noexcept;
    [[nodiscard]] const std::vector<std::int64_t>&
    resolved_hidden_dims() const noexcept;

    // Public to mirror Python's `self.lins` and preserve state-dict names
    // such as `lins.0.weight`, `lins.1.running_mean`, and `lins.3.bias`.
    torch::nn::Sequential lins{nullptr};

private:
    MlpOptions options_;
    std::vector<std::int64_t> resolved_hidden_dims_;
};

TORCH_MODULE(MLPNet);

} // namespace virne::solver::learning::neural_network
