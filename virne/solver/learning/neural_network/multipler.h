#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace virne::solver::learning::neural_network {

struct MultiplerNetOptions {
    std::int64_t state_dim = 0;
};

class MultiplerNetImpl final : public torch::nn::Module {
public:
    explicit MultiplerNetImpl(MultiplerNetOptions options);

    torch::Tensor forward(const torch::Tensor& state);

    [[nodiscard]] const MultiplerNetOptions& options() const noexcept {
        return options_;
    }

    // Fixed handles preserve Python state-dict names and avoid name lookup in
    // the forward path.
    torch::nn::Linear l1{nullptr};
    torch::nn::Linear l2{nullptr};
    torch::nn::Linear l3{nullptr};

private:
    MultiplerNetOptions options_{};
};

TORCH_MODULE(MultiplerNet);

}  // namespace virne::solver::learning::neural_network
