#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <optional>

namespace virne::solver::learning::neural_network {

struct BarlowTwinsContrastiveLossOptions {
    std::optional<double> lambda_ = std::nullopt;
    bool batch_norm = true;
    double eps = 1e-5;
};

class BarlowTwinsContrastiveLossImpl final : public torch::nn::Module {
public:
    explicit BarlowTwinsContrastiveLossImpl(
        BarlowTwinsContrastiveLossOptions options = {});

    torch::Tensor forward(
        const torch::Tensor& h1,
        const torch::Tensor& h2);

    [[nodiscard]] const BarlowTwinsContrastiveLossOptions&
    options() const noexcept {
        return options_;
    }

private:
    BarlowTwinsContrastiveLossOptions options_{};
};

TORCH_MODULE(BarlowTwinsContrastiveLoss);

}  // namespace virne::solver::learning::neural_network
