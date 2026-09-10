#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace virne::solver::learning::neural_network {

struct SinkhornNetworkOptions {
    std::int64_t max_iter = 10;
    double tau = 1.0;
    double epsilon = 1e-4;
    bool log_forward = true;
    bool batched_operation = false;
};

struct SinkhornNetworkCallOptions {
    std::optional<std::vector<std::int64_t>> nrows;
    std::optional<std::vector<std::int64_t>> ncols;
    std::optional<torch::Tensor> r;
    std::optional<torch::Tensor> c;
    bool dummy_row = false;
    c10::ScalarType dtype = torch::kFloat32;
    bool dummy = false;
};

class SinkhornNetworkImpl final : public torch::nn::Module {
public:
    explicit SinkhornNetworkImpl(SinkhornNetworkOptions options = {});

    torch::Tensor forward(
        const torch::Tensor& scores,
        const SinkhornNetworkCallOptions& call = {});
    torch::Tensor forward_log(
        const torch::Tensor& scores,
        const SinkhornNetworkCallOptions& call = {});
    torch::Tensor forward_log_dummy(
        const torch::Tensor& scores,
        const SinkhornNetworkCallOptions& call = {});

    [[nodiscard]] const SinkhornNetworkOptions& options() const noexcept {
        return options_;
    }

private:
    SinkhornNetworkOptions options_{};
};

TORCH_MODULE(SinkhornNetwork);

torch::Tensor logsumexp(
    const torch::Tensor& inputs,
    std::optional<std::int64_t> dim = std::nullopt,
    bool keepdim = false);

struct SinkhornOptions {
    std::int64_t sinkhorn_iters = 5;
    double tau = 0.01;
};

class SinkhornImpl final : public torch::nn::Module {
public:
    explicit SinkhornImpl(SinkhornOptions options = {});

    torch::Tensor row_norm(const torch::Tensor& input) const;
    torch::Tensor col_norm(const torch::Tensor& input) const;
    torch::Tensor forward(const torch::Tensor& input, double eps = 1e-6);

    [[nodiscard]] const SinkhornOptions& options() const noexcept {
        return options_;
    }

private:
    SinkhornOptions options_{};
};

TORCH_MODULE(Sinkhorn);

}  // namespace virne::solver::learning::neural_network
