#include "loss.h"

#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

BarlowTwinsContrastiveLossOptions validate_options(
    BarlowTwinsContrastiveLossOptions options) {
    if (!std::isfinite(options.eps) || options.eps < 0.0) {
        throw std::invalid_argument(
            "BarlowTwinsContrastiveLoss eps must be finite and non-negative");
    }
    if (options.lambda_.has_value()
        && !std::isfinite(*options.lambda_)) {
        throw std::invalid_argument(
            "BarlowTwinsContrastiveLoss lambda must be finite");
    }
    return options;
}

void validate_embeddings(
    const torch::Tensor& h1,
    const torch::Tensor& h2) {
    if (!h1.defined() || !h2.defined()) {
        throw std::invalid_argument(
            "BarlowTwinsContrastiveLoss embeddings must be defined");
    }
    if (h1.sizes() != h2.sizes()) {
        throw std::invalid_argument(
            "BarlowTwinsContrastiveLoss embedding shapes differ");
    }
    if (h1.dim() < 2) {
        throw std::invalid_argument(
            "BarlowTwinsContrastiveLoss embeddings must have rank at least two");
    }
}

}  // namespace

BarlowTwinsContrastiveLossImpl::BarlowTwinsContrastiveLossImpl(
    BarlowTwinsContrastiveLossOptions options)
    : options_(validate_options(std::move(options))) {}

torch::Tensor BarlowTwinsContrastiveLossImpl::forward(
    const torch::Tensor& h1,
    const torch::Tensor& h2) {
    validate_embeddings(h1, h2);

    const auto batch_size = h1.size(0);
    const auto feature_dim = h1.size(1);
    const double lambda = options_.lambda_.value_or(
        1.0 / static_cast<double>(feature_dim));

    torch::Tensor cross_correlation;
    if (options_.batch_norm) {
        // torch.std(dim=0) in Python uses Bessel correction (correction=1).
        // Preserve Python's graph creation order as well as forward arithmetic.
        // Reversing mean/std creation changes autograd accumulation rounding.
        const auto centered1 = h1 - h1.mean(0);
        const auto scale1 = torch::std(h1, {0}, true, false) + options_.eps;
        const auto z1_norm = centered1 / scale1;
        const auto centered2 = h2 - h2.mean(0);
        const auto scale2 = torch::std(h2, {0}, true, false) + options_.eps;
        const auto z2_norm = centered2 / scale2;
        cross_correlation = torch::matmul(
            z1_norm.transpose(-2, -1), z2_norm)
            / batch_size;
    } else {
        cross_correlation = torch::matmul(
            h1.transpose(-2, -1), h2)
            / batch_size;
    }

    // Python creates this mask for every call. Creating it on the result
    // device preserves the same indexing semantics while avoiding a CUDA/CPU
    // synchronization or unsupported cross-device index path.
    const auto off_diagonal_mask = ~torch::eye(
        feature_dim,
        torch::TensorOptions()
            .dtype(torch::kBool)
            .device(cross_correlation.device()));
    auto loss = (1 - cross_correlation.diagonal()).pow(2).sum();
    loss += lambda
        * cross_correlation.index({off_diagonal_mask}).pow(2).sum();
    return loss.mean();
}

}  // namespace virne::solver::learning::neural_network
