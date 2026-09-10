#include "../../virne/solver/learning/neural_network/loss.h"

#include <torch/torch.h>

#include <iostream>
#include <optional>
#include <stdexcept>

namespace nn = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor embedding(
    const std::int64_t batch,
    const std::int64_t features,
    const double offset = 0.0) {
    return torch::arange(batch * features, torch::kFloat64)
        .reshape({batch, features})
        .remainder(17.0)
        .sub(8.0)
        .div(7.0)
        .add(offset);
}

torch::Tensor python_formula(
    const torch::Tensor& h1,
    const torch::Tensor& h2,
    const std::optional<double> lambda,
    const bool batch_norm,
    const double eps) {
    const auto batch_size = h1.size(0);
    const auto feature_dim = h1.size(1);
    const double resolved_lambda = lambda.value_or(
        1.0 / static_cast<double>(feature_dim));
    torch::Tensor correlation;
    if (batch_norm) {
        const auto z1 = (h1 - h1.mean(0))
            / (torch::std(h1, {0}, true, false) + eps);
        const auto z2 = (h2 - h2.mean(0))
            / (torch::std(h2, {0}, true, false) + eps);
        correlation = torch::matmul(z1.transpose(-2, -1), z2)
            / batch_size;
    } else {
        correlation = torch::matmul(h1.transpose(-2, -1), h2)
            / batch_size;
    }
    const auto mask = ~torch::eye(
        feature_dim,
        torch::TensorOptions()
            .dtype(torch::kBool)
            .device(correlation.device()));
    auto loss = (1 - correlation.diagonal()).pow(2).sum();
    loss += resolved_lambda * correlation.index({mask}).pow(2).sum();
    return loss.mean();
}

void test_default_and_explicit_options() {
    const auto h1 = embedding(8, 4);
    const auto h2 = embedding(8, 4, 0.2).flip({0});

    nn::BarlowTwinsContrastiveLoss default_loss;
    const auto actual_default = default_loss->forward(h1, h2);
    const auto expected_default = python_formula(
        h1, h2, std::nullopt, true, 1e-5);
    require(torch::equal(actual_default, expected_default),
            "default Barlow Twins loss formula mismatch");
    require(actual_default.dim() == 0,
            "Barlow Twins loss is not scalar");

    nn::BarlowTwinsContrastiveLoss explicit_loss(
        nn::BarlowTwinsContrastiveLossOptions{0.125, false, 1e-4});
    const auto actual_explicit = explicit_loss->forward(h1, h2);
    const auto expected_explicit = python_formula(
        h1, h2, 0.125, false, 1e-4);
    require(torch::equal(actual_explicit, expected_explicit),
            "explicit Barlow Twins loss formula mismatch");
    require(actual_explicit.scalar_type() == h1.scalar_type(),
            "Barlow Twins loss changed dtype");
}

void test_validation_gradient_and_device() {
    nn::BarlowTwinsContrastiveLoss loss;
    bool rejected = false;
    try {
        static_cast<void>(loss->forward(embedding(3, 2), embedding(4, 2)));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Barlow Twins loss accepted different shapes");

    auto h1 = embedding(6, 3).set_requires_grad(true);
    auto h2 = embedding(6, 3, 0.3).set_requires_grad(true);
    const auto value = loss->forward(h1, h2);
    value.backward();
    require(h1.grad().defined() && h2.grad().defined(),
            "Barlow Twins loss broke autograd");
    require(torch::isfinite(h1.grad()).all().item<bool>()
                && torch::isfinite(h2.grad()).all().item<bool>(),
            "Barlow Twins loss produced non-finite gradients");

    if (torch::cuda::is_available()) {
        const auto cuda_h1 = embedding(6, 3).to(torch::kCUDA);
        const auto cuda_h2 = embedding(6, 3, 0.1).to(torch::kCUDA);
        require(loss->forward(cuda_h1, cuda_h2).device().is_cuda(),
                "CUDA Barlow Twins loss returned a CPU tensor");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_default_and_explicit_options();
        test_validation_gradient_and_device();
        std::cout << "barlow_twins_loss_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "barlow_twins_loss_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
