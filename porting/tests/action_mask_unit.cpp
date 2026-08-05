#include "../../virne/solver/learning/rl_core/action_mask.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl = virne::solver::learning::rl_core;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_allclose(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message) {
    require(actual.scalar_type() == expected.scalar_type(),
            message + " (dtype)");
    require(actual.sizes() == expected.sizes(), message + " (shape)");
    require(torch::allclose(actual, expected, 0.0, 0.0),
            message + " (values)");
}

void test_identity() {
    const auto values = torch::tensor({1.5F, -2.25F, 0.0F, 4.75F})
                            .reshape({2, 2});
    const auto logit = rl::apply_mask_to_logit(values);
    const auto prob = rl::apply_mask_to_prob(values);
    require(logit.unsafeGetTensorImpl() == values.unsafeGetTensorImpl(),
            "logit identity must preserve tensor handle");
    require(prob.unsafeGetTensorImpl() == values.unsafeGetTensorImpl(),
            "probability identity must preserve tensor handle");
}

void test_float32_masks() {
    const auto values = torch::tensor({1.5F, -2.25F, 0.0F, 4.75F})
                            .reshape({2, 2});
    const auto mixed = torch::tensor({1, 0, 1, 0}, torch::kBool)
                           .reshape({1, 4});
    const auto expected_logit = torch::tensor(
        {{1.5F, -1.0e8F}, {0.0F, -1.0e8F}});
    const auto expected_prob = torch::tensor({
        {1.5F, 0.0F}, {0.0F, 0.0F}});
    require_allclose(rl::apply_mask_to_logit(values, mixed), expected_logit,
                     "float32 bool logit mask");
    require_allclose(rl::apply_mask_to_prob(values, mixed), expected_prob,
                     "float32 bool probability mask");

    const auto integer_mask = torch::tensor({2, 0, -1, 0}, torch::kInt64);
    require_allclose(
        rl::apply_mask_to_prob(values, integer_mask), expected_prob,
        "integer mask converts through bool");

    const auto vector_mask = std::vector<std::uint8_t>{1U, 0U, 1U, 0U};
    require_allclose(
        rl::apply_mask_to_logit(values, vector_mask), expected_logit,
        "vector mask overload");

    const auto all_true = torch::ones({2, 2}, torch::kBool);
    const auto all_false = torch::zeros({2, 2}, torch::kBool);
    require_allclose(rl::apply_mask_to_logit(values, all_true), values,
                     "all true logit mask");
    require_allclose(rl::apply_mask_to_prob(values, all_false),
                     torch::zeros_like(values), "all false probability mask");
}

void test_float64_mask() {
    const auto values = torch::tensor({1.25, -3.5, 2.75, 8.0}, torch::kFloat64)
                            .reshape({2, 2});
    const auto mask = torch::tensor({1.0, 0.0, -2.0, 0.25}, torch::kFloat64)
                          .reshape({4});
    const auto expected = torch::tensor(
        {{1.25, -1.0e8}, {2.75, 8.0}}, torch::kFloat64);
    require_allclose(rl::apply_mask_to_logit(values, mask), expected,
                     "float64 numeric mask");
    require_allclose(
        rl::apply_mask_to_prob(values, mask),
        torch::tensor({{1.25, 0.0}, {2.75, 8.0}}, torch::kFloat64),
        "float64 probability mask");
}

void test_mixed_precision_mask() {
    const auto mask = torch::tensor({1, 0}, torch::kBool);
    const auto half = rl::apply_mask_to_logit(
        torch::tensor({1.0F, 2.0F}).to(torch::kFloat16), mask);
    require(half.scalar_type() == torch::kFloat16
                && half.index({0}).item<float>() == 1.0F
                && torch::isneginf(half.index({1})).item<bool>(),
            "float16 mask matches Python overflow cast");

    const auto bfloat = rl::apply_mask_to_logit(
        torch::tensor({1.0F, 2.0F}).to(torch::kBFloat16), mask);
    require(bfloat.scalar_type() == torch::kBFloat16
                && bfloat.index({0}).item<float>() == 1.0F
                && bfloat.index({1}).item<float>() < -9.0e7F,
            "bfloat16 mask keeps the large negative sentinel");
}

void test_mismatch_rejected() {
    const auto values = torch::ones({2, 2}, torch::kFloat32);
    bool rejected = false;
    try {
        static_cast<void>(rl::apply_mask_to_logit(
            values, torch::ones({3}, torch::kBool)));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "mask reshape mismatch must be rejected");
}

void test_typed_errors() {
    bool values_rejected = false;
    try {
        static_cast<void>(rl::apply_mask_to_logit(torch::Tensor()));
    } catch (const rl::ActionMaskException& error) {
        values_rejected =
            error.code() == rl::ActionMaskErrorCode::undefined_values;
    }
    require(values_rejected, "undefined values typed error");

    bool mask_rejected = false;
    try {
        static_cast<void>(rl::apply_mask_to_prob(
            torch::ones({2, 2}, torch::kFloat32), torch::Tensor()));
    } catch (const rl::ActionMaskException& error) {
        mask_rejected =
            error.code() == rl::ActionMaskErrorCode::undefined_mask;
    }
    require(mask_rejected, "undefined mask typed error");
}

void test_cuda_if_available() {
    if (!torch::cuda::is_available()) {
        return;
    }
    const auto device = torch::Device(torch::kCUDA);
    const auto values = torch::tensor({1.0F, -2.0F},
                                      torch::TensorOptions().device(device));
    const auto mask = torch::tensor(
        {1, 0}, torch::TensorOptions().dtype(torch::kBool).device(device));
    const auto output = rl::apply_mask_to_logit(values, mask);
    require(output.device() == values.device(), "CUDA mask device mismatch");
    require(output.index({0}).item<float>() == 1.0F,
            "CUDA retained value mismatch");
    require(output.index({1}).item<float>() == -1.0e8F,
            "CUDA masked value mismatch");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        test_identity();
        test_float32_masks();
        test_float64_mask();
        test_mixed_precision_mask();
        test_mismatch_rejected();
        test_typed_errors();
        test_cuda_if_available();
        std::cout << "action_mask_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "action_mask_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
