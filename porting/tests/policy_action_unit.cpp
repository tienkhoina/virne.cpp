#include "../../virne/solver/learning/rl_core/policy_action.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace rl = virne::solver::learning::rl_core;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message,
    const double tolerance = 1e-6) {
    require(actual.sizes() == expected.sizes(), message + " shape");
    require(actual.scalar_type() == expected.scalar_type(), message + " dtype");
    require(torch::allclose(actual, expected, tolerance, tolerance),
            message + " values");
}

torch::Tensor logits_fixture(const torch::Device& device = torch::kCPU) {
    return torch::tensor(
        {{1.0F, -0.5F, 0.25F, 2.0F},
         {-1.0F, 0.5F, 1.5F, 0.0F},
         {0.1F, 0.2F, 0.3F, 0.4F}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor mask_fixture(const torch::Device& device = torch::kCPU) {
    return torch::tensor(
        {{1, 0, 1, 1}, {0, 1, 1, 0}, {0, 0, 0, 0}},
        torch::TensorOptions().dtype(torch::kBool).device(device));
}

void test_distribution_and_selection() {
    const auto logits = logits_fixture().set_requires_grad(true);
    const auto mask = mask_fixture();
    rl::PolicyActionOptions options;
    options.softmax_temperature = 0.75;

    const auto distribution = rl::policy_action_distribution(
        logits, mask, options);
    const auto fill = torch::full({}, -1e8, logits.options());
    const auto candidates = torch::where(mask, logits, fill);
    require_close(distribution.candidate_logits, candidates,
                  "candidate logits");
    require_close(distribution.probabilities,
                  torch::softmax(candidates / 0.75, -1),
                  "candidate probabilities");
    require(!distribution.probabilities.requires_grad()
                && !distribution.candidate_logits.requires_grad(),
            "inference distribution is detached");
    const auto no_mask_distribution = rl::policy_action_distribution(
        logits, std::nullopt, options);
    require(!no_mask_distribution.candidate_logits.requires_grad(),
            "unmasked candidate logits are detached");

    const auto greedy = rl::select_policy_action(
        logits, mask, rl::PolicyActionMode::greedy, options);
    require(torch::equal(greedy.actions, candidates.argmax(-1)),
            "greedy uses candidate logits");

    options.maskable_policy = false;
    const auto raw_log = logits / options.softmax_temperature;
    const auto raw_normalized = raw_log - raw_log.logsumexp(-1, true);
    const auto raw_greedy = rl::select_policy_action(
        logits, mask, rl::PolicyActionMode::greedy, options);
    const auto expected_log_prob = raw_normalized.gather(
        -1, raw_greedy.actions.unsqueeze(-1)).squeeze(-1);
    require_close(raw_greedy.log_probabilities, expected_log_prob,
                  "maskable false uses raw log probability");

    options.maskable_policy = true;
    torch::manual_seed(1234);
    const auto sampled = rl::select_policy_action(
        logits, mask, rl::PolicyActionMode::sample, options);
    torch::manual_seed(1234);
    const auto normalized = candidates / options.softmax_temperature;
    const auto probs = torch::softmax(
        normalized - normalized.logsumexp(-1, true), -1);
    const auto expected_actions = torch::multinomial(
        probs.reshape({-1, probs.size(-1)}), 1, true).t().reshape({3});
    require(torch::equal(sampled.actions, expected_actions),
            "categorical sample RNG continuation");
}

void test_evaluation_and_gradients() {
    auto logits = logits_fixture().clone().set_requires_grad(true);
    const auto mask = mask_fixture();
    const auto actions = torch::tensor({3, 2, 1}, torch::kInt64);
    auto values = torch::tensor(
        {{0.5F}, {1.5F}, {-0.25F}}, torch::kFloat32)
                      .set_requires_grad(true);
    auto predicted = torch::tensor(
        {{0.8F, 0.2F, 0.7F, 0.9F},
         {0.1F, 0.6F, 0.8F, 0.2F},
         {0.4F, 0.3F, 0.2F, 0.1F}},
        torch::kFloat32).set_requires_grad(true);
    rl::PolicyActionOptions options;
    options.softmax_temperature = 0.5;
    const auto output = rl::evaluate_policy_actions(
        logits, actions, values, mask, predicted, true, options);
    require(output.values.has_value(), "value output present");
    require(output.masked_action_probability.has_value(),
            "mask metric present");
    require(output.prediction_loss.has_value(), "prediction loss present");
    require(output.log_probabilities.sizes() == torch::IntArrayRef({3}),
            "log probability shape");
    require(output.entropy.sizes() == torch::IntArrayRef({3}),
            "entropy shape");

    auto loss = output.log_probabilities.mean()
        + output.entropy.mean()
        + output.values->mean()
        + *output.masked_action_probability
        + *output.prediction_loss;
    loss.backward();
    require(logits.grad().defined() && torch::isfinite(logits.grad()).all().item<bool>(),
            "logit gradients finite");
    require(values.grad().defined() && torch::isfinite(values.grad()).all().item<bool>(),
            "value gradients finite");
    require(predicted.grad().defined()
                && torch::isfinite(predicted.grad()).all().item<bool>(),
            "predictor gradients finite");

    options.mask_actions = false;
    const auto raw = rl::evaluate_policy_actions(
        logits.detach(), actions, std::nullopt, mask, std::nullopt, false,
        options);
    const auto raw_probs = torch::softmax(logits.detach() / 0.5, -1);
    const auto normalized_probs = raw_probs / raw_probs.sum(-1, true);
    const auto expected_log = normalized_probs
        .clamp(std::numeric_limits<float>::epsilon(),
               1.0F - std::numeric_limits<float>::epsilon())
        .log()
        .gather(-1, actions.unsqueeze(-1))
        .squeeze(-1);
    require_close(raw.log_probabilities, expected_log,
                  "mask_actions false evaluation uses raw distribution");
}

void test_errors_and_cuda() {
    bool invalid_temperature = false;
    try {
        rl::PolicyActionOptions options;
        options.softmax_temperature = 0.0;
        static_cast<void>(rl::select_policy_action(
            logits_fixture(), std::nullopt,
            rl::PolicyActionMode::greedy, options));
    } catch (const rl::PolicyActionException& error) {
        invalid_temperature = error.code()
            == rl::PolicyActionErrorCode::invalid_temperature;
    }
    require(invalid_temperature, "invalid temperature rejected");

    bool wrong_actions = false;
    try {
        static_cast<void>(rl::evaluate_policy_actions(
            logits_fixture(), torch::tensor({0, 1}, torch::kInt64)));
    } catch (const rl::PolicyActionException& error) {
        wrong_actions = error.code()
            == rl::PolicyActionErrorCode::invalid_actions;
    }
    require(wrong_actions, "wrong action shape rejected");

    for (const auto dtype : {torch::kFloat16, torch::kBFloat16}) {
        const auto logits = logits_fixture().to(dtype);
        const auto actions = torch::tensor({3, 2, 1}, torch::kInt64);
        const auto valid_mask = torch::tensor(
            {{1, 0, 1, 1}, {0, 1, 1, 0}, {0, 0, 0, 1}},
            torch::TensorOptions().dtype(torch::kBool));
        const auto output = rl::evaluate_policy_actions(
            logits, actions, std::nullopt, valid_mask);
        require(output.log_probabilities.scalar_type() == dtype
                    && torch::isfinite(output.log_probabilities).all().item<bool>()
                    && torch::isfinite(output.entropy).all().item<bool>(),
                "mixed precision evaluation");
    }

    if (torch::cuda::is_available()) {
        const torch::Device cuda(torch::kCUDA);
        const auto output = rl::select_policy_action(
            logits_fixture(cuda), mask_fixture(cuda),
            rl::PolicyActionMode::greedy);
        require(output.actions.device() == cuda
                    && output.log_probabilities.device() == cuda,
                "CUDA selection device");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        test_distribution_and_selection();
        test_evaluation_and_gradients();
        test_errors_and_cuda();
        std::cout << "policy_action_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "policy_action_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
