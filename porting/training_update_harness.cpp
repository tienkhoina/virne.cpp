#include "tensor_bytes.h"
#include "../virne/solver/learning/rl_core/training_update.h"
#include "../virne/solver/learning/rl_policy/mlp_policy.h"
#include <torch/torch.h>
#include <chrono>
#include <stdexcept>

namespace rl = virne::solver::learning::rl_core;
namespace policy = virne::solver::learning::rl_policy;
using virne::porting::emit_tensor_bytes;
using virne::porting::emit_module_bytes;

namespace {
torch::Tensor values(const std::vector<std::int64_t>& shape) {
    std::int64_t size = 1; for (auto extent : shape) size *= extent;
    return torch::arange(size, torch::kFloat32).reshape(shape).remainder(17).sub(8).div(16);
}
void optimizer_state(const rl::Adam& optimizer) {
    for (std::size_t i = 0; i < optimizer.parameters().size(); ++i) {
        const auto name = "optimizer_" + std::to_string(i) + ".";
        emit_tensor_bytes(name + "parameter", optimizer.parameters()[i]);
        emit_tensor_bytes(name + "gradient", optimizer.parameters()[i].grad());
        const auto& state = optimizer.moments()[i];
        emit_tensor_bytes(name + "step", state.step);
        emit_tensor_bytes(name + "mean", state.exp_avg);
        emit_tensor_bytes(name + "variance", state.exp_avg_sq);
        emit_tensor_bytes(name + "maximum", state.max_exp_avg_sq);
    }
}
void adam_case(const int variant) {
    auto a = values({2, 3});
    auto b = values({3});
    if (variant == 1) { a = a.to(torch::kFloat64); b = b.to(torch::kFloat64); }
    if (variant == 2) { a = a.transpose(0, 1); b = b.to(torch::kFloat64); }
    if (variant == 3) { a = torch::complex(a, a * 0.25); b = torch::complex(b, b * -0.5); }
    a.set_requires_grad(true); b.set_requires_grad(true);
    rl::AdamOptions first{0.003, 0.8, 0.95, 1e-7, 0.02, variant >= 2, variant == 3};
    rl::AdamOptions second{0.007, 0.85, 0.98, 1e-8, 0.01, variant >= 2, false};
    rl::Adam optimizer(std::vector<rl::AdamParameterGroup>{{{a}, first}, {{b}, second}});
    std::optional<rl::AdamSnapshot> saved;
    for (int step = 0; step < 4; ++step) {
        optimizer.zero_grad(step != 2);
        a.mutable_grad() = torch::full_like(a, 0.1 + step * 0.03);
        if (step != 1) b.mutable_grad() = torch::full_like(b, -0.2 + step * 0.02);
        if (step == 2) {
            optimizer.restore(*saved);
            optimizer.options(0).lr = 0.002;
        }
        const auto norm = rl::clip_grad_norm({a, b}, 0.25);
        optimizer.step();
        std::cout << "STEP " << step << '\n';
        emit_tensor_bytes("norm", norm);
        optimizer_state(optimizer);
        if (step == 0) saved = optimizer.snapshot();
    }
}

struct TrainingFixture {
    policy::MlpActorCritic model{nullptr};
    std::unique_ptr<rl::Adam> optimizer;
    torch::Tensor observations, masks;
    explicit TrainingFixture(const bool masked) {
        torch::manual_seed(401);
        model = policy::MlpActorCritic(policy::MlpActorCriticOptions{3, 4, 2, 8, 0.0, false});
        model->train();
        optimizer = std::make_unique<rl::Adam>(std::vector<rl::AdamParameterGroup>{
            {rl::unique_parameters(*model->actor), rl::AdamOptions{0.003, 0.9, 0.999, 1e-8, 0.01}},
            {rl::unique_parameters(*model->critic), rl::AdamOptions{0.007, 0.9, 0.999, 1e-8, 0.01}}});
        observations = values({6, 4, 3});
        if (masked) {
            masks = torch::ones({6, 4}, torch::kBool);
            masks.select(1, 3).zero_(); masks.index_put_({3, 3}, true);
        }
    }
    rl::TrainingEvaluator evaluator() {
        return [this](const torch::Tensor& indices, const torch::Tensor& actions) {
            const auto obs = observations.index_select(0, indices);
            const auto logits = model->act(obs);
            const auto value = model->evaluate(obs);
            const auto mask = masks.defined() ? std::optional<torch::Tensor>{masks.index_select(0, indices)} : std::nullopt;
            return rl::evaluate_policy_actions(logits, actions, value, mask, std::nullopt, true);
        };
    }
};
rl::TrainingBatch batch(const bool stop = false) {
    return {torch::tensor({0, 1, 2, 3, 0, 1}, torch::kInt64),
        torch::tensor({0.1F, -0.2F, 0.4F, 0.3F, -0.1F, 0.7F}),
        torch::tensor({1.0F, -0.5F, 2.0F, 0.3F, -1.0F, 1.2F}),
        torch::full({6}, stop ? 3.0 : -1.4)};
}
rl::TrainingOptions options(const int algorithm, const bool stop) {
    rl::TrainingOptions o;
    o.algorithm = static_cast<rl::UpdateAlgorithm>(algorithm);
    o.batch_size = 4; o.repeat_times = 1; o.normalize_returns = true; o.normalize_advantage = true;
    o.max_grad_norm = 0.25; o.mask_coefficient = 0.3;
    if (stop) o.target_kl = 1e-12;
    return o;
}
void training_case(const int algorithm, const bool masked, const bool stop) {
    TrainingFixture fixture(masked);
    rl::PolicyGradientTrainer trainer(*fixture.optimizer, fixture.evaluator(), options(algorithm, stop));
    std::vector<std::int64_t> scheduler;
    for (int update = 0; update < 2; ++update) {
        auto data = batch(stop);
        trainer.after_update = [&] { scheduler.push_back(data.size()); scheduler.push_back(trainer.update_time()); };
        const auto result = trainer.update(data);
        std::cout << "UPDATE " << update << '\n';
        emit_tensor_bytes("loss", result.loss);
        emit_tensor_bytes("update_time", torch::tensor(trainer.update_time(), torch::kInt64));
        emit_tensor_bytes("buffer_size", torch::tensor(data.size(), torch::kInt64));
        optimizer_state(*fixture.optimizer);
        emit_tensor_bytes("rng_tail", torch::rand({4}));
    }
    emit_tensor_bytes("scheduler", torch::tensor(scheduler, torch::kInt64));
}
void exact() {
    for (int i = 0; i < 4; ++i) { std::cout << "CASE adam_" << i << '\n'; adam_case(i); }
    for (int i = 0; i < 5; ++i) { std::cout << "CASE training_" << i << '\n'; training_case(i, false, false); }
    std::cout << "CASE training_mask\n"; training_case(2, true, false);
    std::cout << "CASE training_early_stop\n"; training_case(2, false, true);
}
void benchmark() {
    std::vector<std::int64_t> samples;
    torch::Tensor final_loss;
    std::vector<torch::Tensor> reference;
    for (int repetition = -3; repetition < 5; ++repetition) {
        TrainingFixture fixture(true);
        rl::PolicyGradientTrainer trainer(*fixture.optimizer, fixture.evaluator(), options(2, false));
        auto data = batch();
        const auto start = std::chrono::steady_clock::now();
        const auto result = trainer.update(data);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        final_loss = result.loss;
        if (reference.empty()) for (const auto& parameter : fixture.optimizer->parameters()) reference.push_back(parameter.detach().clone());
        else for (std::size_t i = 0; i < reference.size(); ++i) if (!torch::equal(reference[i], fixture.optimizer->parameters()[i])) throw std::runtime_error("training benchmark output drift");
        if (repetition >= 0) samples.push_back(elapsed);
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
    std::cout << "]}\n";
    emit_tensor_bytes("loss", final_loss);
    for (std::size_t i = 0; i < reference.size(); ++i) emit_tensor_bytes("parameter_" + std::to_string(i), reference[i]);
}
}
int main(int argc, char** argv) {
    try {
        torch::set_num_threads(1); torch::set_num_interop_threads(1);
        if (argc > 1 && std::string(argv[1]) == "--benchmark") benchmark(); else exact();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
