#include "tensor_bytes.h"
#include "../virne/solver/learning/rl_core/dqn_update.h"
#include "../virne/solver/learning/rl_policy/mlp_policy.h"
#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>
#include <chrono>

namespace rl = virne::solver::learning::rl_core;
namespace p = virne::solver::learning::rl_policy;
using virne::porting::emit_tensor_bytes;
using virne::porting::emit_module_bytes;
namespace {
torch::Tensor values() { return torch::arange(72, torch::kFloat32).reshape({6,4,3}).remainder(17).sub(8).div(16); }
rl::DqnBatch batch() {
    return {torch::tensor({0,1,2,3,0,1}, torch::kInt64),
        torch::tensor({.1F,-.2F,.4F,.3F,-.1F,.7F}), torch::tensor({0.F,0.F,1.F,0.F,1.F,1.F})};
}
struct Fixture {
    p::MlpActorCritic policy{nullptr}, target{nullptr};
    std::unique_ptr<rl::Adam> optimizer;
    std::unique_ptr<rl::DqnTrainer> trainer;
    torch::Tensor observations, next_observations;
    Fixture(const bool double_q, const bool batch_norm) {
        torch::manual_seed(503);
        const p::MlpActorCriticOptions options{3,4,2,8,0.,batch_norm};
        policy = p::MlpActorCritic(options);
        auto generator = at::detail::getDefaultCPUGenerator(); const auto rng = generator.get_state();
        target = p::MlpActorCritic(options); generator.set_state(rng);
        optimizer = std::make_unique<rl::Adam>(std::vector<rl::AdamParameterGroup>{
            {rl::unique_parameters(*policy->actor), rl::AdamOptions{.003,.9,.999,1e-8,.01}},
            {rl::unique_parameters(*policy->critic), rl::AdamOptions{.007,.9,.999,1e-8,.01}}});
        observations = values(); next_observations = observations.flip({1}) + .125;
        trainer = std::make_unique<rl::DqnTrainer>(*optimizer, *policy, *target,
            [this](const torch::Tensor& indices, const bool next) { return policy->act((next ? next_observations : observations).index_select(0, indices)); },
            [this](const torch::Tensor& indices, const bool next) { return target->act((next ? next_observations : observations).index_select(0, indices)); },
            rl::DqnOptions{4,2,.97,.25,double_q});
    }
};
void optimizer_state(const rl::Adam& optimizer) {
    for (std::size_t i = 0; i < optimizer.parameters().size(); ++i) {
        const auto name = "optimizer_" + std::to_string(i) + ".";
        const auto& state = optimizer.moments()[i];
        emit_tensor_bytes(name + "parameter", optimizer.parameters()[i]);
        emit_tensor_bytes(name + "gradient", optimizer.parameters()[i].grad());
        emit_tensor_bytes(name + "step", state.step); emit_tensor_bytes(name + "mean", state.exp_avg);
        emit_tensor_bytes(name + "variance", state.exp_avg_sq); emit_tensor_bytes(name + "maximum", state.max_exp_avg_sq);
    }
}
void run(const bool double_q, const bool batch_norm) {
    Fixture fixture(double_q, batch_norm);
    std::vector<std::int64_t> logging_order;
    for (int update = 0; update < 4; ++update) {
        auto data = batch();
        fixture.trainer->after_update = [&](const rl::DqnUpdateResult& result) {
            logging_order.push_back(data.size()); logging_order.push_back(result.update_time);
        };
        const auto result = fixture.trainer->update(data);
        std::cout << "UPDATE " << update << '\n';
        emit_tensor_bytes("loss", result.loss);
        emit_tensor_bytes("update_time", torch::tensor(result.update_time));
        emit_tensor_bytes("buffer_size", torch::tensor(data.size()));
        emit_tensor_bytes("target_synchronized", torch::tensor(result.target_synchronized));
        emit_module_bytes(*fixture.policy, true);
        std::cout << "TARGET\n"; emit_module_bytes(*fixture.target, false);
        optimizer_state(*fixture.optimizer);
        emit_tensor_bytes("rng_tail", torch::rand({4}));
    }
    emit_tensor_bytes("logging_order", torch::tensor(logging_order, torch::kInt64));
}
void benchmark() {
    std::vector<std::int64_t> samples;
    torch::Tensor loss;
    std::vector<torch::Tensor> parameters;
    for (int repetition = -3; repetition < 5; ++repetition) {
        Fixture fixture(true, false); auto data = batch();
        const auto start = std::chrono::steady_clock::now();
        const auto result = fixture.trainer->update(data);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        if (repetition >= 0) samples.push_back(elapsed);
        if (repetition == -3) {
            loss = result.loss;
            for (const auto& p : fixture.optimizer->parameters()) parameters.push_back(p.detach().clone());
        } else {
            if (!torch::equal(loss, result.loss)) throw std::runtime_error("DQN benchmark loss drift");
            for (std::size_t i = 0; i < parameters.size(); ++i)
                if (!torch::equal(parameters[i], fixture.optimizer->parameters()[i])) throw std::runtime_error("DQN benchmark state drift");
        }
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
    std::cout << "]}\n"; emit_tensor_bytes("loss", loss);
    for (std::size_t i = 0; i < parameters.size(); ++i) emit_tensor_bytes("parameter_" + std::to_string(i), parameters[i]);
}
}
int main(int argc, char** argv) {
    try {
        torch::set_num_threads(1); torch::set_num_interop_threads(1);
        if (argc == 2 && std::string(argv[1]) == "--benchmark") { benchmark(); return 0; }
        if (argc != 1) throw std::invalid_argument("invalid DQN harness arguments");
        for (const bool batch_norm : {false,true}) for (const bool double_q : {false,true}) {
            std::cout << "CASE " << (double_q ? "double_dqn" : "dqn") << (batch_norm ? "_bn" : "") << '\n';
            run(double_q,batch_norm);
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
