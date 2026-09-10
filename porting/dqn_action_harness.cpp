#include "tensor_bytes.h"
#include "../virne/solver/learning/rl_core/dqn_action.h"
#include <torch/torch.h>
#include <chrono>

namespace rl = virne::solver::learning::rl_core;
using virne::porting::emit_tensor_bytes;
namespace {
void rng_state(const NumpyRandomState::Snapshot& s) {
    const std::vector<std::int64_t> keys(s.key.begin(),s.key.end());
    emit_tensor_bytes("rng_key",torch::tensor(keys,torch::kInt64));
    emit_tensor_bytes("rng_position",torch::tensor(static_cast<std::int64_t>(s.position)));
    emit_tensor_bytes("rng_has_gaussian",torch::tensor(s.has_gaussian));
    emit_tensor_bytes("rng_gaussian",torch::tensor(s.gaussian,torch::kFloat64));
}
void run(const int variant) {
    rl::DqnExplorationOptions o{.9,.1,7.,4,variant != 1,variant != 2,713};
    rl::DqnActionSelector selector(o);
    const auto values = torch::tensor({{.2F,.6F,-.4F,.8F}});
    const auto mask = torch::tensor({{true,true,false,false}},torch::kBool);
    std::vector<std::int64_t> actions,steps,calls;
    std::vector<double> epsilons;
    std::int64_t call_count = 0;
    rl::DqnExplorationState saved;
    torch::Tensor logprob;
    for (int i = 0; i < 40; ++i) {
        if (i == 18) selector.restore(saved);
        const auto chosen = selector.select([&] { ++call_count; return values; },mask,i != 8 && i != 23);
        const auto state = selector.snapshot();
        actions.push_back(chosen.action); steps.push_back(state.steps_done); calls.push_back(call_count); epsilons.push_back(state.epsilon);
        logprob = chosen.log_probability;
        if (i == 10) saved = state;
    }
    emit_tensor_bytes("actions",torch::tensor(actions,torch::kInt64));
    emit_tensor_bytes("steps",torch::tensor(steps,torch::kInt64));
    emit_tensor_bytes("calls",torch::tensor(calls,torch::kInt64));
    emit_tensor_bytes("epsilon",torch::tensor(epsilons,torch::kFloat64));
    emit_tensor_bytes("logprob",logprob);
    rng_state(selector.snapshot().random);
}
void gaussian_resume() {
    NumpyRandomState random(917);
    random.normal();
    const auto saved = random.snapshot(); rng_state(saved);
    random.random(); random.normal(); random.restore(saved);
    const std::vector<double> values{random.normal(),random.normal(),random.normal()};
    emit_tensor_bytes("normal",torch::tensor(values,torch::kFloat64)); rng_state(random.snapshot());
    auto invalid = saved; invalid.position = 625;
    bool rejected = false;
    try { random.restore(invalid); } catch (const std::invalid_argument&) { rejected = true; }
    emit_tensor_bytes("invalid_rejected",torch::tensor(rejected)); rng_state(random.snapshot());
}
}
int main(int argc,char**) {
    try {
        if (argc != 1) throw std::invalid_argument("invalid DQN action harness arguments");
        torch::set_num_threads(1); torch::set_num_interop_threads(1);
        for (int variant = 0; variant < 3; ++variant) { std::cout << "CASE epsilon_" << variant << '\n'; run(variant); }
        std::cout << "CASE rng_gaussian_resume\n"; gaussian_resume();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
