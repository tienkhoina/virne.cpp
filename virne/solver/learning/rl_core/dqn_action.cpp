#include "dqn_action.h"
#include "action_mask.h"
#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace virne::solver::learning::rl_core {
DqnActionSelector::DqnActionSelector(DqnExplorationOptions options)
    : options_(options), random_(options.seed), epsilon_(options.epsilon_start) {
    if (options.num_actions <= 0 || !(options.epsilon_decay > 0) || !std::isfinite(options.epsilon_decay)
        || !std::isfinite(options.epsilon_start) || !std::isfinite(options.epsilon_end)) throw std::invalid_argument("invalid DQN exploration options");
}
DqnActionSelection DqnActionSelector::select(const std::function<torch::Tensor()>& forward, const std::optional<torch::Tensor>& mask, const bool sample) {
    const auto result = [](const std::int64_t action) { return DqnActionSelection{action, torch::zeros({1,1}, torch::kFloat32)}; };
    if (sample) {
        if (steps_done_ == std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("DQN action counter overflow");
        ++steps_done_;
        const auto exponent = torch::tensor(-1. * static_cast<double>(steps_done_) / options_.epsilon_decay,
            torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
        // Torch's pinned CPU exp matches NumPy here; libm exp differs by an ULP.
        epsilon_ = options_.epsilon_end + (options_.epsilon_start - options_.epsilon_end)
            * exponent.exp().item<double>();
        if (random_.rand() < epsilon_) return result(random_.randint(options_.num_actions));
    }
    if (!forward) throw std::invalid_argument("missing DQN policy callback");
    torch::NoGradGuard no_grad;
    const auto q_values = forward();
    if (!q_values.defined() || q_values.numel() != options_.num_actions) throw std::invalid_argument("DQN greedy action requires one Q-value per action");
    const auto logits = mask && options_.mask_actions && options_.maskable_policy
        ? apply_mask_to_logit(q_values, *mask) : q_values;
    return result(logits.argmax().item<std::int64_t>());
}
void DqnActionSelector::seed(const std::uint32_t seed) { random_.seed(seed); }
DqnExplorationState DqnActionSelector::snapshot() const noexcept { return {random_.snapshot(), steps_done_, epsilon_}; }
void DqnActionSelector::restore(const DqnExplorationState& state) {
    if (state.steps_done < 0 || !std::isfinite(state.epsilon)) throw std::invalid_argument("invalid DQN exploration state");
    random_.restore(state.random); steps_done_ = state.steps_done; epsilon_ = state.epsilon;
}
}  // namespace virne::solver::learning::rl_core
