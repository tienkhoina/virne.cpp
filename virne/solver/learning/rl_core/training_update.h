#pragma once

#include "training_types.h"

#include "adam.h"
#include "policy_action.h"
#include <functional>
#include <optional>

namespace virne::solver::learning::rl_core {

struct TrainingBatch {
    torch::Tensor actions, rewards, returns, old_log_probabilities;
    std::int64_t size() const noexcept { return actions.defined() ? actions.numel() : 0; }
    void clear() {
        actions = torch::Tensor{}; rewards = torch::Tensor{};
        returns = torch::Tensor{}; old_log_probabilities = torch::Tensor{};
    }
};
struct TrainingResult {
    torch::Tensor loss, actor_loss, critic_loss, entropy, gradient_norm, approximate_kl;
    std::int64_t applied_steps = 0;
    bool early_stopped = false;
};
using TrainingEvaluator = std::function<PolicyActionEvaluation(const torch::Tensor&, const torch::Tensor&)>;

// The evaluator gathers typed observations by CPU sample indices and executes
// the real policy with gradients. It must not run an inference/no-grad adapter.
class PolicyGradientTrainer final {
public:
    PolicyGradientTrainer(Adam& optimizer, TrainingEvaluator evaluator, TrainingOptions options = {});
    TrainingResult update(TrainingBatch& batch);
    std::int64_t update_time() const noexcept { return update_time_; }
    void set_update_time(std::int64_t value);
    std::function<void()> after_update;
private:
    Adam& optimizer_;
    TrainingEvaluator evaluator_;
    TrainingOptions options_;
    std::int64_t update_time_ = 0;
};

}  // namespace virne::solver::learning::rl_core
