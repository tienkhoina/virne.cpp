#include "training_update.h"
#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace virne::solver::learning::rl_core {
namespace {
torch::Tensor normalized(const torch::Tensor& input, const double epsilon) {
    const auto centered = input - input.mean();
    const auto deviation = input.std() + epsilon;
    return centered / deviation;
}
torch::Tensor detached(const torch::Tensor& input) { return input.defined() ? input.detach() : torch::Tensor{}; }
}
PolicyGradientTrainer::PolicyGradientTrainer(Adam& optimizer, TrainingEvaluator evaluator, TrainingOptions options)
    : optimizer_(optimizer), evaluator_(std::move(evaluator)), options_(options) {
    if (options.algorithm > UpdateAlgorithm::source_ddpg || !evaluator_ || options.batch_size <= 0 || !std::isfinite(options.repeat_times)
        || options.repeat_times < 0 || !std::isfinite(options.eps_clip)) throw std::invalid_argument("invalid training options");
}
void PolicyGradientTrainer::set_update_time(const std::int64_t value) {
    if (value < 0) throw std::invalid_argument("update_time must be non-negative");
    update_time_ = value;
}
TrainingResult PolicyGradientTrainer::update(TrainingBatch& batch) {
    torch::AutoGradMode enable_grad(true);
    const auto size = batch.size();
    if (size <= 0 || !batch.returns.defined() || batch.returns.numel() != size) throw std::invalid_argument("empty or inconsistent training batch");
    const bool pg = options_.algorithm == UpdateAlgorithm::pg;
    const bool a2c = options_.algorithm == UpdateAlgorithm::a2c;
    const bool repeated = !pg && !a2c;
    if (options_.algorithm == UpdateAlgorithm::arppo && size < options_.batch_size) throw std::invalid_argument("ARPPO buffer smaller than batch size");
    if (repeated && (!batch.old_log_probabilities.defined() || batch.old_log_probabilities.numel() != size)) throw std::invalid_argument("missing old action log probabilities");
    auto returns = batch.returns;
    if (repeated && options_.algorithm != UpdateAlgorithm::arppo && options_.normalize_returns) returns = normalized(returns, 1e-9);
    const double repeat_count = static_cast<double>(size) * options_.repeat_times / static_cast<double>(options_.batch_size);
    if (repeat_count >= static_cast<double>(std::numeric_limits<std::int64_t>::max() - 1)) throw std::overflow_error("training repeat count overflow");
    const std::int64_t sample_times = repeated ? 1 + static_cast<std::int64_t>(repeat_count) : 1;
    TrainingResult result;
    for (std::int64_t iteration = 0; iteration < sample_times; ++iteration) {
        const auto indices = repeated
            ? torch::randint(0, size, {options_.batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
            : torch::arange(size, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        const auto actions = batch.actions.index_select(0, indices.to(batch.actions.device()));
        const auto selected_returns = returns.index_select(0, indices.to(returns.device()));
        const auto selected_rewards = batch.rewards.index_select(0, indices.to(batch.rewards.device()));
        const auto evaluated = evaluator_(indices, actions);
        const auto target = selected_returns.to(evaluated.log_probabilities.device());
        torch::Tensor actor, critic, entropy, approximate_kl;
        torch::Tensor loss;
        if (pg) {
            loss = -(evaluated.log_probabilities * target).mean();
            result.log_probability_mean = evaluated.log_probabilities.mean().detach();
            result.return_mean = target.mean().detach();
            result.reward_mean = selected_rewards.to(evaluated.log_probabilities.device()).mean().detach();
        } else {
            if (!evaluated.values) throw std::invalid_argument("actor-critic update requires critic values");
            const auto value = *evaluated.values;
            auto advantage = target - value.detach();
            if (options_.normalize_advantage && (a2c || value.numel() != 0)) advantage = normalized(advantage, a2c ? 1e-8 : 1e-9);
            if (a2c) {
                actor = -(evaluated.log_probabilities * advantage).mean();
            } else {
                const auto old_logp = batch.old_log_probabilities.index_select(0, indices.to(batch.old_log_probabilities.device())).to(evaluated.log_probabilities.device());
                const auto ratio = torch::exp(evaluated.log_probabilities - old_logp);
                const auto first = ratio * advantage;
                const auto second = torch::clamp(ratio, 1 - options_.eps_clip, 1 + options_.eps_clip) * advantage;
                actor = -torch::min(first, second).mean();
                approximate_kl = (old_logp - evaluated.log_probabilities).mean().detach();
            }
            critic = torch::mse_loss(target, value);
            entropy = evaluated.entropy.mean();
            loss = actor + options_.critic_coefficient * critic;
            // A2C's positive entropy sign is intentional in the reference.
            loss = a2c ? loss + options_.entropy_coefficient * entropy : loss - options_.entropy_coefficient * entropy;
            if (repeated) {
                loss = evaluated.masked_action_probability ? loss + options_.mask_coefficient * *evaluated.masked_action_probability : loss + 0.0;
                loss = evaluated.prediction_loss ? loss + *evaluated.prediction_loss : loss + 0.0;
            }
            result.log_probability_mean = evaluated.log_probabilities.mean().detach();
            result.old_log_probability_mean = repeated
                ? batch.old_log_probabilities.index_select(0, indices.to(batch.old_log_probabilities.device())).to(evaluated.log_probabilities.device()).mean().detach()
                : torch::Tensor{};
            result.value_mean = value.mean().detach();
            result.return_mean = target.mean().detach();
            result.advantage_mean = advantage.mean().detach();
            result.reward_mean = selected_rewards.to(evaluated.log_probabilities.device()).mean().detach();
        }
        result.loss = loss.detach(); result.actor_loss = detached(actor); result.critic_loss = detached(critic);
        result.entropy = detached(entropy); result.approximate_kl = approximate_kl;
        if (options_.algorithm == UpdateAlgorithm::ppo && options_.target_kl
            && (approximate_kl > 1.5 * *options_.target_kl).item<bool>()) {
            result.early_stopped = true;
            break;
        }
        optimizer_.zero_grad();
        loss.backward();
        if (options_.clip_grad) result.gradient_norm = clip_grad_norm(optimizer_.parameters(), options_.max_grad_norm);
        optimizer_.step();
        if (repeated) ++update_time_;
        ++result.applied_steps;
    }
    if (!repeated) batch.clear();
    if (after_update) after_update();
    if (repeated) batch.clear();
    else ++update_time_;
    return result;
}

}  // namespace virne::solver::learning::rl_core
