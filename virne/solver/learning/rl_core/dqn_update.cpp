#include "dqn_update.h"
#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace virne::solver::learning::rl_core {
DqnTrainer::DqnTrainer(Adam& optimizer, torch::nn::Module& policy, torch::nn::Module& target,
    QForward online_forward, QForward target_forward, DqnOptions options)
    : optimizer_(optimizer), online_forward_(std::move(online_forward)), target_forward_(std::move(target_forward)),
      options_(options), parameters_(unique_parameters(policy)) {
    if (!online_forward_ || !target_forward_ || options.batch_size <= 0 || options.target_update_interval <= 0
        || !std::isfinite(options.gamma) || !std::isfinite(options.max_grad_norm)) throw std::invalid_argument("invalid DQN options");
    const auto resolve = [&](const auto& source, const auto& destination) {
        if (source.size() != destination.size()) throw std::invalid_argument("target network state count mismatch");
        for (const auto& item : source) {
            const auto* matched = destination.find(item.key());
            if (!matched || matched->sizes() != item.value().sizes() || matched->scalar_type() != item.value().scalar_type())
                throw std::invalid_argument("target network state layout mismatch");
            target_copies_.emplace_back(*matched, item.value());
        }
    };
    resolve(policy.named_parameters(true), target.named_parameters(true));
    resolve(policy.named_buffers(true), target.named_buffers(true));
    synchronize_target();
}
void DqnTrainer::synchronize_target() {
    torch::NoGradGuard no_grad;
    for (const auto& pair : target_copies_) pair.first.copy_(pair.second);
}
void DqnTrainer::set_update_time(const std::int64_t value) {
    if (value < 0) throw std::invalid_argument("DQN update count must be non-negative");
    update_time_ = value;
}
DqnUpdateResult DqnTrainer::update(DqnBatch& batch) {
    const auto size = batch.size();
    if (size <= 0 || batch.actions.dim() != 1 || batch.actions.scalar_type() != torch::kInt64
        || !batch.rewards.defined() || !batch.dones.defined() || batch.rewards.dim() != 1 || batch.dones.dim() != 1
        || batch.rewards.numel() != size || batch.dones.numel() != size) throw std::invalid_argument("invalid DQN batch");
    if (update_time_ == std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("DQN update counter overflow");
    torch::AutoGradMode enable_grad(true);
    const auto indices = torch::randint(0, size, {options_.batch_size}, torch::kInt64);
    const auto all_values = online_forward_(indices, false);
    if (all_values.dim() != 2 || all_values.size(0) != options_.batch_size) throw std::invalid_argument("DQN policy requires a batch/action matrix");
    const auto device = all_values.device();
    const auto actions = batch.actions.index_select(0, indices.to(batch.actions.device())).to(device);
    const auto rewards = batch.rewards.index_select(0, indices.to(batch.rewards.device())).to(device);
    const auto dones = batch.dones.index_select(0, indices.to(batch.dones.device())).to(device);
    const auto q_values = all_values.gather(1, actions.unsqueeze(-1)).squeeze(-1);
    torch::Tensor target_q_values;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor next_values;
        if (options_.double_q) {
            const auto next_actions = std::get<1>(online_forward_(indices, true).max(1));
            next_values = target_forward_(indices, true).gather(1, next_actions.unsqueeze(-1)).squeeze(-1);
        } else next_values = std::get<0>(target_forward_(indices, true).max(1));
        target_q_values = rewards + options_.gamma * next_values * (1 - dones);
    }
    const auto loss = torch::mse_loss(q_values, target_q_values);
    optimizer_.zero_grad(); loss.backward();
    const auto norm = clip_grad_norm(parameters_, options_.max_grad_norm);
    optimizer_.step();
    // The source synchronizes on update zero, before incrementing its counter.
    const bool synchronize = update_time_ % options_.target_update_interval == 0;
    if (synchronize) synchronize_target();
    DqnUpdateResult result{loss.detach(), q_values.detach(), target_q_values.detach(), rewards.detach(), dones.detach(),
        norm.detach(), ++update_time_, synchronize};
    if (after_update) after_update(result);
    batch.clear();
    return result;
}
}  // namespace virne::solver::learning::rl_core
