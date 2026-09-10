#pragma once
#include "adam.h"
#include <functional>
#include <utility>

namespace virne::solver::learning::rl_core {
struct DqnOptions {
    std::int64_t batch_size = 64, target_update_interval = 100;
    double gamma = .99, max_grad_norm = .5;
    bool double_q = false;
};
struct DqnBatch {
    torch::Tensor actions, rewards, dones;
    std::int64_t size() const { return actions.defined() ? actions.numel() : 0; }
    void clear() { actions = torch::Tensor{}; rewards = torch::Tensor{}; dones = torch::Tensor{}; }
};
struct DqnUpdateResult {
    torch::Tensor loss, q_values, target_q_values, rewards, dones, gradient_norm;
    std::int64_t update_time = 0;
    bool target_synchronized = false;
};
using QForward = std::function<torch::Tensor(const torch::Tensor& indices, bool next_observation)>;

class DqnTrainer final {
public:
    DqnTrainer(Adam& optimizer, torch::nn::Module& policy, torch::nn::Module& target,
        QForward online_forward, QForward target_forward, DqnOptions options = {});
    DqnUpdateResult update(DqnBatch& batch);
    void synchronize_target();
    std::int64_t update_time() const noexcept { return update_time_; }
    void set_update_time(std::int64_t value);
    std::function<void(const DqnUpdateResult&)> after_update;
private:
    Adam& optimizer_;
    QForward online_forward_, target_forward_;
    DqnOptions options_;
    std::vector<torch::Tensor> parameters_;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> target_copies_;
    std::int64_t update_time_ = 0;
};
}  // namespace virne::solver::learning::rl_core
