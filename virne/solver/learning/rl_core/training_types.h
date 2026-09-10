#pragma once
#include <cstdint>
#include <optional>

namespace virne::solver::learning::rl_core {
enum class UpdateAlgorithm : std::uint8_t { pg, a2c, ppo, arppo, source_ddpg, dqn, double_dqn };
enum class AdvantageMethod : std::uint8_t { gae, monte_carlo, temporal_difference };
struct TrainingOptions {
    UpdateAlgorithm algorithm = UpdateAlgorithm::ppo;
    std::int64_t batch_size = 32;
    double repeat_times = 10.0, eps_clip = 0.2;
    bool normalize_returns = false, normalize_advantage = false, clip_grad = true;
    double max_grad_norm = 0.5, critic_coefficient = 0.5, entropy_coefficient = 0.01, mask_coefficient = 0.0;
    std::optional<double> target_kl;
};
struct PolicyLearningRates { std::optional<double> actor, critic, encoder, cost_critic, lambda_net; };
struct DqnConfiguration {
    std::int64_t target_update_interval = 100;
    double epsilon_start = 1., epsilon_end = .1, epsilon_decay = 10000.;
    std::uint32_t seed = 0;
    bool mask_actions = true, maskable_policy = true;
};
}  // namespace virne::solver::learning::rl_core
