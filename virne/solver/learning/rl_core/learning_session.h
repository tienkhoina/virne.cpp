#pragma once
#include "instance_environment.h"
#include "training_types.h"
#include "../rl_policy/policy_types.h"

namespace virne::solver::learning::rl_core {

struct LearningSessionOptions {
    rl_policy::PolicyKind policy = rl_policy::PolicyKind::mlp;
    TrainingOptions training;
    PolicyLearningRates learning_rates{1e-3, 1e-3, 1e-3, 1e-3, 1e-3};
    std::int64_t embedding_dim = 128, num_layers = 3, target_steps = 128;
    std::uint64_t seed = 0;
    double dropout = 0.0, weight_decay = 0.0, gamma = 0.99, gae_lambda = 0.98;
    bool batch_norm = false, include_failed = false;
    AdvantageMethod advantage = AdvantageMethod::gae;
    std::optional<std::size_t> max_episode_steps;
    std::optional<int> intraop_threads;
    DqnConfiguration dqn;
};
struct EpisodeLearningResult {
    EpisodeInfo solution;
    std::vector<std::int64_t> actions;
    std::vector<double> rewards, values, log_probabilities, returns;
    std::vector<std::uint8_t> dones;
    double last_value = 0.0;
    std::optional<double> update_loss;
    std::optional<double> actor_loss, critic_loss, entropy_loss,
        approximate_kl, gradient_norm, log_probability_mean,
        old_log_probability_mean, value_mean, return_mean,
        advantage_mean, reward_mean, learning_rate;
    std::int64_t applied_update_steps = 0;
    std::int64_t update_time = 0, pending_steps = 0;
    std::optional<double> exploration_epsilon;
    std::int64_t exploration_steps = 0;
};

// Tensor-free public boundary. Implementation and optimizer ownership live in
// the Torch ABI; InstanceEnvironment owns all graph objects in the graph ABI.
class LearningSession final {
public:
    explicit LearningSession(LearningSessionOptions options = {});
    ~LearningSession();
    LearningSession(LearningSession&&) noexcept;
    LearningSession& operator=(LearningSession&&) noexcept;
    LearningSession(const LearningSession&) = delete;
    LearningSession& operator=(const LearningSession&) = delete;
    EpisodeLearningResult learn(InstanceEnvironment& environment);
    EpisodeLearningResult solve(InstanceEnvironment& environment, bool sample = false);
    std::optional<double> update();
    std::int64_t pending_steps() const noexcept;
    std::int64_t update_time() const noexcept;
    void set_sampling_seed(std::uint64_t seed);
    void set_exploration_seed(std::uint32_t seed);
    std::vector<double> parameter_values() const;
    std::vector<double> target_parameter_values() const;
    std::vector<std::uint8_t> checkpoint() const;
    void restore_checkpoint(const std::vector<std::uint8_t>& checkpoint);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace virne::solver::learning::rl_core
