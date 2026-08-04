#pragma once

#include "../neural_network/mlp.h"

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace virne::solver::learning::rl_policy {

struct MlpActorCriticOptions {
    std::int64_t feature_dim = 0;

    // Retained for exact compatibility with Python MlpActorCritic. Neither
    // Actor nor Critic uses this value when constructing its MLP.
    std::int64_t action_dim = 0;

    std::int64_t num_layers = 3;
    std::int64_t embedding_dim = 128;
    double dropout_prob = 0.0;
    bool batch_norm = false;
};

class ActorImpl final : public torch::nn::Module {
public:
    explicit ActorImpl(MlpActorCriticOptions options);

    torch::Tensor forward(const torch::Tensor& p_net_x);

    [[nodiscard]] const MlpActorCriticOptions& options() const noexcept;

    // Public and registered as `net` to preserve Python state names.
    neural_network::MLPNet net{nullptr};

private:
    MlpActorCriticOptions options_;
};

TORCH_MODULE(Actor);

class CriticImpl final : public torch::nn::Module {
public:
    explicit CriticImpl(MlpActorCriticOptions options);

    torch::Tensor forward(const torch::Tensor& p_net_x);

    [[nodiscard]] const MlpActorCriticOptions& options() const noexcept;

    // Public and registered as `net` to preserve Python state names.
    neural_network::MLPNet net{nullptr};

private:
    MlpActorCriticOptions options_;
};

TORCH_MODULE(Critic);

class MlpActorCriticImpl final : public torch::nn::Module {
public:
    explicit MlpActorCriticImpl(MlpActorCriticOptions options);

    torch::Tensor act(const torch::Tensor& p_net_x);
    torch::Tensor evaluate(const torch::Tensor& p_net_x);

    [[nodiscard]] const MlpActorCriticOptions& options() const noexcept;

    // Registration order and names match Python BaseActorCritic checkpoints.
    Actor actor{nullptr};
    Critic critic{nullptr};

private:
    MlpActorCriticOptions options_;
};

TORCH_MODULE(MlpActorCritic);

} // namespace virne::solver::learning::rl_policy
