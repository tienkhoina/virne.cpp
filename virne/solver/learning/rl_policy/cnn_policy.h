#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace virne::solver::learning::rl_policy {

struct CnnActorCriticOptions {
    std::int64_t feature_dim = 0;
    // Accepted but unused by the Python CNN architecture.
    std::int64_t action_dim = 0;
    std::int64_t embedding_dim = 128;
};

class CnnActorImpl final : public torch::nn::Module {
public:
    explicit CnnActorImpl(CnnActorCriticOptions options);
    torch::Tensor forward(const torch::Tensor& p_net_x);
    torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(CnnActor);

class CnnCriticImpl final : public torch::nn::Module {
public:
    explicit CnnCriticImpl(CnnActorCriticOptions options);
    torch::Tensor forward(const torch::Tensor& p_net_x);
    torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(CnnCritic);

class CnnActorCriticImpl final : public torch::nn::Module {
public:
    explicit CnnActorCriticImpl(CnnActorCriticOptions options);
    torch::Tensor act(const torch::Tensor& p_net_x);
    torch::Tensor evaluate(const torch::Tensor& p_net_x);
    CnnActor actor{nullptr};
    CnnCritic critic{nullptr};
};
TORCH_MODULE(CnnActorCritic);

}  // namespace virne::solver::learning::rl_policy
