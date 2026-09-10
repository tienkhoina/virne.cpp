#include "cnn_policy.h"

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>

#include <stdexcept>

namespace virne::solver::learning::rl_policy {
namespace {

torch::nn::Conv2d make_convolution(const CnnActorCriticOptions& options) {
    if (options.feature_dim <= 0) {
        throw std::invalid_argument("CNN feature_dim must be positive");
    }
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {1, options.feature_dim}).stride({1, 1}));
}

}  // namespace

CnnActorImpl::CnnActorImpl(CnnActorCriticOptions options) {
    net = register_module("net", torch::nn::Sequential(
        make_convolution(options), torch::nn::ReLU(), torch::nn::Flatten()));
}

torch::Tensor CnnActorImpl::forward(const torch::Tensor& p_net_x) {
    return net->forward(p_net_x);
}

CnnCriticImpl::CnnCriticImpl(CnnActorCriticOptions options) {
    net = register_module("net", torch::nn::Sequential(
        make_convolution(options), torch::nn::Flatten(), torch::nn::ReLU()));
}

torch::Tensor CnnCriticImpl::forward(const torch::Tensor& p_net_x) {
    return net->forward(p_net_x).mean(1);
}

CnnActorCriticImpl::CnnActorCriticImpl(CnnActorCriticOptions options) {
    actor = register_module("actor", CnnActor(options));
    critic = register_module("critic", CnnCritic(options));
}

torch::Tensor CnnActorCriticImpl::act(const torch::Tensor& p_net_x) {
    return actor->forward(p_net_x);
}

torch::Tensor CnnActorCriticImpl::evaluate(const torch::Tensor& p_net_x) {
    return critic->forward(p_net_x);
}

}  // namespace virne::solver::learning::rl_policy
