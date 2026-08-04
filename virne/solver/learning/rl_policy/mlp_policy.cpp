#include "mlp_policy.h"

#include <stdexcept>
#include <utility>

namespace virne::solver::learning::rl_policy {
namespace {

MlpActorCriticOptions validate_options(MlpActorCriticOptions options) {
    if (options.feature_dim <= 0) {
        throw std::invalid_argument(
            "MlpActorCritic feature_dim must be positive");
    }
    if (options.action_dim <= 0) {
        throw std::invalid_argument(
            "MlpActorCritic action_dim must be positive");
    }
    return options;
}

neural_network::MlpOptions make_mlp_options(
    const MlpActorCriticOptions& options) {
    neural_network::MlpOptions result{};
    result.input_dim = options.feature_dim;
    result.output_dim = 1;
    result.num_layers = options.num_layers;
    result.embedding_dims = options.embedding_dim;
    result.batch_norm = options.batch_norm;
    result.dropout_prob = options.dropout_prob;
    return result;
}

} // namespace

ActorImpl::ActorImpl(MlpActorCriticOptions options)
    : options_(validate_options(std::move(options))) {
    net = register_module(
        "net",
        neural_network::MLPNet(make_mlp_options(options_)));
}

torch::Tensor ActorImpl::forward(const torch::Tensor& p_net_x) {
    return net->forward(p_net_x).squeeze(-1);
}

const MlpActorCriticOptions& ActorImpl::options() const noexcept {
    return options_;
}

CriticImpl::CriticImpl(MlpActorCriticOptions options)
    : options_(validate_options(std::move(options))) {
    net = register_module(
        "net",
        neural_network::MLPNet(make_mlp_options(options_)));
}

torch::Tensor CriticImpl::forward(const torch::Tensor& p_net_x) {
    return net->forward(p_net_x).mean(1);
}

const MlpActorCriticOptions& CriticImpl::options() const noexcept {
    return options_;
}

MlpActorCriticImpl::MlpActorCriticImpl(MlpActorCriticOptions options)
    : options_(validate_options(std::move(options))) {
    actor = register_module("actor", Actor(options_));
    critic = register_module("critic", Critic(options_));
}

torch::Tensor MlpActorCriticImpl::act(const torch::Tensor& p_net_x) {
    return actor->forward(p_net_x);
}

torch::Tensor MlpActorCriticImpl::evaluate(const torch::Tensor& p_net_x) {
    return critic->forward(p_net_x);
}

const MlpActorCriticOptions& MlpActorCriticImpl::options() const noexcept {
    return options_;
}

} // namespace virne::solver::learning::rl_policy
