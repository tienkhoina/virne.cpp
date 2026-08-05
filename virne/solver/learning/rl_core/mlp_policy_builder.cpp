#include "mlp_policy_builder.h"

#include <limits>

namespace virne::solver::learning::rl_core {
namespace {

std::int64_t checked_feature_sum(
    const std::int64_t p_net_x_dim,
    const std::int64_t v_node_x_dim) {
    if (p_net_x_dim <= 0 || v_node_x_dim <= 0) {
        throw PolicyConfigException(
            PolicyConfigErrorCode::invalid_feature_dimension,
            "MLP policy feature dimensions must be positive");
    }
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (p_net_x_dim > maximum - v_node_x_dim) {
        throw PolicyConfigException(
            PolicyConfigErrorCode::feature_dimension_overflow,
            "combined MLP policy feature dimension exceeds the int64 "
            "LibTorch shape domain");
    }
    return p_net_x_dim + v_node_x_dim;
}

}  // namespace

rl_policy::MlpActorCriticOptions make_mlp_actor_critic_options(
    const PolicyConfiguration& configuration) {
    const auto network = validate_policy_general_network_config(
        configuration.neural_network);
    const auto& dimensions = configuration.feature_dimensions;
    if (dimensions.p_net_num_nodes <= 0) {
        throw PolicyConfigException(
            PolicyConfigErrorCode::invalid_physical_node_count,
            "MLP policy action dimension must be positive");
    }

    rl_policy::MlpActorCriticOptions options;
    options.feature_dim = checked_feature_sum(
        dimensions.p_net_x_dim,
        dimensions.v_node_x_dim);
    options.action_dim = dimensions.p_net_num_nodes;
    // Python PolicyBuilder does not pass num_layers, so preserve the policy's
    // constructor default of three instead of introducing another config key.
    options.embedding_dim = network.embedding_dim;
    options.dropout_prob = network.dropout_prob;
    options.batch_norm = network.batch_norm;
    return options;
}

rl_policy::MlpActorCritic build_mlp_policy(
    const PolicyConfiguration& configuration,
    const torch::Device& device) {
    rl_policy::MlpActorCritic policy(
        make_mlp_actor_critic_options(configuration));
    policy->to(device);
    return policy;
}

}  // namespace virne::solver::learning::rl_core
