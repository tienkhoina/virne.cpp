#include "../../virne/solver/learning/rl_core/mlp_policy_builder.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace core = virne::solver::learning::rl_core;
namespace policy = virne::solver::learning::rl_policy;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

core::PolicyConfiguration configuration(
    const std::int64_t p_net_x_dim = 14,
    const std::int64_t v_node_x_dim = 11,
    const std::int64_t p_net_num_nodes = 500,
    const std::int64_t embedding_dim = 128,
    const double dropout_prob = 0.0,
    const bool batch_norm = false) {
    core::PolicyConfiguration result;
    result.feature_dimensions.p_net_x_dim = p_net_x_dim;
    result.feature_dimensions.v_node_x_dim = v_node_x_dim;
    result.feature_dimensions.p_net_num_nodes = p_net_num_nodes;
    result.neural_network.embedding_dim = embedding_dim;
    result.neural_network.dropout_prob = dropout_prob;
    result.neural_network.batch_norm = batch_norm;
    return result;
}

void test_exact_option_mapping() {
    const auto options = core::make_mlp_actor_critic_options(
        configuration(14, 11, 500, 256, 0.25, true));
    require(options.feature_dim == 25,
            "feature_dim must sum p_net_x_dim and v_node_x_dim");
    require(options.action_dim == 500,
            "action_dim must use the physical node count");
    require(options.num_layers == 3,
            "PolicyBuilder must preserve MlpActorCritic's default layers");
    require(options.embedding_dim == 256,
            "embedding_dim was not forwarded");
    require(options.dropout_prob == 0.25,
            "dropout_prob was not forwarded");
    require(options.batch_norm,
            "batch_norm was not forwarded");
}

void test_constructs_on_requested_device() {
    torch::manual_seed(31);
    auto policy = core::build_mlp_policy(
        configuration(4, 3, 7, 5), torch::Device(torch::kCPU));
    const auto& options = policy->options();
    require(options.feature_dim == 7, "constructed feature_dim mismatch");
    require(options.action_dim == 7, "constructed action_dim mismatch");

    for (const auto& parameter : policy->parameters()) {
        require(parameter.device().is_cpu(),
                "policy parameter was not moved to requested CPU device");
    }
    for (const auto& buffer : policy->buffers()) {
        require(buffer.device().is_cpu(),
                "policy buffer was not moved to requested CPU device");
    }

    const auto input = torch::zeros({2, 7, 7});
    require(policy->act(input).sizes().vec()
                == std::vector<std::int64_t>({2, 7}),
            "constructed actor output shape mismatch");
    require(policy->evaluate(input).sizes().vec()
                == std::vector<std::int64_t>({2, 1}),
            "constructed critic output shape mismatch");
}

void require_same_state(
    const policy::MlpActorCritic& left,
    const policy::MlpActorCritic& right) {
    const auto left_parameters = left->named_parameters(true);
    const auto right_parameters = right->named_parameters(true);
    require(left_parameters.size() == right_parameters.size(),
            "adapter changed policy parameter count");
    for (const auto& item : left_parameters) {
        require(right_parameters.contains(item.key()),
                "adapter changed policy parameter names");
        require(torch::equal(item.value(), right_parameters[item.key()]),
                "adapter changed initialized policy parameters");
    }

    const auto left_buffers = left->named_buffers(true);
    const auto right_buffers = right->named_buffers(true);
    require(left_buffers.size() == right_buffers.size(),
            "adapter changed policy buffer count");
    for (const auto& item : left_buffers) {
        require(right_buffers.contains(item.key()),
                "adapter changed policy buffer names");
        require(torch::equal(item.value(), right_buffers[item.key()]),
                "adapter changed initialized policy buffers");
    }
}

void test_matches_direct_python_shaped_construction() {
    const auto config = configuration(6, 5, 23, 9, 0.125, false);
    const auto options = core::make_mlp_actor_critic_options(config);

    torch::manual_seed(90210);
    auto built = core::build_mlp_policy(config, torch::Device(torch::kCPU));
    const auto built_rng_continuation = torch::rand({4});

    torch::manual_seed(90210);
    policy::MlpActorCritic direct(options);
    direct->to(torch::Device(torch::kCPU));
    const auto direct_rng_continuation = torch::rand({4});

    require_same_state(built, direct);
    require(torch::equal(built_rng_continuation, direct_rng_continuation),
            "adapter changed constructor RNG consumption");
    require(built->is_training() == direct->is_training(),
            "adapter changed module training mode");
}

template <typename Function>
void require_rejected(Function&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

void test_cold_validation() {
    require_rejected(
        [] {
            static_cast<void>(core::make_mlp_actor_critic_options(
                configuration(0, 3)));
        },
        "non-positive physical feature dimension accepted");
    require_rejected(
        [] {
            static_cast<void>(core::make_mlp_actor_critic_options(
                configuration(3, 0)));
        },
        "non-positive virtual-node feature dimension accepted");
    require_rejected(
        [] {
            static_cast<void>(core::make_mlp_actor_critic_options(
                configuration(3, 4, 0)));
        },
        "non-positive action dimension accepted");
    require_rejected(
        [] {
            static_cast<void>(core::make_mlp_actor_critic_options(
                configuration(
                    std::numeric_limits<std::int64_t>::max(), 1)));
        },
        "combined feature dimension overflow accepted");
    require_rejected(
        [] {
            static_cast<void>(core::make_mlp_actor_critic_options(
                configuration(3, 4, 5, 0)));
        },
        "invalid neural network config accepted");
}

void test_cuda_device_smoke_if_available() {
    if (!torch::cuda::is_available()) {
        return;
    }
    auto policy = core::build_mlp_policy(
        configuration(4, 3, 7, 5), torch::Device(torch::kCUDA, 0));
    for (const auto& parameter : policy->parameters()) {
        require(parameter.device().is_cuda(),
                "policy parameter was not moved to requested CUDA device");
    }
    const auto input = torch::zeros(
        {2, 7, 7}, torch::TensorOptions().device(torch::kCUDA));
    require(policy->act(input).device().is_cuda(),
            "constructed CUDA actor returned a non-CUDA tensor");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_exact_option_mapping();
        test_constructs_on_requested_device();
        test_matches_direct_python_shaped_construction();
        test_cold_validation();
        test_cuda_device_smoke_if_available();
        std::cout << "mlp_policy_builder_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_policy_builder_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
