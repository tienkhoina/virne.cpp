#include "../virne/solver/learning/rl_policy/mlp_policy.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace policy = virne::solver::learning::rl_policy;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

policy::MlpActorCriticOptions options(
    const std::int64_t feature_dim,
    const std::int64_t action_dim,
    const std::int64_t layers = 3,
    const std::int64_t embedding_dim = 7,
    const bool batch_norm = false,
    const double dropout_prob = 0.0) {
    policy::MlpActorCriticOptions result{};
    result.feature_dim = feature_dim;
    result.action_dim = action_dim;
    result.num_layers = layers;
    result.embedding_dim = embedding_dim;
    result.dropout_prob = dropout_prob;
    result.batch_norm = batch_norm;
    return result;
}

void require_same_state(
    policy::MlpActorCritic& left,
    policy::MlpActorCritic& right) {
    const auto left_parameters = left->named_parameters(true);
    const auto right_parameters = right->named_parameters(true);
    require(left_parameters.size() == right_parameters.size(),
            "action_dim changed parameter count");
    for (const auto& item : left_parameters) {
        require(right_parameters.contains(item.key()),
                "action_dim changed parameter names");
        require(torch::equal(item.value(), right_parameters[item.key()]),
                "action_dim changed initialized parameters");
    }

    const auto left_buffers = left->named_buffers(true);
    const auto right_buffers = right->named_buffers(true);
    require(left_buffers.size() == right_buffers.size(),
            "action_dim changed buffer count");
    for (const auto& item : left_buffers) {
        require(right_buffers.contains(item.key()),
                "action_dim changed buffer names");
        require(torch::equal(item.value(), right_buffers[item.key()]),
                "action_dim changed initialized buffers");
    }
}

void test_composite_layout_and_shapes() {
    policy::MlpActorCritic model(options(4, 5));
    const auto parameters = model->named_parameters(true);
    require(parameters.contains("actor.net.lins.0.weight"),
            "missing Python-compatible actor state name");
    require(parameters.contains("critic.net.lins.6.bias"),
            "missing Python-compatible critic state name");

    const auto input = torch::zeros({2, 5, 4});
    const auto logits = model->act(input);
    const auto values = model->evaluate(input);
    require(logits.sizes().vec() == std::vector<std::int64_t>({2, 5}),
            "actor rank-three output shape mismatch");
    require(values.sizes().vec() == std::vector<std::int64_t>({2, 1}),
            "critic rank-three output shape mismatch");
    require(torch::equal(logits, model->actor->forward(input)),
            "composite act did not delegate directly to actor");
    require(torch::equal(values, model->critic->forward(input)),
            "composite evaluate did not delegate directly to critic");
}

void test_action_dim_is_compatibility_only() {
    torch::manual_seed(13);
    policy::MlpActorCritic one(options(4, 1));
    torch::manual_seed(13);
    policy::MlpActorCritic many(options(4, 997));
    require_same_state(one, many);

    const auto input = torch::arange(0, 40, torch::kFloat32).view({2, 5, 4});
    require(torch::equal(one->act(input), many->act(input)),
            "action_dim changed actor output");
    require(torch::equal(one->evaluate(input), many->evaluate(input)),
            "action_dim changed critic output");
}

void test_individual_modules_and_batch_norm() {
    policy::Actor actor(options(3, 8, 3, 5, true));
    policy::Critic critic(options(3, 8, 3, 5, true));
    require(actor->named_buffers(true).contains("net.lins.1.running_mean"),
            "actor BatchNorm state name mismatch");
    require(critic->named_buffers(true).contains("net.lins.4.running_var"),
            "critic BatchNorm state name mismatch");

    actor->eval();
    critic->eval();
    const auto input = torch::ones({8, 3});
    require(actor->forward(input).sizes().vec() ==
                std::vector<std::int64_t>({8}),
            "rank-two actor output shape mismatch");
    require(critic->forward(input).sizes().vec() ==
                std::vector<std::int64_t>({8}),
            "rank-two critic output shape mismatch");
}

void test_validation() {
    bool bad_feature = false;
    try {
        policy::MlpActorCritic invalid(options(0, 3));
    } catch (const std::invalid_argument&) {
        bad_feature = true;
    }
    require(bad_feature, "non-positive feature_dim was accepted");

    bool bad_action = false;
    try {
        policy::MlpActorCritic invalid(options(4, 0));
    } catch (const std::invalid_argument&) {
        bad_action = true;
    }
    require(bad_action, "non-positive action_dim was accepted");
}

} // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_composite_layout_and_shapes();
        test_action_dim_is_compatibility_only();
        test_individual_modules_and_batch_norm();
        test_validation();
        std::cout << "mlp_policy_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_policy_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
