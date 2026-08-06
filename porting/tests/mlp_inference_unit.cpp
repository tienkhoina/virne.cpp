#include "../../virne/solver/learning/rl_core/mlp_inference.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace core = virne::solver::learning::rl_core;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

core::PolicyConfiguration configuration() {
    core::PolicyConfiguration result;
    result.feature_dimensions.p_net_x_dim = 3;
    result.feature_dimensions.v_node_x_dim = 2;
    result.feature_dimensions.p_net_num_nodes = 5;
    result.neural_network.embedding_dim = 8;
    result.neural_network.dropout_prob = 0.0;
    result.neural_network.batch_norm = false;
    return result;
}

core::DenseObservationBatch batch_observation() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto p_net_x = torch::arange(2 * 5 * 3, options)
        .reshape({2, 5, 3})
        .remainder(19.0)
        .sub(9.0)
        .div(8.0);
    auto v_node_x = torch::arange(2 * 2, options)
        .remainder(11.0)
        .sub(5.0)
        .div(4.0)
        .reshape({2, 2});
    auto action_mask = torch::remainder(
        torch::arange(2 * 5, options), 3.0).ne(0).reshape({2, 5});
    core::DenseObservationBatch result;
    result.p_net_x = std::move(p_net_x);
    result.v_node_x = std::move(v_node_x);
    result.curr_v_node_id = torch::tensor({1, 2}, torch::kInt64);
    result.action_mask = std::move(action_mask);
    result.v_net_size = torch::tensor({4.0, 5.0}, options);
    return result;
}

void test_forward_and_conversion() {
    torch::manual_seed(31);
    core::MlpInferenceSession session(
        configuration(), {}, torch::Device(torch::kCPU));
    const auto observation = batch_observation();
    const auto prepared = core::TensorConvertor::obs_as_tensor_for_mlp(
        observation, torch::Device(torch::kCPU));
    const auto output = session.forward(observation);
    require(output.action_logits.sizes().vec()
                == std::vector<std::int64_t>({2, 5}),
            "inference actor output shape mismatch");
    require(output.values.sizes().vec()
                == std::vector<std::int64_t>({2, 1}),
            "inference critic output shape mismatch");
    require(!output.action_logits.requires_grad()
                && !output.values.requires_grad(),
            "inference output must be detached");
    const auto direct_policy = session.policy()->act(prepared.p_net_x);
    require(torch::equal(output.action_logits, direct_policy),
            "session actor path differs from prepared policy path");
}

void test_action_and_evaluation() {
    torch::manual_seed(37);
    core::PolicyActionOptions options;
    options.softmax_temperature = 0.75;
    options.mask_actions = true;
    options.maskable_policy = true;
    core::MlpInferenceSession session(
        configuration(), options, torch::Device(torch::kCPU));
    const auto observation = batch_observation();
    const auto prepared = core::TensorConvertor::obs_as_tensor_for_mlp(
        observation, torch::Device(torch::kCPU));
    const auto distribution = session.get_action_prob_dist(observation);
    require(distribution.probabilities.sizes().vec()
                == std::vector<std::int64_t>({2, 5}),
            "probability shape mismatch");
    require(torch::allclose(
                distribution.probabilities.sum(-1),
                torch::ones({2}),
                1e-6,
                1e-6),
            "probabilities must normalize per batch");

    const auto selection = session.select_action(
        observation, core::PolicyActionMode::greedy);
    const auto expected_actions = distribution.candidate_logits.argmax(-1);
    require(torch::equal(selection.actions, expected_actions),
            "greedy action differs from candidate argmax");
    const auto evaluation = session.evaluate_actions(
        observation, selection.actions, true);
    require(evaluation.values.has_value(), "evaluation must return values");
    require(evaluation.log_probabilities.sizes().vec()
                == std::vector<std::int64_t>({2}),
            "evaluation log-probability shape mismatch");
    require(evaluation.entropy.sizes().vec()
                == std::vector<std::int64_t>({2}),
            "evaluation entropy shape mismatch");
    require(evaluation.masked_action_probability.has_value(),
            "evaluation must expose masked probability when requested");
    require(prepared.general.action_mask.has_value(),
            "prepared action mask unexpectedly missing");
}

void test_single_scalar_value() {
    torch::manual_seed(41);
    core::MlpInferenceSession session(
        configuration(), {}, torch::Device(torch::kCPU));
    core::DenseObservation observation;
    observation.p_net_x = batch_observation().p_net_x[0];
    observation.v_node_x = batch_observation().v_node_x[0];
    require(observation.p_net_x.dim() == 2, "single p_net shape mismatch");
    require(observation.v_node_x.dim() == 1, "single v_node shape mismatch");
    const auto value_tensor = session.values(observation);
    require(value_tensor.numel() == 1, "single value must contain one item");
    require(session.estimate_value(observation) == value_tensor.item<double>(),
            "scalar estimate differs from value tensor");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_forward_and_conversion();
        test_action_and_evaluation();
        test_single_scalar_value();
        std::cout << "mlp_inference_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_inference_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
