#include "../virne/solver/learning/rl_core/tensor_convertor.h"
#include "../virne/solver/learning/rl_policy/mlp_policy.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace convertor = virne::solver::learning::rl_core;
namespace policy = virne::solver::learning::rl_policy;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_single_dense_paths() {
    convertor::DenseObservation input{};
    input.p_net_x = torch::tensor(
        {{1, 2}, {3, 4}}, torch::TensorOptions().dtype(torch::kInt64));
    input.v_node_x = torch::tensor(
        {0.5, -1.5}, torch::TensorOptions().dtype(torch::kFloat64));
    input.curr_v_node_id = 3;
    input.action_mask = torch::tensor(
        {1, 0}, torch::TensorOptions().dtype(torch::kInt64));
    input.v_net_size = 7.0;

    const auto mlp = convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
    const auto expected = torch::tensor({
        {{1.0F, 2.0F, 0.5F, -1.5F},
         {3.0F, 4.0F, 0.5F, -1.5F}},
    });
    require(torch::equal(mlp.p_net_x, expected),
            "single MLP concatenation mismatch");
    require(mlp.general.curr_v_node_id->scalar_type() == torch::kInt64,
            "single current-node dtype mismatch");
    require(mlp.general.curr_v_node_id->sizes().vec() ==
                std::vector<std::int64_t>({1}),
            "single current-node shape mismatch");
    require(mlp.general.action_mask->scalar_type() == torch::kFloat32,
            "single action-mask dtype mismatch");
    require(mlp.general.action_mask->sizes().vec() ==
                std::vector<std::int64_t>({1, 2}),
            "single action-mask shape mismatch");
    require(mlp.general.v_net_size->item<float>() == 7.0F,
            "single virtual-network size mismatch");

    const auto cnn = convertor::TensorConvertor::obs_as_tensor_for_cnn(input);
    require(cnn.p_net_x.sizes().vec() ==
                std::vector<std::int64_t>({1, 1, 2, 4}),
            "single CNN layout mismatch");

    const auto fc = convertor::TensorConvertor::obs_as_tensor_for_mlp_fc(input);
    require(torch::equal(
                fc.pv_nodes_x,
                torch::tensor({{1.0F, 2.0F, 3.0F, 4.0F, 0.5F, -1.5F}})),
            "single MLP-FC layout mismatch");

    const auto attention =
        convertor::TensorConvertor::obs_as_tensor_for_att(input);
    require(attention.p_net_x.sizes().vec() ==
                std::vector<std::int64_t>({1, 2, 2}),
            "single attention physical layout mismatch");
    require(attention.v_node_x.sizes().vec() ==
                std::vector<std::int64_t>({1, 2}),
            "single attention virtual-node layout mismatch");
}

void test_batched_dense_paths() {
    convertor::DenseObservationBatch input{};
    input.p_net_x = torch::arange(0, 16, torch::kFloat64).view({2, 4, 2});
    input.v_node_x = torch::tensor(
        {{10, 11, 12}, {20, 21, 22}},
        torch::TensorOptions().dtype(torch::kInt64));
    input.curr_v_node_id = torch::tensor({1.5, 2.5});
    input.action_mask = torch::ones({2, 4}, torch::kFloat64);
    input.v_net_size = torch::tensor({4, 5}, torch::kInt64);

    const auto mlp = convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
    require(mlp.p_net_x.sizes().vec() ==
                std::vector<std::int64_t>({2, 4, 5}),
            "batch MLP layout mismatch");
    require(mlp.p_net_x.scalar_type() == torch::kFloat32,
            "batch MLP dtype mismatch");
    require(mlp.p_net_x.index({1, 3, 4}).item<float>() == 22.0F,
            "batch virtual-node broadcast mismatch");
    require(mlp.general.curr_v_node_id->scalar_type() == torch::kInt64,
            "batch current-node dtype mismatch");
    require(torch::equal(
                *mlp.general.curr_v_node_id,
                torch::tensor({1, 2}, torch::kInt64)),
            "batch current-node cast mismatch");
    require(mlp.general.v_net_size->scalar_type() == torch::kFloat32,
            "batch virtual-network-size dtype mismatch");

    const auto fc = convertor::TensorConvertor::obs_as_tensor_for_mlp_fc(input);
    require(fc.pv_nodes_x.sizes().vec() ==
                std::vector<std::int64_t>({2, 11}),
            "batch MLP-FC layout mismatch");
}

void test_optional_fields_and_policy_composition() {
    convertor::DenseObservationBatch input{};
    input.p_net_x = torch::zeros({2, 5, 14});
    input.v_node_x = torch::ones({2, 11});
    const auto converted =
        convertor::TensorConvertor::obs_as_tensor_for_mlp(input);
    require(!converted.general.curr_v_node_id.has_value(),
            "missing current-node field was materialized");
    require(!converted.general.action_mask.has_value(),
            "missing action-mask field was materialized");
    require(!converted.general.v_net_size.has_value(),
            "missing virtual-network-size field was materialized");

    policy::MlpActorCriticOptions options{};
    options.feature_dim = 25;
    options.action_dim = 5;
    policy::MlpActorCritic model(options);
    const auto logits = model->act(converted.p_net_x);
    const auto values = model->evaluate(converted.p_net_x);
    require(logits.sizes().vec() == std::vector<std::int64_t>({2, 5}),
            "converted observation actor shape mismatch");
    require(values.sizes().vec() == std::vector<std::int64_t>({2, 1}),
            "converted observation critic shape mismatch");
}

void test_validation() {
    convertor::DenseObservation bad_rank{};
    bad_rank.p_net_x = torch::zeros({1, 2, 3});
    bad_rank.v_node_x = torch::zeros({2});
    bool rank_rejected = false;
    try {
        (void)convertor::TensorConvertor::obs_as_tensor_for_mlp(bad_rank);
    } catch (const std::invalid_argument&) {
        rank_rejected = true;
    }
    require(rank_rejected, "invalid single-observation rank was accepted");

    convertor::DenseObservationBatch bad_batch{};
    bad_batch.p_net_x = torch::zeros({2, 4, 3});
    bad_batch.v_node_x = torch::zeros({3, 2});
    bool batch_rejected = false;
    try {
        (void)convertor::TensorConvertor::obs_as_tensor_for_mlp(bad_batch);
    } catch (const std::invalid_argument&) {
        batch_rejected = true;
    }
    require(batch_rejected, "incompatible dense batch was accepted");
}

} // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_single_dense_paths();
        test_batched_dense_paths();
        test_optional_fields_and_policy_composition();
        test_validation();
        std::cout << "tensor_convertor_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor_convertor_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
