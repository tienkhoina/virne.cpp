#include "../../virne/solver/learning/rl_core/observation_sampler.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace rl = virne::solver::learning::rl_core;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message) {
    require(actual.scalar_type() == expected.scalar_type(), message + " dtype");
    require(actual.sizes() == expected.sizes(), message + " shape");
    require(torch::equal(actual.to(torch::kCPU), expected.to(torch::kCPU)),
            message + " values");
}

torch::Tensor indices() {
    return torch::tensor({3, 1, 3, -4}, torch::kInt64);
}

void test_tensor_sampling() {
    const auto source = torch::arange(8, torch::kFloat32).reshape({4, 2});
    const auto actual = rl::sample_observations(source, indices());
    const auto expected = torch::tensor(
        {{6.0F, 7.0F}, {2.0F, 3.0F}, {6.0F, 7.0F}, {0.0F, 1.0F}});
    require_equal(actual, expected, "plain tensor sample");
}

rl::DenseObservationBatch dense_fixture() {
    rl::DenseObservationBatch result;
    result.p_net_x = torch::arange(24, torch::kFloat32).reshape({4, 3, 2});
    result.v_node_x = torch::arange(12, torch::kFloat64).reshape({4, 3});
    result.curr_v_node_id = torch::tensor({10, 11, 12, 13}, torch::kInt64);
    result.action_mask = torch::tensor(
        {{1, 0}, {0, 1}, {1, 1}, {0, 0}}, torch::kBool);
    result.v_net_size = torch::tensor({2.0F, 3.0F, 4.0F, 5.0F});
    return result;
}

void test_dense_sampling() {
    const auto source = dense_fixture();
    const auto selected = torch::tensor({2, 0}, torch::kInt64);
    const auto actual = rl::sample_observations(source, selected);
    require_equal(actual.p_net_x, source.p_net_x.index({selected}),
                  "dense p_net_x");
    require_equal(actual.v_node_x, source.v_node_x.index({selected}),
                  "dense v_node_x");
    require_equal(*actual.curr_v_node_id,
                  source.curr_v_node_id->index({selected}), "dense curr id");
    require_equal(*actual.action_mask,
                  source.action_mask->index({selected}), "dense mask");
    require_equal(*actual.v_net_size,
                  source.v_net_size->index({selected}), "dense vnet size");
}

void test_typed_policy_sampling() {
    const auto selected = torch::tensor({1, 3}, torch::kInt64);
    rl::MlpTensorObservation mlp;
    mlp.p_net_x = torch::arange(24, torch::kFloat32).reshape({4, 3, 2});
    mlp.general.curr_v_node_id = torch::tensor({0, 1, 2, 3}, torch::kInt64);
    mlp.general.action_mask = torch::ones({4, 3}, torch::kBool);
    const auto sampled_mlp = rl::sample_observations(mlp, selected);
    require_equal(sampled_mlp.p_net_x, mlp.p_net_x.index({selected}),
                  "mlp p_net_x");
    require_equal(*sampled_mlp.general.curr_v_node_id,
                  mlp.general.curr_v_node_id->index({selected}), "mlp curr id");
    require(!sampled_mlp.general.v_net_size.has_value(),
            "mlp absent field preserved");

    rl::MlpFcTensorObservation fc;
    fc.pv_nodes_x = torch::arange(40, torch::kFloat32).reshape({4, 10});
    fc.general.v_net_size = torch::tensor({4.0F, 5.0F, 6.0F, 7.0F});
    const auto sampled_fc = rl::sample_observations(fc, selected);
    require_equal(sampled_fc.pv_nodes_x, fc.pv_nodes_x.index({selected}),
                  "mlp-fc input");

    rl::AttentionTensorObservation attention;
    attention.p_net_x = mlp.p_net_x;
    attention.v_node_x = torch::arange(12, torch::kFloat32).reshape({4, 3});
    const auto sampled_attention =
        rl::sample_observations(attention, selected);
    require_equal(sampled_attention.v_node_x,
                  attention.v_node_x.index({selected}), "attention v_node_x");

    const rl::GeneralTensorObservation empty;
    const auto sampled_empty = rl::sample_observations(empty, selected);
    require(!sampled_empty.curr_v_node_id.has_value()
                && !sampled_empty.action_mask.has_value()
                && !sampled_empty.v_net_size.has_value(),
            "empty general observation");
}

pyg::Data graph(
    const std::initializer_list<float> nodes,
    const std::initializer_list<std::int64_t> edges) {
    pyg::Data result;
    result.x = torch::tensor(std::vector<float>(nodes)).reshape(
        {static_cast<std::int64_t>(nodes.size()), 1});
    result.edge_index = torch::tensor(
        std::vector<std::int64_t>(edges), torch::kInt64).reshape({2, -1});
    result.validate();
    return result;
}

void test_pyg_batch_sampling() {
    auto first = graph({1.0F, 2.0F}, {0, 1, 1, 0});
    first.edge_attr = torch::tensor({0.25F, 0.5F}).reshape({2, 1});
    auto second = graph({5.0F, 6.0F, 7.0F}, {0, 1, 1, 2});
    second.edge_attr = torch::tensor({0.75F, 1.0F}).reshape({2, 1});
    const auto batch = pyg::Batch::from_data_list({first, second});
    const auto selected = torch::tensor({1, 0, 1}, torch::kInt64);
    const auto actual = rl::sample_observations(batch, selected);
    const auto expected = pyg::Batch::from_data_list({second, first, second});
    require(actual.num_graphs == 3, "sampled graph count");
    require_equal(actual.x, expected.x, "sampled graph x");
    require_equal(actual.edge_index, expected.edge_index,
                  "sampled graph edge_index");
    require_equal(*actual.edge_attr, *expected.edge_attr,
                  "sampled graph edge_attr");
    require_equal(*actual.batch, *expected.batch, "sampled graph batch");
    require_equal(*actual.ptr, *expected.ptr, "sampled graph ptr");
    require_equal(actual.edge_ptr, expected.edge_ptr, "sampled graph edge_ptr");
    require(actual.max_num_nodes == expected.max_num_nodes,
            "sampled graph max_num_nodes");
}

void test_errors() {
    const auto source = torch::ones({2, 2});
    bool bad_indices = false;
    try {
        static_cast<void>(rl::sample_observations(
            source, torch::tensor({0.0F, 1.0F})));
    } catch (const rl::ObservationSampleException& error) {
        bad_indices = error.code()
            == rl::ObservationSampleErrorCode::invalid_indices;
    }
    require(bad_indices, "float indices rejected");

    bool bad_rank = false;
    try {
        static_cast<void>(rl::sample_observations(
            torch::tensor(1.0F), torch::tensor({0}, torch::kInt64)));
    } catch (const rl::ObservationSampleException& error) {
        bad_rank = error.code()
            == rl::ObservationSampleErrorCode::invalid_tensor_rank;
    }
    require(bad_rank, "scalar source rejected");

    auto malformed = dense_fixture();
    malformed.v_node_x = torch::ones({3, 2});
    bool bad_batch_size = false;
    try {
        static_cast<void>(rl::sample_observations(
            malformed, torch::tensor({-1}, torch::kInt64)));
    } catch (const rl::ObservationSampleException& error) {
        bad_batch_size = error.code()
            == rl::ObservationSampleErrorCode::batch_size_mismatch;
    }
    require(bad_batch_size, "mixed fixed-field batch sizes rejected");
}

void test_cuda_if_available() {
    if (!torch::cuda::is_available()) {
        return;
    }
    const torch::Device cuda(torch::kCUDA);
    const auto source = torch::arange(
        12, torch::TensorOptions().dtype(torch::kFloat32).device(cuda))
                            .reshape({4, 3});
    const auto selected = torch::tensor({3, 0}, torch::kInt64);
    const auto actual = rl::sample_observations(source, selected, cuda);
    require(actual.device() == cuda, "CUDA output device");
    require_equal(actual, source.index({selected.to(cuda)}), "CUDA values");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        test_tensor_sampling();
        test_dense_sampling();
        test_typed_policy_sampling();
        test_pyg_batch_sampling();
        test_errors();
        test_cuda_if_available();
        std::cout << "observation_sampler_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "observation_sampler_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
