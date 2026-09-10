#include "../../virne/solver/learning/neural_network/gnn.h"
#include "../../virne/solver/learning/neural_network/loss.h"
#include "../../virne/solver/learning/neural_network/multipler.h"
#include "../../virne/solver/learning/neural_network/res.h"
#include "../../virne/solver/learning/neural_network/sinkhorn.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace nn = virne::solver::learning::neural_network;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor values(
    const std::vector<std::int64_t>& shape,
    const double offset = 0.0) {
    std::int64_t count = 1;
    for (const auto extent : shape) count *= extent;
    return torch::arange(count, torch::kFloat32)
        .reshape(shape)
        .remainder(29.0)
        .sub(14.0)
        .div(9.0)
        .add(offset);
}

pyg::Batch graph_batch() {
    pyg::Data first;
    first.x = values({3, 3});
    first.edge_index = torch::tensor(
        {{0, 1, 1, 2}, {1, 0, 2, 1}}, torch::kInt64);
    first.edge_attr = values({4, 2}, 0.1);
    pyg::Data second;
    second.x = values({4, 3}, 0.25);
    second.edge_index = torch::tensor(
        {{0, 1, 2, 2, 3}, {1, 0, 1, 3, 2}}, torch::kInt64);
    second.edge_attr = values({5, 2}, -0.2);
    return pyg::Batch::from_data_list({first, second});
}

void fill_parameters(torch::nn::Module& module, const double scale = 0.01) {
    torch::NoGradGuard guard;
    std::int64_t parameter_id = 1;
    std::unordered_set<const c10::TensorImpl*> seen;
    for (auto& item : module.named_parameters(true)) {
        if (!seen.insert(item.value().unsafeGetTensorImpl()).second) continue;
        item.value().copy_(
            torch::arange(
                item.value().numel(),
                item.value().options())
                .reshape(item.value().sizes())
                .add(static_cast<double>(parameter_id))
                .mul(scale));
        ++parameter_id;
    }
}

void test_multiplier_loss_and_residual() {
    torch::manual_seed(41);
    nn::MultiplerNet multiplier(nn::MultiplerNetOptions{5});
    const auto state = values({7, 5});
    const auto multiplier_output = multiplier->forward(state);
    const auto multiplier_expected = torch::softplus(
        multiplier->l3->forward(torch::leaky_relu(
            multiplier->l2->forward(torch::leaky_relu(
                multiplier->l1->forward(state), 0.01)), 0.01)),
        1.0,
        20.0);
    require(torch::equal(multiplier_output, multiplier_expected),
            "MultiplerNet formula mismatch");
    require(multiplier->named_parameters(true).contains("l3.bias"),
            "MultiplerNet state name mismatch");

    auto h1 = values({8, 4}).set_requires_grad(true);
    auto h2 = values({8, 4}, 0.2).flip({0}).set_requires_grad(true);
    nn::BarlowTwinsContrastiveLoss contrastive;
    const auto loss = contrastive->forward(h1, h2);
    require(loss.dim() == 0 && torch::isfinite(loss).item<bool>(),
            "Barlow Twins output is not a finite scalar");
    loss.backward();
    require(h1.grad().defined() && h2.grad().defined(),
            "Barlow Twins autograd is disconnected");

    torch::manual_seed(43);
    nn::ResNetBlock residual(nn::ResNetBlockOptions{3});
    const auto image = values({2, 3, 4, 5});
    const auto residual_output = residual->forward(image);
    const auto residual_expected = torch::leaky_relu(
        residual->conv_2->forward(torch::leaky_relu(
            residual->conv_1->forward(image), 0.01)), 0.01) + image;
    require(torch::equal(residual_output, residual_expected),
            "ResNetBlock formula mismatch");

    bool rejected = false;
    try {
        nn::ResLinearNet broken(nn::ResLinearNetOptions{4, 8, 3, true});
        static_cast<void>(broken);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "broken Python ResLinearNet boundary was accepted");
}

void test_sinkhorn_family() {
    const auto scores = torch::tensor({
        {0.2F, -0.1F, 0.7F},
        {0.4F, 0.3F, -0.5F},
        {-0.2F, 0.8F, 0.1F},
    });
    nn::Sinkhorn simple(nn::SinkhornOptions{7, 0.7});
    const auto normalized = simple->forward(scores.unsqueeze(0));
    require(torch::allclose(
                normalized.sum(2), torch::ones({1, 3}), 2e-3, 2e-3),
            "Sinkhorn row normalization mismatch");

    nn::SinkhornNetwork sequential(
        nn::SinkhornNetworkOptions{8, 0.8, 1e-4, true, false});
    nn::SinkhornNetwork batched(
        nn::SinkhornNetworkOptions{8, 0.8, 1e-4, true, true});
    require(torch::allclose(
                sequential->forward(scores),
                batched->forward(scores),
                1e-6,
                1e-6),
            "SinkhornNetwork batch mode mismatch");

    nn::SinkhornNetworkCallOptions padded_call;
    padded_call.nrows = std::vector<std::int64_t>{3, 2};
    padded_call.ncols = std::vector<std::int64_t>{3, 2};
    const auto padded = sequential->forward(
        torch::stack({scores, scores + 0.25}), padded_call);
    require(padded.index({1, 2}).eq(0).all().item<bool>(),
            "SinkhornNetwork padded row was not masked");

    nn::SinkhornNetworkCallOptions dummy;
    dummy.dummy = true;
    require(sequential->forward(scores, dummy).index({2, 2}).item<float>()
                == 0.0F,
            "SinkhornNetwork terminal dummy edge was not forbidden");
}

void test_graph_helpers_and_wrappers() {
    require(nn::gnn_kind_from_name("deep_edge_gat")
                == nn::GnnKind::deep_edge_gat,
            "typed GNN registry mismatch");
    bool rejected = false;
    try {
        static_cast<void>(nn::gnn_kind_from_name("pna"));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "invalid Python PNA wrapper was registered");

    const auto dense = values({2, 3, 4});
    const auto mask = torch::tensor(
        {{true, false, true}, {false, true, true}}, torch::kBool);
    const auto sparse = nn::to_sparse_batch(dense, mask);
    require(sparse.sizes().vec() == std::vector<std::int64_t>({4, 4}),
            "to_sparse_batch shape mismatch");

    nn::DenseToSparse dense_to_sparse;
    const auto adjacency = torch::tensor({
        {{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}},
        {{0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
    });
    const auto converted = dense_to_sparse->forward(
        values({2, 3, 2}), adjacency, values({2, 3, 3}, 1.0));
    require(converted.x.sizes().vec()
                == std::vector<std::int64_t>({6, 2})
            && converted.edge_index.sizes().vec()
                == std::vector<std::int64_t>({2, 6})
            && converted.batch.sizes().vec()
                == std::vector<std::int64_t>({6}),
            "DenseToSparse output shape mismatch");

    const auto batch = graph_batch();
    torch::manual_seed(47);
    nn::GraphAttentionPooling attention_pool(4);
    const auto pooled = attention_pool->forward(
        values({7, 4}), *batch.batch, batch.num_graphs);
    require(pooled.sizes().vec() == std::vector<std::int64_t>({2, 4}),
            "GraphAttentionPooling shape mismatch");

    nn::GATConvNet gat(nn::GraphConvNetOptions{
        3, 4, 2, 5, 1, 2, false, 1.0, false,
        nn::GraphPoolingKind::attention});
    gat->eval();
    require(gat->forward(batch).sizes().vec()
                == std::vector<std::int64_t>({2, 4}),
            "GATConvNet attention-pool shape mismatch");

    nn::EdgeFusionGATConvNet edge_fusion(nn::GraphConvNetOptions{
        3, 4, 2, 5, 1, 2, false, 1.0, true,
        nn::GraphPoolingKind::none});
    edge_fusion->eval();
    require(edge_fusion->forward(batch).sizes().vec()
                == std::vector<std::int64_t>({2, 4, 4}),
            "EdgeFusionGATConvNet dense shape mismatch");

    nn::DeepEdgeFeatureGAT deep(nn::DeepEdgeFeatureGATOptions{
        3, 4, 2, 3, 0.2, 0.2, 5, 1, false, 1.0, false,
        nn::GraphPoolingKind::mean});
    fill_parameters(*deep);
    deep->eval();
    const auto deep_output = deep->forward(batch);
    require(deep_output.sizes().vec()
                == std::vector<std::int64_t>({2, 4})
            && torch::isfinite(deep_output).all().item<bool>(),
            "DeepEdgeFeatureGAT output mismatch");
}

void test_cuda_dispatch() {
    if (!torch::cuda::is_available()) return;
    auto batch = graph_batch().to(torch::Device(torch::kCUDA, 0));
    nn::GATConvNet gat(nn::GraphConvNetOptions{
        3, 4, 2, 5, 1, 2, false, 1.0, false,
        nn::GraphPoolingKind::mean});
    gat->to(torch::Device(torch::kCUDA, 0));
    require(gat->forward(batch).device().is_cuda(),
            "GATConvNet moved CUDA data to CPU");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_multiplier_loss_and_residual();
        test_sinkhorn_family();
        test_graph_helpers_and_wrappers();
        test_cuda_dispatch();
        std::cout << "neural_network_remaining_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neural_network_remaining_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
