#include "../../virne/solver/learning/neural_network/gnn.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = virne::solver::learning::neural_network;
namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor node_input(const std::int64_t count, const std::int64_t features) {
    return torch::arange(count * features, torch::kFloat32)
        .reshape({count, features})
        .remainder(29.0)
        .sub(14.0)
        .div(9.0);
}

pyg::Batch make_batch() {
    pyg::Data first;
    first.x = node_input(3, 3);
    first.edge_index = torch::tensor(
        {{0, 1, 1, 2}, {1, 0, 2, 1}}, torch::kInt64);
    pyg::Data second;
    second.x = node_input(5, 3).add(0.25);
    second.edge_index = torch::tensor(
        {{0, 1, 2, 3, 3, 4}, {1, 0, 3, 2, 4, 3}}, torch::kInt64);
    return pyg::Batch::from_data_list({first, second});
}

void test_layers_and_state() {
    torch::manual_seed(31);
    const auto batch = make_batch();
    nn::GCNConvNet network(nn::GCNConvNetOptions{
        3, 5, 6, 3, true, 1.0, true, nn::GraphPoolingKind::none});
    network->eval();
    const auto output = network->forward(batch);
    require(output.sizes().vec()
                == std::vector<std::int64_t>({2, 5, 5}),
            "dense GCN output shape mismatch");
    require(network->named_parameters(true).contains("conv_0.lin.weight"),
            "conv state name mismatch");
    require(network->named_buffers(true).contains("norm_0.running_mean"),
            "batch norm state name mismatch");

    nn::GCNConvNet one_layer(nn::GCNConvNetOptions{
        3, 4, 6, 1, false, 1.0, false, nn::GraphPoolingKind::none});
    require(one_layer->forward(batch).sizes().vec()
                == std::vector<std::int64_t>({8, 4}),
            "one-layer sparse GCN output shape mismatch");
}

void test_pooling_and_validation() {
    const auto batch = make_batch();
    for (const auto kind : {nn::GraphPoolingKind::add,
                            nn::GraphPoolingKind::mean,
                            nn::GraphPoolingKind::max}) {
        nn::GCNConvNet network(nn::GCNConvNetOptions{
            3, 4, 6, 2, false, 1.0, false, kind});
        const auto output = network->forward(batch);
        require(output.sizes().vec()
                    == std::vector<std::int64_t>({2, 4}),
                "graph pooling output shape mismatch");
    }

    bool rejected = false;
    try {
        nn::GCNConvNet invalid(nn::GCNConvNetOptions{
            3, 4, 6, 0, false, 1.0, false, nn::GraphPoolingKind::none});
        static_cast<void>(invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "zero-layer GCN accepted");
    require(nn::graph_pooling_kind_from_name("sum")
                == nn::GraphPoolingKind::add,
            "sum pooling name did not resolve to add");
}

void test_device() {
    if (torch::cuda::is_available()) {
        auto batch = make_batch().to(torch::Device(torch::kCUDA, 0));
        nn::GCNConvNet network(nn::GCNConvNetOptions{
            3, 4, 6, 2, false, 1.0, false, nn::GraphPoolingKind::mean});
        network->to(torch::Device(torch::kCUDA, 0));
        require(network->forward(batch).device().is_cuda(),
                "CUDA GCN returned a CPU tensor");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_layers_and_state();
        test_pooling_and_validation();
        test_device();
        std::cout << "gcn_conv_net_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gcn_conv_net_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
