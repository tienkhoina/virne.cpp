#include "../../virne/solver/learning/pyg_lite/message_passing.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pyg = virne::solver::learning::pyg_lite;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    const std::string& message) {
    require(actual.sizes() == expected.sizes(), message + " (shape)");
    require(
        torch::allclose(actual, expected, 1e-5, 1e-6),
        message + " (values)");
}

torch::TensorOptions indices() {
    return torch::TensorOptions().dtype(torch::kInt64);
}

void test_generic_propagation() {
    const torch::Tensor x = torch::tensor(
        {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    const torch::Tensor edge_index = torch::tensor(
        {{0, 2, 1}, {1, 1, 2}}, indices());
    pyg::PropagationOptions mean_options{};
    mean_options.aggregation = pyg::ScatterReduce::mean;
    mean_options.target_num_nodes = 3;
    const torch::Tensor mean = pyg::propagate_homogeneous(
        x,
        edge_index,
        std::nullopt,
        mean_options,
        [](const torch::Tensor&, const torch::Tensor& x_j,
           const std::optional<torch::Tensor>&) { return x_j; });
    require_close(
        mean,
        torch::tensor({{0.0, 0.0}, {3.0, 4.0}, {3.0, 4.0}}),
        "mean propagation");

    const torch::Tensor edge_weight = torch::tensor({2.0, 1.0, 3.0});
    const torch::Tensor weighted = pyg::propagate_homogeneous(
        x,
        edge_index,
        edge_weight,
        mean_options,
        [](const torch::Tensor&, const torch::Tensor& x_j,
           const std::optional<torch::Tensor>& edge) {
            return edge->view({-1, 1}) * x_j;
        });
    require_close(
        weighted,
        torch::tensor({{0.0, 0.0}, {3.5, 5.0}, {9.0, 12.0}}),
        "weighted propagation");
}

void test_graph_vine() {
    pyg::GraphViNEConvOptions options{};
    options.in_channels = 2;
    options.out_channels = 2;
    options.bias = false;
    pyg::GraphViNEConv layer(options);
    {
        torch::NoGradGuard guard;
        layer->lin_rel->weight.copy_(torch::eye(2));
        layer->lin_root->weight.copy_(torch::eye(2));
    }
    const torch::Tensor x = torch::tensor(
        {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    const torch::Tensor edge_index = torch::tensor(
        {{0, 2, 1}, {1, 1, 2}}, indices());
    const torch::Tensor output = layer->forward(
        x,
        edge_index,
        torch::tensor({2.0, 1.0, 3.0}));
    require_close(
        output,
        torch::tensor({
            {1.0, 2.0},
            {6.5, 9.0},
            {14.0, 18.0}}),
        "GraphViNEConv output");
}

void test_edge_fusion() {
    pyg::EdgeFusionGATConvOptions options{};
    options.in_channels = 2;
    options.out_channels = 2;
    options.heads = 1;
    options.add_self_loops = false;
    options.bias = true;
    pyg::EdgeFusionGATConv layer(options);
    torch::Tensor x = torch::randn({3, 2});
    x.requires_grad_(true);
    const torch::Tensor edge_index = torch::tensor(
        {{0, 2, 1}, {1, 1, 2}}, indices());
    const torch::Tensor output = layer->forward(x, edge_index);
    require(output.sizes() == torch::IntArrayRef({3, 2}),
            "EdgeFusionGATConv output shape");
    output.square().sum().backward();
    require(x.grad().defined(), "EdgeFusionGATConv input gradient");

    pyg::EdgeFusionGATConvOptions edge_options = options;
    edge_options.edge_dim = 1;
    edge_options.add_self_loops = true;
    pyg::EdgeFusionGATConv edge_layer(edge_options);
    const torch::Tensor edge_attr = torch::tensor({{1.0}, {2.0}, {3.0}});
    const torch::Tensor edge_output = edge_layer->forward(
        x.detach(), edge_index, edge_attr);
    require(edge_output.sizes() == torch::IntArrayRef({3, 2}),
            "EdgeFusionGATConv edge output shape");
}

}  // namespace

int main() {
    try {
        torch::manual_seed(0);
        test_generic_propagation();
        test_graph_vine();
        test_edge_fusion();
        std::cout << "message_passing_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "message_passing_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
