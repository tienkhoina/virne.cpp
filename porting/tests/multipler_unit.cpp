#include "../../virne/solver/learning/neural_network/multipler.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor input(const std::int64_t batch, const std::int64_t features) {
    return torch::arange(batch * features, torch::kFloat32)
        .reshape({batch, features})
        .remainder(29.0)
        .sub(14.0)
        .div(9.0);
}

void test_formula_and_state_names() {
    torch::manual_seed(41);
    nn::MultiplerNet network(nn::MultiplerNetOptions{5});
    const auto state = input(7, 5);
    const auto output = network->forward(state);
    const auto expected = torch::softplus(
        network->l3->forward(torch::leaky_relu(
            network->l2->forward(torch::leaky_relu(
                network->l1->forward(state), 0.01)), 0.01)),
        1.0,
        20.0);

    require(output.sizes().vec() == std::vector<std::int64_t>({7, 1}),
            "MultiplerNet output shape mismatch");
    require(torch::equal(output, expected),
            "MultiplerNet formula mismatch");
    require(torch::all(output > 0).item<bool>(),
            "MultiplerNet output is not strictly positive");
    const auto parameters = network->named_parameters(true);
    require(parameters.contains("l1.weight")
                && parameters.contains("l1.bias")
                && parameters.contains("l2.weight")
                && parameters.contains("l2.bias")
                && parameters.contains("l3.weight")
                && parameters.contains("l3.bias"),
            "MultiplerNet state names differ from Python");
}

void test_validation_rank_and_device() {
    bool rejected = false;
    try {
        nn::MultiplerNet invalid(nn::MultiplerNetOptions{0});
        static_cast<void>(invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "MultiplerNet accepted zero state_dim");

    nn::MultiplerNet network(nn::MultiplerNetOptions{3});
    rejected = false;
    try {
        static_cast<void>(network->forward(input(2, 4)));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "MultiplerNet accepted wrong final dimension");

    const auto batched = input(12, 3).reshape({2, 2, 3, 3});
    require(network->forward(batched).sizes().vec()
                == std::vector<std::int64_t>({2, 2, 3, 1}),
            "MultiplerNet did not preserve leading dimensions");

    if (torch::cuda::is_available()) {
        network->to(torch::Device(torch::kCUDA, 0));
        require(network->forward(input(4, 3).to(torch::kCUDA))
                    .device().is_cuda(),
                "CUDA MultiplerNet returned a CPU tensor");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_formula_and_state_names();
        test_validation_rank_and_device();
        std::cout << "multipler_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multipler_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
