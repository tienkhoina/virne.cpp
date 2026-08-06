#include "../../virne/solver/learning/neural_network/neural_tensor.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor input(
    const std::int64_t batch,
    const std::int64_t features,
    const double offset = 0.0) {
    return torch::arange(batch * features, torch::kFloat32)
        .reshape({batch, features})
        .remainder(19.0)
        .sub(9.0)
        .div(7.0)
        .add(offset);
}

void test_shape_and_parameters() {
    torch::manual_seed(17);
    nn::NeuralTensorNetwork network(nn::NeuralTensorNetworkOptions{3, 2});
    const auto result = network->forward(input(4, 3), input(4, 3, 0.25));
    require(result.sizes().vec() == std::vector<std::int64_t>({4, 2}),
            "neural tensor output shape mismatch");
    require(network->weight_matrix.sizes().vec()
                == std::vector<std::int64_t>({3, 3, 2}),
            "tensor parameter shape mismatch");
    require(network->weight_matrix_block.sizes().vec()
                == std::vector<std::int64_t>({2, 6}),
            "block parameter shape mismatch");
    require(network->bias.sizes().vec() == std::vector<std::int64_t>({2, 1}),
            "bias shape mismatch");
    require(torch::all(result >= 0).item<bool>(),
            "neural tensor output did not apply ReLU");
}

void test_validation_and_device() {
    bool rejected = false;
    try {
        nn::NeuralTensorNetwork invalid(nn::NeuralTensorNetworkOptions{0, 2});
        static_cast<void>(invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "invalid neural tensor dimensions accepted");

    nn::NeuralTensorNetwork network(nn::NeuralTensorNetworkOptions{4, 3});
    rejected = false;
    try {
        static_cast<void>(network->forward(input(2, 4), input(3, 4)));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "different embedding batch sizes accepted");

    if (torch::cuda::is_available()) {
        network->to(torch::Device(torch::kCUDA, 0));
        const auto value = input(2, 4).to(torch::kCUDA);
        require(network->forward(value, value).device().is_cuda(),
                "CUDA neural tensor returned a CPU tensor");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_shape_and_parameters();
        test_validation_and_device();
        std::cout << "neural_tensor_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neural_tensor_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
