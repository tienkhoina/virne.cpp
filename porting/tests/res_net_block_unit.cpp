#include "../../virne/solver/learning/neural_network/res.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor input(const std::int64_t batch, const std::int64_t channels,
                    const std::int64_t height, const std::int64_t width) {
    return torch::arange(batch * channels * height * width, torch::kFloat32)
        .reshape({batch, channels, height, width})
        .remainder(37.0)
        .sub(18.0)
        .div(11.0);
}

void test_shape_and_residual() {
    torch::manual_seed(31);
    nn::ResNetBlock block(nn::ResNetBlockOptions{3});
    const auto value = input(2, 3, 4, 5);
    const auto output = block->forward(value);
    require(output.sizes().vec()
                == std::vector<std::int64_t>({2, 3, 4, 5}),
            "ResNet block shape mismatch");
    const auto expected = torch::leaky_relu(
        block->conv_2->forward(torch::leaky_relu(
            block->conv_1->forward(value), 0.01)), 0.01) + value;
    require(torch::equal(output, expected),
            "ResNet block residual formula mismatch");
    require(block->named_parameters(true).contains("conv_1.weight"),
            "first convolution state name mismatch");
    require(block->named_parameters(true).contains("conv_2.bias"),
            "second convolution state name mismatch");
}

void test_validation_and_device() {
    bool rejected = false;
    try {
        nn::ResNetBlock invalid(nn::ResNetBlockOptions{0});
        static_cast<void>(invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "zero-channel ResNet block accepted");

    nn::ResNetBlock block(nn::ResNetBlockOptions{2});
    rejected = false;
    try {
        static_cast<void>(block->forward(input(1, 3, 2, 2)));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "wrong-channel ResNet input accepted");

    if (torch::cuda::is_available()) {
        block->to(torch::Device(torch::kCUDA, 0));
        require(block->forward(input(1, 2, 3, 3).to(torch::kCUDA))
                    .device().is_cuda(),
                "CUDA ResNet block returned a CPU tensor");
    }
}

void test_res_linear_python_boundary() {
    bool rejected = false;
    try {
        nn::ResLinearNet invalid(nn::ResLinearNetOptions{4, 8, 3, true});
        static_cast<void>(invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "broken Python ResLinearNet constructor was accepted");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_shape_and_residual();
        test_validation_and_device();
        test_res_linear_python_boundary();
        std::cout << "res_net_block_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "res_net_block_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
