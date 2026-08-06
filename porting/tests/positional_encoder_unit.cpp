#include "../../virne/solver/learning/neural_network/attention.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor input(const std::int64_t batch, const std::int64_t length,
                    const std::int64_t features) {
    return torch::arange(batch * length * features, torch::kFloat32)
        .reshape({batch, length, features})
        .div(10.0)
        .sub(0.7);
}

void test_add_and_concat() {
    nn::PositionalEncoder add(nn::PositionalEncoderOptions{
        6, 1.0, 8, nn::PositionalEncodingMethod::add});
    const auto add_input = input(2, 5, 6);
    const auto add_output = add->forward(add_input);
    require(add_output.sizes().vec()
                == std::vector<std::int64_t>({2, 5, 6}),
            "add output shape mismatch");
    require(torch::equal(
                add_output,
                add_input + add->positional_buffer().narrow(1, 0, 5).repeat({2, 1, 1})),
            "add output does not use registered positional buffer");

    nn::PositionalEncoder concat(nn::PositionalEncoderOptions{
        6, 1.0, 8, nn::PositionalEncodingMethod::concat});
    const auto concat_output = concat->forward(input(2, 5, 4));
    require(concat_output.sizes().vec()
                == std::vector<std::int64_t>({2, 5, 10}),
            "concat output shape mismatch");
}

void test_eval_and_device() {
    nn::PositionalEncoder encoder(nn::PositionalEncoderOptions{
        8, 0.25, 10, nn::PositionalEncodingMethod::add});
    encoder->eval();
    const auto value = input(3, 7, 8);
    require(torch::equal(encoder->forward(value), encoder->forward(value)),
            "evaluation mode must be deterministic");
    encoder->to(torch::Device(torch::kCPU));
    require(encoder->positional_buffer().device().is_cpu(),
            "positional buffer was not placed on CPU");
    if (torch::cuda::is_available()) {
        encoder->to(torch::Device(torch::kCUDA, 0));
        const auto cuda_value = value.to(torch::kCUDA);
        require(encoder->forward(cuda_value).device().is_cuda(),
                "CUDA positional encoder returned a CPU tensor");
    }
}

template <typename Function>
void require_rejected(Function&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

void test_validation() {
    require_rejected(
        [] { nn::PositionalEncoder(nn::PositionalEncoderOptions{3}); },
        "odd embedding dimension accepted");
    require_rejected(
        [] {
            nn::PositionalEncoder encoder(nn::PositionalEncoderOptions{4, 1.0, 2});
            static_cast<void>(encoder->forward(input(1, 3, 4)));
        },
        "sequence longer than max_len accepted");
    require_rejected(
        [] {
            nn::PositionalEncoder encoder(nn::PositionalEncoderOptions{4});
            static_cast<void>(encoder->forward(input(1, 2, 3)));
        },
        "add width mismatch accepted");
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_add_and_concat();
        test_eval_and_device();
        test_validation();
        std::cout << "positional_encoder_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "positional_encoder_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
