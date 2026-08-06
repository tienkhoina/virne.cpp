#include "../../virne/solver/learning/neural_network/attention.h"

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
    const std::int64_t length,
    const std::int64_t features,
    const double offset = 0.0) {
    return torch::arange(batch * length * features, torch::kFloat32)
        .reshape({batch, length, features})
        .remainder(31.0)
        .sub(15.0)
        .div(8.0)
        .add(offset);
}

void test_shapes_defaults_and_mask() {
    torch::manual_seed(17);
    nn::MultiHeadAttentionWithOutput attention(
        nn::MultiHeadAttentionWithOutputOptions{4, 2, 3, 2, 0.0});
    attention->eval();
    const auto query = input(2, 3, 4);
    const auto key = input(2, 4, 4, 0.25);
    const auto value = input(2, 4, 4, -0.5);
    auto mask = torch::ones({2, 1, 3, 4}, torch::kFloat32);
    mask.index_put_({0, 0, 1, 2}, 0.0);
    const auto result = attention->forward(query, key, value, mask);
    require(result.output.sizes().vec()
                == std::vector<std::int64_t>({2, 3, 4}),
            "attention output shape mismatch");
    require(result.weights.sizes().vec()
                == std::vector<std::int64_t>({2, 2, 3, 4}),
            "attention weight shape mismatch");
    require(result.weights.index({0, 0, 1, 2}).item<float>() == 0.0F,
            "masked attention entry was not zero");
    require(attention->W_q->weight.sizes().vec()
                == std::vector<std::int64_t>({6, 4}),
            "query projection shape mismatch");
    require(attention->W_k->weight.sizes().vec()
                == std::vector<std::int64_t>({6, 4}),
            "key projection shape mismatch");
    require(attention->W_v->weight.sizes().vec()
                == std::vector<std::int64_t>({4, 4}),
            "value projection shape mismatch");
    require(attention->fc->weight.sizes().vec()
                == std::vector<std::int64_t>({4, 4}),
            "output projection shape mismatch");
}

void test_defaults_and_validation() {
    torch::manual_seed(19);
    nn::MultiHeadAttentionWithOutput attention(
        nn::MultiHeadAttentionWithOutputOptions{3, 1, 0, 0, 0.1});
    attention->eval();
    const auto result = attention->forward(input(1, 2, 3), input(1, 2, 3),
                                           input(1, 2, 3));
    require(result.output.sizes().vec()
                == std::vector<std::int64_t>({1, 2, 3}),
            "default dimension output shape mismatch");
    require(result.weights.sizes().vec()
                == std::vector<std::int64_t>({1, 1, 2, 2}),
            "default dimension weight shape mismatch");

    bool rejected = false;
    try {
        nn::MultiHeadAttentionWithOutput invalid(
            nn::MultiHeadAttentionWithOutputOptions{3, 1, -1, 0, 0.1});
        static_cast<void>(invalid);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "negative key dimension accepted");
}

void test_device() {
    nn::MultiHeadAttentionWithOutput attention(
        nn::MultiHeadAttentionWithOutputOptions{4, 2, 2, 2, 0.1});
    attention->eval();
    const auto query = input(1, 2, 4);
    if (torch::cuda::is_available()) {
        attention->to(torch::Device(torch::kCUDA, 0));
        const auto result = attention->forward(
            query.to(torch::kCUDA), query.to(torch::kCUDA),
            query.to(torch::kCUDA));
        require(result.output.device().is_cuda(),
                "CUDA attention output stayed on CPU");
        require(result.weights.device().is_cuda(),
                "CUDA attention weights stayed on CPU");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_shapes_defaults_and_mask();
        test_defaults_and_validation();
        test_device();
        std::cout << "multi_head_attention_with_output_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multi_head_attention_with_output_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
