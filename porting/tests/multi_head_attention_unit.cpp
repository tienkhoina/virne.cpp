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

torch::Tensor input(const std::int64_t batch, const std::int64_t length,
                    const std::int64_t features) {
    return torch::arange(batch * length * features, torch::kFloat32)
        .reshape({batch, length, features})
        .remainder(47.0)
        .sub(23.0)
        .div(16.0);
}

void test_shapes_and_defaults() {
    torch::manual_seed(7);
    nn::MultiHeadSelfAttention attention(nn::MultiHeadSelfAttentionOptions{
        4, 8, 0, 0, 2});
    const auto query = input(2, 3, 4);
    const auto key = input(2, 5, 4);
    require(attention->forward(query).sizes().vec()
                == std::vector<std::int64_t>({2, 3, 3}),
            "self-attention shape mismatch");
    require(attention->forward(query, key).sizes().vec()
                == std::vector<std::int64_t>({2, 3, 5}),
            "cross-attention shape mismatch");
    require(attention->W_query.sizes().vec()
                == std::vector<std::int64_t>({2, 4, 4}),
            "query parameter shape mismatch");
    require(attention->W_key.sizes().vec()
                == std::vector<std::int64_t>({2, 4, 4}),
            "key parameter shape mismatch");
}

void test_validation_and_device() {
    bool rejected = false;
    try {
        nn::MultiHeadSelfAttention attention(nn::MultiHeadSelfAttentionOptions{
            4, 7, 0, 0, 2});
        static_cast<void>(attention);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "non-divisible embed_dim accepted");

    nn::MultiHeadSelfAttention attention(nn::MultiHeadSelfAttentionOptions{
        4, 8, 0, 0, 2});
    if (torch::cuda::is_available()) {
        attention->to(torch::Device(torch::kCUDA, 0));
        require(attention->forward(input(2, 3, 4).to(torch::kCUDA)).device().is_cuda(),
                "CUDA multi-head attention returned a CPU tensor");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_shapes_and_defaults();
        test_validation_and_device();
        std::cout << "multi_head_attention_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multi_head_attention_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
