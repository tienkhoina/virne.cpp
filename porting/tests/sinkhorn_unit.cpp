#include "../../virne/solver/learning/neural_network/sinkhorn.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

torch::Tensor scores_2d() {
    return torch::tensor({
        {0.2F, -0.1F, 0.7F},
        {0.4F, 0.3F, -0.5F},
        {-0.2F, 0.8F, 0.1F},
    });
}

void test_simple_sinkhorn() {
    nn::Sinkhorn layer(nn::SinkhornOptions{7, 0.7});
    const auto result = layer->forward(scores_2d().unsqueeze(0));
    require(result.sizes().vec()
                == std::vector<std::int64_t>({1, 3, 3}),
            "Sinkhorn output shape mismatch");
    require(torch::isfinite(result).all().item<bool>(),
            "Sinkhorn output contains non-finite values");
    require(torch::allclose(
                result.sum(2),
                torch::ones({1, 3}),
                2e-3,
                2e-3),
            "Sinkhorn row sums are not normalized");
    require(torch::allclose(
                result.sum(1),
                torch::ones({1, 3}),
                2e-3,
                2e-3),
            "Sinkhorn column sums are not normalized");

    const auto values = torch::tensor({1.0F, 2.0F, 3.0F});
    require(torch::allclose(
                nn::logsumexp(values),
                torch::log(torch::exp(values).sum())),
            "standalone logsumexp mismatch");
}

void test_network_matrix_and_batch_modes() {
    nn::SinkhornNetwork sequential(
        nn::SinkhornNetworkOptions{8, 0.8, 1e-4, true, false});
    nn::SinkhornNetwork batched(
        nn::SinkhornNetworkOptions{8, 0.8, 1e-4, true, true});
    const auto matrix = scores_2d();
    const auto sequential_result = sequential->forward(matrix);
    const auto batched_result = batched->forward(matrix);
    require(sequential_result.sizes().vec()
                == std::vector<std::int64_t>({3, 3}),
            "matrix input was not squeezed");
    require(torch::allclose(sequential_result, batched_result, 1e-6, 1e-6),
            "batched and per-item full-matrix results differ");

    auto batch = torch::stack({matrix, matrix + 0.25F});
    nn::SinkhornNetworkCallOptions call;
    call.nrows = std::vector<std::int64_t>{3, 2};
    call.ncols = std::vector<std::int64_t>{3, 2};
    const auto padded = sequential->forward(batch, call);
    require(padded.sizes().vec()
                == std::vector<std::int64_t>({2, 3, 3}),
            "padded batch output shape mismatch");
    require(padded.index({1, 2}).eq(0).all().item<bool>(),
            "padded row was not zeroed");
    require(padded.index({1, torch::indexing::Slice(), 2})
                .eq(0)
                .all()
                .item<bool>(),
            "padded column was not zeroed");
}

void test_marginals_and_dummy_paths() {
    nn::SinkhornNetwork network(
        nn::SinkhornNetworkOptions{6, 0.9, 1e-4, true, false});
    const auto matrix = scores_2d();
    nn::SinkhornNetworkCallOptions call;
    call.r = torch::tensor({{0.5F, 1.0F, 1.5F}});
    call.c = torch::tensor({{1.0F, 0.75F, 1.25F}});
    const auto marginals = network->forward(matrix, call);
    require(torch::isfinite(marginals).all().item<bool>(),
            "custom marginal result contains non-finite values");

    nn::SinkhornNetworkCallOptions forbidden;
    forbidden.dummy = true;
    const auto dummy = network->forward(matrix, forbidden);
    require(dummy.index({2, 2}).item<float>() == 0.0F,
            "dummy mode did not forbid the terminal assignment");

    auto rectangular = torch::tensor({
        {0.2F, -0.1F, 0.7F, 0.4F},
        {0.4F, 0.3F, -0.5F, 0.9F},
    });
    nn::SinkhornNetworkCallOptions dummy_row;
    dummy_row.dummy_row = true;
    const auto rectangular_result = network->forward(rectangular, dummy_row);
    require(rectangular_result.sizes().vec()
                == std::vector<std::int64_t>({2, 4}),
            "dummy-row output shape mismatch");
    require(torch::isfinite(rectangular_result).all().item<bool>(),
            "dummy-row result contains non-finite values");

    nn::SinkhornNetwork batched(
        nn::SinkhornNetworkOptions{6, 0.9, 1e-4, true, true});
    dummy_row.dummy = true;
    const auto combined = batched->forward(rectangular, dummy_row);
    require(combined.sizes().vec()
                == std::vector<std::int64_t>({2, 4}),
            "batched dummy + dummy-row output shape mismatch");
}

void test_validation_and_device() {
    nn::SinkhornNetwork network;
    bool rejected = false;
    try {
        static_cast<void>(network->forward(torch::ones({2, 3, 4, 5})));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid score rank was accepted");

    if (torch::cuda::is_available()) {
        const auto cuda_scores = scores_2d().to(torch::kCUDA);
        require(network->forward(cuda_scores).device().is_cuda(),
                "SinkhornNetwork moved CUDA input to CPU");
        nn::Sinkhorn simple;
        require(simple->forward(cuda_scores.unsqueeze(0)).device().is_cuda(),
                "Sinkhorn moved CUDA input to CPU");
    }
}

}  // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_simple_sinkhorn();
        test_network_matrix_and_batch_modes();
        test_marginals_and_dummy_paths();
        test_validation_and_device();
        std::cout << "sinkhorn_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sinkhorn_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
