#include "../virne/solver/learning/neural_network/mlp.h"

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ml = virne::solver::learning::neural_network;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ml::MlpOptions options(
    const std::int64_t input_dim,
    const std::int64_t output_dim,
    const std::int64_t layers,
    ml::MlpHiddenDims hidden = std::monostate{},
    const bool batch_norm = false,
    const double dropout_prob = 1.0) {
    ml::MlpOptions result{};
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.num_layers = layers;
    result.embedding_dims = std::move(hidden);
    result.batch_norm = batch_norm;
    result.dropout_prob = dropout_prob;
    return result;
}

void fill_parameters(ml::MLPNet& model) {
    torch::NoGradGuard no_grad;
    std::int64_t tensor_id = 0;
    for (auto& item : model->named_parameters(true)) {
        auto flat = item.value().view({-1});
        for (std::int64_t index = 0; index < flat.numel(); ++index) {
            const auto numerator =
                ((tensor_id * 37 + index * 17) % 61) - 30;
            flat[index] = static_cast<double>(numerator) / 32.0;
        }
        ++tensor_id;
    }
}

void test_layout_and_shapes() {
    ml::MLPNet model(options(4, 3, 2));
    require(model->resolved_hidden_dims() == std::vector<std::int64_t>{8},
            "default hidden dimension mismatch");

    const auto parameters = model->named_parameters(true);
    require(parameters.contains("lins.0.weight"),
            "missing first Python-compatible state name");
    require(parameters.contains("lins.3.weight"),
            "missing output Python-compatible state name");

    const auto output = model->forward(torch::zeros({7, 4}));
    require(output.sizes() == torch::IntArrayRef({7, 3}),
            "rank-two forward shape mismatch");

    ml::MLPNet rank_three(options(5, 1, 3, std::int64_t{7}));
    const auto rank_three_output =
        rank_three->forward(torch::zeros({2, 4, 5}));
    require(rank_three_output.sizes() == torch::IntArrayRef({2, 4, 1}),
            "rank-three forward shape mismatch");
}

void test_explicit_dims_and_batch_norm() {
    ml::MLPNet model(options(
        3,
        2,
        4,
        std::vector<std::int64_t>{5, 4, 6},
        true));
    require(model->resolved_hidden_dims() ==
                std::vector<std::int64_t>({5, 4, 6}),
            "explicit hidden dimensions mismatch");
    require(model->named_buffers(true).contains("lins.1.running_mean"),
            "missing BatchNorm running mean");

    model->train();
    const auto before =
        model->named_buffers(true)["lins.1.num_batches_tracked"].item<std::int64_t>();
    const auto output = model->forward(torch::ones({8, 3}));
    const auto after =
        model->named_buffers(true)["lins.1.num_batches_tracked"].item<std::int64_t>();
    require(output.sizes() == torch::IntArrayRef({8, 2}),
            "BatchNorm forward shape mismatch");
    require(after == before + 1, "BatchNorm buffer did not advance");
}

void test_dropout_compatibility() {
    torch::manual_seed(11);
    ml::MLPNet zero(options(4, 3, 2, std::monostate{}, false, 0.0));
    torch::manual_seed(11);
    ml::MLPNet one(options(4, 3, 2, std::monostate{}, false, 1.0));
    fill_parameters(zero);
    fill_parameters(one);
    const auto input = torch::arange(0, 28, torch::kFloat32).view({7, 4}) / 8.0;
    require(torch::equal(zero->forward(input), one->forward(input)),
            "dropout_prob must remain compatibility-only");
}

void test_validation() {
    bool bad_layers = false;
    try {
        ml::MLPNet invalid(options(4, 3, 1));
    } catch (const std::invalid_argument&) {
        bad_layers = true;
    }
    require(bad_layers, "single-layer Python error boundary was not rejected");

    bool bad_dims = false;
    try {
        ml::MLPNet invalid(options(
            4,
            3,
            3,
            std::vector<std::int64_t>{7}));
    } catch (const std::invalid_argument&) {
        bad_dims = true;
    }
    require(bad_dims, "invalid explicit hidden dimension count was accepted");
}

} // namespace

int main() {
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        test_layout_and_shapes();
        test_explicit_dims_and_batch_norm();
        test_dropout_compatibility();
        test_validation();
        std::cout << "mlp_net_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mlp_net_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
