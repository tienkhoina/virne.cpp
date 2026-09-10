#include "multipler.h"

#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace virne::solver::learning::neural_network {
namespace {

MultiplerNetOptions validate_options(MultiplerNetOptions options) {
    if (options.state_dim <= 0) {
        throw std::invalid_argument(
            "MultiplerNet state_dim must be positive");
    }
    return options;
}

void validate_state(
    const torch::Tensor& state,
    const std::int64_t state_dim) {
    if (!state.defined() || state.dim() == 0
        || state.size(-1) != state_dim) {
        throw std::invalid_argument(
            "MultiplerNet state must end with state_dim features");
    }
}

}  // namespace

MultiplerNetImpl::MultiplerNetImpl(MultiplerNetOptions options)
    : options_(validate_options(std::move(options))) {
    // Construction order matches Python and therefore preserves seeded default
    // Linear initialization exactly for the same LibTorch version.
    l1 = register_module(
        "l1", torch::nn::Linear(options_.state_dim, 256));
    l2 = register_module("l2", torch::nn::Linear(256, 256));
    l3 = register_module("l3", torch::nn::Linear(256, 1));
}

torch::Tensor MultiplerNetImpl::forward(const torch::Tensor& state) {
    validate_state(state, options_.state_dim);
    auto activation = torch::leaky_relu(l1->forward(state), 0.01);
    activation = torch::leaky_relu(l2->forward(activation), 0.01);
    return torch::softplus(l3->forward(activation), 1.0, 20.0);
}

}  // namespace virne::solver::learning::neural_network
