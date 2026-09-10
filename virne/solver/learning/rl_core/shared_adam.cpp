#include "shared_adam.h"
#include <torch/torch.h>
#include <stdexcept>
#include <unordered_map>

namespace virne::solver::learning::rl_core {

SharedAdam::SharedAdam(std::vector<torch::Tensor> parameters, AdamOptions options)
    : optimizer_(parameters, options) { initialize_moments({{std::move(parameters), options}}); }
SharedAdam::SharedAdam(std::vector<AdamParameterGroup> groups)
    : optimizer_(groups) { initialize_moments(groups); }

void SharedAdam::initialize_moments(const std::vector<AdamParameterGroup>& groups) {
    std::unordered_map<const c10::TensorImpl*, bool> amsgrad;
    for (const auto& group : groups) for (const auto& parameter : group.parameters)
        amsgrad.emplace(parameter.unsafeGetTensorImpl(), group.options.amsgrad);
    auto state = optimizer_.snapshot();
    for (std::size_t i = 0; i < state.moments.size(); ++i) {
        const auto& parameter = optimizer_.parameters()[i];
        auto& moment = state.moments[i];
        // Unlike the original integer counter, this is valid Torch 2.6 state.
        const auto dtype = c10::get_default_dtype_as_scalartype() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
        moment.step = torch::zeros({}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        moment.exp_avg = torch::zeros_like(parameter);
        moment.exp_avg_sq = torch::zeros_like(parameter);
        if (amsgrad.at(parameter.unsafeGetTensorImpl())) moment.max_exp_avg_sq = torch::zeros_like(parameter);
    }
    optimizer_.restore(state);
}

SharedAdamUpdate SharedAdam::apply_gradients(const std::vector<torch::Tensor>& gradients,
    const std::optional<double> max_gradient_norm) {
    std::lock_guard<std::mutex> lock(mutex_);
    torch::NoGradGuard no_grad;
    const auto& parameters = optimizer_.parameters();
    if (gradients.size() != parameters.size()) throw std::invalid_argument("shared gradient packet size mismatch");
    std::vector<torch::Tensor> copied;
    copied.reserve(gradients.size());
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        if (!gradients[i].defined()) { copied.emplace_back(); continue; }
        if (gradients[i].sizes() != parameters[i].sizes() || gradients[i].scalar_type() != parameters[i].scalar_type()
            || gradients[i].device() != parameters[i].device() || gradients[i].is_sparse()) {
            throw std::invalid_argument("shared gradient packet shape/dtype/device mismatch");
        }
        copied.push_back(gradients[i].detach().clone());
    }
    optimizer_.zero_grad();
    for (std::size_t i = 0; i < copied.size(); ++i) parameters[i].mutable_grad() = copied[i];
    torch::Tensor norm;
    if (max_gradient_norm) norm = clip_grad_norm(parameters, *max_gradient_norm);
    optimizer_.step();
    return {++version_, norm};
}

SharedAdamSnapshot SharedAdam::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    torch::NoGradGuard no_grad;
    SharedAdamSnapshot result;
    for (const auto& parameter : optimizer_.parameters()) result.parameters.push_back(parameter.detach().clone());
    result.optimizer = optimizer_.snapshot();
    result.version = version_;
    return result;
}

void SharedAdam::restore(const SharedAdamSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    torch::NoGradGuard no_grad;
    if (snapshot.parameters.size() != optimizer_.parameters().size()) throw std::invalid_argument("shared snapshot parameter count mismatch");
    for (std::size_t i = 0; i < snapshot.parameters.size(); ++i) {
        if (!snapshot.parameters[i].defined() || snapshot.parameters[i].sizes() != optimizer_.parameters()[i].sizes()) {
            throw std::invalid_argument("shared snapshot parameter shape mismatch");
        }
    }
    optimizer_.restore(snapshot.optimizer);
    for (std::size_t i = 0; i < snapshot.parameters.size(); ++i) optimizer_.parameters()[i].copy_(snapshot.parameters[i]);
    version_ = snapshot.version;
}

}  // namespace virne::solver::learning::rl_core
