#include "adam.h"

#include <torch/torch.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_mul.h>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace virne::solver::learning::rl_core {
namespace {
void validate(const AdamOptions& o) {
    if (!(o.lr >= 0) || !(o.eps >= 0) || !(o.weight_decay >= 0)
        || !(o.beta1 >= 0 && o.beta1 < 1) || !(o.beta2 >= 0 && o.beta2 < 1)) {
        throw std::invalid_argument("invalid Adam scalar options");
    }
}
torch::Tensor copied(const torch::Tensor& value) { return value.defined() ? value.clone() : torch::Tensor{}; }
}

std::vector<torch::Tensor> unique_parameters(torch::nn::Module& module) {
    std::vector<torch::Tensor> result;
    std::unordered_set<const c10::TensorImpl*> seen;
    for (const auto& item : module.named_parameters(true)) {
        if (seen.insert(item.value().unsafeGetTensorImpl()).second) result.push_back(item.value());
    }
    return result;
}

torch::Tensor clip_grad_norm(const std::vector<torch::Tensor>& parameters,
    const double max_norm, const double norm_type, const bool error_if_nonfinite) {
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> gradients;
    for (const auto& parameter : parameters) if (parameter.grad().defined()) gradients.push_back(parameter.grad());
    if (gradients.empty()) return torch::tensor(0.0);
    std::vector<torch::Tensor> norms;
    // Single-tensor ATen norms preserve the same per-tensor reductions as foreach.
    for (const auto& gradient : gradients) {
        norms.push_back(torch::linalg_vector_norm(gradient, norm_type).to(gradients.front().device()));
    }
    const auto total = torch::linalg_vector_norm(torch::stack(norms), norm_type);
    if (error_if_nonfinite && !torch::isfinite(total).item<bool>()) throw std::runtime_error("non-finite gradient norm");
    const auto coefficient = torch::clamp_max(max_norm / (total + 1e-6), 1.0);
    for (auto& gradient : gradients) gradient.mul_(coefficient.to(gradient.device()));
    return total;
}

void sync_gradients(torch::nn::Module& shared, torch::nn::Module& local) {
    const auto destination = unique_parameters(shared);
    const auto source = unique_parameters(local);
    for (std::size_t i = 0; i < std::min(destination.size(), source.size()); ++i) {
        if (source[i].grad().defined()) destination[i].mutable_grad() = source[i].grad().clone().to(destination[i].device());
    }
}

Adam::Adam(std::vector<torch::Tensor> parameters, AdamOptions options)
    : Adam(std::vector<AdamParameterGroup>{{std::move(parameters), options}}) {
    if (parameters_.empty()) throw std::invalid_argument("optimizer got an empty parameter list");
}

Adam::Adam(std::vector<AdamParameterGroup> groups) {
    if (groups.empty()) throw std::invalid_argument("optimizer got an empty parameter list");
    std::unordered_map<const c10::TensorImpl*, std::pair<std::size_t, std::size_t>> ids;
    for (std::size_t group_id = 0; group_id < groups.size(); ++group_id) {
        validate(groups[group_id].options);
        Group group{{}, groups[group_id].options};
        for (auto& parameter : groups[group_id].parameters) {
            if (!parameter.defined() || !parameter.is_leaf()) throw std::invalid_argument("Adam requires defined leaf parameters");
            const auto key = parameter.unsafeGetTensorImpl();
            auto found = ids.find(key);
            if (found == ids.end()) {
                const auto state_id = parameters_.size();
                parameters_.push_back(parameter);
                ids.emplace(key, std::make_pair(group_id, state_id));
                group.state_ids.push_back(state_id);
            } else {
                if (found->second.first != group_id) throw std::invalid_argument("parameter appears in more than one Adam group");
                group.state_ids.push_back(found->second.second);
            }
        }
        groups_.push_back(std::move(group));
    }
    moments_.resize(parameters_.size());
}

void Adam::zero_grad(const bool set_to_none) {
    for (auto& parameter : parameters_) {
        if (!parameter.grad().defined()) continue;
        if (set_to_none) parameter.mutable_grad() = torch::Tensor{};
        else {
            if (parameter.grad().grad_fn()) parameter.mutable_grad() = parameter.grad().detach();
            else parameter.mutable_grad().set_requires_grad(false);
            parameter.mutable_grad().zero_();
        }
    }
}

void Adam::step() {
    torch::NoGradGuard no_grad;
    for (const auto& group : groups_) {
        const auto& o = group.options;
        // Python initializes every used state in a group before updating any parameter.
        for (const auto id : group.state_ids) {
            const auto& parameter = parameters_[id];
            if (!parameter.grad().defined()) continue;
            if (parameter.grad().is_sparse()) throw std::runtime_error("Adam does not support sparse gradients");
            auto& state = moments_[id];
            if (!state.step.defined()) {
                const auto dtype = c10::get_default_dtype_as_scalartype() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
                state.step = torch::zeros({}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
                state.exp_avg = torch::zeros_like(parameter, {}, torch::MemoryFormat::Preserve);
                state.exp_avg_sq = torch::zeros_like(parameter, {}, torch::MemoryFormat::Preserve);
                if (o.amsgrad) state.max_exp_avg_sq = torch::zeros_like(parameter, {}, torch::MemoryFormat::Preserve);
            }
        }
        for (const auto id : group.state_ids) {
            auto parameter = parameters_[id];
            if (!parameter.grad().defined()) continue;
            auto gradient = o.maximize ? -parameter.grad() : parameter.grad();
            auto& state = moments_[id];
            state.step.add_(1);
            if (o.weight_decay != 0) gradient = gradient.add(parameter, o.weight_decay);
            auto mean = state.exp_avg;
            auto variance = state.exp_avg_sq;
            auto maximum = state.max_exp_avg_sq;
            if (parameter.is_complex()) {
                parameter = torch::view_as_real(parameter);
                gradient = torch::view_as_real(gradient);
                mean = torch::view_as_real(mean);
                variance = torch::view_as_real(variance);
                if (o.amsgrad) maximum = torch::view_as_real(maximum);
            }
            mean.lerp_(gradient, 1 - o.beta1);
            variance.mul_(o.beta2).addcmul_(gradient, gradient.conj(), 1 - o.beta2);
            const double step = state.step.item<double>();
            const double correction1 = 1 - std::pow(o.beta1, step);
            const double correction2 = 1 - std::pow(o.beta2, step);
            if (o.amsgrad) maximum.copy_(torch::maximum(maximum, variance));
            const auto denominator = ((o.amsgrad ? maximum : variance).sqrt() / std::pow(correction2, 0.5)).add_(o.eps);
            parameter.addcdiv_(mean, denominator, -o.lr / correction1);
        }
    }
}

AdamSnapshot Adam::snapshot() const {
    torch::NoGradGuard no_grad;
    AdamSnapshot result;
    for (const auto& state : moments_) result.moments.push_back({copied(state.step), copied(state.exp_avg), copied(state.exp_avg_sq), copied(state.max_exp_avg_sq)});
    for (const auto& group : groups_) result.options.push_back(group.options);
    return result;
}
void Adam::restore(const AdamSnapshot& snapshot) {
    if (snapshot.moments.size() != moments_.size() || snapshot.options.size() != groups_.size()) throw std::invalid_argument("Adam snapshot layout mismatch");
    for (const auto& options : snapshot.options) validate(options);
    auto restored = snapshot.moments;
    for (std::size_t i = 0; i < restored.size(); ++i) {
        auto& state = restored[i];
        if (!state.step.defined()) continue;
        if (state.exp_avg.sizes() != parameters_[i].sizes() || state.exp_avg_sq.sizes() != parameters_[i].sizes()) throw std::invalid_argument("Adam moment shape mismatch");
        state.step = state.step.detach().cpu().clone();
        state.exp_avg = state.exp_avg.detach().to(parameters_[i].options()).clone();
        state.exp_avg_sq = state.exp_avg_sq.detach().to(parameters_[i].options()).clone();
        if (state.max_exp_avg_sq.defined()) state.max_exp_avg_sq = state.max_exp_avg_sq.detach().to(parameters_[i].options()).clone();
    }
    moments_ = std::move(restored);
    for (std::size_t i = 0; i < groups_.size(); ++i) groups_[i].options = snapshot.options[i];
}
AdamOptions& Adam::options(const std::size_t group) { return groups_.at(group).options; }

}  // namespace virne::solver::learning::rl_core
