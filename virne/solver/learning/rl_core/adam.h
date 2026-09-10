#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>
#include <cstdint>
#include <vector>

namespace virne::solver::learning::rl_core {

struct AdamOptions {
    double lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0;
    bool amsgrad = false, maximize = false;
};
struct AdamParameterGroup { std::vector<torch::Tensor> parameters; AdamOptions options; };
struct AdamMoments { torch::Tensor step, exp_avg, exp_avg_sq, max_exp_avg_sq; };
struct AdamSnapshot { std::vector<AdamMoments> moments; std::vector<AdamOptions> options; };

std::vector<torch::Tensor> unique_parameters(torch::nn::Module& module);
torch::Tensor clip_grad_norm(const std::vector<torch::Tensor>& parameters, double max_norm,
    double norm_type = 2.0, bool error_if_nonfinite = false);
void sync_gradients(torch::nn::Module& shared, torch::nn::Module& local);

// Non-capturable, non-differentiable Adam, matching Torch 2.6's CPU default path.
// State IDs are resolved at construction; step/zero_grad use only direct lanes.
class Adam final {
public:
    explicit Adam(std::vector<AdamParameterGroup> groups);
    explicit Adam(std::vector<torch::Tensor> parameters, AdamOptions options = {});
    void zero_grad(bool set_to_none = true);
    void step();
    AdamSnapshot snapshot() const;
    void restore(const AdamSnapshot& snapshot);
    AdamOptions& options(std::size_t group);
    const std::vector<AdamMoments>& moments() const noexcept { return moments_; }
    const std::vector<torch::Tensor>& parameters() const noexcept { return parameters_; }
private:
    struct Group { std::vector<std::size_t> state_ids; AdamOptions options; };
    std::vector<Group> groups_;
    std::vector<torch::Tensor> parameters_;
    std::vector<AdamMoments> moments_;
};

}  // namespace virne::solver::learning::rl_core
