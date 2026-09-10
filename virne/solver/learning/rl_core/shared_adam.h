#pragma once
#include "adam.h"
#include <mutex>
#include <optional>

namespace virne::solver::learning::rl_core {

struct SharedAdamSnapshot {
    std::vector<torch::Tensor> parameters;
    AdamSnapshot optimizer;
    std::uint64_t version = 0;
};
struct SharedAdamUpdate { std::uint64_t version = 0; torch::Tensor gradient_norm; };

// Native worker threads share one owner. Gradient packets are applied under
// one lock; callers obtain coherent parameter/moment snapshots through snapshot.
class SharedAdam final {
public:
    explicit SharedAdam(std::vector<torch::Tensor> parameters,
        AdamOptions options = AdamOptions{1e-3, 0.9, 0.99, 1e-8, 0.0, false, false});
    explicit SharedAdam(std::vector<AdamParameterGroup> groups);
    SharedAdamUpdate apply_gradients(const std::vector<torch::Tensor>& gradients,
        std::optional<double> max_gradient_norm = std::nullopt);
    SharedAdamSnapshot snapshot() const;
    void restore(const SharedAdamSnapshot& snapshot);
private:
    void initialize_moments(const std::vector<AdamParameterGroup>& groups);
    mutable std::mutex mutex_;
    Adam optimizer_;
    std::uint64_t version_ = 0;
};

}  // namespace virne::solver::learning::rl_core
