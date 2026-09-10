#pragma once
#include "../../../../random/numpy_random_state.h"
#include <torch/types.h>
#include <functional>
#include <optional>

namespace virne::solver::learning::rl_core {
struct DqnExplorationOptions {
    double epsilon_start = 1., epsilon_end = .1, epsilon_decay = 10000.;
    std::int64_t num_actions = 0;
    bool mask_actions = true, maskable_policy = true;
    std::uint32_t seed = 0;
};
struct DqnExplorationState {
    NumpyRandomState::Snapshot random;
    std::int64_t steps_done = 0;
    double epsilon = 1.;
};
struct DqnActionSelection { std::int64_t action = 0; torch::Tensor log_probability; };
class DqnActionSelector final {
public:
    explicit DqnActionSelector(DqnExplorationOptions options);
    DqnActionSelection select(const std::function<torch::Tensor()>& q_values,
        const std::optional<torch::Tensor>& mask = std::nullopt, bool sample = true);
    void seed(std::uint32_t seed);
    DqnExplorationState snapshot() const noexcept;
    void restore(const DqnExplorationState& state);
private:
    DqnExplorationOptions options_;
    NumpyRandomState random_;
    std::int64_t steps_done_ = 0;
    double epsilon_;
};
}  // namespace virne::solver::learning::rl_core
