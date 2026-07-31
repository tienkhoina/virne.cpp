#pragma once

#include "online_system.h"

#include <functional>
#include <vector>

namespace virne::system {

struct ChangeableStage {
    ChangeableStage(
        core::SolutionStepEnvironment& environment_value,
        SystemRunConfig config_value = {});

    std::reference_wrapper<core::SolutionStepEnvironment> environment;
    SystemRunConfig config;
};

struct ChangeableStageSummary {
    std::size_t stage_index = 0U;
    std::size_t first_epoch_index = 0U;
    std::size_t num_epochs = 0U;
    std::size_t first_step_index = 0U;
    std::size_t num_steps = 0U;
};

struct ChangeableRunResult {
    SystemRunResult run;
    std::vector<ChangeableStageSummary> stages;
};

class ChangeableSystem final {
public:
    ChangeableSystem(
        std::vector<ChangeableStage> stages,
        solver::Solver& solver);

    ChangeableRunResult run(RandomContext& random);

    const std::vector<ChangeableStage>& stages() const noexcept;
    solver::Solver& solver_instance() const noexcept;

private:
    std::vector<ChangeableStage> stages_;
    std::reference_wrapper<solver::Solver> solver_;
};

} // namespace virne::system
