#include "changeable_system.h"

#include <iterator>
#include <utility>

namespace virne::system {

ChangeableStage::ChangeableStage(
    core::SolutionStepEnvironment& environment_value,
    SystemRunConfig config_value)
    : environment(environment_value), config(std::move(config_value)) {}

ChangeableSystem::ChangeableSystem(
    std::vector<ChangeableStage> stages,
    solver::Solver& solver)
    : stages_(std::move(stages)), solver_(solver) {
    if (stages_.empty()) {
        throw SystemException(
            SystemErrorCode::empty_changeable_stages,
            SystemOperation::construct_changeable,
            "changeable system requires at least one typed stage");
    }
}

ChangeableRunResult ChangeableSystem::run(RandomContext& random) {
    ChangeableRunResult result;
    result.stages.reserve(stages_.size());

    std::size_t next_epoch_index = 0U;
    for (std::size_t stage_index = 0U;
         stage_index < stages_.size();
        ++stage_index) {
        const auto& stage = stages_[stage_index];
        OnlineSystem online(stage.environment.get(), solver_.get());
        SystemRunConfig stage_config = stage.config;
        stage_config.progress_epoch_offset += next_epoch_index;
        SystemRunResult stage_run = online.run(random, stage_config);

        ChangeableStageSummary summary;
        summary.stage_index = stage_index;
        summary.first_epoch_index = next_epoch_index;
        summary.num_epochs = stage_run.epochs.size();
        summary.first_step_index = result.run.steps.size();
        summary.num_steps = stage_run.steps.size();

        for (auto& step : stage_run.steps) {
            step.epoch_index += next_epoch_index;
            step.stage_index = stage_index;
        }
        for (auto& epoch : stage_run.epochs) {
            epoch.epoch_index += next_epoch_index;
        }

        next_epoch_index += stage_run.epochs.size();
        result.run.steps.insert(
            result.run.steps.end(),
            std::make_move_iterator(stage_run.steps.begin()),
            std::make_move_iterator(stage_run.steps.end()));
        result.run.epochs.insert(
            result.run.epochs.end(),
            std::make_move_iterator(stage_run.epochs.begin()),
            std::make_move_iterator(stage_run.epochs.end()));
        result.stages.push_back(summary);
    }
    return result;
}

const std::vector<ChangeableStage>& ChangeableSystem::stages() const noexcept {
    return stages_;
}

solver::Solver& ChangeableSystem::solver_instance() const noexcept {
    return solver_.get();
}

} // namespace virne::system
