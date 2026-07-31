#include "online_system.h"

#include "../../random/random_context.h"

#include <utility>

namespace virne::system {

SystemRunResult OnlineSystem::run(
    RandomContext& random,
    const SystemRunConfig& config) {
    ready();
    SystemRunResult result;
    result.epochs.reserve(config.num_simulations);

    for (std::size_t epoch = 0U;
         epoch < config.num_simulations;
         ++epoch) {
        const std::size_t progress_epoch =
            config.progress_epoch_offset + epoch;
        if (config.renew_virtual_networks || config.renew_event_schedule) {
            core::EnvironmentResetOptions options;
            options.seed = config.seed;
            options.renew_virtual_networks = config.renew_virtual_networks;
            options.renew_event_schedule = config.renew_event_schedule;
            options.simulator_workers = config.simulator_workers;
            environment().renew_and_reset(random, options);
        } else {
            environment().reset();
        }

        SystemEpochSummary epoch_summary;
        epoch_summary.epoch_index = epoch;
        const std::size_t total = environment().simulator().num_v_nets();
        if (config.progress != nullptr) {
            config.progress->begin_epoch(progress_epoch, total);
        }
        while (true) {
            const core::EnvironmentEventState& current_event =
                environment().current_event();
            const auto request_id = current_event.request_id;
            const double event_time = current_event.time;
            auto transaction = environment().solver_transaction();
            solver::MutableSolverResult solve_result = [&]() {
                try {
                    return solver_instance().solve_mutable(
                        solver::MutableSolverInstance{
                            environment().current_virtual_network(),
                            transaction.physical_network,
                            transaction.mutation});
                } catch (...) {
                    transaction.mutation.rollback_transaction();
                    throw;
                }
            }();
            const core::EnvironmentSolutionState solution_state =
                solve_result.mutation_state ==
                        solver::SolverMutationState::committed
                    ? core::EnvironmentSolutionState::committed
                    : core::EnvironmentSolutionState::detached;
            core::Solution solution = std::move(solve_result.solution);
            core::EnvironmentStepResult step =
                environment().step(solution, solution_state);

            ++epoch_summary.arrival_steps;
            if (step.accepted) {
                ++epoch_summary.accepted;
            } else {
                ++epoch_summary.rejected;
            }
            epoch_summary.auto_released_events +=
                step.auto_released_events;
            if (step.done) {
                epoch_summary.summary = step.summary.has_value()
                    ? std::move(*step.summary)
                    : environment().summary();
            }
            if (config.progress != nullptr) {
                const SystemProgressUpdate progress_update{
                    progress_epoch,
                    epoch_summary.arrival_steps,
                    total,
                    step.record.state.success_count,
                    step.record.state.inservice_count,
                    step.record.state.long_term_r2c_ratio,
                };
                config.progress->update(progress_update);
                if (step.done) {
                    config.progress->end_epoch(progress_update);
                }
            }
            if (config.capture_solutions) {
                result.steps.emplace_back(
                    epoch,
                    request_id,
                    std::move(solution),
                    step.accepted,
                    step.failure_reason,
                    step.auto_released_events,
                    event_time);
            }

            if (step.done) {
                break;
            }
        }
        result.epochs.push_back(std::move(epoch_summary));
    }
    return result;
}

} // namespace virne::system
