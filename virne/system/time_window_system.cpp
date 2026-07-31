#include "time_window_system.h"

#include "../../random/random_context.h"

#include <cmath>
#include <limits>
#include <utility>

namespace virne::system {
namespace {

std::size_t resolve_window_index(
    double event_time,
    const TimeWindowRunConfig& config) {
    if (!std::isfinite(event_time) || event_time < config.window_origin) {
        throw SystemException(
            SystemErrorCode::invalid_event_time,
            SystemOperation::run_time_window,
            "time-window arrival must be finite and not precede the origin");
    }
    const long double raw = std::floor(
        (static_cast<long double>(event_time) -
         static_cast<long double>(config.window_origin)) /
        static_cast<long double>(config.window_size));
    const long double maximum = static_cast<long double>(
        std::numeric_limits<std::size_t>::max());
    if (!std::isfinite(raw) || raw > maximum) {
        throw SystemException(
            SystemErrorCode::time_window_index_overflow,
            SystemOperation::run_time_window,
            "time-window index exceeds the native index lane");
    }
    return static_cast<std::size_t>(raw);
}

} // namespace

TimeWindowRunResult TimeWindowSystem::run(
    RandomContext& random,
    const TimeWindowRunConfig& config) {
    if (!std::isfinite(config.window_size) || config.window_size <= 0.0 ||
        !std::isfinite(config.window_origin)) {
        throw SystemException(
            SystemErrorCode::invalid_time_window,
            SystemOperation::run_time_window,
            "time-window size must be finite and positive and origin finite");
    }

    ready();
    TimeWindowRunResult result;
    result.run.epochs.reserve(config.system.num_simulations);

    for (std::size_t epoch_index = 0U;
         epoch_index < config.system.num_simulations;
         ++epoch_index) {
        const std::size_t progress_epoch =
            config.system.progress_epoch_offset + epoch_index;
        if (config.system.renew_virtual_networks ||
            config.system.renew_event_schedule) {
            core::EnvironmentResetOptions options;
            options.seed = config.system.seed;
            options.renew_virtual_networks =
                config.system.renew_virtual_networks;
            options.renew_event_schedule =
                config.system.renew_event_schedule;
            options.simulator_workers = config.system.simulator_workers;
            environment().renew_and_reset(random, options);
        } else {
            environment().reset();
        }

        SystemEpochSummary epoch;
        epoch.epoch_index = epoch_index;
        const std::size_t total = environment().simulator().num_v_nets();
        if (config.system.progress != nullptr) {
            config.system.progress->begin_epoch(progress_epoch, total);
        }
        while (true) {
            const auto& event = environment().current_event();
            const std::size_t window_index =
                resolve_window_index(event.time, config);
            if (result.windows.empty() ||
                result.windows.back().epoch_index != epoch_index ||
                result.windows.back().window_index != window_index) {
                TimeWindowSummary window;
                window.epoch_index = epoch_index;
                window.window_index = window_index;
                window.first_event_time = event.time;
                window.last_event_time = event.time;
                result.windows.push_back(window);
            }

            const auto request_id = event.request_id;
            const double event_time = event.time;
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

            auto& window = result.windows.back();
            window.last_event_time = event_time;
            ++window.arrival_steps;
            ++epoch.arrival_steps;
            if (step.accepted) {
                ++window.accepted;
                ++epoch.accepted;
            } else {
                ++window.rejected;
                ++epoch.rejected;
            }
            window.auto_released_events += step.auto_released_events;
            epoch.auto_released_events += step.auto_released_events;
            if (step.done) {
                epoch.summary = step.summary.has_value()
                    ? *step.summary
                    : environment().summary();
            }
            if (config.system.progress != nullptr) {
                const SystemProgressUpdate progress_update{
                    progress_epoch,
                    epoch.arrival_steps,
                    total,
                    step.record.state.success_count,
                    step.record.state.inservice_count,
                    step.record.state.long_term_r2c_ratio,
                };
                config.system.progress->update(progress_update);
                if (step.done) {
                    config.system.progress->end_epoch(progress_update);
                }
            }

            if (config.system.capture_solutions) {
                result.run.steps.emplace_back(
                    epoch_index,
                    request_id,
                    std::move(solution),
                    step.accepted,
                    step.failure_reason,
                    step.auto_released_events,
                    event_time,
                    0U,
                    window_index);
            }

            if (step.done) {
                break;
            }
        }
        result.run.epochs.push_back(std::move(epoch));
    }
    return result;
}

} // namespace virne::system
