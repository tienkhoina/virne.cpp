#include "offline_system.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace virne::system {
namespace {

core::EnvironmentFailureReason classify_failure(
    const core::Solution& solution) noexcept {
    if (solution.result) {
        return core::EnvironmentFailureReason::none;
    }
    if (solution.early_rejection) {
        return core::EnvironmentFailureReason::early_rejection;
    }
    if (!solution.place_result) {
        return core::EnvironmentFailureReason::placement;
    }
    if (!solution.route_result) {
        return core::EnvironmentFailureReason::routing;
    }
    return core::EnvironmentFailureReason::unknown;
}

void accumulate_offline_summary(
    core::CounterSummary& summary,
    const core::Solution& solution) noexcept {
    summary.average_r2c_ratio += solution.v_net_r2c_ratio;
    summary.total_cost += solution.v_net_cost;
    summary.total_revenue += solution.v_net_revenue;
    summary.total_time_cost += solution.v_net_time_cost;
    summary.total_time_revenue += solution.v_net_time_revenue;
    summary.total_simulation_time = std::max(
        summary.total_simulation_time,
        solution.v_net_arrival_time + solution.v_net_lifetime);
    summary.total_violation +=
        solution.v_net_total_hard_constraint_violation;
    if (solution.early_rejection) {
        ++summary.early_rejection_count;
    } else if (!solution.place_result) {
        ++summary.place_failure_count;
    } else if (!solution.route_result) {
        ++summary.route_failure_count;
    }
}

void finish_offline_summary(SystemEpochSummary& epoch) noexcept {
    epoch.summary.success_count = static_cast<std::int64_t>(epoch.accepted);
    if (epoch.arrival_steps != 0U) {
        epoch.summary.acceptance_rate =
            static_cast<double>(epoch.accepted) /
            static_cast<double>(epoch.arrival_steps);
        epoch.summary.average_r2c_ratio /=
            static_cast<double>(epoch.arrival_steps);
    }
    if (epoch.summary.total_cost != 0.0) {
        epoch.summary.long_term_r2c_ratio =
            epoch.summary.total_revenue / epoch.summary.total_cost;
    }
    if (epoch.summary.total_time_cost != 0.0) {
        epoch.summary.long_term_time_r2c_ratio =
            epoch.summary.total_time_revenue /
            epoch.summary.total_time_cost;
    }
    if (epoch.summary.total_simulation_time != 0.0) {
        epoch.summary.long_term_average_revenue =
            epoch.summary.total_revenue /
            epoch.summary.total_simulation_time;
        epoch.summary.long_term_average_cost =
            epoch.summary.total_cost /
            epoch.summary.total_simulation_time;
        epoch.summary.long_term_average_time_revenue =
            epoch.summary.total_time_revenue /
            epoch.summary.total_simulation_time;
    }
    epoch.summary.maximum_inservice_count = epoch.accepted == 0U ? 0 : 1;
}

core::Solution solve_independently(
    solver::Solver& solver,
    const network::VirtualNetwork& request,
    network::PhysicalNetwork& physical_network,
    core::controller::PreparedControllerMutation& mutation) {
    mutation.begin_transaction();
    try {
        auto result = solver.solve_mutable(solver::MutableSolverInstance{
            request,
            physical_network,
            mutation});

        // Offline semantics always evaluate the next request against the
        // same physical snapshot. The exact checkpoint handles both mutable
        // committed solvers and the detached const-solve compatibility path.
        mutation.rollback_transaction();
        return std::move(result.solution);
    } catch (...) {
        if (mutation.transaction_active()) {
            mutation.rollback_transaction();
        }
        throw;
    }
}

} // namespace

OfflineSystem::OfflineSystem(
    const network::PhysicalNetwork& physical_network,
    const network::VirtualNetworkRequestSimulator& simulator,
    solver::Solver& solver)
    : physical_network_(physical_network),
      simulator_(simulator),
      solver_(solver) {}

SystemRunResult OfflineSystem::run(const OfflineRunConfig& config) {
    solver_.get().ready();

    SystemRunResult result;
    result.epochs.reserve(config.num_simulations);
    const auto& requests = simulator_.get().v_nets();
    auto working_physical_network = physical_network_.get().clone();
    std::vector<core::PreparedCounter> prepared_counters;
    std::vector<core::controller::PreparedControllerMutation>
        prepared_mutations;
    prepared_counters.reserve(requests.size());
    prepared_mutations.reserve(requests.size());
    for (const auto& request : requests) {
        prepared_counters.push_back(
            solver_.get().counter().prepare(request));
        prepared_mutations.push_back(
            solver_.get().controller().prepare_mutation(
                request, working_physical_network));
    }
    if (config.capture_solutions) {
        result.steps.reserve(config.num_simulations * requests.size());
    }

    for (std::size_t epoch_index = 0U;
         epoch_index < config.num_simulations;
         ++epoch_index) {
        SystemEpochSummary epoch;
        epoch.epoch_index = epoch_index;
        auto* const progress = config.progress;
        if (progress != nullptr) {
            progress->begin_epoch(epoch_index, requests.size());
        }
        SystemProgressUpdate progress_update{
            epoch_index, 0U, requests.size(), 0, 0, 0.0};
        std::priority_queue<
            double,
            std::vector<double>,
            std::greater<double>> active_departures;
        for (std::size_t request_index = 0U;
             request_index < requests.size();
            ++request_index) {
            const auto& request = requests[request_index];
            core::Solution solution = solve_independently(
                solver_.get(),
                request,
                working_physical_network,
                prepared_mutations[request_index]);
            prepared_counters[request_index].count_solution(
                solution,
                core::CounterOptions{config.counter_workers});
            const bool accepted = solution.result;
            ++epoch.arrival_steps;
            if (accepted) {
                ++epoch.accepted;
            } else {
                ++epoch.rejected;
            }
            accumulate_offline_summary(epoch.summary, solution);
            if (progress != nullptr) {
                const double arrival_time = solution.v_net_arrival_time;
                while (!active_departures.empty() &&
                       active_departures.top() < arrival_time) {
                    active_departures.pop();
                }
                if (accepted) {
                    active_departures.push(
                        arrival_time + solution.v_net_lifetime);
                }
                progress_update.completed = epoch.arrival_steps;
                progress_update.success_count =
                    static_cast<std::int64_t>(epoch.accepted);
                progress_update.inservice_count =
                    static_cast<std::int64_t>(active_departures.size());
                progress_update.long_term_r2c_ratio =
                    epoch.summary.total_cost == 0.0
                    ? 0.0
                    : epoch.summary.total_revenue / epoch.summary.total_cost;
                progress->update(progress_update);
            }
            if (config.capture_solutions) {
                const auto request_id = solution.v_net_id;
                const double arrival_time = solution.v_net_arrival_time;
                const auto failure_reason = classify_failure(solution);
                result.steps.emplace_back(
                    epoch_index,
                    request_id,
                    std::move(solution),
                    accepted,
                    failure_reason,
                    0U,
                    arrival_time);
            }
        }
        finish_offline_summary(epoch);
        if (progress != nullptr) {
            progress->end_epoch(progress_update);
        }
        result.epochs.push_back(std::move(epoch));
    }
    return result;
}

const network::PhysicalNetwork& OfflineSystem::physical_network() const
    noexcept {
    return physical_network_.get();
}

const network::VirtualNetworkRequestSimulator& OfflineSystem::simulator() const
    noexcept {
    return simulator_.get();
}

solver::Solver& OfflineSystem::solver_instance() const noexcept {
    return solver_.get();
}

} // namespace virne::system
