#pragma once

#include "../core/environment.h"
#include "../solver/base_solver.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class RandomContext;

namespace virne::system {

class SystemProgressSink;

enum class SystemMode : std::uint8_t {
    online,
    offline,
    changeable,
    time_window,
};

enum class SystemErrorCode : std::uint8_t {
    empty_changeable_stages,
    invalid_time_window,
    invalid_event_time,
    time_window_index_overflow,
};

enum class SystemOperation : std::uint8_t {
    construct_changeable,
    run_time_window,
};

class SystemException final : public std::runtime_error {
public:
    SystemException(
        SystemErrorCode code,
        SystemOperation operation,
        std::string message);

    SystemErrorCode code() const noexcept;
    SystemOperation operation() const noexcept;

private:
    SystemErrorCode code_;
    SystemOperation operation_;
};

struct SystemRunConfig {
    std::size_t num_simulations = 1U;
    std::optional<std::uint32_t> seed;
    bool renew_virtual_networks = false;
    bool renew_event_schedule = false;
    network::VirtualSimulationWorkers simulator_workers;
    bool capture_solutions = true;
    // Borrowed, single-producer callback. The sink must outlive synchronous
    // run(); worker threads must marshal updates through the coordinator.
    SystemProgressSink* progress = nullptr;
    // Only affects progress labels. Result/trace epoch indexes remain local.
    std::size_t progress_epoch_offset = 0U;
};

struct SystemProgressUpdate {
    std::size_t epoch_index = 0U;
    std::size_t completed = 0U;
    std::size_t total = 0U;
    std::int64_t success_count = 0;
    std::int64_t inservice_count = 0;
    double long_term_r2c_ratio = 0.0;
};

class SystemProgressSink {
public:
    virtual ~SystemProgressSink() = default;

    virtual void begin_epoch(
        std::size_t epoch_index,
        std::size_t total) = 0;
    virtual void update(const SystemProgressUpdate& update) = 0;
    virtual void end_epoch(const SystemProgressUpdate& update) = 0;
};

struct SystemStepTrace {
    SystemStepTrace(
        std::size_t epoch_index_value,
        network::VirtualRequestId request_id_value,
        core::Solution solution_value,
        bool accepted_value,
        core::EnvironmentFailureReason failure_reason_value,
        std::size_t auto_released_events_value,
        double event_time_value = 0.0,
        std::size_t stage_index_value = 0U,
        std::size_t window_index_value = 0U);

    std::size_t epoch_index = 0U;
    network::VirtualRequestId request_id = 0;
    core::Solution solution;
    bool accepted = false;
    core::EnvironmentFailureReason failure_reason =
        core::EnvironmentFailureReason::none;
    std::size_t auto_released_events = 0U;
    double event_time = 0.0;
    std::size_t stage_index = 0U;
    std::size_t window_index = 0U;
};

struct SystemEpochSummary {
    std::size_t epoch_index = 0U;
    std::size_t arrival_steps = 0U;
    std::size_t accepted = 0U;
    std::size_t rejected = 0U;
    std::size_t auto_released_events = 0U;
    core::CounterSummary summary;
};

struct SystemRunResult {
    std::vector<SystemStepTrace> steps;
    std::vector<SystemEpochSummary> epochs;
};

class BaseSystem {
public:
    BaseSystem(
        core::SolutionStepEnvironment& environment,
        solver::Solver& solver);
    virtual ~BaseSystem() = default;

    BaseSystem(const BaseSystem&) = delete;
    BaseSystem& operator=(const BaseSystem&) = delete;

    core::SolutionStepEnvironment& environment() const noexcept;
    solver::Solver& solver_instance() const noexcept;

protected:
    void ready();

private:
    std::reference_wrapper<core::SolutionStepEnvironment> environment_;
    std::reference_wrapper<solver::Solver> solver_;
};

} // namespace virne::system
