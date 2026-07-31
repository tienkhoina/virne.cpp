#pragma once

#include "controller/controller.h"
#include "counter.h"
#include "recorder.h"
#include "../network/physical_network.h"
#include "../network/virtual_network_request_simulator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class RandomContext;

namespace virne::core {

enum class EnvironmentPhase : std::uint8_t {
    unready,
    active,
    finished,
};

enum class EnvironmentFailureReason : std::uint8_t {
    none,
    early_rejection,
    placement,
    routing,
    unknown,
};

// Describes whether a submitted Solution is a detached journal or has already
// committed its recorded resource mutations to this Environment's p-net.
enum class EnvironmentSolutionState : std::uint8_t {
    detached,
    committed,
};

struct EnvironmentWorkers {
    std::size_t counter_workers = 1U;
    std::size_t recorder_workers = 1U;
    std::size_t mutation_workers = 1U;
};

struct EnvironmentAdmissionPolicy {
    double r2c_ratio_threshold = 0.0;
    std::size_t virtual_network_size_threshold = 10000U;
};

struct EnvironmentConfig {
    controller::ControllerSelection controller;
    CounterSelection counter;
    RecorderConfig recorder;
    EnvironmentWorkers workers;
    EnvironmentAdmissionPolicy admission;
};

struct EnvironmentResetOptions {
    std::optional<std::uint32_t> seed;
    bool renew_virtual_networks = true;
    bool renew_event_schedule = true;
    network::VirtualSimulationWorkers simulator_workers;
};

struct EnvironmentEventState {
    std::size_t schedule_index = 0U;
    network::VirtualEventId event_id = 0U;
    network::VirtualEventType type = network::VirtualEventType::leave;
    network::VirtualRequestId request_id = 0;
    std::size_t request_index = 0U;
    double time = 0.0;
};

struct EnvironmentState {
    EnvironmentPhase phase = EnvironmentPhase::unready;
    std::size_t num_processed_events = 0U;
    std::optional<EnvironmentEventState> current_event;
};

enum class EnvironmentErrorCode : std::uint8_t {
    empty_event_schedule,
    missing_request_id,
    duplicate_request_id,
    unknown_event_request,
    event_id_overflow,
    event_id_mismatch,
    environment_not_ready,
    environment_finished,
    arrival_event_required,
    leave_event_required,
    solution_request_mismatch,
    incomplete_node_mapping,
    incomplete_link_mapping,
    missing_arrival_record,
    missing_solver_transaction,
};

enum class EnvironmentOperation : std::uint8_t {
    construct,
    reset,
    prepare_schedule,
    prepare_requests,
    ready,
    step,
    release,
    transit,
    access,
};

class EnvironmentException final : public std::runtime_error {
public:
    EnvironmentException(
        EnvironmentErrorCode code,
        EnvironmentOperation operation,
        std::string message,
        std::optional<std::size_t> schedule_index = std::nullopt,
        std::optional<network::VirtualRequestId> request_id = std::nullopt);

    EnvironmentErrorCode code() const noexcept;
    EnvironmentOperation operation() const noexcept;
    const std::optional<std::size_t>& schedule_index() const noexcept;
    const std::optional<network::VirtualRequestId>& request_id() const noexcept;

private:
    EnvironmentErrorCode code_;
    EnvironmentOperation operation_;
    std::optional<std::size_t> schedule_index_;
    std::optional<network::VirtualRequestId> request_id_;
};

struct EnvironmentDrainResult {
    bool done = false;
    std::size_t released_events = 0U;
    std::optional<CounterSummary> summary;
};

struct EnvironmentStepResult {
    EnvironmentStepResult(
        RecorderRecord record_value,
        bool done_value,
        bool accepted_value,
        EnvironmentFailureReason failure_reason_value,
        std::size_t auto_released_events_value,
        std::optional<CounterSummary> summary_value = std::nullopt);

    RecorderRecord record;
    bool done = false;
    bool accepted = false;
    EnvironmentFailureReason failure_reason =
        EnvironmentFailureReason::none;
    std::size_t auto_released_events = 0U;
    std::optional<CounterSummary> summary;
};

struct EnvironmentSolverTransaction {
    network::PhysicalNetwork& physical_network;
    controller::PreparedControllerMutation& mutation;
};

class BaseEnvironment {
public:
    BaseEnvironment(
        network::PhysicalNetwork physical_network,
        network::VirtualNetworkRequestSimulator simulator,
        EnvironmentConfig config);

    ~BaseEnvironment() = default;

    BaseEnvironment(const BaseEnvironment&) = delete;
    BaseEnvironment& operator=(const BaseEnvironment&) = delete;
    BaseEnvironment(BaseEnvironment&&) = delete;
    BaseEnvironment& operator=(BaseEnvironment&&) = delete;

    void reset();
    void renew_and_reset(
        RandomContext& random,
        EnvironmentResetOptions options = {});
    EnvironmentDrainResult drain_leaves();

    const EnvironmentState& state() const noexcept;
    const EnvironmentEventState& current_event() const;
    const network::VirtualNetwork& current_virtual_network() const;
    const Solution& current_solution() const;
    Solution make_solution() const;
    EnvironmentSolverTransaction solver_transaction();
    const network::PhysicalNetwork& physical_network() const noexcept;
    const network::VirtualNetworkRequestSimulator& simulator() const noexcept;
    const Recorder& recorder() const noexcept;
    CounterSummary summary() const;

protected:
    EnvironmentStepResult step_solution(
        Solution& solution,
        EnvironmentSolutionState solution_state);

private:
    void begin_reset();
    void finish_reset();
    void build_event_plan();
    void prepare_requests();
    void ready(std::size_t schedule_index);
    void release_current();
    EnvironmentDrainResult transit_after_arrival();
    EnvironmentFailureReason failure_reason(const Solution& solution) const
        noexcept;
    void apply_failure(
        Solution& solution,
        EnvironmentFailureReason reason) const;
    const EnvironmentEventState& require_current(
        EnvironmentOperation operation) const;
    void require_active(EnvironmentOperation operation) const;

    network::PhysicalNetwork initial_physical_network_;
    network::PhysicalNetwork physical_network_;
    bool physical_network_pristine_ = true;
    network::VirtualNetworkRequestSimulator simulator_;
    EnvironmentConfig config_;
    controller::Controller controller_;
    Counter counter_;
    Recorder recorder_;

    // Both vectors are non-owning prepared views. They are destroyed before
    // either owned network is replaced and indexed directly in event loops.
    std::vector<PreparedCounter> prepared_counters_;
    std::vector<controller::PreparedControllerMutation> prepared_mutations_;
    std::vector<std::size_t> event_request_indices_;
    std::vector<std::optional<std::size_t>> arrival_record_indices_;

    EnvironmentState state_;
    std::optional<Solution> current_solution_;
    std::optional<CounterSummary> completed_summary_;
};

class SolutionStepEnvironment final : public BaseEnvironment {
public:
    using BaseEnvironment::BaseEnvironment;

    EnvironmentStepResult step(
        Solution& solution,
        EnvironmentSolutionState solution_state =
            EnvironmentSolutionState::detached);
};

} // namespace virne::core
