#include "environment.h"

#include "random_context.h"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

namespace virne::core {

EnvironmentException::EnvironmentException(
    EnvironmentErrorCode code,
    EnvironmentOperation operation,
    std::string message,
    std::optional<std::size_t> schedule_index,
    std::optional<network::VirtualRequestId> request_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      schedule_index_(schedule_index),
      request_id_(request_id) {}

EnvironmentErrorCode EnvironmentException::code() const noexcept {
    return code_;
}

EnvironmentOperation EnvironmentException::operation() const noexcept {
    return operation_;
}

const std::optional<std::size_t>&
EnvironmentException::schedule_index() const noexcept {
    return schedule_index_;
}

const std::optional<network::VirtualRequestId>&
EnvironmentException::request_id() const noexcept {
    return request_id_;
}

EnvironmentStepResult::EnvironmentStepResult(
    RecorderRecord record_value,
    bool done_value,
    bool accepted_value,
    EnvironmentFailureReason failure_reason_value,
    std::size_t auto_released_events_value,
    std::optional<CounterSummary> summary_value)
    : record(std::move(record_value)),
      done(done_value),
      accepted(accepted_value),
      failure_reason(failure_reason_value),
      auto_released_events(auto_released_events_value),
      summary(std::move(summary_value)) {}

BaseEnvironment::BaseEnvironment(
    network::PhysicalNetwork physical_network,
    network::VirtualNetworkRequestSimulator simulator,
    EnvironmentConfig config)
    : initial_physical_network_(physical_network.clone()),
      physical_network_(std::move(physical_network)),
      simulator_(std::move(simulator)),
      config_(std::move(config)),
      controller_(config_.controller),
      counter_(config_.counter),
      recorder_(counter_, config_.recorder) {}

void BaseEnvironment::begin_reset() {
    network::PhysicalNetwork reset_physical_network =
        initial_physical_network_.clone();

    state_ = EnvironmentState{};
    current_solution_.reset();
    completed_summary_.reset();
    event_request_indices_.clear();
    arrival_record_indices_.clear();
    prepared_controllers_.clear();
    prepared_counters_.clear();

    physical_network_ = std::move(reset_physical_network);
    recorder_.reset();
    recorder_.count_initial_physical_network(
        physical_network_,
        RecorderOptions{config_.workers.recorder_workers});
}

void BaseEnvironment::finish_reset() {
    build_event_plan();
    prepare_requests();
    ready(0U);
    state_.phase = EnvironmentPhase::active;
}

void BaseEnvironment::reset() {
    begin_reset();
    finish_reset();
}

void BaseEnvironment::renew_and_reset(
    RandomContext& random,
    EnvironmentResetOptions options) {
    begin_reset();
    simulator_.renew(
        random,
        options.renew_virtual_networks,
        options.renew_event_schedule,
        options.seed,
        options.simulator_workers);
    finish_reset();
}

void BaseEnvironment::build_event_plan() {
    const auto& virtual_networks = simulator_.v_nets();
    const auto& events = simulator_.events();
    if (events.empty()) {
        throw EnvironmentException(
            EnvironmentErrorCode::empty_event_schedule,
            EnvironmentOperation::prepare_schedule,
            "environment event schedule is empty");
    }

    std::unordered_map<network::VirtualRequestId, std::size_t> request_indices;
    request_indices.reserve(virtual_networks.size());
    for (std::size_t index = 0U; index < virtual_networks.size(); ++index) {
        const auto& request_id = virtual_networks[index].request_id();
        if (!request_id.has_value()) {
            throw EnvironmentException(
                EnvironmentErrorCode::missing_request_id,
                EnvironmentOperation::prepare_schedule,
                "virtual network has no request ID",
                std::nullopt,
                std::nullopt);
        }
        const auto inserted = request_indices.emplace(*request_id, index);
        if (!inserted.second) {
            throw EnvironmentException(
                EnvironmentErrorCode::duplicate_request_id,
                EnvironmentOperation::prepare_schedule,
                "virtual network request ID is duplicated",
                std::nullopt,
                *request_id);
        }
    }

    event_request_indices_.clear();
    event_request_indices_.reserve(events.size());
    for (std::size_t index = 0U; index < events.size(); ++index) {
        const auto& event = events[index];
        if (static_cast<std::uintmax_t>(event.id()) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw EnvironmentException(
                EnvironmentErrorCode::event_id_overflow,
                EnvironmentOperation::prepare_schedule,
                "event ID cannot be represented by Recorder",
                index,
                event.virtual_network_id());
        }
        if (event.id() != index) {
            throw EnvironmentException(
                EnvironmentErrorCode::event_id_mismatch,
                EnvironmentOperation::prepare_schedule,
                "event ID must equal its dense schedule index",
                index,
                event.virtual_network_id());
        }
        const auto found = request_indices.find(event.virtual_network_id());
        if (found == request_indices.end()) {
            throw EnvironmentException(
                EnvironmentErrorCode::unknown_event_request,
                EnvironmentOperation::prepare_schedule,
                "event references an unknown virtual network request",
                index,
                event.virtual_network_id());
        }
        event_request_indices_.push_back(found->second);
    }
}

void BaseEnvironment::prepare_requests() {
    const auto& virtual_networks = simulator_.v_nets();
    prepared_counters_.clear();
    prepared_controllers_.clear();
    prepared_counters_.reserve(virtual_networks.size());
    prepared_controllers_.reserve(virtual_networks.size());
    arrival_record_indices_.assign(virtual_networks.size(), std::nullopt);

    for (const auto& virtual_network : virtual_networks) {
        prepared_counters_.push_back(counter_.prepare(virtual_network));
        prepared_controllers_.push_back(
            controller_.prepare(virtual_network, physical_network_));
    }
}

void BaseEnvironment::ready(std::size_t schedule_index) {
    const auto& event = simulator_.events().at(schedule_index);
    const std::size_t request_index =
        event_request_indices_.at(schedule_index);

    state_.current_event = EnvironmentEventState{
        schedule_index,
        event.id(),
        event.type(),
        event.virtual_network_id(),
        request_index,
        event.time(),
    };
    ++state_.num_processed_events;
    current_solution_ =
        Solution::from_v_net(simulator_.v_nets().at(request_index));
    recorder_.set_event(RecorderEvent{
        static_cast<std::int64_t>(event.id()),
        event.type(),
    });
}

const EnvironmentState& BaseEnvironment::state() const noexcept {
    return state_;
}

const EnvironmentEventState& BaseEnvironment::require_current(
    EnvironmentOperation operation) const {
    if (!state_.current_event.has_value()) {
        throw EnvironmentException(
            EnvironmentErrorCode::environment_not_ready,
            operation,
            "environment has no current event");
    }
    return *state_.current_event;
}

void BaseEnvironment::require_active(EnvironmentOperation operation) const {
    if (state_.phase == EnvironmentPhase::unready) {
        throw EnvironmentException(
            EnvironmentErrorCode::environment_not_ready,
            operation,
            "environment must be reset before use");
    }
    if (state_.phase == EnvironmentPhase::finished) {
        throw EnvironmentException(
            EnvironmentErrorCode::environment_finished,
            operation,
            "environment episode is finished");
    }
}

const EnvironmentEventState& BaseEnvironment::current_event() const {
    return require_current(EnvironmentOperation::access);
}

const network::VirtualNetwork&
BaseEnvironment::current_virtual_network() const {
    const auto& event = require_current(EnvironmentOperation::access);
    return simulator_.v_nets().at(event.request_index);
}

const Solution& BaseEnvironment::current_solution() const {
    if (!current_solution_.has_value()) {
        throw EnvironmentException(
            EnvironmentErrorCode::environment_not_ready,
            EnvironmentOperation::access,
            "environment has no current Solution");
    }
    return *current_solution_;
}

Solution BaseEnvironment::make_solution() const {
    return Solution::from_v_net(current_virtual_network());
}

const network::PhysicalNetwork&
BaseEnvironment::physical_network() const noexcept {
    return physical_network_;
}

const network::VirtualNetworkRequestSimulator&
BaseEnvironment::simulator() const noexcept {
    return simulator_;
}

const Recorder& BaseEnvironment::recorder() const noexcept {
    return recorder_;
}

CounterSummary BaseEnvironment::summary() const {
    if (completed_summary_.has_value()) {
        return *completed_summary_;
    }
    return recorder_.summary_records();
}

EnvironmentFailureReason BaseEnvironment::failure_reason(
    const Solution& solution) const noexcept {
    if (solution.early_rejection) {
        return EnvironmentFailureReason::early_rejection;
    }
    if (!solution.place_result) {
        return EnvironmentFailureReason::placement;
    }
    if (!solution.route_result) {
        return EnvironmentFailureReason::routing;
    }
    return EnvironmentFailureReason::unknown;
}

void BaseEnvironment::apply_failure(
    Solution& solution,
    EnvironmentFailureReason reason) const {
    switch (reason) {
    case EnvironmentFailureReason::early_rejection:
        solution.description = "Early Rejection";
        solution.early_rejection = true;
        return;
    case EnvironmentFailureReason::placement:
        solution.description = "Place Failure";
        solution.place_result = false;
        return;
    case EnvironmentFailureReason::routing:
        solution.description = "Route Failure";
        solution.route_result = false;
        return;
    case EnvironmentFailureReason::unknown:
        solution.description = "Unknown Reason";
        return;
    case EnvironmentFailureReason::none:
        return;
    }
}

RecorderRecord BaseEnvironment::release_current() {
    require_active(EnvironmentOperation::release);
    const auto& event = require_current(EnvironmentOperation::release);
    if (event.type != network::VirtualEventType::leave) {
        throw EnvironmentException(
            EnvironmentErrorCode::leave_event_required,
            EnvironmentOperation::release,
            "current event is not a leave event",
            event.schedule_index,
            event.request_id);
    }

    const auto& arrival_index = arrival_record_indices_.at(event.request_index);
    if (!arrival_index.has_value() ||
        *arrival_index >= recorder_.memory().size()) {
        throw EnvironmentException(
            EnvironmentErrorCode::missing_arrival_record,
            EnvironmentOperation::release,
            "leave event has no committed arrival record",
            event.schedule_index,
            event.request_id);
    }
    const RecorderRecord& arrival = recorder_.memory().at(*arrival_index);
    (void)prepared_controllers_.at(event.request_index)
        .release(
            arrival.solution,
            controller::ControllerMutationOptions{
                config_.workers.mutation_workers});

    Solution& leave_solution = *current_solution_;
    leave_solution.description = "Leave Event";
    RecorderRecord record = recorder_.count_prepared(
        prepared_counters_.at(event.request_index),
        physical_network_,
        leave_solution,
        RecorderOptions{config_.workers.recorder_workers});
    (void)recorder_.add_record(record);
    return record;
}

EnvironmentDrainResult BaseEnvironment::drain_leaves() {
    require_active(EnvironmentOperation::transit);
    if (current_event().type != network::VirtualEventType::leave) {
        throw EnvironmentException(
            EnvironmentErrorCode::leave_event_required,
            EnvironmentOperation::transit,
            "current event is not a leave event",
            current_event().schedule_index,
            current_event().request_id);
    }

    EnvironmentDrainResult result;
    while (current_event().type == network::VirtualEventType::leave) {
        (void)release_current();
        ++result.released_events;

        const std::size_t schedule_index = current_event().schedule_index;
        const std::size_t event_count = simulator_.events().size();
        if (schedule_index >= event_count - 1U) {
            completed_summary_ = recorder_.summary_records();
            state_.phase = EnvironmentPhase::finished;
            result.done = true;
            result.summary = completed_summary_;
            return result;
        }
        ready(schedule_index + 1U);
    }
    return result;
}

EnvironmentDrainResult BaseEnvironment::transit_after_arrival() {
    const auto& current = require_current(EnvironmentOperation::transit);
    const std::size_t event_count = simulator_.events().size();
    if (current.schedule_index >= event_count - 1U) {
        completed_summary_ = recorder_.summary_records();
        state_.phase = EnvironmentPhase::finished;
        return EnvironmentDrainResult{true, 0U, completed_summary_};
    }

    ready(current.schedule_index + 1U);
    if (current_event().type == network::VirtualEventType::arrival) {
        return EnvironmentDrainResult{};
    }
    return drain_leaves();
}

EnvironmentStepResult BaseEnvironment::step_solution(Solution& solution) {
    require_active(EnvironmentOperation::step);
    const auto& event = require_current(EnvironmentOperation::step);
    if (event.type != network::VirtualEventType::arrival) {
        throw EnvironmentException(
            EnvironmentErrorCode::arrival_event_required,
            EnvironmentOperation::step,
            "current event is not an arrival event",
            event.schedule_index,
            event.request_id);
    }
    if (solution.v_net_id != event.request_id) {
        throw EnvironmentException(
            EnvironmentErrorCode::solution_request_mismatch,
            EnvironmentOperation::step,
            "Solution request ID does not match the current event",
            event.schedule_index,
            event.request_id);
    }

    current_solution_ = solution;
    EnvironmentFailureReason reason = EnvironmentFailureReason::none;
    RecorderRecord record = [&]() -> RecorderRecord {
      try {
        const auto& virtual_network =
            simulator_.v_nets().at(event.request_index);
        if (solution.result &&
            solution.v_net_total_hard_constraint_violation > 0.0) {
            solution.result = false;
        }
        if (solution.result &&
            solution.v_net_r2c_ratio <
                config_.admission.r2c_ratio_threshold &&
            virtual_network.num_nodes() >
                config_.admission.virtual_network_size_threshold) {
            solution.result = false;
            solution.description = "r2c_ratio < threshold";
        }

        prepared_counters_.at(event.request_index)
            .count_solution(
                solution,
                CounterOptions{config_.workers.counter_workers});

        if (solution.result) {
            if (solution.node_slots.size() != virtual_network.num_nodes()) {
                throw EnvironmentException(
                    EnvironmentErrorCode::incomplete_node_mapping,
                    EnvironmentOperation::step,
                    "successful Solution does not map every virtual node",
                    event.schedule_index,
                    event.request_id);
            }
            if (solution.link_paths.size() != virtual_network.num_links()) {
                throw EnvironmentException(
                    EnvironmentErrorCode::incomplete_link_mapping,
                    EnvironmentOperation::step,
                    "successful Solution does not route every virtual link",
                    event.schedule_index,
                    event.request_id);
            }
            solution.description = "Success";
            (void)prepared_controllers_.at(event.request_index)
                .deploy(
                    solution,
                    controller::ControllerMutationOptions{
                        config_.workers.mutation_workers});
        } else {
            reason = failure_reason(solution);
            apply_failure(solution, reason);
        }

        RecorderRecord arrival_record = recorder_.count_prepared(
            prepared_counters_.at(event.request_index),
            physical_network_,
            solution,
            RecorderOptions{config_.workers.recorder_workers});
        const std::size_t record_index = recorder_.memory().size();
        (void)recorder_.add_record(arrival_record);
        arrival_record_indices_.at(event.request_index) = record_index;
        current_solution_ = solution;
        return arrival_record;
      } catch (...) {
        current_solution_ = solution;
        throw;
      }
    }();

    const EnvironmentDrainResult transit = transit_after_arrival();
    return EnvironmentStepResult(
        std::move(record),
        transit.done,
        solution.result,
        reason,
        transit.released_events,
        completed_summary_);
}

EnvironmentStepResult SolutionStepEnvironment::step(Solution& solution) {
    return step_solution(solution);
}

} // namespace virne::core
