#include "base_system.h"

#include <utility>

namespace virne::system {

SystemException::SystemException(
    SystemErrorCode code,
    SystemOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

SystemErrorCode SystemException::code() const noexcept {
    return code_;
}

SystemOperation SystemException::operation() const noexcept {
    return operation_;
}

SystemStepTrace::SystemStepTrace(
    std::size_t epoch_index_value,
    network::VirtualRequestId request_id_value,
    core::Solution solution_value,
    bool accepted_value,
    core::EnvironmentFailureReason failure_reason_value,
    std::size_t auto_released_events_value,
    double event_time_value,
    std::size_t stage_index_value,
    std::size_t window_index_value)
    : epoch_index(epoch_index_value),
      request_id(request_id_value),
      solution(std::move(solution_value)),
      accepted(accepted_value),
      failure_reason(failure_reason_value),
      auto_released_events(auto_released_events_value),
      event_time(event_time_value),
      stage_index(stage_index_value),
      window_index(window_index_value) {}

BaseSystem::BaseSystem(
    core::SolutionStepEnvironment& environment,
    solver::Solver& solver)
    : environment_(environment), solver_(solver) {}

core::SolutionStepEnvironment& BaseSystem::environment() const noexcept {
    return environment_.get();
}

solver::Solver& BaseSystem::solver_instance() const noexcept {
    return solver_.get();
}

void BaseSystem::ready() {
    solver_.get().ready();
}

} // namespace virne::system
