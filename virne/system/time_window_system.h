#pragma once

#include "base_system.h"

namespace virne::system {

struct TimeWindowRunConfig {
    SystemRunConfig system;
    double window_size = 100.0;
    double window_origin = 0.0;
};

struct TimeWindowSummary {
    std::size_t epoch_index = 0U;
    std::size_t window_index = 0U;
    double first_event_time = 0.0;
    double last_event_time = 0.0;
    std::size_t arrival_steps = 0U;
    std::size_t accepted = 0U;
    std::size_t rejected = 0U;
    std::size_t auto_released_events = 0U;
};

struct TimeWindowRunResult {
    SystemRunResult run;
    std::vector<TimeWindowSummary> windows;
};

class TimeWindowSystem final : public BaseSystem {
public:
    using BaseSystem::BaseSystem;

    TimeWindowRunResult run(
        RandomContext& random,
        const TimeWindowRunConfig& config = {});
};

} // namespace virne::system
