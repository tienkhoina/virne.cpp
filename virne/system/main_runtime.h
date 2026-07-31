#pragma once

#include "changeable_system.h"
#include "main_config.h"
#include "time_window_system.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace virne::system {

struct RuntimeSelections {
    core::controller::ControllerSelection controller;
    core::CounterSelection counter;
};

RuntimeSelections runtime_selections_from_virtual_config(
    const network::VirtualNetworkSimulationConfig& config,
    bool reusable);

struct MainRunReport {
    SystemMode mode = SystemMode::online;
    solver::SolverId solver_id;
    std::string solver_name;
    std::size_t physical_nodes = 0U;
    std::size_t physical_links = 0U;
    std::size_t virtual_requests = 0U;
    std::size_t scheduled_events = 0U;
    std::uint64_t setup_time_ns = 0U;
    std::uint64_t run_time_ns = 0U;
    SystemRunResult run;
    std::vector<ChangeableStageSummary> stages;
    std::vector<TimeWindowSummary> windows;
};

MainRunReport run_main_config(const MainConfig& config);

void write_main_report_json(
    std::ostream& output,
    const MainRunReport& report);

void write_main_summary_json(
    std::ostream& output,
    const MainRunReport& report);

} // namespace virne::system
