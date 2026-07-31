#pragma once

#include "base_system.h"

#include "../core/logger.h"
#include "../network/physical_network.h"
#include "../solver/heuristic/bfs_trials.h"
#include "../utils/setting.h"
#include "config/config.h"
#include "py_random.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace virne::system {

enum class MainConfigErrorCode : std::uint8_t {
    invalid_seed,
    invalid_matching_method,
    invalid_shortest_path_method,
    invalid_logger_level,
    invalid_logger_backend,
    invalid_changeable_stage_count,
    invalid_report_mode,
    invalid_progress_width,
};

enum class MainReportMode : std::uint8_t {
    summary,
    full,
    none,
};

struct MainProgressConfig {
    bool enabled = true;
    std::size_t width = 30U;
    std::uint64_t minimum_interval_ms = 100U;
};

class MainConfigException final : public std::runtime_error {
public:
    MainConfigException(MainConfigErrorCode code, std::string message);
    MainConfigErrorCode code() const noexcept;

private:
    MainConfigErrorCode code_;
};

struct MainRuntimeWorkers {
    network::PhysicalNetworkBuildOptions physical;
    network::VirtualSimulationWorkers simulator;
    core::EnvironmentWorkers environment;
    solver::heuristic::NodeRankSolverWorkers node_rank;
};

struct MainConfig {
    std::optional<std::uint32_t> seed;
    std::size_t num_simulations = 1U;
    SystemMode mode = SystemMode::online;
    bool renew_virtual_networks = false;
    bool renew_event_schedule = false;
    bool capture_solutions = false;
    MainReportMode report_mode = MainReportMode::summary;
    MainProgressConfig progress;
    std::size_t changeable_stage_count = 1U;
    double time_window_size = 100.0;
    double time_window_origin = 0.0;

    // This string exists only at the cold configuration boundary. The runtime
    // resolves it exactly once to SolverId before any request loop.
    std::string solver_name;
    solver::SolverConfig solver_config;
    solver::heuristic::BfsSolverParameters bfs_solver_parameters;
    core::RecorderConfig recorder_config;
    std::string records_file_name = "records.csv";
    std::string summary_file_name = "summary.csv";
    core::LoggerConfig logger_config;

    // Full resolved Hydra-compatible tree, serialized once at the cold config
    // boundary when INFO has a standard backend. Hydra's internal subtree and
    // the native C++ extension are kept out so this view matches the Python
    // application's resolved DictConfig exactly. Runtime logging reads these
    // fixed strings directly and never traverses YAML or performs string
    // lookup in a request loop. They stay empty when Logger suppresses INFO.
    std::string resolved_config_yaml;
    std::string resolved_native_config_yaml;
    core::EnvironmentAdmissionPolicy admission;
    bool save_records = true;
    bool save_config = true;
    bool save_physical_network = false;
    bool save_virtual_networks = false;
    bool load_physical_network = false;
    bool load_virtual_networks = false;

    std::filesystem::path physical_dataset_dir;
    std::string physical_dataset_file = "p_net.gml";
    std::filesystem::path virtual_dataset_dir;

    virne::utils::SettingDocument physical_setting;
    virne::utils::SettingDocument virtual_setting;
    MainRuntimeWorkers workers;
};

MainConfig main_config_from_hydra(Config& config, PyRandom& random);

// Returns the resolved Python application view: Hydra internals and the
// native C++ extension are separate control planes and are excluded.
YAML::Node python_compatible_config_root(const Config& config);

std::string_view system_mode_name(SystemMode mode) noexcept;
std::string_view main_report_mode_name(MainReportMode mode) noexcept;

} // namespace virne::system
