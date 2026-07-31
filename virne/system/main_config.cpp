#include "main_config.h"

#include "../network/attribute/attribute_factory.h"
#include "../network/dataset_generator.h"
#include "../utils/utils_config.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>
#include <vector>

namespace virne::system {
namespace {

std::string lowercase(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char byte) {
            return static_cast<char>(std::tolower(byte));
        });
    return value;
}

std::optional<std::uint32_t> read_seed(const Config& config) {
    const YAML::Node node = config.get_raw("experiment.seed");
    if (!node.IsDefined() || node.IsNull()) {
        return std::nullopt;
    }
    const std::int64_t value = node.as<std::int64_t>();
    if (value < 0 ||
        static_cast<std::uint64_t>(value) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_seed,
            "experiment.seed must fit the unsigned 32-bit RNG lane");
    }
    return static_cast<std::uint32_t>(value);
}

core::controller::NodeMatchingMethod matching_method_from_config(
    std::string value) {
    value = lowercase(std::move(value));
    if (value == "greedy") {
        return core::controller::NodeMatchingMethod::greedy;
    }
    if (value == "l2s2") {
        return core::controller::NodeMatchingMethod::l2s2;
    }
    throw MainConfigException(
        MainConfigErrorCode::invalid_matching_method,
        "unknown solver.matching_mathod value");
}

core::controller::ShortestPathMethod shortest_method_from_config(
    std::string value) {
    value = lowercase(std::move(value));
    if (value == "first_shortest") {
        return core::controller::ShortestPathMethod::first_shortest;
    }
    if (value == "k_shortest") {
        return core::controller::ShortestPathMethod::k_shortest;
    }
    if (value == "k_shortest_length") {
        return core::controller::ShortestPathMethod::k_shortest_length;
    }
    if (value == "all_shortest") {
        return core::controller::ShortestPathMethod::all_shortest;
    }
    if (value == "bfs_shortest") {
        return core::controller::ShortestPathMethod::bfs_shortest;
    }
    if (value == "available_shortest") {
        return core::controller::ShortestPathMethod::available_shortest;
    }
    throw MainConfigException(
        MainConfigErrorCode::invalid_shortest_path_method,
        "unknown solver.shortest_method value");
}

core::LoggerLevel logger_level_from_config(std::string value) {
    value = lowercase(std::move(value));
    if (value == "debug") {
        return core::LoggerLevel::debug;
    }
    if (value == "info") {
        return core::LoggerLevel::info;
    }
    if (value == "warning" || value == "warn") {
        return core::LoggerLevel::warning;
    }
    if (value == "error") {
        return core::LoggerLevel::error;
    }
    if (value == "critical") {
        return core::LoggerLevel::critical;
    }
    throw MainConfigException(
        MainConfigErrorCode::invalid_logger_level,
        "unknown logger.level value");
}

core::LoggerBackends logger_backends_from_config(const Config& config) {
    core::LoggerBackends result;
    result.console = false;
    result.file = false;
    const YAML::Node node = config.get_raw("logger.backends");
    if (!node.IsDefined() || node.IsNull()) {
        return result;
    }
    if (!node.IsSequence()) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_logger_backend,
            "logger.backends must be a list");
    }
    for (const auto& item : node) {
        const std::string name = lowercase(item.as<std::string>());
        if (name == "console") {
            result.console = true;
        } else if (name == "file") {
            result.file = true;
        } else if (name == "tensorboard" || name == "wandb") {
            // Explicit non-ML boundary: these Python backends are ignored.
        } else {
            throw MainConfigException(
                MainConfigErrorCode::invalid_logger_backend,
                "unknown non-ML logger backend");
        }
    }
    return result;
}

MainReportMode report_mode_from_config(std::string value) {
    value = lowercase(std::move(value));
    if (value == "summary") {
        return MainReportMode::summary;
    }
    if (value == "full") {
        return MainReportMode::full;
    }
    if (value == "none") {
        return MainReportMode::none;
    }
    throw MainConfigException(
        MainConfigErrorCode::invalid_report_mode,
        "native.output.report must be summary, full or none");
}

virne::utils::SettingDocument subtree_setting(
    const Config& config,
    const std::string& path) {
    return virne::utils::parse_setting(
        YAML::Dump(config.get_raw(path)),
        virne::utils::SettingFormat::yaml);
}

using AttributeKind = virne::network::attribute::AttributeKind;
using SettingDocument = virne::utils::SettingDocument;
using SettingKeyId = virne::utils::SettingKeyId;
using SettingList = virne::utils::SettingList;
using SettingObject = virne::utils::SettingObject;
using SettingValue = virne::utils::SettingValue;
using SettingValueKind = virne::utils::SettingValueKind;

const SettingObject& setting_root(const SettingDocument& setting) {
    if (setting.root.kind() != SettingValueKind::object) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_report_mode,
            "network setting root must be a mapping");
    }
    return setting.root.as_object();
}

const SettingList& setting_attribute_list(
    const SettingDocument& setting,
    const std::string_view key) {
    const SettingObject& root = setting_root(setting);
    const std::optional<SettingKeyId> id = root.find_key_id(key);
    if (!id.has_value() || root.at(*id).kind() != SettingValueKind::list) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_report_mode,
            "network attribute setting must be a list");
    }
    return root.at(*id).as_list();
}

std::vector<AttributeKind> attribute_kinds_from_setting(
    const SettingDocument& setting,
    const std::string_view key) {
    const SettingList& values = setting_attribute_list(setting, key);
    std::vector<AttributeKind> result;
    result.reserve(values.size());
    for (const SettingValue& value : values) {
        if (value.kind() != SettingValueKind::object) {
            throw MainConfigException(
                MainConfigErrorCode::invalid_report_mode,
                "network attribute setting item must be a mapping");
        }
        result.push_back(
            virne::network::attribute::attribute_factory_spec_from_setting(
                value.as_object())
                .kind);
    }
    return result;
}

std::optional<std::size_t> extracted_attribute_index(
    const std::string_view name) noexcept {
    if (name == "resource") {
        return 0U;
    }
    if (name == "extrema") {
        return 1U;
    }
    if (name == "status") {
        return 2U;
    }
    if (name == "position") {
        return 3U;
    }
    if (name == "latency") {
        return 4U;
    }
    return std::nullopt;
}

virne::utils::ExtractedAttributeKinds extracted_attribute_kinds(
    const Config& config) {
    virne::utils::ExtractedAttributeKinds result;
    const YAML::Node configured = config.get_raw(
        "rl.feature_constructor.extracted_attr_types");
    if (!configured.IsDefined() || configured.IsNull()) {
        return result;
    }
    if (!configured.IsSequence()) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_report_mode,
            "rl.feature_constructor.extracted_attr_types must be a list");
    }
    for (const YAML::Node& item : configured) {
        const auto index = extracted_attribute_index(item.as<std::string>());
        if (index.has_value()) {
            result.included[*index] = true;
        }
    }
    return result;
}

void set_simulation_derived_field(
    Config& config,
    const std::string_view path,
    const std::int64_t value) {
    config.set(std::string(path), YAML::Node(value));
}

void set_simulation_derived_field(
    Config& config,
    const std::string_view path,
    const std::size_t value) {
    config.set(
        std::string(path),
        YAML::Node(static_cast<std::uint64_t>(value)));
}

void set_simulation_derived_field(
    Config& config,
    const std::string_view path,
    const std::filesystem::path& value) {
    config.set(
        std::string(path),
        YAML::Node(value.generic_string()));
}

void add_python_simulation_fields(
    Config& config,
    const SettingDocument& physical_setting,
    const SettingDocument& virtual_setting,
    const std::optional<std::uint32_t>& seed) {
    const auto virtual_config =
        virne::network::virtual_network_simulation_config_from_setting(
            virtual_setting);

    virne::utils::SimulationConfigInput input;
    input.physical_dataset =
        virne::network::physical_dataset_setting_from_setting(
            physical_setting);
    input.virtual_dataset =
        virne::network::virtual_dataset_setting_from_config(virtual_config);
    input.physical_node_attributes.kinds = attribute_kinds_from_setting(
        physical_setting,
        "node_attrs_setting");
    input.physical_link_attributes.kinds = attribute_kinds_from_setting(
        physical_setting,
        "link_attrs_setting");
    input.virtual_node_attributes.kinds.reserve(
        virtual_config.node_attribute_specs.size());
    for (const auto& spec : virtual_config.node_attribute_specs) {
        input.virtual_node_attributes.kinds.push_back(spec.kind);
    }
    input.virtual_link_attributes.kinds.reserve(
        virtual_config.link_attribute_specs.size());
    for (const auto& spec : virtual_config.link_attribute_specs) {
        input.virtual_link_attributes.kinds.push_back(spec.kind);
    }
    input.extracted_attribute_kinds = extracted_attribute_kinds(config);
    if (seed.has_value()) {
        input.seed = virne::utils::DatasetScalar{
            static_cast<std::int64_t>(*seed)};
    }

    const virne::utils::SimulationConfigSummary summary =
        virne::utils::derive_simulation_config(input);
    set_simulation_derived_field(
        config,
        "simulation.p_net_dataset_dir",
        summary.p_net_dataset_dir);
    set_simulation_derived_field(
        config,
        "simulation.v_nets_dataset_dir",
        summary.v_nets_dataset_dir);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_nodes",
        summary.p_net_setting_num_nodes);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_node_attrs",
        summary.p_net_setting_num_node_attrs);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_link_attrs",
        summary.p_net_setting_num_link_attrs);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_node_resource_attrs",
        summary.p_net_setting_num_node_resource_attrs);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_link_resource_attrs",
        summary.p_net_setting_num_link_resource_attrs);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_node_extrema_attrs",
        summary.p_net_setting_num_node_extrema_attrs);
    set_simulation_derived_field(
        config,
        "simulation.p_net_setting_num_link_extrema_attrs",
        summary.p_net_setting_num_link_extrema_attrs);
    set_simulation_derived_field(
        config,
        "simulation.v_sim_setting_num_node_attrs",
        summary.v_sim_setting_num_node_attrs);
    set_simulation_derived_field(
        config,
        "simulation.v_sim_setting_num_link_attrs",
        summary.v_sim_setting_num_link_attrs);
    set_simulation_derived_field(
        config,
        "simulation.v_sim_setting_num_node_resource_attrs",
        summary.v_sim_setting_num_node_resource_attrs);
    set_simulation_derived_field(
        config,
        "simulation.v_sim_setting_num_link_resource_attrs",
        summary.v_sim_setting_num_link_resource_attrs);
    set_simulation_derived_field(
        config,
        "simulation.v_sim_setting_num_node_non_status_attrs",
        summary.v_sim_setting_num_node_non_status_attrs);
    set_simulation_derived_field(
        config,
        "simulation.v_sim_setting_num_link_non_status_attrs",
        summary.v_sim_setting_num_link_non_status_attrs);
    set_simulation_derived_field(
        config,
        "simulation.p_net_num_nodes",
        summary.p_net_setting_num_nodes);
    set_simulation_derived_field(
        config,
        "rl.feature_constructor.num_extracted_p_node_attrs",
        summary.feature_constructor.num_extracted_p_node_attrs);
    set_simulation_derived_field(
        config,
        "rl.feature_constructor.num_extracted_p_link_attrs",
        summary.feature_constructor.num_extracted_p_link_attrs);
    set_simulation_derived_field(
        config,
        "rl.feature_constructor.num_extracted_v_node_attrs",
        summary.feature_constructor.num_extracted_v_node_attrs);
    set_simulation_derived_field(
        config,
        "rl.feature_constructor.num_extracted_v_link_attrs",
        summary.feature_constructor.num_extracted_v_link_attrs);
    set_simulation_derived_field(
        config,
        "rl.feature_constructor.p_num_nodes",
        summary.feature_constructor.p_num_nodes);
}

SystemMode mode_from_config(const Config& config) {
    if (config.get<bool>("system.if_changeable_v_nets", false)) {
        return SystemMode::changeable;
    }
    if (config.get<bool>("system.if_offline_system", false)) {
        return SystemMode::offline;
    }
    if (config.get<bool>("system.if_time_window", false)) {
        return SystemMode::time_window;
    }
    return SystemMode::online;
}

} // namespace

MainConfigException::MainConfigException(
    MainConfigErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

MainConfigErrorCode MainConfigException::code() const noexcept {
    return code_;
}

YAML::Node python_compatible_config_root(const Config& config) {
    YAML::Node result = config.root();
    if (result.IsMap()) {
        result.remove("hydra");
        result.remove("native");
    }
    return result;
}

MainConfig main_config_from_hydra(
    Config& config,
    PyRandom& random) {
    // Python replaces "auto" on the composed config before any typed decode
    // or resolved-tree log. Mutate the caller's cold Config explicitly as the
    // public API contract; yaml-cpp nodes share storage, so a nominal Config
    // copy would only disguise the same mutation behind a const reference.
    std::string run_id = config.get<std::string>(
        "experiment.run_id",
        "auto");
    if (run_id == "auto") {
        run_id = virne::utils::generate_run_id(random);
        config.set("experiment.run_id", YAML::Node(run_id));
    }

    MainConfig result;
    result.seed = read_seed(config);
    result.num_simulations =
        config.get<std::size_t>("experiment.num_simulations", 1U);
    result.mode = mode_from_config(config);
    result.renew_virtual_networks =
        config.get<bool>("system.renew_v_net_simulator", false);
    result.renew_event_schedule = result.renew_virtual_networks;
    result.report_mode = report_mode_from_config(
        config.get<std::string>("native.output.report", "summary"));
    result.capture_solutions =
        config.get<bool>(
            "native.capture_solutions",
            result.report_mode == MainReportMode::full);
    result.progress.enabled =
        config.get<bool>("native.progress.enabled", true);
    result.progress.width =
        config.get<std::size_t>("native.progress.width", 30U);
    if (result.progress.width == 0U || result.progress.width > 200U) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_progress_width,
            "native.progress.width must be between 1 and 200");
    }
    result.progress.minimum_interval_ms = config.get<std::uint64_t>(
        "native.progress.minimum_interval_ms", 100U);
    result.changeable_stage_count =
        config.get<std::size_t>("native.changeable_stage_count", 1U);
    if (result.changeable_stage_count == 0U) {
        throw MainConfigException(
            MainConfigErrorCode::invalid_changeable_stage_count,
            "native.changeable_stage_count must be positive");
    }
    result.time_window_size =
        config.get<double>("system.time_window_size", 100.0);
    result.time_window_origin =
        config.get<double>("native.time_window_origin", 0.0);

    result.solver_name = config.get<std::string>("solver.solver_name");
    const std::filesystem::path save_root =
        config.get<std::string>("experiment.save_root_dir", "virne/");

    solver::SolverConfigInput solver_input;
    solver_input.seed = result.seed;
    solver_input.verbose = config.get<int>("logger.verbose", 1);
    solver_input.run_directory = {
        save_root,
        result.solver_name,
        run_id,
    };
    solver_input.reusable = config.get<bool>("solver.reusable", false);
    solver_input.node_ranking_method = solver::rank::node_rank_method_from_string(
        config.get<std::string>("solver.node_ranking_method", "order"));
    solver_input.link_ranking_method = solver::rank::link_rank_method_from_string(
        config.get<std::string>("solver.link_ranking_method", "order"));
    solver_input.matching_method = matching_method_from_config(
        config.get<std::string>("solver.matching_mathod", "greedy"));
    solver_input.shortest_method = shortest_method_from_config(
        config.get<std::string>("solver.shortest_method", "k_shortest"));
    solver_input.k_shortest =
        config.get<std::int64_t>("solver.k_shortest", 10);
    solver_input.allow_rejection =
        config.get<bool>("solver.allow_rejection", false);
    solver_input.allow_revocable =
        config.get<bool>("solver.allow_revocable", false);
    result.solver_config = solver::make_solver_config(solver_input);
    // Python BaseSystem constructs BFS solvers without forwarding
    // config.solver as **kwargs. BfsSolver therefore overwrites the inherited
    // SolverConfig values with its own 50/5/bfs_shortest/10 defaults. Keep
    // the Python-compatible plane exact; explicit native-only overrides live
    // under native.bfs and are excluded from the logged/saved Python tree.
    result.bfs_solver_parameters.max_visit =
        config.get<std::int64_t>("native.bfs.max_visit", 50);
    result.bfs_solver_parameters.max_depth =
        config.get<std::int64_t>("native.bfs.max_depth", 5);
    result.bfs_solver_parameters.shortest_method = shortest_method_from_config(
        config.get<std::string>(
            "native.bfs.shortest_method", "bfs_shortest"));
    result.bfs_solver_parameters.k_shortest =
        config.get<std::int64_t>("native.bfs.k_shortest", 10);

    result.recorder_config.save_root_dir = save_root;
    result.recorder_config.solver_name = result.solver_name;
    result.recorder_config.run_id = run_id;
    result.recorder_config.record_dir_name =
        config.get<std::string>("recorder.record_dir_name", "records");
    result.recorder_config.temporary_records =
        config.get<bool>("recorder.if_temp_save_records", true);
    result.records_file_name = config.get<std::string>(
        "recorder.records_file_name", "records.csv");
    result.summary_file_name = config.get<std::string>(
        "recorder.summary_file_name", "summary.csv");
    result.save_records =
        config.get<bool>("recorder.if_save_records", true);

    result.logger_config.save_root_dir = save_root;
    result.logger_config.solver_name = result.solver_name;
    result.logger_config.run_id = run_id;
    result.logger_config.log_dir_name =
        config.get<std::string>("logger.log_dir_name", "logs");
    result.logger_config.log_file_name =
        config.get<std::string>("logger.log_file_name", "running.log");
    result.logger_config.backends = logger_backends_from_config(config);
    result.logger_config.level = logger_level_from_config(
        config.get<std::string>("logger.level", "INFO"));
    result.logger_config.log_show_interval =
        config.get<std::size_t>("logger.log_show_interval", 20U);
    result.logger_config.project_name =
        config.get<std::string>("logger.project_name", "virne");
    result.logger_config.experiment_name =
        config.get<std::string>(
            "logger.experiment_name", "default_experiment");

    result.save_config =
        config.get<bool>("experiment.if_save_config", true);
    result.save_physical_network =
        config.get<bool>("experiment.if_save_p_net", false);
    result.save_virtual_networks =
        config.get<bool>("experiment.if_save_v_nets", false);
    result.load_physical_network =
        config.get<bool>("experiment.if_load_p_net", false);
    result.load_virtual_networks =
        config.get<bool>("experiment.if_load_v_nets", false);
    result.physical_dataset_dir = config.get<std::string>(
        "p_net_setting.output.save_dir", "dataset/p_net");
    result.physical_dataset_file = config.get<std::string>(
        "p_net_setting.output.file_name", "p_net.gml");
    result.virtual_dataset_dir = config.get<std::string>(
        "v_sim_setting.output.save_dir", "dataset/v_nets");
    result.admission.r2c_ratio_threshold = config.get<double>(
        "native.admission.r2c_ratio_threshold", 0.0);
    result.admission.virtual_network_size_threshold =
        config.get<std::size_t>(
            "native.admission.virtual_network_size_threshold", 10000U);

    result.physical_setting = subtree_setting(config, "p_net_setting");
    result.virtual_setting = subtree_setting(config, "v_sim_setting");
    // Hydra's Python entry point mutates these derived fields before
    // BaseSystem construction.  Resolve them once at the same cold boundary
    // so the saved/logged tree and all dataset paths are identical; the
    // runtime keeps only the typed summaries above.
    add_python_simulation_fields(
        config,
        result.physical_setting,
        result.virtual_setting,
        result.seed);

    result.workers.physical.factory_workers =
        config.get<std::size_t>("native.workers.factory", 1U);
    result.workers.physical.attribute_workers =
        config.get<std::size_t>("native.workers.attribute", 1U);
    result.workers.simulator.factory_workers =
        result.workers.physical.factory_workers;
    result.workers.simulator.arrangement_workers =
        config.get<std::size_t>("native.workers.arrangement", 1U);
    result.workers.simulator.attribute_workers =
        result.workers.physical.attribute_workers;
    result.workers.simulator.event_workers =
        config.get<std::size_t>("native.workers.event", 1U);
    result.workers.simulator.io_workers =
        config.get<std::size_t>("native.workers.io", 1U);
    result.workers.environment.counter_workers =
        config.get<std::size_t>("native.workers.counter", 1U);
    result.workers.environment.recorder_workers =
        config.get<std::size_t>("native.workers.recorder", 1U);
    result.workers.environment.mutation_workers =
        config.get<std::size_t>("native.workers.mutation", 1U);
    result.workers.node_rank.rank_workers =
        config.get<std::size_t>("native.workers.rank", 1U);
    result.workers.node_rank.node_candidate_workers =
        config.get<std::size_t>("native.workers.node_candidate", 1U);
    result.workers.node_rank.link_topology_constraint_workers =
        config.get<std::size_t>(
            "native.workers.link_topology_constraint", 1U);
    result.workers.node_rank.link_candidate_workers =
        config.get<std::size_t>("native.workers.link_candidate", 1U);

    const bool info_enabled =
        result.logger_config.level == core::LoggerLevel::debug ||
        result.logger_config.level == core::LoggerLevel::info;
    const bool has_standard_backend =
        result.logger_config.backends.console ||
        result.logger_config.backends.file;
    if (info_enabled && has_standard_backend) {
        result.resolved_config_yaml = YAML::Dump(
            python_compatible_config_root(config));
        const YAML::Node native = config.get_raw("native");
        if (native.IsDefined() && !native.IsNull()) {
            result.resolved_native_config_yaml = YAML::Dump(native);
        }
    }
    return result;
}

std::string_view system_mode_name(SystemMode mode) noexcept {
    switch (mode) {
    case SystemMode::online:
        return "online";
    case SystemMode::offline:
        return "offline";
    case SystemMode::changeable:
        return "changeable";
    case SystemMode::time_window:
        return "time_window";
    }
    return "unknown";
}

std::string_view main_report_mode_name(MainReportMode mode) noexcept {
    switch (mode) {
    case MainReportMode::summary:
        return "summary";
    case MainReportMode::full:
        return "full";
    case MainReportMode::none:
        return "none";
    }
    return "unknown";
}

} // namespace virne::system
