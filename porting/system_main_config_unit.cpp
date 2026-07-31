#include "../virne/system/main_config.h"
#include "../virne/system/main_runtime.h"

#include "../config/config_loader.h"
#include "../config/override_parser.h"

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace system = virne::system;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::filesystem::path main_setting_path() {
#if defined(VIRNE_SOURCE_DIR)
    return std::filesystem::path(VIRNE_SOURCE_DIR) / "setting/main.yaml";
#else
    return "setting/main.yaml";
#endif
}

Config configured_main() {
    Config config = ConfigLoader::load(main_setting_path().string());
    override_parser::apply(config, "solver.solver_name=ffd_rank");
    override_parser::apply(config, "experiment.seed=37");
    override_parser::apply(config, "experiment.run_id=system-config-unit");
    override_parser::apply(config, "experiment.num_simulations=3");
    override_parser::apply(config, "system.if_time_window=true");
    override_parser::apply(config, "system.time_window_size=12.5");
    override_parser::apply(config, "logger.backends=[console,file,tensorboard]");
    override_parser::apply(config, "++native.capture_solutions=false");
    override_parser::apply(config, "++native.output.report=full");
    override_parser::apply(config, "++native.progress.enabled=false");
    override_parser::apply(config, "++native.progress.width=42");
    override_parser::apply(
        config, "++native.progress.minimum_interval_ms=250");
    override_parser::apply(config, "++native.time_window_origin=2.5");
    override_parser::apply(config, "++native.changeable_stage_count=4");
    override_parser::apply(config, "++native.workers.factory=2");
    override_parser::apply(config, "++native.workers.attribute=3");
    override_parser::apply(config, "++native.workers.arrangement=4");
    override_parser::apply(config, "++native.workers.event=5");
    override_parser::apply(config, "++native.workers.io=6");
    override_parser::apply(config, "++native.workers.counter=7");
    override_parser::apply(config, "++native.workers.recorder=8");
    override_parser::apply(config, "++native.workers.mutation=9");
    override_parser::apply(config, "++native.workers.rank=10");
    override_parser::apply(config, "++native.workers.node_candidate=11");
    override_parser::apply(
        config, "++native.workers.link_topology_constraint=12");
    override_parser::apply(config, "++native.workers.link_candidate=13");
    return config;
}

void test_decode_and_selection() {
    Config config = configured_main();
    PyRandom random(91U);
    PyRandom baseline(91U);
    const system::MainConfig decoded =
        system::main_config_from_hydra(config, random);

    require(
        random.genrand_uint32() == baseline.genrand_uint32(),
        "explicit run ID unexpectedly consumed RNG");
    require(
        decoded.seed == std::optional<std::uint32_t>{37U} &&
            decoded.num_simulations == 3U &&
            decoded.mode == system::SystemMode::time_window &&
            !decoded.capture_solutions &&
            decoded.report_mode == system::MainReportMode::full &&
            !decoded.progress.enabled && decoded.progress.width == 42U &&
            decoded.progress.minimum_interval_ms == 250U &&
            decoded.changeable_stage_count == 4U &&
            decoded.time_window_size == 12.5 &&
            decoded.time_window_origin == 2.5,
        "typed system fields mismatch");
    require(
        decoded.solver_name == "ffd_rank" &&
            decoded.solver_config.seed ==
                std::optional<std::uint32_t>{37U} &&
            decoded.solver_config.save_dir.filename() ==
                "system-config-unit",
        "typed solver fields mismatch");
    require(
        decoded.logger_config.backends.console &&
            decoded.logger_config.backends.file,
        "non-ML logger backend resolution mismatch");
    require(
        decoded.resolved_config_yaml == YAML::Dump(
            system::python_compatible_config_root(config)),
        "resolved config snapshot omitted or changed a composed field");
    require(
        !decoded.resolved_native_config_yaml.empty(),
        "native config extension snapshot was omitted");
    require(
        config.get<std::int64_t>(
            "simulation.p_net_setting_num_nodes") == 100 &&
            config.get<std::size_t>(
                "simulation.p_net_setting_num_node_attrs") == 2U &&
            config.get<std::size_t>(
                "simulation.p_net_setting_num_link_attrs") == 2U &&
            config.get<std::size_t>(
                "simulation.v_sim_setting_num_node_attrs") == 1U &&
            config.get<std::size_t>(
                "simulation.v_sim_setting_num_link_attrs") == 1U &&
            config.get<std::int64_t>(
                "simulation.p_net_num_nodes") == 100 &&
            config.get<std::size_t>(
                "rl.feature_constructor.num_extracted_p_node_attrs") ==
                1U &&
            config.get<std::size_t>(
                "rl.feature_constructor.num_extracted_p_link_attrs") ==
                1U &&
            config.get<std::size_t>(
                "rl.feature_constructor.num_extracted_v_node_attrs") ==
                1U &&
            config.get<std::size_t>(
                "rl.feature_constructor.num_extracted_v_link_attrs") ==
                1U &&
            config.get<std::int64_t>(
                "rl.feature_constructor.p_num_nodes") == 100,
        "Python add_simulation_into_config parity mismatch");
    require(
        decoded.workers.physical.factory_workers == 2U &&
            decoded.workers.physical.attribute_workers == 3U &&
            decoded.workers.simulator.arrangement_workers == 4U &&
            decoded.workers.simulator.event_workers == 5U &&
            decoded.workers.simulator.io_workers == 6U &&
            decoded.workers.environment.counter_workers == 7U &&
            decoded.workers.environment.recorder_workers == 8U &&
            decoded.workers.environment.mutation_workers == 9U &&
            decoded.workers.node_rank.rank_workers == 10U &&
            decoded.workers.node_rank.node_candidate_workers == 11U &&
            decoded.workers.node_rank.link_topology_constraint_workers ==
                12U &&
            decoded.workers.node_rank.link_candidate_workers == 13U,
        "worker decode mismatch");

    const auto virtual_config =
        virne::network::virtual_network_simulation_config_from_setting(
            decoded.virtual_setting);
    const auto selections = system::runtime_selections_from_virtual_config(
        virtual_config, false);
    require(
        selections.controller.node_resources ==
            std::vector<virne::core::controller::ResourceId>{0U} &&
            selections.controller.link_resources ==
                std::vector<virne::core::controller::ResourceId>{0U} &&
            selections.controller.constraints.node_at_node ==
                std::vector<virne::core::controller::ConstraintId>{0U} &&
            selections.controller.constraints.link_at_link ==
                std::vector<virne::core::controller::ConstraintId>{0U} &&
            selections.controller.hard_node_constraints ==
                std::vector<virne::core::controller::ConstraintId>{0U} &&
            selections.controller.hard_link_constraints ==
                std::vector<virne::core::controller::ConstraintId>{0U},
        "resource/constraint IDs were not resolved once in registry order");
}

void test_auto_run_id_in_resolved_snapshot() {
    Config config = configured_main();
    override_parser::apply(config, "experiment.run_id=auto");
    override_parser::apply(
        config,
        "logger.experiment_name=${experiment.run_id}");
    PyRandom random(91U);
    const system::MainConfig decoded =
        system::main_config_from_hydra(config, random);
    const YAML::Node snapshot = YAML::Load(decoded.resolved_config_yaml);
    require(
        snapshot["experiment"]["run_id"].as<std::string>() ==
                decoded.logger_config.run_id &&
            snapshot["logger"]["experiment_name"].as<std::string>() ==
                decoded.logger_config.run_id &&
            decoded.logger_config.experiment_name ==
                decoded.logger_config.run_id &&
            decoded.logger_config.run_id != "auto" &&
            config.get<std::string>("experiment.run_id") ==
                decoded.logger_config.run_id,
        "auto run ID was not resolved consistently before interpolation");
}

void test_info_level_suppresses_unused_snapshot() {
    Config config = configured_main();
    override_parser::apply(config, "logger.level=WARNING");
    PyRandom random(91U);
    const system::MainConfig decoded =
        system::main_config_from_hydra(config, random);
    require(
        decoded.resolved_config_yaml.empty() &&
            decoded.resolved_native_config_yaml.empty(),
        "Logger level above INFO still serialized the full config");
}

void test_mode_precedence_and_errors() {
    Config config = configured_main();
    override_parser::apply(config, "system.if_offline_system=true");
    override_parser::apply(config, "system.if_changeable_v_nets=true");
    PyRandom random(1U);
    require(
        system::main_config_from_hydra(config, random).mode ==
            system::SystemMode::changeable,
        "Python-compatible mode precedence mismatch");

    Config invalid_stage = configured_main();
    override_parser::apply(
        invalid_stage, "native.changeable_stage_count=0");
    try {
        static_cast<void>(
            system::main_config_from_hydra(invalid_stage, random));
        throw std::runtime_error("zero stage count was accepted");
    } catch (const system::MainConfigException& error) {
        require(
            error.code() ==
                system::MainConfigErrorCode::invalid_changeable_stage_count,
            "zero stage error code mismatch");
    }

    Config invalid_matching = configured_main();
    override_parser::apply(
        invalid_matching, "solver.matching_mathod=not-a-method");
    try {
        static_cast<void>(
            system::main_config_from_hydra(invalid_matching, random));
        throw std::runtime_error("invalid matching method was accepted");
    } catch (const system::MainConfigException& error) {
        require(
            error.code() ==
                system::MainConfigErrorCode::invalid_matching_method,
            "matching-method error code mismatch");
    }

    Config invalid_report = configured_main();
    override_parser::apply(invalid_report, "native.output.report=verbose");
    try {
        static_cast<void>(
            system::main_config_from_hydra(invalid_report, random));
        throw std::runtime_error("invalid report mode was accepted");
    } catch (const system::MainConfigException& error) {
        require(
            error.code() == system::MainConfigErrorCode::invalid_report_mode,
            "report-mode error code mismatch");
    }

    Config invalid_progress = configured_main();
    override_parser::apply(invalid_progress, "native.progress.width=0");
    try {
        static_cast<void>(
            system::main_config_from_hydra(invalid_progress, random));
        throw std::runtime_error("zero progress width was accepted");
    } catch (const system::MainConfigException& error) {
        require(
            error.code() ==
                system::MainConfigErrorCode::invalid_progress_width,
            "progress-width error code mismatch");
    }
}

void test_report_json() {
    system::MainRunReport report;
    report.mode = system::SystemMode::online;
    report.solver_id = {3U};
    report.solver_name = "ffd_rank";
    report.physical_nodes = 7U;
    report.physical_links = 6U;
    report.virtual_requests = 2U;
    report.scheduled_events = 4U;
    report.setup_time_ns = 11U;
    report.run_time_ns = 13U;
    std::ostringstream output;
    system::write_main_report_json(output, report);
    require(
        output.str() ==
            "{\"mode\":\"online\",\"solver\":\"ffd_rank\","
            "\"solver_id\":3,\"physical_nodes\":7,\"physical_links\":6,"
            "\"virtual_requests\":2,\"scheduled_events\":4,"
            "\"setup_time_ns\":11,\"run_time_ns\":13,\"steps\":[],"
            "\"epochs\":[],\"stages\":[],\"windows\":[]}\n",
        "canonical report JSON mismatch");

    std::ostringstream summary;
    system::write_main_summary_json(summary, report);
    require(
        summary.str() ==
            "{\"mode\":\"online\",\"solver\":\"ffd_rank\","
            "\"physical_nodes\":7,\"physical_links\":6,"
            "\"virtual_requests\":2,\"scheduled_events\":4,"
            "\"epochs\":0,\"processed_requests\":0,\"accepted\":0,"
            "\"rejected\":0,\"acceptance_rate\":0,"
            "\"long_term_r2c_ratio\":0,\"setup_time_ns\":11,"
            "\"run_time_ns\":13}\n",
        "concise summary JSON mismatch");
}

system::MainConfig small_runtime_config() {
    Config config = ConfigLoader::load(main_setting_path().string());
    override_parser::apply(config, "solver.solver_name=order_rank");
    override_parser::apply(config, "experiment.seed=7");
    override_parser::apply(config, "experiment.run_id=system-runtime-unit");
    override_parser::apply(config, "experiment.num_simulations=1");
    override_parser::apply(config, "experiment.if_save_config=false");
    override_parser::apply(config, "experiment.if_save_p_net=false");
    override_parser::apply(config, "experiment.if_save_v_nets=false");
    override_parser::apply(config, "recorder.if_save_records=false");
    override_parser::apply(config, "recorder.if_temp_save_records=false");
    override_parser::apply(config, "logger.backends=[]");
    override_parser::apply(config, "p_net_setting.topology.num_nodes=12");
    override_parser::apply(config, "v_sim_setting.num_v_nets=3");
    override_parser::apply(config, "v_sim_setting.v_net_size.low=2");
    override_parser::apply(config, "v_sim_setting.v_net_size.high=4");
    override_parser::apply(config, "++native.capture_solutions=true");
    override_parser::apply(config, "++native.output.report=full");
    override_parser::apply(config, "++native.progress.enabled=false");
    override_parser::apply(config, "++native.changeable_stage_count=2");
    override_parser::apply(config, "++native.workers.rank=2");
    override_parser::apply(config, "++native.workers.node_candidate=2");
    override_parser::apply(
        config, "++native.workers.link_topology_constraint=2");
    override_parser::apply(config, "++native.workers.link_candidate=2");
    PyRandom random(13U);
    return system::main_config_from_hydra(config, random);
}

void test_end_to_end_modes() {
    for (const system::SystemMode mode : {
             system::SystemMode::online,
             system::SystemMode::offline,
             system::SystemMode::time_window,
             system::SystemMode::changeable}) {
        system::MainConfig config = small_runtime_config();
        config.mode = mode;
        config.time_window_size = 40.0;
        const system::MainRunReport report =
            system::run_main_config(config);
        require(
            report.mode == mode && report.solver_id.value == 0U &&
                report.solver_name == "order_rank" &&
                report.physical_nodes == 12U &&
                report.virtual_requests == 3U &&
                report.scheduled_events == 6U,
            "main runtime setup/config integration mismatch");

        const bool changeable = mode == system::SystemMode::changeable;
        require(
            report.run.steps.size() == (changeable ? 6U : 3U) &&
                report.run.epochs.size() == (changeable ? 2U : 1U),
            "main runtime did not execute every configured request");
        require(
            report.stages.size() == (changeable ? 2U : 0U),
            "changeable stage summary mismatch");
        require(
            report.windows.empty() ==
                (mode != system::SystemMode::time_window),
            "time-window summary presence mismatch");
    }
}

void test_multi_resource_main_runtime() {
    Config config = ConfigLoader::load(main_setting_path().string());
    override_parser::apply(
        config, "p_net_setting=p_net_setting_multi_resource");
    override_parser::apply(
        config, "v_sim_setting=v_sim_setting_multi_resource");
    override_parser::apply(config, "solver.solver_name=ffd_rank");
    override_parser::apply(config, "experiment.seed=0");
    override_parser::apply(config, "experiment.run_id=multi-resource-unit");
    override_parser::apply(config, "experiment.num_simulations=1");
    override_parser::apply(config, "experiment.if_save_config=false");
    override_parser::apply(config, "experiment.if_save_p_net=false");
    override_parser::apply(config, "experiment.if_save_v_nets=false");
    override_parser::apply(config, "recorder.if_save_records=false");
    override_parser::apply(config, "recorder.if_temp_save_records=false");
    override_parser::apply(config, "logger.backends=[]");
    override_parser::apply(config, "p_net_setting.topology.num_nodes=20");
    override_parser::apply(config, "v_sim_setting.num_v_nets=8");
    override_parser::apply(config, "v_sim_setting.v_net_size.low=2");
    override_parser::apply(config, "v_sim_setting.v_net_size.high=4");
    override_parser::apply(config, "++native.capture_solutions=false");
    override_parser::apply(config, "++native.output.report=summary");
    override_parser::apply(config, "++native.progress.enabled=false");

    PyRandom random(17U);
    const system::MainConfig decoded =
        system::main_config_from_hydra(config, random);
    require(
        decoded.resolved_config_yaml.empty(),
        "disabled Logger backend still serialized the full config");
    const auto virtual_config =
        virne::network::virtual_network_simulation_config_from_setting(
            decoded.virtual_setting);
    const auto selections = system::runtime_selections_from_virtual_config(
        virtual_config, false);

    const std::vector<virne::core::controller::ResourceId>
        expected_node_resources{0U, 1U, 2U};
    require(
        selections.controller.node_resources == expected_node_resources &&
            selections.controller.constraints.node_at_node ==
                expected_node_resources &&
            selections.controller.hard_node_constraints ==
                expected_node_resources,
        "multi-resource controller IDs were not retained in virtual order");
    require(
        !selections.counter.node_resources.has_value() &&
            !selections.counter.link_resources.has_value(),
        "counter must bind all resources independently per registry");

    const system::MainRunReport report = system::run_main_config(decoded);
    require(
        report.mode == system::SystemMode::online &&
            report.solver_name == "ffd_rank" &&
            report.physical_nodes == 20U &&
            report.virtual_requests == 8U &&
            report.scheduled_events == 16U &&
            report.run.epochs.size() == 1U &&
            report.run.epochs.front().arrival_steps == 8U,
        "multi-resource main runtime did not process the complete fixture");
}

} // namespace

int main() {
    try {
        test_decode_and_selection();
        test_auto_run_id_in_resolved_snapshot();
        test_info_level_suppresses_unused_snapshot();
        test_mode_precedence_and_errors();
        test_report_json();
        test_end_to_end_modes();
        test_multi_resource_main_runtime();
        std::cout << "system main-config unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "system main-config unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
