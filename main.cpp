#include "config/config_loader.h"
#include "config/override_parser.h"
#include "random/py_random.h"
#include "virne/system/main_config.h"
#include "virne/system/main_runtime.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CommandLine {
    std::filesystem::path config_path;
    std::vector<std::string> overrides;
    bool help = false;
};

bool apply_python_group_merge(
    Config& config,
    const std::filesystem::path& main_config_path,
    const std::string& argument) {
    static constexpr std::string_view physical_group =
        "+p_net_setting=";
    static constexpr std::string_view virtual_group =
        "+v_sim_setting=";
    const std::string_view view(argument);
    std::string_view group;
    if (view.rfind(physical_group, 0U) == 0U) {
        group = "p_net_setting";
    } else if (view.rfind(virtual_group, 0U) == 0U) {
        group = "v_sim_setting";
    } else {
        return false;
    }

    const std::size_t equals = view.find('=');
    const std::string option(view.substr(equals + 1U));
    if (option.empty()) {
        throw std::invalid_argument(
            "Python-compatible config group requires an option: " +
            argument);
    }

    std::filesystem::path option_path = option;
    if (option_path.extension().empty()) {
        option_path += ".yaml";
    }
    const std::filesystem::path group_path =
        main_config_path.parent_path() / std::string(group) / option_path;
    const Config selected = ConfigLoader::load(group_path.string());

    // Hydra's '+group=option' appends a second package and recursively merges
    // it over the already composed default group.  Native 'group=option'
    // remains the existing replacement spelling.  Keep this compatibility
    // behavior at the CLI boundary instead of changing the frozen override
    // parser contract.
    YAML::Node overlay(YAML::NodeType::Map);
    overlay[std::string(group)] = selected.root();
    config.merge(overlay);
    return true;
}

std::filesystem::path default_config_path() {
    std::filesystem::path local = "setting/main.yaml";
    if (std::filesystem::exists(local)) {
        return local;
    }
#if defined(VIRNE_SOURCE_DIR)
    return std::filesystem::path(VIRNE_SOURCE_DIR) / "setting/main.yaml";
#else
    return local;
#endif
}

CommandLine parse_command_line(int argc, char** argv) {
    CommandLine result;
    result.config_path = default_config_path();
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            result.help = true;
        } else if (argument == "--config") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--config requires a path");
            }
            ++index;
            result.config_path = argv[index];
        } else if (argument.rfind("--config=", 0U) == 0U) {
            result.config_path = argument.substr(9U);
        } else {
            result.overrides.push_back(argument);
        }
    }
    return result;
}

void print_help() {
    std::cout
        << "Usage: main [--config PATH] [hydra.override=value ...]\n"
        << "Example:\n"
        << "  main solver.solver_name=ffd_rank "
           "v_sim_setting.num_v_nets=32 "
           "experiment.run_id=e2e "
           "recorder.if_temp_save_records=false "
           "++native.output.report=summary "
           "++native.workers.rank=8\n";
}

} // namespace

int main(int argc, char** argv) {
    // The native CLI never mixes C stdio with iostreams.  Disable the
    // compatibility synchronization before the first stream operation and
    // detach input from output so an eventual read cannot flush stdout.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cerr.tie(nullptr);
    std::clog.tie(nullptr);

    try {
        const CommandLine command = parse_command_line(argc, argv);
        if (command.help) {
            print_help();
            return 0;
        }

        Config config = ConfigLoader::load(command.config_path.string());
        for (const auto& expression : command.overrides) {
            if (!apply_python_group_merge(
                    config,
                    command.config_path,
                    expression)) {
                override_parser::apply(config, expression);
            }
        }

        const auto tick = std::chrono::system_clock::now()
                              .time_since_epoch()
                              .count();
        PyRandom run_id_random(static_cast<std::uint64_t>(tick));
        const virne::system::MainConfig main_config =
            virne::system::main_config_from_hydra(config, run_id_random);

        if (main_config.save_config) {
            // Python's get_run_id_dir adds this derived path only when
            // persisting the config. Keep the resolved log tree unchanged
            // while making the saved YAML compatible at this field boundary.
            config.set(
                "experiment.save_run_id_dir",
                YAML::Node(
                    main_config.solver_config.save_dir.generic_string()));
            std::filesystem::create_directories(
                main_config.solver_config.save_dir);
            Config(
                virne::system::python_compatible_config_root(config))
                .save(
                (main_config.solver_config.save_dir / "config.yaml").string());
            const YAML::Node native = config.get_raw("native");
            if (native.IsDefined() && !native.IsNull()) {
                YAML::Node native_root(YAML::NodeType::Map);
                native_root["native"] = native;
                Config(native_root).save(
                    (main_config.solver_config.save_dir /
                     "native_config.yaml")
                        .string());
            }
        }

        const virne::system::MainRunReport report =
            virne::system::run_main_config(main_config);
        switch (main_config.report_mode) {
        case virne::system::MainReportMode::summary:
            virne::system::write_main_summary_json(std::cout, report);
            break;
        case virne::system::MainReportMode::full:
            virne::system::write_main_report_json(std::cout, report);
            break;
        case virne::system::MainReportMode::none:
            break;
        }
        std::cout.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "virne main: FAIL: " << error.what() << '\n';
        return 1;
    }
}
