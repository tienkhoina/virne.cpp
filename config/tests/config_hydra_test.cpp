#include "config/config_loader.h"
#include "config/override_parser.h"
#include "config/yaml_merge.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK(condition)                                                     \
    do                                                                       \
    {                                                                        \
        if (!(condition))                                                    \
        {                                                                    \
            std::cerr << "FAIL " << __FILE__ << ':' << __LINE__            \
                      << ": " #condition << '\n';                           \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

template<typename Function>
void check_throws(
    Function&& function,
    const std::string& expected)
{
    try
    {
        function();
        std::cerr << "FAIL expected exception containing: "
                  << expected << '\n';
        ++failures;
    }
    catch (const std::exception& error)
    {
        if (std::string(error.what()).find(expected) == std::string::npos)
        {
            std::cerr << "FAIL exception '" << error.what()
                      << "' does not contain '" << expected << "'\n";
            ++failures;
        }
    }
}

void write_file(
    const fs::path& path,
    const std::string& contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path);

    if (!output)
        throw std::runtime_error("Cannot create fixture: " + path.string());

    output << contents;
}

class TempConfigTree
{
public:
    TempConfigTree()
    {
        const auto suffix =
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count();
        root = fs::temp_directory_path() /
            ("virne-config-test-" + std::to_string(suffix));
        fs::create_directories(root);
    }

    ~TempConfigTree()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    fs::path root;
};

void test_real_settings()
{
    const auto source = fs::path(VIRNE_SOURCE_DIR);
    auto cfg = ConfigLoader::load((source / "setting/main.yaml").string());

    CHECK(!cfg.contains("defaults"));
    CHECK(cfg.get<std::string>("solver.solver_name") == "ppo_dual_gat+");
    CHECK(cfg.get<int>("p_net_setting.topology.num_nodes") == 100);
    CHECK(cfg.get<int>("v_sim_setting.num_v_nets") == 1000);
    CHECK(cfg.get<double>("training.learning_rate") == 0.001);

    // A present null is still an existing key, while the defaulting overload
    // returns its supplied fallback.
    CHECK(cfg.contains("training.seed"));
    CHECK(cfg.get<int>("training.seed", 77) == 77);

    const auto old_cwd = fs::current_path();
    fs::current_path(fs::temp_directory_path());
    override_parser::apply(
        cfg,
        "p_net_setting=p_net_setting_flagvne");
    fs::current_path(old_cwd);

    CHECK(cfg.get<int>("p_net_setting.topology.num_nodes") == 999);
    CHECK(cfg.get_raw("p_net_setting.node_attrs_setting").size() == 6);

    override_parser::apply(
        cfg,
        "v_sim_setting=v_sim_setting_flagvne");
    CHECK(std::abs(
        cfg.get<double>("v_sim_setting.arrival_rate.lam") - 0.1667) < 1e-12);
    CHECK(cfg.get_raw("v_sim_setting.node_attrs_setting").size() == 3);

    override_parser::apply(cfg, "experiment.seed=1_234");
    override_parser::apply(cfg, "training.use_cuda=FALSE");
    override_parser::apply(cfg, "training.learning_rate=2.5e-4");
    override_parser::apply(cfg, "output_subdir=results");
    override_parser::apply(cfg, "logger.backends=[console, 'json file']");
    override_parser::apply(cfg, "+runtime.tags=[fast, null, true]");
    override_parser::apply(
        cfg,
        "++runtime.records=[{name: first}, {name: second}]");
    override_parser::apply(cfg, "runtime.records.1.name=changed");

    CHECK(cfg.get<int>("experiment.seed") == 1234);
    CHECK(!cfg.get<bool>("training.use_cuda"));
    CHECK(std::abs(
        cfg.get<double>("training.learning_rate") - 0.00025) < 1e-12);
    CHECK(cfg.get<std::string>("output_subdir") == "results");
    CHECK((cfg.get<std::vector<std::string>>("logger.backends") ==
           std::vector<std::string>{"console", "json file"}));
    CHECK(cfg.get_raw("runtime.tags")[1].IsNull());
    CHECK(cfg.get_raw("runtime.tags")[2].as<bool>());
    CHECK(cfg.get<std::string>("runtime.records[1].name") == "changed");

    check_throws(
        [&]() { override_parser::apply(cfg, "+experiment.seed=2"); },
        "already exists");
    check_throws(
        [&]() { override_parser::apply(cfg, "does.not.exist=2"); },
        "Unknown key");
    check_throws(
        [&]() {
            override_parser::apply(
                cfg,
                "p_net_setting=missing_option");
        },
        "Config group option not found");

    override_parser::apply(cfg, "++does.not.exist=2");
    CHECK(cfg.get<int>("does.not.exist") == 2);
}

void test_original_setting_matrix()
{
    const auto source = fs::path(VIRNE_SOURCE_DIR);
    const auto main_setting = (source / "setting/main.yaml").string();

    struct GroupCase
    {
        const char* option;
        std::size_t node_attribute_count;
        std::size_t link_attribute_count;
    };

    const std::vector<GroupCase> physical_cases{
        {"default", 2, 2},
        {"p_net_setting", 2, 2},
        {"p_net_setting_ltc", 2, 3},
        {"p_net_setting_multi_resource", 6, 2},
        {"p_net_setting_single_resource", 2, 2},
        {"wx100_p_net_setting", 2, 2},
        // Project-specific option retained in addition to all original files.
        {"p_net_setting_flagvne", 6, 2},
    };

    for (const auto& test : physical_cases)
    {
        auto cfg = ConfigLoader::load(main_setting);
        override_parser::apply(
            cfg,
            std::string("p_net_setting=") + test.option);
        CHECK(cfg.get<int>("p_net_setting.topology.num_nodes") > 0);
        CHECK(cfg.get<std::string>("p_net_setting.topology.type") == "waxman");
        CHECK(cfg.get_raw("p_net_setting.node_attrs_setting").size() ==
              test.node_attribute_count);
        CHECK(cfg.get_raw("p_net_setting.link_attrs_setting").size() ==
              test.link_attribute_count);
        CHECK(cfg.get<std::string>("p_net_setting.output.file_name") ==
              "p_net.gml");
    }

    const std::vector<GroupCase> virtual_cases{
        {"default", 1, 1},
        {"v_sim_setting", 1, 1},
        {"v_sim_setting_ltc", 1, 2},
        {"v_sim_setting_multi_resource", 3, 1},
        {"v_sim_setting_single_resource", 1, 1},
        // Project-specific option retained in addition to all original files.
        {"v_sim_setting_flagvne", 3, 1},
    };

    for (const auto& test : virtual_cases)
    {
        auto cfg = ConfigLoader::load(main_setting);
        override_parser::apply(
            cfg,
            std::string("v_sim_setting=") + test.option);
        CHECK(cfg.get<int>("v_sim_setting.num_v_nets") == 1000);
        CHECK(cfg.get<std::string>("v_sim_setting.topology.type") == "random");
        CHECK(cfg.get_raw("v_sim_setting.node_attrs_setting").size() ==
              test.node_attribute_count);
        CHECK(cfg.get_raw("v_sim_setting.link_attrs_setting").size() ==
              test.link_attribute_count);
        CHECK(cfg.get<std::string>("v_sim_setting.output.save_dir") ==
              "dataset/v_nets");
    }

    auto latency = ConfigLoader::load(main_setting);
    override_parser::apply(latency, "p_net_setting=p_net_setting_ltc");
    CHECK(latency.get<std::string>(
              "p_net_setting.link_attrs_setting[2].name") == "ltc");

    auto multi = ConfigLoader::load(main_setting);
    override_parser::apply(
        multi,
        "v_sim_setting=v_sim_setting_multi_resource");
    CHECK(multi.get<std::string>(
              "v_sim_setting.node_attrs_setting[2].name") == "ram");
}

void create_composition_fixtures(
    const fs::path& root)
{
    write_file(
        root / "base.yaml",
        "shared:\n"
        "  from_base: 1\n"
        "  order: base\n"
        "replace_list: [1, 2]\n");

    write_file(
        root / "global/common.yaml",
        "from_absolute: common\n");

    write_file(
        root / "group/base.yaml",
        "nested:\n"
        "  inherited: inherited\n"
        "  overridden: base\n");

    write_file(
        root / "group/first.yaml",
        "defaults:\n"
        "  - base\n"
        "  - /global/common@_here_\n"
        "  - _self_\n"
        "nested:\n"
        "  overridden: first\n"
        "own: 9\n");

    write_file(
        root / "group/second.yaml",
        "only_second: true\n");

    write_file(
        root / "main.yaml",
        "defaults:\n"
        "  - base\n"
        "  - group: first\n"
        "  - optional missing_group: absent\n"
        "  - _self_\n"
        "shared:\n"
        "  order: main\n"
        "  local: 2\n"
        "replace_list: [9]\n"
        "base_values:\n"
        "  host: localhost\n"
        "  port: 8080\n"
        "url: http://${base_values.host}:${base_values.port}\n"
        "port_copy: ${base_values.port}\n"
        "nested:\n"
        "  sibling: local\n"
        "  sibling_copy: ${.sibling}\n"
        "  root_copy: ${..base_values.host}\n"
        "values: [1, 2, 3]\n"
        "values_copy: ${values}\n"
        "selected: ${oc.select:missing.value,42}\n"
        "nullable: null\n"
        "selected_null: ${oc.select:nullable,42}\n"
        "environment: ${oc.env:VIRNE_CONFIG_TEST_ENV,fallback}\n"
        "date_a: ${now:%Y-%m-%d}\n"
        "date_b: ${now:%Y-%m-%d}\n");

    write_file(
        root / "early_self.yaml",
        "defaults:\n"
        "  - _self_\n"
        "  - base\n"
        "shared:\n"
        "  order: early\n");

    write_file(
        root / "replace_group.yaml",
        "defaults:\n"
        "  - group: first\n"
        "  - override group: second\n");

    write_file(
        root / "cycle_a.yaml",
        "defaults: [cycle_b]\n");
    write_file(
        root / "cycle_b.yaml",
        "defaults: [cycle_a]\n");
}

void test_composition_and_interpolation()
{
    TempConfigTree fixtures;
    create_composition_fixtures(fixtures.root);

#if defined(_WIN32)
    _putenv_s("VIRNE_CONFIG_TEST_ENV", "from-env");
#else
    setenv("VIRNE_CONFIG_TEST_ENV", "from-env", 1);
#endif

    auto cfg = ConfigLoader::load((fixtures.root / "main.yaml").string());

    CHECK(cfg.get<int>("shared.from_base") == 1);
    CHECK(cfg.get<std::string>("shared.order") == "main");
    CHECK(cfg.get<int>("shared.local") == 2);
    CHECK((cfg.get<std::vector<int>>("replace_list") ==
           std::vector<int>{9}));
    CHECK(cfg.get<std::string>("group.nested.inherited") == "inherited");
    CHECK(cfg.get<std::string>("group.nested.overridden") == "first");
    CHECK(cfg.get<std::string>("group.from_absolute") == "common");
    CHECK(cfg.get<int>("group.own") == 9);

    CHECK(cfg.get<std::string>("url") == "http://localhost:8080");
    CHECK(cfg.get<int>("port_copy") == 8080);
    CHECK(cfg.get<std::string>("nested.sibling_copy") == "local");
    CHECK(cfg.get<std::string>("nested.root_copy") == "localhost");
    CHECK((cfg.get<std::vector<int>>("values_copy") ==
           std::vector<int>{1, 2, 3}));
    CHECK(cfg.get<int>("selected") == 42);
    CHECK(cfg.get_raw("selected_null").IsNull());
    CHECK(cfg.get<std::string>("environment") == "from-env");
    CHECK(cfg.get<std::string>("date_a") == cfg.get<std::string>("date_b"));

    // Interpolation is lazy: a CLI override immediately changes all aliases.
    override_parser::apply(cfg, "base_values.port=9090");
    CHECK(cfg.get<int>("port_copy") == 9090);
    CHECK(cfg.get<std::string>("url") == "http://localhost:9090");

    const auto resolved_root = cfg.root();
    CHECK(resolved_root["port_copy"].as<int>() == 9090);
    CHECK(resolved_root["values_copy"].IsSequence());

    auto early = ConfigLoader::load(
        (fixtures.root / "early_self.yaml").string());
    CHECK(early.get<std::string>("shared.order") == "base");

    auto replacement = ConfigLoader::load(
        (fixtures.root / "replace_group.yaml").string());
    CHECK(replacement.get<bool>("group.only_second"));
    CHECK(!replacement.contains("group.own"));
    CHECK(!replacement.contains("group.nested.inherited"));

    check_throws(
        [&]() {
            ConfigLoader::load((fixtures.root / "cycle_a.yaml").string());
        },
        "Config cycle detected");

    Config interpolation_cycle(YAML::Load(
        "a: ${b}\n"
        "b: ${a}\n"));
    check_throws(
        [&]() { (void)interpolation_cycle.get<std::string>("a"); },
        "Interpolation cycle detected");

    Config missing_interpolation(YAML::Load("a: ${missing}\n"));
    check_throws(
        [&]() { (void)missing_interpolation.get<std::string>("a"); },
        "Missing interpolation key");
}

void test_recursive_merge()
{
    const auto base = YAML::Load(
        "map:\n"
        "  keep: 1\n"
        "  replace: old\n"
        "list: [1, 2]\n"
        "scalar: old\n");
    const auto overlay = YAML::Load(
        "map:\n"
        "  replace: new\n"
        "  add: 2\n"
        "list: [9]\n"
        "scalar: null\n");
    const auto merged = yaml_merge::merge(base, overlay);

    CHECK(merged["map"]["keep"].as<int>() == 1);
    CHECK(merged["map"]["replace"].as<std::string>() == "new");
    CHECK(merged["map"]["add"].as<int>() == 2);
    CHECK(merged["list"].size() == 1);
    CHECK(merged["list"][0].as<int>() == 9);
    CHECK(merged["scalar"].IsNull());
}

} // namespace

int main()
{
    try
    {
        test_real_settings();
        test_original_setting_matrix();
        test_composition_and_interpolation();
        test_recursive_merge();
    }
    catch (const std::exception& error)
    {
        std::cerr << "UNEXPECTED EXCEPTION: " << error.what() << '\n';
        return 2;
    }

    if (failures != 0)
    {
        std::cerr << failures << " config test(s) failed\n";
        return 1;
    }

    std::cout << "ALL CONFIG HYDRA-COMPAT TESTS PASSED\n";
    return 0;
}
