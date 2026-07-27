#include "config/config_loader.h"
#include "config/override_parser.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void check_throws(Fn&& fn, const char* message)
{
    bool threw = false;
    try
    {
        fn();
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    check(threw, message);
}

void write_file(const fs::path& path, const std::string& contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("cannot create config fixture: " + path.string());
    }
    output << contents;
}

class TempConfigTree
{
public:
    TempConfigTree()
    {
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        root = fs::temp_directory_path() /
               ("virne-config-test-" + std::to_string(stamp));
        fs::create_directories(root);
    }

    ~TempConfigTree()
    {
        std::error_code error;
        fs::remove_all(root, error);
    }

    fs::path root;
};

void check_repository_settings()
{
    const fs::path source = VIRNE_SOURCE_DIR;
    Config cfg = ConfigLoader::load((source / "setting/main.yaml").string());

    check(cfg.get<std::string>("solver.solver_name") == "ppo_dual_gat+",
          "main setting scalar missing");
    check(cfg.get<int>("training.batch_size") == 128,
          "learning defaults were not merged globally");
    check(cfg.get<int>("p_net_setting.topology.num_nodes") == 100,
          "physical-network group was not packaged");
    check(cfg.get<int>("v_sim_setting.num_v_nets") == 1000,
          "virtual-network group was not packaged");
    check(cfg.get<std::vector<std::string>>("logger.backends").size() == 3,
          "YAML sequence conversion failed");
    check(cfg.contains("training.seed"),
          "Hydra-style contains must retain explicit null keys");
    check(cfg.get<int>("training.seed", 73) == 73,
          "null value did not use typed default");

    override_parser::apply(cfg, "p_net_setting=p_net_setting_flagvne");
    override_parser::apply(cfg, "v_sim_setting=v_sim_setting_flagvne");
    check(cfg.get<int>("p_net_setting.topology.num_nodes") == 999,
          "normal config-group override failed");
    check(cfg.get<std::string>("p_net_setting.node_attrs_setting[4].name") == "ram",
          "list-index read from group override failed");
    check(std::abs(cfg.get<double>("v_sim_setting.arrival_rate.lam") - 0.1667) < 1e-12,
          "second config-group override failed");

    override_parser::apply(cfg, "training.batch_size=256");
    override_parser::apply(cfg, "logger.backends=[console, file]");
    override_parser::apply(cfg, "p_net_setting.node_attrs_setting[0].high=77");
    override_parser::apply(cfg, "++runtime.flags={fast: true, retries: 0x10}");
    override_parser::apply(cfg, "++runtime.empty=null");
    override_parser::apply(cfg, "solver.pretrained_model_path='model=a.pt'");

    check(cfg.get<int>("training.batch_size") == 256,
          "integer override failed");
    check(cfg.get<std::vector<std::string>>("logger.backends") ==
              std::vector<std::string>({"console", "file"}),
          "list override failed");
    check(cfg.get<int>("p_net_setting.node_attrs_setting.0.high") == 77,
          "list-index write failed");
    check(cfg.get<bool>("runtime.flags.fast"),
          "map override failed");
    check(cfg.get<int>("runtime.flags.retries") == 16,
          "base-prefixed integer override failed");
    check(cfg.contains("runtime.empty") &&
              cfg.get<int>("runtime.empty", 9) == 9,
          "null override semantics failed");
    check(cfg.get<std::string>("solver.pretrained_model_path") == "model=a.pt",
          "quoted string containing '=' failed");

    check_throws([&] { override_parser::apply(cfg, "+training.batch_size=1"); },
                 "single-plus override replaced an existing key");
    check_throws([&] { override_parser::apply(cfg, "missing.value=1"); },
                 "plain override created an unknown key");
}

void check_defaults_and_resolvers()
{
    TempConfigTree fixture;
    write_file(
        fixture.root / "base.yaml",
        "value: base\n"
        "from_base: true\n");
    write_file(
        fixture.root / "group/one.yaml",
        "x: 4\n"
        "items: [a, b]\n");
    write_file(
        fixture.root / "group/two.yaml",
        "x: 8\n");
    write_file(
        fixture.root / "main.yaml",
        "defaults:\n"
        "  - base\n"
        "  - optional missing\n"
        "  - group: one\n"
        "  - override group: two\n"
        "  - _self_\n"
        "value: self\n"
        "copy: ${group.x}\n"
        "message: value-${group.x}\n"
        "env_default: ${oc.env:VIRNE_CONFIG_TEST_MISSING,17}\n"
        "selected: ${oc.select:not.present,23}\n"
        "year: ${now:%Y}\n"
        "nested:\n"
        "  value: 31\n"
        "  copy: ${.value}\n");

    Config cfg = ConfigLoader::load((fixture.root / "main.yaml").string());
    check(cfg.get<std::string>("value") == "self",
          "_self_ ordering differs from Hydra");
    check(cfg.get<bool>("from_base"),
          "plain defaults entry failed");
    check(cfg.get<int>("group.x") == 8,
          "override defaults entry failed");
    check(cfg.get<int>("copy") == 8,
          "typed interpolation failed");
    check(cfg.get<std::string>("message") == "value-8",
          "embedded interpolation failed");
    check(cfg.get<int>("env_default") == 17,
          "oc.env default resolver failed");
    check(cfg.get<int>("selected") == 23,
          "oc.select default resolver failed");
    check(cfg.get<int>("nested.copy") == 31,
          "relative interpolation failed");
    check(cfg.get<std::string>("year").size() == 4,
          "now resolver failed");

    write_file(
        fixture.root / "implicit.yaml",
        "defaults: [base]\n"
        "value: implicit-self\n");
    check(ConfigLoader::load((fixture.root / "implicit.yaml").string())
              .get<std::string>("value") == "implicit-self",
          "implicit _self_ was not merged last");

    write_file(
        fixture.root / "package.yaml",
        "defaults:\n"
        "  - group@alias: one\n");
    check(ConfigLoader::load((fixture.root / "package.yaml").string())
              .get<int>("alias.x") == 4,
          "defaults package override failed");

    write_file(fixture.root / "cycle_a.yaml", "defaults: [cycle_b]\n");
    write_file(fixture.root / "cycle_b.yaml", "defaults: [cycle_a]\n");
    check_throws(
        [&] { (void)ConfigLoader::load((fixture.root / "cycle_a.yaml").string()); },
        "defaults cycle was not rejected");

    Config interpolated(YAML::Load("a: ${b}\nb: ${a}\n"));
    check_throws([&] { (void)interpolated.get<std::string>("a"); },
                 "interpolation cycle was not rejected");
}

} // namespace

int main()
{
    try
    {
        check_repository_settings();
        check_defaults_and_resolvers();
        std::cout << "test_config: ALL TESTS PASSED\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_config: " << error.what() << '\n';
        return 1;
    }
}
