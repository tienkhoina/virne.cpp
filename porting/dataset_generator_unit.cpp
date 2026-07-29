#include "dataset_generator.h"

#include "config.h"
#include "config_loader.h"
#include "override_parser.h"
#include "random_context.h"
#include "setting.h"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#ifndef VIRNE_SOURCE_DIR
#define VIRNE_SOURCE_DIR "."
#endif

namespace {

namespace fs = std::filesystem;
namespace network = virne::network;
namespace utils = virne::utils;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Function>
void expect_error(Function&& function, const std::string& context) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return;
    }
    fail(context + ": expected an exception");
}

std::uint64_t raw64(const double value) {
    std::uint64_t result = 0U;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::string attr_scalar(const AttrValue& value) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return "i" + std::to_string(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return "d" + std::to_string(raw64(*item));
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item ? "b1" : "b0";
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        return "s" + *item;
    }
    return "recursive";
}

std::string simulator_snapshot(
    const network::VirtualNetworkRequestSimulator& simulator) {
    std::ostringstream output;
    output << simulator.num_v_nets() << ':' << simulator.num_events() << '|';
    for (const network::VirtualNetwork& request : simulator.v_nets()) {
        output << request.request_id().value_or(-1) << ','
               << raw64(request.arrival_time().value_or(-1.0)) << ','
               << raw64(request.lifetime().value_or(-1.0)) << ','
               << request.graph().num_nodes() << ','
               << request.graph().num_edges() << ';';
        const auto cpu = request.bind_node_attribute("cpu");
        if (cpu) {
            const auto rows = network::get_node_attrs_data(
                request, {cpu->registry_id}, 1U);
            for (const AttrValue& item : rows.front()) {
                output << attr_scalar(item) << ',';
            }
        }
        const auto bandwidth = request.bind_link_attribute("bandwidth");
        if (bandwidth) {
            const auto rows = network::get_link_attrs_data(
                request, {bandwidth->registry_id}, 1U);
            for (const AttrValue& item : rows.front()) {
                output << attr_scalar(item) << ',';
            }
        }
        output << '|';
    }
    for (const network::VirtualNetworkEvent& event : simulator.events()) {
        output << event.id() << ','
               << static_cast<unsigned>(event.type()) << ','
               << event.virtual_network_id() << ','
               << raw64(event.time()) << ';';
    }
    return output.str();
}

std::pair<std::uint32_t, std::uint32_t> next_words(RandomContext& random) {
    return {
        random.python().getrandbits32(),
        random.numpy().next_uint32(),
    };
}

std::string quote_yaml(const fs::path& path) {
    std::string value = path.generic_string();
    std::string result = "\"";
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    result.push_back('"');
    return result;
}

Config make_config(
    const std::optional<std::uint32_t> root_seed = 17U,
    const std::size_t request_count = 8U,
    const bool with_attributes = true,
    const fs::path& save_root = "generator-unit-output") {
    std::ostringstream yaml;
    if (root_seed) {
        yaml << "seed: " << *root_seed << '\n';
    }
    yaml
        << "experiment:\n"
        << "  seed: 23\n"
        << "p_net_setting:\n"
        << "  topology:\n"
        << "    type: path\n"
        << "    num_nodes: 4\n"
        << "    wm_alpha: 0.5\n"
        << "    wm_beta: 0.2\n"
        << "  node_attrs_setting: []\n"
        << "  link_attrs_setting: []\n"
        << "  output:\n"
        << "    save_dir: " << quote_yaml(save_root / "physical") << '\n'
        << "v_sim_setting:\n"
        << "  num_v_nets: " << request_count << '\n'
        << "  topology:\n"
        << "    type: path\n"
        << "  v_net_size:\n"
        << "    distribution: uniform\n"
        << "    dtype: int\n"
        << "    low: 2\n"
        << "    high: 3\n"
        << "  arrival_rate:\n"
        << "    distribution: poisson\n"
        << "    dtype: float\n"
        << "    lam: 1.0\n"
        << "  lifetime:\n"
        << "    distribution: exponential\n"
        << "    dtype: float\n"
        << "    scale: 2\n";
    if (with_attributes) {
        yaml
            << "  node_attrs_setting:\n"
            << "    - name: cpu\n"
            << "      type: resource\n"
            << "      owner: node\n"
            << "      distribution: uniform\n"
            << "      dtype: int\n"
            << "      generative: true\n"
            << "      low: 0\n"
            << "      high: 3\n"
            << "  link_attrs_setting:\n"
            << "    - name: bandwidth\n"
            << "      type: resource\n"
            << "      owner: link\n"
            << "      distribution: uniform\n"
            << "      dtype: int\n"
            << "      generative: true\n"
            << "      low: 0\n"
            << "      high: 5\n";
    } else {
        yaml
            << "  node_attrs_setting: []\n"
            << "  link_attrs_setting: []\n";
    }
    yaml
        << "  output:\n"
        << "    save_dir: " << quote_yaml(save_root / "virtual") << '\n'
        << "    events_file_name: events.yaml\n"
        << "    setting_file_name: v_sim_setting.yaml\n";
    return Config(YAML::Load(yaml.str()));
}

network::GeneratorWorkers all_workers(const std::size_t width) {
    network::GeneratorWorkers result;
    result.physical_factory_workers = width;
    result.physical_attribute_workers = width;
    result.virtual_simulation.factory_workers = width;
    result.virtual_simulation.arrangement_workers = width;
    result.virtual_simulation.attribute_workers = width;
    result.virtual_simulation.event_workers = width;
    result.virtual_simulation.io_workers = width;
    return result;
}

void test_disabled_and_error_seed_order() {
    Config empty(YAML::Load("{}"));
    network::GeneratorSelection selection;
    selection.physical_network = false;
    selection.virtual_networks = false;
    RandomContext actual(71U);
    RandomContext expected(71U);
    const auto result = network::Generator::generate_dataset(
        empty, actual, selection);
    expect(!result.physical_network && !result.virtual_networks,
           "disabled selection result");
    expect(next_words(actual) == next_words(expected),
           "disabled selection touched RNG");

    Config missing(YAML::Load("seed: 99\n"));
    RandomContext missing_actual(72U);
    RandomContext missing_expected(72U);
    expect_error(
        [&] {
            static_cast<void>(
                network::Generator::generate_v_nets_dataset_from_config(
                    missing, missing_actual));
        },
        "missing subtree");
    expect(next_words(missing_actual) == next_words(missing_expected),
           "missing subtree reseeded before membership failure");

    Config present_null(YAML::Load("seed: 99\nv_sim_setting: null\n"));
    RandomContext null_actual(73U);
    expect_error(
        [&] {
            static_cast<void>(
                network::Generator::generate_v_nets_dataset_from_config(
                    present_null, null_actual));
        },
        "null subtree");
    RandomContext null_expected(99U);
    expect(next_words(null_actual) == next_words(null_expected),
           "present invalid subtree did not seed first");
}

void test_selection_seed_and_workers() {
    std::optional<std::string> baseline;
    std::optional<std::pair<std::uint32_t, std::uint32_t>> baseline_next;
    for (const std::size_t width : {0U, 1U, 2U, 8U}) {
        const Config config = make_config();
        RandomContext random(900U);
        network::GeneratorSelection selection;
        selection.physical_network = false;
        selection.virtual_networks = true;
        auto generated = network::Generator::generate_dataset(
            config, random, selection, all_workers(width));
        expect(!generated.physical_network && generated.virtual_networks,
               "virtual-only selection");
        const std::string current =
            simulator_snapshot(*generated.virtual_networks);
        const auto next = next_words(random);
        if (!baseline) {
            baseline = current;
            baseline_next = next;
        } else {
            expect(current == *baseline, "Generator worker output drift");
            expect(next == *baseline_next, "Generator worker RNG drift");
        }
    }

    const Config config = make_config();
    network::GeneratorSelection both;
    RandomContext both_random(1U);
    auto both_result = network::Generator::generate_dataset(
        config, both_random, both, all_workers(2U));
    expect(both_result.physical_network && both_result.virtual_networks,
           "both selection result");
    expect(both_result.physical_network->graph().num_nodes() == 4U &&
               both_result.physical_network->graph().num_edges() == 3U,
           "physical generated before virtual");

    network::GeneratorSelection virtual_only;
    virtual_only.physical_network = false;
    RandomContext virtual_random(2U);
    auto virtual_result = network::Generator::generate_dataset(
        config, virtual_random, virtual_only, all_workers(2U));
    expect(
        simulator_snapshot(*both_result.virtual_networks) ==
            simulator_snapshot(*virtual_result.virtual_networks),
        "second seed reset did not isolate virtual generation");
    expect(next_words(both_random) == next_words(virtual_random),
           "second seed reset continuation drift");

    network::GeneratorSelection physical_only;
    physical_only.virtual_networks = false;
    RandomContext physical_random(3U);
    auto physical_result = network::Generator::generate_dataset(
        config, physical_random, physical_only);
    expect(physical_result.physical_network && !physical_result.virtual_networks,
           "physical-only selection");
}

void test_absent_and_composed_seed_modes() {
    const Config without_seed = make_config(std::nullopt, 8U, false);
    RandomContext combined_random(81U);
    auto combined = network::Generator::generate_dataset(
        without_seed, combined_random);

    RandomContext explicit_random(81U);
    auto physical = network::Generator::generate_p_net_dataset_from_config(
        without_seed, explicit_random);
    auto virtuals = network::Generator::generate_v_nets_dataset_from_config(
        without_seed, explicit_random);
    expect(physical.graph().num_nodes() ==
               combined.physical_network->graph().num_nodes(),
           "absent-seed physical order");
    expect(simulator_snapshot(virtuals) ==
               simulator_snapshot(*combined.virtual_networks),
           "absent-seed stream continuation");
    expect(next_words(explicit_random) == next_words(combined_random),
           "absent-seed final continuation");

    const Config dual_seed = make_config(17U, 8U, false);
    RandomContext composed_random(0U);
    auto composed = network::Generator::generate_v_nets_dataset_from_config(
        dual_seed,
        composed_random,
        network::GeneratorPersistence::memory_only,
        {},
        network::GeneratorSeedMode::composed_experiment_seed);

    Config root_23 = make_config(23U, 8U, false);
    RandomContext root_random(0U);
    auto root = network::Generator::generate_v_nets_dataset_from_config(
        root_23, root_random);
    expect(simulator_snapshot(composed) == simulator_snapshot(root),
           "explicit composed seed path");
    expect(next_words(composed_random) == next_words(root_random),
           "explicit composed seed continuation");
}

void expect_event_index(
    const network::VirtualNetworkRequestSimulator& simulator) {
    expect(simulator.num_events() == simulator.num_v_nets() * 2U,
           "changeable event cardinality");
    double previous = -1.0;
    for (const network::VirtualNetworkEvent& event : simulator.events()) {
        expect(event.time() >= previous, "changeable event order");
        previous = event.time();
        expect(simulator.event_id(event.virtual_network_id(), event.type()) ==
                   std::optional<std::size_t>{event.id()},
               "changeable compact event index");
    }
}

void test_changeable_and_validation() {
    std::optional<std::string> baseline;
    std::optional<std::pair<std::uint32_t, std::uint32_t>> baseline_next;
    for (const std::size_t width : {0U, 1U, 2U, 8U}) {
        const Config config = make_config(41U, 8U, true);
        RandomContext random(0U);
        auto simulator =
            network::Generator::generate_changeable_v_nets_dataset_from_config(
                config,
                random,
                network::GeneratorPersistence::memory_only,
                all_workers(width));
        expect(simulator.num_v_nets() == 8U, "changeable request count");
        for (std::size_t index = 0U; index < simulator.num_v_nets(); ++index) {
            expect(
                simulator.v_nets()[index].request_id() ==
                    std::optional<std::int64_t>{
                        static_cast<std::int64_t>(index)},
                "changeable quarter request ID");
        }
        expect_event_index(simulator);
        const std::string current = simulator_snapshot(simulator);
        const auto next = next_words(random);
        if (!baseline) {
            baseline = current;
            baseline_next = next;
        } else {
            expect(current == *baseline, "changeable worker output drift");
            expect(next == *baseline_next, "changeable worker RNG drift");
        }
    }

    const Config invalid = make_config(91U, 6U, false);
    RandomContext actual(0U);
    expect_error(
        [&] {
            static_cast<void>(
                network::Generator::generate_changeable_v_nets_dataset_from_config(
                    invalid, actual));
        },
        "changeable invalid divisibility");
    RandomContext expected(91U);
    expect(next_words(actual) == next_words(expected),
           "changeable validation seed order");
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = fs::temp_directory_path() /
            ("virne_generator_unit_" + std::to_string(raw64(17.25)));
        std::error_code error;
        fs::remove_all(path_, error);
        fs::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path& path() const noexcept {
        return path_;
    }

private:
    fs::path path_;
};

const utils::SettingValue& setting_value(
    const utils::SettingObject& object,
    const std::string_view key) {
    const auto id = object.find_key_id(key);
    if (!id) {
        fail("missing saved setting key " + std::string(key));
    }
    return object.at(*id);
}

void test_changeable_save_path_and_original_setting() {
    TemporaryDirectory temporary;
    const Config config = make_config(53U, 8U, false, temporary.path());
    RandomContext random(0U);
    auto simulator =
        network::Generator::generate_changeable_v_nets_dataset_from_config(
            config,
            random,
            network::GeneratorPersistence::save,
            all_workers(2U));
    expect(simulator.num_v_nets() == 8U, "saved changeable request count");

    const fs::path save_root = temporary.path() / "virtual";
    std::vector<fs::path> datasets;
    for (const fs::directory_entry& entry : fs::directory_iterator(save_root)) {
        if (entry.is_directory()) {
            datasets.push_back(entry.path());
        }
    }
    expect(datasets.size() == 1U, "changeable save directory count");
    expect(datasets.front().filename().string().find("[2-6]") !=
               std::string::npos,
           "changeable save path did not use stage-four size high");
    expect(fs::is_regular_file(datasets.front() / "events.yaml") &&
               fs::is_regular_file(datasets.front() / "v_sim_setting.yaml"),
           "changeable save files");

    const utils::SettingDocument saved = utils::read_setting(
        (datasets.front() / "v_sim_setting.yaml").string());
    const utils::SettingObject& root = saved.root.as_object();
    const utils::SettingObject& size =
        setting_value(root, "v_net_size").as_object();
    expect(
        setting_value(size, "high").as_integer().convert_to<std::int64_t>() ==
            3,
        "changeable saved setting was not original");
}

void test_config_loader_integration() {
    Config config = ConfigLoader::load(
        std::string(VIRNE_SOURCE_DIR) + "/setting/main.yaml");
    expect(config.contains("p_net_setting") &&
               config.contains("v_sim_setting"),
           "ConfigLoader composed subtrees");
    override_parser::apply(config, "v_sim_setting.num_v_nets=4");
    override_parser::apply(config, "v_sim_setting.topology.type=path");
    RandomContext random(0U);
    auto simulator = network::Generator::generate_v_nets_dataset_from_config(
        config,
        random,
        network::GeneratorPersistence::memory_only,
        all_workers(2U),
        network::GeneratorSeedMode::composed_experiment_seed);
    expect(simulator.num_v_nets() == 4U && simulator.num_events() == 8U,
           "ConfigLoader Generator integration");
}

}  // namespace

int main() {
    try {
        test_disabled_and_error_seed_order();
        test_selection_seed_and_workers();
        test_absent_and_composed_seed_modes();
        test_changeable_and_validation();
        test_changeable_save_path_and_original_setting();
        test_config_loader_integration();
        std::cout << "dataset Generator unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dataset Generator unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
