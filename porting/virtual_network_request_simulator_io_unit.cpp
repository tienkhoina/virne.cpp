#include "virtual_network_request_simulator.h"

#include "random_context.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace network = virne::network;
namespace utils = virne::utils;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Callable>
network::VirtualNetworkSimulatorException expect_simulator_error(
    Callable&& callable,
    network::VirtualNetworkSimulatorErrorCode code,
    network::VirtualNetworkSimulatorOperation operation,
    std::string_view context) {
    try {
        std::forward<Callable>(callable)();
    } catch (const network::VirtualNetworkSimulatorException& error) {
        expect(error.code() == code, std::string(context) + ": code drift");
        expect(
            error.operation() == operation,
            std::string(context) + ": operation drift");
        return error;
    }
    fail(std::string(context) + ": expected simulator exception");
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        base_ = fs::temp_directory_path();
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        for (std::uint32_t attempt = 0U; attempt < 256U; ++attempt) {
            const fs::path candidate = base_ /
                ("virne_simulator_io_unit_" + std::to_string(stamp) + "_" +
                 std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(candidate, error)) {
                path_ = candidate;
                return;
            }
            if (error) {
                throw fs::filesystem_error(
                    "reserve simulator unit directory", candidate, error);
            }
        }
        fail("unable to reserve simulator unit directory");
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory() {
        if (!path_.empty() && path_.parent_path() == base_) {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }
    }

    const fs::path& path() const noexcept { return path_; }

private:
    fs::path base_;
    fs::path path_;
};

std::string setting_yaml(
    std::size_t count = 2U,
    std::string_view events_name = "events.yaml",
    std::string_view setting_name = "v_sim_setting.yaml",
    bool include_attribute = false) {
    std::string result =
        "num_v_nets: " + std::to_string(count) + "\n"
        "topology:\n"
        "  type: path\n"
        "v_net_size:\n"
        "  distribution: uniform\n"
        "  dtype: int\n"
        "  low: 2\n"
        "  high: 2\n"
        "arrival_rate:\n"
        "  distribution: poisson\n"
        "  dtype: float\n"
        "  lam: 0.0\n"
        "lifetime:\n"
        "  distribution: uniform\n"
        "  dtype: float\n"
        "  low: 3.0\n"
        "  high: 3.0\n";
    if (include_attribute) {
        result +=
            "node_attrs_setting:\n"
            "  - name: cpu\n"
            "    owner: node\n"
            "    type: resource\n"
            "    generative: true\n"
            "    distribution: uniform\n"
            "    dtype: int\n"
            "    low: 1\n"
            "    high: 1\n";
    } else {
        result += "node_attrs_setting: []\n";
    }
    result +=
        "link_attrs_setting: []\n"
        "output:\n"
        "  save_dir: saved-root\n"
        "  events_file_name: " + std::string(events_name) + "\n"
        "  setting_file_name: " + std::string(setting_name) + "\n";
    return result;
}

utils::SettingDocument setting_document(
    std::size_t count = 2U,
    std::string_view events_name = "events.yaml",
    std::string_view setting_name = "v_sim_setting.yaml",
    bool include_attribute = false) {
    return utils::parse_setting(
        setting_yaml(count, events_name, setting_name, include_attribute),
        utils::SettingFormat::yaml);
}

network::VirtualNetworkRequestSimulator generated_simulator(
    const utils::SettingDocument& setting,
    std::uint32_t seed = 19U,
    std::size_t workers = 1U) {
    RandomContext random(seed);
    auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
        setting, random, std::nullopt);
    network::VirtualSimulationWorkers configured;
    configured.factory_workers = workers;
    configured.arrangement_workers = workers;
    configured.attribute_workers = workers;
    configured.event_workers = workers;
    simulator.renew(random, true, true, std::nullopt, configured);
    return simulator;
}

std::string read_bytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        fail("unable to read fixture: " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

void expect_same_state(
    const network::VirtualNetworkRequestSimulator& left,
    const network::VirtualNetworkRequestSimulator& right,
    std::string_view context) {
    expect(left.num_v_nets() == right.num_v_nets(),
           std::string(context) + ": request count");
    expect(left.events() == right.events(),
           std::string(context) + ": event drift");
    for (std::size_t index = 0U; index < left.num_v_nets(); ++index) {
        const auto& lhs = left.v_nets()[index];
        const auto& rhs = right.v_nets()[index];
        expect(lhs.request_id() == rhs.request_id(),
               std::string(context) + ": id drift");
        expect(lhs.arrival_time() == rhs.arrival_time(),
               std::string(context) + ": arrival drift");
        expect(lhs.lifetime() == rhs.lifetime(),
               std::string(context) + ": lifetime drift");
        expect(lhs.live_num_nodes() == rhs.live_num_nodes(),
               std::string(context) + ": node drift");
        expect(lhs.live_num_links() == rhs.live_num_links(),
               std::string(context) + ": link drift");
    }
}

network::VirtualNetwork make_duplicate_network(
    std::int64_t marker) {
    network::BaseNetworkConstruction construction;
    construction.config.graph_attributes = {
        {"id", std::int64_t{7}},
        {"arrival_time", 0.0},
        {"lifetime", 1.0},
        {"marker", marker},
    };
    network::VirtualNetwork result(std::move(construction));
    network::TopologyRequest request;
    request.type = network::TopologyType::Path;
    request.num_nodes = 2;
    result.generate_topology(request);
    return result;
}

void test_raw_decode_and_seed_order() {
    const auto document = setting_document(3U, "custom-events.yaml",
                                           "custom-setting.yaml", true);
    const auto config =
        network::virtual_network_simulation_config_from_setting(document);
    expect(config.num_virtual_networks == 3U, "decoded request count");
    expect(config.topology_type == network::TopologyType::Path,
           "decoded topology enum");
    expect(config.virtual_network_size.value_kind ==
               utils::DatasetValueKind::integer,
           "decoded size dtype");
    expect(config.node_attribute_specs.size() == 1U,
           "decoded attribute specs");
    expect(config.node_attribute_specs[0].name == "cpu",
           "decoded attribute name");
    expect(config.output.events_file_name == "custom-events.yaml",
           "decoded event filename");
    expect(config.output.save_dir ==
               std::optional<std::string>{"saved-root"},
           "decoded save directory");
    expect(config.source_setting.has_value(), "missing deep source snapshot");
    expect(config.source_setting->root.object_ptr().get() !=
               document.root.object_ptr().get(),
           "source setting was shallow copied");

    utils::SettingDocument invalid = utils::parse_setting(
        "num_v_nets: 1\n", utils::SettingFormat::yaml);
    RandomContext actual(3U);
    try {
        static_cast<void>(
            network::VirtualNetworkRequestSimulator::from_setting(
                invalid, actual, 77U));
        fail("invalid raw config unexpectedly decoded");
    } catch (const std::exception&) {
    }
    RandomContext expected(77U);
    expect(actual.python().getrandbits32() ==
               expected.python().getrandbits32(),
           "seed-before-decode Python continuation");
    expect(actual.numpy().next_uint32() == expected.numpy().next_uint32(),
           "seed-before-decode NumPy continuation");
}

void test_roundtrip_workers_and_bytes(const fs::path& root) {
    const auto document = setting_document();
    auto simulator = generated_simulator(document);
    std::optional<std::string> sequential_events;
    std::optional<std::string> sequential_network;
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        const fs::path directory = root / ("roundtrip_w" +
            std::to_string(workers));
        simulator.save_dataset(directory, workers);
        expect(fs::exists(directory / "v_nets" / "v_net-00000.gml"),
               "missing first request file");
        expect(fs::exists(directory / "v_nets" / "v_net-00001.gml"),
               "missing second request file");
        expect(fs::exists(directory / "events.yaml"),
               "missing events file");
        expect(fs::exists(directory / "v_sim_setting.yaml"),
               "missing setting file");

        const std::string event_bytes = read_bytes(directory / "events.yaml");
        const std::string network_bytes = read_bytes(
            directory / "v_nets" / "v_net-00000.gml");
        if (!sequential_events) {
            sequential_events = event_bytes;
            sequential_network = network_bytes;
        } else {
            expect(event_bytes == *sequential_events,
                   "save worker event-byte drift");
            expect(network_bytes == *sequential_network,
                   "save worker GML-byte drift");
        }
        for (const fs::directory_entry& entry :
             fs::directory_iterator(directory / "v_nets")) {
            expect(entry.path().filename().string().find(
                       ".virne-save-stage-") == std::string::npos,
                   "staging directory leaked");
        }

        network::VirtualNetworkDatasetCache cache;
        auto loaded = network::VirtualNetworkRequestSimulator::load_dataset(
            directory, cache, workers);
        expect_same_state(simulator, loaded, "roundtrip");
        expect(cache.size() == 1U, "successful load not cached");
        expect(loaded.config().source_setting->root.object_ptr().get() !=
                   simulator.config().source_setting->root.object_ptr().get(),
               "roundtrip source setting aliases original");
    }
}

void test_duplicate_ordered_commit(const fs::path& root) {
    auto config = network::virtual_network_simulation_config_from_setting(
        setting_document(2U));
    std::vector<network::VirtualNetwork> requests;
    requests.push_back(make_duplicate_network(11));
    requests.push_back(make_duplicate_network(22));
    std::vector<network::VirtualNetworkEvent> events;
    events.emplace_back(0U, network::VirtualEventType::arrival, 7, 0.0);
    events.emplace_back(1U, network::VirtualEventType::arrival, 7, 0.0);
    events.emplace_back(2U, network::VirtualEventType::leave, 7, 1.0);
    events.emplace_back(3U, network::VirtualEventType::leave, 7, 1.0);
    auto simulator = network::VirtualNetworkRequestSimulator::from_state(
        std::move(config), std::move(requests), std::move(events));
    const fs::path directory = root / "duplicates";
    simulator.save_dataset(directory, 8U);
    expect(fs::exists(directory / "v_nets" / "v_net-00007.gml"),
           "duplicate target missing");
    expect(std::distance(
               fs::directory_iterator(directory / "v_nets"),
               fs::directory_iterator{}) == 1,
           "duplicate IDs did not overwrite one target");
    network::BaseNetwork loaded = network::BaseNetwork::from_gml(
        (directory / "v_nets" / "v_net-00007.gml").string());
    const AttrId marker = loaded.bind_graph_attribute("marker");
    expect(std::get<std::int64_t>(loaded.graph_attribute(marker)) == 22,
           "parallel duplicate commit was not last-write-wins");
}

void test_cache_and_hardcoded_load_names(const fs::path& root) {
    const auto document = setting_document();
    auto simulator = generated_simulator(document);
    const fs::path cached_directory = root / "seed_cached";
    simulator.save_dataset(cached_directory, 2U);
    network::VirtualNetworkDatasetCache cache;
    auto first = network::VirtualNetworkRequestSimulator::load_dataset(
        cached_directory, cache, 2U);
    fs::remove(cached_directory / "events.yaml");
    auto second = network::VirtualNetworkRequestSimulator::load_dataset(
        cached_directory, cache, 8U);
    expect_same_state(first, second, "cached reload");
    expect(&first.v_nets()[0].graph() != &second.v_nets()[0].graph(),
           "cache returned shared graph");
    expect(first.config().source_setting->root.object_ptr().get() !=
               second.config().source_setting->root.object_ptr().get(),
           "cache returned shared setting");

    const fs::path ordinary = root / "ordinary_cache_write_only";
    simulator.save_dataset(ordinary, 1U);
    network::VirtualNetworkDatasetCache ordinary_cache;
    static_cast<void>(network::VirtualNetworkRequestSimulator::load_dataset(
        ordinary, ordinary_cache, 1U));
    fs::remove(ordinary / "events.yaml");
    expect_simulator_error(
        [&] {
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    ordinary, ordinary_cache, 1U));
        },
        network::VirtualNetworkSimulatorErrorCode::invalid_dataset_layout,
        network::VirtualNetworkSimulatorOperation::load_layout,
        "non-seed cache read");

    const auto custom = setting_document(
        2U, "custom-events.yaml", "custom-setting.yaml");
    auto custom_simulator = generated_simulator(custom);
    const fs::path custom_directory = root / "custom_names";
    custom_simulator.save_dataset(custom_directory, 1U);
    expect(fs::exists(custom_directory / "custom-events.yaml"),
           "custom save events name ignored");
    expect(fs::exists(custom_directory / "custom-setting.yaml"),
           "custom save setting name ignored");
    network::VirtualNetworkDatasetCache custom_cache;
    expect_simulator_error(
        [&] {
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    custom_directory, custom_cache, 1U));
        },
        network::VirtualNetworkSimulatorErrorCode::invalid_dataset_layout,
        network::VirtualNetworkSimulatorOperation::load_layout,
        "hard-coded load names");
}

void test_errors_and_all_entries(const fs::path& root) {
    network::VirtualNetworkDatasetCache cache;
    const fs::path missing = root / "missing";
    auto error = expect_simulator_error(
        [&] {
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    missing, cache, 1U));
        },
        network::VirtualNetworkSimulatorErrorCode::invalid_dataset_layout,
        network::VirtualNetworkSimulatorOperation::load_layout,
        "missing root");
    expect(error.path() == missing, "missing-root error path");

    const fs::path partial = root / "partial";
    fs::create_directory(partial);
    error = expect_simulator_error(
        [&] {
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    partial, cache, 1U));
        },
        network::VirtualNetworkSimulatorErrorCode::invalid_dataset_layout,
        network::VirtualNetworkSimulatorOperation::load_layout,
        "missing v_nets");
    expect(error.path().filename() == "v_nets", "layout check order drift");

    auto simulator = generated_simulator(setting_document());
    const fs::path mismatch = root / "event_mismatch";
    simulator.save_dataset(mismatch, 1U);
    utils::SettingDocument events = utils::read_setting(
        (mismatch / "events.yaml").string());
    events.root.as_list().pop_back();
    utils::write_setting_strict(
        events,
        (mismatch / "events.yaml").string(),
        utils::SettingMode::write_update);
    expect_simulator_error(
        [&] {
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    mismatch, cache, 2U));
        },
        network::VirtualNetworkSimulatorErrorCode::event_count_mismatch,
        network::VirtualNetworkSimulatorOperation::validate_dataset,
        "event count mismatch");

    const fs::path junk = root / "all_entries";
    simulator.save_dataset(junk, 1U);
    fs::create_directory(junk / "v_nets" / "000-not-gml");
    error = expect_simulator_error(
        [&] {
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    junk, cache, 8U));
        },
        network::VirtualNetworkSimulatorErrorCode::io_failure,
        network::VirtualNetworkSimulatorOperation::load_networks,
        "all directory entries");
    expect(error.input_index() == std::optional<std::size_t>{0U},
           "parallel lowest load-error index");

    auto typed = network::virtual_network_simulation_config_from_setting(
        setting_document());
    typed.source_setting.reset();
    auto no_source =
        network::VirtualNetworkRequestSimulator::from_setting(std::move(typed));
    expect_simulator_error(
        [&] { no_source.save_setting(root / "missing_source.yaml"); },
        network::VirtualNetworkSimulatorErrorCode::missing_source_setting,
        network::VirtualNetworkSimulatorOperation::save_setting,
        "missing source setting");
}

void test_cache_concurrency() {
    auto simulator = generated_simulator(setting_document());
    network::VirtualNetworkDatasetCache cache;
    cache.store("seed_shared", simulator);
    std::vector<std::future<std::size_t>> futures;
    for (std::size_t worker = 0U; worker < 8U; ++worker) {
        futures.push_back(std::async(
            std::launch::async,
            [&cache, &simulator, worker] {
                for (std::size_t iteration = 0U; iteration < 16U; ++iteration) {
                    auto found = cache.find("seed_shared");
                    if (!found || found->num_events() != 4U) {
                        return std::size_t{0U};
                    }
                    cache.store(
                        "ordinary_" + std::to_string(worker), simulator);
                }
                return std::size_t{1U};
            }));
    }
    for (auto& future : futures) {
        expect(future.get() == 1U, "concurrent cache clone drift");
    }
    expect(cache.size() == 9U, "concurrent cache size drift");
}

}  // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        test_raw_decode_and_seed_order();
        test_roundtrip_workers_and_bytes(temporary.path());
        test_duplicate_ordered_commit(temporary.path());
        test_cache_and_hardcoded_load_names(temporary.path());
        test_errors_and_all_entries(temporary.path());
        test_cache_concurrency();
        std::cout << "VirtualNetworkRequestSimulator I/O unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkRequestSimulator I/O unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
