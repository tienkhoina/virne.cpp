#include "virtual_network_request_simulator.h"

#include "random_context.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace network = virne::network;
namespace utils = virne::utils;

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

void emit(const std::string_view name, const std::string& payload) {
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        base_ = fs::temp_directory_path();
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        for (std::uint32_t attempt = 0U; attempt < 256U; ++attempt) {
            const fs::path candidate = base_ /
                ("virne_sim_io_diff_" + std::to_string(stamp) + "_" +
                 std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(candidate, error)) {
                path_ = candidate;
                return;
            }
            if (error) {
                throw fs::filesystem_error(
                    "reserve simulator differential directory",
                    candidate,
                    error);
            }
        }
        throw std::runtime_error("unable to reserve simulator differential directory");
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
    const std::string_view events_name = "events.yaml",
    const std::string_view setting_name = "v_sim_setting.yaml") {
    return
        "num_v_nets: 2\n"
        "v_net_size:\n"
        "  distribution: uniform\n"
        "  dtype: int\n"
        "  low: 2\n"
        "  high: 2\n"
        "lifetime:\n"
        "  distribution: uniform\n"
        "  dtype: float\n"
        "  low: 3.0\n"
        "  high: 3.0\n"
        "arrival_rate:\n"
        "  distribution: poisson\n"
        "  dtype: float\n"
        "  lam: 0.0\n"
        "topology:\n"
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n"
        "output:\n"
        "  events_file_name: " + std::string(events_name) + "\n"
        "  setting_file_name: " + std::string(setting_name) + "\n";
}

utils::SettingDocument make_setting(
    const std::string_view events_name = "events.yaml",
    const std::string_view setting_name = "v_sim_setting.yaml") {
    return utils::parse_setting(
        setting_yaml(events_name, setting_name), utils::SettingFormat::yaml);
}

network::VirtualSimulationWorkers all_workers(const std::size_t workers) {
    network::VirtualSimulationWorkers result;
    result.factory_workers = workers;
    result.arrangement_workers = workers;
    result.attribute_workers = workers;
    result.event_workers = workers;
    result.io_workers = workers;
    return result;
}

network::VirtualNetworkRequestSimulator make_generated(
    const utils::SettingDocument& setting,
    const std::size_t workers = 1U) {
    RandomContext random(19U);
    auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
        setting, random, std::nullopt);
    simulator.renew(
        random, true, true, std::nullopt, all_workers(workers));
    return simulator;
}

std::string json_file(const fs::path& path) {
    return utils::dump_setting(
        utils::read_setting(path.string(), utils::SettingMode::read_update),
        utils::SettingFormat::json);
}

std::string graph_payload(const Graph& graph) {
    std::string result = "N=[";
    auto [node, node_end] = graph.nodes();
    bool first = true;
    for (; node != node_end; ++node) {
        if (!first) {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(*node);
    }
    result += "];E=[";
    auto [edge, edge_end] = graph.edges();
    first = true;
    for (; edge != edge_end; ++edge) {
        if (!first) {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(graph.source(*edge)) + "-" +
            std::to_string(graph.target(*edge));
    }
    result.push_back(']');
    return result;
}

std::string networks_payload(
    const std::vector<network::VirtualNetwork>& values) {
    std::string result = "[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            result.push_back('/');
        }
        const auto& value = values[index];
        if (!value.request_id() || !value.arrival_time() || !value.lifetime()) {
            throw std::runtime_error("loaded virtual network metadata is missing");
        }
        result += "id=" + std::to_string(*value.request_id()) +
            ",arrival=" + double_token(*value.arrival_time()) +
            ",lifetime=" + double_token(*value.lifetime()) +
            ",max=" +
            (value.max_latency() ? double_token(*value.max_latency()) : "none") +
            "," + graph_payload(value.graph());
    }
    result.push_back(']');
    return result;
}

std::string filenames_payload(const fs::path& directory) {
    std::vector<std::string> names;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    std::string result = "[";
    for (std::size_t index = 0U; index < names.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += names[index];
    }
    result.push_back(']');
    return result;
}

std::string dataset_payload(
    const fs::path& directory,
    const network::VirtualNetworkRequestSimulator& loaded) {
    return "setting=" + json_file(directory / "v_sim_setting.yaml") +
        ";events=" + json_file(directory / "events.yaml") +
        ";gml_files=" + filenames_payload(directory / "v_nets") +
        ";networks=" + networks_payload(loaded.v_nets());
}

std::string layout_target(
    const fs::path& directory,
    const std::size_t workers = 1U) {
    network::VirtualNetworkDatasetCache cache;
    try {
        static_cast<void>(
            network::VirtualNetworkRequestSimulator::load_dataset(
                directory, cache, workers));
    } catch (const network::VirtualNetworkSimulatorException& error) {
        if (error.code() !=
                network::VirtualNetworkSimulatorErrorCode::invalid_dataset_layout ||
            error.operation() !=
                network::VirtualNetworkSimulatorOperation::load_layout) {
            throw;
        }
        return error.path() == directory
            ? "root"
            : error.path().filename().string();
    }
    throw std::runtime_error("expected dataset layout error");
}

void emit_raw_decode_snapshot(const fs::path& root) {
    auto setting = make_setting();
    RandomContext random(19U);
    auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
        setting, random, std::nullopt);
    auto& object = setting.root.as_object();
    const auto count_id = object.find_key_id("num_v_nets");
    if (!count_id) {
        throw std::runtime_error("raw setting count ID is missing");
    }
    object.set(*count_id, utils::SettingValue(std::int64_t{99}));
    const fs::path saved = root / "snapshot.yaml";
    simulator.save_setting(saved);

    const auto& config = simulator.config();
    const bool decoded =
        config.num_virtual_networks == 2U &&
        config.topology_type == network::TopologyType::Path &&
        config.virtual_network_size.value_kind ==
            utils::DatasetValueKind::integer &&
        config.virtual_network_size.distribution.kind ==
            utils::DistributionKind::uniform &&
        config.node_attribute_specs.empty() &&
        config.link_attribute_specs.empty() &&
        config.output.events_file_name == "events.yaml" &&
        config.output.setting_file_name == "v_sim_setting.yaml";
    emit(
        "raw_decode_snapshot",
        std::string("decoded=") + (decoded ? "1" : "0") +
            ";snapshot=" + json_file(saved));
}

void emit_worker_roundtrips(const fs::path& root) {
    const auto setting = make_setting();
    auto simulator = make_generated(setting, 1U);
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        const fs::path directory = root / ("roundtrip_w" +
            std::to_string(workers));
        simulator.save_dataset(directory, workers);
        network::VirtualNetworkDatasetCache cache;
        auto loaded = network::VirtualNetworkRequestSimulator::load_dataset(
            directory, cache, workers);
        emit(
            "save_load_w" + std::to_string(workers),
            dataset_payload(directory, loaded));
    }
}

void emit_custom_names(const fs::path& root) {
    const auto setting = make_setting(
        "custom-events.yaml", "custom-setting.yaml");
    auto simulator = make_generated(setting);
    const fs::path directory = root / "custom_names";
    simulator.save_dataset(directory, 2U);
    emit(
        "hardcoded_load_names",
        "files=" + filenames_payload(directory) +
            ";load=layout:" + layout_target(directory, 2U));
}

void emit_layout_order(const fs::path& root) {
    const fs::path missing_root = root / "missing_root";
    emit("layout_missing_root", "layout:" + layout_target(missing_root));

    const fs::path missing_vnets = root / "missing_vnets";
    fs::create_directory(missing_vnets);
    emit("layout_missing_vnets", "layout:" + layout_target(missing_vnets));

    const fs::path missing_events = root / "missing_events";
    fs::create_directories(missing_events / "v_nets");
    emit("layout_missing_events", "layout:" + layout_target(missing_events));

    const fs::path missing_setting = root / "missing_setting";
    fs::create_directories(missing_setting / "v_nets");
    const auto empty_events = utils::parse_setting(
        "[]\n", utils::SettingFormat::yaml);
    utils::write_setting_strict(
        empty_events,
        (missing_setting / "events.yaml").string(),
        utils::SettingMode::write_update);
    emit("layout_missing_setting", "layout:" + layout_target(missing_setting));
}

void emit_event_errors(const fs::path& root) {
    auto simulator = make_generated(make_setting());

    const fs::path mismatch = root / "event_cardinality";
    simulator.save_dataset(mismatch, 1U);
    auto events = utils::read_setting(
        (mismatch / "events.yaml").string(),
        utils::SettingMode::read_update);
    events.root.as_list().pop_back();
    utils::write_setting_strict(
        events,
        (mismatch / "events.yaml").string(),
        utils::SettingMode::write_update);
    try {
        network::VirtualNetworkDatasetCache cache;
        static_cast<void>(network::VirtualNetworkRequestSimulator::load_dataset(
            mismatch, cache, 2U));
        throw std::runtime_error("expected event cardinality error");
    } catch (const network::VirtualNetworkSimulatorException& error) {
        if (error.code() !=
                network::VirtualNetworkSimulatorErrorCode::event_count_mismatch ||
            error.operation() !=
                network::VirtualNetworkSimulatorOperation::validate_dataset) {
            throw;
        }
        emit("event_cardinality", "event_count_mismatch");
    }

    const fs::path invalid = root / "invalid_event";
    simulator.save_dataset(invalid, 1U);
    const auto invalid_events = utils::parse_setting(
        "- id: 0\n  type: 2\n  v_net_id: 0\n  time: 0.0\n",
        utils::SettingFormat::yaml);
    utils::write_setting_strict(
        invalid_events,
        (invalid / "events.yaml").string(),
        utils::SettingMode::write_update);
    try {
        network::VirtualNetworkDatasetCache cache;
        static_cast<void>(network::VirtualNetworkRequestSimulator::load_dataset(
            invalid, cache, 1U));
        throw std::runtime_error("expected invalid event setting error");
    } catch (const network::VirtualNetworkSimulatorException& error) {
        if (error.code() !=
                network::VirtualNetworkSimulatorErrorCode::invalid_event_setting ||
            error.operation() !=
                network::VirtualNetworkSimulatorOperation::load_events) {
            throw;
        }
        emit("invalid_event_setting", "invalid_event_setting");
    }
}

void emit_sorted_entry_errors(const fs::path& root) {
    auto simulator = make_generated(make_setting());
    for (const std::size_t workers : {1U, 8U}) {
        const fs::path directory = root / ("sorted_error_w" +
            std::to_string(workers));
        simulator.save_dataset(directory, 1U);
        fs::create_directory(directory / "v_nets" / "000-bad");
        fs::create_directory(directory / "v_nets" / "zzz-bad");
        try {
            network::VirtualNetworkDatasetCache cache;
            static_cast<void>(
                network::VirtualNetworkRequestSimulator::load_dataset(
                    directory, cache, workers));
            throw std::runtime_error("expected sorted entry load error");
        } catch (const network::VirtualNetworkSimulatorException& error) {
            if (error.code() !=
                    network::VirtualNetworkSimulatorErrorCode::io_failure ||
                error.operation() !=
                    network::VirtualNetworkSimulatorOperation::load_networks ||
                !error.input_index()) {
                throw;
            }
            emit(
                "sorted_entry_error_w" + std::to_string(workers),
                "first=" + error.path().filename().string() +
                    ";index=" + std::to_string(*error.input_index()));
        }
    }
}

void emit_cache_cases(const fs::path& root) {
    auto simulator = make_generated(make_setting());

    const fs::path seeded = root / "seed_cached";
    simulator.save_dataset(seeded, 2U);
    network::VirtualNetworkDatasetCache seeded_cache;
    auto first = network::VirtualNetworkRequestSimulator::load_dataset(
        seeded, seeded_cache, 2U);
    fs::remove(seeded / "events.yaml");
    auto second = network::VirtualNetworkRequestSimulator::load_dataset(
        seeded, seeded_cache, 8U);
    const bool same = networks_payload(first.v_nets()) ==
        networks_payload(second.v_nets());
    const bool independent_graph =
        &first.v_nets()[0].graph() != &second.v_nets()[0].graph();
    const bool independent_setting =
        first.config().source_setting && second.config().source_setting &&
        first.config().source_setting->root.object_ptr().get() !=
            second.config().source_setting->root.object_ptr().get();
    emit(
        "cache_seed",
        std::string("reload=ok;same=") + (same ? "1" : "0") +
            ";graph_independent=" + (independent_graph ? "1" : "0") +
            ";setting_independent=" + (independent_setting ? "1" : "0") +
            ";stored=" + (seeded_cache.size() == 1U ? "1" : "0"));

    const fs::path ordinary = root / "ordinary_cache";
    simulator.save_dataset(ordinary, 1U);
    network::VirtualNetworkDatasetCache ordinary_cache;
    static_cast<void>(network::VirtualNetworkRequestSimulator::load_dataset(
        ordinary, ordinary_cache, 1U));
    fs::remove(ordinary / "events.yaml");
    emit(
        "cache_nonseed",
        "reload=layout:" + layout_target(ordinary, 8U) +
            ";stored=" + (ordinary_cache.size() == 1U ? "1" : "0"));
}

void emit_cases(const fs::path& root) {
    emit_raw_decode_snapshot(root);
    emit_worker_roundtrips(root);
    emit_custom_names(root);
    emit_layout_order(root);
    emit_event_errors(root);
    emit_sorted_entry_errors(root);
    emit_cache_cases(root);
}

}  // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        emit_cases(temporary.path());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkRequestSimulator I/O harness: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
