#include "physical_network.h"

#include "random_context.h"
#include "setting.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;
namespace utils = virne::utils;

std::string hex_text(const std::string_view value)
{
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

std::string double_token(const double value)
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

void emit(const std::string_view name, const std::string& payload)
{
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

utils::SettingDocument yaml(const std::string& bytes)
{
    return utils::parse_setting(bytes, utils::SettingFormat::yaml);
}

std::string quote_yaml(const std::filesystem::path& path)
{
    std::string escaped;
    const std::string value = path.generic_string();
    escaped.reserve(value.size());
    for (const char item : value) {
        escaped.push_back(item);
        if (item == '\'') {
            escaped.push_back('\'');
        }
    }
    return "'" + escaped + "'";
}

std::string generated_config()
{
    return
        "topology:\n"
        "  num_nodes: 5\n"
        "  type: path\n"
        "node_attrs_setting:\n"
        "  - name: cpu\n"
        "    owner: node\n"
        "    type: resource\n"
        "    generative: true\n"
        "    distribution: uniform\n"
        "    dtype: int\n"
        "    low: 1\n"
        "    high: 9\n"
        "link_attrs_setting:\n"
        "  - name: bandwidth\n"
        "    owner: link\n"
        "    type: resource\n"
        "    generative: true\n"
        "    distribution: uniform\n"
        "    dtype: int\n"
        "    low: 10\n"
        "    high: 19\n";
}

std::string integer_vector_token(const std::vector<AttrValue>& values)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += std::to_string(std::get<std::int64_t>(values[index]));
    }
    result.push_back(']');
    return result;
}

std::string generated_payload(
    const network::PhysicalNetwork& value,
    RandomContext& random,
    const std::size_t workers)
{
    const auto cpu = value.bind_node_attribute("cpu");
    const auto bandwidth = value.bind_link_attribute("bandwidth");
    if (!cpu || !bandwidth) {
        throw std::runtime_error("generated attribute binding failed");
    }
    const auto node_rows = network::get_node_attrs_data(
        value, {cpu->registry_id}, workers);
    const auto link_rows = network::get_link_attrs_data(
        value, {bandwidth->registry_id}, workers);
    if (node_rows.size() != 1U || link_rows.size() != 1U) {
        throw std::runtime_error("generated row count drift");
    }
    return
        "nodes=" + std::to_string(value.live_num_nodes()) +
        ";links=" + std::to_string(value.live_num_links()) +
        ";cpu=" + integer_vector_token(node_rows.front()) +
        ";bandwidth=" + integer_vector_token(link_rows.front()) +
        ";py=" + std::to_string(random.python().getrandbits32()) +
        ";np=" + std::to_string(random.numpy().next_uint32());
}

std::string registry_names(const attribute::NodeAttributeRegistry& registry)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < registry.entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += registry.entries()[index].name;
    }
    result.push_back(']');
    return result;
}

std::string registry_names(const attribute::LinkAttributeRegistry& registry)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < registry.entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += registry.entries()[index].name;
    }
    result.push_back(']');
    return result;
}

std::string file_config(
    const std::filesystem::path& path,
    const bool with_num_nodes)
{
    return
        "topology:\n"
        "  file_path: " + quote_yaml(path) + "\n" +
        (with_num_nodes ? "  num_nodes: 3\n" : "") +
        "  type: path\n"
        "node_attrs_setting: []\n"
        "link_attrs_setting: []\n";
}

std::string origin_name(const network::PhysicalTopologyOrigin origin)
{
    switch (origin) {
    case network::PhysicalTopologyOrigin::generated:
        return "generated";
    case network::PhysicalTopologyOrigin::loaded_gml:
        return "loaded";
    case network::PhysicalTopologyOrigin::generated_after_gml_error:
        return "fallback";
    }
    throw std::runtime_error("invalid physical origin");
}

void generated_cases()
{
    const auto config = yaml(generated_config());
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        RandomContext random(999U);
        network::PhysicalNetworkBuildOptions options;
        options.seed = 77U;
        options.factory_workers = workers;
        options.attribute_workers = workers;
        auto value = network::PhysicalNetwork::from_setting(
            config, random, options);
        emit(
            "generated_w" + std::to_string(workers),
            generated_payload(value, random, workers));
    }

    RandomContext continuation(31U);
    static_cast<void>(continuation.python().random());
    static_cast<void>(continuation.numpy().random());
    network::PhysicalNetworkBuildOptions options;
    options.factory_workers = 2U;
    options.attribute_workers = 8U;
    auto value = network::PhysicalNetwork::from_setting(
        config, continuation, options);
    emit("continued", generated_payload(value, continuation, 8U));
}

void loaded_cases(
    const std::filesystem::path& loaded_path,
    const std::filesystem::path& empty_path,
    const std::filesystem::path& broken_path,
    const std::filesystem::path& missing_path,
    const std::filesystem::path& dataset_path)
{
    RandomContext loaded_random(5U);
    auto loaded = network::PhysicalNetwork::from_setting(
        yaml(file_config(loaded_path, false)), loaded_random, {});
    const auto loaded_node = loaded.bind_node_attribute("loaded_node");
    const auto loaded_link = loaded.bind_link_attribute("loaded_link");
    if (!loaded_node || !loaded_link) {
        throw std::runtime_error("loaded status binding failed");
    }
    const auto edges = loaded.graph().edges();
    if (edges.first == edges.second) {
        throw std::runtime_error("loaded fixture edge missing");
    }
    const AttrValue* topology = loaded.graph_attributes().find("topology");
    if (topology == nullptr || !std::holds_alternative<std::string>(*topology)) {
        throw std::runtime_error("loaded topology metadata missing");
    }
    emit(
        "loaded",
        "origin=" + origin_name(loaded.build_report().origin) +
        ";nodes=" + std::to_string(loaded.live_num_nodes()) +
        ";links=" + std::to_string(loaded.live_num_links()) +
        ";node_names=" + registry_names(loaded.node_attributes()) +
        ";link_names=" + registry_names(loaded.link_attributes()) +
        ";later=" +
            std::string(loaded.bind_node_attribute("later_only") ? "1" : "0") +
        ";node=" + std::to_string(std::get<std::int64_t>(
            loaded.graph().node_attrs(0U).at(loaded_node->value_id))) +
        ";link=" + double_token(std::get<double>(
            loaded.graph().edge_attrs(*edges.first).at(loaded_link->value_id))) +
        ";topology=" + std::get<std::string>(*topology) +
        ";snapshots=" +
            std::string(
                loaded.graph_attributes().find("node_attrs_setting") == nullptr &&
                loaded.graph_attributes().find("link_attrs_setting") == nullptr
                ? "0" : "1"));

    RandomContext empty_random(5U);
    auto empty = network::PhysicalNetwork::from_setting(
        yaml(file_config(empty_path, false)), empty_random, {});
    emit(
        "loaded_empty",
        "origin=" + origin_name(empty.build_report().origin) +
        ";nodes=" + std::to_string(empty.live_num_nodes()) +
        ";links=" + std::to_string(empty.live_num_links()) +
        ";node_names=" + registry_names(empty.node_attributes()) +
        ";link_names=" + registry_names(empty.link_attributes()));

    RandomContext broken_random(11U);
    auto fallback = network::PhysicalNetwork::from_setting(
        yaml(file_config(broken_path, true)), broken_random, {});
    emit(
        "broken_fallback",
        "origin=" + origin_name(fallback.build_report().origin) +
        ";error=" + (fallback.build_report().gml_error ? "1" : "0") +
        ";nodes=" + std::to_string(fallback.live_num_nodes()) +
        ";links=" + std::to_string(fallback.live_num_links()));

    RandomContext missing_random(11U);
    auto missing = network::PhysicalNetwork::from_setting(
        yaml(file_config(missing_path, true)), missing_random, {});
    emit(
        "missing_fallback",
        "origin=" + origin_name(missing.build_report().origin) +
        ";requested=" + (missing.build_report().requested_file ? "1" : "0") +
        ";error=" + (missing.build_report().gml_error ? "1" : "0") +
        ";nodes=" + std::to_string(missing.live_num_nodes()) +
        ";links=" + std::to_string(missing.live_num_links()));

    bool missing_num_error = false;
    try {
        RandomContext random(11U);
        static_cast<void>(network::PhysicalNetwork::from_setting(
            yaml(file_config(broken_path, false)), random, {}));
    } catch (const std::exception&) {
        missing_num_error = true;
    }
    emit("broken_missing_num", missing_num_error ? "error=1" : "error=0");

    auto dataset_loaded = network::PhysicalNetwork::load_dataset(
        dataset_path.string());
    emit(
        "dataset_load",
        "origin=" + origin_name(dataset_loaded.build_report().origin) +
        ";nodes=" + std::to_string(dataset_loaded.live_num_nodes()) +
        ";links=" + std::to_string(dataset_loaded.live_num_links()) +
        ";requested=" +
            std::string(dataset_loaded.build_report().requested_file ? "1" : "0"));

    auto clone = loaded.clone();
    network::PhysicalNetwork moved(std::move(clone));
    emit(
        "native_clone_move",
        "origin=" + origin_name(moved.build_report().origin) +
        ";nodes=" + std::to_string(moved.live_num_nodes()) +
        ";node_binding=" +
            std::string(moved.bind_node_attribute("loaded_node") ? "1" : "0"));
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 6) {
            throw std::invalid_argument(
                "usage: physical_network_harness <loaded.gml> <empty.gml> "
                "<broken.gml> <missing.gml> <dataset-dir>");
        }
        generated_cases();
        loaded_cases(argv[1], argv[2], argv[3], argv[4], argv[5]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysicalNetwork harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
