#include "base_network.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace network = virne::network;
namespace attribute = virne::network::attribute;

struct Result {
    std::uint64_t elapsed_ns = 0U;
    std::uint64_t checksum = 0U;
    std::size_t output_bytes = 0U;
    std::size_t entry_count = 0U;
};

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

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

std::string value_token(const AttrValue& value) {
    return std::visit(
        [](const auto& item) -> std::string {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, std::int64_t>) {
                return "i:" + std::to_string(item);
            } else if constexpr (std::is_same_v<Item, double>) {
                return double_token(item);
            } else if constexpr (std::is_same_v<Item, bool>) {
                return item ? "b:1" : "b:0";
            } else if constexpr (std::is_same_v<Item, std::string>) {
                return "s:" + hex_text(item);
            } else {
                throw std::runtime_error("recursive benchmark value");
            }
        },
        value);
}

std::string canonical_rows(
    const std::vector<std::vector<AttrValue>>& rows) {
    std::string result = "[";
    for (std::size_t row = 0U; row < rows.size(); ++row) {
        if (row != 0U) {
            result.push_back(',');
        }
        result.push_back('[');
        for (std::size_t column = 0U; column < rows[row].size(); ++column) {
            if (column != 0U) {
                result.push_back(',');
            }
            result += value_token(rows[row][column]);
        }
        result.push_back(']');
    }
    result.push_back(']');
    return result;
}

std::string canonical_map(
    const std::optional<attribute::AttributeBenchmarkMap>& value) {
    if (!value) {
        return "none";
    }
    std::string result = "{";
    for (std::size_t index = 0U; index < value->entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& entry = value->entries()[index];
        result += hex_text(entry.name);
        result.push_back('=');
        result += double_token(entry.value);
    }
    result.push_back('}');
    return result;
}

std::string canonical_benchmarks(
    const attribute::AttributeBenchmarks& value) {
    return "node=" + canonical_map(value.node_attr_benchmarks)
        + ";link=" + canonical_map(value.link_attr_benchmarks)
        + ";link_sum=" + canonical_map(value.link_sum_attr_benchmarks);
}

std::uint64_t fingerprint(const std::string_view value) {
    std::uint64_t result = 14695981039346656037ULL;
    for (const char raw_byte : value) {
        result ^= static_cast<unsigned char>(raw_byte);
        result *= 1099511628211ULL;
    }
    return result;
}

attribute::AttributeFactorySpec make_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const attribute::AttributeKind kind) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = kind;
    return result;
}

attribute::AttributeRegistryId require_node_id(
    const network::BaseNetwork& value,
    const std::string_view name) {
    const auto id = value.node_attributes().bind(name);
    if (!id) {
        throw std::runtime_error("missing node definition");
    }
    return *id;
}

attribute::AttributeRegistryId require_link_id(
    const network::BaseNetwork& value,
    const std::string_view name) {
    const auto id = value.link_attributes().bind(name);
    if (!id) {
        throw std::runtime_error("missing link definition");
    }
    return *id;
}

network::BaseNetwork make_fixture(const std::size_t count) {
    Graph graph;
    for (std::size_t index = 0U; index < count; ++index) {
        graph.add_node();
    }
    for (Vertex index = 0U; index + 1U < count; ++index) {
        graph.add_edge(index, index + 1U);
    }
    for (Vertex index = 0U; index + 2U < count; ++index) {
        graph.add_edge(index, index + 2U);
    }

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(std::move(graph));
    construction.config.node_attribute_specs.push_back(make_spec(
        "cpu", attribute::AttributeOwner::node,
        attribute::AttributeKind::resource));
    auto peak = make_spec(
        "peak", attribute::AttributeOwner::node,
        attribute::AttributeKind::extrema);
    peak.originator_name = "cpu";
    construction.config.node_attribute_specs.push_back(std::move(peak));
    construction.config.node_attribute_specs.push_back(make_spec(
        "state", attribute::AttributeOwner::node,
        attribute::AttributeKind::status));
    construction.config.link_attribute_specs.push_back(make_spec(
        "bw", attribute::AttributeOwner::link,
        attribute::AttributeKind::resource));
    auto peak_bw = make_spec(
        "peak_bw", attribute::AttributeOwner::link,
        attribute::AttributeKind::extrema);
    peak_bw.originator_name = "bw";
    construction.config.link_attribute_specs.push_back(std::move(peak_bw));
    construction.config.link_attribute_specs.push_back(make_spec(
        "delay", attribute::AttributeOwner::link,
        attribute::AttributeKind::latency));

    network::BaseNetwork result(std::move(construction));
    const std::size_t edges = result.graph().num_edges();
    std::vector<AttrValue> cpu(count);
    std::vector<AttrValue> peak_values(count);
    std::vector<AttrValue> state(count);
    for (std::size_t index = 0U; index < count; ++index) {
        const auto raw = static_cast<std::int64_t>((index * 17U) % 1000U);
        cpu[index] = raw;
        peak_values[index] = static_cast<double>(raw) + 0.25;
        state[index] = (index % 2U) == 0U;
    }
    std::vector<AttrValue> bw(edges);
    std::vector<AttrValue> link_peak(edges);
    std::vector<AttrValue> delay(edges);
    for (std::size_t index = 0U; index < edges; ++index) {
        const auto raw = static_cast<std::int64_t>((index * 13U) % 1000U + 1U);
        bw[index] = raw;
        link_peak[index] = static_cast<double>(raw) + 0.5;
        delay[index] = 0.25 * static_cast<double>(index) + 0.5;
    }
    result.set_node_attrs_data({
        {require_node_id(result, "cpu"), network::AttributeDataLayout::dense,
         {}, std::move(cpu)},
        {require_node_id(result, "peak"), network::AttributeDataLayout::dense,
         {}, std::move(peak_values)},
        {require_node_id(result, "state"), network::AttributeDataLayout::dense,
         {}, std::move(state)},
    });
    result.set_link_attrs_data({
        {require_link_id(result, "bw"), network::AttributeDataLayout::dense,
         {}, std::move(bw)},
        {require_link_id(result, "peak_bw"), network::AttributeDataLayout::dense,
         {}, std::move(link_peak)},
        {require_link_id(result, "delay"), network::AttributeDataLayout::dense,
         {}, std::move(delay)},
    });
    return result;
}

Result finish_result(
    const std::uint64_t elapsed_ns,
    std::string output,
    const std::size_t entries) {
    const std::uint64_t checksum = fingerprint(output);
    return {elapsed_ns, checksum, output.size(), entries};
}

Result run_get(
    const std::size_t count,
    const std::size_t workers) {
    const auto value = make_fixture(count);
    const std::vector<attribute::AttributeRegistryId> node_ids = {
        require_node_id(value, "cpu"), require_node_id(value, "peak")};
    const std::vector<attribute::AttributeRegistryId> link_ids = {
        require_link_id(value, "bw"), require_link_id(value, "peak_bw")};
    const auto begin = std::chrono::steady_clock::now();
    auto node_rows = network::get_node_attrs_data(value, node_ids, workers);
    auto link_rows = network::get_link_attrs_data(value, link_ids, workers);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    const std::size_t entries = 2U * count + 2U * value.graph().num_edges();
    return finish_result(
        elapsed,
        "node=" + canonical_rows(node_rows) + ";link=" + canonical_rows(link_rows),
        entries);
}

Result run_set(
    const std::size_t count,
    const std::size_t workers) {
    auto value = make_fixture(count);
    const auto cpu = require_node_id(value, "cpu");
    const auto peak = require_node_id(value, "peak");
    const auto bw = require_link_id(value, "bw");
    const auto peak_bw = require_link_id(value, "peak_bw");
    const std::size_t edges = value.graph().num_edges();
    std::vector<AttrValue> cpu_values(count);
    std::vector<AttrValue> peak_values(count);
    for (std::size_t index = 0U; index < count; ++index) {
        cpu_values[index] = static_cast<std::int64_t>((index * 3U) % 997U);
        peak_values[index] = static_cast<double>((index * 5U) % 991U) + 0.75;
    }
    std::vector<AttrValue> bw_values(edges);
    std::vector<AttrValue> link_peak_values(edges);
    for (std::size_t index = 0U; index < edges; ++index) {
        bw_values[index] = static_cast<std::int64_t>((index * 7U) % 983U);
        link_peak_values[index] =
            static_cast<double>((index * 11U) % 977U) + 0.125;
    }
    std::vector<network::NodeAttributeDataUpdate> node_updates;
    node_updates.push_back(
        {cpu, network::AttributeDataLayout::dense, {}, std::move(cpu_values)});
    node_updates.push_back(
        {peak, network::AttributeDataLayout::dense, {}, std::move(peak_values)});
    std::vector<network::LinkAttributeDataUpdate> link_updates;
    link_updates.push_back(
        {bw, network::AttributeDataLayout::dense, {}, std::move(bw_values)});
    link_updates.push_back({
        peak_bw, network::AttributeDataLayout::dense, {},
        std::move(link_peak_values)});
    const auto begin = std::chrono::steady_clock::now();
    value.set_node_attrs_data(node_updates, workers);
    value.set_link_attrs_data(link_updates, workers);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    const auto node_rows = network::get_node_attrs_data(value, {cpu, peak});
    const auto link_rows = network::get_link_attrs_data(value, {bw, peak_bw});
    const std::size_t entries = 2U * count + 2U * edges;
    return finish_result(
        elapsed,
        "node=" + canonical_rows(node_rows) + ";link=" + canonical_rows(link_rows),
        entries);
}

Result run_manager(
    const std::size_t count,
    const std::size_t workers) {
    const auto value = make_fixture(count);
    network::BaseNetworkBenchmarkSelection selection;
    selection.workers = workers;
    const auto begin = std::chrono::steady_clock::now();
    const auto benchmarks = network::get_attribute_benchmarks(value, selection);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    std::size_t entries = 0U;
    if (benchmarks.node_attr_benchmarks) {
        entries += benchmarks.node_attr_benchmarks->entries().size();
    }
    if (benchmarks.link_attr_benchmarks) {
        entries += benchmarks.link_attr_benchmarks->entries().size();
    }
    if (benchmarks.link_sum_attr_benchmarks) {
        entries += benchmarks.link_sum_attr_benchmarks->entries().size();
    }
    return finish_result(elapsed, canonical_benchmarks(benchmarks), entries);
}

Result run(
    const std::string_view operation,
    const std::size_t count,
    const std::size_t workers) {
    if (operation == "get") {
        return run_get(count, workers);
    }
    if (operation == "set") {
        return run_set(count, workers);
    }
    if (operation == "manager") {
        return run_manager(count, workers);
    }
    throw std::invalid_argument("unknown BaseNetwork benchmark operation");
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: base_network_benchmark <get|set|manager> <count> <workers>");
        }
        const std::string operation(argv[1]);
        const auto count = static_cast<std::size_t>(std::stoull(argv[2]));
        const auto workers = static_cast<std::size_t>(std::stoull(argv[3]));
        if (count < 4U) {
            throw std::invalid_argument("count must be at least four");
        }
        const Result result = run(operation, count, workers);
        std::cout << "protocol=1\n"
                  << "kind=base_network\n"
                  << "operation=" << operation << '\n'
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "type_tag=attrvalue_bits_order_v1\n"
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "entry_count=" << result.entry_count << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base_network_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
