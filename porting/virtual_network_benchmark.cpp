#include "virtual_network.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

attribute::AttributeFactorySpec make_spec(
    std::string name,
    const attribute::AttributeOwner owner) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    return result;
}

network::VirtualNetwork make_fixture(const std::size_t count) {
    Graph graph;
    for (std::size_t node = 1U; node < count; ++node) {
        graph.add_edge(node - 1U, node);
    }
    network::BaseNetworkConstruction construction;
    construction.incoming_graph = std::move(graph);
    construction.config.node_attribute_specs = {
        make_spec("cpu", attribute::AttributeOwner::node),
        make_spec("peak", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        make_spec("bw", attribute::AttributeOwner::link),
        make_spec("peak_bw", attribute::AttributeOwner::link),
    };
    network::VirtualNetwork result(std::move(construction));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto peak = result.bind_node_attribute("peak");
    const auto bw = result.bind_link_attribute("bw");
    const auto peak_bw = result.bind_link_attribute("peak_bw");
    if (!cpu || !peak || !bw || !peak_bw) {
        throw std::runtime_error("benchmark binding failed");
    }
    std::vector<AttrValue> cpu_values(count);
    std::vector<AttrValue> peak_values(count);
    for (std::size_t index = 0U; index < count; ++index) {
        cpu_values[index] = static_cast<std::int64_t>((index * 17U) % 1009U);
        peak_values[index] = static_cast<double>((index * 19U) % 1013U) + 0.25;
    }
    const std::size_t edges = count - 1U;
    std::vector<AttrValue> bw_values(edges);
    std::vector<AttrValue> peak_bw_values(edges);
    for (std::size_t index = 0U; index < edges; ++index) {
        bw_values[index] = static_cast<std::int64_t>((index * 23U) % 1019U);
        peak_bw_values[index] =
            static_cast<double>((index * 29U) % 1021U) + 0.125;
    }
    result.set_node_attrs_data({
        {cpu->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(cpu_values)},
        {peak->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(peak_values)},
    });
    result.set_link_attrs_data({
        {bw->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(bw_values)},
        {peak_bw->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(peak_bw_values)},
    });
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

std::uint64_t fingerprint(const std::string& value) {
    std::uint64_t checksum = fnv_offset;
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        checksum = (checksum ^ byte) * fnv_prime;
    }
    return checksum;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: virtual_network_benchmark <count> <workers>");
        }
        const auto count = static_cast<std::size_t>(std::stoull(argv[1]));
        const auto workers = static_cast<std::size_t>(std::stoull(argv[2]));
        if (count < 4U) {
            throw std::invalid_argument("count must be at least four");
        }
        auto value = make_fixture(count);
        const auto begin = std::chrono::steady_clock::now();
        const double total = value.total_resource_demand(workers);
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count());
        const double node = value.total_node_resource_demand(workers);
        const double link = value.total_link_resource_demand(workers);
        const std::string output =
            "node=" + double_token(node) + ";link=" + double_token(link) +
            ";total=" + double_token(total);
        std::cout << "protocol=1\n"
                  << "kind=virtual_network_demand\n"
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "type_tag=raw64_order_v1\n"
                  << "elapsed_ns=" << elapsed << '\n'
                  << "checksum=" << fingerprint(output) << '\n'
                  << "output_bytes=" << output.size() << '\n'
                  << "entry_count=" << (2U * count + 2U * (count - 1U)) << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "virtual_network_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
