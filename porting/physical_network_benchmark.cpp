#include "physical_network.h"

#include "random_context.h"
#include "setting.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace network = virne::network;
namespace utils = virne::utils;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void add_u64(std::uint64_t& checksum, const std::uint64_t value)
{
    for (std::size_t shift = 0U; shift < 64U; shift += 8U) {
        checksum =
            (checksum ^ static_cast<std::uint8_t>(value >> shift)) * fnv_prime;
    }
}

void add_u32(std::uint64_t& checksum, const std::uint32_t value)
{
    for (std::size_t shift = 0U; shift < 32U; shift += 8U) {
        checksum =
            (checksum ^ static_cast<std::uint8_t>(value >> shift)) * fnv_prime;
    }
}

std::string config_bytes(const std::size_t count)
{
    return
        "topology:\n"
        "  num_nodes: " + std::to_string(count) + "\n"
        "  type: path\n"
        "node_attrs_setting:\n"
        "  - name: cpu\n"
        "    owner: node\n"
        "    type: resource\n"
        "    generative: true\n"
        "    distribution: uniform\n"
        "    dtype: int\n"
        "    low: 1\n"
        "    high: 1009\n"
        "link_attrs_setting:\n"
        "  - name: bandwidth\n"
        "    owner: link\n"
        "    type: resource\n"
        "    generative: true\n"
        "    distribution: uniform\n"
        "    dtype: int\n"
        "    low: 10\n"
        "    high: 1021\n";
}

void add_integer_row(
    std::uint64_t& checksum,
    const std::vector<AttrValue>& row)
{
    for (const AttrValue& value : row) {
        add_u64(
            checksum,
            static_cast<std::uint64_t>(std::get<std::int64_t>(value)));
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: physical_network_benchmark <count> <workers>");
        }
        const auto count = static_cast<std::size_t>(std::stoull(argv[1]));
        const auto workers = static_cast<std::size_t>(std::stoull(argv[2]));
        if (count < 4U) {
            throw std::invalid_argument("physical benchmark count must be >= 4");
        }

        const utils::SettingDocument config = utils::parse_setting(
            config_bytes(count), utils::SettingFormat::yaml);
        RandomContext random(999U);
        network::PhysicalNetworkBuildOptions options;
        options.seed = 77U;
        options.factory_workers = workers;
        options.attribute_workers = workers;

        const auto begin = std::chrono::steady_clock::now();
        auto value = network::PhysicalNetwork::from_setting(
            config, random, options);
        const auto end = std::chrono::steady_clock::now();

        const auto cpu = value.bind_node_attribute("cpu");
        const auto bandwidth = value.bind_link_attribute("bandwidth");
        if (!cpu || !bandwidth) {
            throw std::runtime_error("physical benchmark binding failed");
        }
        const auto node_rows = network::get_node_attrs_data(
            value, {cpu->registry_id}, workers);
        const auto link_rows = network::get_link_attrs_data(
            value, {bandwidth->registry_id}, workers);
        if (node_rows.size() != 1U || link_rows.size() != 1U) {
            throw std::runtime_error("physical benchmark row-count drift");
        }

        std::uint64_t checksum = fnv_offset;
        add_u64(checksum, value.live_num_nodes());
        add_u64(checksum, value.live_num_links());
        add_integer_row(checksum, node_rows.front());
        add_integer_row(checksum, link_rows.front());
        add_u32(checksum, random.python().getrandbits32());
        add_u32(checksum, random.numpy().next_uint32());

        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin).count();
        const std::size_t entries = count + (count - 1U);
        const std::size_t output_bytes =
            16U + entries * sizeof(std::int64_t) + 8U;
        std::cout
            << "protocol=1\n"
            << "kind=physical_from_setting\n"
            << "count=" << count << '\n'
            << "workers=" << workers << '\n'
            << "elapsed_ns=" << elapsed << '\n'
            << "checksum=" << checksum << '\n'
            << "output_bytes=" << output_bytes << '\n'
            << "entry_count=" << entries << '\n'
            << "type_tag=le64_order_v1\n"
            << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysicalNetwork benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
