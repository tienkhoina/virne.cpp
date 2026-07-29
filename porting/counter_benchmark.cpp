#include "counter.h"

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
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace core = virne::core;
namespace network = virne::network;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    attribute::AttributeOwner owner) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    return result;
}

network::BaseNetwork make_fixture(std::size_t node_count) {
    std::vector<EdgeEndpoints> edges;
    edges.reserve(node_count - 1U);
    for (std::size_t node = 1U; node < node_count; ++node) {
        edges.emplace_back(node - 1U, node);
    }

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(node_count, std::move(edges));
    construction.config.node_attribute_specs = {
        resource_spec("node_integer", attribute::AttributeOwner::node),
        resource_spec("node_floating", attribute::AttributeOwner::node),
        resource_spec("node_boolean", attribute::AttributeOwner::node),
    };
    construction.config.link_attribute_specs = {
        resource_spec("link_integer", attribute::AttributeOwner::link),
        resource_spec("link_floating", attribute::AttributeOwner::link),
        resource_spec("link_boolean", attribute::AttributeOwner::link),
    };
    network::BaseNetwork result(std::move(construction));

    // Every dynamic name is resolved once during fixture setup. The timed
    // PreparedCounter call below sees only registry IDs and graph-local IDs.
    const auto node_integer = result.bind_node_attribute("node_integer");
    const auto node_floating = result.bind_node_attribute("node_floating");
    const auto node_boolean = result.bind_node_attribute("node_boolean");
    const auto link_integer = result.bind_link_attribute("link_integer");
    const auto link_floating = result.bind_link_attribute("link_floating");
    const auto link_boolean = result.bind_link_attribute("link_boolean");
    if (!node_integer || !node_floating || !node_boolean ||
        !link_integer || !link_floating || !link_boolean) {
        throw std::runtime_error("Counter benchmark binding failed");
    }

    std::vector<AttrValue> node_integer_values(node_count);
    std::vector<AttrValue> node_floating_values(node_count);
    std::vector<AttrValue> node_boolean_values(node_count);
    for (std::size_t index = 0U; index < node_count; ++index) {
        node_integer_values[index] =
            static_cast<std::int64_t>((index * 17U) % 1009U);
        node_floating_values[index] =
            static_cast<double>((index * 19U) % 1013U) + 0.25;
        node_boolean_values[index] = (index % 3U) == 0U;
    }

    const std::size_t link_count = node_count - 1U;
    std::vector<AttrValue> link_integer_values(link_count);
    std::vector<AttrValue> link_floating_values(link_count);
    std::vector<AttrValue> link_boolean_values(link_count);
    for (std::size_t index = 0U; index < link_count; ++index) {
        link_integer_values[index] =
            static_cast<std::int64_t>((index * 23U) % 1019U);
        link_floating_values[index] =
            static_cast<double>((index * 29U) % 1021U) + 0.125;
        link_boolean_values[index] = (index % 5U) == 0U;
    }

    result.set_node_attrs_data({
        {node_integer->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(node_integer_values)},
        {node_floating->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(node_floating_values)},
        {node_boolean->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(node_boolean_values)},
    });
    result.set_link_attrs_data({
        {link_integer->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(link_integer_values)},
        {link_floating->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(link_floating_values)},
        {link_boolean->registry_id,
         network::AttributeDataLayout::dense,
         {},
         std::move(link_boolean_values)},
    });
    return result;
}

std::size_t parse_size(const char* text) {
    const std::string value(text);
    std::size_t position = 0U;
    const unsigned long long parsed = std::stoull(value, &position, 10);
    if (position != value.size()) {
        throw std::invalid_argument("Counter benchmark argument is not unsigned");
    }
    return static_cast<std::size_t>(parsed);
}

std::string number_token(const core::CounterNumber& value) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return "i:" + std::to_string(*integer);
    }
    const double floating = std::get<double>(value);
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(floating));
    std::memcpy(&bits, &floating, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16)
           << bits;
    return stream.str();
}

std::uint64_t fingerprint(const std::string& value) noexcept {
    std::uint64_t result = fnv_offset;
    for (const char raw_byte : value) {
        result =
            (result ^ static_cast<unsigned char>(raw_byte)) * fnv_prime;
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: counter_benchmark <node_count> <workers>");
        }
        const std::size_t node_count = parse_size(argv[1]);
        const std::size_t workers = parse_size(argv[2]);
        if (node_count < 128U) {
            throw std::invalid_argument(
                "Counter benchmark node_count must be at least 128");
        }

        network::BaseNetwork network = make_fixture(node_count);
        const core::PreparedCounter prepared = core::Counter{}.prepare(network);

        const auto started = std::chrono::steady_clock::now();
        const core::CounterNumber sum = prepared.calculate_sum_network_resource(
            true,
            true,
            core::CounterOptions{workers});
        const auto stopped = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - started).count();

        if (!std::holds_alternative<double>(sum)) {
            throw std::runtime_error(
                "Mixed Counter benchmark did not produce the double lane");
        }
        const std::string payload = "sum=" + number_token(sum);
        const std::size_t entry_count =
            3U * node_count + 3U * (node_count - 1U);
        std::cout
            << "protocol=1\n"
            << "kind=counter_mixed_whole_network_sum\n"
            << "semantics=resource_major_numpy_flatten_v1\n"
            << "node_count=" << node_count << '\n'
            << "workers=" << workers << '\n'
            << "type_tag=counter_number_double_raw64\n"
            << "elapsed_ns=" << elapsed << '\n'
            << "checksum=" << fingerprint(payload) << '\n'
            << "output_bytes=" << payload.size() << '\n'
            << "entry_count=" << entry_count << '\n'
            << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "counter_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
