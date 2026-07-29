#include "controller.h"

#include "attribute/attribute_factory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace network = virne::network;

constexpr std::size_t item_count = 8192U;
constexpr std::size_t warmup_count = 1U;
constexpr std::size_t sample_count = 3U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Fixture
{
    network::VirtualNetwork virtual_network;
    network::PhysicalNetwork physical_network;
    core::Solution solution;
    controller::Controller lifecycle;
    controller::PreparedController prepared;
    AttrId physical_node_value = 0U;
    AttrId physical_link_value = 0U;

    Fixture(
        network::VirtualNetwork virtual_value,
        network::PhysicalNetwork physical_value,
        core::Solution solution_value,
        controller::ControllerSelection selection,
        AttrId physical_node_value_id,
        AttrId physical_link_value_id)
        : virtual_network(std::move(virtual_value)),
          physical_network(std::move(physical_value)),
          solution(std::move(solution_value)),
          lifecycle(std::move(selection)),
          prepared(lifecycle.prepare(virtual_network, physical_network)),
          physical_node_value(physical_node_value_id),
          physical_link_value(physical_link_value_id)
    {
    }
};

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    const attribute::AttributeOwner owner)
{
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = owner == attribute::AttributeOwner::node
        ? attribute::CheckingLevel::node
        : attribute::CheckingLevel::link;
    return result;
}

network::BaseNetworkConstruction construction(
    Graph graph,
    const std::string& node_name,
    const std::string& link_name)
{
    network::BaseNetworkConstruction result;
    result.incoming_graph = std::move(graph);
    result.config.node_attribute_specs.push_back(
        resource_spec(node_name, attribute::AttributeOwner::node));
    result.config.link_attribute_specs.push_back(
        resource_spec(link_name, attribute::AttributeOwner::link));
    return result;
}

double node_capacity(const std::size_t index)
{
    return 1000.0 + static_cast<double>(index % 97U) * 0.25;
}

double link_capacity(const std::size_t index)
{
    return 2000.0 + static_cast<double>(index % 89U) * 0.5;
}

double node_demand(const std::size_t index)
{
    return 0.5 + static_cast<double>(index % 13U) * 0.125;
}

double link_demand(const std::size_t index)
{
    return 0.25 + static_cast<double>(index % 11U) * 0.0625;
}

Fixture make_fixture()
{
    constexpr std::string_view node_name = "cpu";
    constexpr std::string_view link_name = "bandwidth";

    std::vector<EdgeEndpoints> edges;
    edges.reserve(item_count);
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        const Vertex source = static_cast<Vertex>(index);
        const Vertex target = static_cast<Vertex>((index + 1U) % item_count);
        edges.emplace_back(source, target);
    }

    Graph virtual_graph(item_count, edges);
    Graph physical_graph(item_count, edges);
    network::VirtualNetwork virtual_network(construction(
        std::move(virtual_graph),
        std::string(node_name),
        std::string(link_name)));
    network::PhysicalNetwork physical_network(construction(
        std::move(physical_graph),
        std::string(node_name),
        std::string(link_name)));

    const auto virtual_node_binding =
        virtual_network.bind_node_attribute(node_name);
    const auto virtual_link_binding =
        virtual_network.bind_link_attribute(link_name);
    const auto physical_node_binding =
        physical_network.bind_node_attribute(node_name);
    const auto physical_link_binding =
        physical_network.bind_link_attribute(link_name);
    if (!virtual_node_binding.has_value() ||
        !virtual_link_binding.has_value() ||
        !physical_node_binding.has_value() ||
        !physical_link_binding.has_value())
    {
        throw std::runtime_error("controller benchmark binding failure");
    }

    std::vector<AttrValue> virtual_node_values(item_count);
    std::vector<AttrValue> physical_node_values(item_count);
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        virtual_node_values[index] = node_demand(index);
        physical_node_values[index] = node_capacity(index);
    }
    std::vector<AttrValue> virtual_link_values;
    std::vector<AttrValue> physical_link_values;
    virtual_link_values.reserve(item_count);
    physical_link_values.reserve(item_count);
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        virtual_link_values.emplace_back(link_demand(index));
        physical_link_values.emplace_back(link_capacity(index));
    }
    virtual_network.set_node_attrs_data({network::NodeAttributeDataUpdate{
        virtual_node_binding->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(virtual_node_values)}});
    virtual_network.set_link_attrs_data({network::LinkAttributeDataUpdate{
        virtual_link_binding->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(virtual_link_values)}});
    physical_network.set_node_attrs_data({network::NodeAttributeDataUpdate{
        physical_node_binding->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(physical_node_values)}});
    physical_network.set_link_attrs_data({network::LinkAttributeDataUpdate{
        physical_link_binding->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(physical_link_values)}});

    core::Solution solution(core::SolutionMetadata{
        7, 1.0, 0.0, item_count, item_count});
    solution.result = true;
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        const auto node = static_cast<core::SolutionNodeId>(index);
        core::SolutionAttributeValues node_values;
        node_values.set(virtual_node_binding->registry_id, node_demand(index));
        solution.node_slots.insert_or_assign(node, node);
        solution.node_slots_info.insert_or_assign(
            core::NodeSlotInfoKey{node, node},
            std::move(node_values));

        const core::SolutionLink virtual_link{
            static_cast<core::SolutionNodeId>(index),
            static_cast<core::SolutionNodeId>((index + 1U) % item_count)};
        const core::SolutionLink physical_link = virtual_link;
        core::SolutionAttributeValues link_values;
        link_values.set(virtual_link_binding->registry_id, link_demand(index));
        solution.link_paths.insert_or_assign(virtual_link, {physical_link});
        solution.link_paths_info.insert_or_assign(
            core::LinkPathInfoKey{virtual_link, physical_link},
            std::move(link_values));
    }

    controller::ControllerSelection selection;
    selection.node_resources = {virtual_node_binding->registry_id};
    selection.link_resources = {virtual_link_binding->registry_id};
    return Fixture(
        std::move(virtual_network),
        std::move(physical_network),
        std::move(solution),
        std::move(selection),
        physical_node_binding->value_id,
        physical_link_binding->value_id);
}

void restore_baseline(Fixture& fixture)
{
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        fixture.physical_network.graph().node_attrs(
            static_cast<Vertex>(index)).set(
                fixture.physical_node_value,
                node_capacity(index));
    }
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        const Edge edge = fixture.physical_network.graph().edge(
            static_cast<Vertex>(index),
            static_cast<Vertex>((index + 1U) % item_count));
        fixture.physical_network.graph().edge_attrs(edge).set(
                fixture.physical_link_value,
                link_capacity(index));
    }
}

double numeric_value(const AttrValue& value)
{
    if (const auto* number = std::get_if<double>(&value))
    {
        return *number;
    }
    if (const auto* number = std::get_if<std::int64_t>(&value))
    {
        return static_cast<double>(*number);
    }
    if (const auto* number = std::get_if<bool>(&value))
    {
        return *number ? 1.0 : 0.0;
    }
    throw std::runtime_error("controller benchmark nonnumeric output");
}

void hash_u64(std::uint64_t& hash, const std::uint64_t value)
{
    for (unsigned int shift = 0U; shift < 64U; shift += 8U)
    {
        hash ^= (value >> shift) & 0xffU;
        hash *= fnv_prime;
    }
}

std::uint64_t hash_double(std::uint64_t hash, const double value)
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "double width drift");
    std::memcpy(&bits, &value, sizeof(bits));
    hash_u64(hash, bits);
    return hash;
}

std::uint64_t output_checksum(const Fixture& fixture)
{
    std::uint64_t hash = fnv_offset;
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        const auto& value = fixture.physical_network.graph().node_attrs(
            static_cast<Vertex>(index)).at(fixture.physical_node_value);
        hash = hash_double(hash, numeric_value(value));
    }
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        const Edge edge = fixture.physical_network.graph().edge(
            static_cast<Vertex>(index),
            static_cast<Vertex>((index + 1U) % item_count));
        const auto& value = fixture.physical_network.graph().edge_attrs(edge).at(
            fixture.physical_link_value);
        hash = hash_double(hash, numeric_value(value));
    }
    return hash;
}

std::uint64_t expected_checksum()
{
    std::uint64_t hash = fnv_offset;
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        hash = hash_double(hash, node_capacity(index));
    }
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        hash = hash_double(hash, link_capacity(index));
    }
    return hash;
}

std::uint64_t expected_deployed_checksum()
{
    std::uint64_t hash = fnv_offset;
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        hash = hash_double(
            hash, node_capacity(index) - node_demand(index));
    }
    for (std::size_t index = 0U; index < item_count; ++index)
    {
        hash = hash_double(
            hash, link_capacity(index) - link_demand(index));
    }
    return hash;
}

void run_once(Fixture& fixture, const std::size_t workers)
{
    if (!fixture.prepared.deploy(
            fixture.solution, controller::ControllerMutationOptions{workers}) ||
        !fixture.prepared.release(
            fixture.solution, controller::ControllerMutationOptions{workers}))
    {
        throw std::runtime_error("controller benchmark lifecycle returned false");
    }
}

double median(std::array<double, sample_count> values)
{
    std::sort(values.begin(), values.end());
    return values[sample_count / 2U];
}

void benchmark()
{
    Fixture fixture = make_fixture();
    const std::uint64_t expected = expected_checksum();
    const std::uint64_t expected_deployed = expected_deployed_checksum();

    std::cout << std::setprecision(17);
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        restore_baseline(fixture);
        if (!fixture.prepared.deploy(
                fixture.solution,
                controller::ControllerMutationOptions{workers}) ||
            output_checksum(fixture) != expected_deployed ||
            !fixture.prepared.release(
                fixture.solution,
                controller::ControllerMutationOptions{workers}) ||
            output_checksum(fixture) != expected)
        {
            throw std::runtime_error(
                "controller benchmark lifecycle output gate mismatch");
        }
        for (std::size_t warmup = 0U; warmup < warmup_count; ++warmup)
        {
            restore_baseline(fixture);
            run_once(fixture, workers);
            if (output_checksum(fixture) != expected)
            {
                throw std::runtime_error("controller benchmark warmup checksum mismatch");
            }
        }

        std::array<double, sample_count> samples{};
        for (std::size_t sample = 0U; sample < sample_count; ++sample)
        {
            restore_baseline(fixture);
            const auto begin = std::chrono::steady_clock::now();
            run_once(fixture, workers);
            const auto end = std::chrono::steady_clock::now();
            if (output_checksum(fixture) != expected)
            {
                throw std::runtime_error("controller benchmark checksum mismatch");
            }
            samples[sample] = std::chrono::duration<double, std::milli>(
                end - begin).count();
        }
        std::cout
            << "workers=" << workers
            << ";items=" << item_count
            << ";operations=" << item_count * 4U
            << ";deployed_checksum=" << expected_deployed
            << ";checksum=" << expected
            << ";output_bytes="
            << (item_count * 2U * sizeof(double))
            << ";median_ms=" << median(samples)
            << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 ||
            (argc == 2 && std::string_view(argv[1]) == "benchmark"))
        {
            benchmark();
            return 0;
        }
        throw std::invalid_argument(
            "usage: controller_benchmark [benchmark]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "controller benchmark: " << error.what() << '\n';
        return 1;
    }
}
