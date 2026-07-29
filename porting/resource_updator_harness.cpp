#include "resource_updator.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
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
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeNumber;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ResourceUpdateOperation;
using controller::ConstraintLink;
using controller::NodeResourceUpdateRequest;
using controller::PreparedResourceUpdator;
using controller::ResourceId;
using controller::ResourceUpdator;
using controller::ResourceUpdatorErrorCode;
using controller::ResourceUpdatorException;
using controller::ResourceUpdatorSelection;

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner,
    CheckingLevel level)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.checking_level = level;
    return result;
}

network::BaseNetworkConstruction construction(
    std::size_t nodes,
    std::vector<EdgeEndpoints> edges)
{
    network::BaseNetworkConstruction result;
    result.incoming_graph.emplace(nodes, std::move(edges));
    result.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node, CheckingLevel::node),
        resource_spec("memory", AttributeOwner::node, CheckingLevel::node),
        resource_spec("flag", AttributeOwner::node, CheckingLevel::node)};
    result.config.link_attribute_specs = {
        resource_spec("bandwidth", AttributeOwner::link, CheckingLevel::link),
        resource_spec("burst", AttributeOwner::link, CheckingLevel::link)};
    return result;
}

template <typename Network>
network::NodeNetworkAttributeBinding node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing node harness binding");
    }
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing link harness binding");
    }
    return *binding;
}

network::NodeAttributeDataUpdate node_update(
    ResourceId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate link_update(
    ResourceId id,
    std::vector<AttrValue> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

struct Fixture
{
    network::VirtualNetwork virtual_network{
        construction(2U, {{0U, 1U}})};
    network::PhysicalNetwork physical_network{
        construction(3U, {{0U, 1U}, {1U, 2U}})};

    network::NodeNetworkAttributeBinding v_cpu =
        node_binding(virtual_network, "cpu");
    network::NodeNetworkAttributeBinding v_memory =
        node_binding(virtual_network, "memory");
    network::NodeNetworkAttributeBinding v_flag =
        node_binding(virtual_network, "flag");
    network::LinkNetworkAttributeBinding v_bandwidth =
        link_binding(virtual_network, "bandwidth");
    network::LinkNetworkAttributeBinding v_burst =
        link_binding(virtual_network, "burst");

    network::NodeNetworkAttributeBinding p_cpu =
        node_binding(physical_network, "cpu");
    network::NodeNetworkAttributeBinding p_memory =
        node_binding(physical_network, "memory");
    network::NodeNetworkAttributeBinding p_flag =
        node_binding(physical_network, "flag");
    network::LinkNetworkAttributeBinding p_bandwidth =
        link_binding(physical_network, "bandwidth");
    network::LinkNetworkAttributeBinding p_burst =
        link_binding(physical_network, "burst");

    Fixture()
    {
        virtual_network.set_node_attrs_data({
            node_update(v_cpu.registry_id, {std::int64_t{2}, std::int64_t{3}}),
            node_update(v_memory.registry_id, {1.5, 2.5}),
            node_update(v_flag.registry_id, {true, false})});
        physical_network.set_node_attrs_data({
            node_update(
                p_cpu.registry_id,
                {std::int64_t{10}, std::int64_t{20}, std::int64_t{30}}),
            node_update(p_memory.registry_id, {10.0, 20.0, 30.0}),
            node_update(p_flag.registry_id, {true, false, true})});
        virtual_network.set_link_attrs_data({
            link_update(v_bandwidth.registry_id, {std::int64_t{3}}),
            link_update(v_burst.registry_id, {1.5})});
        physical_network.set_link_attrs_data({
            link_update(
                p_bandwidth.registry_id,
                {std::int64_t{10}, std::int64_t{12}}),
            link_update(p_burst.registry_id, {8.0, 9.0})});
    }

    PreparedResourceUpdator prepare(
        bool duplicate_bandwidth = false,
        bool empty = false)
    {
        ResourceUpdatorSelection selection;
        if (!empty)
        {
            selection.node_resources = {
                v_cpu.registry_id, v_memory.registry_id, v_flag.registry_id};
            selection.link_resources = {
                v_bandwidth.registry_id, v_burst.registry_id};
            if (duplicate_bandwidth)
            {
                selection.link_resources.push_back(v_bandwidth.registry_id);
            }
        }
        return ResourceUpdator(std::move(selection)).prepare(
            virtual_network, physical_network);
    }
};

std::string number_token(const AttrValue& value)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? "b:1" : "b:0";
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        std::uint64_t bits = 0U;
        std::memcpy(&bits, floating, sizeof(bits));
        std::ostringstream stream;
        stream << "d:" << std::hex << std::setfill('0')
               << std::setw(16) << bits;
        return stream.str();
    }
    throw std::runtime_error("nonnumeric harness snapshot");
}

const AttrValue& node_value(
    const Fixture& fixture,
    Vertex node,
    const network::NodeNetworkAttributeBinding& binding)
{
    return fixture.physical_network.graph().node_attrs(node).at(binding.value_id);
}

const AttrValue& link_value(
    const Fixture& fixture,
    ConstraintLink link,
    const network::LinkNetworkAttributeBinding& binding)
{
    const Graph& graph = fixture.physical_network.graph();
    return graph.edge_attrs(graph.edge(link.source, link.target)).at(
        binding.value_id);
}

std::string snapshot(const Fixture& fixture)
{
    std::string result = "nodes=[";
    for (Vertex node = 0U; node < 3U; ++node)
    {
        if (node != 0U)
        {
            result.push_back(',');
        }
        result += std::to_string(node) + "{" +
            number_token(node_value(fixture, node, fixture.p_cpu)) + "," +
            number_token(node_value(fixture, node, fixture.p_memory)) + "," +
            number_token(node_value(fixture, node, fixture.p_flag)) + "}";
    }
    result += "];links=[";
    const std::vector<ConstraintLink> links = {{0U, 1U}, {1U, 2U}};
    for (std::size_t index = 0U; index < links.size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const ConstraintLink link = links[index];
        result += std::to_string(link.source) + ":" +
            std::to_string(link.target) + "{" +
            number_token(link_value(fixture, link, fixture.p_bandwidth)) +
            "," + number_token(link_value(fixture, link, fixture.p_burst)) +
            "}";
    }
    result.push_back(']');
    return result;
}

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

void differential()
{
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        prepared.update_node_resource(
            0U,
            {fixture.v_cpu.registry_id, std::int64_t{3}},
            ResourceUpdateOperation::subtract,
            true);
        emit("node_sub", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        prepared.update_node_resource(
            0U,
            {fixture.v_memory.registry_id, 2.5},
            ResourceUpdateOperation::add,
            true);
        emit("node_add", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        prepared.update_node_resources(
            1U,
            {{fixture.v_cpu.registry_id, std::int64_t{2}},
             {fixture.v_memory.registry_id, 1.5}},
            ResourceUpdateOperation::subtract,
            true);
        emit("node_list", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        std::string error = "none";
        try
        {
            prepared.update_node_resources(
                1U,
                {{fixture.v_cpu.registry_id, std::int64_t{2}},
                 {fixture.v_memory.registry_id, 999.0}},
                ResourceUpdateOperation::subtract,
                true);
        }
        catch (const ResourceUpdatorException& caught)
        {
            error = caught.code() == ResourceUpdatorErrorCode::insufficient_resource
                ? "insufficient"
                : "unexpected";
        }
        emit("node_partial", "error=" + error + ";" + snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        prepared.update_link_resources(
            {1U, 0U},
            {{fixture.v_bandwidth.registry_id, std::int64_t{2}},
             {fixture.v_burst.registry_id, 0.5}},
            ResourceUpdateOperation::subtract,
            true);
        emit("link_sub", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        prepared.update_link_resource(
            {0U, 1U},
            {fixture.v_bandwidth.registry_id, std::int64_t{13}},
            ResourceUpdateOperation::subtract,
            false);
        emit("link_unsafe", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare(true);
        prepared.update_path_resources(
            {0U, 1U},
            {0U, 1U, 2U},
            ResourceUpdateOperation::subtract,
            true);
        emit("path", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        fixture.physical_network.graph()
            .edge_attrs(fixture.physical_network.graph().edge(1U, 2U))
            .set(fixture.p_bandwidth.value_id, std::int64_t{1});
        std::string error = "none";
        try
        {
            prepared.update_path_resources(
                {0U, 1U},
                {0U, 1U, 2U},
                ResourceUpdateOperation::subtract,
                true);
        }
        catch (const ResourceUpdatorException& caught)
        {
            error = caught.code() == ResourceUpdatorErrorCode::insufficient_resource
                ? "insufficient"
                : "unexpected";
        }
        emit("path_partial", "error=" + error + ";" + snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare(false, true);
        prepared.update_path_resources(
            {99U, 100U},
            {0U},
            ResourceUpdateOperation::subtract,
            true);
        emit("empty", snapshot(fixture));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        prepared.update_node_resource(
            0U,
            {fixture.v_flag.registry_id, true},
            ResourceUpdateOperation::add,
            false);
        emit("bool_add", snapshot(fixture));
    }
}

constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void mix_u64(std::uint64_t& checksum, std::uint64_t value) noexcept
{
    for (std::size_t byte = 0U; byte < 8U; ++byte)
    {
        checksum ^= value & 0xffU;
        checksum *= fnv_prime;
        value >>= 8U;
    }
}

std::size_t parse_size(const char* text)
{
    const std::string value(text);
    std::size_t position = 0U;
    const auto parsed = std::stoull(value, &position, 10);
    if (position != value.size())
    {
        throw std::invalid_argument("invalid unsigned argument");
    }
    return static_cast<std::size_t>(parsed);
}

void benchmark(std::size_t count, std::size_t workers)
{
    network::BaseNetworkConstruction virtual_construction;
    virtual_construction.incoming_graph.emplace(1U, std::vector<EdgeEndpoints>{});
    virtual_construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node, CheckingLevel::node)};
    network::VirtualNetwork virtual_network(std::move(virtual_construction));

    network::BaseNetworkConstruction physical_construction;
    physical_construction.incoming_graph.emplace(
        count, std::vector<EdgeEndpoints>{});
    physical_construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node, CheckingLevel::node)};
    network::PhysicalNetwork physical_network(std::move(physical_construction));

    const auto v_cpu = node_binding(virtual_network, "cpu");
    const auto p_cpu = node_binding(physical_network, "cpu");
    virtual_network.set_node_attrs_data({
        node_update(v_cpu.registry_id, {std::int64_t{1}})});
    std::vector<AttrValue> initial(count);
    std::vector<NodeResourceUpdateRequest> requests(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        initial[index] = static_cast<std::int64_t>(100U + index % 17U);
        requests[index] = {
            static_cast<Vertex>(index),
            {{v_cpu.registry_id,
              static_cast<std::int64_t>(1U + index % 7U)}}};
    }
    physical_network.set_node_attrs_data({
        node_update(p_cpu.registry_id, std::move(initial))});

    ResourceUpdatorSelection selection;
    selection.node_resources = {v_cpu.registry_id};
    auto prepared = ResourceUpdator(std::move(selection)).prepare(
        virtual_network, physical_network);

    const auto started = std::chrono::steady_clock::now();
    prepared.update_node_resources_batch(
        requests, ResourceUpdateOperation::subtract, true, workers);
    std::uint64_t checksum = fnv_offset;
    std::int64_t value_sum = 0;
    for (std::size_t index = 0U; index < count; ++index)
    {
        const auto& value = physical_network.graph().node_attrs(
            static_cast<Vertex>(index)).at(p_cpu.value_id);
        const std::int64_t integer = std::get<std::int64_t>(value);
        value_sum += integer;
        mix_u64(checksum, static_cast<std::uint64_t>(integer));
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();

    std::cout << "elapsed_ns=" << elapsed
              << ";entry_count=" << count
              << ";value_sum=" << value_sum
              << ";checksum=" << checksum << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 ||
            (argc == 2 && std::string_view(argv[1]) == "differential"))
        {
            differential();
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "benchmark")
        {
            benchmark(parse_size(argv[2]), parse_size(argv[3]));
            return 0;
        }
        throw std::invalid_argument(
            "usage: resource_updator_harness [differential | benchmark COUNT WORKERS]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "resource updator harness: " << error.what() << '\n';
        return 1;
    }
}
