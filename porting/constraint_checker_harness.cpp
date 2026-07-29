#include "constraint_checker.h"

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
using attribute::ConstraintRestriction;
using attribute::GraphResourceAttribute;
using attribute::GraphResourceSpec;
using controller::ConstraintCheckResult;
using controller::ConstraintChecker;
using controller::ConstraintCheckerSelection;
using controller::ConstraintId;
using controller::ConstraintLink;
using controller::GraphConstraintSelection;
using controller::NodeConstraintRequest;
using controller::PathConstraintCheckResult;
using controller::PreparedConstraintChecker;

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner,
    ConstraintRestriction restriction,
    CheckingLevel level)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = restriction;
    result.checking_level = level;
    return result;
}

AttributeFactorySpec latency_spec(
    std::string name,
    ConstraintRestriction restriction)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::latency;
    result.restriction = restriction;
    result.checking_level = CheckingLevel::path;
    return result;
}

network::NodeAttributeDataUpdate dense_node_update(
    ConstraintId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    ConstraintId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

network::BaseNetworkConstruction construction(
    std::size_t nodes,
    std::vector<EdgeEndpoints> edges)
{
    network::BaseNetworkConstruction result;
    result.incoming_graph.emplace(nodes, std::move(edges));
    result.config.node_attribute_specs = {
        resource_spec(
            "node_hard", AttributeOwner::node,
            ConstraintRestriction::hard, CheckingLevel::node),
        resource_spec(
            "node_soft", AttributeOwner::node,
            ConstraintRestriction::soft, CheckingLevel::node)};
    result.config.link_attribute_specs = {
        resource_spec(
            "link_hard", AttributeOwner::link,
            ConstraintRestriction::hard, CheckingLevel::link),
        resource_spec(
            "link_soft", AttributeOwner::link,
            ConstraintRestriction::soft, CheckingLevel::link),
        latency_spec("latency_hard", ConstraintRestriction::hard),
        latency_spec("latency_soft", ConstraintRestriction::soft)};
    return result;
}

template <typename Network>
ConstraintId node_id(Network& value, std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing node harness binding");
    }
    return binding->registry_id;
}

template <typename Network>
ConstraintId link_id(Network& value, std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing link harness binding");
    }
    return binding->registry_id;
}

template <typename Network>
void set_graph_value(
    Network& value,
    const GraphResourceAttribute& definition,
    AttrValue graph_value)
{
    const auto binding = definition.bind(value.graph());
    definition.set_data(value.graph(), graph_value, binding);
}

GraphResourceSpec graph_spec(
    std::string name,
    ConstraintRestriction restriction)
{
    GraphResourceSpec result;
    result.name = std::move(name);
    result.restriction = restriction;
    result.checking_level = CheckingLevel::graph;
    return result;
}

struct Fixture
{
    network::VirtualNetwork virtual_network{
        construction(3U, {{0U, 1U}, {1U, 2U}})};
    network::PhysicalNetwork physical_network{
        construction(
            4U,
            {{0U, 1U}, {1U, 2U}, {1U, 3U}, {0U, 3U}})};

    GraphResourceAttribute graph_hard{
        graph_spec("graph_hard", ConstraintRestriction::hard)};
    GraphResourceAttribute graph_soft{
        graph_spec("graph_soft", ConstraintRestriction::soft)};
    GraphResourceAttribute graph_boolean{
        graph_spec("graph_boolean", ConstraintRestriction::hard)};

    ConstraintId node_hard = 0U;
    ConstraintId node_soft = 0U;
    ConstraintId link_hard = 0U;
    ConstraintId link_soft = 0U;
    ConstraintId latency_hard = 0U;
    ConstraintId latency_soft = 0U;

    Fixture()
    {
        node_hard = node_id(virtual_network, "node_hard");
        node_soft = node_id(virtual_network, "node_soft");
        link_hard = link_id(virtual_network, "link_hard");
        link_soft = link_id(virtual_network, "link_soft");
        latency_hard = link_id(virtual_network, "latency_hard");
        latency_soft = link_id(virtual_network, "latency_soft");

        const ConstraintId physical_node_hard =
            node_id(physical_network, "node_hard");
        const ConstraintId physical_node_soft =
            node_id(physical_network, "node_soft");
        const ConstraintId physical_link_hard =
            link_id(physical_network, "link_hard");
        const ConstraintId physical_link_soft =
            link_id(physical_network, "link_soft");
        const ConstraintId physical_latency_hard =
            link_id(physical_network, "latency_hard");
        const ConstraintId physical_latency_soft =
            link_id(physical_network, "latency_soft");

        virtual_network.set_node_attrs_data({
            dense_node_update(
                node_hard,
                {std::int64_t{5}, std::int64_t{7}, std::int64_t{1}}),
            dense_node_update(node_soft, {9.5, 2.0, 1.0})});
        physical_network.set_node_attrs_data({
            dense_node_update(
                physical_node_hard,
                {std::int64_t{8}, std::int64_t{3},
                 std::int64_t{9}, std::int64_t{6}}),
            dense_node_update(
                physical_node_soft,
                {4.0, 10.0, 3.0, 1.5})});

        virtual_network.set_link_attrs_data({
            sparse_link_update(
                link_hard,
                {{0U, 1U, std::int64_t{4}},
                 {1U, 2U, std::int64_t{2}}}),
            sparse_link_update(
                link_soft,
                {{0U, 1U, 9.5}, {1U, 2U, 1.0}}),
            sparse_link_update(
                latency_hard,
                {{0U, 1U, 10.0}, {1U, 2U, 4.0}}),
            sparse_link_update(
                latency_soft,
                {{0U, 1U, 5.0}, {1U, 2U, 3.0}})});
        physical_network.set_link_attrs_data({
            sparse_link_update(
                physical_link_hard,
                {{0U, 1U, std::int64_t{6}},
                 {1U, 2U, std::int64_t{2}},
                 {1U, 3U, std::int64_t{7}},
                 {0U, 3U, std::int64_t{7}}}),
            sparse_link_update(
                physical_link_soft,
                {{0U, 1U, 3.0}, {1U, 2U, 10.0},
                 {1U, 3U, 4.0}, {0U, 3U, 4.0}}),
            sparse_link_update(
                physical_latency_hard,
                {{0U, 1U, 3.0}, {1U, 2U, 4.0},
                 {1U, 3U, 9.0}, {0U, 3U, 6.0}}),
            sparse_link_update(
                physical_latency_soft,
                {{0U, 1U, 4.0}, {1U, 2U, 5.0},
                 {1U, 3U, 8.0}, {0U, 3U, 7.0}})});

        set_graph_value(
            virtual_network, graph_hard, std::int64_t{7});
        set_graph_value(
            physical_network, graph_hard, std::int64_t{10});
        set_graph_value(virtual_network, graph_soft, 20.5);
        set_graph_value(physical_network, graph_soft, 5.0);
        set_graph_value(virtual_network, graph_boolean, false);
        set_graph_value(physical_network, graph_boolean, true);
    }

    ConstraintCheckerSelection selection() const
    {
        ConstraintCheckerSelection result;
        result.node_at_node = {node_hard, node_soft};
        result.link_at_link = {link_hard, link_soft};
        result.link_at_path = {latency_hard, latency_soft};
        result.graph = {
            GraphConstraintSelection{0U, &graph_hard},
            GraphConstraintSelection{1U, &graph_soft},
            GraphConstraintSelection{2U, &graph_boolean}};
        return result;
    }

    PreparedConstraintChecker prepare() const
    {
        return ConstraintChecker(selection()).prepare(
            virtual_network, physical_network);
    }
};

std::string number_token(const AttributeNumber& value)
{
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? "b:1" : "b:0";
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    std::uint64_t bits = 0U;
    const double floating = std::get<double>(value);
    std::memcpy(&bits, &floating, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

std::string offsets_payload(
    const virne::core::SolutionAttributeValues& offsets,
    const std::vector<ConstraintId>& ids)
{
    std::string result = "[";
    bool first = true;
    for (const ConstraintId id : ids)
    {
        const AttributeNumber* value = offsets.find(id);
        if (value == nullptr)
        {
            continue;
        }
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(id) + "=" + number_token(*value);
    }
    result.push_back(']');
    return result;
}

std::string check_payload(
    const ConstraintCheckResult& result,
    const std::vector<ConstraintId>& ids)
{
    return std::string("flag=") + (result.feasible ? "b:1" : "b:0") +
        ";offsets=" + offsets_payload(result.offsets, ids);
}

std::string path_payload(
    const PathConstraintCheckResult& result,
    const std::vector<ConstraintId>& link_ids,
    const std::vector<ConstraintId>& path_ids)
{
    std::string links = "[";
    bool first_attribute = true;
    for (const ConstraintId id : link_ids)
    {
        if (!first_attribute)
        {
            links.push_back(',');
        }
        first_attribute = false;
        links += std::to_string(id) + "={";
        bool first_link = true;
        for (const auto& item : result.link_level)
        {
            const AttributeNumber* value = item.offsets.find(id);
            if (value == nullptr)
            {
                continue;
            }
            if (!first_link)
            {
                links.push_back(',');
            }
            first_link = false;
            links += std::to_string(item.physical_link.source) + ":" +
                std::to_string(item.physical_link.target) + "=" +
                number_token(*value);
        }
        links.push_back('}');
    }
    links.push_back(']');
    return std::string("flag=") + (result.feasible ? "b:1" : "b:0") +
        ";links=" + links +
        ";path=" + offsets_payload(result.path_level, path_ids);
}

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

void differential()
{
    Fixture fixture;
    const PreparedConstraintChecker prepared = fixture.prepare();
    const std::vector<ConstraintId> graph_ids = {0U, 1U, 2U};
    const std::vector<ConstraintId> node_ids = {
        fixture.node_hard, fixture.node_soft};
    const std::vector<ConstraintId> link_ids = {
        fixture.link_hard, fixture.link_soft};
    const std::vector<ConstraintId> path_ids = {
        fixture.latency_hard, fixture.latency_soft};

    emit(
        "graph",
        check_payload(prepared.check_graph_constraints(), graph_ids));
    emit(
        "node_pass",
        check_payload(
            prepared.check_node_level_constraints(0U, 0U), node_ids));
    emit(
        "node_fail",
        check_payload(
            prepared.check_node_level_constraints(0U, 1U), node_ids));
    emit(
        "node_mixed",
        check_payload(
            prepared.check_node_level_constraints(1U, 2U), node_ids));
    emit(
        "link_pass",
        check_payload(
            prepared.check_link_level_constraints(
                {0U, 1U}, {0U, 1U}),
            link_ids));
    emit(
        "link_fail",
        check_payload(
            prepared.check_link_level_constraints(
                {0U, 1U}, {1U, 2U}),
            link_ids));
    emit(
        "link_reversed",
        check_payload(
            prepared.check_link_level_constraints(
                {1U, 0U}, {1U, 0U}),
            link_ids));
    emit(
        "path_direct",
        path_payload(
            prepared.check_path_level_constraints(
                {0U, 1U}, {0U, 3U}),
            link_ids,
            path_ids));
    emit(
        "path_link_fail",
        path_payload(
            prepared.check_path_level_constraints(
                {0U, 1U}, {0U, 1U, 2U}),
            link_ids,
            path_ids));
    emit(
        "path_latency_fail",
        path_payload(
            prepared.check_path_level_constraints(
                {0U, 1U}, {0U, 1U, 3U}),
            link_ids,
            path_ids));

    const PreparedConstraintChecker empty =
        ConstraintChecker(ConstraintCheckerSelection{}).prepare(
            fixture.virtual_network, fixture.physical_network);
    emit(
        "empty_graph",
        check_payload(empty.check_graph_constraints(), {}));
    emit(
        "empty_node",
        check_payload(
            empty.check_node_level_constraints(0U, 0U), {}));
    emit(
        "empty_link",
        check_payload(
            empty.check_link_level_constraints(
                {0U, 1U}, {0U, 1U}),
            {}));
    emit(
        "empty_path",
        path_payload(
            empty.check_path_level_constraints(
                {0U, 1U}, {0U, 3U}),
            {},
            {}));
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

std::uint64_t number_bits(const AttributeNumber& value)
{
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? 1U : 0U;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return static_cast<std::uint64_t>(*integer);
    }
    std::uint64_t bits = 0U;
    const double floating = std::get<double>(value);
    std::memcpy(&bits, &floating, sizeof(bits));
    return bits;
}

std::size_t parse_size(const char* text)
{
    std::size_t position = 0U;
    const std::string value(text);
    const auto parsed = std::stoull(value, &position, 10);
    if (position != value.size())
    {
        throw std::invalid_argument("invalid unsigned argument");
    }
    return static_cast<std::size_t>(parsed);
}

void benchmark(std::size_t count, std::size_t workers)
{
    Fixture fixture;
    ConstraintCheckerSelection selection;
    selection.node_at_node = {fixture.node_hard, fixture.node_soft};
    const PreparedConstraintChecker prepared =
        ConstraintChecker(std::move(selection)).prepare(
            fixture.virtual_network, fixture.physical_network);

    std::vector<NodeConstraintRequest> requests(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        requests[index] = {
            static_cast<Vertex>(index % 3U),
            static_cast<Vertex>((index * 3U + 1U) % 4U)};
    }

    const auto started = std::chrono::steady_clock::now();
    const auto results =
        prepared.check_node_level_constraints_batch(requests, workers);
    std::uint64_t checksum = fnv_offset;
    std::size_t feasible_count = 0U;
    for (const ConstraintCheckResult& result : results)
    {
        feasible_count += result.feasible ? 1U : 0U;
        mix_u64(checksum, result.feasible ? 1U : 0U);
        const AttributeNumber* hard = result.offsets.find(fixture.node_hard);
        const AttributeNumber* soft = result.offsets.find(fixture.node_soft);
        if (hard == nullptr || soft == nullptr)
        {
            throw std::runtime_error("missing benchmark offset");
        }
        mix_u64(checksum, number_bits(*hard));
        mix_u64(checksum, number_bits(*soft));
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();

    std::cout << "elapsed_ns=" << elapsed
              << ";entry_count=" << results.size()
              << ";feasible_count=" << feasible_count
              << ";checksum=" << checksum << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "differential"))
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
            "usage: constraint_checker_harness [differential | benchmark COUNT WORKERS]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "constraint checker harness: " << error.what() << '\n';
        return 1;
    }
}
