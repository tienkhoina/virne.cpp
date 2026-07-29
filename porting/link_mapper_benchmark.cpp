#include "controller/link_mapper.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace network = virne::network;

struct Fixture
{
    std::unique_ptr<network::VirtualNetwork> virtual_network;
    std::unique_ptr<network::PhysicalNetwork> physical_network;
    controller::PreparedLinkMapper mapper;
    core::Solution solution;
    std::vector<EdgeEndpoints> physical_edges;
    AttrId physical_bandwidth_value_id = 0U;
    controller::ConstraintId bandwidth_id = 0U;
    Vertex target = 0U;
};

struct BenchmarkResult
{
    std::uint64_t elapsed_ns = 0U;
    std::uint64_t checksum = 0U;
    std::size_t output_bytes = 0U;
};

std::uint64_t fingerprint(const std::string_view value)
{
    std::uint64_t result = 14695981039346656037ULL;
    for (const char raw_byte : value)
    {
        result ^= static_cast<unsigned char>(raw_byte);
        result *= 1099511628211ULL;
    }
    return result;
}

std::string double_token(const double value)
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16)
           << bits;
    return stream.str();
}

template <typename NumericVariant>
std::string numeric_token(const NumericVariant& value)
{
    return std::visit(
        [](const auto& item) -> std::string
        {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, bool>)
            {
                return item ? "b:1" : "b:0";
            }
            else if constexpr (std::is_same_v<Item, std::int64_t>)
            {
                return "i:" + std::to_string(item);
            }
            else if constexpr (std::is_same_v<Item, double>)
            {
                return double_token(item);
            }
            else
            {
                throw std::runtime_error(
                    "LinkMapper benchmark entered a non-numeric lane");
            }
        },
        value);
}

attribute::AttributeFactorySpec bandwidth_spec()
{
    attribute::AttributeFactorySpec result;
    result.name = "bw";
    result.owner = attribute::AttributeOwner::link;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = attribute::CheckingLevel::link;
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    const controller::ConstraintId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.link_attribute_specs = {bandwidth_spec()};
    network::VirtualNetwork result(std::move(construction));
    const auto binding = result.bind_link_attribute("bw");
    if (!binding)
    {
        throw std::runtime_error("virtual bandwidth binding is missing");
    }
    result.set_link_attrs_data({sparse_link_update(
        binding->registry_id,
        {{0U, 1U, std::int64_t{2}}})});
    return result;
}

network::PhysicalNetwork make_physical_network(
    const std::size_t candidate_count,
    std::vector<EdgeEndpoints>& physical_edges)
{
    const Vertex target = static_cast<Vertex>(candidate_count + 1U);
    physical_edges.clear();
    physical_edges.reserve(candidate_count * 2U);
    std::vector<attribute::LinkAttributeAssignment> capacities;
    capacities.reserve(candidate_count * 2U);

    for (std::size_t index = 0U; index < candidate_count; ++index)
    {
        const Vertex intermediate = static_cast<Vertex>(index + 1U);
        const std::int64_t capacity =
            index + 1U == candidate_count ? std::int64_t{2}
                                          : std::int64_t{1};
        physical_edges.emplace_back(0U, intermediate);
        physical_edges.emplace_back(intermediate, target);
        capacities.push_back({0U, intermediate, capacity});
        capacities.push_back({intermediate, target, capacity});
    }

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        candidate_count + 2U, physical_edges);
    construction.config.link_attribute_specs = {bandwidth_spec()};
    network::PhysicalNetwork result(std::move(construction));
    const auto binding = result.bind_link_attribute("bw");
    if (!binding)
    {
        throw std::runtime_error("physical bandwidth binding is missing");
    }
    result.set_link_attrs_data({sparse_link_update(
        binding->registry_id, std::move(capacities))});
    return result;
}

Fixture make_fixture(const std::size_t candidate_count)
{
    if (candidate_count >
        static_cast<std::size_t>(std::numeric_limits<Vertex>::max()) - 2U)
    {
        throw std::invalid_argument("candidate count exceeds Vertex range");
    }

    auto virtual_network =
        std::make_unique<network::VirtualNetwork>(make_virtual_network());
    std::vector<EdgeEndpoints> physical_edges;
    auto physical_network = std::make_unique<network::PhysicalNetwork>(
        make_physical_network(candidate_count, physical_edges));

    const auto virtual_bandwidth =
        virtual_network->bind_link_attribute("bw");
    const auto physical_bandwidth =
        physical_network->bind_link_attribute("bw");
    if (!virtual_bandwidth || !physical_bandwidth)
    {
        throw std::runtime_error("bandwidth benchmark binding is missing");
    }

    controller::LinkMapperSelection selection;
    selection.link_constraints = {virtual_bandwidth->registry_id};
    selection.link_resources = {virtual_bandwidth->registry_id};
    selection.hard_constraints = {virtual_bandwidth->registry_id};
    auto prepared = controller::LinkMapper(std::move(selection)).prepare(
        *virtual_network, *physical_network);

    core::SolutionMetadata metadata;
    metadata.v_net_id = 0;
    metadata.v_net_num_nodes = 2U;
    metadata.v_net_num_edges = 1U;
    core::Solution solution(metadata);

    return Fixture{
        std::move(virtual_network),
        std::move(physical_network),
        std::move(prepared),
        std::move(solution),
        std::move(physical_edges),
        physical_bandwidth->value_id,
        virtual_bandwidth->registry_id,
        static_cast<Vertex>(candidate_count + 1U),
    };
}

std::string constraint_link_token(const controller::ConstraintLink link)
{
    return "(" + std::to_string(link.source) + "," +
           std::to_string(link.target) + ")";
}

std::string solution_link_token(const core::SolutionLink link)
{
    return "(" + std::to_string(link.source) + "," +
           std::to_string(link.target) + ")";
}

std::string values_payload(
    const core::SolutionAttributeValues& values,
    const controller::ConstraintId bandwidth_id)
{
    std::size_t populated = 0U;
    for (const auto& slot : values.slots())
    {
        populated += slot.has_value() ? 1U : 0U;
    }
    const auto* bandwidth = values.find(bandwidth_id);
    if (bandwidth == nullptr)
    {
        if (populated != 0U)
        {
            throw std::runtime_error(
                "unexpected non-bandwidth Solution value");
        }
        return "{}";
    }
    if (populated != 1U)
    {
        throw std::runtime_error("unexpected extra Solution value");
    }
    return "{bw=" + numeric_token(*bandwidth) + "}";
}

template <typename Table>
std::string constraint_table_payload(
    const Table& table,
    const controller::ConstraintId bandwidth_id)
{
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0U; index < table.entries().size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        const auto& entry = table.entries()[index];
        stream << solution_link_token(entry.key) << ':'
               << values_payload(entry.value, bandwidth_id);
    }
    stream << ']';
    return stream.str();
}

std::string state_payload(
    const Fixture& fixture,
    const controller::LinkRouteResult& route)
{
    std::ostringstream stream;
    stream << "routed=" << (route.routed ? '1' : '0')
           << ";placeholder=" << (route.check.placeholder ? '1' : '0')
           << ";feasible="
           << (route.check.constraints.feasible ? '1' : '0')
           << ";phys=[";

    const Graph& graph = fixture.physical_network->graph();
    for (std::size_t index = 0U;
         index < fixture.physical_edges.size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        const auto [source, target] = fixture.physical_edges[index];
        const AttrValue& value = graph.edge_attrs(
            graph.edge(source, target)).at(fixture.physical_bandwidth_value_id);
        stream << '(' << source << ',' << target << ")="
               << numeric_token(value);
    }

    stream << "];paths=[";
    for (std::size_t index = 0U;
         index < fixture.solution.link_paths.entries().size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        const auto& entry = fixture.solution.link_paths.entries()[index];
        stream << solution_link_token(entry.key) << ":[";
        for (std::size_t link_index = 0U;
             link_index < entry.value.size(); ++link_index)
        {
            if (link_index != 0U)
            {
                stream << ',';
            }
            stream << solution_link_token(entry.value[link_index]);
        }
        stream << ']';
    }

    stream << "];info=[";
    for (std::size_t index = 0U;
         index < fixture.solution.link_paths_info.entries().size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        const auto& entry = fixture.solution.link_paths_info.entries()[index];
        stream << solution_link_token(entry.key.virtual_link) << '@'
               << solution_link_token(entry.key.physical_link) << ':'
               << values_payload(entry.value, fixture.bandwidth_id);
    }

    stream << "];check_link=[";
    for (std::size_t index = 0U;
         index < route.check.constraints.link_level.size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        const auto& item = route.check.constraints.link_level[index];
        stream << constraint_link_token(item.physical_link) << ':'
               << values_payload(item.offsets, fixture.bandwidth_id);
    }
    stream << "];check_path="
           << values_payload(
                  route.check.constraints.path_level,
                  fixture.bandwidth_id)
           << ";offset_link="
           << constraint_table_payload(
                  fixture.solution.v_net_constraint_offsets.link_level,
                  fixture.bandwidth_id)
           << ";offset_path="
           << constraint_table_payload(
                  fixture.solution.v_net_constraint_offsets.path_level,
                  fixture.bandwidth_id)
           << ";violation_link="
           << constraint_table_payload(
                  fixture.solution.v_net_constraint_violations.link_level,
                  fixture.bandwidth_id)
           << ";violation_path="
           << constraint_table_payload(
                  fixture.solution.v_net_constraint_violations.path_level,
                  fixture.bandwidth_id)
           << ";total="
           << double_token(
                  fixture.solution.v_net_total_hard_constraint_violation)
           << ";flags=" << (fixture.solution.result ? '1' : '0') << ','
           << (fixture.solution.place_result ? '1' : '0') << ','
           << (fixture.solution.route_result ? '1' : '0');
    return stream.str();
}

void validate_selected_path(
    const Fixture& fixture,
    const controller::LinkRouteResult& result,
    const std::size_t candidate_count)
{
    if (!result.routed || result.check.placeholder ||
        !result.check.constraints.feasible)
    {
        throw std::runtime_error("valid all-shortest route failed");
    }
    const core::SolutionLink virtual_link{0, 1};
    const auto route_id = fixture.solution.link_paths.find_id(virtual_link);
    if (!route_id)
    {
        throw std::runtime_error("selected route is missing");
    }
    const auto& path = fixture.solution.link_paths.at(*route_id);
    const core::SolutionNodeId intermediate =
        static_cast<core::SolutionNodeId>(candidate_count);
    const core::SolutionNodeId target =
        static_cast<core::SolutionNodeId>(candidate_count + 1U);
    if (path.size() != 2U ||
        path[0] != core::SolutionLink{0, intermediate} ||
        path[1] != core::SolutionLink{intermediate, target})
    {
        throw std::runtime_error("all-shortest candidate order drifted");
    }
    if (fixture.solution.link_paths_info.size() != 2U ||
        fixture.solution.v_net_constraint_offsets.link_level.size() != 1U ||
        fixture.solution.v_net_constraint_offsets.path_level.size() != 1U ||
        fixture.solution.v_net_constraint_violations.link_level.size() != 1U ||
        fixture.solution.v_net_constraint_violations.path_level.size() != 1U)
    {
        throw std::runtime_error("complete route state is missing");
    }
}

BenchmarkResult run_benchmark(
    const std::size_t candidate_count,
    const std::size_t candidate_workers)
{
    Fixture fixture = make_fixture(candidate_count);
    controller::LinkRouteOptions options;
    options.shortest_method = controller::ShortestPathMethod::all_shortest;
    options.k = static_cast<std::int64_t>(candidate_count);
    options.topology_constraint_workers = 1U;
    options.candidate_workers = candidate_workers;
    options.allow_constraint_violation = false;
    options.record_constraint_violation = true;

    const auto begin = std::chrono::steady_clock::now();
    const controller::LinkRouteResult route = fixture.mapper.route(
        {0U, 1U}, {0U, fixture.target}, fixture.solution, options);
    const auto end = std::chrono::steady_clock::now();

    validate_selected_path(fixture, route, candidate_count);
    const std::string payload = state_payload(fixture, route);
    return BenchmarkResult{
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - begin).count()),
        fingerprint(payload),
        payload.size(),
    };
}

std::size_t parse_size(const char* text, const char* name)
{
    const std::string value(text);
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size())
    {
        throw std::invalid_argument(std::string(name) + " is invalid");
    }
    if (parsed > std::numeric_limits<std::size_t>::max())
    {
        throw std::out_of_range(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(const int argc, char** argv)
{
    try
    {
        if (argc != 3)
        {
            throw std::invalid_argument(
                "usage: link_mapper_benchmark candidate_paths candidate_workers");
        }
        const std::size_t candidate_count =
            parse_size(argv[1], "candidate_paths");
        const std::size_t candidate_workers =
            parse_size(argv[2], "candidate_workers");
        if (candidate_count < 2U)
        {
            throw std::invalid_argument(
                "candidate_paths must be at least two");
        }

        const BenchmarkResult result =
            run_benchmark(candidate_count, candidate_workers);
        std::cout << "protocol=1\n"
                  << "kind=link_mapper_safe_all_shortest\n"
                  << "candidate_paths=" << candidate_count << '\n'
                  << "physical_edges=" << candidate_count * 2U << '\n'
                  << "candidate_workers=" << candidate_workers << '\n'
                  << "type_tag=ordered_link_route_state_v1\n"
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "status=PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "link_mapper_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
