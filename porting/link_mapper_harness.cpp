#include "link_mapper.h"

#include <algorithm>
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
using controller::ConstraintId;
using controller::ConstraintLink;
using controller::LinkMapper;
using controller::LinkMapperErrorCode;
using controller::LinkMapperException;
using controller::LinkMapperSelection;
using controller::LinkMappingOptions;
using controller::LinkPathRanker;
using controller::LinkRouteCheckInfo;
using controller::LinkRouteOptions;
using controller::LinkRouteResult;
using controller::PhysicalLinkConstraintResult;
using controller::PreparedLinkMapper;
using controller::ShortestPathMethod;
using virne::core::LinkPathInfoKey;
using virne::core::Solution;
using virne::core::SolutionAttributeValues;
using virne::core::SolutionLink;
using virne::core::SolutionMetadata;

constexpr ConstraintLink virtual_link{0U, 1U};

const std::vector<ConstraintLink>& physical_links()
{
    static const std::vector<ConstraintLink> value{
        {0U, 1U}, {1U, 4U},
        {0U, 2U}, {2U, 4U},
        {0U, 3U}, {3U, 4U}};
    return value;
}

AttributeFactorySpec resource_spec(std::string name)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = CheckingLevel::link;
    return result;
}

AttributeFactorySpec latency_spec(std::string name)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::latency;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = CheckingLevel::path;
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

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.link_attribute_specs = {
        resource_spec("capacity"), latency_spec("latency")};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        6U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 4U},
            {0U, 2U}, {2U, 4U},
            {0U, 3U}, {3U, 4U}});
    // Reverse independent physical registry order.
    construction.config.link_attribute_specs = {
        latency_spec("latency"), resource_spec("capacity")};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing link mapper harness binding");
    }
    return *binding;
}

Solution make_solution(bool with_slots = false)
{
    SolutionMetadata metadata;
    metadata.v_net_id = 11;
    metadata.v_net_num_nodes = 2U;
    metadata.v_net_num_edges = 1U;
    Solution result(metadata);
    result.route_result = true;
    result.result = true;
    if (with_slots)
    {
        result.node_slots.insert_or_assign(0, 0);
        result.node_slots.insert_or_assign(1, 4);
    }
    return result;
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();
    ConstraintId virtual_capacity = 0U;
    ConstraintId virtual_latency = 0U;
    ConstraintId physical_capacity = 0U;
    ConstraintId physical_latency = 0U;
    AttrId physical_capacity_value = 0U;
    bool reusable = false;

    explicit Fixture(bool reusable_value = false)
        : reusable(reusable_value)
    {
        const auto virtual_capacity_binding = require_link_binding(
            virtual_network, "capacity");
        const auto virtual_latency_binding = require_link_binding(
            virtual_network, "latency");
        const auto physical_capacity_binding = require_link_binding(
            physical_network, "capacity");
        const auto physical_latency_binding = require_link_binding(
            physical_network, "latency");

        virtual_capacity = virtual_capacity_binding.registry_id;
        virtual_latency = virtual_latency_binding.registry_id;
        physical_capacity = physical_capacity_binding.registry_id;
        physical_latency = physical_latency_binding.registry_id;
        physical_capacity_value = physical_capacity_binding.value_id;

        if (virtual_capacity == physical_capacity ||
            virtual_latency == physical_latency)
        {
            throw std::runtime_error(
                "link mapper fixture registry IDs are not independent");
        }

        set_virtual(std::int64_t{5}, 6.0);
        physical_network.set_link_attrs_data({
            sparse_link_update(
                physical_capacity,
                {{0U, 1U, std::int64_t{4}},
                 {1U, 4U, std::int64_t{4}},
                 {0U, 2U, std::int64_t{5}},
                 {2U, 4U, std::int64_t{5}},
                 {0U, 3U, std::int64_t{5}},
                 {3U, 4U, std::int64_t{5}}}),
            sparse_link_update(
                physical_latency,
                {{0U, 1U, 3.0}, {1U, 4U, 3.0},
                 {0U, 2U, 3.0}, {2U, 4U, 3.0},
                 {0U, 3U, 3.0}, {3U, 4U, 3.0}})});
    }

    void set_virtual(std::int64_t capacity, double latency)
    {
        virtual_network.set_link_attrs_data({
            sparse_link_update(
                virtual_capacity, {{0U, 1U, capacity}}),
            sparse_link_update(
                virtual_latency, {{0U, 1U, latency}})});
    }

    PreparedLinkMapper prepare()
    {
        LinkMapperSelection selection;
        selection.link_constraints = {virtual_capacity};
        selection.path_constraints = {virtual_latency};
        selection.link_resources = {virtual_capacity};
        selection.hard_constraints = {
            virtual_capacity, virtual_latency};
        selection.reusable = reusable;
        return LinkMapper(std::move(selection)).prepare(
            virtual_network, physical_network);
    }
};

std::string double_token(double value)
{
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

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
    return double_token(std::get<double>(value));
}

std::string attr_token(const AttrValue& value)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        return double_token(*floating);
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? "b:1" : "b:0";
    }
    throw std::runtime_error("non-numeric link mapper fixture value");
}

std::string values_payload(
    const SolutionAttributeValues& values,
    const std::vector<ConstraintId>& order)
{
    std::string result = "{";
    bool first = true;
    for (const ConstraintId id : order)
    {
        const AttributeNumber* value = values.find(id);
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
    result.push_back('}');
    return result;
}

std::string link_token(const SolutionLink& link)
{
    return std::to_string(link.source) + ":" +
        std::to_string(link.target);
}

std::string solution_payload(
    const Fixture& fixture,
    const Solution& solution)
{
    std::string result = "phys=[";
    const Graph& graph = fixture.physical_network.graph();
    for (std::size_t index = 0U;
         index < physical_links().size();
         ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const ConstraintLink link = physical_links()[index];
        result += attr_token(
            graph.edge_attrs(graph.edge(link.source, link.target)).at(
                fixture.physical_capacity_value));
    }
    result += "];paths=[";
    bool first = true;
    for (const auto& entry : solution.link_paths.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += link_token(entry.key) + "{";
        for (std::size_t index = 0U;
             index < entry.value.size();
             ++index)
        {
            if (index != 0U)
            {
                result.push_back(',');
            }
            result += link_token(entry.value[index]);
        }
        result.push_back('}');
    }
    result += "];info=[";
    first = true;
    for (const auto& entry : solution.link_paths_info.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += link_token(entry.key.virtual_link) + "/" +
            link_token(entry.key.physical_link) +
            values_payload(entry.value, {fixture.virtual_capacity});
    }
    const std::vector<ConstraintId> link_order{fixture.virtual_capacity};
    const std::vector<ConstraintId> path_order{fixture.virtual_latency};
    result += "];lo=[";
    first = true;
    for (const auto& entry :
         solution.v_net_constraint_offsets.link_level.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += link_token(entry.key) +
            values_payload(entry.value, link_order);
    }
    result += "];po=[";
    first = true;
    for (const auto& entry :
         solution.v_net_constraint_offsets.path_level.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += link_token(entry.key) +
            values_payload(entry.value, path_order);
    }
    result += "];lv=[";
    first = true;
    for (const auto& entry :
         solution.v_net_constraint_violations.link_level.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += link_token(entry.key) +
            values_payload(entry.value, link_order);
    }
    result += "];pv=[";
    first = true;
    for (const auto& entry :
         solution.v_net_constraint_violations.path_level.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += link_token(entry.key) +
            values_payload(entry.value, path_order);
    }
    result += "];total=" + double_token(
        solution.v_net_total_hard_constraint_violation);
    result += ";flags=";
    result.push_back(solution.route_result ? '1' : '0');
    result.push_back(',');
    result.push_back(solution.result ? '1' : '0');
    return result;
}

std::string route_payload(
    const LinkRouteResult& route,
    const Fixture& fixture,
    const Solution& solution)
{
    return std::string("routed=") + (route.routed ? "1" : "0") +
        ";placeholder=" + (route.check.placeholder ? "1" : "0") +
        ";" + solution_payload(fixture, solution);
}

std::string mapping_payload(
    bool mapped,
    const Fixture& fixture,
    const Solution& solution)
{
    return std::string("mapped=") + (mapped ? "1" : "0") +
        ";" + solution_payload(fixture, solution);
}

LinkRouteOptions route_options(bool record = true)
{
    LinkRouteOptions result;
    result.shortest_method = ShortestPathMethod::k_shortest;
    result.k = 3;
    result.record_constraint_violation = record;
    return result;
}

LinkMappingOptions mapping_options(bool inplace = true)
{
    LinkMappingOptions result;
    result.shortest_method = ShortestPathMethod::k_shortest;
    result.k = 3;
    result.inplace = inplace;
    return result;
}

LinkRouteCheckInfo manual_check(
    ConstraintId capacity,
    ConstraintId latency,
    const std::vector<std::int64_t>& link_offsets,
    double path_offset)
{
    LinkRouteCheckInfo result;
    result.constraints.feasible = false;
    for (std::size_t index = 0U;
         index < link_offsets.size();
         ++index)
    {
        SolutionAttributeValues offsets;
        offsets.set(capacity, link_offsets[index]);
        const ConstraintLink endpoints = physical_links()[index];
        result.constraints.link_level.push_back(
            PhysicalLinkConstraintResult{endpoints, std::move(offsets)});
    }
    result.constraints.path_level.set(latency, path_offset);
    return result;
}

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

std::string run_safe(std::size_t workers)
{
    Fixture fixture;
    PreparedLinkMapper mapper = fixture.prepare();
    Solution solution = make_solution();
    LinkRouteOptions options = route_options();
    options.candidate_workers = workers;
    const LinkRouteResult route = mapper.route(
        virtual_link, {0U, 4U}, solution, options);
    return route_payload(route, fixture, solution);
}

void differential()
{
    {
        Fixture fixture(true);
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 0U}, solution, route_options());
        emit(
            "same_node_reusable",
            route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        std::string error = "none";
        try
        {
            static_cast<void>(mapper.route(
                virtual_link, {0U, 0U}, solution, route_options()));
        }
        catch (const LinkMapperException& exception)
        {
            if (exception.code() != LinkMapperErrorCode::same_physical_node)
            {
                throw;
            }
            error = "same_node";
        }
        emit(
            "same_node_nonreusable",
            "error=" + error + ";" +
                solution_payload(fixture, solution));
    }

    emit("safe_success", run_safe(1U));

    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 5U}, solution, route_options());
        emit("safe_no_path", route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        fixture.set_virtual(std::int64_t{5}, 5.0);
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 4U}, solution, route_options());
        emit(
            "safe_all_infeasible",
            route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const LinkPathRanker reverse_ranker =
            [](controller::PhysicalPaths& paths)
            {
                std::reverse(paths.begin(), paths.end());
            };
        LinkRouteOptions options = route_options();
        options.ranker = &reverse_ranker;
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 4U}, solution, options);
        emit(
            "safe_ranker_reverse",
            route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        static_cast<void>(mapper.route(
            virtual_link, {0U, 4U}, solution, route_options(false)));
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 5U}, solution, route_options(false));
        emit(
            "reroute_resource_leak",
            route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        LinkRouteOptions options = route_options();
        options.allow_constraint_violation = true;
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 4U}, solution, options);
        emit(
            "unsafe_first_feasible",
            route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        fixture.set_virtual(std::int64_t{6}, 5.0);
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        LinkRouteOptions options = route_options();
        options.allow_constraint_violation = true;
        const LinkRouteResult route = mapper.route(
            virtual_link, {0U, 4U}, solution, options);
        emit(
            "unsafe_least_violation_tie",
            route_payload(route, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const LinkRouteCheckInfo check = manual_check(
            fixture.virtual_capacity,
            fixture.virtual_latency,
            {-2, 3, 4},
            -1.0);
        mapper.record_route_constraint_violation(
            virtual_link, check, solution);
        emit("pooling_mixed", solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const LinkRouteCheckInfo check = manual_check(
            fixture.virtual_capacity,
            fixture.virtual_latency,
            {-2, 3, 4},
            -1.0);
        mapper.record_route_constraint_violation(
            virtual_link, check, solution);
        mapper.record_route_constraint_violation(
            virtual_link, check, solution);
        emit("pooling_repeated", solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        static_cast<void>(mapper.route(
            virtual_link, {0U, 4U}, solution, route_options(false)));
        const bool undone = mapper.undo_route(virtual_link, solution);
        emit(
            "undo_success",
            std::string("undone=") + (undone ? "1;" : "0;") +
                solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        static_cast<void>(mapper.route(
            virtual_link, {0U, 4U}, solution, route_options(false)));
        solution.link_paths_info.erase(
            LinkPathInfoKey{
                SolutionLink{0, 1}, SolutionLink{2, 4}});
        std::string error = "none";
        try
        {
            static_cast<void>(mapper.undo_route(virtual_link, solution));
        }
        catch (const LinkMapperException& exception)
        {
            if (exception.code() != LinkMapperErrorCode::route_info_not_found)
            {
                throw;
            }
            error = "missing_info";
        }
        emit(
            "undo_partial_missing_info",
            "error=" + error + ";" +
                solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution(true);
        const bool mapped = mapper.link_mapping(
            solution, mapping_options());
        emit(
            "mapping_success",
            mapping_payload(mapped, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution(true);
        const bool mapped = mapper.link_mapping(
            solution, mapping_options(false));
        emit(
            "mapping_clone",
            mapping_payload(mapped, fixture, solution));
    }
    {
        Fixture fixture;
        fixture.set_virtual(std::int64_t{5}, 5.0);
        PreparedLinkMapper mapper = fixture.prepare();
        Solution solution = make_solution(true);
        const bool mapped = mapper.link_mapping(
            solution, mapping_options());
        emit(
            "mapping_failure",
            mapping_payload(mapped, fixture, solution));
    }

    std::string workers_payload;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        if (!workers_payload.empty())
        {
            workers_payload.push_back('|');
        }
        workers_payload += run_safe(workers);
    }
    emit("workers_0_1_2_8", workers_payload);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 ||
            (argc == 2 &&
             std::string_view(argv[1]) == "differential"))
        {
            differential();
            return 0;
        }
        throw std::invalid_argument(
            "usage: link_mapper_harness [differential]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "link mapper harness: " << error.what() << '\n';
        return 1;
    }
}
