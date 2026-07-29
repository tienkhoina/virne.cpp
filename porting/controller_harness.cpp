#include "controller.h"

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
#include <utility>
#include <variant>
#include <vector>

namespace {

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
using controller::Controller;
using controller::ControllerFailurePhase;
using controller::ControllerMutationOptions;
using controller::ControllerSelection;
using controller::PlaceAndRouteOptions;
using controller::PlaceAndRouteResult;
using controller::PreparedController;
using controller::ShortestPathMethod;
using virne::core::LinkPathInfoKey;
using virne::core::NodeSlotInfoKey;
using virne::core::Solution;
using virne::core::SolutionAttributeValues;
using virne::core::SolutionLink;
using virne::core::SolutionMetadata;

constexpr std::size_t virtual_node_count = 3U;
constexpr std::size_t physical_node_count = 6U;

const std::vector<ConstraintLink>& physical_links()
{
    static const std::vector<ConstraintLink> links{
        {0U, 1U}, {1U, 4U}, {0U, 2U}, {2U, 5U}};
    return links;
}

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner,
    CheckingLevel level)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = level;
    return result;
}

network::NodeAttributeDataUpdate node_update(
    ConstraintId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate link_update(
    ConstraintId id,
    std::vector<AttrValue> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        virtual_node_count,
        std::vector<EdgeEndpoints>{{0U, 1U}, {0U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node, CheckingLevel::node)};
    construction.config.link_attribute_specs = {
        resource_spec(
            "bandwidth", AttributeOwner::link, CheckingLevel::link)};
    network::VirtualNetwork result(std::move(construction));
    result.set_request_id(17);
    result.set_arrival_time(1.25);
    result.set_lifetime(4.0);
    return result;
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        physical_node_count,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 4U}, {0U, 2U}, {2U, 5U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node, CheckingLevel::node)};
    construction.config.link_attribute_specs = {
        resource_spec(
            "bandwidth", AttributeOwner::link, CheckingLevel::link)};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing Controller node binding");
    }
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing Controller link binding");
    }
    return *binding;
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();
    network::NodeNetworkAttributeBinding virtual_cpu =
        require_node_binding(virtual_network, "cpu");
    network::LinkNetworkAttributeBinding virtual_bandwidth =
        require_link_binding(virtual_network, "bandwidth");
    network::NodeNetworkAttributeBinding physical_cpu =
        require_node_binding(physical_network, "cpu");
    network::LinkNetworkAttributeBinding physical_bandwidth =
        require_link_binding(physical_network, "bandwidth");
    std::optional<Controller> controller;

    Fixture()
    {
        virtual_network.set_node_attrs_data({
            node_update(
                virtual_cpu.registry_id,
                {std::int64_t{4}, std::int64_t{2}, std::int64_t{2}})});
        virtual_network.set_link_attrs_data({
            link_update(
                virtual_bandwidth.registry_id,
                {std::int64_t{3}, std::int64_t{6}})});
        physical_network.set_node_attrs_data({
            node_update(
                physical_cpu.registry_id,
                {std::int64_t{10}, std::int64_t{10},
                 std::int64_t{10}, std::int64_t{3},
                 std::int64_t{10}, std::int64_t{10}})});
        physical_network.set_link_attrs_data({
            link_update(
                physical_bandwidth.registry_id,
                {std::int64_t{5}, std::int64_t{5},
                 std::int64_t{4}, std::int64_t{4}})});

        ControllerSelection selection;
        selection.constraints.node_at_node = {virtual_cpu.registry_id};
        selection.constraints.link_at_link = {
            virtual_bandwidth.registry_id};
        selection.node_resources = {virtual_cpu.registry_id};
        selection.link_resources = {virtual_bandwidth.registry_id};
        selection.hard_node_constraints = {virtual_cpu.registry_id};
        selection.hard_link_constraints = {
            virtual_bandwidth.registry_id};
        controller.emplace(std::move(selection));
    }

    PreparedController prepare()
    {
        return controller->prepare(virtual_network, physical_network);
    }
};

Solution make_solution(bool result_value = true)
{
    SolutionMetadata metadata;
    metadata.v_net_id = 17;
    metadata.v_net_lifetime = 4.0;
    metadata.v_net_arrival_time = 1.25;
    metadata.v_net_num_nodes = virtual_node_count;
    metadata.v_net_num_edges = 2U;
    Solution result(metadata);
    result.result = result_value;
    result.place_result = true;
    result.route_result = true;
    return result;
}

SolutionAttributeValues one_value(ConstraintId id, std::int64_t value)
{
    SolutionAttributeValues result;
    result.set(id, value);
    return result;
}

Solution manual_solution(
    const Fixture& fixture,
    bool result_value = true,
    std::int64_t second_cpu = 2)
{
    Solution result = make_solution(result_value);
    result.node_slots.insert_or_assign(0, 0);
    result.node_slots.insert_or_assign(1, 4);
    result.node_slots_info.insert_or_assign(
        NodeSlotInfoKey{0, 0},
        one_value(fixture.virtual_cpu.registry_id, 4));
    result.node_slots_info.insert_or_assign(
        NodeSlotInfoKey{1, 4},
        one_value(fixture.virtual_cpu.registry_id, second_cpu));

    const SolutionLink virtual_link{0, 1};
    result.link_paths.insert_or_assign(
        virtual_link,
        std::vector<SolutionLink>{{0, 1}, {1, 4}});
    result.link_paths_info.insert_or_assign(
        LinkPathInfoKey{virtual_link, {0, 1}},
        one_value(fixture.virtual_bandwidth.registry_id, 3));
    result.link_paths_info.insert_or_assign(
        LinkPathInfoKey{virtual_link, {1, 4}},
        one_value(fixture.virtual_bandwidth.registry_id, 3));
    return result;
}

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
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? "b:1" : "b:0";
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        return double_token(*floating);
    }
    throw std::runtime_error("Controller fixture contains nonnumeric data");
}

std::string values_payload(
    const SolutionAttributeValues& values,
    ConstraintId id)
{
    const AttributeNumber* value = values.find(id);
    return value == nullptr
        ? "{}"
        : "{" + std::to_string(id) + "=" + number_token(*value) + "}";
}

std::string link_token(const SolutionLink link)
{
    return std::to_string(link.source) + ":" +
        std::to_string(link.target);
}

std::string physical_payload(const Fixture& fixture)
{
    const Graph& graph = fixture.physical_network.graph();
    std::string result = "N[";
    for (Vertex node = 0U; node < physical_node_count; ++node)
    {
        if (node != 0U)
        {
            result.push_back(',');
        }
        result += attr_token(
            graph.node_attrs(node).at(fixture.physical_cpu.value_id));
    }
    result += "]L[";
    for (std::size_t index = 0U; index < physical_links().size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const ConstraintLink link = physical_links()[index];
        result += attr_token(
            graph.edge_attrs(graph.edge(link.source, link.target)).at(
                fixture.physical_bandwidth.value_id));
    }
    result.push_back(']');
    return result;
}

std::string solution_payload(const Fixture& fixture, const Solution& solution)
{
    const ConstraintId cpu = fixture.virtual_cpu.registry_id;
    const ConstraintId bandwidth = fixture.virtual_bandwidth.registry_id;
    std::string result = physical_payload(fixture) + ";slots=[";
    bool first = true;
    for (const auto& entry : solution.node_slots.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += std::to_string(entry.key) + ":" +
            std::to_string(entry.value);
    }
    result += "];paths=[";
    first = true;
    for (const auto& entry : solution.link_paths.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += link_token(entry.key) + "{";
        for (std::size_t index = 0U; index < entry.value.size(); ++index)
        {
            if (index != 0U) result.push_back(',');
            result += link_token(entry.value[index]);
        }
        result.push_back('}');
    }
    result += "];ni=[";
    first = true;
    for (const auto& entry : solution.node_slots_info.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += std::to_string(entry.key.virtual_node) + ":" +
            std::to_string(entry.key.physical_node) +
            values_payload(entry.value, cpu);
    }
    result += "];li=[";
    first = true;
    for (const auto& entry : solution.link_paths_info.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += link_token(entry.key.virtual_link) + "/" +
            link_token(entry.key.physical_link) +
            values_payload(entry.value, bandwidth);
    }
    result += "];no=[";
    first = true;
    for (const auto& entry : solution.v_net_constraint_offsets.node_level.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += std::to_string(entry.key) +
            values_payload(entry.value, cpu);
    }
    result += "];lo=[";
    first = true;
    for (const auto& entry : solution.v_net_constraint_offsets.link_level.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += link_token(entry.key) +
            values_payload(entry.value, bandwidth);
    }
    result += "];nv=[";
    first = true;
    for (const auto& entry : solution.v_net_constraint_violations.node_level.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += std::to_string(entry.key) +
            values_payload(entry.value, cpu);
    }
    result += "];lv=[";
    first = true;
    for (const auto& entry : solution.v_net_constraint_violations.link_level.entries())
    {
        if (!first) result.push_back(',');
        first = false;
        result += link_token(entry.key) +
            values_payload(entry.value, bandwidth);
    }
    result += "];step=N" + values_payload(
        solution.v_net_single_step_constraint_offset.node_level, cpu);
    result += "L" + values_payload(
        solution.v_net_single_step_constraint_offset.link_level, bandwidth);
    result += "P{};total=" + double_token(
        solution.v_net_total_hard_constraint_violation);
    result += ";single=" + double_token(
        solution.v_net_single_step_hard_constraint_offset);
    result += ";maximum=" + double_token(
        solution.v_net_max_single_step_hard_constraint_violation);
    result += ";flags=";
    result.push_back(solution.place_result ? '1' : '0');
    result.push_back(',');
    result.push_back(solution.route_result ? '1' : '0');
    result.push_back(',');
    result.push_back(solution.result ? '1' : '0');
    return result;
}

std::string phase_token(ControllerFailurePhase phase)
{
    switch (phase)
    {
    case ControllerFailurePhase::none: return "none";
    case ControllerFailurePhase::place: return "place";
    case ControllerFailurePhase::route: return "route";
    }
    throw std::runtime_error("invalid Controller phase");
}

std::string place_payload(
    const PlaceAndRouteResult& operation,
    const Fixture& fixture,
    const Solution& solution)
{
    const std::string last = !operation.last_route.has_value()
        ? "none"
        : (operation.last_route->routed ? "1" : "0");
    return std::string("ok=") + (operation.succeeded ? "1" : "0") +
        ";phase=" + phase_token(operation.failure_phase) +
        ";placed=" + (operation.placement.placed ? "1" : "0") +
        ";routes=" + std::to_string(operation.attempted_routes) +
        ";last=" + last + ";" + solution_payload(fixture, solution);
}

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

PlaceAndRouteOptions route_options(std::size_t workers = 1U)
{
    PlaceAndRouteOptions result;
    result.shortest_method = ShortestPathMethod::bfs_shortest;
    result.workers.topology_constraint_workers = workers;
    result.workers.candidate_workers = workers;
    return result;
}

void differential()
{
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = make_solution();
        const auto result = prepared.place_and_route(
            1U, 4U, solution, route_options());
        emit("safe_no_neighbor", place_payload(result, fixture, solution));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = make_solution();
        (void)prepared.place_and_route(1U, 4U, solution, route_options());
        const auto result = prepared.place_and_route(
            0U, 0U, solution, route_options());
        emit("safe_one_route", place_payload(result, fixture, solution));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = make_solution();
        const auto result = prepared.place_and_route(
            0U, 3U, solution, route_options());
        emit("placement_failure", place_payload(result, fixture, solution));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = make_solution();
        (void)prepared.place_and_route(1U, 4U, solution, route_options());
        (void)prepared.place_and_route(2U, 5U, solution, route_options());
        const auto result = prepared.place_and_route(
            0U, 0U, solution, route_options());
        emit("route_partial_failure", place_payload(result, fixture, solution));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = make_solution();
        (void)prepared.place_and_route(1U, 4U, solution, route_options());
        (void)prepared.place_and_route(0U, 0U, solution, route_options());
        const bool undone = prepared.undo_place_and_route(0U, solution);
        emit(
            "undo_place_route",
            std::string("undone=") + (undone ? "1;" : "0;") +
                solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = manual_solution(fixture, false);
        const bool deployed = prepared.deploy(solution);
        const bool released = prepared.release(solution);
        emit(
            "unsuccessful_noop",
            std::string("deploy=") + (deployed ? "1" : "0") +
                ";release=" + (released ? "1;" : "0;") +
                solution_payload(fixture, solution));
    }

    std::string roundtrip;
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = manual_solution(fixture);
        const bool deployed = prepared.deploy(
            solution, ControllerMutationOptions{workers});
        const std::string after_deploy = solution_payload(fixture, solution);
        const bool released = prepared.release(
            solution, ControllerMutationOptions{workers});
        if (!roundtrip.empty()) roundtrip.push_back('|');
        roundtrip += "w=" + std::to_string(workers) +
            "{d=" + (deployed ? "1;" : "0;") + after_deploy +
            ";r=" + (released ? "1;" : "0;") +
            solution_payload(fixture, solution) + "}";
    }
    emit("roundtrip_workers_1_2_8", roundtrip);

    std::string partial;
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = manual_solution(fixture, true, 99);
        std::string error = "none";
        try
        {
            (void)prepared.deploy(
                solution, ControllerMutationOptions{workers});
        }
        catch (const std::exception&)
        {
            error = "resource";
        }
        if (!partial.empty()) partial.push_back('|');
        partial += "w=" + std::to_string(workers) +
            "{error=" + error + ";" +
            solution_payload(fixture, solution) + "}";
    }
    emit("deploy_partial_workers_1_2_8", partial);

    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = manual_solution(fixture);
        (void)prepared.deploy(solution);
        (void)solution.node_slots_info.erase(NodeSlotInfoKey{1, 4});
        std::string error = "none";
        try
        {
            (void)prepared.release(solution);
        }
        catch (const std::exception&)
        {
            error = "missing";
        }
        emit(
            "release_partial_missing_info",
            "error=" + error + ";" + solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        auto prepared = fixture.prepare();
        Solution solution = manual_solution(fixture);
        (void)prepared.deploy(solution);
        const bool undone = prepared.undo_deploy(solution);
        emit(
            "undo_deploy_quirk",
            std::string("undone=") + (undone ? "1;" : "0;") +
                solution_payload(fixture, solution));
    }
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
        throw std::invalid_argument(
            "usage: controller_harness [differential]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "controller_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
