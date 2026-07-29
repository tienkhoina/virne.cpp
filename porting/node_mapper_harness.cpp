#include "node_mapper.h"

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
using controller::NodeMapper;
using controller::NodeMapperSelection;
using controller::NodeMappingOptions;
using controller::NodeMatchingMethod;
using controller::NodePlacementOptions;
using controller::NodePlacementResult;
using controller::PreparedNodeMapper;
using virne::core::Solution;
using virne::core::SolutionAttributeValues;
using virne::core::SolutionMetadata;

AttributeFactorySpec resource_spec(
    std::string name,
    ConstraintRestriction restriction)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::node;
    result.kind = AttributeKind::resource;
    result.restriction = restriction;
    result.checking_level = CheckingLevel::node;
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

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", ConstraintRestriction::hard),
        resource_spec("quality", ConstraintRestriction::soft)};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U, std::vector<EdgeEndpoints>{});
    // Deliberately reverse the independent registry order.
    construction.config.node_attribute_specs = {
        resource_spec("quality", ConstraintRestriction::soft),
        resource_spec("cpu", ConstraintRestriction::hard)};
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
        throw std::runtime_error("missing node mapper harness binding");
    }
    return *binding;
}

Solution make_solution()
{
    SolutionMetadata metadata;
    metadata.v_net_id = 7;
    metadata.v_net_num_nodes = 2U;
    Solution result(metadata);
    result.place_result = true;
    result.result = true;
    return result;
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();
    ConstraintId virtual_cpu = 0U;
    ConstraintId virtual_quality = 0U;
    ConstraintId physical_cpu = 0U;
    ConstraintId physical_quality = 0U;
    AttrId physical_cpu_value = 0U;

    Fixture()
    {
        const auto virtual_cpu_binding = require_node_binding(
            virtual_network, "cpu");
        const auto virtual_quality_binding = require_node_binding(
            virtual_network, "quality");
        const auto physical_cpu_binding = require_node_binding(
            physical_network, "cpu");
        const auto physical_quality_binding = require_node_binding(
            physical_network, "quality");

        virtual_cpu = virtual_cpu_binding.registry_id;
        virtual_quality = virtual_quality_binding.registry_id;
        physical_cpu = physical_cpu_binding.registry_id;
        physical_quality = physical_quality_binding.registry_id;
        physical_cpu_value = physical_cpu_binding.value_id;

        if (virtual_cpu == physical_cpu ||
            virtual_quality == physical_quality)
        {
            throw std::runtime_error(
                "node mapper fixture did not separate registry IDs");
        }

        virtual_network.set_node_attrs_data({
            dense_node_update(
                virtual_cpu,
                {std::int64_t{4}, std::int64_t{6}}),
            dense_node_update(
                virtual_quality,
                {5.0, 1.0})});
        physical_network.set_node_attrs_data({
            dense_node_update(
                physical_cpu,
                {std::int64_t{5}, std::int64_t{3},
                 std::int64_t{10}, std::int64_t{1}}),
            dense_node_update(
                physical_quality,
                {2.0, 10.0, 10.0, 1.0})});
    }

    PreparedNodeMapper prepare()
    {
        NodeMapperSelection selection;
        selection.node_constraints = {virtual_cpu, virtual_quality};
        selection.node_resources = {virtual_cpu};
        selection.hard_constraints = {virtual_cpu};
        return NodeMapper(std::move(selection)).prepare(
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

std::string numeric_attr_token(const AttrValue& value)
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
    throw std::runtime_error("non-numeric node mapper fixture value");
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

std::string solution_payload(
    const Fixture& fixture,
    const Solution& solution)
{
    const std::vector<ConstraintId> constraints{
        fixture.virtual_cpu, fixture.virtual_quality};
    const std::vector<ConstraintId> resources{fixture.virtual_cpu};
    std::string result = "phys=[";
    for (Vertex node = 0U;
         node < fixture.physical_network.graph().num_nodes();
         ++node)
    {
        if (node != 0U)
        {
            result.push_back(',');
        }
        result += numeric_attr_token(
            fixture.physical_network.graph().node_attrs(node).at(
                fixture.physical_cpu_value));
    }
    result += "];slots=[";
    bool first = true;
    for (const auto& entry : solution.node_slots.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(entry.key) + ":" +
            std::to_string(entry.value);
    }
    result += "];info=[";
    first = true;
    for (const auto& entry : solution.node_slots_info.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(entry.key.virtual_node) + ":" +
            std::to_string(entry.key.physical_node) +
            values_payload(entry.value, resources);
    }
    result += "];offsets=[";
    first = true;
    for (const auto& entry :
         solution.v_net_constraint_offsets.node_level.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(entry.key) +
            values_payload(entry.value, constraints);
    }
    result += "];violations=[";
    first = true;
    for (const auto& entry :
         solution.v_net_constraint_violations.node_level.entries())
    {
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(entry.key) +
            values_payload(entry.value, constraints);
    }
    result += "];total=" + double_token(
        solution.v_net_total_hard_constraint_violation);
    result += ";flags=";
    result.push_back(solution.place_result ? '1' : '0');
    result.push_back(',');
    result.push_back(solution.result ? '1' : '0');
    return result;
}

std::string placement_payload(
    const NodePlacementResult& placement,
    const Fixture& fixture,
    const Solution& solution)
{
    return std::string("placed=") + (placement.placed ? "1" : "0") +
        ";feasible=" + (placement.check.feasible ? "1" : "0") +
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

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

std::string run_greedy(std::size_t workers, bool inplace = true)
{
    Fixture fixture;
    PreparedNodeMapper mapper = fixture.prepare();
    Solution solution = make_solution();
    NodeMappingOptions options;
    options.method = NodeMatchingMethod::greedy;
    options.inplace = inplace;
    options.candidate_workers = workers;
    const bool mapped = mapper.node_mapping(
        {0U, 1U}, {0U, 1U, 2U}, solution, options);
    return mapping_payload(mapped, fixture, solution);
}

void differential()
{
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const NodePlacementResult placement = mapper.place(
            0U, 0U, solution);
        emit(
            "safe_success",
            placement_payload(placement, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        const NodePlacementResult placement = mapper.place(
            0U, 1U, solution);
        emit(
            "safe_failure",
            placement_payload(placement, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        NodePlacementOptions options;
        options.allow_constraint_violation = true;
        const NodePlacementResult placement = mapper.place(
            0U, 1U, solution, options);
        emit(
            "unsafe_failure_places",
            placement_payload(placement, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        NodePlacementOptions options;
        options.record_constraint_violation = false;
        const NodePlacementResult placement = mapper.place(
            0U, 0U, solution, options);
        emit(
            "safe_success_no_record",
            placement_payload(placement, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        SolutionAttributeValues offsets;
        offsets.set(fixture.virtual_cpu, std::int64_t{2});
        offsets.set(fixture.virtual_quality, 3.0);
        mapper.record_place_constraint_violation(0U, offsets, solution);
        mapper.record_place_constraint_violation(0U, offsets, solution);
        emit("record_repeated", solution_payload(fixture, solution));
    }
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        NodePlacementOptions options;
        options.record_constraint_violation = false;
        static_cast<void>(mapper.place(0U, 0U, solution, options));
        const bool undone = mapper.undo_place(0U, solution);
        emit(
            "undo_success",
            std::string("undone=") + (undone ? "1;" : "0;") +
                solution_payload(fixture, solution));
    }

    emit("mapping_greedy", run_greedy(1U));

    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        NodeMappingOptions options;
        options.method = NodeMatchingMethod::l2s2;
        const bool mapped = mapper.node_mapping(
            {0U, 1U}, {0U, 1U, 2U}, solution, options);
        emit(
            "mapping_l2s2_failure",
            mapping_payload(mapped, fixture, solution));
    }
    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        NodeMappingOptions options;
        options.reusable = true;
        const bool mapped = mapper.node_mapping(
            {0U, 1U}, {2U}, solution, options);
        emit(
            "mapping_reusable",
            mapping_payload(mapped, fixture, solution));
    }

    emit("mapping_inplace_false", run_greedy(1U, false));

    {
        Fixture fixture;
        PreparedNodeMapper mapper = fixture.prepare();
        Solution solution = make_solution();
        solution.v_net_total_hard_constraint_violation = 2.0;
        NodeMappingOptions options;
        options.method = NodeMatchingMethod::l2s2;
        const bool mapped = mapper.node_mapping(
            {0U, 1U}, {0U, 1U, 2U}, solution, options);
        emit(
            "mapping_failure_preserves_total",
            mapping_payload(mapped, fixture, solution));
    }

    std::string worker_payload;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        if (!worker_payload.empty())
        {
            worker_payload.push_back('|');
        }
        worker_payload += run_greedy(workers);
    }
    emit("mapping_workers_0_1_2_8", worker_payload);
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
            "usage: node_mapper_harness [differential]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "node mapper harness: " << error.what() << '\n';
        return 1;
    }
}
