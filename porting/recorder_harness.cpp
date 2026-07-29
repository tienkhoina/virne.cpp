#include "recorder.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

namespace
{

namespace attribute = virne::network::attribute;
namespace core = virne::core;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using core::Counter;
using core::Recorder;
using core::RecorderConfig;
using core::RecorderErrorCode;
using core::RecorderException;
using core::RecorderInitialPhysicalState;
using core::RecorderRecord;
using core::RecorderState;
using core::Solution;
using core::SolutionAttributeValues;
using core::SolutionLink;
using core::SolutionMetadata;
using core::SolutionNodeId;

struct ScopedDirectory
{
    explicit ScopedDirectory(const std::string_view label)
    {
        static std::uint64_t sequence = 0U;
        const auto stamp = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("virne_recorder_diff_" + std::string(label) + "_" +
             std::to_string(stamp) + "_" +
             std::to_string(sequence++));
    }

    ~ScopedDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    ScopedDirectory(const ScopedDirectory&) = delete;
    ScopedDirectory& operator=(const ScopedDirectory&) = delete;

    std::filesystem::path path;
};

RecorderConfig recorder_config(
    const std::filesystem::path& root,
    std::string run_id)
{
    RecorderConfig result;
    result.save_root_dir = root;
    result.solver_name = "recorder-differential";
    result.run_id = std::move(run_id);
    result.temporary_records = false;
    return result;
}

AttributeFactorySpec resource_spec(
    std::string name,
    const AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = owner == AttributeOwner::node
        ? CheckingLevel::node
        : CheckingLevel::link;
    return result;
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    const std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    if (!binding.has_value())
    {
        throw std::runtime_error("missing Recorder node binding");
    }
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    const std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding.has_value())
    {
        throw std::runtime_error("missing Recorder link binding");
    }
    return *binding;
}

network::NodeAttributeDataUpdate dense_node_update(
    const attribute::AttributeRegistryId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    const attribute::AttributeRegistryId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bandwidth", AttributeOwner::link)};

    network::PhysicalNetwork result(std::move(construction));
    const auto node = require_node_binding(result, "cpu");
    const auto link = require_link_binding(result, "bandwidth");
    result.set_node_attrs_data({dense_node_update(
        node.registry_id, {10.0, 20.0, 30.0})});
    result.set_link_attrs_data({sparse_link_update(
        link.registry_id, {{0U, 1U, 5.0}, {1U, 2U, 7.0}})});
    return result;
}

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {
        resource_spec("cpu", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bandwidth", AttributeOwner::link)};

    network::VirtualNetwork result(std::move(construction));
    const auto node = require_node_binding(result, "cpu");
    const auto link = require_link_binding(result, "bandwidth");
    result.set_node_attrs_data({dense_node_update(
        node.registry_id, {2.0, 4.0})});
    result.set_link_attrs_data({sparse_link_update(
        link.registry_id, {{0U, 1U, 3.0}})});
    result.set_lifetime(1.0);
    return result;
}

void seed_preserved_solution_fields(Solution& value)
{
    value.v_net_total_hard_constraint_violation = 0.25;
    value.v_net_single_step_hard_constraint_offset = -0.5;
    value.v_net_max_single_step_hard_constraint_violation = 2.25;
    value.v_net_single_step_violation_list = {
        true, std::int64_t{-2}, 1.5};
    value.revoke_times = 2;
    value.selected_actions = {4, 5};
    value.num_interactions = 3;
    value.num_attempt_times = 7U;

    value.v_net_single_step_constraint_offset.node_level.set(0U, 1.25);
    SolutionAttributeValues node_offset;
    node_offset.set(0U, -0.25);
    value.v_net_constraint_offsets.node_level.insert_or_assign(
        0, std::move(node_offset));
    SolutionAttributeValues link_violation;
    link_violation.set(0U, 0.5);
    value.v_net_constraint_violations.link_level.insert_or_assign(
        SolutionLink{0, 1}, std::move(link_violation));
}

Solution make_success_solution(
    network::VirtualNetwork& virtual_network,
    const SolutionNodeId virtual_network_id,
    const SolutionNodeId first_physical_node,
    const SolutionNodeId second_physical_node,
    const bool empty_route = false)
{
    SolutionMetadata metadata;
    metadata.v_net_id = virtual_network_id;
    metadata.v_net_lifetime = 1.0;
    metadata.v_net_arrival_time =
        static_cast<double>(virtual_network_id + 1);
    metadata.v_net_num_nodes = 2U;
    metadata.v_net_num_edges = 1U;
    Solution result(metadata);
    result.result = true;
    result.description = "original";
    result.v_net_reward = static_cast<double>(virtual_network_id);
    result.node_slots.insert_or_assign(0, first_physical_node);
    result.node_slots.insert_or_assign(1, second_physical_node);
    seed_preserved_solution_fields(result);

    const SolutionLink virtual_link{0, 1};
    if (empty_route)
    {
        result.link_paths.insert_or_assign(virtual_link, {});
        return result;
    }

    const std::vector<SolutionLink> path{{0, 1}, {1, 2}};
    result.link_paths.insert_or_assign(virtual_link, path);
    const auto binding =
        require_link_binding(virtual_network, "bandwidth");
    SolutionAttributeValues route_info;
    route_info.set(binding.registry_id, 3.0);
    for (const SolutionLink physical_link : path)
    {
        result.link_paths_info.insert_or_assign(
            {virtual_link, physical_link}, route_info);
    }
    return result;
}

Solution make_failure_solution(const SolutionNodeId virtual_network_id)
{
    SolutionMetadata metadata;
    metadata.v_net_id = virtual_network_id;
    metadata.v_net_lifetime = 1.0;
    metadata.v_net_arrival_time =
        static_cast<double>(virtual_network_id + 1);
    metadata.v_net_num_nodes = 2U;
    metadata.v_net_num_edges = 1U;
    Solution result(metadata);
    result.result = false;
    result.v_net_reward = static_cast<double>(virtual_network_id);
    seed_preserved_solution_fields(result);
    return result;
}

std::string double_token(const double value)
{
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0')
           << std::setw(16) << bits;
    return stream.str();
}

std::string hex_text(const std::string_view value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char character : value)
    {
        const auto byte = static_cast<unsigned char>(character);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string number_token(const attribute::AttributeNumber& value)
{
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return std::string("b:") + (*boolean ? "1" : "0");
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return "i:" + std::to_string(*integer);
    }
    return double_token(std::get<double>(value));
}

std::string attribute_values_payload(const SolutionAttributeValues& values)
{
    std::string result = "[";
    bool first = true;
    const auto& slots = values.slots();
    for (std::size_t id = 0U; id < slots.size(); ++id)
    {
        if (!slots[id].has_value())
        {
            continue;
        }
        if (!first)
        {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(id) + "=" + number_token(*slots[id]);
    }
    result.push_back(']');
    return result;
}

std::string link_token(const SolutionLink& link)
{
    return std::to_string(link.source) + ":" +
        std::to_string(link.target);
}

std::string node_slots_payload(const core::NodeSlots& slots)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < slots.entries().size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const auto& entry = slots.entries()[index];
        result += std::to_string(entry.key) + "=" +
            std::to_string(entry.value);
    }
    result.push_back(']');
    return result;
}

std::string link_paths_payload(const core::LinkPaths& paths)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < paths.entries().size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const auto& entry = paths.entries()[index];
        result += link_token(entry.key) + "=[";
        for (std::size_t path = 0U; path < entry.value.size(); ++path)
        {
            if (path != 0U)
            {
                result.push_back(',');
            }
            result += link_token(entry.value[path]);
        }
        result.push_back(']');
    }
    result.push_back(']');
    return result;
}

std::string link_info_payload(const core::LinkPathsInfo& info)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < info.entries().size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const auto& entry = info.entries()[index];
        result += link_token(entry.key.virtual_link) + "/" +
            link_token(entry.key.physical_link) + "=" +
            attribute_values_payload(entry.value);
    }
    result.push_back(']');
    return result;
}

template <typename Table, typename KeyFormatter>
std::string table_payload(
    const Table& table,
    KeyFormatter&& key_formatter)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < table.entries().size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        const auto& entry = table.entries()[index];
        result += key_formatter(entry.key) + "=" +
            attribute_values_payload(entry.value);
    }
    result.push_back(']');
    return result;
}

std::string optional_size_token(const std::optional<std::size_t>& value)
{
    return value.has_value() ? "i:" + std::to_string(*value) : "n";
}

std::string solution_payload(const Solution& value)
{
    std::ostringstream stream;
    stream << "meta=" << value.v_net_id << ','
           << double_token(value.v_net_lifetime) << ','
           << double_token(value.v_net_arrival_time) << ','
           << value.v_net_num_nodes << ',' << value.v_net_num_edges
           << ";result=" << (value.result ? '1' : '0')
           << ";slots=" << node_slots_payload(value.node_slots)
           << ";paths=" << link_paths_payload(value.link_paths)
           << ";node_info=" << value.node_slots_info.size()
           << ";link_info=" << link_info_payload(value.link_paths_info)
           << ";metrics="
           << double_token(value.v_net_cost) << ','
           << double_token(value.v_net_revenue) << ','
           << double_token(value.v_net_demand) << ','
           << double_token(value.v_net_node_demand) << ','
           << double_token(value.v_net_link_demand) << ','
           << double_token(value.v_net_node_revenue) << ','
           << double_token(value.v_net_link_revenue) << ','
           << double_token(value.v_net_node_cost) << ','
           << double_token(value.v_net_link_cost) << ','
           << double_token(value.v_net_path_cost) << ','
           << double_token(value.v_net_r2c_ratio) << ','
           << double_token(value.v_net_time_cost) << ','
           << double_token(value.v_net_time_revenue) << ','
           << double_token(value.v_net_time_rc_ratio)
           << ";description=" << hex_text(value.description)
           << ";hard="
           << double_token(value.v_net_total_hard_constraint_violation)
           << ";step="
           << attribute_values_payload(
                  value.v_net_single_step_constraint_offset.node_level)
           << '/'
           << attribute_values_payload(
                  value.v_net_single_step_constraint_offset.link_level)
           << '/'
           << attribute_values_payload(
                  value.v_net_single_step_constraint_offset.path_level)
           << ";offsets="
           << table_payload(
                  value.v_net_constraint_offsets.node_level,
                  [](const SolutionNodeId node)
                  {
                      return std::to_string(node);
                  })
           << '/'
           << table_payload(
                  value.v_net_constraint_offsets.link_level,
                  [](const SolutionLink& link)
                  {
                      return link_token(link);
                  })
           << '/'
           << table_payload(
                  value.v_net_constraint_offsets.path_level,
                  [](const SolutionLink& link)
                  {
                      return link_token(link);
                  })
           << ";violations="
           << table_payload(
                  value.v_net_constraint_violations.node_level,
                  [](const SolutionNodeId node)
                  {
                      return std::to_string(node);
                  })
           << '/'
           << table_payload(
                  value.v_net_constraint_violations.link_level,
                  [](const SolutionLink& link)
                  {
                      return link_token(link);
                  })
           << '/'
           << table_payload(
                  value.v_net_constraint_violations.path_level,
                  [](const SolutionLink& link)
                  {
                      return link_token(link);
                  })
           << ";violation_list=[";
    for (std::size_t index = 0U;
         index < value.v_net_single_step_violation_list.size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        stream << number_token(value.v_net_single_step_violation_list[index]);
    }
    stream << "]"
           << ";single="
           << double_token(value.v_net_single_step_hard_constraint_offset)
           << ','
           << double_token(
                  value.v_net_max_single_step_hard_constraint_violation)
           << ";flags=" << (value.place_result ? '1' : '0')
           << (value.route_result ? '1' : '0')
           << (value.early_rejection ? '1' : '0')
           << ";revoke=" << value.revoke_times
           << ";actions=[";
    for (std::size_t index = 0U;
         index < value.selected_actions.size(); ++index)
    {
        if (index != 0U)
        {
            stream << ',';
        }
        stream << value.selected_actions[index];
    }
    stream << "]"
           << ";interactions=" << value.num_interactions
           << ";reward=" << double_token(value.v_net_reward)
           << ";counts="
           << optional_size_token(value.num_placed_nodes) << ','
           << optional_size_token(value.num_routed_links) << ','
           << optional_size_token(value.num_attempt_times);
    return stream.str();
}

std::string state_payload(const RecorderState& value)
{
    std::ostringstream stream;
    stream << "event=";
    if (value.event.has_value())
    {
        stream << value.event->event_id << ':'
               << static_cast<unsigned int>(value.event->type);
    }
    else
    {
        stream << 'n';
    }
    stream << ";counts=" << value.virtual_network_count << ','
           << value.success_count << ',' << value.inservice_count
           << ";totals=" << double_token(value.total_revenue) << ','
           << double_token(value.total_cost) << ','
           << double_token(value.total_time_revenue) << ','
           << double_token(value.total_time_cost)
           << ";ratios=" << double_token(value.long_term_r2c_ratio)
           << ',' << double_token(value.long_term_time_r2c_ratio)
           << ";running=" << value.running_physical_node_count
           << ";physical="
           << double_token(value.physical_available_resource) << ','
           << double_token(value.physical_node_available_resource) << ','
           << double_token(value.physical_link_available_resource)
           << ";utilization="
           << double_token(value.physical_node_resource_utilization)
           << ','
           << double_token(value.physical_link_resource_utilization);
    return stream.str();
}

std::string initial_payload(const RecorderInitialPhysicalState& value)
{
    return double_token(value.available_resource) + "," +
        double_token(value.node_available_resource) + "," +
        double_token(value.link_available_resource);
}

std::string nodes_payload(const std::vector<SolutionNodeId>& nodes)
{
    std::string result = "[";
    for (std::size_t index = 0U; index < nodes.size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back(',');
        }
        result += std::to_string(nodes[index]);
    }
    result.push_back(']');
    return result;
}

std::string record_payload(const RecorderRecord& record)
{
    return "{" + state_payload(record.state) + "|" +
        solution_payload(record.solution) + "}";
}

std::string flow_payload(const std::size_t workers)
{
    ScopedDirectory directory("flow");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    Recorder recorder(
        Counter{},
        recorder_config(directory.path, "flow-" +
            std::to_string(workers)));
    recorder.count_initial_physical_network(
        physical_network, {workers});

    std::vector<std::string> stages;
    const auto append_stage = [&stages, &recorder](
        const RecorderRecord& record)
    {
        stages.push_back(
            record_payload(record) + "@" +
            nodes_payload(recorder.running_physical_nodes()));
    };

    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution first = make_success_solution(virtual_network, 10, 2, 2);
    append_stage(recorder.add_record(recorder.count(
        virtual_network, physical_network, first, {workers})));

    recorder.set_event({1, network::VirtualEventType::arrival});
    Solution failure = make_failure_solution(20);
    append_stage(recorder.add_record(recorder.count(
        virtual_network, physical_network, failure, {workers})));

    recorder.set_event({2, network::VirtualEventType::arrival});
    Solution second = make_success_solution(
        virtual_network, 30, 1, 2);
    append_stage(recorder.add_record(recorder.count(
        virtual_network, physical_network, second, {workers})));

    const std::string lookup =
        std::to_string(recorder.record_by_event(-1).solution.v_net_id) +
        "," +
        std::to_string(recorder.record_by_event(-3).solution.v_net_id) +
        "," +
        (recorder.record_by_virtual_network(20).solution.result ? "1" : "0");

    recorder.set_event({3, network::VirtualEventType::leave});
    Solution first_leave = make_success_solution(
        virtual_network, 10, 2, 2);
    first_leave.result = false;
    append_stage(recorder.add_record(recorder.count(
        virtual_network, physical_network, first_leave, {workers})));

    recorder.set_event({4, network::VirtualEventType::leave});
    Solution failed_leave = make_failure_solution(20);
    failed_leave.result = true;
    failed_leave.node_slots.insert_or_assign(0, 99);
    append_stage(recorder.add_record(recorder.count(
        virtual_network, physical_network, failed_leave, {workers})));

    recorder.set_event({5, network::VirtualEventType::leave});
    Solution second_leave = make_success_solution(
        virtual_network, 30, 1, 2);
    second_leave.result = false;
    append_stage(recorder.add_record(recorder.count(
        virtual_network, physical_network, second_leave, {workers})));

    std::string result = "initial=" + initial_payload(
        *recorder.initial_physical_state()) + ";stages=[";
    for (std::size_t index = 0U; index < stages.size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back('#');
        }
        result += stages[index];
    }
    result += "];lookup=" + lookup + ";history=[";
    for (std::size_t index = 0U; index < recorder.memory().size(); ++index)
    {
        if (index != 0U)
        {
            result.push_back('#');
        }
        result += record_payload(recorder.memory()[index]);
    }
    result += "];final={" + state_payload(recorder.state()) + "@" +
        nodes_payload(recorder.running_physical_nodes()) + "}";
    return result;
}

std::string ratio_error_payload(const std::size_t workers)
{
    ScopedDirectory directory("ratio");
    auto physical_network = make_physical_network();
    auto virtual_network = make_virtual_network();
    Recorder recorder(
        Counter{},
        recorder_config(directory.path, "ratio-" +
            std::to_string(workers)));
    recorder.count_initial_physical_network(
        physical_network, {workers});
    recorder.set_event({0, network::VirtualEventType::arrival});
    Solution solution = make_success_solution(
        virtual_network, 53, 0, 1, true);

    std::string error = "none";
    try
    {
        static_cast<void>(recorder.count(
            virtual_network, physical_network, solution, {workers}));
    }
    catch (const RecorderException& exception)
    {
        error = exception.code() == RecorderErrorCode::invalid_time_ratio
            ? "invalid_time_ratio"
            : "other_recorder_error";
    }

    std::string lookup = "record";
    try
    {
        static_cast<void>(recorder.record_by_virtual_network(53));
    }
    catch (const RecorderException& exception)
    {
        lookup = exception.code() ==
                RecorderErrorCode::record_index_out_of_range
            ? "mapped_without_record"
            : "unmapped";
    }

    return "error=" + error + ";initial=" +
        initial_payload(*recorder.initial_physical_state()) +
        ";state={" + state_payload(recorder.state()) + "}" +
        ";solution={" + solution_payload(solution) + "}" +
        ";nodes=" + nodes_payload(recorder.running_physical_nodes()) +
        ";memory=" + std::to_string(recorder.memory().size()) +
        ";lookup=" + lookup;
}

void emit_differential()
{
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        std::cout << "flow_workers_" << workers << '\t'
                  << flow_payload(workers) << '\n';
    }
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        std::cout << "ratio_error_workers_" << workers << '\t'
                  << ratio_error_payload(workers) << '\n';
    }
}

} // namespace

int main(const int argc, char** argv)
{
    try
    {
        if (argc != 2 || std::string_view(argv[1]) != "differential")
        {
            std::cerr << "usage: recorder_harness differential\n";
            return 2;
        }
        emit_differential();
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Recorder harness failure: "
                  << exception.what() << '\n';
        return 1;
    }
}
