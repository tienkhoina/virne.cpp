#include "recorder.h"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace core = virne::core;
namespace network = virne::network;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct ScopedDirectory {
    explicit ScopedDirectory(const std::size_t workers) {
        const auto stamp = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("virne_recorder_benchmark_" + std::to_string(stamp) + "_" +
             std::to_string(workers));
    }

    ~ScopedDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    ScopedDirectory(const ScopedDirectory&) = delete;
    ScopedDirectory& operator=(const ScopedDirectory&) = delete;

    std::filesystem::path path;
};

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    const attribute::AttributeOwner owner) {
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

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    const std::string_view name) {
    const auto binding = value.bind_node_attribute(name);
    if (!binding) {
        throw std::runtime_error("Recorder benchmark node binding failed");
    }
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    const std::string_view name) {
    const auto binding = value.bind_link_attribute(name);
    if (!binding) {
        throw std::runtime_error("Recorder benchmark link binding failed");
    }
    return *binding;
}

network::NodeAttributeDataUpdate dense_node_update(
    const attribute::AttributeRegistryId id,
    std::vector<AttrValue> values) {
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate dense_link_update(
    const attribute::AttributeRegistryId id,
    std::vector<AttrValue> values) {
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

std::vector<EdgeEndpoints> chain_edges(const std::size_t item_count) {
    std::vector<EdgeEndpoints> result;
    result.reserve(item_count - 1U);
    for (std::size_t index = 0U; index + 1U < item_count; ++index) {
        result.emplace_back(index, index + 1U);
    }
    return result;
}

network::PhysicalNetwork make_physical_network(const std::size_t item_count) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(item_count, chain_edges(item_count));
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bandwidth", attribute::AttributeOwner::link)};

    network::PhysicalNetwork result(std::move(construction));
    const auto node = require_node_binding(result, "cpu");
    const auto link = require_link_binding(result, "bandwidth");

    std::vector<AttrValue> node_values(item_count);
    std::vector<AttrValue> link_values(item_count - 1U);
    for (std::size_t index = 0U; index < item_count; ++index) {
        node_values[index] =
            static_cast<double>((index * 23U) % 107U) + 10.0;
    }
    for (std::size_t index = 0U; index + 1U < item_count; ++index) {
        link_values[index] =
            static_cast<double>((index * 29U) % 109U) + 5.0;
    }
    result.set_node_attrs_data({
        dense_node_update(node.registry_id, std::move(node_values))});
    result.set_link_attrs_data({
        dense_link_update(link.registry_id, std::move(link_values))});
    return result;
}

network::VirtualNetwork make_virtual_network(const std::size_t item_count) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(item_count, chain_edges(item_count));
    construction.config.node_attribute_specs = {
        resource_spec("cpu", attribute::AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("bandwidth", attribute::AttributeOwner::link)};

    network::VirtualNetwork result(std::move(construction));
    const auto node = require_node_binding(result, "cpu");
    const auto link = require_link_binding(result, "bandwidth");

    std::vector<AttrValue> node_values(item_count);
    std::vector<AttrValue> link_values(item_count - 1U);
    for (std::size_t index = 0U; index < item_count; ++index) {
        node_values[index] =
            static_cast<double>((index * 17U) % 101U) + 1.25;
    }
    for (std::size_t index = 0U; index + 1U < item_count; ++index) {
        link_values[index] =
            static_cast<double>((index * 19U) % 103U) + 0.5;
    }
    result.set_node_attrs_data({
        dense_node_update(node.registry_id, std::move(node_values))});
    result.set_link_attrs_data({
        dense_link_update(link.registry_id, std::move(link_values))});
    result.set_request_id(7);
    result.set_arrival_time(1.25);
    result.set_lifetime(3.0);
    return result;
}

core::Solution make_solution(
    network::VirtualNetwork& virtual_network,
    const std::size_t item_count) {
    core::SolutionMetadata metadata;
    metadata.v_net_id = 7;
    metadata.v_net_lifetime = 3.0;
    metadata.v_net_arrival_time = 1.25;
    metadata.v_net_num_nodes = item_count;
    metadata.v_net_num_edges = item_count - 1U;
    core::Solution result(metadata);
    result.result = true;
    result.description = "benchmark";

    const auto link_binding = require_link_binding(virtual_network, "bandwidth");
    for (std::size_t index = 0U; index < item_count; ++index) {
        const auto node = static_cast<core::SolutionNodeId>(index);
        result.node_slots.insert_or_assign(node, node);
    }
    for (std::size_t index = 0U; index + 1U < item_count; ++index) {
        const auto source = static_cast<core::SolutionNodeId>(index);
        const auto target = static_cast<core::SolutionNodeId>(index + 1U);
        const core::SolutionLink link{source, target};
        result.link_paths.insert_or_assign(link, {link});

        core::SolutionAttributeValues info;
        info.set(
            link_binding.registry_id,
            static_cast<double>((index * 19U) % 103U) + 0.5);
        result.link_paths_info.insert_or_assign({link, link}, std::move(info));
    }
    return result;
}

std::size_t parse_size(const char* text) {
    const std::string value(text);
    std::size_t position = 0U;
    const unsigned long long parsed = std::stoull(value, &position, 10);
    if (position != value.size() ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "Recorder benchmark argument is not a valid size");
    }
    return static_cast<std::size_t>(parsed);
}

template <typename Integer>
void append_integer(std::string& output, const Integer value) {
    static_assert(std::is_integral_v<Integer>);
    char buffer[32];
    const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("Recorder benchmark integer encoding failed");
    }
    output.append(buffer, result.ptr);
}

void append_double(std::string& output, const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr char digits[] = "0123456789abcdef";
    output += "d:";
    for (unsigned shift = 60U;; shift -= 4U) {
        output.push_back(digits[(bits >> shift) & 0xFU]);
        if (shift == 0U) {
            break;
        }
    }
}

void append_bool(std::string& output, const bool value) {
    output.push_back(value ? '1' : '0');
}

void append_optional_size(
    std::string& output,
    const std::optional<std::size_t>& value) {
    if (value) {
        append_integer(output, *value);
    } else {
        output += "none";
    }
}

void append_attribute_number(
    std::string& output,
    const attribute::AttributeNumber& value) {
    std::visit(
        [&output](const auto number) {
            using Number = std::decay_t<decltype(number)>;
            if constexpr (std::is_same_v<Number, double>) {
                append_double(output, number);
            } else {
                output += "i:";
                append_integer(output, static_cast<std::int64_t>(number));
            }
        },
        value);
}

void append_link(std::string& output, const core::SolutionLink link) {
    output.push_back('(');
    append_integer(output, link.source);
    output.push_back(',');
    append_integer(output, link.target);
    output.push_back(')');
}

std::string snapshot_payload(
    const core::Recorder& recorder,
    const core::RecorderRecord& record,
    const std::size_t item_count) {
    if (!record.state.event || recorder.memory().size() != 1U ||
        !record.extra.empty()) {
        throw std::runtime_error("Recorder benchmark snapshot invariant failed");
    }
    const core::Solution& solution = record.solution;
    if (!solution.node_slots_info.empty() ||
        !solution.v_net_single_step_constraint_offset.node_level.empty() ||
        !solution.v_net_single_step_constraint_offset.link_level.empty() ||
        !solution.v_net_single_step_constraint_offset.path_level.empty() ||
        !solution.v_net_constraint_offsets.node_level.empty() ||
        !solution.v_net_constraint_offsets.link_level.empty() ||
        !solution.v_net_constraint_offsets.path_level.empty() ||
        !solution.v_net_constraint_violations.node_level.empty() ||
        !solution.v_net_constraint_violations.link_level.empty() ||
        !solution.v_net_constraint_violations.path_level.empty()) {
        throw std::runtime_error("Recorder benchmark empty-table invariant failed");
    }

    std::string output;
    output.reserve(item_count * 150U);
    const auto& state = record.state;
    output += "state:event=";
    append_integer(output, state.event->event_id);
    output.push_back(',');
    append_integer(output, static_cast<unsigned>(state.event->type));
    output += ";counts=";
    append_integer(output, state.virtual_network_count);
    output.push_back(',');
    append_integer(output, state.success_count);
    output.push_back(',');
    append_integer(output, state.inservice_count);
    output += ";totals=";
    append_double(output, state.total_revenue);
    output.push_back(',');
    append_double(output, state.total_cost);
    output.push_back(',');
    append_double(output, state.total_time_revenue);
    output.push_back(',');
    append_double(output, state.total_time_cost);
    output.push_back(',');
    append_double(output, state.long_term_r2c_ratio);
    output.push_back(',');
    append_double(output, state.long_term_time_r2c_ratio);
    output += ";physical=";
    append_integer(output, state.running_physical_node_count);
    output.push_back(',');
    append_double(output, state.physical_available_resource);
    output.push_back(',');
    append_double(output, state.physical_node_available_resource);
    output.push_back(',');
    append_double(output, state.physical_link_available_resource);
    output.push_back(',');
    append_double(output, state.physical_node_resource_utilization);
    output.push_back(',');
    append_double(output, state.physical_link_resource_utilization);

    output += ";solution:meta=";
    append_integer(output, solution.v_net_id);
    output.push_back(',');
    append_double(output, solution.v_net_lifetime);
    output.push_back(',');
    append_double(output, solution.v_net_arrival_time);
    output.push_back(',');
    append_integer(output, solution.v_net_num_nodes);
    output.push_back(',');
    append_integer(output, solution.v_net_num_edges);
    output += ";flags=";
    append_bool(output, solution.result);
    output.push_back(',');
    append_bool(output, solution.place_result);
    output.push_back(',');
    append_bool(output, solution.route_result);
    output.push_back(',');
    append_bool(output, solution.early_rejection);

    output += ";node_slots=";
    append_integer(output, solution.node_slots.size());
    output.push_back(':');
    for (const auto& entry : solution.node_slots.entries()) {
        append_integer(output, entry.key);
        output.push_back('>');
        append_integer(output, entry.value);
        output.push_back(',');
    }
    output += ";link_paths=";
    append_integer(output, solution.link_paths.size());
    output.push_back(':');
    for (const auto& entry : solution.link_paths.entries()) {
        append_link(output, entry.key);
        output += ">[";
        for (const auto link : entry.value) {
            append_link(output, link);
            output.push_back(',');
        }
        output += "],";
    }
    output += ";node_slots_info=0;link_paths_info=";
    append_integer(output, solution.link_paths_info.size());
    output.push_back(':');
    for (const auto& entry : solution.link_paths_info.entries()) {
        append_link(output, entry.key.virtual_link);
        output.push_back('@');
        append_link(output, entry.key.physical_link);
        output.push_back('{');
        const auto& slots = entry.value.slots();
        for (std::size_t id = 0U; id < slots.size(); ++id) {
            if (!slots[id]) {
                continue;
            }
            append_integer(output, id);
            output.push_back('=');
            append_attribute_number(output, *slots[id]);
            output.push_back(',');
        }
        output += "},";
    }

    output += ";metrics=";
    for (const double value : {
             solution.v_net_cost,
             solution.v_net_revenue,
             solution.v_net_demand,
             solution.v_net_node_demand,
             solution.v_net_link_demand,
             solution.v_net_node_revenue,
             solution.v_net_link_revenue,
             solution.v_net_node_cost,
             solution.v_net_link_cost,
             solution.v_net_path_cost,
             solution.v_net_r2c_ratio,
             solution.v_net_time_cost,
             solution.v_net_time_revenue,
             solution.v_net_time_rc_ratio,
             solution.v_net_total_hard_constraint_violation}) {
        append_double(output, value);
        output.push_back(',');
    }
    output += ";description=";
    append_integer(output, solution.description.size());
    output.push_back(':');
    output += solution.description;
    output += ";constraint_tables=0,0,0,0,0,0;violation_list=";
    append_integer(output, solution.v_net_single_step_violation_list.size());
    output.push_back(':');
    for (const auto& value : solution.v_net_single_step_violation_list) {
        append_attribute_number(output, value);
        output.push_back(',');
    }
    output += ";hard_offsets=";
    append_double(output, solution.v_net_single_step_hard_constraint_offset);
    output.push_back(',');
    append_double(
        output, solution.v_net_max_single_step_hard_constraint_violation);
    output += ";runtime=";
    append_integer(output, solution.revoke_times);
    output.push_back(',');
    append_integer(output, solution.num_interactions);
    output.push_back(',');
    append_double(output, solution.v_net_reward);
    output += ";actions=";
    append_integer(output, solution.selected_actions.size());
    output.push_back(':');
    for (const auto action : solution.selected_actions) {
        append_integer(output, action);
        output.push_back(',');
    }
    output += ";optional=";
    append_optional_size(output, solution.num_placed_nodes);
    output.push_back(',');
    append_optional_size(output, solution.num_routed_links);
    output.push_back(',');
    append_optional_size(output, solution.num_attempt_times);

    const auto running = recorder.running_physical_nodes();
    output += ";extra=0;history=";
    append_integer(output, recorder.memory().size());
    output += ";running=";
    append_integer(output, running.size());
    output.push_back(':');
    for (const auto node : running) {
        append_integer(output, node);
        output.push_back(',');
    }
    return output;
}

std::uint64_t fingerprint(const std::string& value) noexcept {
    std::uint64_t result = fnv_offset;
    for (const char raw_byte : value) {
        result =
            (result ^ static_cast<unsigned char>(raw_byte)) * fnv_prime;
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: recorder_benchmark <item_count> <workers>");
        }
        const std::size_t item_count = parse_size(argv[1]);
        const std::size_t workers = parse_size(argv[2]);
        if (item_count < 128U ||
            item_count > std::numeric_limits<std::size_t>::max() / 4U ||
            item_count > static_cast<std::size_t>(
                std::numeric_limits<core::SolutionNodeId>::max())) {
            throw std::invalid_argument(
                "Recorder benchmark item_count is outside [128, INT64_MAX]");
        }

        ScopedDirectory directory(workers);
        auto physical_network = make_physical_network(item_count);
        auto virtual_network = make_virtual_network(item_count);
        core::Counter counter;
        const core::PreparedCounter virtual_counter =
            counter.prepare(virtual_network);

        core::RecorderConfig config;
        config.save_root_dir = directory.path;
        config.solver_name = "recorder-benchmark";
        config.run_id = "workers-" + std::to_string(workers);
        config.temporary_records = false;
        core::Recorder recorder(std::move(counter), std::move(config));
        recorder.count_initial_physical_network(
            physical_network, core::RecorderOptions{workers});
        recorder.set_event({0, network::VirtualEventType::arrival});
        core::Solution solution = make_solution(virtual_network, item_count);

        const auto started = std::chrono::steady_clock::now();
        const core::RecorderRecord& stored = recorder.add_record(
            recorder.count_prepared(
                virtual_counter,
                physical_network,
                solution,
                core::RecorderOptions{workers}));
        const auto stopped = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - started).count();

        const std::string payload =
            snapshot_payload(recorder, stored, item_count);
        const std::size_t entry_count = 4U * item_count - 1U;
        std::cout
            << "protocol=1\n"
            << "kind=recorder_prepared_arrival_history_snapshot\n"
            << "semantics=exact_recorder_arrival_and_deep_snapshot_v1\n"
            << "item_count=" << item_count << '\n'
            << "workers=" << workers << '\n'
            << "type_tag=recorder_fixed_fields_raw64_ordered_maps_v1\n"
            << "elapsed_ns=" << elapsed << '\n'
            << "checksum=" << fingerprint(payload) << '\n'
            << "output_bytes=" << payload.size() << '\n'
            << "entry_count=" << entry_count << '\n'
            << "history_size=" << recorder.memory().size() << '\n'
            << "running_node_count="
            << recorder.state().running_physical_node_count << '\n'
            << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "recorder_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
