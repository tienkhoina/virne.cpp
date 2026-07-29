#include "solution.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace core = virne::core;
namespace network = virne::network;
namespace attribute = virne::network::attribute;

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw : value) {
        const auto byte = static_cast<unsigned char>(raw);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

std::string number_token(const attribute::AttributeNumber& value) {
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return std::string("b:") + (*boolean ? "1" : "0");
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return "i:" + std::to_string(*integer);
    }
    return double_token(std::get<double>(value));
}

std::string optional_size_token(const std::optional<std::size_t>& value) {
    return value.has_value() ? "i:" + std::to_string(*value) : "n";
}

std::string attribute_values_payload(
    const core::SolutionAttributeValues& values) {
    std::string result = "[";
    bool first = true;
    const auto& slots = values.slots();
    for (std::size_t id = 0U; id < slots.size(); ++id) {
        if (!slots[id].has_value()) {
            continue;
        }
        if (!first) {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(id) + "=" + number_token(*slots[id]);
    }
    result.push_back(']');
    return result;
}

std::string link_token(const core::SolutionLink& link) {
    return std::to_string(link.source) + ":" + std::to_string(link.target);
}

std::string node_slots_payload(const core::NodeSlots& slots) {
    std::string result = "[";
    for (std::size_t index = 0U; index < slots.entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& entry = slots.entries()[index];
        result += std::to_string(entry.key) + "=" +
            std::to_string(entry.value);
    }
    result.push_back(']');
    return result;
}

std::string link_paths_payload(const core::LinkPaths& paths) {
    std::string result = "[";
    for (std::size_t index = 0U; index < paths.entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& entry = paths.entries()[index];
        result += link_token(entry.key) + "=[";
        for (std::size_t path_index = 0U;
             path_index < entry.value.size(); ++path_index) {
            if (path_index != 0U) {
                result.push_back(',');
            }
            result += link_token(entry.value[path_index]);
        }
        result.push_back(']');
    }
    result.push_back(']');
    return result;
}

std::string node_info_payload(const core::NodeSlotsInfo& info) {
    std::string result = "[";
    for (std::size_t index = 0U; index < info.entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& entry = info.entries()[index];
        result += std::to_string(entry.key.virtual_node) + ":" +
            std::to_string(entry.key.physical_node) + "=" +
            attribute_values_payload(entry.value);
    }
    result.push_back(']');
    return result;
}

std::string link_info_payload(const core::LinkPathsInfo& info) {
    std::string result = "[";
    for (std::size_t index = 0U; index < info.entries().size(); ++index) {
        if (index != 0U) {
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
std::string constraint_table_payload(
    const Table& table,
    KeyFormatter&& key_formatter) {
    std::string result = "[";
    for (std::size_t index = 0U; index < table.entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& entry = table.entries()[index];
        result += key_formatter(entry.key) + "=" +
            attribute_values_payload(entry.value);
    }
    result.push_back(']');
    return result;
}

std::string snapshot(const core::Solution& solution) {
    std::string result;
    result += "id=i:" + std::to_string(solution.v_net_id);
    result += ";life=" + double_token(solution.v_net_lifetime);
    result += ";arrival=" + double_token(solution.v_net_arrival_time);
    result += ";nodes=i:" + std::to_string(solution.v_net_num_nodes);
    result += ";egdes=i:" + std::to_string(solution.v_net_num_edges);
    result += std::string(";result=b:") + (solution.result ? "1" : "0");
    result += ";node_slots=" + node_slots_payload(solution.node_slots);
    result += ";link_paths=" + link_paths_payload(solution.link_paths);
    result += ";node_info=" + node_info_payload(solution.node_slots_info);
    result += ";link_info=" + link_info_payload(solution.link_paths_info);

    const double metrics[] = {
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
    };
    result += ";metrics=[";
    for (std::size_t index = 0U; index < std::size(metrics); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += double_token(metrics[index]);
    }
    result += "]";
    result += ";description=s:" + hex_text(solution.description);
    result += ";total=" +
        double_token(solution.v_net_total_hard_constraint_violation);
    result += ";step=" +
        attribute_values_payload(
            solution.v_net_single_step_constraint_offset.node_level) + "/" +
        attribute_values_payload(
            solution.v_net_single_step_constraint_offset.link_level) + "/" +
        attribute_values_payload(
            solution.v_net_single_step_constraint_offset.path_level);
    result += ";offsets=" +
        constraint_table_payload(
            solution.v_net_constraint_offsets.node_level,
            [](const core::SolutionNodeId node) { return std::to_string(node); }) +
        "/" +
        constraint_table_payload(
            solution.v_net_constraint_offsets.link_level,
            [](const core::SolutionLink& link) { return link_token(link); }) +
        "/" +
        constraint_table_payload(
            solution.v_net_constraint_offsets.path_level,
            [](const core::SolutionLink& link) { return link_token(link); });
    result += ";violations=" +
        constraint_table_payload(
            solution.v_net_constraint_violations.node_level,
            [](const core::SolutionNodeId node) { return std::to_string(node); }) +
        "/" +
        constraint_table_payload(
            solution.v_net_constraint_violations.link_level,
            [](const core::SolutionLink& link) { return link_token(link); }) +
        "/" +
        constraint_table_payload(
            solution.v_net_constraint_violations.path_level,
            [](const core::SolutionLink& link) { return link_token(link); });
    result += ";step_list=[";
    for (std::size_t index = 0U;
         index < solution.v_net_single_step_violation_list.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += number_token(solution.v_net_single_step_violation_list[index]);
    }
    result += "]";
    result += ";step_hard=" +
        double_token(solution.v_net_single_step_hard_constraint_offset);
    result += ";max_hard=" +
        double_token(solution.v_net_max_single_step_hard_constraint_violation);
    result += std::string(";place=b:") +
        (solution.place_result ? "1" : "0");
    result += std::string(";route=b:") +
        (solution.route_result ? "1" : "0");
    result += std::string(";early=b:") +
        (solution.early_rejection ? "1" : "0");
    result += ";revoke=i:" + std::to_string(solution.revoke_times);
    result += ";actions=[";
    for (std::size_t index = 0U; index < solution.selected_actions.size();
         ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += "i:" + std::to_string(solution.selected_actions[index]);
    }
    result += "]";
    result += ";interactions=i:" + std::to_string(solution.num_interactions);
    result += ";reward=" + double_token(solution.v_net_reward);
    result += ";later=" + optional_size_token(solution.num_placed_nodes) +
        "," + optional_size_token(solution.num_routed_links) +
        "," + optional_size_token(solution.num_attempt_times);
    return result;
}

void emit(const std::string_view name, const std::string& payload) {
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

core::SolutionMetadata base_metadata() {
    return core::SolutionMetadata{7, 3.5, -0.0, 4U, 3U};
}

std::string missing_field_payload(const core::SolutionNetworkField expected) {
    network::VirtualNetwork virtual_network;
    if (expected != core::SolutionNetworkField::id) {
        virtual_network.set_request_id(7);
    }
    if (expected == core::SolutionNetworkField::arrival_time) {
        virtual_network.set_lifetime(3.5);
    }
    try {
        static_cast<void>(core::Solution::from_v_net(virtual_network));
    } catch (const core::SolutionException& error) {
        switch (error.field()) {
        case core::SolutionNetworkField::id:
            return "field=id";
        case core::SolutionNetworkField::lifetime:
            return "field=lifetime";
        case core::SolutionNetworkField::arrival_time:
            return "field=arrival_time";
        }
    }
    return "field=none";
}

std::string batch_payload(const std::vector<core::Solution>& solutions) {
    std::string result = "[";
    for (std::size_t index = 0U; index < solutions.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& solution = solutions[index];
        result += std::to_string(solution.v_net_id) + ":" +
            double_token(solution.v_net_lifetime) + ":" +
            double_token(solution.v_net_arrival_time) + ":" +
            std::to_string(solution.v_net_num_nodes) + ":" +
            std::to_string(solution.v_net_num_edges) + ":" +
            (solution.is_feasible() ? "1" : "0");
    }
    result.push_back(']');
    return result;
}

void emit_cases() {
    core::Solution ordinary(base_metadata());
    emit("construction", snapshot(ordinary));
    emit("repr", ordinary.repr());

    for (const auto& [name, result, violation] :
         std::vector<std::tuple<std::string, bool, double>>{
             {"feasible_false", false, 0.0},
             {"feasible_zero", true, 0.0},
             {"feasible_negative", true, -1.0},
             {"feasible_positive", true, 0.25},
             {"feasible_nan", true,
              std::numeric_limits<double>::quiet_NaN()},
         }) {
        ordinary.result = result;
        ordinary.v_net_total_hard_constraint_violation = violation;
        emit(name, ordinary.is_feasible() ? "b:1" : "b:0");
    }
    ordinary.reset();

    ordinary.node_slots.insert_or_assign(9, 90);
    ordinary.node_slots.insert_or_assign(2, 20);
    ordinary.node_slots.insert_or_assign(9, 91);
    ordinary.link_paths.insert_or_assign(
        {0, 1}, {{91, 12}, {12, 20}});
    core::SolutionAttributeValues node_resources;
    node_resources.set(0U, std::int64_t{7});
    node_resources.set(2U, 2.5);
    ordinary.node_slots_info.insert_or_assign({9, 91}, node_resources);
    core::SolutionAttributeValues link_resources;
    link_resources.set(1U, true);
    ordinary.link_paths_info.insert_or_assign(
        {{0, 1}, {91, 12}}, link_resources);
    ordinary.v_net_single_step_constraint_offset.node_level.set(0U, -1.0);
    ordinary.v_net_single_step_constraint_offset.link_level.set(2U, 3.0);
    ordinary.v_net_constraint_offsets.node_level.insert_or_assign(
        9, node_resources);
    ordinary.v_net_constraint_offsets.link_level.insert_or_assign(
        {0, 1}, link_resources);
    ordinary.v_net_constraint_offsets.path_level.insert_or_assign(
        {0, 1}, node_resources);
    ordinary.v_net_constraint_violations.node_level.insert_or_assign(
        9, link_resources);
    ordinary.v_net_constraint_violations.link_level.insert_or_assign(
        {0, 1}, node_resources);
    ordinary.v_net_constraint_violations.path_level.insert_or_assign(
        {0, 1}, link_resources);
    ordinary.v_net_single_step_violation_list = {
        std::int64_t{-1}, 0.5, true};
    ordinary.selected_actions = {4, 8};
    emit("typed_mappings", snapshot(ordinary));

    ordinary.result = true;
    ordinary.v_net_cost = 9.5;
    ordinary.description = "dirty";
    ordinary.place_result = false;
    ordinary.num_placed_nodes = 2U;
    ordinary.num_routed_links = 1U;
    ordinary.num_attempt_times = 5U;
    ordinary.reset();
    emit("reset_dirty", snapshot(ordinary));

    ordinary.result = true;
    ordinary.v_net_cost = 11.25;
    ordinary.description = "typed update";
    ordinary.early_rejection = true;
    emit("typed_update", snapshot(ordinary));

    emit("missing_id", missing_field_payload(core::SolutionNetworkField::id));
    emit(
        "missing_lifetime",
        missing_field_payload(core::SolutionNetworkField::lifetime));
    emit(
        "missing_arrival",
        missing_field_payload(core::SolutionNetworkField::arrival_time));

    std::vector<core::SolutionMetadata> metadata;
    metadata.reserve(64U);
    for (std::size_t index = 0U; index < 64U; ++index) {
        metadata.push_back({
            static_cast<std::int64_t>(index),
            1.0 + static_cast<double>(index),
            static_cast<double>(index) * 0.25,
            2U + index,
            1U + index,
        });
    }
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        emit(
            "batch_w" + std::to_string(workers),
            batch_payload(core::Solution::from_metadata_batch(metadata, workers)));
    }
}

void checksum_byte(std::uint64_t& checksum, const std::uint8_t value) {
    checksum ^= value;
    checksum *= UINT64_C(1099511628211);
}

void checksum_u64(std::uint64_t& checksum, std::uint64_t value) {
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
        checksum_byte(checksum, static_cast<std::uint8_t>(value & UINT64_C(0xff)));
        value >>= 8U;
    }
}

std::uint64_t double_bits(const double value) {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::size_t parse_size(const char* value, const std::string_view name) {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != std::strlen(value)) {
        throw std::invalid_argument("invalid " + std::string(name));
    }
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(parsed);
}

void emit_benchmark(const std::size_t count, const std::size_t workers) {
    std::vector<core::SolutionMetadata> metadata;
    metadata.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        metadata.push_back({
            static_cast<std::int64_t>(index),
            1.0 + static_cast<double>(index % 17U) * 0.25,
            static_cast<double>(index) * 0.125,
            2U + index % 31U,
            1U + index % 37U,
        });
    }

    const auto started = std::chrono::steady_clock::now();
    auto solutions = core::Solution::from_metadata_batch(metadata, workers);
    for (std::size_t index = 0U; index < solutions.size(); ++index) {
        auto& solution = solutions[index];
        solution.result = true;
        solution.v_net_cost = static_cast<double>(index) * 0.5;
        solution.description = "dirty";
        solution.node_slots.insert_or_assign(
            static_cast<std::int64_t>(index % 7U),
            static_cast<std::int64_t>(index % 11U));
        solution.num_placed_nodes = index % 13U;
    }
    core::Solution::reset_batch(solutions, workers);
    std::size_t feasible_count = 0U;
    for (std::size_t index = 0U; index < solutions.size(); ++index) {
        auto& solution = solutions[index];
        solution.result = index % 3U != 0U;
        solution.v_net_total_hard_constraint_violation =
            index % 5U == 0U ? 0.5 : -0.25;
        feasible_count += solution.is_feasible() ? 1U : 0U;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();

    std::uint64_t checksum = UINT64_C(1469598103934665603);
    for (const auto& solution : solutions) {
        checksum_u64(checksum, static_cast<std::uint64_t>(solution.v_net_id));
        checksum_u64(checksum, double_bits(solution.v_net_lifetime));
        checksum_u64(checksum, double_bits(solution.v_net_arrival_time));
        checksum_u64(checksum, solution.v_net_num_nodes);
        checksum_u64(checksum, solution.v_net_num_edges);
        checksum_u64(checksum, solution.result ? 1U : 0U);
        checksum_u64(
            checksum,
            double_bits(solution.v_net_total_hard_constraint_violation));
        checksum_u64(checksum, solution.num_placed_nodes.value_or(0U));
    }
    std::cout << "elapsed_ns=" << elapsed
              << ";entry_count=" << solutions.size()
              << ";output_bytes=" << solutions.size() * 64U
              << ";feasible_count=" << feasible_count
              << ";checksum=" << checksum << '\n';
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc == 4 && std::string_view(argv[1]) == "benchmark") {
            emit_benchmark(
                parse_size(argv[2], "count"),
                parse_size(argv[3], "workers"));
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: vne_solution_harness [benchmark count workers]");
        }
        emit_cases();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Solution harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
