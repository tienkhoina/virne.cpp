#include "solution.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace core = virne::core;
namespace network = virne::network;
namespace attribute = virne::network::attribute;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

core::SolutionMetadata metadata(const std::size_t index = 0U) {
    return core::SolutionMetadata{
        static_cast<std::int64_t>(100U + index),
        3.5 + static_cast<double>(index),
        1.25 + static_cast<double>(index),
        4U + index,
        3U + index,
    };
}

void expect_reset_state(
    const core::Solution& solution,
    const core::SolutionMetadata& expected,
    const std::string& context) {
    expect(solution.v_net_id == expected.v_net_id, context + ": id");
    expect(solution.v_net_lifetime == expected.v_net_lifetime,
           context + ": lifetime");
    expect(solution.v_net_arrival_time == expected.v_net_arrival_time,
           context + ": arrival");
    expect(solution.v_net_num_nodes == expected.v_net_num_nodes,
           context + ": node count");
    expect(solution.v_net_num_edges == expected.v_net_num_edges,
           context + ": edge count");
    expect(!solution.result, context + ": result");
    expect(solution.node_slots.empty(), context + ": node slots");
    expect(solution.link_paths.empty(), context + ": link paths");
    expect(solution.node_slots_info.empty(), context + ": node info");
    expect(solution.link_paths_info.empty(), context + ": link info");
    expect(solution.v_net_cost == 0.0, context + ": cost");
    expect(solution.v_net_revenue == 0.0, context + ": revenue");
    expect(solution.v_net_demand == 0.0, context + ": demand");
    expect(solution.v_net_node_demand == 0.0, context + ": node demand");
    expect(solution.v_net_link_demand == 0.0, context + ": link demand");
    expect(solution.v_net_node_revenue == 0.0,
           context + ": node revenue");
    expect(solution.v_net_link_revenue == 0.0,
           context + ": link revenue");
    expect(solution.v_net_node_cost == 0.0, context + ": node cost");
    expect(solution.v_net_link_cost == 0.0, context + ": link cost");
    expect(solution.v_net_path_cost == 0.0, context + ": path cost");
    expect(solution.v_net_r2c_ratio == 0.0, context + ": r2c");
    expect(solution.v_net_time_cost == 0.0, context + ": time cost");
    expect(solution.v_net_time_revenue == 0.0,
           context + ": time revenue");
    expect(solution.v_net_time_rc_ratio == 0.0,
           context + ": time ratio");
    expect(solution.description.empty(), context + ": description");
    expect(solution.v_net_total_hard_constraint_violation == 0.0,
           context + ": total violation");
    expect(solution.v_net_single_step_constraint_offset.node_level.empty(),
           context + ": step node constraints");
    expect(solution.v_net_constraint_offsets.node_level.empty(),
           context + ": node constraints");
    expect(solution.v_net_constraint_violations.path_level.empty(),
           context + ": path violations");
    expect(solution.v_net_single_step_violation_list.empty(),
           context + ": violation list");
    expect(std::isinf(solution.v_net_single_step_hard_constraint_offset) &&
               std::signbit(
                   solution.v_net_single_step_hard_constraint_offset),
           context + ": step negative infinity");
    expect(
        std::isinf(
            solution.v_net_max_single_step_hard_constraint_violation) &&
            std::signbit(
                solution.v_net_max_single_step_hard_constraint_violation),
        context + ": max negative infinity");
    expect(solution.place_result, context + ": place result");
    expect(solution.route_result, context + ": route result");
    expect(!solution.early_rejection, context + ": early rejection");
    expect(solution.revoke_times == 0, context + ": revoke times");
    expect(solution.selected_actions.empty(), context + ": actions");
    expect(solution.num_interactions == 0, context + ": interactions");
    expect(solution.v_net_reward == 0.0, context + ": reward");
}

void test_construction_reset_and_feasibility() {
    const auto expected = metadata();
    core::Solution solution(expected);
    expect_reset_state(solution, expected, "construction");
    expect(!solution.num_placed_nodes.has_value(), "derived node presence");
    expect(!solution.num_routed_links.has_value(), "derived link presence");
    expect(!solution.num_attempt_times.has_value(), "attempt presence");
    expect(solution.repr() ==
               "Solution({k: v for k, v in self.__dict__.items()})",
           "repr parity");

    expect(!solution.is_feasible(), "false result feasibility");
    solution.result = true;
    expect(solution.is_feasible(), "zero violation feasibility");
    solution.v_net_total_hard_constraint_violation = -2.0;
    expect(solution.is_feasible(), "negative violation feasibility");
    solution.v_net_total_hard_constraint_violation = 0.25;
    expect(!solution.is_feasible(), "positive violation feasibility");
    solution.v_net_total_hard_constraint_violation =
        std::numeric_limits<double>::quiet_NaN();
    expect(!solution.is_feasible(), "NaN violation feasibility");

    solution.node_slots.insert_or_assign(3, 7);
    solution.link_paths.insert_or_assign(
        core::SolutionLink{0, 1},
        std::vector<core::SolutionLink>{{7, 8}});
    core::SolutionAttributeValues resources;
    resources.set(2U, std::int64_t{9});
    solution.node_slots_info.insert_or_assign({3, 7}, resources);
    solution.v_net_single_step_constraint_offset.link_level.set(1U, 0.5);
    solution.v_net_constraint_offsets.node_level.insert_or_assign(3, resources);
    solution.v_net_constraint_violations.path_level.insert_or_assign(
        {0, 1}, resources);
    solution.v_net_cost = 8.0;
    solution.description = "dirty";
    solution.place_result = false;
    solution.selected_actions = {4, 5};
    solution.num_placed_nodes = 1U;
    solution.num_routed_links = 2U;
    solution.num_attempt_times = 3U;
    solution.reset();
    expect_reset_state(solution, expected, "reset");
    expect(solution.num_placed_nodes == std::optional<std::size_t>{1U},
           "reset removed later node field");
    expect(solution.num_routed_links == std::optional<std::size_t>{2U},
           "reset removed later link field");
    expect(solution.num_attempt_times == std::optional<std::size_t>{3U},
           "reset removed later attempt field");
}

void test_ordered_compact_id_containers() {
    core::NodeSlots slots;
    const auto first = slots.insert_or_assign(9, 90);
    const auto second = slots.insert_or_assign(2, 20);
    expect(first.value == 0U && second.value == 1U, "entry ID order");
    const auto overwritten = slots.insert_or_assign(9, 91);
    expect(overwritten == first, "overwrite moved entry ID");
    expect(slots.entries().size() == 2U, "overwrite appended entry");
    expect(slots.entries()[0].key == 9 && slots.entries()[0].value == 91,
           "overwrite value/order");
    expect(slots.at(first) == 91, "direct ID lookup");
    expect(slots.find_id(2) == std::optional{second}, "numeric bind");
    expect(!slots.find_id(8).has_value(), "missing numeric bind");

    expect(slots.erase(9), "erase existing");
    expect(!slots.erase(9), "erase missing");
    const auto rebound = slots.find_id(2);
    expect(rebound.has_value() && rebound->value == 0U,
           "erase did not rebuild compact IDs");

    core::SolutionAttributeValues values;
    values.set(3U, true);
    expect(values.slot_count() == 4U, "attribute slots not direct indexed");
    expect(std::get<bool>(values.at(3U)), "attribute direct lookup");
    expect(values.find(2U) == nullptr, "unset attribute slot");
    try {
        static_cast<void>(values.at(2U));
        fail("missing attribute value did not throw");
    } catch (const std::out_of_range&) {
    }
}

void expect_missing_field(
    const network::VirtualNetwork& virtual_network,
    const core::SolutionNetworkField field,
    const std::string& context) {
    try {
        static_cast<void>(core::Solution::from_v_net(virtual_network));
        fail(context + ": missing field accepted");
    } catch (const core::SolutionException& error) {
        expect(error.field() == field, context + ": field order");
    }
}

void test_virtual_network_boundary() {
    network::VirtualNetwork virtual_network;
    expect_missing_field(
        virtual_network, core::SolutionNetworkField::id, "missing id");
    virtual_network.set_request_id(42);
    expect_missing_field(
        virtual_network,
        core::SolutionNetworkField::lifetime,
        "missing lifetime");
    virtual_network.set_lifetime(6.0);
    expect_missing_field(
        virtual_network,
        core::SolutionNetworkField::arrival_time,
        "missing arrival");
    virtual_network.set_arrival_time(2.0);
    network::TopologyRequest topology;
    topology.type = network::TopologyType::Path;
    topology.num_nodes = 4;
    virtual_network.generate_topology(topology);

    const auto solution = core::Solution::from_v_net(virtual_network);
    expect(solution.v_net_id == 42, "network id");
    expect(solution.v_net_lifetime == 6.0, "network lifetime");
    expect(solution.v_net_arrival_time == 2.0, "network arrival");
    expect(solution.v_net_num_nodes == 4U, "network node count");
    expect(solution.v_net_num_edges == 3U, "network edge count");
}

void test_batches_and_concurrent_callers() {
    std::vector<core::SolutionMetadata> inputs;
    inputs.reserve(257U);
    for (std::size_t index = 0U; index < 257U; ++index) {
        inputs.push_back(metadata(index));
    }

    const auto sequential = core::Solution::from_metadata_batch(inputs, 1U);
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        auto actual = core::Solution::from_metadata_batch(inputs, workers);
        expect(actual.size() == inputs.size(), "batch size");
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            expect_reset_state(actual[index], inputs[index], "batch");
            expect(actual[index].v_net_id == sequential[index].v_net_id,
                   "batch order drift");
            actual[index].result = true;
            actual[index].v_net_cost = static_cast<double>(index + 1U);
            actual[index].node_slots.insert_or_assign(
                static_cast<std::int64_t>(index), 7);
        }
        core::Solution::reset_batch(actual, workers);
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            expect_reset_state(actual[index], inputs[index], "batch reset");
        }
    }

    auto first = std::async(std::launch::async, [&inputs] {
        return core::Solution::from_metadata_batch(inputs, 8U);
    });
    auto second = std::async(std::launch::async, [&inputs] {
        return core::Solution::from_metadata_batch(inputs, 2U);
    });
    expect(first.get().size() == inputs.size(), "first concurrent batch");
    expect(second.get().size() == inputs.size(), "second concurrent batch");
}

} // namespace

int main() {
    try {
        test_construction_reset_and_feasibility();
        test_ordered_compact_id_containers();
        test_virtual_network_boundary();
        test_batches_and_concurrent_callers();
        std::cout << "Solution unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Solution unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
