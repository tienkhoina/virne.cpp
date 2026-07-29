#include "../virne/solver/heuristic/node_rank.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/core/solution.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace heuristic = virne::solver::heuristic;
namespace network = virne::network;
namespace rank = virne::solver::rank;
namespace solver = virne::solver;

constexpr attribute::AttributeRegistryId node_resource_id{0U};
constexpr attribute::AttributeRegistryId link_resource_id{0U};

void require(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

std::filesystem::path unique_temp_root()
{
    static std::atomic<std::uint64_t> sequence{0U};
    const auto tick = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
    const auto suffix = sequence.fetch_add(1U, std::memory_order_relaxed);
    return std::filesystem::temp_directory_path() /
           ("virne-heuristic-node-rank-unit-" + std::to_string(tick) + "-" +
            std::to_string(suffix));
}

struct CleanupRoot
{
    std::filesystem::path path;

    explicit CleanupRoot(std::filesystem::path value)
        : path(std::move(value))
    {
    }

    ~CleanupRoot()
    {
        std::error_code error;
        const auto temporary_root =
            std::filesystem::temp_directory_path(error);
        const std::string filename = path.filename().string();
        if (error || path.empty() || path.parent_path() != temporary_root ||
            filename.rfind("virne-heuristic-node-rank-unit-", 0U) != 0U)
        {
            return;
        }
        error.clear();
        std::filesystem::remove_all(path, error);
    }
};

core::RecorderConfig recorder_config(const std::filesystem::path& root)
{
    core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-node-rank-unit";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

core::LoggerConfig logger_config(const std::filesystem::path& root)
{
    core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-node-rank-unit";
    config.run_id = "logger";
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = false;
    config.backends.file = false;
    config.level = core::LoggerLevel::critical;
    config.log_show_interval = 1U;
    return config;
}

controller::ControllerSelection controller_selection()
{
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {node_resource_id};
    selection.constraints.link_at_link = {link_resource_id};
    selection.node_resources = {node_resource_id};
    selection.link_resources = {link_resource_id};
    selection.hard_node_constraints = {node_resource_id};
    selection.hard_link_constraints = {link_resource_id};
    selection.reusable = false;
    return selection;
}

struct RuntimeFixture
{
    CleanupRoot cleanup;
    std::filesystem::path root;
    controller::Controller controller;
    core::Counter counter;
    core::Recorder recorder;
    core::Logger logger;

    RuntimeFixture()
        : cleanup(unique_temp_root()),
          root(cleanup.path),
          controller(controller_selection()),
          counter(core::CounterSelection{}),
          recorder(core::Counter(core::CounterSelection{}),
                   recorder_config(root)),
          logger(logger_config(root))
    {
    }

    solver::SolverDependencies dependencies()
    {
        return solver::SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger),
        };
    }
};

attribute::AttributeFactorySpec node_resource_spec()
{
    attribute::AttributeFactorySpec spec;
    spec.name = "cpu";
    spec.owner = attribute::AttributeOwner::node;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::node;
    return spec;
}

attribute::AttributeFactorySpec link_resource_spec()
{
    attribute::AttributeFactorySpec spec;
    spec.name = "bandwidth";
    spec.owner = attribute::AttributeOwner::link;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::link;
    return spec;
}

std::vector<AttrValue> attribute_values(
    const std::vector<std::int64_t>& values)
{
    std::vector<AttrValue> result;
    result.reserve(values.size());
    for (const std::int64_t value : values)
    {
        result.emplace_back(value);
    }
    return result;
}

network::NodeAttributeDataUpdate node_update(
    const std::vector<std::int64_t>& values)
{
    network::NodeAttributeDataUpdate update;
    update.registry_id = node_resource_id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = attribute_values(values);
    return update;
}

network::LinkAttributeDataUpdate link_update(
    const std::vector<std::int64_t>& values)
{
    network::LinkAttributeDataUpdate update;
    update.registry_id = link_resource_id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = attribute_values(values);
    return update;
}

network::VirtualNetwork make_virtual_network(
    std::size_t node_count,
    std::vector<EdgeEndpoints> edges,
    const std::vector<std::int64_t>& node_demands,
    const std::vector<std::int64_t>& link_demands)
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(node_count, std::move(edges));
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};

    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update(node_demands)});
    result.set_link_attrs_data({link_update(link_demands)});
    result.set_request_id(17);
    result.set_arrival_time(2.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_physical_network(
    std::size_t node_count,
    std::vector<EdgeEndpoints> edges,
    const std::vector<std::int64_t>& node_capacities,
    const std::vector<std::int64_t>& link_capacities)
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(node_count, std::move(edges));
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};

    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update(node_capacities)});
    result.set_link_attrs_data({link_update(link_capacities)});
    return result;
}

struct NetworkPair
{
    network::VirtualNetwork virtual_network;
    network::PhysicalNetwork physical_network;
};

NetworkPair make_no_link_success_case()
{
    return NetworkPair{
        make_virtual_network(1U, {}, {3}, {}),
        make_physical_network(1U, {}, {10}, {}),
    };
}

NetworkPair make_link_success_case()
{
    return NetworkPair{
        make_virtual_network(2U, {{0U, 1U}}, {2, 3}, {1}),
        make_physical_network(
            3U, {{0U, 1U}, {1U, 2U}}, {10, 10, 10}, {10, 10}),
    };
}

NetworkPair make_node_failure_case()
{
    return NetworkPair{
        make_virtual_network(2U, {}, {1, 100}, {}),
        make_physical_network(2U, {}, {10, 10}, {}),
    };
}

NetworkPair make_disconnected_case()
{
    return NetworkPair{
        make_virtual_network(2U, {{0U, 1U}}, {1, 1}, {1}),
        make_physical_network(2U, {}, {10, 10}, {}),
    };
}

NetworkPair make_multi_link_order_case()
{
    return NetworkPair{
        make_virtual_network(
            4U,
            {{2U, 3U}, {0U, 2U}, {1U, 3U}, {0U, 1U}},
            {1, 1, 1, 1},
            {1, 1, 1, 1}),
        make_physical_network(
            4U,
            {{0U, 1U}, {1U, 2U}, {2U, 3U}},
            {10, 10, 10, 10},
            {10, 10, 10}),
    };
}

NetworkPair make_matching_case()
{
    return NetworkPair{
        make_virtual_network(1U, {}, {5}, {}),
        make_physical_network(2U, {}, {1, 10}, {}),
    };
}

solver::SolverConfig solver_config(const std::filesystem::path& root)
{
    solver::SolverConfig config;
    config.seed = 123U;
    config.verbose = 0;
    config.save_dir = root;
    config.reusable = true;
    config.node_ranking_method = rank::NodeRankMethod::nps;
    config.link_ranking_method = rank::LinkRankMethod::ffd;
    config.matching_method = controller::NodeMatchingMethod::greedy;
    config.shortest_method = controller::ShortestPathMethod::k_shortest;
    config.k_shortest = 10;
    config.allow_rejection = true;
    config.allow_revocable = true;
    return config;
}

struct ResourceSnapshot
{
    std::vector<std::vector<AttrValue>> nodes;
    std::vector<std::vector<AttrValue>> links;
};

ResourceSnapshot resource_snapshot(const network::PhysicalNetwork& value)
{
    return ResourceSnapshot{
        network::get_node_attrs_data(value, {node_resource_id}, 1U),
        network::get_link_attrs_data(value, {link_resource_id}, 1U),
    };
}

bool same_resources(
    const ResourceSnapshot& left,
    const ResourceSnapshot& right)
{
    return left.nodes == right.nodes && left.links == right.links;
}

class WordWriter
{
public:
    void append(std::uint64_t value)
    {
        words_.push_back(value);
    }

    void append(bool value)
    {
        append(value ? UINT64_C(1) : UINT64_C(0));
    }

    void append(std::int64_t value)
    {
        append(static_cast<std::uint64_t>(value));
    }

    void append(double value)
    {
        std::uint64_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value), "double width mismatch");
        std::memcpy(&bits, &value, sizeof(bits));
        append(bits);
    }

    void append(std::string_view value)
    {
        append(value.size());
        for (const char byte : value)
        {
            append(static_cast<std::uint64_t>(
                static_cast<unsigned char>(byte)));
        }
    }

    std::vector<std::uint64_t> take()
    {
        return std::move(words_);
    }

private:
    std::vector<std::uint64_t> words_;
};

void append_link(WordWriter& writer, const core::SolutionLink& value)
{
    writer.append(value.source);
    writer.append(value.target);
}

void append_attribute_number(
    WordWriter& writer,
    const attribute::AttributeNumber& value)
{
    writer.append(value.index());
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        writer.append(*boolean);
    }
    else if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        writer.append(*integer);
    }
    else
    {
        writer.append(std::get<double>(value));
    }
}

void append_attribute_values(
    WordWriter& writer,
    const core::SolutionAttributeValues& values)
{
    writer.append(values.slot_count());
    for (const auto& slot : values.slots())
    {
        writer.append(slot.has_value());
        if (slot.has_value())
        {
            append_attribute_number(writer, *slot);
        }
    }
}

template <typename Table, typename AppendKey>
void append_attribute_table(
    WordWriter& writer,
    const Table& table,
    AppendKey append_key)
{
    writer.append(table.size());
    for (const auto& entry : table.entries())
    {
        append_key(writer, entry.key);
        append_attribute_values(writer, entry.value);
    }
}

void append_node_table(WordWriter& writer, const core::NodeConstraintTable& table)
{
    append_attribute_table(
        writer, table,
        [](WordWriter& output, core::SolutionNodeId key)
        {
            output.append(key);
        });
}

void append_node_table(WordWriter& writer, const core::NodeViolationTable& table)
{
    append_attribute_table(
        writer, table,
        [](WordWriter& output, core::SolutionNodeId key)
        {
            output.append(key);
        });
}

template <typename Table>
void append_link_table(WordWriter& writer, const Table& table)
{
    append_attribute_table(
        writer, table,
        [](WordWriter& output, const core::SolutionLink& key)
        {
            append_link(output, key);
        });
}

template <typename Value>
void append_optional_size(WordWriter& writer, const std::optional<Value>& value)
{
    writer.append(value.has_value());
    if (value.has_value())
    {
        writer.append(static_cast<std::size_t>(*value));
    }
}

std::vector<std::uint64_t> solution_snapshot(const core::Solution& value)
{
    WordWriter writer;
    writer.append(value.v_net_id);
    writer.append(value.v_net_lifetime);
    writer.append(value.v_net_arrival_time);
    writer.append(value.v_net_num_nodes);
    writer.append(value.v_net_num_edges);
    writer.append(value.result);

    writer.append(value.node_slots.size());
    for (const auto& entry : value.node_slots.entries())
    {
        writer.append(entry.key);
        writer.append(entry.value);
    }

    writer.append(value.link_paths.size());
    for (const auto& entry : value.link_paths.entries())
    {
        append_link(writer, entry.key);
        writer.append(entry.value.size());
        for (const core::SolutionLink& link : entry.value)
        {
            append_link(writer, link);
        }
    }

    writer.append(value.node_slots_info.size());
    for (const auto& entry : value.node_slots_info.entries())
    {
        writer.append(entry.key.virtual_node);
        writer.append(entry.key.physical_node);
        append_attribute_values(writer, entry.value);
    }

    writer.append(value.link_paths_info.size());
    for (const auto& entry : value.link_paths_info.entries())
    {
        append_link(writer, entry.key.virtual_link);
        append_link(writer, entry.key.physical_link);
        append_attribute_values(writer, entry.value);
    }

    writer.append(value.v_net_cost);
    writer.append(value.v_net_revenue);
    writer.append(value.v_net_demand);
    writer.append(value.v_net_node_demand);
    writer.append(value.v_net_link_demand);
    writer.append(value.v_net_node_revenue);
    writer.append(value.v_net_link_revenue);
    writer.append(value.v_net_node_cost);
    writer.append(value.v_net_link_cost);
    writer.append(value.v_net_path_cost);
    writer.append(value.v_net_r2c_ratio);
    writer.append(value.v_net_time_cost);
    writer.append(value.v_net_time_revenue);
    writer.append(value.v_net_time_rc_ratio);
    writer.append(std::string_view(value.description));
    writer.append(value.v_net_total_hard_constraint_violation);

    append_attribute_values(
        writer, value.v_net_single_step_constraint_offset.node_level);
    append_attribute_values(
        writer, value.v_net_single_step_constraint_offset.link_level);
    append_attribute_values(
        writer, value.v_net_single_step_constraint_offset.path_level);
    append_node_table(writer, value.v_net_constraint_offsets.node_level);
    append_link_table(writer, value.v_net_constraint_offsets.link_level);
    append_link_table(writer, value.v_net_constraint_offsets.path_level);
    append_node_table(writer, value.v_net_constraint_violations.node_level);
    append_link_table(writer, value.v_net_constraint_violations.link_level);
    append_link_table(writer, value.v_net_constraint_violations.path_level);

    writer.append(value.v_net_single_step_violation_list.size());
    for (const attribute::AttributeNumber& item :
         value.v_net_single_step_violation_list)
    {
        append_attribute_number(writer, item);
    }
    writer.append(value.v_net_single_step_hard_constraint_offset);
    writer.append(value.v_net_max_single_step_hard_constraint_violation);
    writer.append(value.place_result);
    writer.append(value.route_result);
    writer.append(value.early_rejection);
    writer.append(value.revoke_times);
    writer.append(value.selected_actions.size());
    for (const std::int64_t action : value.selected_actions)
    {
        writer.append(action);
    }
    writer.append(value.num_interactions);
    writer.append(value.v_net_reward);
    append_optional_size(writer, value.num_placed_nodes);
    append_optional_size(writer, value.num_routed_links);
    append_optional_size(writer, value.num_attempt_times);
    return writer.take();
}

core::Solution solve_order(
    RuntimeFixture& runtime,
    NetworkPair& networks,
    solver::SolverConfig config,
    heuristic::NodeRankSolverWorkers workers = {})
{
    heuristic::OrderRankSolver order_solver(
        runtime.dependencies(), std::move(config), workers);
    return order_solver.solve(solver::SolverInstance{
        networks.virtual_network,
        networks.physical_network,
    });
}

void require_single_slot(
    const core::Solution& solution,
    core::SolutionNodeId virtual_node,
    core::SolutionNodeId physical_node,
    std::string_view message)
{
    require(solution.node_slots.size() == 1U, message);
    const auto& entry = solution.node_slots.entries().front();
    require(
        entry.key == virtual_node && entry.value == physical_node,
        message);
}

void require_two_identity_slots(
    const core::Solution& solution,
    std::string_view message)
{
    require(solution.node_slots.size() == 2U, message);
    const auto& entries = solution.node_slots.entries();
    require(
        entries[0U].key == 0 && entries[0U].value == 0 &&
            entries[1U].key == 1 && entries[1U].value == 1,
        message);
}

void test_accessors_and_registration()
{
    RuntimeFixture runtime;
    const heuristic::NodeRankSolverWorkers workers{0U, 2U, 8U, 1U};
    heuristic::OrderRankSolver direct(
        runtime.dependencies(), solver_config(runtime.root), workers);
    require(
        direct.node_rank_method() == rank::NodeRankMethod::order,
        "direct solver method mismatch");
    require(
        direct.workers().rank_workers == 0U &&
            direct.workers().node_candidate_workers == 2U &&
            direct.workers().link_topology_constraint_workers == 8U &&
            direct.workers().link_candidate_workers == 1U,
        "direct solver worker snapshot mismatch");

    solver::SolverRegistry registry;
    const solver::SolverId id =
        heuristic::register_order_rank_solver(registry, workers);
    require(id == solver::SolverId{0U}, "registered compact ID mismatch");
    const solver::SolverId resolved = registry.resolve("order_rank");
    require(resolved == id, "cold name resolution mismatch");
    registry.freeze();
    const solver::SolverDescriptor& descriptor = registry.descriptor(id);
    require(
        descriptor.id == id && descriptor.name == "order_rank" &&
            descriptor.category == solver::SolverCategory::node_ranking,
        "registered descriptor mismatch");

    std::unique_ptr<solver::Solver> created = registry.create(
        id, runtime.dependencies(), solver_config(runtime.root));
    const auto* typed =
        dynamic_cast<const heuristic::OrderRankSolver*>(created.get());
    require(typed != nullptr, "registered factory type mismatch");
    require(
        typed->node_rank_method() == rank::NodeRankMethod::order &&
            typed->workers().rank_workers == 0U &&
            typed->workers().node_candidate_workers == 2U &&
            typed->workers().link_topology_constraint_workers == 8U &&
            typed->workers().link_candidate_workers == 1U,
        "registered factory state mismatch");
}

void test_success_without_link()
{
    RuntimeFixture runtime;
    NetworkPair networks = make_no_link_success_case();
    const ResourceSnapshot before = resource_snapshot(networks.physical_network);
    const core::Solution solution = solve_order(
        runtime, networks, solver_config(runtime.root));

    require(
        solution.result && solution.place_result && solution.route_result,
        "no-link success flags mismatch");
    require_single_slot(solution, 0, 0, "no-link node slot mismatch");
    require(solution.link_paths.empty(), "no-link solution wrote a path");
    require(
        same_resources(before, resource_snapshot(networks.physical_network)),
        "no-link solve mutated input physical resources");
}

void test_success_with_link()
{
    RuntimeFixture runtime;
    NetworkPair networks = make_link_success_case();
    const ResourceSnapshot before = resource_snapshot(networks.physical_network);
    const core::Solution solution = solve_order(
        runtime, networks, solver_config(runtime.root));

    require(
        solution.result && solution.place_result && solution.route_result,
        "link success flags mismatch");
    require_two_identity_slots(solution, "link success slots mismatch");
    require(solution.link_paths.size() == 1U, "link success path count mismatch");
    const auto& path = solution.link_paths.entries().front();
    require(
        path.key == core::SolutionLink{0, 1} && path.value.size() == 1U &&
            path.value[0U] == core::SolutionLink{0, 1},
        "link success route mismatch");
    require(
        same_resources(before, resource_snapshot(networks.physical_network)),
        "link solve mutated input physical resources");
}

void test_partial_node_failure()
{
    RuntimeFixture runtime;
    NetworkPair networks = make_node_failure_case();
    const ResourceSnapshot before = resource_snapshot(networks.physical_network);
    const core::Solution solution = solve_order(
        runtime, networks, solver_config(runtime.root));

    require(
        !solution.result && !solution.place_result && solution.route_result,
        "node failure flag/write order mismatch");
    require_single_slot(solution, 0, 0, "node failure lost partial placement");
    require(solution.link_paths.empty(), "node failure unexpectedly routed");
    require(
        same_resources(before, resource_snapshot(networks.physical_network)),
        "node failure mutated input physical resources");
}

void test_disconnected_route_failure()
{
    RuntimeFixture runtime;
    NetworkPair networks = make_disconnected_case();
    const ResourceSnapshot before = resource_snapshot(networks.physical_network);
    const core::Solution solution = solve_order(
        runtime, networks, solver_config(runtime.root));

    require(
        !solution.result && solution.place_result && !solution.route_result,
        "route failure flags mismatch");
    require_two_identity_slots(solution, "route failure lost node placements");
    require(
        same_resources(before, resource_snapshot(networks.physical_network)),
        "route failure mutated input physical resources");
}

void test_live_multi_link_order()
{
    RuntimeFixture runtime;
    NetworkPair networks = make_multi_link_order_case();
    const core::Solution solution = solve_order(
        runtime, networks, solver_config(runtime.root));

    require(solution.result, "multi-link order case failed");
    const auto& paths = solution.link_paths.entries();
    const std::array<core::SolutionLink, 4U> expected{{
        {0, 2},
        {0, 1},
        {1, 3},
        {2, 3},
    }};
    require(paths.size() == expected.size(), "multi-link path count mismatch");
    for (std::size_t index = 0U; index < expected.size(); ++index)
    {
        require(
            paths[index].key == expected[index],
            "virtual links did not retain live graph edge order");
    }
}

void test_rank_config_ignored()
{
    RuntimeFixture baseline_runtime;
    NetworkPair baseline_networks = make_link_success_case();
    solver::SolverConfig baseline_config = solver_config(baseline_runtime.root);
    baseline_config.node_ranking_method = rank::NodeRankMethod::order;
    baseline_config.link_ranking_method = rank::LinkRankMethod::order;
    const auto baseline = solution_snapshot(solve_order(
        baseline_runtime, baseline_networks, baseline_config));

    RuntimeFixture ignored_runtime;
    NetworkPair ignored_networks = make_link_success_case();
    solver::SolverConfig ignored_config = solver_config(ignored_runtime.root);
    ignored_config.node_ranking_method = rank::NodeRankMethod::random;
    ignored_config.link_ranking_method = rank::LinkRankMethod::ffd;
    const auto ignored = solution_snapshot(solve_order(
        ignored_runtime, ignored_networks, ignored_config));
    require(ignored == baseline, "config rank fields changed order solver output");
}

void test_matching_config_forwarded()
{
    RuntimeFixture greedy_runtime;
    NetworkPair greedy_networks = make_matching_case();
    solver::SolverConfig greedy_config = solver_config(greedy_runtime.root);
    greedy_config.matching_method = controller::NodeMatchingMethod::greedy;
    const core::Solution greedy = solve_order(
        greedy_runtime, greedy_networks, greedy_config);
    require(
        greedy.result && greedy.place_result,
        "greedy did not scan to a feasible candidate");
    require_single_slot(greedy, 0, 1, "greedy candidate order mismatch");

    RuntimeFixture l2s2_runtime;
    NetworkPair l2s2_networks = make_matching_case();
    solver::SolverConfig l2s2_config = solver_config(l2s2_runtime.root);
    l2s2_config.matching_method = controller::NodeMatchingMethod::l2s2;
    const core::Solution l2s2 = solve_order(
        l2s2_runtime, l2s2_networks, l2s2_config);
    require(
        !l2s2.result && !l2s2.place_result && l2s2.route_result &&
            l2s2.node_slots.empty(),
        "l2s2 matching config was not forwarded");
}

void test_shortest_method_and_k_forwarded()
{
    RuntimeFixture success_runtime;
    NetworkPair success_networks = make_link_success_case();
    solver::SolverConfig success_config = solver_config(success_runtime.root);
    success_config.shortest_method =
        controller::ShortestPathMethod::k_shortest_length;
    success_config.k_shortest = 2;
    const core::Solution success = solve_order(
        success_runtime, success_networks, success_config);
    require(success.result && success.route_result,
            "inclusive path-node cutoff was not forwarded");

    RuntimeFixture failure_runtime;
    NetworkPair failure_networks = make_link_success_case();
    solver::SolverConfig failure_config = solver_config(failure_runtime.root);
    failure_config.shortest_method =
        controller::ShortestPathMethod::k_shortest_length;
    failure_config.k_shortest = 1;
    const core::Solution failure = solve_order(
        failure_runtime, failure_networks, failure_config);
    require(
        !failure.result && failure.place_result && !failure.route_result,
        "exclusive path-node cutoff was not forwarded");
    require_two_identity_slots(failure, "k cutoff lost placements");
}

void test_worker_exact_equality()
{
    constexpr std::array<std::size_t, 4U> widths{{0U, 1U, 2U, 8U}};
    std::optional<std::vector<std::uint64_t>> expected;
    for (const std::size_t width : widths)
    {
        RuntimeFixture runtime;
        NetworkPair networks = make_link_success_case();
        const ResourceSnapshot before = resource_snapshot(networks.physical_network);
        const heuristic::NodeRankSolverWorkers workers{
            width, width, width, width};
        const core::Solution solution = solve_order(
            runtime, networks, solver_config(runtime.root), workers);
        const auto snapshot = solution_snapshot(solution);
        if (!expected.has_value())
        {
            expected = snapshot;
        }
        else
        {
            require(snapshot == *expected, "worker width changed exact output");
        }
        require(
            same_resources(before, resource_snapshot(networks.physical_network)),
            "worker solve mutated input physical resources");
    }
}

void test_empty_virtual_precedes_physical()
{
    RuntimeFixture runtime;
    NetworkPair networks{
        make_virtual_network(0U, {}, {}, {}),
        make_physical_network(0U, {}, {}, {}),
    };
    try
    {
        static_cast<void>(solve_order(
            runtime, networks, solver_config(runtime.root)));
    }
    catch (const rank::NodeRankException& error)
    {
        require(
            error.operation() == rank::NodeRankOperation::reduce,
            "empty virtual rank operation mismatch");
        return;
    }
    throw std::runtime_error(
        "empty virtual network did not propagate NodeRankException");
}

void test_concurrent_registry_readers_and_independent_solves()
{
    constexpr std::size_t caller_count = 8U;
    const heuristic::NodeRankSolverWorkers workers{8U, 8U, 8U, 8U};
    solver::SolverRegistry registry;
    const solver::SolverId id =
        heuristic::register_order_rank_solver(registry, workers);
    registry.freeze();

    std::vector<std::vector<std::uint64_t>> snapshots(caller_count);
    std::vector<std::exception_ptr> errors(caller_count);
    std::vector<std::thread> callers;
    callers.reserve(caller_count);
    for (std::size_t caller = 0U; caller < caller_count; ++caller)
    {
        callers.emplace_back([&, caller]
        {
            try
            {
                const solver::SolverDescriptor& descriptor =
                    registry.descriptor(id);
                require(
                    descriptor.id == id &&
                        descriptor.category == solver::SolverCategory::node_ranking,
                    "concurrent descriptor mismatch");

                RuntimeFixture runtime;
                NetworkPair networks = make_link_success_case();
                const ResourceSnapshot before =
                    resource_snapshot(networks.physical_network);
                std::unique_ptr<solver::Solver> instance = registry.create(
                    id, runtime.dependencies(), solver_config(runtime.root));
                core::Solution solution = instance->solve(solver::SolverInstance{
                    networks.virtual_network,
                    networks.physical_network,
                });
                require(
                    same_resources(
                        before, resource_snapshot(networks.physical_network)),
                    "concurrent solve mutated input physical resources");
                snapshots[caller] = solution_snapshot(solution);
            }
            catch (...)
            {
                errors[caller] = std::current_exception();
            }
        });
    }
    for (std::thread& caller : callers)
    {
        caller.join();
    }
    for (const std::exception_ptr& error : errors)
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }
    for (std::size_t caller = 1U; caller < caller_count; ++caller)
    {
        require(
            snapshots[caller] == snapshots[0U],
            "concurrent independent solver output drift");
    }
}

} // namespace

int main()
{
    try
    {
        test_accessors_and_registration();
        test_success_without_link();
        test_success_with_link();
        test_partial_node_failure();
        test_disconnected_route_failure();
        test_live_multi_link_order();
        test_rank_config_ignored();
        test_matching_config_forwarded();
        test_shortest_method_and_k_forwarded();
        test_worker_exact_equality();
        test_empty_virtual_precedes_physical();
        test_concurrent_registry_readers_and_independent_solves();
        std::cout << "heuristic node rank unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "heuristic node rank unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
