#include "../virne/solver/heuristic/ffd_rank.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/core/solution.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

std::string json_string(std::string_view value)
{
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const char raw_item : value)
    {
        const auto item = static_cast<unsigned char>(raw_item);
        switch (item)
        {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result.push_back(static_cast<char>(item));
            break;
        }
    }
    result.push_back('"');
    return result;
}

const char* json_bool(bool value) noexcept
{
    return value ? "true" : "false";
}

struct CleanupRoot
{
    std::filesystem::path path;

    ~CleanupRoot()
    {
        std::error_code error;
        const std::filesystem::path temporary_root =
            std::filesystem::temp_directory_path(error);
        const std::string filename = path.filename().string();
        if (error || path.empty() || path.parent_path() != temporary_root ||
            filename.rfind("virne-heuristic-ffd-rank-harness-", 0U) != 0U)
        {
            return;
        }
        error.clear();
        std::filesystem::remove_all(path, error);
    }
};

std::filesystem::path unique_temp_root()
{
    const auto tick = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
    return std::filesystem::temp_directory_path() /
           ("virne-heuristic-ffd-rank-harness-" + std::to_string(tick));
}

core::RecorderConfig recorder_config(const std::filesystem::path& root)
{
    core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-ffd-rank-differential";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

core::LoggerConfig logger_config(const std::filesystem::path& root)
{
    core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-ffd-rank-differential";
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
    CleanupRoot cleanup{unique_temp_root()};
    std::filesystem::path root{cleanup.path};
    controller::Controller controller{controller_selection()};
    core::Counter counter{core::CounterSelection{}};
    core::Recorder recorder{
        core::Counter(core::CounterSelection{}), recorder_config(root)};
    core::Logger logger{logger_config(root)};

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

NetworkPair ffd_permutation_case()
{
    return NetworkPair{
        make_virtual_network(3U, {}, {2, 9, 5}, {}),
        make_physical_network(3U, {}, {6, 3, 12}, {}),
    };
}

NetworkPair stable_tie_case()
{
    return NetworkPair{
        make_virtual_network(3U, {}, {8, 7, 1}, {}),
        make_physical_network(3U, {}, {10, 3, 10}, {}),
    };
}

NetworkPair link_success_case()
{
    return NetworkPair{
        make_virtual_network(2U, {{0U, 1U}}, {3, 2}, {1}),
        make_physical_network(
            3U, {{0U, 1U}, {1U, 2U}}, {10, 10, 10}, {10, 10}),
    };
}

NetworkPair partial_node_failure_case()
{
    return NetworkPair{
        make_virtual_network(2U, {}, {9, 8}, {}),
        make_physical_network(2U, {}, {10, 7}, {}),
    };
}

NetworkPair disconnected_case()
{
    return NetworkPair{
        make_virtual_network(2U, {{0U, 1U}}, {2, 1}, {1}),
        make_physical_network(2U, {}, {10, 10}, {}),
    };
}

NetworkPair empty_case()
{
    return NetworkPair{
        make_virtual_network(0U, {}, {}, {}),
        make_physical_network(0U, {}, {}, {}),
    };
}

solver::SolverConfig solver_config(
    const std::filesystem::path& root,
    bool ignored_rank_config)
{
    solver::SolverConfig config;
    config.seed = 123U;
    config.verbose = 0;
    config.save_dir = root;
    config.reusable = true;
    config.node_ranking_method = ignored_rank_config
                                     ? rank::NodeRankMethod::random
                                     : rank::NodeRankMethod::order;
    config.link_ranking_method = ignored_rank_config
                                     ? rank::LinkRankMethod::ffd
                                     : rank::LinkRankMethod::order;
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

    friend bool operator==(
        const ResourceSnapshot& left,
        const ResourceSnapshot& right)
    {
        return left.nodes == right.nodes && left.links == right.links;
    }
};

ResourceSnapshot resource_snapshot(const network::PhysicalNetwork& value)
{
    return ResourceSnapshot{
        network::get_node_attrs_data(value, {node_resource_id}, 1U),
        network::get_link_attrs_data(value, {link_resource_id}, 1U),
    };
}

std::string solution_json(std::string_view name, const core::Solution& solution)
{
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\"name\":" << json_string(name)
           << ",\"kind\":\"solution\",\"metadata\":{"
           << "\"v_net_id\":" << solution.v_net_id
           << ",\"lifetime\":" << solution.v_net_lifetime
           << ",\"arrival_time\":" << solution.v_net_arrival_time
           << ",\"num_nodes\":" << solution.v_net_num_nodes
           << ",\"num_edges\":" << solution.v_net_num_edges << "}"
           << ",\"result\":" << json_bool(solution.result)
           << ",\"place_result\":" << json_bool(solution.place_result)
           << ",\"route_result\":" << json_bool(solution.route_result)
           << ",\"node_slots\":[";

    bool first = true;
    for (const auto& entry : solution.node_slots.entries())
    {
        if (!first)
        {
            output << ',';
        }
        first = false;
        output << '[' << entry.key << ',' << entry.value << ']';
    }

    output << "],\"link_paths\":[";
    first = true;
    for (const auto& entry : solution.link_paths.entries())
    {
        if (!first)
        {
            output << ',';
        }
        first = false;
        output << "{\"virtual\":[" << entry.key.source << ','
               << entry.key.target << "],\"path\":[";
        bool first_link = true;
        for (const core::SolutionLink& link : entry.value)
        {
            if (!first_link)
            {
                output << ',';
            }
            first_link = false;
            output << '[' << link.source << ',' << link.target << ']';
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

std::string solve_case(
    std::string_view name,
    RuntimeFixture& runtime,
    const solver::SolverRegistry& registry,
    solver::SolverId solver_id,
    NetworkPair networks,
    bool ignored_rank_config)
{
    const ResourceSnapshot before = resource_snapshot(networks.physical_network);
    std::unique_ptr<solver::Solver> instance = registry.create(
        solver_id,
        runtime.dependencies(),
        solver_config(runtime.root, ignored_rank_config));
    const core::Solution solution = instance->solve(solver::SolverInstance{
        networks.virtual_network,
        networks.physical_network,
    });
    if (!(before == resource_snapshot(networks.physical_network)))
    {
        throw std::runtime_error("solver mutated input physical resources");
    }
    return solution_json(name, solution);
}

std::size_t parse_workers(int argc, char** argv)
{
    if (argc != 3 || std::string_view(argv[1]) != "--workers")
    {
        throw std::runtime_error(
            "usage: heuristic_ffd_rank_harness --workers N");
    }
    return static_cast<std::size_t>(std::stoull(argv[2]));
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::size_t worker_width = parse_workers(argc, argv);
        RuntimeFixture runtime;
        solver::SolverRegistry registry;
        const heuristic::NodeRankSolverWorkers workers{
            worker_width, worker_width, worker_width, worker_width};
        const solver::SolverId solver_id =
            heuristic::register_ffd_rank_solver(registry, workers);
        registry.freeze();
        const solver::SolverDescriptor& descriptor = registry.descriptor(solver_id);
        if (descriptor.name != "ffd_rank" ||
            descriptor.category != solver::SolverCategory::node_ranking)
        {
            throw std::runtime_error("FFD-rank descriptor mismatch");
        }

        std::vector<std::string> cases;
        cases.reserve(6U);
        cases.push_back(solve_case(
            "ffd_permutation",
            runtime,
            registry,
            solver_id,
            ffd_permutation_case(),
            false));
        cases.push_back(solve_case(
            "stable_tie",
            runtime,
            registry,
            solver_id,
            stable_tie_case(),
            false));
        cases.push_back(solve_case(
            "link_success",
            runtime,
            registry,
            solver_id,
            link_success_case(),
            false));
        cases.push_back(solve_case(
            "partial_node_failure",
            runtime,
            registry,
            solver_id,
            partial_node_failure_case(),
            false));
        cases.push_back(solve_case(
            "disconnected_route_failure",
            runtime,
            registry,
            solver_id,
            disconnected_case(),
            false));
        cases.push_back(solve_case(
            "empty_success",
            runtime,
            registry,
            solver_id,
            empty_case(),
            false));

        std::cout << "{\"component\":\"solver.heuristic.node_rank\","
                     "\"solver\":\"ffd_rank\","
                     "\"category\":\"node_ranking\",\"solver_id\":"
                  << solver_id.value << ",\"workers\":" << worker_width
                  << ",\"native_input_physical_unchanged\":true,\"cases\":[";
        for (std::size_t index = 0U; index < cases.size(); ++index)
        {
            if (index != 0U)
            {
                std::cout << ',';
            }
            std::cout << cases[index];
        }
        std::cout << "]}\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "heuristic FFD rank harness: FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
