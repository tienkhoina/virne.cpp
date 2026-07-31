#include "../virne/solver/heuristic/custom_rank_variants.h"
#include "../virne/solver/heuristic/ffd_rank.h"
#include "../virne/solver/heuristic/random_rank.h"
#include "../virne/solver/heuristic/standard_rank_variants.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../random/numpy_random_state.h"

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
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
namespace solver = virne::solver;

constexpr attribute::AttributeRegistryId node_resource_id{0U};
constexpr attribute::AttributeRegistryId link_resource_id{0U};

std::string json_string(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const char item : value) {
        if (item == '"' || item == '\\') {
            result.push_back('\\');
        }
        result.push_back(item);
    }
    result.push_back('"');
    return result;
}

const char* json_bool(bool value) noexcept {
    return value ? "true" : "false";
}

struct CleanupRoot {
    std::filesystem::path path;

    ~CleanupRoot() {
        std::error_code error;
        const auto root = std::filesystem::temp_directory_path(error);
        const std::string name = path.filename().string();
        if (!error && path.parent_path() == root &&
            name.rfind("virne-node-rank-variants-harness-", 0U) == 0U) {
            std::filesystem::remove_all(path, error);
        }
    }
};

std::filesystem::path unique_temp_root() {
    const auto tick = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
    return std::filesystem::temp_directory_path() /
        ("virne-node-rank-variants-harness-" + std::to_string(tick));
}

controller::ControllerSelection controller_selection() {
    controller::ControllerSelection result;
    result.constraints.node_at_node = {node_resource_id};
    result.constraints.link_at_link = {link_resource_id};
    result.node_resources = {node_resource_id};
    result.link_resources = {link_resource_id};
    result.hard_node_constraints = {node_resource_id};
    result.hard_link_constraints = {link_resource_id};
    return result;
}

core::RecorderConfig recorder_config(const std::filesystem::path& root) {
    core::RecorderConfig result;
    result.save_root_dir = root;
    result.solver_name = "node-rank-variants-differential";
    result.run_id = "recorder";
    result.temporary_records = false;
    return result;
}

core::LoggerConfig logger_config(const std::filesystem::path& root) {
    core::LoggerConfig result;
    result.save_root_dir = root;
    result.solver_name = "node-rank-variants-differential";
    result.run_id = "logger";
    result.backends.console = false;
    result.backends.file = false;
    result.level = core::LoggerLevel::critical;
    return result;
}

struct RuntimeFixture {
    CleanupRoot cleanup{unique_temp_root()};
    controller::Controller controller{controller_selection()};
    core::Counter counter{core::CounterSelection{}};
    core::Recorder recorder{
        core::Counter(core::CounterSelection{}),
        recorder_config(cleanup.path)};
    core::Logger logger{logger_config(cleanup.path)};

    solver::SolverDependencies dependencies() {
        return {
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger),
        };
    }
};

attribute::AttributeFactorySpec node_resource_spec() {
    attribute::AttributeFactorySpec result;
    result.name = "cpu";
    result.owner = attribute::AttributeOwner::node;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = attribute::CheckingLevel::node;
    return result;
}

attribute::AttributeFactorySpec link_resource_spec() {
    attribute::AttributeFactorySpec result;
    result.name = "bandwidth";
    result.owner = attribute::AttributeOwner::link;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = attribute::CheckingLevel::link;
    return result;
}

std::vector<AttrValue> values(const std::vector<std::int64_t>& input) {
    std::vector<AttrValue> result;
    result.reserve(input.size());
    for (const auto value : input) {
        result.emplace_back(value);
    }
    return result;
}

network::NodeAttributeDataUpdate node_update(
    const std::vector<std::int64_t>& input) {
    network::NodeAttributeDataUpdate result;
    result.registry_id = node_resource_id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = values(input);
    return result;
}

network::LinkAttributeDataUpdate link_update(
    const std::vector<std::int64_t>& input) {
    network::LinkAttributeDataUpdate result;
    result.registry_id = link_resource_id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = values(input);
    return result;
}

network::VirtualNetwork make_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update({3, 2})});
    result.set_link_attrs_data({link_update({2})});
    result.set_request_id(31);
    result.set_arrival_time(2.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}, {0U, 2U}});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update({10, 12, 9})});
    result.set_link_attrs_data({link_update({10, 8, 6})});
    return result;
}

network::VirtualNetwork make_sparse_tie_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        1U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update({10})});
    result.set_link_attrs_data({link_update({})});
    result.set_request_id(37);
    result.set_arrival_time(0.0);
    result.set_lifetime(1.0);
    return result;
}

network::PhysicalNetwork make_sparse_tie_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        9U, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update(
        {0, 10, 0, 0, 0, 0, 0, 0, 10})});
    result.set_link_attrs_data({link_update({})});
    return result;
}

std::vector<EdgeEndpoints> path_edges(std::size_t node_count) {
    std::vector<EdgeEndpoints> result;
    if (node_count == 0U) {
        return result;
    }
    result.reserve(node_count - 1U);
    for (Vertex node = 1U; node < node_count; ++node) {
        result.push_back(EdgeEndpoints{node - 1U, node});
    }
    return result;
}

network::VirtualNetwork make_benchmark_virtual_network() {
    constexpr std::size_t node_count = 8U;
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        node_count, path_edges(node_count));
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::VirtualNetwork result(std::move(construction));

    std::vector<std::int64_t> nodes(node_count);
    for (std::size_t index = 0U; index < node_count; ++index) {
        nodes[index] = 5 + static_cast<std::int64_t>((index * 3U) % 7U);
    }
    std::vector<std::int64_t> links(node_count - 1U);
    for (std::size_t index = 0U; index < links.size(); ++index) {
        links[index] = 2 + static_cast<std::int64_t>((index * 5U) % 9U);
    }
    result.set_node_attrs_data({node_update(nodes)});
    result.set_link_attrs_data({link_update(links)});
    result.set_request_id(41);
    result.set_arrival_time(3.0);
    result.set_lifetime(20.0);
    return result;
}

network::PhysicalNetwork make_benchmark_physical_network() {
    constexpr std::size_t node_count = 48U;
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        node_count, path_edges(node_count));
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));

    std::vector<std::int64_t> nodes(node_count);
    for (std::size_t index = 0U; index < node_count; ++index) {
        nodes[index] =
            100 + static_cast<std::int64_t>((index * 17U) % 31U);
    }
    std::vector<std::int64_t> links(node_count - 1U);
    for (std::size_t index = 0U; index < links.size(); ++index) {
        links[index] =
            100 + static_cast<std::int64_t>((index * 13U) % 29U);
    }
    result.set_node_attrs_data({node_update(nodes)});
    result.set_link_attrs_data({link_update(links)});
    return result;
}

solver::SolverConfig solver_config(const std::filesystem::path& root) {
    solver::SolverConfig result;
    result.seed = 77U;
    result.verbose = 0;
    result.save_dir = root;
    result.matching_method = controller::NodeMatchingMethod::greedy;
    result.shortest_method = controller::ShortestPathMethod::bfs_shortest;
    result.k_shortest = 10;
    return result;
}

std::vector<solver::SolverId> register_all_solvers(
    solver::SolverRegistry& registry,
    NumpyRandomState& random,
    heuristic::NodeRankSolverWorkers workers) {
    return {
        heuristic::register_order_rank_solver(registry, workers),
        heuristic::register_random_rank_solver(registry, random, workers),
        heuristic::register_grc_rank_solver(registry, {}, workers),
        heuristic::register_ffd_rank_solver(registry, workers),
        heuristic::register_nrm_rank_solver(registry, workers),
        heuristic::register_pl_rank_solver(registry, workers),
        heuristic::register_nea_rank_solver(registry, workers),
        heuristic::register_random_walk_rank_solver(registry, {}, workers),
    };
}

struct ResourceSnapshot {
    std::vector<std::vector<AttrValue>> nodes;
    std::vector<std::vector<AttrValue>> links;

    friend bool operator==(
        const ResourceSnapshot& left,
        const ResourceSnapshot& right) {
        return left.nodes == right.nodes && left.links == right.links;
    }
};

ResourceSnapshot snapshot(const network::PhysicalNetwork& network_value) {
    return {
        network::get_node_attrs_data(
            network_value, {node_resource_id}, 1U),
        network::get_link_attrs_data(
            network_value, {link_resource_id}, 1U),
    };
}

std::string solution_json(const core::Solution& solution) {
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"result\":" << json_bool(solution.result)
           << ",\"place_result\":" << json_bool(solution.place_result)
           << ",\"route_result\":" << json_bool(solution.route_result)
           << ",\"node_slots\":[";
    bool first = true;
    for (const auto& entry : solution.node_slots.entries()) {
        if (!first) {
            output.put(',');
        }
        first = false;
        output << '[' << entry.key << ',' << entry.value << ']';
    }
    output << "],\"link_paths\":[";
    first = true;
    for (const auto& entry : solution.link_paths.entries()) {
        if (!first) {
            output.put(',');
        }
        first = false;
        output << "{\"virtual\":[" << entry.key.source << ','
               << entry.key.target << "],\"path\":[";
        bool first_link = true;
        for (const auto link : entry.value) {
            if (!first_link) {
                output.put(',');
            }
            first_link = false;
            output << '[' << link.source << ',' << link.target << ']';
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

struct CommandLine {
    bool benchmark = false;
    std::size_t workers = 1U;
    std::size_t warmups = 1U;
    std::size_t repetitions = 3U;
    std::size_t iterations = 2U;
};

std::size_t parse_size(const char* value, std::string_view option) {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != std::string_view(value).size()) {
        throw std::invalid_argument(
            std::string(option) + " requires an unsigned integer");
    }
    return static_cast<std::size_t>(parsed);
}

CommandLine parse_command_line(int argc, char** argv) {
    CommandLine result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--benchmark") {
            result.benchmark = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(
                std::string(option) + " requires a value");
        }
        ++index;
        if (option == "--workers") {
            result.workers = parse_size(argv[index], option);
        } else if (option == "--warmups") {
            result.warmups = parse_size(argv[index], option);
        } else if (option == "--repetitions") {
            result.repetitions = parse_size(argv[index], option);
        } else if (option == "--iterations") {
            result.iterations = parse_size(argv[index], option);
        } else {
            throw std::invalid_argument(
                "unknown harness option: " + std::string(option));
        }
    }
    if (argc == 1 || result.repetitions == 0U || result.iterations == 0U) {
        throw std::invalid_argument(
            "usage: heuristic_node_rank_variants_harness "
            "[--benchmark] --workers N [--warmups N] "
            "[--repetitions N] [--iterations N]");
    }
    return result;
}

std::string verification_json(
    const std::vector<core::Solution>& solutions,
    const std::optional<std::uint32_t>& rng_next) {
    std::ostringstream output;
    output << "{\"solutions\":[";
    for (std::size_t index = 0U; index < solutions.size(); ++index) {
        if (index != 0U) {
            output.put(',');
        }
        output << solution_json(solutions[index]);
    }
    output << "],\"rng_next\":";
    if (rng_next.has_value()) {
        output << *rng_next;
    } else {
        output << "null";
    }
    output.put('}');
    return output.str();
}

int run_differential(std::size_t width) {
    const heuristic::NodeRankSolverWorkers workers{
        width, width, width, width};
    RuntimeFixture runtime;
    NumpyRandomState random(77U);
    solver::SolverRegistry registry;
    const auto ids = register_all_solvers(registry, random, workers);
    registry.freeze();

    std::cout
        << "{\"component\":\"solver.heuristic.node_rank.variants\"," 
           "\"workers\":"
        << width
        << ",\"native_input_physical_unchanged\":true,\"solvers\":[";
    for (std::size_t index = 0U; index < ids.size(); ++index) {
        auto virtual_network = make_virtual_network();
        auto physical_network = make_physical_network();
        const auto before = snapshot(physical_network);
        const auto& descriptor = registry.descriptor(ids[index]);
        auto instance = registry.create(
            ids[index],
            runtime.dependencies(),
            solver_config(runtime.cleanup.path));
        const core::Solution solution = instance->solve({
            virtual_network,
            physical_network,
        });
        if (!(snapshot(physical_network) == before)) {
            throw std::runtime_error(
                "variant mutated the const physical input");
        }
        if (index != 0U) {
            std::cout.put(',');
        }
        std::cout << "{\"solver\":" << json_string(descriptor.name)
                  << ",\"solver_id\":" << descriptor.id.value
                  << ",\"category\":\"node_ranking\",\"solution\":"
                  << solution_json(solution) << ",\"rng_next\":";
        if (descriptor.name == "random_rank") {
            std::cout << random.next_uint32();
        } else {
            std::cout << "null";
        }
        std::cout.put('}');
    }
    std::cout << "],\"sparse_candidate_ties\":[";
    constexpr std::array<std::size_t, 2U> sparse_solver_indexes{{5U, 6U}};
    for (std::size_t output_index = 0U;
         output_index < sparse_solver_indexes.size();
         ++output_index) {
        const std::size_t solver_index =
            sparse_solver_indexes[output_index];
        auto virtual_network = make_sparse_tie_virtual_network();
        auto physical_network = make_sparse_tie_physical_network();
        const auto before = snapshot(physical_network);
        const auto& descriptor = registry.descriptor(ids[solver_index]);
        auto instance = registry.create(
            ids[solver_index],
            runtime.dependencies(),
            solver_config(runtime.cleanup.path));
        const core::Solution solution = instance->solve({
            virtual_network,
            physical_network,
        });
        if (!(snapshot(physical_network) == before)) {
            throw std::runtime_error(
                "sparse candidate tie mutated the const physical input");
        }
        if (output_index != 0U) {
            std::cout.put(',');
        }
        std::cout << "{\"solver\":" << json_string(descriptor.name)
                  << ",\"solution\":" << solution_json(solution) << '}';
    }
    std::cout << "]}\n";
    return 0;
}

int run_benchmark(const CommandLine& command) {
    const heuristic::NodeRankSolverWorkers workers{
        command.workers,
        command.workers,
        command.workers,
        command.workers};
    RuntimeFixture runtime;
    NumpyRandomState random(77U);
    solver::SolverRegistry registry;
    const auto ids = register_all_solvers(registry, random, workers);
    registry.freeze();

    std::cout
        << "{\"component\":\"solver.heuristic.node_rank.variants\"," 
           "\"mode\":\"benchmark\",\"workers\":"
        << command.workers << ",\"warmups\":" << command.warmups
        << ",\"repetitions\":" << command.repetitions
        << ",\"iterations\":" << command.iterations
        << ",\"solvers\":[";

    for (std::size_t solver_index = 0U;
         solver_index < ids.size();
         ++solver_index) {
        const auto& descriptor = registry.descriptor(ids[solver_index]);
        std::vector<std::uint64_t> samples;
        samples.reserve(command.repetitions);
        std::optional<std::string> expected_verification;
        const std::size_t sample_count =
            command.warmups + command.repetitions;

        for (std::size_t sample = 0U; sample < sample_count; ++sample) {
            random.seed(77U);
            auto instance = registry.create(
                ids[solver_index],
                runtime.dependencies(),
                solver_config(runtime.cleanup.path));
            instance->ready();
            auto virtual_network = make_benchmark_virtual_network();
            auto physical_network = make_benchmark_physical_network();
            const auto before = snapshot(physical_network);
            std::vector<core::Solution> solutions;
            solutions.reserve(command.iterations);

            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t iteration = 0U;
                 iteration < command.iterations;
                 ++iteration) {
                solutions.push_back(instance->solve({
                    virtual_network,
                    physical_network,
                }));
            }
            const auto end = std::chrono::steady_clock::now();
            if (!(snapshot(physical_network) == before)) {
                throw std::runtime_error(
                    "benchmark solve mutated the const physical input");
            }

            std::optional<std::uint32_t> rng_next;
            if (descriptor.name == "random_rank") {
                rng_next = random.next_uint32();
            }
            const std::string observed =
                verification_json(solutions, rng_next);
            if (sample >= command.warmups) {
                if (!expected_verification.has_value()) {
                    expected_verification = observed;
                } else if (*expected_verification != observed) {
                    throw std::runtime_error(
                        "benchmark output changed between samples");
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin);
                samples.push_back(
                    static_cast<std::uint64_t>(elapsed.count()));
            }
        }

        if (solver_index != 0U) {
            std::cout.put(',');
        }
        std::cout << "{\"solver\":" << json_string(descriptor.name)
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < samples.size(); ++index) {
            if (index != 0U) {
                std::cout.put(',');
            }
            std::cout << samples[index];
        }
        std::cout << "],\"verification\":"
                  << *expected_verification << '}';
    }
    std::cout << "]}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CommandLine command = parse_command_line(argc, argv);
        return command.benchmark
            ? run_benchmark(command)
            : run_differential(command.workers);
    } catch (const std::exception& error) {
        std::cerr << "heuristic node-rank variants harness: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
