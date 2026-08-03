#include "../virne/solver/meta_heuristic/meta_heuristic.h"

#include "../random/py_random.h"
#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace meta = virne::solver::meta;
namespace network = virne::network;
namespace solver = virne::solver;

constexpr attribute::AttributeRegistryId resource_id{0U};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

attribute::AttributeFactorySpec node_resource_spec() {
    attribute::AttributeFactorySpec spec;
    spec.name = "cpu";
    spec.owner = attribute::AttributeOwner::node;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::node;
    return spec;
}

attribute::AttributeFactorySpec link_resource_spec() {
    attribute::AttributeFactorySpec spec;
    spec.name = "bandwidth";
    spec.owner = attribute::AttributeOwner::link;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::link;
    return spec;
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
    result.registry_id = resource_id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = values(input);
    return result;
}

network::LinkAttributeDataUpdate link_update(
    const std::vector<std::int64_t>& input) {
    network::LinkAttributeDataUpdate result;
    result.registry_id = resource_id;
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
    result.set_request_id(17);
    result.set_arrival_time(0.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_physical_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {0U, 2U}, {0U, 3U},
                                   {1U, 2U}, {1U, 3U}, {2U, 3U}});
    construction.config.node_attribute_specs = {node_resource_spec()};
    construction.config.link_attribute_specs = {link_resource_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data({node_update({20, 18, 16, 14})});
    result.set_link_attrs_data({link_update({20, 20, 20, 20, 20, 20})});
    return result;
}

controller::Controller make_controller() {
    controller::ControllerSelection selection;
    selection.constraints.node_at_node = {resource_id};
    selection.constraints.link_at_link = {resource_id};
    selection.node_resources = {resource_id};
    selection.link_resources = {resource_id};
    selection.hard_node_constraints = {resource_id};
    selection.hard_link_constraints = {resource_id};
    return controller::Controller(std::move(selection));
}

core::RecorderConfig recorder_config(
    const std::filesystem::path& root) {
    core::RecorderConfig result;
    result.save_root_dir = root;
    result.solver_name = "meta-unit";
    result.run_id = "unit";
    result.temporary_records = false;
    return result;
}

core::LoggerConfig logger_config() {
    core::LoggerConfig result;
    result.save_root_dir =
        std::filesystem::temp_directory_path() / "virne-meta-unit";
    result.solver_name = "meta-unit";
    result.run_id = "logger";
    result.log_dir_name = "logs";
    result.log_file_name = "run.log";
    result.backends.console = false;
    result.backends.file = false;
    result.level = core::LoggerLevel::critical;
    return result;
}

struct Runtime {
    std::filesystem::path root =
        std::filesystem::temp_directory_path() / "virne-meta-unit";
    controller::Controller controller = make_controller();
    core::Counter counter{core::CounterSelection{}};
    core::Recorder recorder{core::Counter(core::CounterSelection{}),
                            recorder_config(root)};
    core::Logger logger{logger_config()};

    ~Runtime() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    solver::SolverDependencies dependencies() {
        return solver::SolverDependencies{
            std::cref(controller), std::ref(recorder),
            std::cref(counter), std::ref(logger)};
    }
};

solver::SolverConfig solver_config() {
    solver::SolverConfig result;
    result.verbose = 0;
    result.matching_method = controller::NodeMatchingMethod::greedy;
    result.shortest_method = controller::ShortestPathMethod::bfs_shortest;
    result.k_shortest = 10;
    return result;
}

std::vector<std::int64_t> signature(const core::Solution& solution) {
    std::vector<std::int64_t> result{
        solution.result ? 1 : 0,
        solution.place_result ? 1 : 0,
        solution.route_result ? 1 : 0};
    for (const auto& entry : solution.node_slots.entries()) {
        result.push_back(entry.key);
        result.push_back(entry.value);
    }
    for (const auto& entry : solution.link_paths.entries()) {
        result.push_back(entry.key.source);
        result.push_back(entry.key.target);
        result.push_back(static_cast<std::int64_t>(entry.value.size()));
        for (const auto link : entry.value) {
            result.push_back(link.source);
            result.push_back(link.target);
        }
    }
    return result;
}

struct Run {
    std::array<std::vector<std::int64_t>, 5U> signatures;
    std::uint32_t continuation = 0U;
    double milliseconds = 0.0;
};

Run run(std::size_t workers) {
    Runtime runtime;
    PyRandom random(0U);
    solver::SolverRegistry registry;
    meta::MetaHeuristicOptions options;
    options.population_size = 4U;
    options.max_iterations = 3U;
    options.max_attempts = 2U;
    options.evaluation_workers = workers;
    options.candidate_workers = workers;
    options.topology_constraint_workers = workers;
    const auto ids = meta::register_meta_heuristic_solvers(
        registry, random, options);
    const std::array<std::pair<solver::SolverId, std::string_view>, 5U> expected{{
        {ids.ga_meta, "ga_meta"}, {ids.sa_meta, "sa_meta"},
        {ids.ts_meta, "ts_meta"}, {ids.pso_meta, "pso_meta"},
        {ids.aco_meta, "aco_meta"}}};
    require(registry.size() == expected.size(), "meta registry size mismatch");
    registry.freeze();
    for (const auto& entry : expected) {
        require(registry.resolve(entry.second) == entry.first,
                "meta registry name mismatch");
        require(registry.descriptor(entry.first).category ==
                    solver::SolverCategory::meta_heuristic,
                "meta registry category mismatch");
    }
    Run result;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        auto virtual_network = make_virtual_network();
        auto physical_network = make_physical_network();
        auto instance = registry.create(
            expected[index].first, runtime.dependencies(), solver_config());
        require(instance != nullptr, "meta factory returned null");
        const core::Solution solution = instance->solve(
            solver::SolverInstance{virtual_network, physical_network});
        require(solution.result && solution.place_result && solution.route_result,
                "meta solver rejected feasible request");
        result.signatures[index] = signature(solution);
    }
    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.continuation = random.genrand_uint32();
    return result;
}

} // namespace

int main() {
    try {
        const Run sequential = run(1U);
        const Run parallel = run(4U);
        require(sequential.signatures == parallel.signatures,
                "meta worker width changed output");
        require(sequential.continuation == parallel.continuation,
                "meta worker width changed RNG stream");
        std::cout << "meta timing: workers=1 " << sequential.milliseconds
                  << " ms, workers=4 " << parallel.milliseconds << " ms\n";
        std::cout << "meta heuristic unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "meta heuristic unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
