#include "../virne/solver/heuristic/node_rank.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/core/solution.h"
#include "../virne/network/attribute/attribute_factory.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace heuristic = virne::solver::heuristic;
namespace network = virne::network;
namespace solver = virne::solver;

constexpr std::size_t virtual_node_count = 16U;
constexpr std::size_t physical_node_count = 32U;
constexpr attribute::AttributeRegistryId node_constraint_id{0U};
constexpr attribute::AttributeRegistryId link_constraint_id{0U};

std::filesystem::path unique_temp_root() {
    static std::atomic<std::uint64_t> next_id{0U};
    return std::filesystem::temp_directory_path() /
           ("virne-heuristic-node-rank-benchmark-" +
            std::to_string(next_id.fetch_add(1U, std::memory_order_relaxed)));
}

class CleanupRoot {
public:
    explicit CleanupRoot(std::filesystem::path root) : root_(std::move(root)) {}

    CleanupRoot(const CleanupRoot&) = delete;
    CleanupRoot& operator=(const CleanupRoot&) = delete;

    ~CleanupRoot() {
        std::error_code error;
        const auto temporary_root =
            std::filesystem::temp_directory_path(error);
        const std::string filename = root_.filename().string();
        if (error || root_.empty() || root_.parent_path() != temporary_root ||
            filename.rfind("virne-heuristic-node-rank-benchmark-", 0U) !=
                0U) {
            return;
        }
        error.clear();
        std::filesystem::remove_all(root_, error);
    }

private:
    std::filesystem::path root_;
};

core::RecorderConfig recorder_config(const std::filesystem::path& root) {
    core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-node-rank-benchmark";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

core::LoggerConfig logger_config(const std::filesystem::path& root) {
    core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "heuristic-node-rank-benchmark";
    config.run_id = "logger";
    config.log_dir_name = "logs";
    config.log_file_name = "benchmark.log";
    config.backends.console = false;
    config.backends.file = false;
    config.level = core::LoggerLevel::critical;
    return config;
}

struct RuntimeFixture {
    explicit RuntimeFixture(const std::filesystem::path& root)
        : controller([] {
              controller::ControllerSelection selection;
              selection.constraints.node_at_node = {node_constraint_id};
              selection.constraints.link_at_link = {link_constraint_id};
              selection.hard_node_constraints = {node_constraint_id};
              selection.hard_link_constraints = {link_constraint_id};
              return selection;
          }()),
          counter(core::CounterSelection{}),
          recorder(core::Counter(core::CounterSelection{}), recorder_config(root)),
          logger(logger_config(root)) {}

    solver::SolverDependencies dependencies() {
        return solver::SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger),
        };
    }

    controller::Controller controller;
    core::Counter counter;
    core::Recorder recorder;
    core::Logger logger;
};

attribute::AttributeFactorySpec node_constraint_spec() {
    attribute::AttributeFactorySpec spec;
    spec.name = "cpu";
    spec.owner = attribute::AttributeOwner::node;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::node;
    return spec;
}

attribute::AttributeFactorySpec link_constraint_spec() {
    attribute::AttributeFactorySpec spec;
    spec.name = "bandwidth";
    spec.owner = attribute::AttributeOwner::link;
    spec.kind = attribute::AttributeKind::resource;
    spec.restriction = attribute::ConstraintRestriction::hard;
    spec.checking_level = attribute::CheckingLevel::link;
    return spec;
}

std::vector<AttrValue> repeated_values(
    std::size_t count,
    std::int64_t value) {
    std::vector<AttrValue> values;
    values.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        values.emplace_back(value);
    }
    return values;
}

network::NodeAttributeDataUpdate node_constraint_values(
    std::size_t count,
    std::int64_t value) {
    network::NodeAttributeDataUpdate update;
    update.registry_id = node_constraint_id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = repeated_values(count, value);
    return update;
}

network::LinkAttributeDataUpdate link_constraint_values(
    std::size_t count,
    std::int64_t value) {
    network::LinkAttributeDataUpdate update;
    update.registry_id = link_constraint_id;
    update.layout = network::AttributeDataLayout::dense;
    update.dense_values = repeated_values(count, value);
    return update;
}

std::vector<EdgeEndpoints> virtual_edges() {
    // Already grouped in the public Graph/NetworkX edge-view order.
    return {
        {0U, 2U},
        {0U, 5U},
        {1U, 3U},
        {2U, 4U},
        {3U, 5U},
        {4U, 6U},
        {5U, 7U},
        {5U, 10U},
        {6U, 8U},
        {7U, 9U},
        {8U, 10U},
        {9U, 11U},
        {10U, 12U},
        {10U, 15U},
        {11U, 13U},
        {12U, 14U},
        {13U, 15U},
    };
}

std::vector<EdgeEndpoints> physical_edges() {
    std::vector<EdgeEndpoints> edges;
    edges.reserve(physical_node_count - 1U);
    for (Vertex source = 0U;
         source < static_cast<Vertex>(physical_node_count - 1U);
         ++source) {
        edges.push_back(EdgeEndpoints{source, source + 1U});
    }
    return edges;
}

network::VirtualNetwork make_virtual_network(
    std::vector<EdgeEndpoints> edges) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(virtual_node_count, std::move(edges));
    construction.config.node_attribute_specs = {node_constraint_spec()};
    construction.config.link_attribute_specs = {link_constraint_spec()};
    network::VirtualNetwork result(std::move(construction));
    result.set_node_attrs_data(
        {node_constraint_values(virtual_node_count, 1)});
    result.set_link_attrs_data(
        {link_constraint_values(virtual_edges().size(), 1)});
    result.set_request_id(17);
    result.set_arrival_time(2.0);
    result.set_lifetime(10.0);
    return result;
}

network::PhysicalNetwork make_physical_network(
    std::vector<EdgeEndpoints> edges) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(physical_node_count, std::move(edges));
    construction.config.node_attribute_specs = {node_constraint_spec()};
    construction.config.link_attribute_specs = {link_constraint_spec()};
    network::PhysicalNetwork result(std::move(construction));
    result.set_node_attrs_data(
        {node_constraint_values(physical_node_count, 1'000)});
    result.set_link_attrs_data(
        {link_constraint_values(physical_node_count - 1U, 1'000)});
    return result;
}

struct PreparedNetworks {
    network::VirtualNetwork virtual_network;
    network::PhysicalNetwork physical_network;
    std::vector<EdgeEndpoints> expected_virtual_edges;
};

PreparedNetworks prepare_networks() {
    auto expected_edges = virtual_edges();
    auto virtual_network = make_virtual_network(expected_edges);
    auto physical_network = make_physical_network(physical_edges());
    return PreparedNetworks{
        std::move(virtual_network),
        std::move(physical_network),
        std::move(expected_edges),
    };
}

solver::SolverConfig solver_config(const std::filesystem::path& root) {
    solver::SolverConfig config;
    config.seed = 0U;
    config.verbose = 0;
    config.save_dir = root / "solver";
    config.reusable = false;
    config.matching_method = controller::NodeMatchingMethod::greedy;
    config.shortest_method = controller::ShortestPathMethod::bfs_shortest;
    config.k_shortest = 1;
    config.allow_rejection = false;
    config.allow_revocable = false;
    return config;
}

std::size_t parse_size(const char* text, const char* label, bool allow_zero) {
    std::size_t parsed_count = 0U;
    const auto value = std::stoull(text, &parsed_count, 10);
    if (text[parsed_count] != '\0' || (!allow_zero && value == 0U)) {
        throw std::invalid_argument(std::string(label) + " must be a positive integer");
    }
    return static_cast<std::size_t>(value);
}

void validate_solution(
    const core::Solution& solution,
    const std::vector<EdgeEndpoints>& expected_edges) {
    if (!solution.result || !solution.place_result || !solution.route_result) {
        throw std::runtime_error("OrderRank did not produce a successful solution");
    }

    const auto& slots = solution.node_slots.entries();
    if (slots.size() != virtual_node_count) {
        throw std::runtime_error("unexpected node-slot count");
    }
    for (std::size_t index = 0U; index < slots.size(); ++index) {
        const auto expected = static_cast<std::int64_t>(index);
        if (slots[index].key != expected || slots[index].value != expected) {
            throw std::runtime_error("node slots are not in exact OrderRank order");
        }
    }

    const auto& routed = solution.link_paths.entries();
    if (routed.size() != expected_edges.size()) {
        throw std::runtime_error("unexpected routed-link count");
    }
    for (std::size_t link_index = 0U; link_index < routed.size(); ++link_index) {
        const auto expected_link = expected_edges[link_index];
        const auto& actual = routed[link_index];
        if (actual.key.source != expected_link.first ||
            actual.key.target != expected_link.second) {
            throw std::runtime_error("routed links are not in exact graph order");
        }

        const auto expected_length = static_cast<std::size_t>(
            expected_link.second - expected_link.first);
        if (actual.value.size() != expected_length) {
            throw std::runtime_error("unexpected shortest-path length");
        }
        for (std::size_t path_index = 0U; path_index < expected_length;
             ++path_index) {
            const auto expected_source =
                expected_link.first + static_cast<Vertex>(path_index);
            const auto& actual_edge = actual.value[path_index];
            if (actual_edge.source != expected_source ||
                actual_edge.target != expected_source + 1) {
                throw std::runtime_error("shortest-path edge order differs");
            }
        }
    }
}

class HashWriter {
public:
    void byte(std::uint8_t value) noexcept {
        checksum_ ^= static_cast<std::uint64_t>(value);
        checksum_ *= fnv_prime;
        ++byte_count_;
    }

    void boolean(bool value) noexcept {
        byte(value ? std::uint8_t{1U} : std::uint8_t{0U});
    }

    void u64(std::uint64_t value) noexcept {
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            byte(static_cast<std::uint8_t>((value >> shift) & 0xffU));
        }
    }

    void i64(std::int64_t value) noexcept {
        u64(static_cast<std::uint64_t>(value));
    }

    std::size_t byte_count() const noexcept { return byte_count_; }
    std::uint64_t checksum() const noexcept { return checksum_; }

private:
    static constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
    static constexpr std::uint64_t fnv_prime = 1099511628211ULL;

    std::uint64_t checksum_ = fnv_offset;
    std::size_t byte_count_ = 0U;
};

void append_solution(HashWriter& writer, const core::Solution& solution) {
    writer.boolean(solution.result);
    writer.boolean(solution.place_result);
    writer.boolean(solution.route_result);

    const auto& slots = solution.node_slots.entries();
    writer.u64(static_cast<std::uint64_t>(slots.size()));
    for (const auto& entry : slots) {
        writer.i64(entry.key);
        writer.i64(entry.value);
    }

    const auto& paths = solution.link_paths.entries();
    writer.u64(static_cast<std::uint64_t>(paths.size()));
    for (const auto& entry : paths) {
        writer.i64(entry.key.source);
        writer.i64(entry.key.target);
        writer.u64(static_cast<std::uint64_t>(entry.value.size()));
        for (const auto& edge : entry.value) {
            writer.i64(edge.source);
            writer.i64(edge.target);
        }
    }
}

struct OutputGate {
    std::size_t entries = 0U;
    std::size_t bytes = 0U;
    std::uint64_t checksum = 0U;

    friend bool operator==(
        const OutputGate& left,
        const OutputGate& right) noexcept {
        return left.entries == right.entries && left.bytes == right.bytes &&
               left.checksum == right.checksum;
    }
};

OutputGate gate_outputs(
    const std::vector<core::Solution>& outputs,
    const std::vector<EdgeEndpoints>& expected_edges) {
    HashWriter writer;
    writer.u64(static_cast<std::uint64_t>(outputs.size()));
    for (const auto& output : outputs) {
        validate_solution(output, expected_edges);
        append_solution(writer, output);
    }
    return OutputGate{outputs.size(), writer.byte_count(), writer.checksum()};
}

struct SolveBatch {
    std::uint64_t elapsed_ns = 0U;
    std::vector<core::Solution> outputs;
};

SolveBatch solve_batch(
    solver::Solver& order_solver,
    const solver::SolverInstance& instance,
    std::size_t repetitions,
    bool measure) {
    SolveBatch batch;
    batch.outputs.reserve(repetitions);
    for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
        if (!measure) {
            batch.outputs.push_back(order_solver.solve(instance));
            continue;
        }

        const auto begin = std::chrono::steady_clock::now();
        auto solution = order_solver.solve(instance);
        const auto end = std::chrono::steady_clock::now();
        batch.elapsed_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        batch.outputs.push_back(std::move(solution));
    }
    return batch;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

void print_result(
    std::size_t workers,
    const OutputGate& gate,
    const std::vector<std::uint64_t>& samples) {
    std::cout << "{\"workers\":" << workers << ",\"entries\":" << gate.entries
              << ",\"bytes\":" << gate.bytes << ",\"checksum\":"
              << gate.checksum << ",\"median_ns\":" << median(samples)
              << ",\"samples_ns\":[";
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        std::cout << samples[index];
    }
    std::cout << "]}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            throw std::invalid_argument(
                "usage: heuristic_node_rank_benchmark "
                "WORKERS WARMUPS SAMPLES REPETITIONS");
        }

        const auto workers = parse_size(argv[1], "workers", false);
        const auto warmups = parse_size(argv[2], "warmups", true);
        const auto sample_count = parse_size(argv[3], "samples", false);
        const auto repetitions = parse_size(argv[4], "repetitions", false);

        const auto root = unique_temp_root();
        CleanupRoot cleanup(root);
        RuntimeFixture runtime(root);
        auto networks = prepare_networks();

        heuristic::NodeRankSolverWorkers worker_config;
        worker_config.rank_workers = workers;
        worker_config.node_candidate_workers = workers;
        worker_config.link_topology_constraint_workers = workers;
        worker_config.link_candidate_workers = workers;

        solver::SolverRegistry registry;
        const auto solver_id =
            heuristic::register_order_rank_solver(registry, worker_config);
        registry.freeze();
        auto order_solver = registry.create(
            solver_id, runtime.dependencies(), solver_config(root));
        const solver::SolverInstance instance{
            networks.virtual_network,
            networks.physical_network,
        };

        for (std::size_t warmup = 0U; warmup < warmups; ++warmup) {
            auto batch = solve_batch(*order_solver, instance, repetitions, false);
            static_cast<void>(
                gate_outputs(batch.outputs, networks.expected_virtual_edges));
        }

        std::vector<std::uint64_t> samples;
        samples.reserve(sample_count);
        OutputGate reference_gate;
        bool have_reference = false;
        for (std::size_t sample = 0U; sample < sample_count; ++sample) {
            auto batch = solve_batch(*order_solver, instance, repetitions, true);
            const auto gate =
                gate_outputs(batch.outputs, networks.expected_virtual_edges);
            if (have_reference && !(gate == reference_gate)) {
                throw std::runtime_error("native output gate changed between samples");
            }
            reference_gate = gate;
            have_reference = true;
            samples.push_back(batch.elapsed_ns);
        }

        print_result(workers, reference_gate, samples);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "heuristic_node_rank_benchmark: " << error.what() << '\n';
        return 1;
    }
}
