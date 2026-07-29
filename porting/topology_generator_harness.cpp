#include "topology/topology_generator.h"

#include "py_random.h"
#include "random_context.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using virne::network::TopologyGenerator;
using virne::network::TopologyOptions;
using virne::network::TopologyRequest;
using virne::network::TopologyType;

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::size_t kPathNodes = 65536;
constexpr std::size_t kStarNodes = 4096;
constexpr std::size_t kGridRows = 256;
constexpr std::size_t kGridColumns = 256;
constexpr std::size_t kRandomNodes = 500;
constexpr std::size_t kWaxmanNodes = 500;

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string hex_u64(std::uint64_t value, int width = 16)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(width) << value;
    return stream.str();
}

std::string encode_graph(const Graph& graph)
{
    std::ostringstream stream;
    stream << "N:";
    auto [node, node_end] = graph.nodes();
    bool first = true;
    for (; node != node_end; ++node)
    {
        if (!first)
        {
            stream << ',';
        }
        first = false;
        stream << *node;
    }

    stream << ";E:";
    auto [edge, edge_end] = graph.edges();
    first = true;
    for (; edge != edge_end; ++edge)
    {
        if (!first)
        {
            stream << ',';
        }
        first = false;
        stream << graph.source(*edge) << '-' << graph.target(*edge);
    }

    stream << ";Q:";
    for (Vertex vertex = 0; vertex < graph.num_nodes(); ++vertex)
    {
        if (vertex != 0)
        {
            stream << '/';
        }
        stream << vertex << ':';
        const auto [neighbor, neighbor_end] = graph.neighbors(vertex);
        bool first_neighbor = true;
        for (auto current = neighbor; current != neighbor_end; ++current)
        {
            if (!first_neighbor)
            {
                stream << ',';
            }
            first_neighbor = false;
            stream << *current;
        }
    }

    stream << ";P:";
    first = true;
    std::size_t node_attributes = 0;
    const std::optional<AttrId> pos_id =
        graph.attribute_registry().find("pos");
    for (Vertex vertex = 0; vertex < graph.num_nodes(); ++vertex)
    {
        node_attributes += graph.node_attrs(vertex).size();
        const AttrValue* value = pos_id.has_value()
            ? graph.node_attrs(vertex).find(*pos_id)
            : nullptr;
        if (value == nullptr)
        {
            continue;
        }
        const AttrList* position = attr_list(*value);
        if (position == nullptr || position->values.size() != 2)
        {
            throw std::runtime_error("invalid Waxman pos attribute");
        }
        if (!first)
        {
            stream << ',';
        }
        first = false;
        stream << vertex << ':'
               << hex_u64(double_bits(attr_to_double(position->values[0])))
               << ':'
               << hex_u64(double_bits(attr_to_double(position->values[1])));
    }

    std::size_t edge_attributes = 0;
    auto [attribute_edge, attribute_edge_end] = graph.edges();
    for (; attribute_edge != attribute_edge_end; ++attribute_edge)
    {
        edge_attributes += graph.edge_attrs(*attribute_edge).size();
    }
    stream << ";A:" << graph.graph_attrs().size() << ','
           << node_attributes << ',' << edge_attributes;
    return stream.str();
}

std::string encode_rng(PyRandom& random)
{
    std::ostringstream stream;
    for (int index = 0; index < 8; ++index)
    {
        if (index != 0)
        {
            stream << ',';
        }
        stream << hex_u64(random.getrandbits32(), 8);
    }
    return stream.str();
}

std::string encode_graphs(const std::vector<Graph>& graphs)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < graphs.size(); ++index)
    {
        if (index != 0)
        {
            stream << "||";
        }
        stream << encode_graph(graphs[index]);
    }
    return stream.str();
}

template <typename Action>
std::string error_and_state(std::uint64_t seed, Action&& action)
{
    PyRandom random(seed);
    try
    {
        action(random);
        return "NO_ERROR|R:" + encode_rng(random);
    }
    catch (const std::exception& error)
    {
        return "ERROR:" + std::string(error.what()) +
               "|R:" + encode_rng(random);
    }
}

template <typename Action>
std::string error_flag_and_state(std::uint64_t seed, Action&& action)
{
    PyRandom random(seed);
    try
    {
        action(random);
        return "0|R:" + encode_rng(random);
    }
    catch (...)
    {
        return "1|R:" + encode_rng(random);
    }
}

void emit_parity(std::string_view key, const std::string& value)
{
    std::cout << "PARITY\t" << key << '\t' << value << '\n';
}

std::string deterministic_case(
    std::string_view type,
    std::int64_t num_nodes,
    const TopologyOptions& options = {})
{
    PyRandom random(91234567);
    const Graph graph =
        TopologyGenerator::generate(type, num_nodes, options, random);
    return encode_graph(graph) + "|R:" + encode_rng(random);
}

std::string stochastic_sequence(
    std::string_view type,
    std::int64_t num_nodes,
    const TopologyOptions& options,
    std::uint64_t seed,
    std::size_t count)
{
    PyRandom random(seed);
    std::vector<Graph> graphs;
    graphs.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        graphs.push_back(
            TopologyGenerator::generate(
                type, num_nodes, options, random));
    }
    return encode_graphs(graphs) + "|R:" + encode_rng(random);
}

TopologyOptions grid_options(std::int64_t rows, std::int64_t columns)
{
    TopologyOptions options;
    options.m = rows;
    options.n = columns;
    return options;
}

TopologyOptions random_options(double probability)
{
    TopologyOptions options;
    options.random_prob = probability;
    return options;
}

TopologyOptions waxman_options(double wm_alpha, double wm_beta)
{
    TopologyOptions options;
    options.wm_alpha = wm_alpha;
    options.wm_beta = wm_beta;
    return options;
}

std::vector<TopologyRequest> parity_batch_requests()
{
    return {
        {TopologyType::Path, 7, {}, 3},
        {TopologyType::Star, 6, {}, 5},
        {TopologyType::Grid2D, 999, grid_options(3, 4), 7},
        {TopologyType::Random, 10, random_options(0.23), 0},
        {TopologyType::Random, 10, random_options(0.23), 2},
        {TopologyType::Waxman, 12, waxman_options(0.8, 0.45), 15},
        {TopologyType::Waxman, 12, waxman_options(0.8, 0.45), 1},
    };
}

void emit_parity_cases()
{
    emit_parity("path_1", deterministic_case("path", 1));
    emit_parity("path_2", deterministic_case("path", 2));
    emit_parity("path_7", deterministic_case("path", 7));
    emit_parity("star_1", deterministic_case("star", 1));
    emit_parity("star_2", deterministic_case("star", 2));
    emit_parity("star_6", deterministic_case("star", 6));
    emit_parity(
        "grid_1x1", deterministic_case("grid_2d", 1, grid_options(1, 1)));
    emit_parity(
        "grid_1x4", deterministic_case("grid_2d", 1, grid_options(1, 4)));
    emit_parity(
        "grid_3x4", deterministic_case("grid_2d", 1, grid_options(3, 4)));
    emit_parity(
        "grid_3x4_ignored_num_nodes",
        deterministic_case("grid_2d", 999, grid_options(3, 4)));
    emit_parity(
        "grid_0x4", deterministic_case("grid_2d", 1, grid_options(0, 4)));
    emit_parity(
        "grid_mapping_3x4",
        "0,0,0;1,0,1;2,0,2;3,0,3;4,1,0;5,1,1;6,1,2;"
        "7,1,3;8,2,0;9,2,1;10,2,2;11,2,3");

    emit_parity(
        "error_num_nodes_zero",
        error_and_state(11, [](PyRandom& random)
        {
            static_cast<void>(
                TopologyGenerator::generate("path", 0, {}, random));
        }));
    emit_parity(
        "error_num_nodes_negative",
        error_and_state(11, [](PyRandom& random)
        {
            static_cast<void>(
                TopologyGenerator::generate("path", -1, {}, random));
        }));
    emit_parity(
        "error_unsupported",
        error_and_state(11, [](PyRandom& random)
        {
            static_cast<void>(
                TopologyGenerator::generate("PATH", 3, {}, random));
        }));
    emit_parity(
        "error_grid_missing_both",
        error_and_state(11, [](PyRandom& random)
        {
            static_cast<void>(
                TopologyGenerator::generate("grid_2d", 1, {}, random));
        }));
    emit_parity(
        "error_grid_missing_m",
        error_and_state(11, [](PyRandom& random)
        {
            TopologyOptions options;
            options.n = 4;
            static_cast<void>(
                TopologyGenerator::generate("grid_2d", 1, options, random));
        }));
    emit_parity(
        "error_grid_missing_n",
        error_and_state(11, [](PyRandom& random)
        {
            TopologyOptions options;
            options.m = 3;
            static_cast<void>(
                TopologyGenerator::generate("grid_2d", 1, options, random));
        }));
    emit_parity(
        "error_grid_negative",
        error_and_state(11, [](PyRandom& random)
        {
            static_cast<void>(TopologyGenerator::generate(
                "grid_2d", 1, grid_options(-1, 4), random));
        }));

    const TopologyOptions defaults;
    emit_parity(
        "random_default_seed0_two",
        stochastic_sequence("random", 10, defaults, 0, 2));
    emit_parity(
        "random_retry_seed0",
        stochastic_sequence("random", 10, random_options(0.23), 0, 1));
    emit_parity(
        "random_retry_seed2",
        stochastic_sequence("random", 10, random_options(0.23), 2, 1));
    emit_parity(
        "random_n1_p0",
        stochastic_sequence("random", 1, random_options(0.0), 42, 1));
    emit_parity(
        "random_n8_p1",
        stochastic_sequence("random", 8, random_options(1.0), 42, 1));

    emit_parity(
        "waxman_default_seed0",
        stochastic_sequence("waxman", 50, defaults, 0, 1));
    emit_parity(
        "waxman_retry_seed15",
        stochastic_sequence(
            "waxman", 12, waxman_options(0.8, 0.45), 15, 1));
    emit_parity(
        "waxman_retry_seed1",
        stochastic_sequence(
            "waxman", 12, waxman_options(0.8, 0.45), 1, 1));
    emit_parity(
        "waxman_sequence_seed42",
        stochastic_sequence(
            "waxman", 12, waxman_options(0.8, 0.45), 42, 2));
    emit_parity(
        "waxman_n1_error",
        error_flag_and_state(7, [](PyRandom& random)
        {
            static_cast<void>(
                TopologyGenerator::generate("waxman", 1, {}, random));
        }));

    global_py_random().seed(4294967303ULL);
    std::vector<Graph> global_random_graphs;
    global_random_graphs.push_back(
        TopologyGenerator::generate("random", 10, random_options(0.23)));
    global_random_graphs.push_back(
        TopologyGenerator::generate("random", 10, random_options(0.23)));
    emit_parity(
        "random_global_sequence",
        encode_graphs(global_random_graphs) +
            "|R:" + encode_rng(global_py_random()));

    global_py_random().seed(4294967303ULL);
    std::vector<Graph> global_waxman_graphs;
    global_waxman_graphs.push_back(TopologyGenerator::generate(
        "waxman", 12, waxman_options(0.8, 0.45)));
    global_waxman_graphs.push_back(TopologyGenerator::generate(
        "waxman", 12, waxman_options(0.8, 0.45)));
    emit_parity(
        "waxman_global_sequence",
        encode_graphs(global_waxman_graphs) +
            "|R:" + encode_rng(global_py_random()));

    const auto requests = parity_batch_requests();
    emit_parity(
        "batch_workers_1",
        encode_graphs(TopologyGenerator::generate_batch(requests, 1)));
    emit_parity(
        "batch_workers_8",
        encode_graphs(TopologyGenerator::generate_batch(requests, 8)));
}

void fnv_add_u64(std::uint64_t& hash, std::uint64_t value)
{
    for (int shift = 0; shift < 64; shift += 8)
    {
        hash ^= static_cast<unsigned char>(value >> shift);
        hash *= kFnvPrime;
    }
}

std::uint64_t checksum_graph_into(
    std::uint64_t hash,
    const Graph& graph)
{
    fnv_add_u64(hash, graph.num_nodes());
    fnv_add_u64(hash, graph.num_edges());
    const std::optional<AttrId> pos_id =
        graph.attribute_registry().find("pos");
    auto [node, node_end] = graph.nodes();
    for (; node != node_end; ++node)
    {
        fnv_add_u64(hash, *node);
        fnv_add_u64(hash, graph.node_attrs(*node).size());
        const AttrValue* position_value = pos_id.has_value()
            ? graph.node_attrs(*node).find(*pos_id)
            : nullptr;
        if (position_value != nullptr)
        {
            fnv_add_u64(hash, 1);
            const AttrList* position = attr_list(*position_value);
            if (position == nullptr || position->values.size() != 2)
            {
                throw std::runtime_error("invalid Waxman pos attribute");
            }
            fnv_add_u64(
                hash,
                double_bits(attr_to_double(position->values[0])));
            fnv_add_u64(
                hash,
                double_bits(attr_to_double(position->values[1])));
        }
        else
        {
            fnv_add_u64(hash, 0);
        }

        const auto [neighbor, neighbor_end] = graph.neighbors(*node);
        std::uint64_t neighbor_count = 0;
        for (auto current = neighbor; current != neighbor_end; ++current)
        {
            ++neighbor_count;
        }
        fnv_add_u64(hash, neighbor_count);
        for (auto current = neighbor; current != neighbor_end; ++current)
        {
            fnv_add_u64(hash, *current);
        }
    }

    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        fnv_add_u64(hash, graph.source(*edge));
        fnv_add_u64(hash, graph.target(*edge));
        fnv_add_u64(hash, graph.edge_attrs(*edge).size());
    }
    fnv_add_u64(hash, graph.graph_attrs().size());
    return hash;
}

std::uint64_t checksum_graph(const Graph& graph)
{
    return checksum_graph_into(kFnvOffset, graph);
}

std::uint64_t checksum_graphs(const std::vector<Graph>& graphs)
{
    std::uint64_t hash = kFnvOffset;
    fnv_add_u64(hash, graphs.size());
    for (const Graph& graph : graphs)
    {
        hash = checksum_graph_into(hash, graph);
    }
    return hash;
}

struct BenchResult
{
    std::vector<std::uint64_t> samples;
    std::uint64_t checksum = 0;
};

template <typename Prepare, typename Action, typename Checksum>
BenchResult measure(
    std::size_t warmups,
    std::size_t repetitions,
    Prepare&& prepare,
    Action&& action,
    Checksum&& checksum)
{
    BenchResult result;
    result.samples.reserve(repetitions);
    for (std::size_t sample = 0;
         sample < warmups + repetitions;
         ++sample)
    {
        auto prepared = prepare(sample);
        const auto start = Clock::now();
        auto value = action(prepared);
        const auto stop = Clock::now();
        result.checksum = checksum(value);
        if (sample >= warmups)
        {
            result.samples.push_back(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        stop - start).count()));
        }
    }
    return result;
}

void emit_bench(
    std::string_view name,
    std::size_t workers,
    const BenchResult& result)
{
    std::cout << "BENCH\t" << name << '\t' << workers << '\t';
    for (std::size_t index = 0; index < result.samples.size(); ++index)
    {
        if (index != 0)
        {
            std::cout << ',';
        }
        std::cout << result.samples[index];
    }
    std::cout << '\t' << result.checksum << '\n';
}

std::vector<TopologyRequest> repeated_requests(
    TopologyType type,
    std::size_t count,
    std::int64_t num_nodes,
    const TopologyOptions& options,
    std::uint64_t seed_base)
{
    std::vector<TopologyRequest> requests;
    requests.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        requests.push_back({
            type,
            num_nodes,
            options,
            seed_base + index * 104729ULL});
    }
    return requests;
}

void emit_benchmark_cases(
    std::size_t warmups,
    std::size_t repetitions,
    std::size_t workers)
{
    std::cout << "META\tpath_nodes\t" << kPathNodes << '\n';
    std::cout << "META\tstar_nodes\t" << kStarNodes << '\n';
    std::cout << "META\tgrid_rows\t" << kGridRows << '\n';
    std::cout << "META\tgrid_columns\t" << kGridColumns << '\n';
    std::cout << "META\trandom_nodes\t" << kRandomNodes << '\n';
    std::cout << "META\twaxman_nodes\t" << kWaxmanNodes << '\n';

    const auto no_prepare = [](std::size_t) { return 0; };
    emit_bench(
        "path.cpp",
        1,
        measure(
            warmups,
            repetitions,
            no_prepare,
            [](int&) { return TopologyGenerator::generate("path", kPathNodes); },
            checksum_graph));
    emit_bench(
        "star.cpp",
        1,
        measure(
            warmups,
            repetitions,
            no_prepare,
            [](int&) { return TopologyGenerator::generate("star", kStarNodes); },
            checksum_graph));
    emit_bench(
        "grid_2d.cpp",
        1,
        measure(
            warmups,
            repetitions,
            no_prepare,
            [](int&)
            {
                return TopologyGenerator::generate(
                    "grid_2d",
                    1,
                    grid_options(kGridRows, kGridColumns));
            },
            checksum_graph));

    emit_bench(
        "random.cpp",
        1,
        measure(
            warmups,
            repetitions,
            [](std::size_t sample)
            {
                return PyRandom(1009 + sample);
            },
            [](PyRandom& random)
            {
                return TopologyGenerator::generate(
                    "random",
                    kRandomNodes,
                    random_options(0.03),
                    random);
            },
            checksum_graph));

    emit_bench(
        "waxman.cpp",
        1,
        measure(
            warmups,
            repetitions,
            [](std::size_t sample)
            {
                return PyRandom(2003 + sample);
            },
            [](PyRandom& random)
            {
                return TopologyGenerator::generate(
                    "waxman", kWaxmanNodes, {}, random);
            },
            checksum_graph));

    const auto path_requests = repeated_requests(
        TopologyType::Path, 64, 2048, {}, 3001);
    const auto star_requests = repeated_requests(
        TopologyType::Star, 64, 2048, {}, 4001);
    const auto grid_requests = repeated_requests(
        TopologyType::Grid2D, 32, 1, grid_options(64, 64), 5001);
    const auto random_requests = repeated_requests(
        TopologyType::Random, 32, 250, random_options(0.04), 6001);
    const auto waxman_requests = repeated_requests(
        TopologyType::Waxman, 16, 250, {}, 7001);

    const auto emit_batch =
        [&](std::string_view name,
            const std::vector<TopologyRequest>& requests)
        {
            emit_bench(
                std::string(name) + ".st",
                1,
                measure(
                    warmups,
                    repetitions,
                    no_prepare,
                    [&](int&)
                    {
                        return TopologyGenerator::generate_batch(requests, 1);
                    },
                    checksum_graphs));
            emit_bench(
                std::string(name) + ".mt",
                workers,
                measure(
                    warmups,
                    repetitions,
                    no_prepare,
                    [&](int&)
                    {
                        return TopologyGenerator::generate_batch(
                            requests, workers);
                    },
                    checksum_graphs));
        };

    emit_batch("batch_path", path_requests);
    emit_batch("batch_star", star_requests);
    emit_batch("batch_grid_2d", grid_requests);
    emit_batch("batch_random", random_requests);
    emit_batch("batch_waxman", waxman_requests);
}

std::size_t parse_size(const char* text)
{
    return static_cast<std::size_t>(std::stoull(text));
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "--parity")
        {
            emit_parity_cases();
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--benchmark")
        {
            const std::size_t warmups = parse_size(argv[2]);
            const std::size_t repetitions = parse_size(argv[3]);
            const std::size_t workers = parse_size(argv[4]);
            if (warmups == 0 || repetitions == 0)
            {
                throw std::invalid_argument(
                    "warmups and repetitions must be positive");
            }
            emit_benchmark_cases(warmups, repetitions, workers);
            return 0;
        }
        throw std::invalid_argument(
            "usage: topology_generator_harness --parity | "
            "--benchmark WARMUPS REPETITIONS WORKERS");
    }
    catch (const std::exception& error)
    {
        std::cerr << "topology_generator_harness: " << error.what() << '\n';
        return 2;
    }
}
