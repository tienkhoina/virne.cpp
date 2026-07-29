#include "topology/topological_metric_calculator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using virne::network::MetricColumn;
using virne::network::TopologicalMetricCalculator;
using virne::network::TopologicalMetricOptions;
using virne::network::TopologicalMetrics;

enum MetricBits : std::uint8_t
{
    Degree = 1U << 0U,
    Closeness = 1U << 1U,
    Eigenvector = 1U << 2U,
    Betweenness = 1U << 3U,
    All = Degree | Closeness | Eigenvector | Betweenness,
};

enum class FixtureKind : std::uint8_t
{
    Empty,
    Singleton,
    Branch,
    Disconnected,
    SelfLoop,
    CycleChord,
    Complete,
    Path20,
    Ordered,
};

struct DifferentialCase
{
    const char* name;
    FixtureKind fixture;
    std::uint8_t metric_bits;
    bool normalize;
    std::size_t workers;
    bool constructor_defaults = false;
};

struct BenchmarkCase
{
    const char* name;
    std::size_t node_count;
    std::uint8_t metric_bits;
};

constexpr DifferentialCase kDifferentialCases[] = {
    {"empty_none", FixtureKind::Empty, 0, true, 1},
    {"empty_degree", FixtureKind::Empty, Degree, true, 1},
    {"empty_closeness", FixtureKind::Empty, Closeness, true, 4},
    {"empty_betweenness", FixtureKind::Empty, Betweenness, true, 4},
    {"empty_eigenvector_error", FixtureKind::Empty, Eigenvector, true, 1},
    {"empty_all_error", FixtureKind::Empty, All, true, 4},
    {"singleton_all_raw", FixtureKind::Singleton, All, false, 1},
    {"singleton_all_normalized", FixtureKind::Singleton, All, true, 8},
    {"branch_degree_normalized", FixtureKind::Branch, Degree, true, 1},
    {"branch_all_raw", FixtureKind::Branch, All, false, 1},
    {"branch_all_normalized_w1", FixtureKind::Branch, All, true, 1},
    {"branch_all_normalized_w2", FixtureKind::Branch, All, true, 2},
    {"branch_all_normalized_w4", FixtureKind::Branch, All, true, 4},
    {"branch_all_normalized_w8", FixtureKind::Branch, All, true, 8},
    {"disconnected_dcb_raw", FixtureKind::Disconnected,
     static_cast<std::uint8_t>(Degree | Closeness | Betweenness), false, 4},
    {"disconnected_dcb_normalized", FixtureKind::Disconnected,
     static_cast<std::uint8_t>(Degree | Closeness | Betweenness), true, 8},
    {"self_loop_all_raw", FixtureKind::SelfLoop, All, false, 4},
    {"self_loop_all_normalized", FixtureKind::SelfLoop, All, true, 8},
    {"cycle_chord_all_raw", FixtureKind::CycleChord, All, false, 1},
    {"cycle_chord_all_normalized", FixtureKind::CycleChord, All, true, 4},
    {"complete_all_normalized", FixtureKind::Complete, All, true, 8},
    {"path20_eigenvector_error", FixtureKind::Path20, Eigenvector, false, 1},
    {"path20_none", FixtureKind::Path20, 0, true, 8},
    {"ordered_all_raw_w1", FixtureKind::Ordered, All, false, 1},
    {"ordered_all_raw_w8", FixtureKind::Ordered, All, false, 8},
    {"ordered_cb_normalized_w2", FixtureKind::Ordered,
     static_cast<std::uint8_t>(Closeness | Betweenness), true, 2},
    {"ordered_cb_normalized_w8", FixtureKind::Ordered,
     static_cast<std::uint8_t>(Closeness | Betweenness), true, 8},
    {"branch_constructor_defaults", FixtureKind::Branch, Degree, true, 0, true},
};

constexpr BenchmarkCase kBenchmarkCases[] = {
    {"degree_20000", 20000, Degree},
    {"eigenvector_4000", 4000, Eigenvector},
    {"closeness_500", 500, Closeness},
    {"betweenness_240", 240, Betweenness},
    {"all_metrics_240", 240, All},
};

constexpr std::size_t kCorpusCaseCount = 64;

Graph graph_with_nodes(std::size_t node_count)
{
    Graph graph;
    for (std::size_t node = 0; node < node_count; ++node)
    {
        graph.add_node();
    }
    return graph;
}

void add_edges(
    Graph& graph,
    std::initializer_list<EdgeEndpoints> edges)
{
    graph.add_edges_from(std::vector<EdgeEndpoints>(edges));
}

Graph make_fixture(FixtureKind kind)
{
    switch (kind)
    {
    case FixtureKind::Empty:
        return {};

    case FixtureKind::Singleton:
        return graph_with_nodes(1);

    case FixtureKind::Branch:
    {
        Graph graph = graph_with_nodes(6);
        add_edges(graph, {{0, 1}, {1, 2}, {2, 3}, {2, 4}, {4, 5}});
        return graph;
    }

    case FixtureKind::Disconnected:
    {
        Graph graph = graph_with_nodes(8);
        add_edges(graph, {{0, 1}, {1, 2}, {3, 4}, {4, 5}, {5, 3}});
        return graph;
    }

    case FixtureKind::SelfLoop:
    {
        Graph graph = graph_with_nodes(5);
        add_edges(
            graph,
            {{0, 0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 1}});
        return graph;
    }

    case FixtureKind::CycleChord:
    {
        Graph graph = graph_with_nodes(7);
        add_edges(
            graph,
            {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
             {6, 0}, {0, 3}, {2, 5}});
        return graph;
    }

    case FixtureKind::Complete:
    {
        Graph graph = graph_with_nodes(6);
        for (Vertex left = 0; left < graph.num_nodes(); ++left)
        {
            for (Vertex right = left + 1; right < graph.num_nodes(); ++right)
            {
                graph.add_edge(left, right);
            }
        }
        return graph;
    }

    case FixtureKind::Path20:
    {
        Graph graph = graph_with_nodes(20);
        for (Vertex node = 1; node < graph.num_nodes(); ++node)
        {
            graph.add_edge(node - 1, node);
        }
        return graph;
    }

    case FixtureKind::Ordered:
    {
        Graph graph = graph_with_nodes(9);
        add_edges(
            graph,
            {{4, 1}, {4, 7}, {1, 0}, {7, 8}, {1, 2}, {7, 6},
             {2, 5}, {6, 3}, {5, 3}, {0, 8}, {2, 6}, {1, 7}});
        return graph;
    }
    }

    throw std::invalid_argument("unknown FixtureKind");
}

Graph make_benchmark_graph(std::size_t node_count)
{
    constexpr std::size_t offsets[] = {1, 7, 31, 127, 509};
    Graph graph = graph_with_nodes(node_count);
    for (Vertex node = 0; node < node_count; ++node)
    {
        for (std::size_t offset : offsets)
        {
            const Vertex neighbor =
                static_cast<Vertex>((node + offset) % node_count);
            if (neighbor != node && !graph.has_edge(node, neighbor))
            {
                graph.add_edge(node, neighbor);
            }
        }
    }
    return graph;
}

std::uint64_t corpus_random(std::uint64_t& state) noexcept
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

void corpus_shuffle(
    std::vector<EdgeEndpoints>& edges,
    std::uint64_t& state)
{
    for (std::size_t remaining = edges.size(); remaining > 1; --remaining)
    {
        const std::size_t swap_index = static_cast<std::size_t>(
            corpus_random(state) % remaining);
        std::swap(edges[remaining - 1], edges[swap_index]);
    }
}

Graph make_corpus_graph(std::size_t case_index)
{
    const std::size_t node_count = 3 + case_index % 19;
    Graph graph = graph_with_nodes(node_count);
    std::uint64_t state =
        0x9e3779b97f4a7c15ULL ^
        (static_cast<std::uint64_t>(case_index) *
         0xd1b54a32d192ed03ULL);

    std::vector<EdgeEndpoints> ring;
    ring.reserve(node_count);
    for (Vertex node = 0; node < node_count; ++node)
    {
        ring.emplace_back(node, (node + 1) % node_count);
    }
    corpus_shuffle(ring, state);
    for (const auto& edge : ring)
    {
        if (!graph.has_edge(edge.first, edge.second))
        {
            graph.add_edge(edge.first, edge.second);
        }
    }

    std::vector<EdgeEndpoints> candidates;
    const bool include_self_loops = case_index % 5 == 0;
    for (Vertex left = 0; left < node_count; ++left)
    {
        const Vertex first_right = include_self_loops ? left : left + 1;
        for (Vertex right = first_right; right < node_count; ++right)
        {
            candidates.emplace_back(left, right);
        }
    }
    corpus_shuffle(candidates, state);
    const std::size_t extra_target =
        node_count + (case_index * 5) % (node_count * 2 + 1);
    std::size_t added = 0;
    for (const auto& edge : candidates)
    {
        if (added == extra_target)
        {
            break;
        }
        if (!graph.has_edge(edge.first, edge.second))
        {
            graph.add_edge(edge.first, edge.second);
            ++added;
        }
    }
    return graph;
}

TopologicalMetricOptions make_options(
    std::uint8_t metric_bits,
    bool normalize,
    std::size_t workers)
{
    TopologicalMetricOptions options;
    options.degree = (metric_bits & Degree) != 0;
    options.closeness = (metric_bits & Closeness) != 0;
    options.eigenvector = (metric_bits & Eigenvector) != 0;
    options.betweenness = (metric_bits & Betweenness) != 0;
    options.normalize = normalize;
    options.worker_count = workers;
    return options;
}

std::uint32_t float_bits(float value) noexcept
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float32 is required");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void emit_column(
    std::string_view prefix,
    const std::optional<MetricColumn>& column)
{
    std::cout << prefix << ".present=" << (column.has_value() ? 1 : 0) << '\n';
    if (!column)
    {
        return;
    }
    std::cout << prefix << ".rows=" << column->rows() << '\n';
    std::cout << prefix << ".columns=" << column->columns() << '\n';
    std::cout << prefix << ".bits=";
    for (std::size_t row = 0; row < column->rows(); ++row)
    {
        if (row != 0)
        {
            std::cout << ',';
        }
        std::cout << float_bits((*column)[row]);
    }
    std::cout << '\n';
}

void emit_calculation(
    const std::string& prefix,
    const Graph& graph,
    std::uint8_t metric_bits,
    bool normalize,
    std::size_t workers,
    bool constructor_defaults)
{
    try
    {
        TopologicalMetrics metrics;
        if (constructor_defaults)
        {
            const TopologicalMetricCalculator calculator(graph);
            metrics = calculator.metrics;
        }
        else
        {
            metrics = TopologicalMetricCalculator::calculate(
                graph,
                make_options(metric_bits, normalize, workers));
        }
        std::cout << prefix << ".status=ok\n";
        emit_column(prefix + ".degree", metrics.node_degree_centrality);
        emit_column(prefix + ".closeness", metrics.node_closeness_centrality);
        emit_column(prefix + ".eigenvector", metrics.node_eigenvector_centrality);
        emit_column(prefix + ".betweenness", metrics.node_betweenness_centrality);
    }
    catch (const std::domain_error& error)
    {
        std::cout << prefix << ".status=error\n";
        std::cout << prefix << ".error_kind=null_graph\n";
        std::cout << prefix << ".error_message=" << error.what() << '\n';
    }
    catch (const std::runtime_error& error)
    {
        std::cout << prefix << ".status=error\n";
        std::cout << prefix << ".error_kind=nonconvergence\n";
        std::cout << prefix << ".error_message=" << error.what() << '\n';
    }
}

void emit_differential()
{
    std::cout << "diff_count="
              << (sizeof(kDifferentialCases) / sizeof(kDifferentialCases[0]))
              << '\n';
    for (std::size_t index = 0;
         index < sizeof(kDifferentialCases) / sizeof(kDifferentialCases[0]);
         ++index)
    {
        const DifferentialCase& item = kDifferentialCases[index];
        const std::string prefix = "diff[" + std::to_string(index) + "]";
        std::cout << prefix << ".name=" << item.name << '\n';
        const Graph graph = make_fixture(item.fixture);
        emit_calculation(
            prefix,
            graph,
            item.metric_bits,
            item.normalize,
            item.workers,
            item.constructor_defaults);
    }

    constexpr std::size_t corpus_workers[] = {1, 2, 4, 8};
    std::cout << "corpus_count=" << kCorpusCaseCount << '\n';
    for (std::size_t index = 0; index < kCorpusCaseCount; ++index)
    {
        const std::string prefix = "corpus[" + std::to_string(index) + "]";
        const bool normalize = (index & 1U) != 0;
        const std::size_t workers = corpus_workers[index % 4];
        std::cout << prefix << ".name=generated_" << index << '\n';
        std::cout << prefix << ".normalize=" << (normalize ? 1 : 0) << '\n';
        std::cout << prefix << ".workers=" << workers << '\n';
        emit_calculation(
            prefix,
            make_corpus_graph(index),
            All,
            normalize,
            workers,
            false);
    }
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept
{
    constexpr std::uint64_t fnv_prime = 1099511628211ULL;
    for (unsigned byte = 0; byte < 8; ++byte)
    {
        hash ^= (value >> (byte * 8U)) & 0xffU;
        hash *= fnv_prime;
    }
}

void hash_column(
    std::uint64_t& hash,
    const std::optional<MetricColumn>& column)
{
    hash_u64(hash, column.has_value() ? 1 : 0);
    if (!column)
    {
        return;
    }
    hash_u64(hash, column->rows());
    for (float value : column->values)
    {
        hash_u64(hash, float_bits(value));
    }
}

std::uint64_t metrics_checksum(const TopologicalMetrics& metrics)
{
    std::uint64_t hash = 1469598103934665603ULL;
    hash_column(hash, metrics.node_degree_centrality);
    hash_column(hash, metrics.node_closeness_centrality);
    hash_column(hash, metrics.node_eigenvector_centrality);
    hash_column(hash, metrics.node_betweenness_centrality);
    return hash;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0)
    {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5;
}

double percentile95(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    return values[index];
}

void emit_benchmark(
    std::size_t workers,
    std::size_t warmups,
    std::size_t repetitions)
{
    std::cout << "bench_count="
              << (sizeof(kBenchmarkCases) / sizeof(kBenchmarkCases[0]))
              << '\n';
    std::cout << "bench_workers=" << workers << '\n';
    std::cout << "bench_warmups=" << warmups << '\n';
    std::cout << "bench_repetitions=" << repetitions << '\n';

    for (std::size_t index = 0;
         index < sizeof(kBenchmarkCases) / sizeof(kBenchmarkCases[0]);
         ++index)
    {
        const BenchmarkCase& item = kBenchmarkCases[index];
        const Graph graph = make_benchmark_graph(item.node_count);
        const TopologicalMetricOptions options = make_options(
            item.metric_bits, true, workers);

        for (std::size_t warmup = 0; warmup < warmups; ++warmup)
        {
            const TopologicalMetrics result =
                TopologicalMetricCalculator::calculate(graph, options);
            if (metrics_checksum(result) == 0)
            {
                throw std::runtime_error("unreachable warm-up checksum");
            }
        }

        std::vector<double> samples;
        samples.reserve(repetitions);
        std::uint64_t checksum = 0;
        for (std::size_t repetition = 0;
             repetition < repetitions;
             ++repetition)
        {
            const auto start = Clock::now();
            const TopologicalMetrics result =
                TopologicalMetricCalculator::calculate(graph, options);
            const auto stop = Clock::now();
            samples.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            const std::uint64_t current = metrics_checksum(result);
            if (repetition == 0)
            {
                checksum = current;
            }
            else if (current != checksum)
            {
                throw std::runtime_error(
                    "benchmark result changed between repetitions");
            }
        }

        std::vector<double> deviations;
        const double sample_median = median(samples);
        deviations.reserve(samples.size());
        for (double sample : samples)
        {
            deviations.push_back(std::abs(sample - sample_median));
        }

        const std::string prefix = "bench[" + std::to_string(index) + "]";
        std::cout << prefix << ".name=" << item.name << '\n';
        std::cout << prefix << ".nodes=" << graph.num_nodes() << '\n';
        std::cout << prefix << ".edges=" << graph.num_edges() << '\n';
        std::cout << std::setprecision(17);
        std::cout << prefix << ".cpp_median_ms=" << sample_median << '\n';
        std::cout << prefix << ".cpp_mad_ms=" << median(deviations) << '\n';
        std::cout << prefix << ".cpp_p95_ms=" << percentile95(samples) << '\n';
        std::cout << prefix << ".checksum=" << checksum << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        bool benchmark = false;
        std::size_t workers = 0;
        std::size_t warmups = 3;
        std::size_t repetitions = 11;

        for (int index = 1; index < argc; ++index)
        {
            const std::string_view argument(argv[index]);
            if (argument == "--differential")
            {
                benchmark = false;
            }
            else if (argument == "--benchmark")
            {
                benchmark = true;
            }
            else if (argument == "--workers" && index + 1 < argc)
            {
                workers = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--warmups" && index + 1 < argc)
            {
                warmups = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--repetitions" && index + 1 < argc)
            {
                repetitions = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--help")
            {
                std::cout
                    << "usage: vne_topological_metric_calculator_harness "
                       "[--differential|--benchmark] [--workers N] "
                       "[--warmups N] [--repetitions N]\n";
                return 0;
            }
            else
            {
                throw std::invalid_argument(
                    "unknown or incomplete argument: " + std::string(argument));
            }
        }

        if (repetitions == 0)
        {
            throw std::invalid_argument("--repetitions must be positive");
        }
        if (benchmark)
        {
            emit_benchmark(workers, warmups, repetitions);
        }
        else
        {
            emit_differential();
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vne_topological_metric_calculator_harness: "
                  << error.what() << '\n';
        return 1;
    }
}
