#include "graph.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

volatile uint64_t benchmark_sink = 0;

struct TimingPair
{
    double raw;
    double view;
};

template <typename RawFunction, typename ViewFunction>
TimingPair paired_median_ms(
    RawFunction&& raw_function,
    ViewFunction&& view_function)
{
    std::vector<double> raw_samples;
    std::vector<double> view_samples;
    raw_samples.reserve(11);
    view_samples.reserve(11);

    const auto measure = []<typename Function>(
                             Function&& function)
    {
        const auto start = Clock::now();
        const uint64_t checksum = function();
        const auto stop = Clock::now();
        benchmark_sink ^= checksum;
        return std::chrono::duration<
            double,
            std::milli>(stop - start)
            .count();
    };

    for (size_t sample = 0;
         sample < 15;
         ++sample)
    {
        double raw_ms = 0.0;
        double view_ms = 0.0;
        // Ten short, alternating blocks share scheduler and frequency drift
        // far more evenly than two long back-to-back regions.
        for (size_t block = 0;
             block < 10;
             ++block)
        {
            if (((sample + block) & 1U) == 0)
            {
                raw_ms += measure(raw_function);
                view_ms += measure(view_function);
            }
            else
            {
                view_ms += measure(view_function);
                raw_ms += measure(raw_function);
            }
        }
        if (sample >= 4)
        {
            raw_samples.push_back(raw_ms);
            view_samples.push_back(view_ms);
        }
    }
    std::sort(
        raw_samples.begin(),
        raw_samples.end());
    std::sort(
        view_samples.begin(),
        view_samples.end());
    return {
        raw_samples[raw_samples.size() / 2],
        view_samples[view_samples.size() / 2]};
}

Graph make_graph()
{
    constexpr size_t node_count = 2048;
    Graph graph;
    for (size_t node = 0;
         node < node_count;
         ++node)
    {
        graph.add_node();
        graph.node_attrs(node)["feature"] =
            static_cast<int64_t>(
                node % 97);
    }
    for (Vertex u = 0;
         u < node_count;
         ++u)
    {
        for (Vertex offset = 1;
             offset <= 8;
             ++offset)
        {
            const Vertex v =
                (u + offset) % node_count;
            graph.edge_attrs(
                graph.add_edge(u, v))["weight"] =
                    static_cast<int64_t>(
                        (u + v) % 31 + 1);
        }
    }
    return graph;
}

void print_row(
    const std::string& name,
    double indexed,
    double view)
{
    std::cout << "| " << name
              << " | " << std::fixed
              << std::setprecision(3)
              << indexed
              << " | " << view
              << " | " << view / indexed
              << "x |\n";
}

} // namespace

int main()
{
    Graph graph = make_graph();
    const AttrId feature =
        graph.attr_id("feature");
    const AttrId weight =
        graph.attr_id("weight");
    // paired_median_ms aggregates ten interleaved blocks, for 400 traversals
    // per recorded sample without exposing either path to one long time slice.
    constexpr size_t repeats = 40;

    const TimingPair node_timing = paired_median_ms(
        [&]
        {
            uint64_t sum = 0;
            for (size_t repeat = 0;
                 repeat < repeats;
                 ++repeat)
            {
                for (Vertex node = 0;
                     node < graph.num_nodes();
                     ++node)
                {
                    sum += static_cast<uint64_t>(
                        attr_to_double(
                            graph.node_attrs(node).at(
                                feature)));
                }
            }
            return sum;
        },
        [&]
        {
            uint64_t sum = 0;
            const auto nodes =
                graph.node_view();
            for (size_t repeat = 0;
                 repeat < repeats;
                 ++repeat)
            {
                for (auto item : nodes.data())
                {
                    sum += static_cast<uint64_t>(
                        attr_to_double(
                            item.attrs.at(feature)));
                }
            }
            return sum;
        });

    const TimingPair edge_timing = paired_median_ms(
        [&]
        {
            uint64_t sum = 0;
            for (size_t repeat = 0;
                 repeat < repeats;
                 ++repeat)
            {
                auto [edge, edge_end] =
                    graph.edges();
                for (; edge != edge_end; ++edge)
                {
                    sum += static_cast<uint64_t>(
                        attr_to_double(
                            graph.edge_attrs(*edge).at(
                                weight)));
                }
            }
            return sum;
        },
        [&]
        {
            uint64_t sum = 0;
            const auto edges =
                graph.edge_view();
            for (size_t repeat = 0;
                 repeat < repeats;
                 ++repeat)
            {
                for (auto item : edges.data())
                {
                    sum += static_cast<uint64_t>(
                        attr_to_double(
                            item.attrs.at(weight)));
                }
            }
            return sum;
        });

    const TimingPair adjacency_timing = paired_median_ms(
        [&]
        {
            uint64_t sum = 0;
            for (size_t repeat = 0;
                 repeat < repeats;
                 ++repeat)
            {
                for (Vertex node = 0;
                     node < graph.num_nodes();
                     ++node)
                {
                    for (const auto& edge :
                         graph.neighbors_fast(node))
                    {
                        sum += edge.get_target();
                    }
                }
            }
            return sum;
        },
        [&]
        {
            uint64_t sum = 0;
            const auto adjacency =
                graph.adjacency_view();
            for (size_t repeat = 0;
                 repeat < repeats;
                 ++repeat)
            {
                for (Vertex node = 0;
                     node < graph.num_nodes();
                     ++node)
                {
                    for (const Vertex neighbor :
                         adjacency[node])
                    {
                        sum += neighbor;
                    }
                }
            }
            return sum;
        });

    std::cout
        << "| traversal | indexed/raw ms | view ms | view/raw |\n"
        << "|---|---:|---:|---:|\n";
    print_row(
        "node attributes",
        node_timing.raw,
        node_timing.view);
    print_row(
        "edge attributes",
        edge_timing.raw,
        edge_timing.view);
    print_row(
        "adjacency",
        adjacency_timing.raw,
        adjacency_timing.view);
    return benchmark_sink ==
                   uint64_t{0xFFFFFFFFFFFFFFFFULL}
        ? 1
        : 0;
}
