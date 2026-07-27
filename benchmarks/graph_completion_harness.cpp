#include "generators/topology_generators.h"
#include "generators/waxman_generator.h"
#include "graph.h"
#include "nx/centrality.h"
#include "nx/attributes.h"
#include "nx/shortest_paths.h"
#include "nx/sparse.h"
#include "nx/subgraph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace
{

volatile uint64_t benchmark_sink = 0;

uint64_t mix(
    uint64_t hash,
    uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ULL;
    return hash;
}

uint64_t hash_path(
    uint64_t hash,
    const std::vector<Vertex>& path)
{
    hash = mix(hash, path.size());
    for (const Vertex vertex : path)
    {
        hash = mix(hash, vertex + 1);
    }
    return hash;
}

Graph benchmark_graph(
    size_t n)
{
    Graph graph;
    for (size_t i = 0; i < n; ++i)
    {
        graph.add_node();
    }
    const AttrId weight = graph.attr_id("weight");
    const AttrId capacity = graph.attr_id("capacity");

    auto add =
        [&](Vertex u, Vertex v, double value)
        {
            if (v >= n || graph.has_edge(u, v))
            {
                return;
            }
            const Edge edge = graph.add_edge(u, v);
            graph.edge_attrs(edge).set(weight, value);
            graph.edge_attrs(edge).set(
                capacity,
                static_cast<double>(
                    graph.edge_id(edge) % 9 + 1));
        };

    for (Vertex u = 0; u < n; ++u)
    {
        add(u, u + 1, 1.0 + (u % 7) * 0.013);
        add(u, u + 5, 2.2 + (u % 11) * 0.017);
        add(u, u + 13, 3.7 + (u % 13) * 0.019);
    }
    return graph;
}

std::vector<std::vector<double>> dense_matrix_fixture(
    size_t n)
{
    std::vector<std::vector<double>> matrix(
        n, std::vector<double>(n, 0.0));
    for (size_t u = 0; u < n; ++u)
    {
        if (u % 17 == 0)
        {
            matrix[u][u] = static_cast<double>(u % 13 + 1);
        }
        for (const size_t offset : {size_t{1}, size_t{7}, size_t{23}})
        {
            const size_t v = u + offset;
            if (v >= n)
            {
                continue;
            }
            matrix[u][v] = static_cast<double>((u * 11 + v) % 31 + 1);
            if ((u + offset) % 3 != 0)
            {
                matrix[v][u] =
                    static_cast<double>((u * 7 + v * 3) % 29 + 1);
            }
        }
    }
    return matrix;
}

DiGraph layered_graph(
    size_t layers)
{
    DiGraph graph;
    const size_t node_count = 2 * layers + 2;
    for (size_t i = 0; i < node_count; ++i)
    {
        graph.add_node();
    }
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    for (size_t layer = 0;
         layer + 1 < layers;
         ++layer)
    {
        const Vertex left = 1 + 2 * layer;
        const Vertex next = left + 2;
        graph.add_edge(left, next);
        graph.add_edge(left, next + 1);
        graph.add_edge(left + 1, next);
        graph.add_edge(left + 1, next + 1);
    }
    const Vertex last = 1 + 2 * (layers - 1);
    const Vertex target = 2 * layers + 1;
    graph.add_edge(last, target);
    graph.add_edge(last + 1, target);
    return graph;
}

template <typename GraphType>
GraphType reverse_layer_graph()
{
    GraphType graph;
    for (size_t i = 0; i < 7; ++i)
    {
        graph.add_node();
    }
    for (const auto [u, v] : {
             std::pair<Vertex, Vertex>{0, 3},
             {0, 2}, {0, 1},
             {3, 5}, {3, 4},
             {2, 5}, {2, 4},
             {1, 5}, {1, 4},
             {5, 6}, {4, 6}})
    {
        graph.add_edge(u, v);
    }
    return graph;
}

template <typename GraphType>
GraphType ordered_result_graph()
{
    GraphType graph;
    for (size_t i = 0; i < 7; ++i)
    {
        graph.add_node();
    }
    const AttrId weight = graph.attr_id("weight");
    for (const auto [u, v, value] : {
             std::tuple<Vertex, Vertex, double>{0, 4, 2.0},
             {0, 2, 1.0}, {0, 5, 1.0},
             {4, 3, 1.0}, {2, 3, 2.0},
             {5, 6, 1.0}, {3, 1, 1.0},
             {6, 1, 1.0}})
    {
        graph.edge_attrs(graph.add_edge(u, v)).set(
            weight, value);
    }
    return graph;
}

uint64_t all_pairs_length_checksum(
    const Graph& graph)
{
    const auto lengths =
        nx::shortest_path_length(graph);
    uint64_t checksum = 1469598103934665603ULL;
    for (Vertex source = 0;
         source < graph.num_nodes();
         ++source)
    {
        for (Vertex target = 0;
             target < graph.num_nodes();
             ++target)
        {
            const auto found_source = lengths.find(source);
            if (found_source == lengths.end())
            {
                continue;
            }
            const auto found_target =
                found_source->second.find(target);
            if (found_target == found_source->second.end())
            {
                continue;
            }
            checksum = mix(checksum, source + 1);
            checksum = mix(checksum, target + 1);
            checksum = mix(
                checksum,
                found_target->second + 1);
        }
    }
    return checksum;
}

uint64_t all_pairs_path_checksum(
    const Graph& graph)
{
    const auto paths = nx::shortest_path(graph);
    uint64_t checksum = 1469598103934665603ULL;
    for (Vertex source = 0;
         source < graph.num_nodes();
         ++source)
    {
        for (Vertex target = 0;
             target < graph.num_nodes();
             ++target)
        {
            checksum = mix(checksum, source + 1);
            checksum = mix(checksum, target + 1);
            checksum = hash_path(
                checksum,
                paths.at(source).at(target));
        }
    }
    return checksum;
}

uint64_t all_shortest_checksum(
    const DiGraph& graph)
{
    auto paths = nx::all_shortest_paths(
        graph,
        0,
        graph.num_nodes() - 1);
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& path : paths)
    {
        checksum = hash_path(checksum, path);
    }
    return checksum;
}

template <typename GraphLike>
uint64_t all_shortest_weighted_order_checksum(
    const GraphLike& graph)
{
    const auto paths = nx::all_shortest_paths(
        graph, 0, 6, "weight");
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& path : paths)
    {
        checksum = hash_path(checksum, path);
    }
    return checksum;
}

uint64_t filtered_path_checksum(
    const nx::GraphView& view)
{
    const auto path = nx::dijkstra_path(
        view,
        0,
        view.graph().num_nodes() - 1,
        "weight");
    return hash_path(
        1469598103934665603ULL,
        path);
}

uint64_t lazy_yen_checksum(
    const Graph& graph)
{
    ShortestSimplePathOptions options;
    options.max_paths = 20;
    options.weight_attr = "weight";
    auto paths = nx::shortest_simple_paths(
        graph,
        0,
        graph.num_nodes() - 1,
        options);

    uint64_t checksum = 1469598103934665603ULL;
    while (auto path = paths.next())
    {
        checksum = hash_path(
            checksum,
            path->path);
    }
    return checksum;
}

template <typename GraphLike>
uint64_t unweighted_yen_checksum(
    const GraphLike& graph)
{
    auto paths = nx::shortest_simple_paths(
        graph, 0, 6);
    uint64_t checksum = 1469598103934665603ULL;
    while (auto path = paths.next())
    {
        checksum = hash_path(checksum, path->path);
    }
    return checksum;
}

uint64_t score_checksum(
    const nx::NodeScores& scores)
{
    uint64_t checksum = 1469598103934665603ULL;
    for (const double score : scores)
    {
        const uint64_t scaled =
            static_cast<uint64_t>(
                std::llround(score * 1e12));
        checksum = mix(checksum, scaled);
    }
    return checksum;
}

template <typename GraphLike>
uint64_t ordered_unweighted_map_checksum(
    const GraphLike& graph)
{
    const auto result =
        nx::single_source_shortest_path_length(
            graph, 0);
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& [vertex, distance] : result)
    {
        checksum = mix(checksum, vertex + 1);
        checksum = mix(checksum, distance + 1);
    }
    return checksum;
}

template <typename GraphLike>
uint64_t ordered_weighted_map_checksum(
    const GraphLike& graph)
{
    const auto result =
        nx::single_source_dijkstra_path_length(
            graph, 0, "weight");
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& [vertex, distance] : result)
    {
        checksum = mix(checksum, vertex + 1);
        checksum = mix(
            checksum,
            static_cast<uint64_t>(
                std::llround(distance * 1e6)));
    }
    return checksum;
}

template <typename GraphLike>
uint64_t ordered_all_pairs_checksum(
    const GraphLike& graph)
{
    const auto result = nx::shortest_path_length(graph);
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& [source, lengths] : result)
    {
        checksum = mix(checksum, source + 1);
        for (const auto& [target, distance] : lengths)
        {
            checksum = mix(checksum, target + 1);
            checksum = mix(checksum, distance + 1);
        }
    }
    return checksum;
}

uint64_t sparse_checksum(
    const Graph& graph,
    const std::vector<Vertex>& order)
{
    const SparseMatrix matrix =
        nx::attr_sparse_matrix(
            graph,
            "capacity",
            true,
            order);

    uint64_t checksum = 1469598103934665603ULL;
    for (size_t i = 0;
         i < matrix.nnz();
         ++i)
    {
        const uint64_t scaled =
            static_cast<uint64_t>(
                matrix.value[i] * 1e12 + 0.5);
        checksum = mix(checksum, matrix.row[i] + 1);
        checksum = mix(checksum, matrix.col[i] + 1);
        checksum = mix(checksum, scaled);
    }
    return checksum;
}

uint64_t directed_checksum(
    const Graph& graph)
{
    const DiGraph directed = graph.to_directed();
    uint64_t checksum =
        directed.num_nodes() * 1000003ULL +
        directed.num_edges();
    auto [edge, end] = directed.edges();
    for (; edge != end; ++edge)
    {
        checksum +=
            (directed.source(*edge) + 1) *
                1000033ULL +
            (directed.target(*edge) + 1) *
                1000037ULL;
    }
    return checksum;
}

uint64_t in_out_checksum(
    const DiGraph& graph)
{
    uint64_t checksum = 0;
    for (Vertex v = 0;
         v < graph.num_nodes();
         ++v)
    {
        checksum +=
            (v + 1) *
            (graph.in_edges_fast(v).size() * 31ULL +
             graph.out_edges_fast(v).size() * 37ULL);
    }
    return checksum;
}

uint64_t graph_shape_checksum(
    const Graph& graph)
{
    uint64_t checksum =
        graph.num_nodes() * 1000003ULL +
        graph.num_edges();
    auto [edge, end] = graph.edges();
    for (; edge != end; ++edge)
    {
        checksum = mix(checksum, graph.source(*edge) + 1);
        checksum = mix(checksum, graph.target(*edge) + 1);
    }
    return checksum;
}

uint64_t dense_matrix_checksum(
    const std::vector<std::vector<double>>& matrix)
{
    const Graph graph(matrix);
    uint64_t checksum = graph_shape_checksum(graph);
    const AttrId weight_id =
        graph.attr_id("weight");
    auto [edge, end] = graph.edges();
    for (; edge != end; ++edge)
    {
        const double weight = attr_to_double(
            graph.edge_attrs(*edge).at(weight_id));
        checksum = mix(
            checksum,
            static_cast<uint64_t>(std::llround(weight * 1e6)));
    }
    return checksum;
}

uint64_t endpoint_attribute_checksum(
    const Graph& graph)
{
    const auto values =
        nx::get_edge_attributes(graph, "capacity");
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& [edge, value] : values)
    {
        checksum = mix(checksum, edge.first + 1);
        checksum = mix(checksum, edge.second + 1);
        checksum = mix(
            checksum,
            static_cast<uint64_t>(
                std::llround(attr_to_double(value) * 1e6)));
    }
    return checksum;
}

uint64_t unweighted_dijkstra_checksum(
    const Graph& graph)
{
    return hash_path(
        1469598103934665603ULL,
        nx::dijkstra_path(
            graph, 0, graph.num_nodes() - 1, std::nullopt));
}

uint64_t cutoff_dijkstra_checksum(
    const Graph& graph)
{
    const auto distances =
        nx::single_source_dijkstra_path_length(
            graph, 0, 4.0, std::string_view{"weight"});
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& [vertex, distance] : distances)
    {
        checksum = mix(checksum, vertex + 1);
        checksum = mix(
            checksum,
            static_cast<uint64_t>(std::llround(distance * 1e6)));
    }
    return checksum;
}

uint64_t cutoff_unweighted_checksum(
    const Graph& graph)
{
    const auto distances =
        nx::single_source_shortest_path_length(
            graph, 0, 2.1);
    uint64_t checksum = 1469598103934665603ULL;
    for (const auto& [vertex, distance] : distances)
    {
        checksum = mix(checksum, vertex + 1);
        checksum = mix(checksum, distance);
    }
    return checksum;
}

uint64_t waxman_checksum(
    const Graph& graph)
{
    uint64_t checksum = graph_shape_checksum(graph);
    const AttrId position_id =
        graph.attr_id("pos");
    for (Vertex v = 0;
         v < graph.num_nodes();
         ++v)
    {
        const AttrList* position = attr_list(
            graph.node_attrs(v).at(position_id));
        for (const AttrValue& value : position->values)
        {
            const double coordinate = attr_to_double(value);
            uint64_t bits = 0;
            std::memcpy(&bits, &coordinate, sizeof(bits));
            checksum = mix(checksum, bits);
        }
    }
    return checksum;
}

template <typename Function>
void benchmark(
    const std::string& name,
    size_t repeats,
    Function&& function)
{
    const uint64_t checksum = function();
    const auto start =
        std::chrono::steady_clock::now();
    for (size_t i = 0; i < repeats; ++i)
    {
        benchmark_sink ^= function();
    }
    const auto end =
        std::chrono::steady_clock::now();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    std::cout
        << "RESULT " << name << ' '
        << checksum << ' '
        << std::fixed << std::setprecision(6)
        << milliseconds << ' '
        << repeats << '\n';
}

} // namespace

int main(
    int argc,
    char** argv)
{
    const Graph graph = benchmark_graph(120);
    const auto dense_matrix = dense_matrix_fixture(120);

    if (argc == 2 &&
        std::string(argv[1]) == "--dump-yen")
    {
        ShortestSimplePathOptions options;
        options.max_paths = 20;
        options.weight_attr = "weight";
        auto paths = nx::shortest_simple_paths(
            graph, 0, graph.num_nodes() - 1,
            options);
        while (auto path = paths.next())
        {
            std::cout
                << std::setprecision(17)
                << path->cost;
            for (Vertex vertex : path->path)
            {
                std::cout << ' ' << vertex;
            }
            std::cout << '\n';
        }
        return 0;
    }

    const Graph path = nx::path_graph(100);
    const DiGraph layered = layered_graph(10);
    const nx::GraphView view = nx::subgraph_view(
        graph,
        nx::NodeFilter{},
        [](Vertex u, Vertex v)
        {
            const Vertex delta = u > v ? u - v : v - u;
            return delta == 1 || (u + v) % 4 != 0;
        });
    const DiGraph directed = graph.to_directed();
    const Graph tie_graph =
        reverse_layer_graph<Graph>();
    const DiGraph tie_digraph =
        reverse_layer_graph<DiGraph>();
    const nx::DiGraphView tie_view =
        nx::subgraph_view(
            tie_digraph,
            nx::NodeFilter{},
            [](Vertex u, Vertex v)
            {
                return !(u == 2 && v == 5);
            });
    const Graph ordered_graph =
        ordered_result_graph<Graph>();
    const DiGraph ordered_digraph =
        ordered_result_graph<DiGraph>();
    const nx::GraphView ordered_graph_view =
        nx::subgraph_view(
            ordered_graph,
            [](Vertex v)
            {
                return v != 4;
            });
    const nx::DiGraphView ordered_digraph_view =
        nx::subgraph_view(
            ordered_digraph,
            [](Vertex v)
            {
                return v != 4;
            });
    std::vector<Vertex> reverse_order(graph.num_nodes());
    for (Vertex v = 0; v < graph.num_nodes(); ++v)
    {
        reverse_order[v] = graph.num_nodes() - 1 - v;
    }

    benchmark("all_pairs_length", 4, [&]
    {
        return all_pairs_length_checksum(graph);
    });
    benchmark("all_pairs_path", 2, [&]
    {
        return all_pairs_path_checksum(path);
    });
    benchmark("ordered_unweighted_graph", 500, [&]
    {
        return ordered_unweighted_map_checksum(
            ordered_graph);
    });
    benchmark("ordered_unweighted_digraph", 500, [&]
    {
        return ordered_unweighted_map_checksum(
            ordered_digraph);
    });
    benchmark("ordered_unweighted_graph_view", 500, [&]
    {
        return ordered_unweighted_map_checksum(
            ordered_graph_view);
    });
    benchmark("ordered_unweighted_digraph_view", 500, [&]
    {
        return ordered_unweighted_map_checksum(
            ordered_digraph_view);
    });
    benchmark("ordered_weighted_graph", 500, [&]
    {
        return ordered_weighted_map_checksum(
            ordered_graph);
    });
    benchmark("ordered_weighted_digraph", 500, [&]
    {
        return ordered_weighted_map_checksum(
            ordered_digraph);
    });
    benchmark("ordered_weighted_graph_view", 500, [&]
    {
        return ordered_weighted_map_checksum(
            ordered_graph_view);
    });
    benchmark("ordered_weighted_digraph_view", 500, [&]
    {
        return ordered_weighted_map_checksum(
            ordered_digraph_view);
    });
    benchmark("ordered_all_pairs_graph_view", 100, [&]
    {
        return ordered_all_pairs_checksum(
            ordered_graph_view);
    });
    benchmark("ordered_all_pairs_digraph_view", 100, [&]
    {
        return ordered_all_pairs_checksum(
            ordered_digraph_view);
    });
    benchmark("all_shortest_paths", 15, [&]
    {
        return all_shortest_checksum(layered);
    });
    benchmark("all_shortest_weighted_graph_order", 100, [&]
    {
        return all_shortest_weighted_order_checksum(
            tie_graph);
    });
    benchmark("all_shortest_weighted_digraph_order", 100, [&]
    {
        return all_shortest_weighted_order_checksum(
            tie_digraph);
    });
    benchmark("all_shortest_weighted_view_order", 100, [&]
    {
        return all_shortest_weighted_order_checksum(
            tie_view);
    });
    benchmark("filtered_dijkstra", 500, [&]
    {
        return filtered_path_checksum(view);
    });
    benchmark("lazy_yen_20", 8, [&]
    {
        return lazy_yen_checksum(graph);
    });
    benchmark("yen_unweighted_graph_order", 40, [&]
    {
        return unweighted_yen_checksum(tie_graph);
    });
    benchmark("yen_unweighted_digraph_order", 80, [&]
    {
        return unweighted_yen_checksum(tie_digraph);
    });
    benchmark("yen_unweighted_view_order", 80, [&]
    {
        return unweighted_yen_checksum(tie_view);
    });
    benchmark("attr_sparse_normalized", 500, [&]
    {
        return sparse_checksum(graph, reverse_order);
    });

    benchmark("dense_matrix_constructor", 20, [&]
    {
        return dense_matrix_checksum(dense_matrix);
    });

    benchmark("edge_attributes_endpoint", 500, [&]
    {
        return endpoint_attribute_checksum(graph);
    });

    benchmark("dijkstra_weight_none", 1000, [&]
    {
        return unweighted_dijkstra_checksum(graph);
    });

    benchmark("single_source_dijkstra_cutoff", 1000, [&]
    {
        return cutoff_dijkstra_checksum(graph);
    });
    benchmark("single_source_shortest_cutoff", 1000, [&]
    {
        return cutoff_unweighted_checksum(graph);
    });
    benchmark("to_directed", 100, [&]
    {
        return directed_checksum(graph);
    });
    benchmark("digraph_in_out_fast", 20000, [&]
    {
        return in_out_checksum(directed);
    });
    benchmark("betweenness_unweighted", 3, [&]
    {
        return score_checksum(
            nx::betweenness_centrality(graph));
    });
    benchmark("betweenness_weighted", 3, [&]
    {
        return score_checksum(
            nx::betweenness_centrality(
                graph, "weight"));
    });
    benchmark("erdos_renyi", 25, []
    {
        return graph_shape_checksum(
            nx::erdos_renyi_graph(
                220, 0.08, 42));
    });
    benchmark("connected_erdos_renyi", 15, []
    {
        return graph_shape_checksum(
            nx::connected_erdos_renyi_graph(
                160, 0.035, 42));
    });
    benchmark("waxman", 15, []
    {
        WaxmanConfig config;
        config.num_nodes = 150;
        config.alpha = 0.35;
        config.beta = 0.65;
        config.seed = 42;
        return waxman_checksum(
            WaxmanGenerator::generate(config));
    });

    return benchmark_sink == 0x12345678ULL ? 1 : 0;
}
