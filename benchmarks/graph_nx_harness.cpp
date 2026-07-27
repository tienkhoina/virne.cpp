// Small, deterministic oracle/benchmark driver for the local graph API.
//
// The companion compare_nx.py script runs the same fixture through
// NetworkX 3.4.2 and compares every value emitted by --parity.  Keeping the
// fixture in this executable (rather than reading a generated file) makes
// the benchmark reproducible and keeps graph construction out of timings.

#include "graph/algorithms/bfs.h"
#include "graph/algorithms/bfs_nx.h"
#include "graph/algorithms/bfs_stats.h"
#include "graph/algorithms/bidirectional_bfs.h"
#include "graph/algorithms/bidirectional_dijkstra.h"
#include "graph/algorithms/dijkstra.h"
#include "graph/algorithms/floyd_warshall.h"
#include "graph/algorithms/k_shortest_paths.h"
#include "graph/cache/weight_cache.h"
#include "graph/graph.h"
#include "graph/nx/centrality.h"
#include "graph/nx/connectivity.h"
#include "graph/nx/shortest_paths.h"
#include "graph/nx/sparse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

struct FixtureEdge
{
    Vertex u;
    Vertex v;
    double weight;
};

// There are no equal weighted routes in the first three Yen results.  The
// reciprocal 1->3/3->1 pair also checks that DiGraph stores arcs separately.
constexpr FixtureEdge kFixtureEdges[] = {
    {0, 1, 1.0},
    {0, 2, 4.0},
    {1, 2, 2.0},
    {1, 3, 5.0},
    {2, 3, 1.0},
    {3, 1, 1.0},
    {3, 4, 3.0},
    {4, 0, 7.0},
};

DiGraph
make_fixture()
{
    DiGraph g;
    for (Vertex v = 0; v < 5; ++v)
    {
        g.add_node();
    }

    for (const FixtureEdge& edge : kFixtureEdges)
    {
        const DiEdge e = g.add_edge(edge.u, edge.v);
        g.edge_attrs(e)["weight"] = edge.weight;
    }

    return g;
}

template <typename T>
void emit_scalar(std::string_view key, const T& value)
{
    std::cout << key << '=' << value << '\n';
}

void emit_double(std::string_view key, double value)
{
    if (std::isinf(value))
    {
        emit_scalar(key, "inf");
        return;
    }
    std::cout << std::setprecision(17) << key << '=' << value << '\n';
}

void emit_path(std::string_view key, const std::vector<Vertex>& path)
{
    std::cout << key << '=';
    for (std::size_t i = 0; i < path.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ',';
        }
        std::cout << path[i];
    }
    std::cout << '\n';
}

void emit_fixture_parity()
{
    const DiGraph g = make_fixture();

    emit_scalar("num_nodes", g.num_nodes());
    emit_scalar("num_edges", g.num_edges());
    emit_scalar("has_edge_0_1", g.has_edge(0, 1) ? 1 : 0);
    emit_scalar("has_edge_1_0", g.has_edge(1, 0) ? 1 : 0);
    emit_scalar("has_edge_3_1", g.has_edge(3, 1) ? 1 : 0);
    emit_scalar("has_edge_1_3", g.has_edge(1, 3) ? 1 : 0);

    const auto degree = nx::degree_centrality(g);
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        emit_double("degree_centrality[" + std::to_string(v) + "]", degree[v]);
    }

    const auto eigen = nx::eigenvector_centrality(g, 10000, 1e-6);
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        emit_double("eigenvector[" + std::to_string(v) + "]", eigen[v]);
    }

    const auto closeness = nx::closeness_centrality(g);
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        emit_double("closeness[" + std::to_string(v) + "]", closeness[v]);
    }

    const auto betweenness = nx::betweenness_centrality(g, "weight");
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        emit_double("betweenness[" + std::to_string(v) + "]", betweenness[v]);
    }

    const BFSResult bfs_result = bfs(g, 0);
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        emit_scalar("bfs_distance[" + std::to_string(v) + "]", bfs_result.distance[v]);
    }

    const auto bfs_map = nx::single_source_shortest_path_length(g, 0);
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        const auto it = bfs_map.find(v);
        emit_scalar("bfs_map[" + std::to_string(v) + "]",
                    it == bfs_map.end() ? std::string("inf") : std::to_string(it->second));
    }

    const DijkstraResult dijkstra_result = dijkstra(g, 0, "weight");
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        emit_double("dijkstra_distance[" + std::to_string(v) + "]", dijkstra_result.distance[v]);
    }

    const auto dijkstra_map = nx::single_source_dijkstra_path_length(g, 0, "weight");
    for (Vertex v = 0; v < g.num_nodes(); ++v)
    {
        const auto it = dijkstra_map.find(v);
        emit_double("dijkstra_map[" + std::to_string(v) + "]",
                    it == dijkstra_map.end() ? std::numeric_limits<double>::infinity() : it->second);
    }

    emit_scalar("shortest_path_length_0_4", nx::shortest_path_length(g, 0, 4));
    emit_path("shortest_path_0_4", nx::shortest_path(g, 0, 4));
    emit_double("dijkstra_path_length_0_4", nx::dijkstra_path_length(g, 0, 4, "weight"));
    emit_path("dijkstra_path_0_4", nx::dijkstra_path(g, 0, 4, "weight"));

    const DistanceMatrix matrix = nx::floyd_warshall(g, "weight");
    for (Vertex u = 0; u < g.num_nodes(); ++u)
    {
        for (Vertex v = 0; v < g.num_nodes(); ++v)
        {
            emit_double("floyd[" + std::to_string(u) + "," + std::to_string(v) + "]",
                        matrix(u, v));
        }
    }

    const auto yen = nx::shortest_simple_paths(g, 0, 4, 3, "weight");
    emit_scalar("yen_count", yen.size());
    for (std::size_t i = 0; i < yen.size(); ++i)
    {
        emit_path("yen_path[" + std::to_string(i) + "]", yen[i]);
        emit_double("yen_cost[" + std::to_string(i) + "]", path_cost(g, yen[i], "weight"));
    }

    const SparseMatrix adjacency = nx::adjacency_matrix(g, "weight");
    for (std::size_t i = 0; i < adjacency.nnz(); ++i)
    {
        emit_double("adj[" + std::to_string(adjacency.row[i]) + "," +
                        std::to_string(adjacency.col[i]) + "]",
                    adjacency.value[i]);
    }

    emit_scalar("is_connected_weak", nx::is_connected(g) ? 1 : 0);
}

DiGraph
make_benchmark_graph(std::size_t n)
{
    DiGraph g;
    for (std::size_t i = 0; i < n; ++i)
    {
        g.add_node();
    }

    // This deterministic edge recipe is duplicated in compare_nx.py.  It
    // gives every vertex a short route to the next vertices and a sparse,
    // repeatable workload for BFS and Dijkstra.
    constexpr std::size_t kOffsets[] = {1, 7, 31, 127, 509};
    for (Vertex u = 0; u < n; ++u)
    {
        for (std::size_t offset : kOffsets)
        {
            const Vertex v = (u + offset) % n;
            if (u == v || g.has_edge(u, v))
            {
                continue;
            }
            const DiEdge e = g.add_edge(u, v);
            // Keep weights positive and avoid large tie sets.
            g.edge_attrs(e)["weight"] = 1.0 + static_cast<double>((u * 13 + offset) % 997) / 997.0;
        }
    }
    return g;
}

template <typename Fn>
double median_ms(Fn&& fn, std::size_t warmups, std::size_t reps, std::uint64_t& sink)
{
    for (std::size_t i = 0; i < warmups; ++i)
    {
        sink += fn();
    }

    std::vector<double> samples;
    samples.reserve(reps);
    for (std::size_t i = 0; i < reps; ++i)
    {
        const auto start = Clock::now();
        sink += fn();
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    return samples.size() % 2 == 0
               ? (samples[middle - 1] + samples[middle]) * 0.5
               : samples[middle];
}

void emit_benchmark(std::size_t n, std::size_t warmups, std::size_t reps)
{
    // Expensive O(VE)/O(V^3) algorithms intentionally use smaller versions
    // of the exact same deterministic graph.  This keeps the default
    // 3-warmup/15-median protocol practical without timing graph creation.
    const std::size_t main_n = std::max<std::size_t>(n, 2);
    const std::size_t eigen_n = std::min<std::size_t>(main_n, 1500);
    const std::size_t central_n = std::min<std::size_t>(main_n, 140);
    const std::size_t floyd_n = std::min<std::size_t>(main_n, 72);
    const std::size_t yen_n = std::min<std::size_t>(main_n, 120);

    const DiGraph g = make_benchmark_graph(main_n);
    const DiGraph eigen_g = make_benchmark_graph(eigen_n);
    const DiGraph central_g = make_benchmark_graph(central_n);
    const DiGraph floyd_g = make_benchmark_graph(floyd_n);
    const DiGraph yen_g = make_benchmark_graph(yen_n);

    // Resolve the hot attribute once before all measurements.  Public APIs
    // that intentionally accept a NetworkX-like string still perform their
    // own lookup, while the AttrId sparse API consumes this cached ID.
    const AttrId weight_id = g.attr_id("weight");
    const DiWeightCache weight_cache(g);
    const Vertex target = main_n - 1;
    const VertexSet banned_vertices = {main_n / 3};
    const EdgeSet banned_edges = {{0, 1}};
    BFSWorkspace bfs_workspace(main_n);
    const DijkstraResult prepared_dijkstra = dijkstra(g, 0, "weight");
    const std::vector<Vertex> prepared_path =
        build_path(prepared_dijkstra, 0, target);
    if (prepared_path.size() < 2)
    {
        throw std::runtime_error(
            "benchmark graph is unexpectedly missing a weighted path");
    }
    const std::size_t split = prepared_path.size() / 2;
    const std::vector<Vertex> join_root(
        prepared_path.begin(), prepared_path.begin() + split + 1);
    const std::vector<Vertex> join_spur(
        prepared_path.begin() + split, prepared_path.end());
    constexpr std::size_t helper_batch = 1024;

    const Vertex yen_target = yen_n - 1;
    const auto first_yen = bidirectional_dijkstra(
        yen_g, 0, yen_target, VertexSet{}, EdgeSet{}, "weight");
    if (!first_yen.found)
    {
        throw std::runtime_error("benchmark Yen graph is unexpectedly disconnected");
    }
    const PathResult first_path{first_yen.path, first_yen.cost};

    std::uint64_t sink = 0;
    struct BenchRow
    {
        std::string name;
        std::size_t nodes;
        std::size_t edges;
        std::size_t calls;
        double milliseconds;
    };
    std::vector<BenchRow> rows;
    rows.reserve(32);

    auto add = [&](
        std::string name,
        const DiGraph& graph,
        auto&& fn,
        std::size_t calls = 1)
    {
        rows.push_back({
            std::move(name), graph.num_nodes(), graph.num_edges(), calls,
            median_ms(std::forward<decltype(fn)>(fn), warmups, reps, sink) /
                static_cast<double>(calls)});
    };

    add("raw_neighbor_scan", g, [&]() -> std::uint64_t
    {
        std::uint64_t sum = 0;
        for (Vertex u = 0; u < g.num_nodes(); ++u)
        {
            for (const auto& edge : g.neighbors_fast(u))
            {
                sum += static_cast<std::uint64_t>(edge.get_target()) +
                       edge.get_property().edge_id;
            }
        }
        return sum;
    });
    add("weight_cache_scan", g, [&]() -> std::uint64_t
    {
        double sum = 0.0;
        for (Vertex u = 0; u < g.num_nodes(); ++u)
        {
            for (const auto& edge : g.neighbors_fast(u))
            {
                sum += weight_cache.value(edge, weight_id);
            }
        }
        return static_cast<std::uint64_t>(sum * 1000000.0);
    });
    add("bfs", g, [&]() -> std::uint64_t
    {
        const auto result = bfs(g, 0);
        return result.distance[target] + result.predecessor[target];
    });
    add("bfs_nx", g, [&]() -> std::uint64_t
    {
        const auto result = bfs_nx(g, 0);
        return result.size() + result.at(target);
    });
    add("bfs_stats", g, [&]() -> std::uint64_t
    {
        const auto result = bfs_stats(g, 0, bfs_workspace);
        return result.reachable + static_cast<std::uint64_t>(result.sum_dist);
    });
    add("bidirectional_bfs", g, [&]() -> std::uint64_t
    {
        const auto result = bidirectional_bfs(g, 0, target);
        return result.distance + result.path.size() + result.found;
    });
    add("nx.single_source_shortest_path_length", g, [&]() -> std::uint64_t
    {
        const auto result = nx::single_source_shortest_path_length(g, 0);
        return result.size() + result.at(target);
    });
    add("nx.shortest_path_length", g, [&]() -> std::uint64_t
    {
        return nx::shortest_path_length(g, 0, target);
    });
    add("nx.shortest_path", g, [&]() -> std::uint64_t
    {
        const auto result = nx::shortest_path(g, 0, target);
        return result.size() + result.back();
    });

    add("dijkstra", g, [&]() -> std::uint64_t
    {
        const auto result = dijkstra(g, 0, "weight");
        return static_cast<std::uint64_t>(result.distance[target] * 1000000.0) +
               result.predecessor[target];
    });
    add("dijkstra_masked", g, [&]() -> std::uint64_t
    {
        const auto result = dijkstra(
            g, 0, banned_vertices, banned_edges, "weight");
        return static_cast<std::uint64_t>(result.distance[target] * 1000000.0) +
               result.predecessor[target];
    });
    add("bidirectional_dijkstra", g, [&]() -> std::uint64_t
    {
        const auto result = bidirectional_dijkstra(
            g, 0, target, VertexSet{}, EdgeSet{}, "weight");
        return static_cast<std::uint64_t>(result.cost * 1000000.0) +
               result.path.size() + result.found;
    });
    add("edge_cost", g, [&]() -> std::uint64_t
    {
        double total = 0.0;
        for (std::size_t i = 0; i < helper_batch; ++i)
        {
            total += edge_cost(g, 0, 1, "weight");
        }
        return static_cast<std::uint64_t>(total * 1000000.0);
    }, helper_batch);
    add("path_cost", g, [&]() -> std::uint64_t
    {
        double total = 0.0;
        for (std::size_t i = 0; i < helper_batch; ++i)
        {
            total += path_cost(g, prepared_path, "weight");
        }
        return static_cast<std::uint64_t>(total * 1000000.0);
    }, helper_batch);
    add("path_prefix_costs", g, [&]() -> std::uint64_t
    {
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < helper_batch; ++i)
        {
            const auto result = path_prefix_costs(g, prepared_path, "weight");
            total += result.size() +
                     static_cast<std::uint64_t>(result.back() * 1000000.0);
        }
        return total;
    }, helper_batch);
    add("build_path", g, [&]() -> std::uint64_t
    {
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < helper_batch; ++i)
        {
            const auto result = build_path(prepared_dijkstra, 0, target);
            total += result.size() + result.back();
        }
        return total;
    }, helper_batch);
    add("join_paths", g, [&]() -> std::uint64_t
    {
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < helper_batch; ++i)
        {
            const auto result = join_paths(join_root, join_spur);
            total += result.size() + result.back();
        }
        return total;
    }, helper_batch);
    add("nx.single_source_dijkstra_path_length", g, [&]() -> std::uint64_t
    {
        const auto result = nx::single_source_dijkstra_path_length(g, 0, "weight");
        return result.size() + static_cast<std::uint64_t>(result.at(target) * 1000000.0);
    });
    add("nx.dijkstra_path_length", g, [&]() -> std::uint64_t
    {
        return static_cast<std::uint64_t>(
            nx::dijkstra_path_length(g, 0, target, "weight") * 1000000.0);
    });
    add("nx.dijkstra_path", g, [&]() -> std::uint64_t
    {
        const auto result = nx::dijkstra_path(g, 0, target, "weight");
        return result.size() + result.back();
    });

    add("floyd_warshall", floyd_g, [&]() -> std::uint64_t
    {
        const auto result = ::floyd_warshall(floyd_g, "weight");
        return static_cast<std::uint64_t>(result(0, floyd_n - 1) * 1000000.0) +
               result.size();
    });
    add("nx.floyd_warshall", floyd_g, [&]() -> std::uint64_t
    {
        const auto result = nx::floyd_warshall(floyd_g, "weight");
        return static_cast<std::uint64_t>(result(0, floyd_n - 1) * 1000000.0) +
               result.size();
    });
    add("yen_k_shortest_paths", yen_g, [&]() -> std::uint64_t
    {
        const auto result = yen_k_shortest_paths(yen_g, 0, yen_target, 5, "weight");
        std::uint64_t sum = result.size();
        for (const auto& path : result) sum += path.path.size();
        return sum;
    });
    add("generate_candidates", yen_g, [&]() -> std::uint64_t
    {
        const auto result = generate_candidates(yen_g, first_path, yen_target, "weight");
        std::uint64_t sum = result.size();
        for (const auto& path : result) sum += path.path.size();
        return sum;
    });
    add("nx.shortest_simple_paths", yen_g, [&]() -> std::uint64_t
    {
        const auto result = nx::shortest_simple_paths(yen_g, 0, yen_target, 5, "weight");
        std::uint64_t sum = result.size();
        for (const auto& path : result) sum += path.size();
        return sum;
    });

    add("nx.adjacency_matrix", g, [&]() -> std::uint64_t
    {
        const auto result = nx::adjacency_matrix(g, "weight");
        return result.nnz() + result.row.back() + result.col.back();
    });
    add("nx.attr_sparse_matrix", g, [&]() -> std::uint64_t
    {
        const auto result = nx::attr_sparse_matrix(g, weight_id);
        return result.nnz() + result.row.back() + result.col.back();
    });
    add("nx.is_connected", g, [&]() -> std::uint64_t
    {
        return nx::is_connected(g) ? g.num_nodes() : 0;
    });
    add("nx.degree_centrality", g, [&]() -> std::uint64_t
    {
        const auto result = nx::degree_centrality(g);
        return result.size() + static_cast<std::uint64_t>(result[0] * 1000000.0);
    });
    add("nx.eigenvector_centrality", eigen_g, [&]() -> std::uint64_t
    {
        const auto result = nx::eigenvector_centrality(eigen_g, 10000, 1e-6);
        return result.size() + static_cast<std::uint64_t>(result[0] * 1000000.0);
    });
    add("nx.closeness_centrality", central_g, [&]() -> std::uint64_t
    {
        const auto result = nx::closeness_centrality(central_g);
        return result.size() + static_cast<std::uint64_t>(result[0] * 1000000.0);
    });
    add("nx.betweenness_centrality", central_g, [&]() -> std::uint64_t
    {
        const auto result = nx::betweenness_centrality(central_g, "weight");
        return result.size() + static_cast<std::uint64_t>(result[0] * 1000000.0);
    });

    emit_scalar("bench_count", rows.size());
    emit_scalar("bench_warmups", warmups);
    emit_scalar("bench_reps", reps);
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const std::string prefix = "bench[" + std::to_string(i) + "].";
        emit_scalar(prefix + "name", rows[i].name);
        emit_scalar(prefix + "nodes", rows[i].nodes);
        emit_scalar(prefix + "edges", rows[i].edges);
        emit_scalar(prefix + "calls", rows[i].calls);
        emit_double(prefix + "cpp_ms", rows[i].milliseconds);
    }
    emit_scalar("bench_sink", sink);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        bool bench = false;
        std::size_t n = 5000;
        std::size_t warmups = 3;
        std::size_t reps = 15;

        for (int i = 1; i < argc; ++i)
        {
            const std::string_view arg(argv[i]);
            if (arg == "--parity")
            {
                bench = false;
            }
            else if (arg == "--bench")
            {
                bench = true;
            }
            else if (arg == "--nodes" && i + 1 < argc)
            {
                n = static_cast<std::size_t>(std::stoull(argv[++i]));
            }
            else if (arg == "--warmups" && i + 1 < argc)
            {
                warmups = static_cast<std::size_t>(std::stoull(argv[++i]));
            }
            else if (arg == "--reps" && i + 1 < argc)
            {
                reps = static_cast<std::size_t>(std::stoull(argv[++i]));
            }
            else if (arg == "--help")
            {
                std::cout << "usage: graph_nx_harness [--parity|--bench] [--nodes N] [--warmups N] [--reps N]\n";
                return 0;
            }
            else
            {
                std::cerr << "unknown argument: " << arg << '\n';
                return 2;
            }
        }

        if (bench)
        {
            if (reps == 0)
            {
                throw std::invalid_argument("--reps must be positive");
            }
            emit_benchmark(n, warmups, reps);
        }
        else
        {
            emit_fixture_parity();
        }
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "graph_nx_harness: " << ex.what() << '\n';
        return 1;
    }
}
