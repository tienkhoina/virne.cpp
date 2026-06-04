#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "graph/algorithms/dijkstra.h"
#include "graph/algorithms/k_shortest_paths.h"
#include "graph/generators/waxman_generator.h"
#include "graph/graph.h"
#include "graph/nx/shortest_paths.h"
#include "random/py_random.h"

#include <chrono>   

template<typename Fn>
double bench_ms(Fn&& fn)
{
    auto start =
        std::chrono::steady_clock::now();

    fn();

    auto end =
        std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start).count();
}

namespace
{

constexpr double kEps = 1e-9;

std::string
path_key(
    const std::vector<Vertex>& path)
{
    std::string key;
    for (auto v : path)
    {
        key += std::to_string(v);
        key += ',';
    }
    return key;
}

bool
is_simple_path(
    const std::vector<Vertex>& path)
{
    std::unordered_set<Vertex> seen;
    for (auto v : path)
    {
        if (!seen.insert(v).second)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
void
print_vec(
    const std::vector<T>& v,
    std::ostream& os)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        if (i)
        {
            os << ' ';
        }
        os << v[i];
    }
}

void
print_vec_double(
    const std::vector<double>& v,
    std::ostream& os)
{
    os << std::fixed << std::setprecision(12);
    for (size_t i = 0; i < v.size(); ++i)
    {
        if (i)
        {
            os << ' ';
        }
        os << v[i];
    }
}

void
print_path(
    const std::vector<Vertex>& path,
    std::ostream& os)
{
    print_vec(path, os);
}

double
get_weight(
    const Graph& g,
    Edge e)
{
    const auto& attrs = g.edge_attrs(e);
    auto it = attrs.find("weight");
    if (it == attrs.end())
    {
        return 1.0;
    }
    return std::get<double>(it->second);
}

void
assign_weights_from_distance_or_default(
    Graph& g)
{
    auto [it, end] = g.edges();
    size_t idx = 0;

    for (; it != end; ++it, ++idx)
    {
        double w = 1.0;

        const auto& attrs = g.edge_attrs(*it);
        auto dit = attrs.find("distance");
        if (dit != attrs.end())
        {
            if (std::holds_alternative<double>(dit->second))
            {
                w = std::get<double>(dit->second);
            }
            else if (std::holds_alternative<int64_t>(dit->second))
            {
                w = static_cast<double>(std::get<int64_t>(dit->second));
            }
        }

        // Tiny deterministic perturbation to reduce ties.
        w += static_cast<double>(idx) * 1e-6;

        g.edge_attrs(*it)["weight"] = w;
    }
}

Graph
build_waxman_graph(
    PyRandom& prng,
    uint64_t& out_waxman_seed,
    double& out_alpha,
    double& out_beta)
{
    WaxmanConfig cfg;

    cfg.num_nodes = 1000;
    cfg.alpha = prng.uniform(0.20, 0.70);
    cfg.beta = prng.uniform(0.20, 0.90);
    cfg.seed = static_cast<uint64_t>(
        prng.randrange(1000000000ULL) + 1);

    out_waxman_seed = cfg.seed;
    out_alpha = cfg.alpha;
    out_beta = cfg.beta;

    Graph g =
        WaxmanGenerator::generate(
            cfg);

    assign_weights_from_distance_or_default(g);

    // Guarantee connectedness deterministically by adding a backbone chain.
    for (size_t i = 0; i + 1 < g.num_nodes(); ++i)
    {
        if (!g.has_edge(i, i + 1))
        {
            auto e = g.add_edge(i, i + 1);
            g.edge_attrs(e)["weight"] = 1.0 + static_cast<double>(i) * 1e-6;
        }
    }

    return g;
}

void
dump_graph(
    const Graph& g,
    uint64_t py_seed,
    uint64_t waxman_seed,
    double alpha,
    double beta,
    Vertex source,
    Vertex target,
    const std::string& path)
{
    std::ofstream out(path);
    out << std::fixed << std::setprecision(12);

    out << "PY_RANDOM_SEED " << py_seed << '\n';
    out << "WAXMAN_NUM_NODES " << g.num_nodes() << '\n';
    out << "WAXMAN_ALPHA " << alpha << '\n';
    out << "WAXMAN_BETA " << beta << '\n';
    out << "WAXMAN_SEED " << waxman_seed << '\n';
    out << "SOURCE " << source << '\n';
    out << "TARGET " << target << '\n';
    out << "NUM_NODES " << g.num_nodes() << '\n';
    out << "NUM_EDGES " << g.num_edges() << '\n';

    auto [it, end] = g.edges();
    for (; it != end; ++it)
    {
        Vertex u = g.source(*it);
        Vertex v = g.target(*it);
        double w = get_weight(g, *it);
        out << "EDGE " << u << ' ' << v << ' ' << w << '\n';
    }
}

void
print_graph_dump(
    const Graph& g,
    uint64_t py_seed,
    uint64_t waxman_seed,
    double alpha,
    double beta,
    Vertex source,
    Vertex target)
{
    std::cout << std::fixed << std::setprecision(12);

    std::cout << "PY_RANDOM_SEED " << py_seed << '\n';
    std::cout << "WAXMAN_NUM_NODES " << g.num_nodes() << '\n';
    std::cout << "WAXMAN_ALPHA " << alpha << '\n';
    std::cout << "WAXMAN_BETA " << beta << '\n';
    std::cout << "WAXMAN_SEED " << waxman_seed << '\n';
    std::cout << "SOURCE " << source << '\n';
    std::cout << "TARGET " << target << '\n';
    std::cout << "NUM_NODES " << g.num_nodes() << '\n';
    std::cout << "NUM_EDGES " << g.num_edges() << '\n';

    std::cout << "EDGE_LIST\n";
    auto [it, end] = g.edges();
    for (; it != end; ++it)
    {
        Vertex u = g.source(*it);
        Vertex v = g.target(*it);
        double w = get_weight(g, *it);
        std::cout << u << ' ' << v << ' ' << w << '\n';
    }
}

void
print_bfs_section(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    std::cout << "\n[BFS]\n";

    size_t length =
        nx::shortest_path_length(
            g,
            source,
            target);

    auto path =
        nx::shortest_path(
            g,
            source,
            target);

    auto dist =
        nx::single_source_shortest_path_length(
            g,
            source);

    std::cout << "shortest_path_length " << length << '\n';
    std::cout << "shortest_path ";
    print_path(path, std::cout);
    std::cout << '\n';

    std::cout << "single_source_shortest_path_length\n";
    for (size_t v = 0; v < g.num_nodes(); ++v)
    {
        std::cout << v << ' ' << dist.at(v) << '\n';
    }
}

void
print_dijkstra_section(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    std::cout << "\n[DIJKSTRA]\n";

    auto path =
        nx::dijkstra_path(
            g,
            source,
            target);

    double cost =
        nx::dijkstra_path_length(
            g,
            source,
            target);

    auto dist =
        nx::single_source_dijkstra_path_length(
            g,
            source);

    auto result =
        dijkstra(
            g,
            source);

    auto rebuilt =
        build_path(
            result,
            source,
            target);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "dijkstra_path_length " << cost << '\n';
    std::cout << "dijkstra_path ";
    print_path(path, std::cout);
    std::cout << '\n';

    std::cout << "single_source_dijkstra_path_length\n";
    for (size_t v = 0; v < g.num_nodes(); ++v)
    {
        std::cout << v << ' ' << dist.at(v) << '\n';
    }

    std::cout << "raw_dijkstra_build_path ";
    print_path(rebuilt, std::cout);
    std::cout << '\n';

    if (!rebuilt.empty())
    {
        assert(rebuilt.front() == source);
        assert(rebuilt.back() == target);
        // assert(std::abs(path_cost(g, rebuilt) - cost) < 1e-8);
    }
}

void
print_floyd_section(
    const Graph& g)
{
    std::cout << "\n[FLOYD_WARSHALL]\n";

    auto fw =
        nx::floyd_warshall(
            g);

    std::cout << std::fixed << std::setprecision(12);

    for (size_t i = 0; i < g.num_nodes(); ++i)
    {
        for (size_t j = 0; j < g.num_nodes(); ++j)
        {
            if (j)
            {
                std::cout << ' ';
            }
            std::cout << fw(i, j);
        }
        std::cout << '\n';
    }
}

void
print_yen_section(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    std::cout << "\n[YEN]\n";

    auto paths =
        nx::shortest_simple_paths(
            g,
            source,
            target,
            10);

    auto raw =
        yen_k_shortest_paths(
            g,
            source,
            target,
            10);

    assert(paths.size() == raw.size());

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "count " << paths.size() << '\n';

    std::unordered_set<std::string> seen;
    double last_cost = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < paths.size(); ++i)
    {
        const auto& p = paths[i];
        double c = path_cost(g, p);

        assert(!p.empty());
        assert(p.front() == source);
        assert(p.back() == target);
        assert(is_simple_path(p));
        assert(seen.insert(path_key(p)).second);
        assert(c + kEps >= last_cost);
        assert(std::abs(c - raw[i].cost) < 1e-8);
        assert(raw[i].path == p);

        last_cost = c;

        std::cout << "path " << i << ' ';
        print_path(p, std::cout);
        std::cout << " cost " << c << '\n';
    }
}

void
run_end_to_end()
{
    constexpr uint64_t py_seed = 42;

    PyRandom prng(py_seed);

    uint64_t waxman_seed = 0;
    double alpha = 0.0;
    double beta = 0.0;

    Graph g =
        build_waxman_graph(
            prng,
            waxman_seed,
            alpha,
            beta);

    Vertex source = 0;
    Vertex target = static_cast<Vertex>(g.num_nodes() - 1);

    dump_graph(
        g,
        py_seed,
        waxman_seed,
        alpha,
        beta,
        source,
        target,
        "graph_dump.txt");

    print_graph_dump(
        g,
        py_seed,
        waxman_seed,
        alpha,
        beta,
        source,
        target);

    print_bfs_section(
        g,
        source,
        target);

    print_dijkstra_section(
        g,
        source,
        target);

    print_floyd_section(g);

    print_yen_section(
        g,
        source,
        target);
    auto bfs_ms =
        bench_ms(
            [&]
            {
                nx::shortest_path(
                    g,
                    source,
                    target);
            });

    std::cout
        << "BFS(ms) "
        << bfs_ms
        << '\n';

    auto dijkstra_ms =
        bench_ms(
            [&]
            {
                nx::dijkstra_path(
                    g,
                    source,
                    target);
            });

    std::cout
        << "DIJKSTRA(ms) "
        << dijkstra_ms
        << '\n';

    auto sssp_ms =
        bench_ms(
            [&]
            {
                nx::single_source_dijkstra_path_length(
                    g,
                    source);
            });

    std::cout
        << "SSSP(ms) "
        << sssp_ms
        << '\n';

    auto fw_ms =
        bench_ms(
            [&]
            {
                nx::floyd_warshall(
                    g);
            });

    std::cout
        << "FLOYD(ms) "
        << fw_ms
        << '\n';

    auto yen_ms =
        bench_ms(
            [&]
            {
                nx::shortest_simple_paths(
                    g,
                    source,
                    target,
                    10);
            });

    std::cout
        << "YEN(ms) "
        << yen_ms
        << '\n';

        std::cout << "\nALL PASS\n";
    }

} // namespace

int main()
{
    run_end_to_end();
    return 0;
}