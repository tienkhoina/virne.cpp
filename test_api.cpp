#include "graph/generators/waxman_generator.h"

#include "graph/nx/attributes.h"
#include "graph/nx/centrality.h"
#include "graph/nx/sparse.h"
#include "graph/nx/shortest_paths.h"
#include "graph/algorithms/bfs.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <limits>
#include <vector>

using Clock =
    std::chrono::
        high_resolution_clock;

#define CHECK(x)                                   \
    do                                             \
    {                                              \
        if (!(x))                                  \
        {                                          \
            std::cerr                              \
                << "[FAIL] "                       \
                << #x                              \
                << '\n';                           \
                                                   \
            std::exit(1);                          \
        }                                          \
                                                   \
        std::cout                                  \
            << "[PASS] "                           \
            << #x                                  \
            << '\n';                               \
    }                                              \
    while (false)

static double elapsed_ms(
    Clock::time_point start)
{
    return
        std::chrono::
            duration<double,
                     std::milli>(
                Clock::now()
                - start)
                .count();
}


static double
benchmark_scan(
    const Graph& g)
{
    using Clock =
        std::chrono::high_resolution_clock;

    volatile uint64_t sum = 0;

    auto start =
        Clock::now();

    for (size_t rep = 0;
         rep < g.num_nodes();
         ++rep)
    {
        for (Vertex u = 0;
             u < g.num_nodes();
             ++u)
        {
            const auto& out =
                g.neighbors_fast(u);

            for (const auto& e : out)
            {
                sum +=
                    e.get_target();
            }
        }
    }

    auto end =
        Clock::now();

    return
        std::chrono::duration<double,
        std::milli>(
            end - start)
            .count();
}

static double
benchmark_bfs_nopred(
    const Graph& g)
{
    using Clock =
        std::chrono::high_resolution_clock;

    constexpr size_t INF =
        std::numeric_limits<
            size_t>::max();

    const size_t n =
        g.num_nodes();

    std::vector<size_t>
        dist(n);

    std::vector<Vertex>
        queue(n);

    auto start =
        Clock::now();

    for (Vertex source = 0;
         source < n;
         ++source)
    {
        std::fill(
            dist.begin(),
            dist.end(),
            INF);

        size_t head = 0;
        size_t tail = 0;

        dist[source] = 0;

        queue[tail++] =
            source;

        while (head < tail)
        {
            const Vertex u =
                queue[head++];

            const size_t nd =
                dist[u] + 1;

            const auto& out =
                g.neighbors_fast(u);

            for (const auto& e : out)
            {
                const Vertex v =
                    e.get_target();

                if (dist[v] != INF)
                {
                    continue;
                }

                dist[v] = nd;

                queue[tail++] = v;
            }
        }
    }

    auto end =
        Clock::now();

    return
        std::chrono::duration<double,
        std::milli>(
            end - start)
            .count();
}

int main()
{
    std::cout
        << std::fixed
        << std::setprecision(
               12);

        std::cout
        << "RawNeighbor="
        << sizeof(RawNeighbor)
        << '\n';


    
    std::cout
    << "Vertex="
    << sizeof(Vertex)
    << '\n';
    //
    // generate graph
    //

    WaxmanConfig cfg;

    cfg.num_nodes = 1000;
    cfg.alpha = 0.4;
    cfg.beta = 0.2;
    cfg.seed = 42;

    Graph g =
        WaxmanGenerator::
            generate(cfg);

    CHECK(
        g.num_nodes()
        == 1000);

    CHECK(
        g.num_edges()
        > 0);

    //
    // node attributes
    //

    

    std::unordered_map<
        Vertex,
        AttrValue>
        cpu_values;

    std::unordered_map<
        Vertex,
        AttrValue>
        gpu_values;

    for (Vertex v = 0;
         v < g.num_nodes();
         ++v)
    {
        cpu_values[v] =
            int64_t(
                100 + v);

        gpu_values[v] =
            int64_t(
                1000 + v);
    }

    nx::set_node_attributes(
        g,
        cpu_values,
        "cpu");

    nx::set_node_attributes(
        g,
        gpu_values,
        "gpu");

    //
    // edge attributes
    //

    std::unordered_map<
        uint32_t,
        AttrValue>
        weight_values;

    std::unordered_map<
        uint32_t,
        AttrValue>
        bw_values;

    auto [eit, eend] =
        g.edges();

    uint32_t idx = 1;

    for (; eit != eend; ++eit)
    {
        Edge e =
            *eit;

        uint32_t eid =
            g.edge_id(e);

        weight_values[eid] =
            double(idx);

        bw_values[eid] =
            double(
                idx * 10);

        ++idx;
    }

    nx::set_edge_attributes(
        g,
        weight_values,
        "weight");

    nx::set_edge_attributes(
        g,
        bw_values,
        "bw");

    //
    // dump graph
    //

    std::cout
    << "SCAN_FULL(ms) "
    << benchmark_scan(g)
    << '\n';

std::cout
    << "BFS_NOPRED(ms) "
    << benchmark_bfs_nopred(g)
    << '\n';

    std::ofstream dump(
        "graph_dump_cpp.txt");

    dump
        << std::fixed
        << std::setprecision(
               12);

    dump
        << "NUM_NODES "
        << g.num_nodes()
        << '\n';

    dump
        << "NUM_EDGES "
        << g.num_edges()
        << '\n';

    //
    // node dump
    //

    auto cpu =
        nx::get_node_attributes(
            g,
            "cpu");

    auto gpu =
        nx::get_node_attributes(
            g,
            "gpu");

    for (Vertex v = 0;
         v < g.num_nodes();
         ++v)
    {
        dump
            << "NODE "
            << v
            << '\n';

        dump
            << "CPU "
            << std::get<int64_t>(
                   cpu[v])
            << '\n';

        dump
            << "GPU "
            << std::get<int64_t>(
                   gpu[v])
            << '\n';
    }

    //
    // edge dump
    //

    auto weights =
        nx::get_edge_attributes(
            g,
            "weight");

    auto bw =
        nx::get_edge_attributes(
            g,
            "bw");

    auto [eit2, eend2] =
        g.edges();

    for (; eit2 != eend2; ++eit2)
    {
        Edge e =
            *eit2;

        uint32_t eid =
            g.edge_id(e);

        dump
            << "EDGE "
            << g.source(e)
            << ' '
            << g.target(e)
            << '\n';

        dump
            << "EDGE_ID "
            << eid
            << '\n';

        dump
            << "WEIGHT "
            << std::get<double>(
                   weights[eid])
            << '\n';

        dump
            << "BW "
            << std::get<double>(
                   bw[eid])
            << '\n';
    }

    //
    // attributes tests
    //

    std::cout
        << "\n===== ATTRIBUTES =====\n";

    CHECK(
        cpu.size()
        ==
        g.num_nodes());

    CHECK(
        gpu.size()
        ==
        g.num_nodes());

    CHECK(
        weights.size()
        ==
        g.num_edges());

    CHECK(
        bw.size()
        ==
        g.num_edges());

    //
    // adjacency matrix
    //

    auto t0 =
        Clock::now();

    SparseMatrix A =
        nx::adjacency_matrix(
            g);

    double adjacency_ms =
        elapsed_ms(
            t0);

    CHECK(
        A.rows
        ==
        g.num_nodes());

    CHECK(
        A.cols
        ==
        g.num_nodes());

    CHECK(
        A.nnz()
        ==
        g.num_edges() * 2);

    CSRMatrix A_csr =
        A.to_csr();

    CHECK(
        A_csr.nnz()
        ==
        A.nnz());

    //
    // weighted matrix
    //

     t0 =
    Clock::now();

for (Vertex source = 0;
     source < g.num_nodes();
     ++source)
{
    auto result =
        nx::single_source_shortest_path_length(
            g,
            source);

    volatile size_t sz =
        result.size();

    (void)sz;
}

auto t1 =
    Clock::now();

std::cout
    << "NX_BFS_FULL(ms) "
    << std::chrono::duration<double,
       std::milli>(
           t1 - t0)
           .count()
    << '\n';

    t0 =
        Clock::now();

    SparseMatrix W =
        nx::attr_sparse_matrix(
            g,
            "weight");

    double weighted_ms =
        elapsed_ms(
            t0);

    CHECK(
        W.nnz()
        ==
        A.nnz());

    CSRMatrix W_csr =
        W.to_csr();

    CHECK(
        W_csr.nnz()
        ==
        W.nnz());

    //
    // dump sparse
    //

    dump
        << "ADJ_NNZ "
        << A.nnz()
        << '\n';

    for (size_t i = 0;
         i < A.nnz();
         ++i)
    {
        dump
            << "ADJ "
            << A.row[i]
            << ' '
            << A.col[i]
            << ' '
            << A.value[i]
            << '\n';
    }

    dump
        << "WEIGHT_NNZ "
        << W.nnz()
        << '\n';

    for (size_t i = 0;
         i < W.nnz();
         ++i)
    {
        dump
            << "W "
            << W.row[i]
            << ' '
            << W.col[i]
            << ' '
            << W.value[i]
            << '\n';
    }

    //
    // centrality
    //

    t0 =
        Clock::now();

    auto degree =
        nx::degree_centrality(
            g);

    double degree_ms =
        elapsed_ms(
            t0);

    t0 =
        Clock::now();

    auto eigen =
        nx::eigenvector_centrality(
            g);

    double eigen_ms =
        elapsed_ms(
            t0);

    t0 =
        Clock::now();

    auto close =
        nx::closeness_centrality(
            g);
    t0 =
    Clock::now();

    for (Vertex source = 0;
        source < g.num_nodes();
        ++source)
    {
        volatile auto result =
            bfs(
                g,
                source);
    }

    double bfs_full_ms =
        elapsed_ms(
            t0);

    double close_ms =
        elapsed_ms(
            t0);

    t0 =
        Clock::now();

    auto between =
        nx::betweenness_centrality(
            g,
            "weight");

    double between_ms =
        elapsed_ms(
            t0);

    CHECK(
        degree.size()
        ==
        g.num_nodes());

    CHECK(
        eigen.size()
        ==
        g.num_nodes());

    CHECK(
        close.size()
        ==
        g.num_nodes());

    CHECK(
        between.size()
        ==
        g.num_nodes());

    //
    // eigen norm
    //

    double norm = 0.0;

    for (double x :
         eigen)
    {
        norm +=
            x * x;
    }

    CHECK(
        std::abs(
            norm - 1.0)
        <
        1e-6);

    //
    // centrality dump
    //

    for (Vertex v = 0;
         v < g.num_nodes();
         ++v)
    {
        dump
            << "DEGREE "
            << v
            << ' '
            << degree[v]
            << '\n';

        dump
            << "EIGEN "
            << v
            << ' '
            << eigen[v]
            << '\n';

        dump
            << "CLOSE "
            << v
            << ' '
            << close[v]
            << '\n';

        dump
            << "BETWEEN "
            << v
            << ' '
            << between[v]
            << '\n';
    }

    dump.close();

    //
    // ranking
    //

    std::cout
        << "\n===== CENTRALITY =====\n";

    for (Vertex v = 0;
         v < g.num_nodes();
         ++v)
    {
        std::cout
            << "NODE "
            << v
            << " DEG="
            << degree[v]
            << " EIG="
            << eigen[v]
            << " CLO="
            << close[v]
            << " BTW="
            << between[v]
            << '\n';
    }

    //
    // performance
    //

    std::cout
        << "\n===== PERFORMANCE =====\n";

    std::cout
        << "ADJ(ms)       "
        << adjacency_ms
        << '\n';

    std::cout
        << "ATTR(ms)      "
        << weighted_ms
        << '\n';

    std::cout
        << "DEGREE(ms)    "
        << degree_ms
        << '\n';

    std::cout
        << "EIGEN(ms)     "
        << eigen_ms
        << '\n';

    std::cout
        << "CLOSE(ms)     "
        << close_ms
        << '\n';

    std::cout
        << "BETWEEN(ms)   "
        << between_ms
        << '\n';

    std::cout
    << "BFS_FULL(ms)  "
    << bfs_full_ms
    << '\n';
    std::cout
        << "\n===== OUTPUT =====\n";

    std::cout
        << "graph_dump_cpp.txt\n";

    std::cout
        << "\n===== ALL TESTS PASSED =====\n";

    

    return 0;
}