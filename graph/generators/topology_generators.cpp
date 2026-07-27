#include "topology_generators.h"

#include "../nx/connectivity.h"
#include "../../random/py_random.h"
#include "../../random/random_context.h"

#include <limits>
#include <stdexcept>

namespace
{

Graph empty_graph(
    size_t num_nodes)
{
    Graph graph;
    for (size_t i = 0;
         i < num_nodes;
         ++i)
    {
        graph.add_node();
    }
    return graph;
}

DiGraph empty_digraph(
    size_t num_nodes)
{
    DiGraph graph;
    for (size_t i = 0;
         i < num_nodes;
         ++i)
    {
        graph.add_node();
    }
    return graph;
}

Graph erdos_renyi_graph_impl(
    size_t num_nodes,
    double probability,
    PyRandom& random)
{
    Graph graph = empty_graph(num_nodes);

    if (probability <= 0.0)
    {
        return graph;
    }

    if (probability >= 1.0)
    {
        for (Vertex u = 0;
             u < num_nodes;
             ++u)
        {
            for (Vertex v = u + 1;
                 v < num_nodes;
                 ++v)
            {
                graph.add_edge(u, v);
            }
        }
        return graph;
    }

    // itertools.combinations(range(n), 2), matching NetworkX 3.4's RNG
    // consumption and insertion order.
    for (Vertex u = 0;
         u < num_nodes;
         ++u)
    {
        for (Vertex v = u + 1;
             v < num_nodes;
             ++v)
        {
            if (random.random() < probability)
            {
                graph.add_edge(u, v);
            }
        }
    }

    return graph;
}

DiGraph erdos_renyi_digraph_impl(
    size_t num_nodes,
    double probability,
    PyRandom& random)
{
    DiGraph graph =
        empty_digraph(num_nodes);

    if (probability <= 0.0)
    {
        return graph;
    }

    for (Vertex u = 0;
         u < num_nodes;
         ++u)
    {
        for (Vertex v = 0;
             v < num_nodes;
             ++v)
        {
            if (u == v)
            {
                continue;
            }

            if (probability >= 1.0 ||
                random.random() < probability)
            {
                graph.add_edge(u, v);
            }
        }
    }

    return graph;
}

} // namespace

namespace nx
{

Graph path_graph(
    size_t num_nodes)
{
    Graph graph = empty_graph(num_nodes);
    for (Vertex v = 1;
         v < num_nodes;
         ++v)
    {
        graph.add_edge(v - 1, v);
    }
    return graph;
}

Graph star_graph(
    size_t outer_nodes)
{
    Graph graph =
        empty_graph(outer_nodes + 1);

    for (Vertex leaf = 1;
         leaf <= outer_nodes;
         ++leaf)
    {
        graph.add_edge(0, leaf);
    }
    return graph;
}

Graph grid_2d_graph(
    size_t rows,
    size_t columns,
    bool periodic)
{
    if (rows != 0 &&
        columns >
            std::numeric_limits<size_t>::max() /
                rows)
    {
        throw std::overflow_error(
            "grid_2d_graph node count overflows size_t");
    }

    Graph graph =
        empty_graph(rows * columns);

    auto index =
        [columns](size_t row, size_t column)
        {
            return static_cast<Vertex>(
                row * columns + column);
        };

    // NetworkX's public EdgeView groups by node order and retains each node's
    // adjacency insertion order. Emit down then right for every row-major
    // node so Graph::edges() has the same observable sequence.
    for (size_t row = 0;
         row < rows;
         ++row)
    {
        for (size_t column = 0;
             column < columns;
             ++column)
        {
            if (row + 1 < rows)
            {
                graph.add_edge(
                    index(row, column),
                    index(row + 1, column));
            }
            if (column + 1 < columns)
            {
                graph.add_edge(
                    index(row, column),
                    index(row, column + 1));
            }
        }
    }

    // NetworkX closes a dimension only when it contains at least three
    // nodes, avoiding duplicate edges for one- and two-node dimensions.
    if (periodic && rows > 2)
    {
        for (size_t column = 0;
             column < columns;
             ++column)
        {
            graph.add_edge(
                index(0, column),
                index(rows - 1, column));
        }
    }
    if (periodic && columns > 2)
    {
        for (size_t row = 0;
             row < rows;
             ++row)
        {
            graph.add_edge(
                index(row, 0),
                index(row, columns - 1));
        }
    }

    return graph;
}

Graph erdos_renyi_graph(
    size_t num_nodes,
    double probability)
{
    return erdos_renyi_graph(
        num_nodes,
        probability,
        global_py_random());
}

Graph erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    uint64_t seed)
{
    PyRandom random(seed);
    return erdos_renyi_graph(
        num_nodes,
        probability,
        random);
}

Graph erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    PyRandom& random)
{
    return erdos_renyi_graph_impl(
        num_nodes,
        probability,
        random);
}

DiGraph erdos_renyi_digraph(
    size_t num_nodes,
    double probability)
{
    return erdos_renyi_digraph(
        num_nodes,
        probability,
        global_py_random());
}

DiGraph erdos_renyi_digraph(
    size_t num_nodes,
    double probability,
    uint64_t seed)
{
    PyRandom random(seed);
    return erdos_renyi_digraph(
        num_nodes,
        probability,
        random);
}

DiGraph erdos_renyi_digraph(
    size_t num_nodes,
    double probability,
    PyRandom& random)
{
    return erdos_renyi_digraph_impl(
        num_nodes,
        probability,
        random);
}

Graph connected_retry(
    const std::function<Graph(size_t)>& generator,
    size_t max_attempts)
{
    if (!generator)
    {
        throw std::invalid_argument(
            "connected_retry requires a generator");
    }
    if (max_attempts == 0)
    {
        throw std::invalid_argument(
            "connected_retry requires max_attempts > 0");
    }

    for (size_t attempt = 0;
         attempt < max_attempts;
         ++attempt)
    {
        Graph graph = generator(attempt);
        if (is_connected(graph))
        {
            return graph;
        }
    }

    throw std::runtime_error(
        "Unable to generate a connected graph within max_attempts");
}

Graph connected_erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    uint64_t seed,
    size_t max_attempts)
{
    PyRandom random(seed);
    return connected_erdos_renyi_graph(
        num_nodes,
        probability,
        random,
        max_attempts);
}

Graph connected_erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    PyRandom& random,
    size_t max_attempts)
{
    if (num_nodes > 1 &&
        probability <= 0.0)
    {
        throw std::runtime_error(
            "A connected graph is impossible when p <= 0 and n > 1");
    }

    return connected_retry(
        [&](size_t)
        {
            return erdos_renyi_graph_impl(
                num_nodes,
                probability,
                random);
        },
        max_attempts);
}

Graph connected_waxman_graph(
    const WaxmanConfig& config,
    size_t max_attempts)
{
    PyRandom random(config.seed);
    return connected_waxman_graph(
        config,
        random,
        max_attempts);
}

Graph connected_waxman_graph(
    const WaxmanConfig& config,
    PyRandom& random,
    size_t max_attempts)
{
    return connected_retry(
        [&](size_t)
        {
            return WaxmanGenerator::generate(
                config,
                random);
        },
        max_attempts);
}

} // namespace nx
