#include "shortest_paths.h"

#include "../algorithms/bfs.h"
#include "../algorithms/bidirectional_bfs.h"
#include "../algorithms/bidirectional_dijkstra.h"
#include "../algorithms/dijkstra.h"
#include "../algorithms/floyd_warshall.h"


#include <algorithm>
#include <limits>
#include <stdexcept>
#include "../algorithms/k_shortest_paths.h"
#include "../algorithms/bfs_nx.h"

namespace nx
{

size_t shortest_path_length(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    auto result =
        bidirectional_bfs(
            g,
            source,
            target);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return result.distance;
}

std::vector<Vertex> shortest_path(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    auto result =
        bidirectional_bfs(
            g,
            source,
            target);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return std::move(
        result.path);
}

std::unordered_map<Vertex, size_t>
single_source_shortest_path_length(
    const Graph& g,
    Vertex source)
{
    return bfs_nx(
        g,
        source);
}

std::vector<Vertex> dijkstra_path(
    const Graph& g,
    Vertex source,
    Vertex target,
    const std::string& weight_attr)
{
    auto result =
        bidirectional_dijkstra(
            g,
            source,
            target,
            VertexSet{},
            EdgeSet{},
            weight_attr);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return std::move(
        result.path);
}

double dijkstra_path_length(
    const Graph& g,
    Vertex source,
    Vertex target,
    const std::string& weight_attr)
{
    auto result =
        bidirectional_dijkstra(
            g,
            source,
            target,
            VertexSet{},
            EdgeSet{},
            weight_attr);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return result.cost;
}

std::unordered_map<
    Vertex,
    double>
single_source_dijkstra_path_length(
    const Graph& g,
    Vertex source,
    const std::string& weight_attr)
{
    auto result =
        dijkstra(
            g,
            source,
            weight_attr);

    std::unordered_map<
        Vertex,
        double> distances;

    for (Vertex v = 0;
         v < g.num_nodes();
         ++v)
    {
        if (
            result.distance[v]
            ==
            std::numeric_limits<double>::max())
        {
            continue;
        }

        distances[v] =
            result.distance[v];
    }

    return distances;
}

DistanceMatrix floyd_warshall(
    const Graph& g,
    const std::string& weight_attr)
{
    return ::floyd_warshall(
        g,
        weight_attr);
}

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    auto results =
        yen_k_shortest_paths(
            g,
            source,
            target,
            k,
            weight_attr);

    std::vector<
        std::vector<Vertex>
    > paths;

    paths.reserve(
        results.size());

    for (auto& r : results)
    {
        paths.push_back(
            std::move(
                r.path));
    }

    return paths;
}

}