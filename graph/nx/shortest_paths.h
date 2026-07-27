#pragma once

#include "../graph.h"
#include "../distance_matrix.h"
#include "../sparse_matrix.h"
#include "../algorithms/k_shortest_paths.h"
#include "ordered_vertex_map.h"
#include "subgraph.h"

#include <vector>
#include <string>
#include <optional>
#include <string_view>
#include <utility>


namespace nx
{

using SingleSourcePathLengths =
    OrderedVertexMap<size_t>;

using AllPairsPathLengths =
    OrderedVertexMap<SingleSourcePathLengths>;

using SingleSourcePaths =
    OrderedVertexMap<std::vector<Vertex>>;

using AllPairsPaths =
    OrderedVertexMap<SingleSourcePaths>;

using SingleSourceDijkstraPathLengths =
    OrderedVertexMap<double>;

size_t shortest_path_length(
    const Graph& g,
    Vertex source,
    Vertex target);

size_t shortest_path_length(
    const DiGraph& g,
    Vertex source,
    Vertex target);

size_t shortest_path_length(
    const GraphView& view,
    Vertex source,
    Vertex target);

size_t shortest_path_length(
    const DiGraphView& view,
    Vertex source,
    Vertex target);

SingleSourcePathLengths shortest_path_length(
    const Graph& g,
    Vertex source);

SingleSourcePathLengths shortest_path_length(
    const DiGraph& g,
    Vertex source);

SingleSourcePathLengths shortest_path_length(
    const GraphView& view,
    Vertex source);

SingleSourcePathLengths shortest_path_length(
    const DiGraphView& view,
    Vertex source);

AllPairsPathLengths shortest_path_length(
    const Graph& g);

AllPairsPathLengths shortest_path_length(
    const DiGraph& g);

AllPairsPathLengths shortest_path_length(
    const GraphView& view);

AllPairsPathLengths shortest_path_length(
    const DiGraphView& view);

std::vector<Vertex> shortest_path(
    const Graph& g,
    Vertex source,
    Vertex target);

std::vector<Vertex> shortest_path(
    const DiGraph& g,
    Vertex source,
    Vertex target);

std::vector<Vertex> shortest_path(
    const GraphView& view,
    Vertex source,
    Vertex target);

std::vector<Vertex> shortest_path(
    const DiGraphView& view,
    Vertex source,
    Vertex target);

SingleSourcePaths shortest_path(
    const Graph& g,
    Vertex source);

SingleSourcePaths shortest_path(
    const DiGraph& g,
    Vertex source);

SingleSourcePaths shortest_path(
    const GraphView& view,
    Vertex source);

SingleSourcePaths shortest_path(
    const DiGraphView& view,
    Vertex source);

AllPairsPaths shortest_path(
    const Graph& g);

AllPairsPaths shortest_path(
    const DiGraph& g);

AllPairsPaths shortest_path(
    const GraphView& view);

AllPairsPaths shortest_path(
    const DiGraphView& view);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const GraphView& view,
    Vertex source,
    Vertex target);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight);

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight);

SingleSourcePathLengths
single_source_shortest_path_length(
    const Graph& g,
    Vertex source,
    std::optional<double> cutoff = std::nullopt);

SingleSourcePathLengths
single_source_shortest_path_length(
    const DiGraph& g,
    Vertex source,
    std::optional<double> cutoff = std::nullopt);

SingleSourcePathLengths
single_source_shortest_path_length(
    const GraphView& view,
    Vertex source,
    std::optional<double> cutoff = std::nullopt);

SingleSourcePathLengths
single_source_shortest_path_length(
    const DiGraphView& view,
    Vertex source,
    std::optional<double> cutoff = std::nullopt);


std::vector<Vertex> dijkstra_path(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

std::vector<Vertex> dijkstra_path(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

std::vector<Vertex> dijkstra_path(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

std::vector<Vertex> dijkstra_path(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

double dijkstra_path_length(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

double dijkstra_path_length(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

double dijkstra_path_length(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

double dijkstra_path_length(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const Graph& g,
    Vertex source,
    std::optional<double> cutoff = std::nullopt,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const DiGraph& g,
    Vertex source,
    std::optional<double> cutoff = std::nullopt,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const GraphView& view,
    Vertex source,
    std::optional<double> cutoff = std::nullopt,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const DiGraphView& view,
    Vertex source,
    std::optional<double> cutoff = std::nullopt,
    std::optional<std::string_view> weight =
        std::string_view{"weight"});

inline SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const Graph& g,
    Vertex source,
    std::string_view weight)
{
    return single_source_dijkstra_path_length(
        g, source, std::nullopt, weight);
}

inline SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const DiGraph& g,
    Vertex source,
    std::string_view weight)
{
    return single_source_dijkstra_path_length(
        g, source, std::nullopt, weight);
}

inline SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const GraphView& view,
    Vertex source,
    std::string_view weight)
{
    return single_source_dijkstra_path_length(
        view, source, std::nullopt, weight);
}

inline SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const DiGraphView& view,
    Vertex source,
    std::string_view weight)
{
    return single_source_dijkstra_path_length(
        view, source, std::nullopt, weight);
}

DistanceMatrix floyd_warshall(
    const Graph& g,
    const std::string& weight_attr =
        "weight");

DistanceMatrix floyd_warshall(
    const DiGraph& g,
    const std::string& weight_attr =
        "weight");

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "");

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "");

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "");

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "");

ShortestSimplePathGenerator
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options = {});

ShortestSimplePathGenerator
shortest_simple_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options = {});

ShortestSimplePathGenerator
shortest_simple_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options = {});

ShortestSimplePathGenerator
shortest_simple_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options = {});

// NetworkX-shaped fourth argument. The three-argument spelling is provided by
// the options overload above; this overload preserves `weight=None` versus an
// attribute-name string without exposing that distinction in hot loops.
inline ShortestSimplePathGenerator shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    ShortestSimplePathOptions options;
    if (weight)
    {
        options.weight_attr = std::string(*weight);
    }
    return shortest_simple_paths(
        g, source, target, std::move(options));
}

inline ShortestSimplePathGenerator shortest_simple_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    ShortestSimplePathOptions options;
    if (weight)
    {
        options.weight_attr = std::string(*weight);
    }
    return shortest_simple_paths(
        g, source, target, std::move(options));
}

inline ShortestSimplePathGenerator shortest_simple_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    ShortestSimplePathOptions options;
    if (weight)
    {
        options.weight_attr = std::string(*weight);
    }
    return shortest_simple_paths(
        view, source, target, std::move(options));
}

inline ShortestSimplePathGenerator shortest_simple_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    ShortestSimplePathOptions options;
    if (weight)
    {
        options.weight_attr = std::string(*weight);
    }
    return shortest_simple_paths(
        view, source, target, std::move(options));
}


}
