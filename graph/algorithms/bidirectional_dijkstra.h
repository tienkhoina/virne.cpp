// graph/algorithms/bidirectional_dijkstra.h
#pragma once

#include "dijkstra.h"

#include <limits>
#include <string>
#include <vector>

struct BidirectionalPathResult
{
    bool found = false;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<Vertex> path;
};

BidirectionalPathResult bidirectional_dijkstra(
    const Graph& g,
    Vertex source,
    Vertex target,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");

BidirectionalPathResult bidirectional_dijkstra(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");

BidirectionalPathResult bidirectional_dijkstra(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");

BidirectionalPathResult bidirectional_dijkstra(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    const VertexSet& banned_vertices = {},
    const EdgeSet& banned_edges = {},
    const std::string& weight_attr = "weight");
