#pragma once

#include "../graph.h"
#include "search_mask.h"

#include <vector>
#include <cstddef>

struct BFSResult
{
    std::vector<size_t> distance;

    std::vector<Vertex> predecessor;

    // NetworkX-compatible dict insertion order: source first, then vertices
    // when they are first discovered by the BFS neighbor scan.
    std::vector<Vertex> discovery_order;
};

BFSResult bfs(
    const Graph& g,
    Vertex source);

BFSResult bfs(
    const DiGraph& g,
    Vertex source);

BFSResult bfs(
    const Graph& g,
    Vertex source,
    const SearchMask& mask);

BFSResult bfs(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask);
