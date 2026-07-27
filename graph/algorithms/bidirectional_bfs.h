#pragma once

#include "../graph.h"
#include "search_mask.h"

#include <vector>

struct BidirectionalBFSResult
{
    bool found = false;

    size_t distance = 0;

    std::vector<Vertex> path;
};

BidirectionalBFSResult
bidirectional_bfs(
    const Graph& g,
    Vertex source,
    Vertex target);

BidirectionalBFSResult
bidirectional_bfs(
    const DiGraph& g,
    Vertex source,
    Vertex target);

BidirectionalBFSResult
bidirectional_bfs(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask);

BidirectionalBFSResult
bidirectional_bfs(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask);
