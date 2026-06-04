#pragma once

#include "../graph.h"

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