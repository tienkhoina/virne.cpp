#pragma once

#include "../graph.h"

#include <vector>
#include <cstddef>

struct BFSResult
{
    std::vector<size_t> distance;

    std::vector<Vertex> predecessor;
};

BFSResult bfs(
    const Graph& g,
    Vertex source);