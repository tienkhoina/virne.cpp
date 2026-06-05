#pragma once

#include "../graph.h"
#include <vector>

struct BFSStats
{
    double sum_dist = 0.0;
    size_t reachable = 0;
};

struct BFSWorkspace
{
    std::vector<size_t>
        dist;

    std::vector<Vertex>
        queue;

    explicit BFSWorkspace(
        size_t n)
        :
        dist(n),
        queue(n)
    {}
};

BFSStats
bfs_stats(
    const Graph& g,
    Vertex source,
    BFSWorkspace& ws);