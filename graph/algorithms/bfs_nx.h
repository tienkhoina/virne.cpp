
#pragma once

#include "../graph.h"

#include <unordered_map>

std::unordered_map<
    Vertex,
    size_t>
bfs_nx(
    const Graph& g,
    Vertex source);


