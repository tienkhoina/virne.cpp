
#pragma once

#include "../graph.h"
#include "../nx/ordered_vertex_map.h"

nx::OrderedVertexMap<size_t>
bfs_nx(
    const Graph& g,
    Vertex source);

nx::OrderedVertexMap<size_t>
bfs_nx(
    const DiGraph& g,
    Vertex source);
