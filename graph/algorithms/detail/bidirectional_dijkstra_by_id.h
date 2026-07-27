#pragma once

#include "../bidirectional_dijkstra.h"

// Internal indexed entry points for composite algorithms.  Public callers
// keep the NetworkX-shaped string overloads; a caller that already resolved
// the attribute must not pay another registry lookup for each nested search.
namespace graph_detail
{

BidirectionalPathResult bidirectional_dijkstra_by_id(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    AttrId weight_attr_id);

BidirectionalPathResult bidirectional_dijkstra_by_id(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    AttrId weight_attr_id);

} // namespace graph_detail
