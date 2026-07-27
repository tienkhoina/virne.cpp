#pragma once

#include "../graph.h"

namespace nx
{

// NetworkX free-function spellings used by the original Virne code. The
// member APIs remain available and have identical ordering.
inline auto neighbors(
    const Graph& graph,
    Vertex node)
{
    return graph.neighbors(node);
}

inline auto neighbors(
    const DiGraph& graph,
    Vertex node)
{
    return graph.neighbors(node);
}

inline DegreeItems degree(
    const Graph& graph)
{
    return graph.degree();
}

inline DegreeItems degree(
    const DiGraph& graph)
{
    return graph.degree();
}

inline DegreeItems in_degree(
    const DiGraph& graph)
{
    return graph.in_degree();
}

inline DegreeItems out_degree(
    const DiGraph& graph)
{
    return graph.out_degree();
}

} // namespace nx
