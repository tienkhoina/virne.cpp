#pragma once

#include "attribute.h"

#include <boost/version.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <type_traits>
#include <utility>

// Intentional ABI/layout coupling for the allocation-free hot path below.
// Upgrade Boost only together with a full graph parity/benchmark run.
static_assert(
    BOOST_VERSION == 108500,
    "Virne graph fast-neighbor access is pinned to Boost 1.85.0");

// Boost 1.85 exposes adjacency_list::m_vertices for these selectors.  Accessing
// it is still an implementation-layout hack (not a documented BGL contract),
// hence the hard version guard above.

struct VertexProperty
{
    AttrMap attrs;
    // Graph collapses Boost's duplicate undirected self-loop incidence record
    // to one NetworkX-style neighbor.  Keep this bit so degree() can restore
    // the NetworkX convention that a self-loop contributes two to degree.
    bool has_self_loop = false;
};

struct EdgeProperty
{
    AttrMap attrs;
    uint32_t edge_id = 0;
};

using BGLGraph =
    boost::adjacency_list<
        boost::vecS,
        boost::vecS,
        boost::undirectedS,
        VertexProperty,
        EdgeProperty>;

using BGLDiGraph =
    boost::adjacency_list<
        boost::vecS,
        boost::vecS,
        boost::bidirectionalS,
        VertexProperty,
        EdgeProperty>;

using Vertex =
    boost::graph_traits<
        BGLGraph>::vertex_descriptor;

using Edge =
    boost::graph_traits<
        BGLGraph>::edge_descriptor;

using DiEdge =
    boost::graph_traits<
        BGLDiGraph>::edge_descriptor;

// The descriptor property pointer and stored-vertex bundle are intentionally
// used directly by graph.cpp/view hot paths. These assertions turn a Boost
// layout change into a compile error instead of silent memory corruption.
static_assert(
    std::is_same_v<
        decltype(std::declval<Edge&>().get_property()),
        void*>,
    "Boost 1.85 edge descriptor property pointer layout changed");
static_assert(
    std::is_same_v<
        decltype(std::declval<DiEdge&>().get_property()),
        void*>,
    "Boost 1.85 directed edge descriptor property pointer layout changed");
static_assert(
    std::is_same_v<
        decltype((std::declval<BGLGraph&>()
                      .m_vertices[0]
                      .m_property)),
        VertexProperty&>,
    "Boost 1.85 stored vertex property layout changed");
static_assert(
    std::is_same_v<
        decltype((std::declval<BGLDiGraph&>()
                      .m_vertices[0]
                      .m_property)),
        VertexProperty&>,
    "Boost 1.85 directed stored vertex property layout changed");

static_assert(
    std::is_same_v<
        Vertex,
        boost::graph_traits<BGLDiGraph>::vertex_descriptor>,
    "Graph and DiGraph must share one contiguous Vertex index type");
static_assert(
    std::is_integral_v<Vertex>,
    "Vertex must remain an integral row index for tensor adapters");
static_assert(
    std::is_same_v<
        typename BGLGraph::edge_list_selector,
        boost::listS>,
    "NetworkX edge-order parity requires Boost listS global edge storage");
static_assert(
    std::is_same_v<
        typename BGLDiGraph::edge_list_selector,
        boost::listS>,
    "NetworkX arc-order parity requires Boost listS global edge storage");

using RawNeighborList =
    decltype(
        std::declval<BGLGraph>()
            .m_vertices[0]
            .m_out_edges);

using RawNeighbor =
    typename RawNeighborList::value_type;

using DiRawNeighborList =
    decltype(
        std::declval<BGLDiGraph>()
            .m_vertices[0]
            .m_out_edges);

using DiRawNeighbor =
    typename DiRawNeighborList::value_type;

using DiRawInNeighborList =
    decltype(
        std::declval<BGLDiGraph>()
            .m_vertices[0]
            .m_in_edges);

using DiRawInNeighbor =
    typename DiRawInNeighborList::value_type;

static_assert(
    std::is_same_v<
        decltype(std::declval<RawNeighbor&>().get_property()),
        EdgeProperty&>,
    "Boost 1.85 raw edge property layout changed");
static_assert(
    std::is_same_v<
        decltype(std::declval<DiRawNeighbor&>().get_property()),
        EdgeProperty&>,
    "Boost 1.85 raw directed edge property layout changed");
