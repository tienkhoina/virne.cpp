#pragma once

#include "attribute.h"

#include <boost/graph/adjacency_list.hpp>
#define private public
#define protected public


#undef private
#undef protected

struct VertexProperty
{
    AttrMap attrs;
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

using Vertex =
    boost::graph_traits<
        BGLGraph>::vertex_descriptor;

using Edge =
    boost::graph_traits<
        BGLGraph>::edge_descriptor;

using RawNeighborList =
    decltype(
        std::declval<BGLGraph>()
            .m_vertices[0]
            .m_out_edges);

using RawNeighbor =
    typename RawNeighborList::value_type;