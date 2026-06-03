#pragma once

#include "attribute.h"

#include <boost/graph/adjacency_list.hpp>

struct VertexProperty
{
    AttrMap attrs;
};

struct EdgeProperty
{
    AttrMap attrs;
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