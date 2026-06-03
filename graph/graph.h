#pragma once

#include "graph_types.h"

class Graph
{
public:

    Vertex add_node();

    Edge add_edge(
        Vertex u,
        Vertex v);

    AttrMap&
    node_attrs(Vertex v);

    AttrMap&
    edge_attrs(Edge e);

    size_t num_nodes() const;

    size_t num_edges() const;

    BGLGraph&
    raw();

private:

    BGLGraph graph_;
};