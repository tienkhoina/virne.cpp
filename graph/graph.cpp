#include "graph.h"

Vertex Graph::add_node()
{
    return boost::add_vertex(
        graph_);
}

Edge Graph::add_edge(
    Vertex u,
    Vertex v)
{
    return boost::add_edge(
        u,
        v,
        graph_).first;
}

AttrMap&
Graph::node_attrs(
    Vertex v)
{
    return graph_[v].attrs;
}

AttrMap&
Graph::edge_attrs(
    Edge e)
{
    return graph_[e].attrs;
}

size_t
Graph::num_nodes() const
{
    return boost::num_vertices(
        graph_);
}

size_t
Graph::num_edges() const
{
    return boost::num_edges(
        graph_);
}

BGLGraph&
Graph::raw()
{
    return graph_;
}