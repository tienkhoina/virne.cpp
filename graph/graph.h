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

    const AttrMap&
    node_attrs(Vertex v) const;

    AttrMap&
    edge_attrs(Edge e);

    const AttrMap&
    edge_attrs(Edge e) const;

    size_t num_nodes() const;

    size_t num_edges() const;

    size_t degree(
    Vertex v) const;

    std::pair<
        boost::graph_traits<BGLGraph>::adjacency_iterator,
        boost::graph_traits<BGLGraph>::adjacency_iterator>
    neighbors(
        Vertex v) const;

    std::pair<
        boost::graph_traits<BGLGraph>::vertex_iterator,
        boost::graph_traits<BGLGraph>::vertex_iterator>
    nodes() const;

    std::pair<
        boost::graph_traits<BGLGraph>::edge_iterator,
        boost::graph_traits<BGLGraph>::edge_iterator>
    edges() const;

    bool has_edge(
        Vertex u,
        Vertex v) const;

    Edge edge(
        Vertex u,
        Vertex v) const;

    bool remove_edge(
        Vertex u,
        Vertex v);

    Vertex source(
        Edge e) const;

    Vertex target(
        Edge e) const;

    const RawNeighborList&
    neighbors_fast(
        Vertex v) const;

    BGLGraph&
    raw();

    const BGLGraph&
    raw() const;

    

private:

    BGLGraph graph_;
};