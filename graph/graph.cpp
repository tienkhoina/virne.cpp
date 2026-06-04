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

const AttrMap&
Graph::node_attrs(
    Vertex v) const
{
    return graph_[v].attrs;
}


AttrMap&
Graph::edge_attrs(
    Edge e)
{
    return graph_[e].attrs;
}

const AttrMap&
Graph::edge_attrs(
    Edge e) const
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

size_t Graph::degree(
    Vertex v) const
{
    return boost::degree(
        v,
        graph_);
}

auto Graph::neighbors(
    Vertex v) const
    ->
    std::pair<
        boost::graph_traits<BGLGraph>::adjacency_iterator,
        boost::graph_traits<BGLGraph>::adjacency_iterator>
{
    return boost::adjacent_vertices(
        v,
        graph_);
}

auto Graph::nodes() const
    ->
    std::pair<
        boost::graph_traits<BGLGraph>::vertex_iterator,
        boost::graph_traits<BGLGraph>::vertex_iterator>
{
    return boost::vertices(
        graph_);
}

auto Graph::edges() const
    ->
    std::pair<
        boost::graph_traits<BGLGraph>::edge_iterator,
        boost::graph_traits<BGLGraph>::edge_iterator>
{
    return boost::edges(
        graph_);
}

bool Graph::has_edge(
    Vertex u,
    Vertex v) const
{
    return
        boost::edge(
            u,
            v,
            graph_)
        .second;
}

Edge Graph::edge(
    Vertex u,
    Vertex v) const
{
    auto [e, ok] =
        boost::edge(
            u,
            v,
            graph_);

    if (!ok)
    {
        throw std::runtime_error(
            "Edge not found");
    }

    return e;
}

bool Graph::remove_edge(
    Vertex u,
    Vertex v)
{
    auto before =
        num_edges();

    boost::remove_edge(
        u,
        v,
        graph_);

    return
        num_edges()
        <
        before;
}

Vertex Graph::source(
    Edge e) const
{
    return boost::source(
        e,
        graph_);
}

Vertex Graph::target(
    Edge e) const
{
    return boost::target(
        e,
        graph_);
}

const RawNeighborList&
Graph::neighbors_fast(
    Vertex v) const
{
    return
        graph_
            .m_vertices[v]
            .m_out_edges;
}

BGLGraph&
Graph::raw()
{
    return graph_;
}



const BGLGraph&
Graph::raw() const
{
    return graph_;
}