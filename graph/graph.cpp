#include "graph.h"

#include <stdexcept>

Vertex Graph::add_node()
{
    Vertex v =
        boost::add_vertex(
            graph_);

    graph_[v].attrs.bind(
        attr_registry_);

    return v;
}

Edge Graph::add_edge(
    Vertex u,
    Vertex v)
{
    auto [existing, exists] =
        boost::edge(
            u,
            v,
            graph_);

    if (exists)
    {
        return existing;
    }

    auto [e, ok] =
        boost::add_edge(
            u,
            v,
            graph_);

    if (!ok)
    {
        throw std::runtime_error(
            "Failed to add edge");
    }

    graph_[e].attrs.bind(
        attr_registry_);

    const uint32_t id =
        next_edge_id_++;

    graph_[e].edge_id =
        id;

    if (id >= edge_endpoints_.size())
    {
        edge_endpoints_.resize(
            id + 1);
    }

    edge_endpoints_[id] =
        std::pair<Vertex, Vertex>{
            u,
            v};

    return e;
}

AttrMap& Graph::node_attrs(
    Vertex v)
{
    return graph_[v].attrs;
}

const AttrMap& Graph::node_attrs(
    Vertex v) const
{
    return graph_[v].attrs;
}

AttrMap& Graph::edge_attrs(
    Edge e)
{
    return graph_[e].attrs;
}

const AttrMap& Graph::edge_attrs(
    Edge e) const
{
    return graph_[e].attrs;
}

size_t Graph::num_nodes() const
{
    return boost::num_vertices(
        graph_);
}

size_t Graph::num_edges() const
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
        boost::graph_traits<
            BGLGraph>::adjacency_iterator,
        boost::graph_traits<
            BGLGraph>::adjacency_iterator>
{
    return boost::adjacent_vertices(
        v,
        graph_);
}

auto Graph::nodes() const
    ->
    std::pair<
        boost::graph_traits<
            BGLGraph>::vertex_iterator,
        boost::graph_traits<
            BGLGraph>::vertex_iterator>
{
    return boost::vertices(
        graph_);
}

auto Graph::edges() const
    ->
    std::pair<
        boost::graph_traits<
            BGLGraph>::edge_iterator,
        boost::graph_traits<
            BGLGraph>::edge_iterator>
{
    return boost::edges(
        graph_);
}

bool Graph::has_edge(
    Vertex u,
    Vertex v) const
{
    return boost::edge(
        u,
        v,
        graph_).second;
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
    auto [e, ok] =
        boost::edge(
            u,
            v,
            graph_);

    if (!ok)
    {
        return false;
    }

    const uint32_t id =
        graph_[e].edge_id;

    if (id < edge_endpoints_.size())
    {
        edge_endpoints_[id].reset();
    }

    boost::remove_edge(
        u,
        v,
        graph_);

    return true;
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

uint32_t Graph::edge_id(
    Edge e) const
{
    return graph_[e].edge_id;
}

Edge Graph::edge_by_id(
    uint32_t id) const
{
    if (id >= edge_endpoints_.size() ||
        !edge_endpoints_[id].has_value())
    {
        throw std::runtime_error(
            "Edge id not found");
    }

    auto [u, v] =
        *edge_endpoints_[id];

    return edge(
        u,
        v);
}

std::pair<Vertex, Vertex>
Graph::edge_endpoints(
    uint32_t id) const
{
    if (id >= edge_endpoints_.size() ||
        !edge_endpoints_[id].has_value())
    {
        throw std::runtime_error(
            "Edge id not found");
    }

    return *edge_endpoints_[id];
}

AttrId Graph::attr_id(
    std::string_view name) const
{
    return attr_registry_->intern(
        name);
}

std::string_view Graph::attr_name(
    AttrId id) const
{
    return attr_registry_->name(
        id);
}

const AttributeRegistry&
Graph::attribute_registry() const
{
    return *attr_registry_;
}

BGLGraph& Graph::raw()
{
    return graph_;
}

const BGLGraph& Graph::raw() const
{
    return graph_;
}
