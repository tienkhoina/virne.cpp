#pragma once

#include "graph_types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

class DiGraph;

namespace graph_views
{
template <typename GraphType>
class NodeView;

template <typename GraphType>
class EdgeView;

template <typename GraphType>
class IncidentEdgeRange;

template <typename GraphType>
class AdjacencyView;
} // namespace graph_views

using EdgeEndpoints =
    std::pair<Vertex, Vertex>;

using DegreeItems =
    std::vector<
        std::pair<Vertex, size_t>>;

struct NodeWithAttrs
{
    Vertex node = 0;
    AttrObject attrs;
};

struct EdgeWithAttrs
{
    Vertex u = 0;
    Vertex v = 0;
    AttrObject attrs;
};

class Graph
{
public:
    Graph() = default;

    explicit Graph(
        const std::vector<EdgeEndpoints>& edge_list);

    explicit Graph(
        const std::vector<EdgeWithAttrs>& edge_list);

    // Dense NumPy-compatible adjacency input.  Zero entries are omitted;
    // nonzero diagonal entries become self-loops and the weight attribute is
    // stored under `weight` (or the supplied name).
    explicit Graph(
        const std::vector<std::vector<double>>& adjacency,
        std::string_view weight_attr = "weight");

    Graph(
        size_t node_count,
        const std::vector<EdgeEndpoints>& edge_list);

    Graph(
        size_t node_count,
        const std::vector<EdgeWithAttrs>& edge_list);

    Graph(const Graph& other);

    Graph& operator=(const Graph& other);

    Graph(Graph&& other);

    Graph& operator=(Graph&& other) noexcept;

    Vertex add_node();

    void add_nodes_from(
        const std::vector<Vertex>& nodes);

    void add_nodes_from(
        const std::vector<NodeWithAttrs>& nodes);

    Edge add_edge(
        Vertex u,
        Vertex v);

    Edge add_edge(
        Vertex u,
        Vertex v,
        const AttrObject& attrs);

    void add_edges_from(
        const std::vector<EdgeEndpoints>& edges);

    void add_edges_from(
        const std::vector<EdgeWithAttrs>& edges);

    AttrMap& node_attrs(
        Vertex v) noexcept
    {
        return graph_.m_vertices[v]
            .m_property.attrs;
    }

    const AttrMap& node_attrs(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_property.attrs;
    }

    AttrMap& edge_attrs(
        Edge e) noexcept
    {
        return static_cast<EdgeProperty*>(
            e.get_property())->attrs;
    }

    const AttrMap& edge_attrs(
        Edge e) const noexcept
    {
        return static_cast<const EdgeProperty*>(
            e.get_property())->attrs;
    }

    AttrMap& graph_attrs();

    const AttrMap& graph_attrs() const;

    size_t num_nodes() const noexcept
    {
        return graph_.m_vertices.size();
    }

    size_t number_of_nodes() const noexcept
    {
        return num_nodes();
    }

    size_t num_edges() const;

    size_t number_of_edges() const noexcept
    {
        return num_edges();
    }

    size_t degree(
        Vertex v) const;

    DegreeItems degree() const;

    std::pair<
        boost::graph_traits<
            BGLGraph>::adjacency_iterator,
        boost::graph_traits<
            BGLGraph>::adjacency_iterator>
    neighbors(
        Vertex v) const;

    std::pair<
        boost::graph_traits<
            BGLGraph>::vertex_iterator,
        boost::graph_traits<
            BGLGraph>::vertex_iterator>
    nodes() const;

    std::pair<
        boost::graph_traits<
            BGLGraph>::edge_iterator,
        boost::graph_traits<
            BGLGraph>::edge_iterator>
    edges() const;

    graph_views::IncidentEdgeRange<Graph>
    edges(
        Vertex v);

    graph_views::IncidentEdgeRange<const Graph>
    edges(
        Vertex v) const;

    graph_views::NodeView<Graph>
    node_view();

    graph_views::NodeView<const Graph>
    node_view() const;

    graph_views::EdgeView<Graph>
    edge_view();

    graph_views::EdgeView<const Graph>
    edge_view() const;

    graph_views::AdjacencyView<Graph>
    adjacency_view();

    graph_views::AdjacencyView<const Graph>
    adjacency_view() const;

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
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_out_edges;
    }

    uint32_t edge_id(
        Edge e) const noexcept
    {
        return static_cast<const EdgeProperty*>(
            e.get_property())->edge_id;
    }

    Edge edge_by_id(
        uint32_t id) const;

    std::pair<
        Vertex,
        Vertex>
    edge_endpoints(
        uint32_t id) const;

    size_t edge_id_capacity() const noexcept;

    DiGraph to_directed() const;

    AttrId attr_id(
        std::string_view name) const;

    std::string_view attr_name(
        AttrId id) const;

    const AttributeRegistry&
    attribute_registry() const;

    BGLGraph& raw();

    const BGLGraph& raw() const;

private:
    void swap_state(Graph& other) noexcept;

    // edges() performs a logically-const lazy normalization of the global
    // listS order.  The storage itself must therefore be mutable; casting a
    // genuinely const Graph's non-mutable subobject and sorting it is UB.
    mutable BGLGraph graph_;

    std::shared_ptr<AttributeRegistry>
        attr_registry_ =
            std::make_shared<AttributeRegistry>();

    AttrMap graph_attrs_{attr_registry_};

    uint32_t next_edge_id_ = 0;

    std::vector<
        std::optional<
            std::pair<Vertex, Vertex>>>
        edge_endpoints_;

    // Boost listS iterators and edge descriptors survive list::sort.  The
    // global list is normalized lazily so insertion stays O(1) in the common
    // generator path while edges() matches NetworkX's node-major order.
    mutable bool edge_order_dirty_ = false;
};

class DiGraph
{
public:
    DiGraph() = default;

    explicit DiGraph(
        const std::vector<EdgeEndpoints>& edge_list);

    explicit DiGraph(
        const std::vector<EdgeWithAttrs>& edge_list);

    explicit DiGraph(
        const std::vector<std::vector<double>>& adjacency,
        std::string_view weight_attr = "weight");

    DiGraph(
        size_t node_count,
        const std::vector<EdgeEndpoints>& edge_list);

    DiGraph(
        size_t node_count,
        const std::vector<EdgeWithAttrs>& edge_list);

    DiGraph(const DiGraph& other);

    DiGraph& operator=(const DiGraph& other);

    DiGraph(DiGraph&& other);

    DiGraph& operator=(DiGraph&& other) noexcept;

    Vertex add_node();

    void add_nodes_from(
        const std::vector<Vertex>& nodes);

    void add_nodes_from(
        const std::vector<NodeWithAttrs>& nodes);

    DiEdge add_edge(
        Vertex u,
        Vertex v);

    DiEdge add_edge(
        Vertex u,
        Vertex v,
        const AttrObject& attrs);

    void add_edges_from(
        const std::vector<EdgeEndpoints>& edges);

    void add_edges_from(
        const std::vector<EdgeWithAttrs>& edges);

    AttrMap& node_attrs(
        Vertex v) noexcept
    {
        return graph_.m_vertices[v]
            .m_property.attrs;
    }

    const AttrMap& node_attrs(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_property.attrs;
    }

    AttrMap& edge_attrs(
        DiEdge e) noexcept
    {
        return static_cast<EdgeProperty*>(
            e.get_property())->attrs;
    }

    const AttrMap& edge_attrs(
        DiEdge e) const noexcept
    {
        return static_cast<const EdgeProperty*>(
            e.get_property())->attrs;
    }

    AttrMap& graph_attrs();

    const AttrMap& graph_attrs() const;

    size_t num_nodes() const noexcept
    {
        return graph_.m_vertices.size();
    }

    size_t number_of_nodes() const noexcept
    {
        return num_nodes();
    }

    size_t num_edges() const;

    size_t number_of_edges() const noexcept
    {
        return num_edges();
    }

    size_t degree(
        Vertex v) const;

    DegreeItems degree() const;

    size_t in_degree(
        Vertex v) const;

    DegreeItems in_degree() const;

    size_t out_degree(
        Vertex v) const;

    DegreeItems out_degree() const;

    std::pair<
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator,
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator>
    neighbors(
        Vertex v) const;

    std::pair<
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator,
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator>
    successors(
        Vertex v) const;

    std::pair<
        BGLDiGraph::inv_adjacency_iterator,
        BGLDiGraph::inv_adjacency_iterator>
    predecessors(
        Vertex v) const;

    std::pair<
        boost::graph_traits<
            BGLDiGraph>::out_edge_iterator,
        boost::graph_traits<
            BGLDiGraph>::out_edge_iterator>
    out_edges(
        Vertex v) const;

    std::pair<
        boost::graph_traits<
            BGLDiGraph>::in_edge_iterator,
        boost::graph_traits<
            BGLDiGraph>::in_edge_iterator>
    in_edges(
        Vertex v) const;

    std::pair<
        boost::graph_traits<
            BGLDiGraph>::vertex_iterator,
        boost::graph_traits<
            BGLDiGraph>::vertex_iterator>
    nodes() const;

    std::pair<
        boost::graph_traits<
            BGLDiGraph>::edge_iterator,
        boost::graph_traits<
            BGLDiGraph>::edge_iterator>
    edges() const;

    graph_views::IncidentEdgeRange<DiGraph>
    edges(
        Vertex v);

    graph_views::IncidentEdgeRange<const DiGraph>
    edges(
        Vertex v) const;

    graph_views::NodeView<DiGraph>
    node_view();

    graph_views::NodeView<const DiGraph>
    node_view() const;

    graph_views::EdgeView<DiGraph>
    edge_view();

    graph_views::EdgeView<const DiGraph>
    edge_view() const;

    graph_views::AdjacencyView<DiGraph>
    adjacency_view();

    graph_views::AdjacencyView<const DiGraph>
    adjacency_view() const;

    bool has_edge(
        Vertex u,
        Vertex v) const;

    DiEdge edge(
        Vertex u,
        Vertex v) const;

    bool remove_edge(
        Vertex u,
        Vertex v);

    Vertex source(
        DiEdge e) const;

    Vertex target(
        DiEdge e) const;

    const DiRawNeighborList&
    neighbors_fast(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_out_edges;
    }

    const DiRawNeighborList&
    successors_fast(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_out_edges;
    }

    const DiRawNeighborList&
    out_edges_fast(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_out_edges;
    }

    const DiRawInNeighborList&
    predecessors_fast(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_in_edges;
    }

    const DiRawInNeighborList&
    in_edges_fast(
        Vertex v) const noexcept
    {
        return graph_.m_vertices[v]
            .m_in_edges;
    }

    uint32_t edge_id(
        DiEdge e) const noexcept
    {
        return static_cast<const EdgeProperty*>(
            e.get_property())->edge_id;
    }

    DiEdge edge_by_id(
        uint32_t id) const;

    std::pair<Vertex, Vertex>
    edge_endpoints(
        uint32_t id) const;

    size_t edge_id_capacity() const noexcept;

    AttrId attr_id(
        std::string_view name) const;

    std::string_view attr_name(
        AttrId id) const;

    const AttributeRegistry&
    attribute_registry() const;

    BGLDiGraph& raw();

    const BGLDiGraph& raw() const;

private:
    void swap_state(DiGraph& other) noexcept;

    mutable BGLDiGraph graph_;

    std::shared_ptr<AttributeRegistry>
        attr_registry_ =
            std::make_shared<AttributeRegistry>();

    AttrMap graph_attrs_{attr_registry_};

    uint32_t next_edge_id_ = 0;

    std::vector<
        std::optional<
            std::pair<Vertex, Vertex>>>
        edge_endpoints_;

    mutable bool edge_order_dirty_ = false;
};

// Complete the lightweight public facade return types after Graph/DiGraph are
// defined. Each view header includes graph.h safely through this file's pragma
// once guard, so consumers only need to include graph.h.
#include "views/node_view.h"
#include "views/edge_view.h"
#include "views/adjacency_view.h"
