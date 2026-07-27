#include "graph.h"
#include "views/adjacency_view.h"
#include "views/edge_view.h"
#include "views/node_view.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

void require_vertex(
    Vertex vertex,
    size_t node_count)
{
    if (vertex >= node_count)
    {
        throw std::out_of_range(
            "Graph vertex is out of range");
    }
}

template <typename EdgeType>
EdgeProperty& raw_edge_property(
    EdgeType edge) noexcept
{
    // Accepted Boost 1.85 descriptor-layout hack. graph_types.h pins and
    // statically verifies the void* property plug used by edge_desc_impl.
    return *static_cast<EdgeProperty*>(
        edge.get_property());
}

template <typename EdgeType>
const EdgeProperty& raw_edge_property_const(
    EdgeType edge) noexcept
{
    return *static_cast<const EdgeProperty*>(
        edge.get_property());
}

void copy_attributes(
    const AttrMap& source,
    AttrMap& target)
{
    for (const AttrId id :
         source.attribute_ids())
    {
        target.set(
            id,
            clone_attr_value(source.at(id)));
    }
}

void apply_attributes(
    const AttrObject& source,
    AttrMap& target)
{
    for (const auto& [name, value] :
         source.entries)
    {
        target[name] =
            clone_attr_value(value);
    }
}

template <typename GraphType>
void add_node_count(
    GraphType& graph,
    size_t count)
{
    for (size_t index = 0;
         index < count;
         ++index)
    {
        graph.add_node();
    }
}

template <typename GraphType>
void ensure_contiguous_nodes(
    GraphType& graph,
    const std::vector<Vertex>& referenced)
{
    if (referenced.empty())
    {
        return;
    }

    const Vertex current =
        static_cast<Vertex>(
            graph.num_nodes());
    const Vertex highest =
        *std::max_element(
            referenced.begin(),
            referenced.end());
    if (highest < current)
    {
        return;
    }
    if (highest ==
        std::numeric_limits<Vertex>::max())
    {
        throw std::invalid_argument(
            "Contiguous vertex range is too large");
    }

    const size_t new_count =
        static_cast<size_t>(
            highest - current) + 1;
    if (new_count > referenced.size())
    {
        throw std::invalid_argument(
            "New vertex labels must cover one contiguous range");
    }

    std::vector<uint8_t> present(new_count, uint8_t{0});
    for (const Vertex node : referenced)
    {
        if (node >= current)
        {
            present[static_cast<size_t>(node - current)] = 1;
        }
    }
    if (std::find(
            present.begin(), present.end(), uint8_t{0}) != present.end())
    {
        throw std::invalid_argument(
            "New vertex labels must cover one contiguous range");
    }

    add_node_count(graph, new_count);
}

template <typename EdgeInput>
std::vector<Vertex> edge_vertices(
    const std::vector<EdgeInput>& edges)
{
    std::vector<Vertex> vertices;
    vertices.reserve(edges.size() * 2);
    for (const auto& edge : edges)
    {
        if constexpr (
            std::is_same_v<
                EdgeInput,
                EdgeEndpoints>)
        {
            vertices.push_back(edge.first);
            vertices.push_back(edge.second);
        }
        else
        {
            vertices.push_back(edge.u);
            vertices.push_back(edge.v);
        }
    }
    return vertices;
}

template <typename EdgeInput>
void validate_edge_bounds(
    size_t node_count,
    const std::vector<EdgeInput>& edges)
{
    for (const auto& edge : edges)
    {
        Vertex u;
        Vertex v;
        if constexpr (
            std::is_same_v<
                EdgeInput,
                EdgeEndpoints>)
        {
            u = edge.first;
            v = edge.second;
        }
        else
        {
            u = edge.u;
            v = edge.v;
        }
        if (u >= node_count ||
            v >= node_count)
        {
            throw std::invalid_argument(
                "Edge endpoint violates contiguous vertex range");
        }
    }
}

template <typename GraphType>
void rebind_attributes(
    GraphType& graph,
    const std::shared_ptr<AttributeRegistry>& registry)
{
    auto [node, node_end] = boost::vertices(graph);
    for (; node != node_end; ++node)
    {
        graph.m_vertices[*node]
            .m_property
            .attrs.bind(registry);
    }

    auto [edge, edge_end] = boost::edges(graph);
    for (; edge != edge_end; ++edge)
    {
        raw_edge_property(*edge)
            .attrs.bind(registry);
    }
}

template <typename IncidenceList>
void restore_incidence_order(
    const IncidenceList& source,
    IncidenceList& target,
    size_t edge_id_capacity)
{
    if (source.size() < 2)
    {
        return;
    }

    const size_t missing =
        std::numeric_limits<size_t>::max();
    std::vector<size_t> rank(
        edge_id_capacity,
        missing);

    size_t index = 0;
    for (const auto& edge : source)
    {
        const uint32_t id =
            edge.get_property().edge_id;
        if (id >= rank.size())
        {
            throw std::logic_error(
                "Edge ID is outside copy bookkeeping");
        }
        rank[id] = index++;
    }

    std::stable_sort(
        target.begin(),
        target.end(),
        [&](const auto& lhs, const auto& rhs)
        {
            return rank[lhs.get_property().edge_id] <
                   rank[rhs.get_property().edge_id];
        });
}

void restore_incidence_order(
    const BGLGraph& source,
    BGLGraph& target,
    size_t edge_id_capacity)
{
    for (Vertex v = 0;
         v < boost::num_vertices(source);
         ++v)
    {
        restore_incidence_order(
            source.m_vertices[v].m_out_edges,
            target.m_vertices[v].m_out_edges,
            edge_id_capacity);
    }
}

void restore_incidence_order(
    const BGLDiGraph& source,
    BGLDiGraph& target,
    size_t edge_id_capacity)
{
    for (Vertex v = 0;
         v < boost::num_vertices(source);
         ++v)
    {
        restore_incidence_order(
            source.m_vertices[v].m_out_edges,
            target.m_vertices[v].m_out_edges,
            edge_id_capacity);
        restore_incidence_order(
            source.m_vertices[v].m_in_edges,
            target.m_vertices[v].m_in_edges,
            edge_id_capacity);
    }
}

void collapse_copied_self_loops(
    BGLGraph& graph)
{
    for (Vertex vertex = 0;
         vertex < boost::num_vertices(graph);
         ++vertex)
    {
        auto& stored_vertex =
            graph.m_vertices[vertex];
        if (!stored_vertex
                 .m_property
                 .has_self_loop)
        {
            continue;
        }

        auto& incidence =
            stored_vertex.m_out_edges;
        auto first = std::find_if(
            incidence.begin(),
            incidence.end(),
            [vertex](const auto& edge)
            {
                return edge.get_target() ==
                       vertex;
            });
        if (first == incidence.end())
        {
            throw std::logic_error(
                "Self-loop bookkeeping has no incidence record");
        }

        const EdgeProperty* property =
            &first->get_property();
        auto duplicate = std::find_if(
            std::next(first),
            incidence.end(),
            [vertex, property](const auto& edge)
            {
                return edge.get_target() ==
                           vertex &&
                       &edge.get_property() ==
                           property;
            });
        if (duplicate == incidence.end())
        {
            throw std::logic_error(
                "Boost copy did not duplicate an undirected self-loop");
        }
        incidence.erase(duplicate);
    }
}

// BGL's bidirectional/undirected adjacency_list edge iterator follows its
// global listS container, which is raw insertion order.  NetworkX instead
// groups EdgeView output by node (source for DiGraph) while retaining
// adjacency insertion order inside each group.  Stable list sorting by source
// gives exactly that order because every wrapper insertion appends once to the
// corresponding adjacency list.  std::list::sort relinks nodes, preserving all
// iterators stored by BGL's incidence lists and every edge descriptor/property
// address.  graph_types.h pins both Boost 1.85 and listS for this layout hack.
template <typename GraphType>
void normalize_networkx_edge_order(
    GraphType& graph,
    bool& dirty)
{
    if (!dirty)
    {
        return;
    }

    graph.m_edges.sort(
        [](const auto& lhs, const auto& rhs)
        {
            return lhs.m_source < rhs.m_source;
        });
    dirty = false;
}

} // namespace

Graph::Graph(
    const std::vector<EdgeEndpoints>& edge_list)
{
    add_edges_from(edge_list);
}

Graph::Graph(
    const std::vector<EdgeWithAttrs>& edge_list)
{
    add_edges_from(edge_list);
}

Graph::Graph(
    const std::vector<std::vector<double>>& adjacency,
    std::string_view weight_attr)
{
    const size_t n = adjacency.size();
    for (const auto& row : adjacency)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument(
                "Adjacency matrix must be square");
        }
    }
    add_node_count(*this, n);
    const AttrId weight_id = attr_id(weight_attr);
    for (size_t u = 0; u < n; ++u)
    {
        for (size_t v = 0; v < n; ++v)
        {
            const double value = adjacency[u][v];
            if (value == 0.0)
            {
                continue;
            }
            const Edge e = add_edge(
                static_cast<Vertex>(u),
                static_cast<Vertex>(v));
            edge_attrs(e).set(weight_id, value);
        }
    }
}

Graph::Graph(
    size_t node_count,
    const std::vector<EdgeEndpoints>& edge_list)
{
    validate_edge_bounds(
        node_count,
        edge_list);
    add_node_count(*this, node_count);
    add_edges_from(edge_list);
}

Graph::Graph(
    size_t node_count,
    const std::vector<EdgeWithAttrs>& edge_list)
{
    validate_edge_bounds(
        node_count,
        edge_list);
    add_node_count(*this, node_count);
    add_edges_from(edge_list);
}

Graph::Graph(
    const Graph& other)
    :
    graph_(other.graph_),
    attr_registry_(
        std::make_shared<AttributeRegistry>(
            *other.attr_registry_)),
    graph_attrs_(other.graph_attrs_),
    next_edge_id_(other.next_edge_id_),
    edge_endpoints_(other.edge_endpoints_),
    edge_order_dirty_(other.edge_order_dirty_)
{
    graph_attrs_.bind(attr_registry_);
    collapse_copied_self_loops(graph_);
    restore_incidence_order(
        other.graph_,
        graph_,
        next_edge_id_);
    rebind_attributes(graph_, attr_registry_);
}

Graph& Graph::operator=(
    const Graph& other)
{
    if (this == &other)
    {
        return *this;
    }

    graph_ = other.graph_;
    attr_registry_ =
        std::make_shared<AttributeRegistry>(
            *other.attr_registry_);
    graph_attrs_ = other.graph_attrs_;
    graph_attrs_.bind(attr_registry_);
    next_edge_id_ = other.next_edge_id_;
    edge_endpoints_ = other.edge_endpoints_;
    edge_order_dirty_ = other.edge_order_dirty_;
    collapse_copied_self_loops(graph_);
    restore_incidence_order(
        other.graph_,
        graph_,
        next_edge_id_);
    rebind_attributes(graph_, attr_registry_);
    return *this;
}

Graph::Graph(Graph&& other)
    : Graph()
{
    swap_state(other);
}

Graph& Graph::operator=(
    Graph&& other) noexcept
{
    if (this != &other)
    {
        swap_state(other);
    }
    return *this;
}

void Graph::swap_state(
    Graph& other) noexcept
{
    // adjacency_list 1.85 has no real move constructor: a defaulted move
    // silently invokes its copy constructor and rebuilds incidence lists in
    // global-edge order.  Swap the pinned vecS/listS storage directly so list
    // iterators, edge-property addresses, and original neighbor order survive.
    graph_.m_edges.swap(other.graph_.m_edges);
    graph_.m_vertices.swap(other.graph_.m_vertices);
    attr_registry_.swap(other.attr_registry_);
    std::swap(graph_attrs_, other.graph_attrs_);
    std::swap(next_edge_id_, other.next_edge_id_);
    edge_endpoints_.swap(other.edge_endpoints_);
    std::swap(edge_order_dirty_, other.edge_order_dirty_);
}

Vertex Graph::add_node()
{
    Vertex v =
        boost::add_vertex(
            graph_);

    graph_.m_vertices[v]
        .m_property
        .attrs.bind(
        attr_registry_);

    return v;
}

void Graph::add_nodes_from(
    const std::vector<Vertex>& nodes)
{
    ensure_contiguous_nodes(
        *this,
        nodes);
}

void Graph::add_nodes_from(
    const std::vector<NodeWithAttrs>& nodes)
{
    std::vector<Vertex> labels;
    labels.reserve(nodes.size());
    for (const NodeWithAttrs& node : nodes)
    {
        labels.push_back(node.node);
    }
    ensure_contiguous_nodes(
        *this,
        labels);

    for (const NodeWithAttrs& node : nodes)
    {
        apply_attributes(
            node.attrs,
            node_attrs(node.node));
    }
}

Edge Graph::add_edge(
    Vertex u,
    Vertex v)
{
    const Vertex highest = std::max(u, v);
    if (highest >= num_nodes())
    {
        if (highest == std::numeric_limits<Vertex>::max())
        {
            throw std::invalid_argument(
                "Vertex label is too large");
        }
        add_node_count(
            *this,
            static_cast<size_t>(highest - num_nodes()) + 1);
    }

    // Nodes are inserted as contiguous indices, so NetworkX EdgeView always
    // presents an undirected edge from its earlier (smaller) endpoint.
    if (v < u)
    {
        std::swap(u, v);
    }

    auto [existing, exists] =
        boost::edge(
            u,
            v,
            graph_);

    if (exists)
    {
        return existing;
    }

    // Keep uint32_t edge IDs as the compact hot-path ABI, but reject ID
    // exhaustion before mutating Boost storage.  UINT32_MAX remains the
    // sentinel so neither the counter nor id + 1 can wrap.
    if (next_edge_id_ ==
        std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error(
            "Graph edge ID space is exhausted");
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

    if (u == v)
    {
        // adjacency_list stores an undirected loop twice in the same vecS
        // incidence list. NetworkX adjacency exposes one neighbor entry, so
        // collapse the just-appended duplicate once at mutation time instead
        // of branching on every hot-loop increment.
        auto& incidence =
            graph_.m_vertices[u]
                .m_out_edges;
        if (incidence.size() < 2 ||
            &incidence[incidence.size() - 1]
                 .get_property() !=
                &incidence[incidence.size() - 2]
                     .get_property())
        {
            throw std::logic_error(
                "Boost self-loop incidence layout changed");
        }
        incidence.pop_back();
        graph_.m_vertices[u]
            .m_property
            .has_self_loop = true;
    }

    // Monotone generators already append in node-major order and pay no sort.
    // Arbitrary insertion is normalized only when EdgeView is requested.
    if (!edge_order_dirty_ &&
        graph_.m_edges.size() > 1)
    {
        const auto last = std::prev(
            graph_.m_edges.end());
        const auto previous = std::prev(last);
        edge_order_dirty_ =
            last->m_source < previous->m_source;
    }

    raw_edge_property(e).attrs.bind(
        attr_registry_);

    const uint32_t id =
        next_edge_id_++;

    raw_edge_property(e).edge_id =
        id;

    if (id >= edge_endpoints_.size())
    {
        edge_endpoints_.resize(
            static_cast<size_t>(id) + size_t{1});
    }

    edge_endpoints_[id] =
        std::pair<Vertex, Vertex>{
            u,
            v};

    return e;
}

Edge Graph::add_edge(
    Vertex u,
    Vertex v,
    const AttrObject& attrs)
{
    const Edge e = add_edge(u, v);
    apply_attributes(attrs, edge_attrs(e));
    return e;
}

void Graph::add_edges_from(
    const std::vector<EdgeEndpoints>& edges)
{
    for (const auto& [u, v] : edges)
    {
        add_edge(u, v);
    }
}

void Graph::add_edges_from(
    const std::vector<EdgeWithAttrs>& edges)
{
    for (const EdgeWithAttrs& input : edges)
    {
        const Edge edge =
            add_edge(input.u, input.v);
        apply_attributes(
            input.attrs,
            edge_attrs(edge));
    }
}

AttrMap& Graph::graph_attrs()
{
    return graph_attrs_;
}

const AttrMap& Graph::graph_attrs() const
{
    return graph_attrs_;
}

size_t Graph::num_edges() const
{
    return boost::num_edges(
        graph_);
}

size_t Graph::degree(
    Vertex v) const
{
    require_vertex(v, num_nodes());
    const size_t incidence =
        boost::degree(
            v,
            graph_);
    return incidence +
           static_cast<size_t>(
               graph_.m_vertices[v]
                   .m_property
                   .has_self_loop);
}

DegreeItems Graph::degree() const
{
    DegreeItems result;
    result.reserve(num_nodes());
    for (Vertex v = 0;
         v < num_nodes();
         ++v)
    {
        result.emplace_back(
            v,
            degree(v));
    }
    return result;
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
    require_vertex(v, num_nodes());
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
    normalize_networkx_edge_order(
        graph_,
        edge_order_dirty_);
    return boost::edges(
        graph_);
}

graph_views::IncidentEdgeRange<Graph>
Graph::edges(Vertex v)
{
    return {*this, v};
}

graph_views::IncidentEdgeRange<const Graph>
Graph::edges(Vertex v) const
{
    return {*this, v};
}

graph_views::NodeView<Graph>
Graph::node_view()
{
    return graph_views::NodeView<Graph>(
        *this);
}

graph_views::NodeView<const Graph>
Graph::node_view() const
{
    return graph_views::NodeView<const Graph>(
        *this);
}

graph_views::EdgeView<Graph>
Graph::edge_view()
{
    return graph_views::EdgeView<Graph>(
        *this);
}

graph_views::EdgeView<const Graph>
Graph::edge_view() const
{
    return graph_views::EdgeView<const Graph>(
        *this);
}

graph_views::AdjacencyView<Graph>
Graph::adjacency_view()
{
    return graph_views::AdjacencyView<Graph>(
        *this);
}

graph_views::AdjacencyView<const Graph>
Graph::adjacency_view() const
{
    return graph_views::AdjacencyView<const Graph>(
        *this);
}

bool Graph::has_edge(
    Vertex u,
    Vertex v) const
{
    if (u >= num_nodes() || v >= num_nodes())
    {
        return false;
    }
    return boost::edge(
        u,
        v,
        graph_).second;
}

Edge Graph::edge(
    Vertex u,
    Vertex v) const
{
    if (u >= num_nodes() || v >= num_nodes())
    {
        throw std::out_of_range(
            "Edge endpoint is out of range");
    }
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
    if (u >= num_nodes() || v >= num_nodes())
    {
        return false;
    }
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
        raw_edge_property_const(e).edge_id;

    if (id < edge_endpoints_.size())
    {
        edge_endpoints_[id].reset();
    }

    if (u == v)
    {
        graph_.m_vertices[u]
            .m_property
            .has_self_loop = false;
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

size_t Graph::edge_id_capacity() const noexcept
{
    return edge_endpoints_.size();
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

DiGraph::DiGraph(
    const std::vector<EdgeEndpoints>& edge_list)
{
    add_edges_from(edge_list);
}

DiGraph::DiGraph(
    const std::vector<EdgeWithAttrs>& edge_list)
{
    add_edges_from(edge_list);
}

DiGraph::DiGraph(
    const std::vector<std::vector<double>>& adjacency,
    std::string_view weight_attr)
{
    const size_t n = adjacency.size();
    for (const auto& row : adjacency)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument(
                "Adjacency matrix must be square");
        }
    }
    add_node_count(*this, n);
    const AttrId weight_id = attr_id(weight_attr);
    for (size_t u = 0; u < n; ++u)
    {
        for (size_t v = 0; v < n; ++v)
        {
            const double value = adjacency[u][v];
            if (value == 0.0)
            {
                continue;
            }
            const DiEdge e = add_edge(
                static_cast<Vertex>(u),
                static_cast<Vertex>(v));
            edge_attrs(e).set(weight_id, value);
        }
    }
}

DiGraph::DiGraph(
    size_t node_count,
    const std::vector<EdgeEndpoints>& edge_list)
{
    validate_edge_bounds(
        node_count,
        edge_list);
    add_node_count(*this, node_count);
    add_edges_from(edge_list);
}

DiGraph::DiGraph(
    size_t node_count,
    const std::vector<EdgeWithAttrs>& edge_list)
{
    validate_edge_bounds(
        node_count,
        edge_list);
    add_node_count(*this, node_count);
    add_edges_from(edge_list);
}

DiGraph::DiGraph(
    const DiGraph& other)
    :
    graph_(other.graph_),
    attr_registry_(
        std::make_shared<AttributeRegistry>(
            *other.attr_registry_)),
    graph_attrs_(other.graph_attrs_),
    next_edge_id_(other.next_edge_id_),
    edge_endpoints_(other.edge_endpoints_),
    edge_order_dirty_(other.edge_order_dirty_)
{
    graph_attrs_.bind(attr_registry_);
    restore_incidence_order(
        other.graph_,
        graph_,
        next_edge_id_);
    rebind_attributes(graph_, attr_registry_);
}

DiGraph& DiGraph::operator=(
    const DiGraph& other)
{
    if (this == &other)
    {
        return *this;
    }

    graph_ = other.graph_;
    attr_registry_ =
        std::make_shared<AttributeRegistry>(
            *other.attr_registry_);
    graph_attrs_ = other.graph_attrs_;
    graph_attrs_.bind(attr_registry_);
    next_edge_id_ = other.next_edge_id_;
    edge_endpoints_ = other.edge_endpoints_;
    edge_order_dirty_ = other.edge_order_dirty_;
    restore_incidence_order(
        other.graph_,
        graph_,
        next_edge_id_);
    rebind_attributes(graph_, attr_registry_);
    return *this;
}

DiGraph::DiGraph(DiGraph&& other)
    : DiGraph()
{
    swap_state(other);
}

DiGraph& DiGraph::operator=(
    DiGraph&& other) noexcept
{
    if (this != &other)
    {
        swap_state(other);
    }
    return *this;
}

void DiGraph::swap_state(
    DiGraph& other) noexcept
{
    graph_.m_edges.swap(other.graph_.m_edges);
    graph_.m_vertices.swap(other.graph_.m_vertices);
    attr_registry_.swap(other.attr_registry_);
    std::swap(graph_attrs_, other.graph_attrs_);
    std::swap(next_edge_id_, other.next_edge_id_);
    edge_endpoints_.swap(other.edge_endpoints_);
    std::swap(edge_order_dirty_, other.edge_order_dirty_);
}

Vertex DiGraph::add_node()
{
    Vertex v =
        boost::add_vertex(
            graph_);

    graph_.m_vertices[v]
        .m_property
        .attrs.bind(
        attr_registry_);

    return v;
}

void DiGraph::add_nodes_from(
    const std::vector<Vertex>& nodes)
{
    ensure_contiguous_nodes(
        *this,
        nodes);
}

void DiGraph::add_nodes_from(
    const std::vector<NodeWithAttrs>& nodes)
{
    std::vector<Vertex> labels;
    labels.reserve(nodes.size());
    for (const NodeWithAttrs& node : nodes)
    {
        labels.push_back(node.node);
    }
    ensure_contiguous_nodes(
        *this,
        labels);

    for (const NodeWithAttrs& node : nodes)
    {
        apply_attributes(
            node.attrs,
            node_attrs(node.node));
    }
}

DiEdge DiGraph::add_edge(
    Vertex u,
    Vertex v)
{
    const Vertex highest = std::max(u, v);
    if (highest >= num_nodes())
    {
        if (highest == std::numeric_limits<Vertex>::max())
        {
            throw std::invalid_argument(
                "Vertex label is too large");
        }
        add_node_count(
            *this,
            static_cast<size_t>(highest - num_nodes()) + 1);
    }

    auto [existing, exists] =
        boost::edge(
            u,
            v,
            graph_);

    if (exists)
    {
        return existing;
    }

    if (next_edge_id_ ==
        std::numeric_limits<uint32_t>::max())
    {
        throw std::overflow_error(
            "DiGraph edge ID space is exhausted");
    }

    auto [e, ok] =
        boost::add_edge(
            u,
            v,
            graph_);

    if (!ok)
    {
        throw std::runtime_error(
            "Failed to add directed edge");
    }

    if (!edge_order_dirty_ &&
        graph_.m_edges.size() > 1)
    {
        const auto last = std::prev(
            graph_.m_edges.end());
        const auto previous = std::prev(last);
        edge_order_dirty_ =
            last->m_source < previous->m_source;
    }

    raw_edge_property(e).attrs.bind(
        attr_registry_);

    const uint32_t id =
        next_edge_id_++;

    raw_edge_property(e).edge_id = id;

    if (id >= edge_endpoints_.size())
    {
        edge_endpoints_.resize(
            static_cast<size_t>(id) + size_t{1});
    }

    edge_endpoints_[id] =
        std::pair<Vertex, Vertex>{u, v};

    return e;
}

DiEdge DiGraph::add_edge(
    Vertex u,
    Vertex v,
    const AttrObject& attrs)
{
    const DiEdge e = add_edge(u, v);
    apply_attributes(attrs, edge_attrs(e));
    return e;
}

void DiGraph::add_edges_from(
    const std::vector<EdgeEndpoints>& edges)
{
    for (const auto& [u, v] : edges)
    {
        add_edge(u, v);
    }
}

void DiGraph::add_edges_from(
    const std::vector<EdgeWithAttrs>& edges)
{
    for (const EdgeWithAttrs& input : edges)
    {
        const DiEdge edge =
            add_edge(input.u, input.v);
        apply_attributes(
            input.attrs,
            edge_attrs(edge));
    }
}

AttrMap& DiGraph::graph_attrs()
{
    return graph_attrs_;
}

const AttrMap& DiGraph::graph_attrs() const
{
    return graph_attrs_;
}

size_t DiGraph::num_edges() const
{
    return boost::num_edges(
        graph_);
}

size_t DiGraph::degree(
    Vertex v) const
{
    require_vertex(v, num_nodes());
    return boost::degree(
        v,
        graph_);
}

DegreeItems DiGraph::degree() const
{
    DegreeItems result;
    result.reserve(num_nodes());
    for (Vertex v = 0;
         v < num_nodes();
         ++v)
    {
        result.emplace_back(
            v,
            degree(v));
    }
    return result;
}

size_t DiGraph::in_degree(
    Vertex v) const
{
    require_vertex(v, num_nodes());
    return boost::in_degree(
        v,
        graph_);
}

DegreeItems DiGraph::in_degree() const
{
    DegreeItems result;
    result.reserve(num_nodes());
    for (Vertex v = 0;
         v < num_nodes();
         ++v)
    {
        result.emplace_back(
            v,
            in_degree(v));
    }
    return result;
}

size_t DiGraph::out_degree(
    Vertex v) const
{
    require_vertex(v, num_nodes());
    return boost::out_degree(
        v,
        graph_);
}

DegreeItems DiGraph::out_degree() const
{
    DegreeItems result;
    result.reserve(num_nodes());
    for (Vertex v = 0;
         v < num_nodes();
         ++v)
    {
        result.emplace_back(
            v,
            out_degree(v));
    }
    return result;
}

auto DiGraph::neighbors(
    Vertex v) const
    ->
    std::pair<
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator,
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator>
{
    require_vertex(v, num_nodes());
    return boost::adjacent_vertices(
        v,
        graph_);
}

auto DiGraph::successors(
    Vertex v) const
    ->
    std::pair<
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator,
        boost::graph_traits<
            BGLDiGraph>::adjacency_iterator>
{
    require_vertex(v, num_nodes());
    return boost::adjacent_vertices(
        v,
        graph_);
}

auto DiGraph::predecessors(
    Vertex v) const
    ->
    std::pair<
        BGLDiGraph::inv_adjacency_iterator,
        BGLDiGraph::inv_adjacency_iterator>
{
    require_vertex(v, num_nodes());
    return boost::inv_adjacent_vertices(
        v,
        graph_);
}

auto DiGraph::out_edges(
    Vertex v) const
    ->
    std::pair<
        boost::graph_traits<
            BGLDiGraph>::out_edge_iterator,
        boost::graph_traits<
            BGLDiGraph>::out_edge_iterator>
{
    require_vertex(v, num_nodes());
    return boost::out_edges(
        v,
        graph_);
}

auto DiGraph::in_edges(
    Vertex v) const
    ->
    std::pair<
        boost::graph_traits<
            BGLDiGraph>::in_edge_iterator,
        boost::graph_traits<
            BGLDiGraph>::in_edge_iterator>
{
    require_vertex(v, num_nodes());
    return boost::in_edges(
        v,
        graph_);
}

auto DiGraph::nodes() const
    ->
    std::pair<
        boost::graph_traits<
            BGLDiGraph>::vertex_iterator,
        boost::graph_traits<
            BGLDiGraph>::vertex_iterator>
{
    return boost::vertices(
        graph_);
}

auto DiGraph::edges() const
    ->
    std::pair<
        boost::graph_traits<
            BGLDiGraph>::edge_iterator,
        boost::graph_traits<
            BGLDiGraph>::edge_iterator>
{
    normalize_networkx_edge_order(
        graph_,
        edge_order_dirty_);
    return boost::edges(
        graph_);
}

graph_views::IncidentEdgeRange<DiGraph>
DiGraph::edges(Vertex v)
{
    return {*this, v};
}

graph_views::IncidentEdgeRange<const DiGraph>
DiGraph::edges(Vertex v) const
{
    return {*this, v};
}

graph_views::NodeView<DiGraph>
DiGraph::node_view()
{
    return graph_views::NodeView<DiGraph>(
        *this);
}

graph_views::NodeView<const DiGraph>
DiGraph::node_view() const
{
    return graph_views::NodeView<const DiGraph>(
        *this);
}

graph_views::EdgeView<DiGraph>
DiGraph::edge_view()
{
    return graph_views::EdgeView<DiGraph>(
        *this);
}

graph_views::EdgeView<const DiGraph>
DiGraph::edge_view() const
{
    return graph_views::EdgeView<const DiGraph>(
        *this);
}

graph_views::AdjacencyView<DiGraph>
DiGraph::adjacency_view()
{
    return graph_views::AdjacencyView<DiGraph>(
        *this);
}

graph_views::AdjacencyView<const DiGraph>
DiGraph::adjacency_view() const
{
    return graph_views::AdjacencyView<const DiGraph>(
        *this);
}

bool DiGraph::has_edge(
    Vertex u,
    Vertex v) const
{
    if (u >= num_nodes() || v >= num_nodes())
    {
        return false;
    }
    return boost::edge(
        u,
        v,
        graph_).second;
}

DiEdge DiGraph::edge(
    Vertex u,
    Vertex v) const
{
    if (u >= num_nodes() || v >= num_nodes())
    {
        throw std::out_of_range(
            "Directed edge endpoint is out of range");
    }
    auto [e, ok] =
        boost::edge(
            u,
            v,
            graph_);

    if (!ok)
    {
        throw std::runtime_error(
            "Directed edge not found");
    }

    return e;
}

bool DiGraph::remove_edge(
    Vertex u,
    Vertex v)
{
    if (u >= num_nodes() || v >= num_nodes())
    {
        return false;
    }
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
        raw_edge_property_const(e).edge_id;

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

Vertex DiGraph::source(
    DiEdge e) const
{
    return boost::source(
        e,
        graph_);
}

Vertex DiGraph::target(
    DiEdge e) const
{
    return boost::target(
        e,
        graph_);
}

DiEdge DiGraph::edge_by_id(
    uint32_t id) const
{
    if (id >= edge_endpoints_.size() ||
        !edge_endpoints_[id].has_value())
    {
        throw std::runtime_error(
            "Directed edge id not found");
    }

    const auto [u, v] =
        *edge_endpoints_[id];

    return edge(u, v);
}

std::pair<Vertex, Vertex>
DiGraph::edge_endpoints(
    uint32_t id) const
{
    if (id >= edge_endpoints_.size() ||
        !edge_endpoints_[id].has_value())
    {
        throw std::runtime_error(
            "Directed edge id not found");
    }

    return *edge_endpoints_[id];
}

size_t DiGraph::edge_id_capacity() const noexcept
{
    return edge_endpoints_.size();
}

AttrId DiGraph::attr_id(
    std::string_view name) const
{
    return attr_registry_->intern(name);
}

std::string_view DiGraph::attr_name(
    AttrId id) const
{
    return attr_registry_->name(id);
}

const AttributeRegistry&
DiGraph::attribute_registry() const
{
    return *attr_registry_;
}

BGLDiGraph& DiGraph::raw()
{
    return graph_;
}

const BGLDiGraph& DiGraph::raw() const
{
    return graph_;
}

DiGraph Graph::to_directed() const
{
    DiGraph directed;

    // Keep AttrId values stable across the conversion.  This lets callers
    // resolve a string once before a hot loop and use the same integer ID on
    // either representation.
    for (AttrId id = 0;
         id < attribute_registry().size();
         ++id)
    {
        static_cast<void>(
            directed.attr_id(
                attr_name(id)));
    }

    copy_attributes(
        graph_attrs(),
        directed.graph_attrs());

    for (Vertex v = 0;
         v < num_nodes();
         ++v)
    {
        const Vertex copy =
            directed.add_node();

        copy_attributes(
            node_attrs(v),
            directed.node_attrs(copy));
    }

    // NetworkX Graph.to_directed() walks each node's adjacency mapping.  This
    // matters when a node's earlier-neighbor and later-neighbor edges were
    // interleaved: forward/reverse insertion from Graph.edges() would produce
    // a different successor order even though the edge set is identical.
    for (Vertex u = 0;
         u < num_nodes();
         ++u)
    {
        auto [neighbor, neighbor_end] =
            neighbors(u);
        for (; neighbor != neighbor_end;
             ++neighbor)
        {
            const Vertex v = *neighbor;
            const Edge source_edge =
                edge(u, v);
            const DiEdge copy =
                directed.add_edge(u, v);

            copy_attributes(
                edge_attrs(source_edge),
                directed.edge_attrs(copy));
        }
    }

    return directed;
}
