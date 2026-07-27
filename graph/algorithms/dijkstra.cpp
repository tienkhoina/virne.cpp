#include "dijkstra.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace
{

constexpr double INF =
    std::numeric_limits<double>::max();

struct HeapNode
{
    double dist;
    uint64_t order;
    Vertex vertex;

    bool operator>(
        const HeapNode& other) const noexcept
    {
        if (dist != other.dist)
        {
            return dist > other.dist;
        }
        return order > other.order;
    }
};

template <typename GraphType>
static DijkstraResult make_empty_result(
    const GraphType& g)
{
    DijkstraResult result;

    const std::size_t n =
        g.num_nodes();

    result.distance.assign(
        n,
        INF);

    result.predecessor.resize(
        n);

    result.settled_order.reserve(n);

    for (Vertex v = 0;
         v < n;
         ++v)
    {
        result.predecessor[v] = v;
    }

    return result;
}

template <typename NeighborType>
inline double fast_edge_weight(
    const NeighborType& edge,
    AttrId weight_attr_id)
{
    const auto& attrs =
        edge.get_property().attrs;

    const AttrValue* value =
        attrs.find(weight_attr_id);

    if (value == nullptr)
    {
        return 1.0;
    }

    return attr_to_double(*value);
}

template <typename GraphType>
static std::unordered_set<uint32_t>
build_banned_edge_id_set(
    const GraphType& g,
    const EdgeSet& banned_edges)
{
    std::unordered_set<uint32_t> banned_ids;

    banned_ids.reserve(
        banned_edges.size());

    for (const auto& key : banned_edges)
    {
        const Vertex u =
            key.first;
        const Vertex v =
            key.second;

        if (!g.has_edge(u, v))
        {
            continue;
        }

        const auto e =
            g.edge(u, v);

        banned_ids.insert(
            g.edge_id(e));
    }

    return banned_ids;
}

template <typename GraphType>
static DijkstraResult run_dijkstra(
    const GraphType& g,
    Vertex source,
    const SearchMask* mask,
    const VertexSet* banned_vertices,
    const EdgeSet* banned_edges,
    AttrId weight_attr_id,
    std::optional<double> cutoff)
{
    DijkstraResult result =
        make_empty_result(g);

    auto& dist =
        result.distance;

    auto& pred =
        result.predecessor;

    std::priority_queue<
        HeapNode,
        std::vector<HeapNode>,
        std::greater<HeapNode>>
        pq;

    std::unordered_set<uint32_t> banned_edge_ids;
    if (banned_edges != nullptr &&
        !banned_edges->empty())
    {
        banned_edge_ids =
            build_banned_edge_id_set(
                g,
                *banned_edges);
    }

    dist[source] = 0.0;
    pred[source] = source;
    uint64_t push_order = 0;

    pq.push({
        0.0,
        push_order++,
        source});

    while (!pq.empty())
    {
        const HeapNode current =
            pq.top();

        pq.pop();

        const double du =
            current.dist;

        const Vertex u =
            current.vertex;

        if (du != dist[u])
        {
            continue;
        }

        result.settled_order.push_back(u);

        const auto& out =
            g.neighbors_fast(
                u);

        for (const auto& edge : out)
        {
            const Vertex v =
                edge.get_target();

            if (mask != nullptr &&
                !mask->allows(
                    u,
                    v,
                    edge.get_property().edge_id))
            {
                continue;
            }

            if (banned_vertices != nullptr)
            {
                if (banned_vertices->find(v) !=
                    banned_vertices->end())
                {
                    continue;
                }
            }

            if (!banned_edge_ids.empty())
            {
                const uint32_t eid =
                    edge.get_property().edge_id;

                if (banned_edge_ids.find(eid) !=
                    banned_edge_ids.end())
                {
                    continue;
                }
            }

            const double w =
                fast_edge_weight(
                    edge,
                    weight_attr_id);

            if (w < 0.0)
            {
                throw std::runtime_error(
                    "Negative edge weights are not supported");
            }

            const double nd =
                du + w;

            if (cutoff && nd > *cutoff)
            {
                continue;
            }

            if (nd < dist[v])
            {
                dist[v] = nd;
                pred[v] = u;

                pq.push({
                    nd,
                    push_order++,
                    v});
            }
        }
    }

    return result;
}

template <typename GraphType, typename EdgeType>
static double read_edge_weight(
    const GraphType& g,
    EdgeType e,
    AttrId weight_attr_id)
{
    const auto& attrs =
        g.edge_attrs(e);

    const AttrValue* value =
        attrs.find(weight_attr_id);

    if (value == nullptr)
    {
        return 1.0;
    }

    return attr_to_double(*value);
}

template <typename GraphType>
DijkstraResult dijkstra_impl(
    const GraphType& g,
    Vertex source,
    const SearchMask* mask,
    const std::string& weight_attr,
    std::optional<double> cutoff = std::nullopt)
{
    if (source >= g.num_nodes())
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }

    if (mask != nullptr &&
        !mask->allows_node(source))
    {
        throw std::runtime_error(
            "source vertex is filtered out");
    }

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    return run_dijkstra(
        g,
        source,
        mask,
        nullptr,
        nullptr,
        weight_attr_id,
        cutoff);
}

template <typename GraphType>
DijkstraResult dijkstra_masked_impl(
    const GraphType& g,
    Vertex source,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    if (source >= g.num_nodes())
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }

    if (mask != nullptr &&
        !mask->allows_node(source))
    {
        throw std::runtime_error(
            "source vertex is filtered out");
    }

    if (banned_vertices.find(source) !=
        banned_vertices.end())
    {
        throw std::runtime_error(
            "source vertex cannot be banned");
    }

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    return run_dijkstra(
        g,
        source,
        mask,
        &banned_vertices,
        &banned_edges,
        weight_attr_id,
        std::nullopt);
}

template <typename GraphType>
double edge_cost_impl(
    const GraphType& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr)
{
    const auto e =
        g.edge(u, v);

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    return read_edge_weight(
        g,
        e,
        weight_attr_id);
}

template <typename GraphType>
double path_cost_impl(
    const GraphType& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    if (path.size() < 2)
    {
        return 0.0;
    }

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    double cost = 0.0;

    for (std::size_t i = 0;
         i + 1 < path.size();
         ++i)
    {
        const auto e =
            g.edge(
                path[i],
                path[i + 1]);

        cost += read_edge_weight(
            g,
            e,
            weight_attr_id);
    }

    return cost;
}

template <typename GraphType>
std::vector<double> path_prefix_costs_impl(
    const GraphType& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    std::vector<double> prefix;

    prefix.reserve(path.size());

    if (path.empty())
    {
        return prefix;
    }

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    prefix.push_back(0.0);

    double running = 0.0;

    for (std::size_t i = 0;
         i + 1 < path.size();
         ++i)
    {
        const auto e =
            g.edge(
                path[i],
                path[i + 1]);

        running += read_edge_weight(
            g,
            e,
            weight_attr_id);

        prefix.push_back(running);
    }

    return prefix;
}

} // namespace

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g,
        source,
        nullptr,
        weight_attr);
}

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g,
        source,
        nullptr,
        weight_attr);
}

DijkstraResult dijkstra_with_cutoff(
    const Graph& g,
    Vertex source,
    std::optional<double> cutoff,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g, source, nullptr, weight_attr, cutoff);
}

DijkstraResult dijkstra_with_cutoff(
    const DiGraph& g,
    Vertex source,
    std::optional<double> cutoff,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g, source, nullptr, weight_attr, cutoff);
}

DijkstraResult dijkstra_with_cutoff(
    const Graph& g,
    Vertex source,
    const SearchMask& mask,
    std::optional<double> cutoff,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g, source, &mask, weight_attr, cutoff);
}

DijkstraResult dijkstra_with_cutoff(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask,
    std::optional<double> cutoff,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g, source, &mask, weight_attr, cutoff);
}

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const SearchMask& mask,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g,
        source,
        &mask,
        weight_attr);
}

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask,
    const std::string& weight_attr)
{
    return dijkstra_impl(
        g,
        source,
        &mask,
        weight_attr);
}

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return dijkstra_masked_impl(
        g,
        source,
        nullptr,
        banned_vertices,
        banned_edges,
        weight_attr);
}

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return dijkstra_masked_impl(
        g,
        source,
        nullptr,
        banned_vertices,
        banned_edges,
        weight_attr);
}


DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const SearchMask& mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return dijkstra_masked_impl(
        g,
        source,
        &mask,
        banned_vertices,
        banned_edges,
        weight_attr);
}

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return dijkstra_masked_impl(
        g,
        source,
        &mask,
        banned_vertices,
        banned_edges,
        weight_attr);
}

double edge_cost(
    const Graph& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr)
{
    return edge_cost_impl(
        g,
        u,
        v,
        weight_attr);
}

double edge_cost(
    const DiGraph& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr)
{
    return edge_cost_impl(
        g,
        u,
        v,
        weight_attr);
}

double path_cost(
    const Graph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    return path_cost_impl(
        g,
        path,
        weight_attr);
}

double path_cost(
    const DiGraph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    return path_cost_impl(
        g,
        path,
        weight_attr);
}

std::vector<double> path_prefix_costs(
    const Graph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    return path_prefix_costs_impl(
        g,
        path,
        weight_attr);
}

std::vector<double> path_prefix_costs(
    const DiGraph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    return path_prefix_costs_impl(
        g,
        path,
        weight_attr);
}

std::vector<Vertex> build_path(
    const DijkstraResult& result,
    Vertex source,
    Vertex target)
{
    if (source >= result.predecessor.size() ||
        target >= result.predecessor.size())
    {
        return {};
    }

    std::vector<Vertex> path;

    Vertex current =
        target;

    while (current != source)
    {
        path.push_back(
            current);

        Vertex parent =
            result.predecessor[current];

        if (parent == current)
        {
            return {};
        }

        current = parent;
    }

    path.push_back(
        source);

    std::reverse(
        path.begin(),
        path.end());

    return path;
}
