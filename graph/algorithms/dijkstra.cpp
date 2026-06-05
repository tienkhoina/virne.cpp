#include "dijkstra.h"

#include <algorithm>
#include <limits>
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
    Vertex vertex;

    bool operator>(
        const HeapNode& other) const noexcept
    {
        return dist > other.dist;
    }
};

static DijkstraResult make_empty_result(
    const Graph& g)
{
    DijkstraResult result;

    const std::size_t n =
        g.num_nodes();

    result.distance.assign(
        n,
        INF);

    result.predecessor.resize(
        n);

    for (Vertex v = 0;
         v < n;
         ++v)
    {
        result.predecessor[v] = v;
    }

    return result;
}

inline double fast_edge_weight(
    const RawNeighbor& edge,
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

    if (std::holds_alternative<double>(
            *value))
    {
        return std::get<double>(
            *value);
    }

    if (std::holds_alternative<int64_t>(
            *value))
    {
        return static_cast<double>(
            std::get<int64_t>(
                *value));
    }

    throw std::runtime_error(
        "Edge weight must be numeric");
}

static std::unordered_set<uint32_t>
build_banned_edge_id_set(
    const Graph& g,
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

        Edge e =
            g.edge(u, v);

        banned_ids.insert(
            g.edge_id(e));
    }

    return banned_ids;
}

static DijkstraResult run_dijkstra(
    const Graph& g,
    Vertex source,
    const VertexSet* banned_vertices,
    const EdgeSet* banned_edges,
    AttrId weight_attr_id)
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

    pq.push({
        0.0,
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

        const auto& out =
            g.neighbors_fast(
                u);

        for (const auto& edge : out)
        {
            const Vertex v =
                edge.get_target();

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

            const double nd =
                du + w;

            if (nd < dist[v])
            {
                dist[v] = nd;
                pred[v] = u;

                pq.push({
                    nd,
                    v});
            }
        }
    }

    return result;
}

static double read_edge_weight(
    const Graph& g,
    Edge e,
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

    if (std::holds_alternative<double>(
            *value))
    {
        return std::get<double>(
            *value);
    }

    if (std::holds_alternative<int64_t>(
            *value))
    {
        return static_cast<double>(
            std::get<int64_t>(
                *value));
    }

    throw std::runtime_error(
        "Edge weight must be numeric");
}

} // namespace

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const std::string& weight_attr)
{
    if (source >= g.num_nodes())
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    return run_dijkstra(
        g,
        source,
        nullptr,
        nullptr,
        weight_attr_id);
}

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    if (source >= g.num_nodes())
    {
        throw std::out_of_range(
            "source vertex is out of range");
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
        &banned_vertices,
        &banned_edges,
        weight_attr_id);
}

double edge_cost(
    const Graph& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr)
{
    Edge e =
        g.edge(u, v);

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    return read_edge_weight(
        g,
        e,
        weight_attr_id);
}

double path_cost(
    const Graph& g,
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
        Edge e =
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

std::vector<double> path_prefix_costs(
    const Graph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr)
{
    std::vector<double> prefix;

    prefix.reserve(
        path.size());

    if (path.empty())
    {
        return prefix;
    }

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    prefix.push_back(
        0.0);

    double running = 0.0;

    for (std::size_t i = 0;
         i + 1 < path.size();
         ++i)
    {
        Edge e =
            g.edge(
                path[i],
                path[i + 1]);

        running += read_edge_weight(
            g,
            e,
            weight_attr_id);

        prefix.push_back(
            running);
    }

    return prefix;
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