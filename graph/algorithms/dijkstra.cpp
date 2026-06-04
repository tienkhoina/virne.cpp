#include "dijkstra.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
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
    const std::string& weight_attr)
{
    const auto& attrs =
        edge.get_property().attrs;

    auto it =
        attrs.find(weight_attr);

    if (it == attrs.end())
    {
        return 1.0;
    }

    if (std::holds_alternative<double>(
            it->second))
    {
        return std::get<double>(
            it->second);
    }

    if (std::holds_alternative<int64_t>(
            it->second))
    {
        return static_cast<double>(
            std::get<int64_t>(
                it->second));
    }

    throw std::runtime_error(
        "Edge weight must be numeric");
}

static DijkstraResult run_dijkstra(
    const Graph& g,
    Vertex source,
    const VertexSet* banned_vertices,
    const EdgeSet* banned_edges,
    const std::string& weight_attr)
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

        for (const auto& edge :
             out)
        {
            const Vertex v =
                edge.get_target();

            if (banned_vertices)
            {
                if (banned_vertices->find(v)
                    != banned_vertices->end())
                {
                    continue;
                }
            }

            if (banned_edges)
            {
                if (banned_edges->find(
                        normalize_edge_key(
                            u,
                            v))
                    != banned_edges->end())
                {
                    continue;
                }
            }

            const double w =
                fast_edge_weight(
                    edge,
                    weight_attr);

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
    const std::string& weight_attr)
{
    const auto& attrs =
        g.edge_attrs(e);

    auto it =
        attrs.find(weight_attr);

    if (it == attrs.end())
    {
        return 1.0;
    }

    if (std::holds_alternative<double>(
            it->second))
    {
        return std::get<double>(
            it->second);
    }

    if (std::holds_alternative<int64_t>(
            it->second))
    {
        return static_cast<double>(
            std::get<int64_t>(
                it->second));
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

    return run_dijkstra(
        g,
        source,
        nullptr,
        nullptr,
        weight_attr);
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

    if (banned_vertices.find(source)
        != banned_vertices.end())
    {
        throw std::runtime_error(
            "source vertex cannot be banned");
    }

    return run_dijkstra(
        g,
        source,
        &banned_vertices,
        &banned_edges,
        weight_attr);
}

double edge_cost(
    const Graph& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr)
{
    Edge e =
        g.edge(u, v);

    return read_edge_weight(
        g,
        e,
        weight_attr);
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

    double cost = 0.0;

    for (std::size_t i = 0;
         i + 1 < path.size();
         ++i)
    {
        cost += edge_cost(
            g,
            path[i],
            path[i + 1],
            weight_attr);
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

    prefix.push_back(
        0.0);

    double running = 0.0;

    for (std::size_t i = 0;
         i + 1 < path.size();
         ++i)
    {
        running += edge_cost(
            g,
            path[i],
            path[i + 1],
            weight_attr);

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
    if (source >= result.predecessor.size()
        || target >= result.predecessor.size())
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