#include "detail/bidirectional_dijkstra_by_id.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace
{

struct NodeState
{
    double dist;
    uint64_t order;
    Vertex v;
};

struct NodeStateGreater
{
    bool operator()(
        const NodeState& a,
        const NodeState& b) const noexcept
    {
        if (a.dist != b.dist)
        {
            return a.dist > b.dist;
        }
        return a.order > b.order;
    }
};

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

const RawNeighborList& reverse_neighbors_fast(
    const Graph& g,
    Vertex v)
{
    return g.neighbors_fast(v);
}

const DiRawInNeighborList& reverse_neighbors_fast(
    const DiGraph& g,
    Vertex v)
{
    return g.raw().m_vertices[v].m_in_edges;
}

template <typename GraphType>
std::unordered_set<uint32_t> build_banned_edge_ids(
    const GraphType& g,
    const EdgeSet& banned_edges)
{
    std::unordered_set<uint32_t> banned_ids;

    banned_ids.reserve(
        banned_edges.size());

    for (const auto& key : banned_edges)
    {
        const Vertex u = key.first;
        const Vertex v = key.second;

        if (!g.has_edge(u, v))
        {
            continue;
        }

        const auto e = g.edge(u, v);
        banned_ids.insert(
            g.edge_id(e));
    }

    return banned_ids;
}

template <typename GraphType, typename ResolveWeightAttr>
BidirectionalPathResult
bidirectional_dijkstra_impl(
    const GraphType& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    ResolveWeightAttr resolve_weight_attr)
{
    BidirectionalPathResult out;

    const size_t n =
        g.num_nodes();

    if (source >= n ||
        target >= n)
    {
        return out;
    }

    if (mask != nullptr &&
        (!mask->allows_node(source) ||
         !mask->allows_node(target)))
    {
        return out;
    }

    if (source == target)
    {
        if (banned_vertices.find(source) !=
            banned_vertices.end())
        {
            return out;
        }

        out.found = true;
        out.cost = 0.0;
        out.path = {source};

        return out;
    }

    if (banned_vertices.find(source) !=
            banned_vertices.end() ||
        banned_vertices.find(target) !=
            banned_vertices.end())
    {
        return out;
    }

    const AttrId weight_attr_id =
        resolve_weight_attr();

    const std::unordered_set<uint32_t>
        banned_edge_ids =
            build_banned_edge_ids(
                g,
                banned_edges);

    constexpr double INF =
        std::numeric_limits<double>::infinity();

    std::vector<double>
        dist_f(n, INF),
        dist_b(n, INF);

    std::vector<Vertex>
        parent_f(n, source),
        parent_b(n, target);

    std::vector<std::vector<Vertex>>
        path_f(n),
        path_b(n);

    std::vector<char>
        seen_f(n, 0),
        seen_b(n, 0);

    std::vector<char>
        settled_f(n, 0),
        settled_b(n, 0);

    std::priority_queue<
        NodeState,
        std::vector<NodeState>,
        NodeStateGreater>
        qf,
        qb;

    dist_f[source] = 0.0;
    dist_b[target] = 0.0;

    parent_f[source] = source;
    parent_b[target] = target;

    path_f[source] = {source};
    path_b[target] = {target};

    seen_f[source] = 1;
    seen_b[target] = 1;

    uint64_t push_order = 0;

    qf.push({
        0.0,
        push_order++,
        source});

    qb.push({
        0.0,
        push_order++,
        target});

    bool have_best = false;
    double best_cost = INF;
    Vertex best_meet = source;
    std::vector<Vertex> best_path;

    auto update_best =
        [&](Vertex v)
        {
            if (!seen_f[v] ||
                !seen_b[v])
            {
                return;
            }

            const double cand =
                dist_f[v] + dist_b[v];

            if (cand < best_cost)
            {
                best_cost = cand;
                best_meet = v;
                have_best = true;

                best_path = path_f[v];
                std::vector<Vertex> reverse =
                    path_b[v];
                std::reverse(
                    reverse.begin(),
                    reverse.end());
                best_path.insert(
                    best_path.end(),
                    reverse.begin() + 1,
                    reverse.end());
            }
        };

    update_best(source);
    update_best(target);

    int dir = 1;

    while (!qf.empty() && !qb.empty())
    {
        dir = 1 - dir;

        auto& q =
            (dir == 0) ? qf : qb;

        NodeState state =
            q.top();

        q.pop();

        const double du =
            state.dist;

        const Vertex u =
            state.v;

        if (dir == 0)
        {
            if (settled_f[u])
            {
                continue;
            }

            settled_f[u] = 1;

            if (settled_b[u] && have_best)
            {
                break;
            }

            const auto& out_edges =
                g.neighbors_fast(u);

            for (const auto& edge : out_edges)
            {
                const Vertex v =
                    edge.get_target();

                if (banned_vertices.find(v) !=
                    banned_vertices.end())
                {
                    continue;
                }

                const uint32_t eid =
                    edge.get_property().edge_id;

                if (mask != nullptr &&
                    !mask->allows(u, v, eid))
                {
                    continue;
                }

                if (!banned_edge_ids.empty() &&
                    banned_edge_ids.find(eid) !=
                        banned_edge_ids.end())
                {
                    continue;
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

                if (nd < dist_f[v])
                {
                    dist_f[v] = nd;
                    parent_f[v] = u;
                    seen_f[v] = 1;

                    path_f[v] = path_f[u];
                    path_f[v].push_back(v);

                    qf.push({
                        nd,
                        push_order++,
                        v});

                    update_best(v);
                }
            }
        }
        else
        {
            if (settled_b[u])
            {
                continue;
            }

            settled_b[u] = 1;

            if (settled_f[u] && have_best)
            {
                break;
            }

            const auto& in_edges =
                reverse_neighbors_fast(
                    g,
                    u);

            for (const auto& edge : in_edges)
            {
                const Vertex v =
                    edge.get_target();

                if (banned_vertices.find(v) !=
                    banned_vertices.end())
                {
                    continue;
                }

                const uint32_t eid =
                    edge.get_property().edge_id;

                if (mask != nullptr &&
                    !mask->allows(v, u, eid))
                {
                    continue;
                }

                if (!banned_edge_ids.empty() &&
                    banned_edge_ids.find(eid) !=
                        banned_edge_ids.end())
                {
                    continue;
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

                if (nd < dist_b[v])
                {
                    dist_b[v] = nd;
                    parent_b[v] = u;
                    seen_b[v] = 1;

                    path_b[v] = path_b[u];
                    path_b[v].push_back(v);

                    qb.push({
                        nd,
                        push_order++,
                        v});

                    update_best(v);
                }
            }
        }
    }

    if (!have_best || best_cost == INF)
    {
        return out;
    }

    out.found = true;
    out.cost = best_cost;
    out.path = std::move(best_path);

    return out;
}

} // namespace

BidirectionalPathResult
bidirectional_dijkstra(
    const Graph& g,
    Vertex source,
    Vertex target,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return bidirectional_dijkstra_impl(
        g,
        source,
        target,
        nullptr,
        banned_vertices,
        banned_edges,
        [&g, &weight_attr] {
            return g.attr_id(weight_attr);
        });
}

BidirectionalPathResult
bidirectional_dijkstra(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return bidirectional_dijkstra_impl(
        g,
        source,
        target,
        nullptr,
        banned_vertices,
        banned_edges,
        [&g, &weight_attr] {
            return g.attr_id(weight_attr);
        });
}

BidirectionalPathResult
bidirectional_dijkstra(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return bidirectional_dijkstra_impl(
        g,
        source,
        target,
        &mask,
        banned_vertices,
        banned_edges,
        [&g, &weight_attr] {
            return g.attr_id(weight_attr);
        });
}

BidirectionalPathResult
bidirectional_dijkstra(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr)
{
    return bidirectional_dijkstra_impl(
        g,
        source,
        target,
        &mask,
        banned_vertices,
        banned_edges,
        [&g, &weight_attr] {
            return g.attr_id(weight_attr);
        });
}

namespace graph_detail
{

BidirectionalPathResult bidirectional_dijkstra_by_id(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    AttrId weight_attr_id)
{
    return bidirectional_dijkstra_impl(
        g,
        source,
        target,
        mask,
        banned_vertices,
        banned_edges,
        [weight_attr_id] {
            return weight_attr_id;
        });
}

BidirectionalPathResult bidirectional_dijkstra_by_id(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    AttrId weight_attr_id)
{
    return bidirectional_dijkstra_impl(
        g,
        source,
        target,
        mask,
        banned_vertices,
        banned_edges,
        [weight_attr_id] {
            return weight_attr_id;
        });
}

} // namespace graph_detail
