#include "bidirectional_dijkstra.h"

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
    Vertex v;
};

struct NodeStateGreater
{
    bool operator()(
        const NodeState& a,
        const NodeState& b) const noexcept
    {
        return a.dist > b.dist;
    }
};

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

inline bool is_banned_edge(
    Vertex u,
    Vertex v,
    const EdgeSet& banned_edges)
{
    return
        banned_edges.find(
            normalize_edge_key(
                u,
                v))
        != banned_edges.end();
}

std::unordered_set<uint32_t> build_banned_edge_ids(
    const Graph& g,
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

        Edge e = g.edge(u, v);
        banned_ids.insert(
            g.edge_id(e));
    }

    return banned_ids;
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
    BidirectionalPathResult out;

    const size_t n =
        g.num_nodes();

    if (source >= n ||
        target >= n)
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
        g.attr_id(weight_attr);

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

    seen_f[source] = 1;
    seen_b[target] = 1;

    qf.push({
        0.0,
        source});

    qb.push({
        0.0,
        target});

    bool have_best = false;
    double best_cost = INF;
    Vertex best_meet = source;

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

                    qf.push({
                        nd,
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

                    qb.push({
                        nd,
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

    std::vector<Vertex> left;

    for (Vertex v = best_meet;;)
    {
        left.push_back(v);

        if (v == source)
        {
            break;
        }

        v = parent_f[v];
    }

    std::reverse(
        left.begin(),
        left.end());

    std::vector<Vertex> right;

    for (Vertex v = best_meet; v != target;)
    {
        Vertex nxt =
            parent_b[v];

        if (nxt == v)
        {
            return out;
        }

        right.push_back(nxt);
        v = nxt;
    }

    out.found = true;
    out.cost = best_cost;
    out.path = std::move(left);
    out.path.insert(
        out.path.end(),
        right.begin(),
        right.end());

    return out;
}