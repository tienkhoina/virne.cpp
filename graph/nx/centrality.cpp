#include "centrality.h"

#include <cmath>

#include "../algorithms/bfs.h"
#include "../algorithms/bfs_stats.h"

#include <queue>
#include <limits>
#include <cmath>

#include "../algorithms/bfs_nx.h"

namespace
{

constexpr double INF =
    std::numeric_limits<
        double>::max();

struct HeapNode
{
    double dist;
    Vertex vertex;

    bool operator>(
        const HeapNode& other) const noexcept
    {
        return dist >
               other.dist;
    }
};

struct BrandesSSSP
{
    std::vector<double>
        dist;

    std::vector<double>
        sigma;

    std::vector<
        std::vector<Vertex>>
        pred;

    std::vector<Vertex>
        stack;
};

inline double brandes_edge_weight(
    const RawNeighbor& edge,
    AttrId weight_attr_id)
{
    const auto& attrs =
        edge.get_property()
            .attrs;

    const AttrValue* value =
        attrs.find(
            weight_attr_id);

    if (value == nullptr)
    {
        return 1.0;
    }

    return attr_to_double(
        *value);
}

static BrandesSSSP
brandes_dijkstra(
    const Graph& g,
    Vertex source,
    AttrId weight_attr_id)
{
    const size_t n =
        g.num_nodes();

    BrandesSSSP result;

    result.dist.assign(
        n,
        INF);

    result.sigma.assign(
        n,
        0.0);

    result.pred.resize(
        n);

    result.stack.reserve(
        n);

    std::priority_queue<
        HeapNode,
        std::vector<HeapNode>,
        std::greater<HeapNode>>
        pq;

    result.dist[source] =
        0.0;

    result.sigma[source] =
        1.0;

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

        if (du != result.dist[u])
        {
            continue;
        }

        result.stack.push_back(
            u);

        const auto& out =
            g.neighbors_fast(
                u);

        for (const auto& edge : out)
        {
            const Vertex v =
                edge.get_target();

            const double w =
                brandes_edge_weight(
                    edge,
                    weight_attr_id);

            const double nd =
                du + w;

            if (nd < result.dist[v])
            {
                result.dist[v] =
                    nd;

                result.sigma[v] =
                    result.sigma[u];

                auto& pred =
                    result.pred[v];

                pred.clear();

                pred.push_back(
                    u);

                pq.push({
                    nd,
                    v});
            }
            else if (
                std::abs(
                    nd -
                    result.dist[v])
                <
                1e-12)
            {
                result.sigma[v]
                    +=
                    result.sigma[u];

                result.pred[v]
                    .push_back(
                        u);
            }
        }
    }

    return result;
}

}

namespace nx
{

NodeScores
degree_centrality(
    const Graph& g)
{
    NodeScores scores(
        g.num_nodes(),
        0.0);

    if (g.num_nodes() == 0)
    {
        return scores;
    }

    const double norm =
        g.num_nodes() > 1
            ?
            static_cast<double>(
                g.num_nodes() - 1)
            :
            1.0;

    auto [it, end] =
        g.nodes();

    for (; it != end; ++it)
    {
        const Vertex v =
            *it;

        scores[v] =
            static_cast<double>(
                g.degree(v))
            /
            norm;
    }

    return scores;
}

NodeScores
eigenvector_centrality(
    const Graph& g,
    size_t max_iter,
    double tol)
{
    const size_t n = g.num_nodes();

    NodeScores x(
        n,
        n > 0 ? 1.0 / static_cast<double>(n) : 0.0);

    if (n == 0)
    {
        return x;
    }

    for (size_t iter = 0; iter < max_iter; ++iter)
    {
        NodeScores xlast = x;
        NodeScores xnew = xlast;   // giống x = xlast.copy()

        auto [eit, eend] = g.edges();
        for (; eit != eend; ++eit)
        {
            const Edge e = *eit;
            const Vertex u = g.source(e);
            const Vertex v = g.target(e);

            xnew[u] += xlast[v];
            xnew[v] += xlast[u];
        }

        double norm = 0.0;
        for (double value : xnew)
        {
            norm += value * value;
        }
        norm = std::sqrt(norm);
        if (norm == 0.0)
        {
            norm = 1.0;
        }

        for (double& value : xnew)
        {
            value /= norm;
        }

        double error = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            error += std::abs(xnew[i] - xlast[i]);
        }

        x.swap(xnew);

        if (error < static_cast<double>(n) * tol)
        {
            return x;
        }
    }

    return x;
}

NodeScores
closeness_centrality(
    const Graph& g)
{
    const size_t n =
        g.num_nodes();

    NodeScores scores(
        n,
        0.0);

    for (Vertex source = 0;
        source < n;
        ++source)
    {
        auto dist =
            bfs_nx(
                g,
                source);

        double sum_dist =
            0.0;

        for (const auto& kv :
            dist)
        {
            sum_dist +=
                static_cast<double>(
                    kv.second);
        }

        const size_t reachable =
            dist.size() - 1;

        if (sum_dist > 0.0 &&
            reachable > 0 &&
            n > 1)
        {
            const double r =
                static_cast<double>(
                    reachable);

            scores[source] =
                (r / sum_dist)
                *
                (
                    r /
                    static_cast<double>(
                        n - 1)
                );
        }
    }

    return scores;
}

NodeScores
betweenness_centrality(
    const Graph& g,
    const std::string& weight_attr)
{
    const size_t n =
        g.num_nodes();

    NodeScores bc(
        n,
        0.0);

    if (n == 0)
    {
        return bc;
    }

    const AttrId weight_attr_id =
        g.attr_id(
            weight_attr);

    std::vector<double>
        delta(n);

    for (Vertex source = 0;
         source < n;
         ++source)
    {
        BrandesSSSP sssp =
            brandes_dijkstra(
                g,
                source,
                weight_attr_id);

        std::fill(
            delta.begin(),
            delta.end(),
            0.0);

        auto& stack =
            sssp.stack;

        auto& pred =
            sssp.pred;

        auto& sigma =
            sssp.sigma;

        while (!stack.empty())
        {
            const Vertex w =
                stack.back();

            stack.pop_back();

            for (Vertex v :
                 pred[w])
            {
                delta[v]
                    +=
                    (
                        sigma[v]
                        /
                        sigma[w]
                    )
                    *
                    (
                        1.0
                        +
                        delta[w]
                    );
            }

            if (w != source)
            {
                bc[w]
                    +=
                    delta[w];
            }
        }
    }

    //
    // undirected graph
    //

//
// undirected graph
//

for (double& value :
     bc)
{
    value *= 0.5;
}

if (n > 2)
{
    const double scale =
        2.0 /
        (
            static_cast<double>(
                n - 1)
            *
            static_cast<double>(
                n - 2)
        );

    for (double& value :
         bc)
    {
        value *= scale;
    }
}

return bc;
}

}