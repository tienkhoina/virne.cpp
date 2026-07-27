#include "centrality.h"

#include "../algorithms/bfs_nx.h"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>

namespace
{

constexpr double INF = std::numeric_limits<double>::infinity();

struct HeapNode
{
    double dist;
    Vertex vertex;

    bool operator>(const HeapNode& other) const noexcept
    {
        return dist > other.dist;
    }
};

template <typename NeighborType>
double edge_weight(const NeighborType& edge, AttrId attr_id)
{
    const AttrValue* value =
        edge.get_property().attrs.find(attr_id);
    return value ? attr_to_double(*value) : 1.0;
}

struct BrandesSSSP
{
    std::vector<double> dist;
    std::vector<double> sigma;
    std::vector<std::vector<Vertex>> pred;
    std::vector<Vertex> stack;
};

template <typename GraphType>
BrandesSSSP brandes_dijkstra(
    const GraphType& g,
    Vertex source,
    AttrId weight_attr_id)
{
    const size_t n = g.num_nodes();
    BrandesSSSP result;
    result.dist.assign(n, INF);
    result.sigma.assign(n, 0.0);
    result.pred.resize(n);
    result.stack.reserve(n);

    std::priority_queue<
        HeapNode,
        std::vector<HeapNode>,
        std::greater<HeapNode>> queue;

    result.dist[source] = 0.0;
    result.sigma[source] = 1.0;
    queue.push({0.0, source});

    while (!queue.empty())
    {
        const HeapNode current = queue.top();
        queue.pop();
        const double du = current.dist;
        const Vertex u = current.vertex;
        if (du != result.dist[u])
        {
            continue;
        }

        result.stack.push_back(u);
        for (const auto& edge : g.neighbors_fast(u))
        {
            const Vertex v = edge.get_target();
            const double weight = edge_weight(edge, weight_attr_id);
            if (weight < 0.0)
            {
                throw std::runtime_error(
                    "Negative edge weights are not supported");
            }
            const double nd = du + weight;
            if (nd < result.dist[v])
            {
                result.dist[v] = nd;
                result.sigma[v] = result.sigma[u];
                result.pred[v].clear();
                result.pred[v].push_back(u);
                queue.push({nd, v});
            }
            else if (nd == result.dist[v])
            {
                result.sigma[v] += result.sigma[u];
                result.pred[v].push_back(u);
            }
        }
    }
    return result;
}

template <typename GraphType>
BrandesSSSP brandes_bfs(
    const GraphType& g,
    Vertex source)
{
    const size_t n = g.num_nodes();
    BrandesSSSP result;
    result.dist.assign(n, INF);
    result.sigma.assign(n, 0.0);
    result.pred.resize(n);
    result.stack.reserve(n);

    std::vector<Vertex> queue(n);
    size_t head = 0;
    size_t tail = 0;
    result.dist[source] = 0.0;
    result.sigma[source] = 1.0;
    queue[tail++] = source;

    while (head < tail)
    {
        const Vertex u = queue[head++];
        result.stack.push_back(u);
        const double next_distance =
            result.dist[u] + 1.0;

        for (const auto& edge :
             g.neighbors_fast(u))
        {
            const Vertex v = edge.get_target();
            if (result.dist[v] == INF)
            {
                result.dist[v] = next_distance;
                queue[tail++] = v;
            }
            if (result.dist[v] == next_distance)
            {
                result.sigma[v] += result.sigma[u];
                result.pred[v].push_back(u);
            }
        }
    }
    return result;
}

template <typename GraphType>
nx::NodeScores degree_centrality_impl(const GraphType& g)
{
    const size_t n = g.num_nodes();
    nx::NodeScores scores(n, 0.0);
    if (n == 0)
    {
        return scores;
    }
    // This is NetworkX's convention for a singleton graph.
    if (n == 1)
    {
        scores[0] = 1.0;
        return scores;
    }
    const double scale = 1.0 / static_cast<double>(n - 1);
    for (Vertex v = 0; v < n; ++v)
    {
        scores[v] = static_cast<double>(g.degree(v)) * scale;
    }
    return scores;
}

nx::NodeScores eigenvector_centrality_impl(
    const Graph& g,
    size_t max_iter,
    double tol)
{
    const size_t n = g.num_nodes();
    nx::NodeScores x(n, n ? 1.0 / static_cast<double>(n) : 0.0);
    if (n == 0)
    {
        return x;
    }

    for (size_t iter = 0; iter < max_iter; ++iter)
    {
        const nx::NodeScores xlast = x;
        nx::NodeScores xnew = xlast; // multiply by A + I
        auto [it, end] = g.edges();
        for (; it != end; ++it)
        {
            const auto e = *it;
            const Vertex u = g.source(e);
            const Vertex v = g.target(e);
            if (u == v)
            {
                xnew[u] += xlast[u];
            }
            else
            {
                xnew[u] += xlast[v];
                xnew[v] += xlast[u];
            }
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

nx::NodeScores eigenvector_centrality_impl(
    const DiGraph& g,
    size_t max_iter,
    double tol)
{
    const size_t n = g.num_nodes();
    nx::NodeScores x(n, n ? 1.0 / static_cast<double>(n) : 0.0);
    if (n == 0)
    {
        return x;
    }

    for (size_t iter = 0; iter < max_iter; ++iter)
    {
        const nx::NodeScores xlast = x;
        nx::NodeScores xnew = xlast; // multiply by A + I (left eigenvector)
        auto [it, end] = g.edges();
        for (; it != end; ++it)
        {
            const auto e = *it;
            const Vertex u = g.source(e);
            const Vertex v = g.target(e);
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

template <typename GraphType>
nx::NodeScores closeness_centrality_impl(const GraphType& g)
{
    const size_t n = g.num_nodes();
    nx::NodeScores scores(n, 0.0);
    for (Vertex source = 0; source < n; ++source)
    {
        std::vector<int64_t> distance(n, -1);
        std::queue<Vertex> queue;
        distance[source] = 0;
        queue.push(source);
        while (!queue.empty())
        {
            const Vertex u = queue.front();
            queue.pop();
            for (const auto& edge : g.neighbors_fast(u))
            {
                const Vertex v = edge.get_target();
                if (distance[v] < 0)
                {
                    distance[v] = distance[u] + 1;
                    queue.push(v);
                }
            }
        }

        double sum = 0.0;
        size_t reachable = 0;
        for (int64_t value : distance)
        {
            if (value >= 0)
            {
                ++reachable;
                sum += static_cast<double>(value);
            }
        }
        --reachable; // omit the source itself
        if (sum > 0.0 && reachable > 0 && n > 1)
        {
            const double r = static_cast<double>(reachable);
            scores[source] = (r / sum) *
                             (r / static_cast<double>(n - 1));
        }
    }
    return scores;
}

nx::NodeScores closeness_centrality_digraph_impl(const DiGraph& g)
{
    const size_t n = g.num_nodes();
    nx::NodeScores scores(n, 0.0);
    for (Vertex source = 0; source < n; ++source)
    {
        // NetworkX measures inward distance for a DiGraph.  The bidirectional
        // adjacency list exposes the predecessor list without an allocation.
        std::vector<int64_t> distance(n, -1);
        std::queue<Vertex> queue;
        distance[source] = 0;
        queue.push(source);
        while (!queue.empty())
        {
            const Vertex u = queue.front();
            queue.pop();
            const auto& incoming = g.raw().m_vertices[u].m_in_edges;
            for (const auto& edge : incoming)
            {
                const Vertex v = edge.get_target();
                if (distance[v] < 0)
                {
                    distance[v] = distance[u] + 1;
                    queue.push(v);
                }
            }
        }

        double sum = 0.0;
        size_t reachable = 0;
        for (int64_t value : distance)
        {
            if (value >= 0)
            {
                ++reachable;
                sum += static_cast<double>(value);
            }
        }
        --reachable;
        if (sum > 0.0 && reachable > 0 && n > 1)
        {
            const double r = static_cast<double>(reachable);
            scores[source] = (r / sum) *
                             (r / static_cast<double>(n - 1));
        }
    }
    return scores;
}

template <typename GraphType>
nx::NodeScores betweenness_centrality_impl(
    const GraphType& g,
    const std::string& weight_attr)
{
    const size_t n = g.num_nodes();
    nx::NodeScores betweenness(n, 0.0);
    if (n == 0)
    {
        return betweenness;
    }

    const bool weighted = !weight_attr.empty();
    const AttrId weight_id = weighted
        ? g.attr_id(weight_attr)
        : AttrId{0};
    std::vector<double> delta(n, 0.0);
    for (Vertex source = 0; source < n; ++source)
    {
        BrandesSSSP sssp = weighted
            ? brandes_dijkstra(g, source, weight_id)
            : brandes_bfs(g, source);
        std::fill(delta.begin(), delta.end(), 0.0);
        while (!sssp.stack.empty())
        {
            const Vertex w = sssp.stack.back();
            sssp.stack.pop_back();
            for (Vertex v : sssp.pred[w])
            {
                if (sssp.sigma[w] != 0.0)
                {
                    delta[v] += (sssp.sigma[v] / sssp.sigma[w]) *
                                (1.0 + delta[w]);
                }
            }
            if (w != source)
            {
                betweenness[w] += delta[w];
            }
        }
    }

    if (n > 2)
    {
        const double denominator =
            static_cast<double>(n - 1) * static_cast<double>(n - 2);
        // NetworkX's undirected normalization uses 2/((n-1)(n-2));
        // Brandes' undirected pass below applies the corresponding 1/2
        // correction because each unordered pair is visited twice.
        const double scale = std::is_same_v<GraphType, Graph>
                                 ? 2.0 / denominator
                                 : 1.0 / denominator;
        for (double& value : betweenness)
        {
            value *= scale;
        }
    }
    // For an undirected graph Brandes counts each unordered pair twice.
    if constexpr (std::is_same_v<GraphType, Graph>)
    {
        for (double& value : betweenness)
        {
            value *= 0.5;
        }
    }
    return betweenness;
}

} // namespace

namespace nx
{

NodeScores degree_centrality(const Graph& g)
{
    return degree_centrality_impl(g);
}

NodeScores degree_centrality(const DiGraph& g)
{
    return degree_centrality_impl(g);
}

NodeScores eigenvector_centrality(
    const Graph& g,
    size_t max_iter,
    double tol)
{
    return eigenvector_centrality_impl(g, max_iter, tol);
}

NodeScores eigenvector_centrality(
    const DiGraph& g,
    size_t max_iter,
    double tol)
{
    return eigenvector_centrality_impl(g, max_iter, tol);
}

NodeScores closeness_centrality(const Graph& g)
{
    return closeness_centrality_impl(g);
}

NodeScores closeness_centrality(const DiGraph& g)
{
    return closeness_centrality_digraph_impl(g);
}

NodeScores betweenness_centrality(
    const Graph& g,
    const std::string& weight_attr)
{
    return betweenness_centrality_impl(g, weight_attr);
}

NodeScores betweenness_centrality(
    const DiGraph& g,
    const std::string& weight_attr)
{
    return betweenness_centrality_impl(g, weight_attr);
}

} // namespace nx
