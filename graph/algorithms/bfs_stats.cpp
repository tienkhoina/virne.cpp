#include "bfs_stats.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

template <typename GraphType>
BFSStats
bfs_stats_impl(
    const GraphType& g,
    Vertex source,
    BFSWorkspace& ws)
{
    const size_t n =
        g.num_nodes();

    if (source >= n)
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }

    if (ws.dist.size() != n ||
        ws.queue.size() != n)
    {
        throw std::invalid_argument(
            "BFS workspace size does not match graph");
    }

    constexpr size_t INF =
        std::numeric_limits<size_t>::max();

    auto& dist = ws.dist;
    auto& queue = ws.queue;

    std::fill(
        dist.begin(),
        dist.end(),
        INF);

    size_t head = 0;
    size_t tail = 0;

    dist[source] = 0;

    queue[tail++] =
        source;

    BFSStats stats;

    while (head < tail)
    {
        const Vertex u =
            queue[head++];

        const size_t next_dist =
            dist[u] + 1;

        const auto& out =
            g.neighbors_fast(u);

        for (const auto& e : out)
        {
            const Vertex v =
                e.get_target();

            if (dist[v] != INF)
            {
                continue;
            }

            dist[v] =
                next_dist;

            stats.sum_dist +=
                static_cast<double>(
                    next_dist);

            ++stats.reachable;

            queue[tail++] =
                v;
        }
    }

    return stats;
}

} // namespace

BFSStats bfs_stats(
    const Graph& g,
    Vertex source,
    BFSWorkspace& ws)
{
    return bfs_stats_impl(g, source, ws);
}

BFSStats bfs_stats(
    const DiGraph& g,
    Vertex source,
    BFSWorkspace& ws)
{
    return bfs_stats_impl(g, source, ws);
}
