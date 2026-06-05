#include "bfs_stats.h"

#include <limits>
#include <vector>


BFSStats
bfs_stats(
    const Graph& g,
    Vertex source,
    BFSWorkspace& ws)
{
    const size_t n =
        g.num_nodes();

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