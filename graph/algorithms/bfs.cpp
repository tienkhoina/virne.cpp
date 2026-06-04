#include "bfs.h"

#include <limits>
#include <vector>

BFSResult bfs(
    const Graph& g,
    Vertex source)
{
    const size_t n =
        g.num_nodes();

    constexpr size_t INF =
        std::numeric_limits<size_t>::max();

    BFSResult result;

    result.distance.assign(
        n,
        INF);

    result.predecessor.assign(
        n,
        Vertex(-1));

    auto* dist =
        result.distance.data();

    auto* pred =
        result.predecessor.data();

    std::vector<Vertex> queue(n);

    size_t head = 0;
    size_t tail = 0;

    dist[source] = 0;
    pred[source] = source;

    queue[tail++] = source;

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

            dist[v] = next_dist;
            pred[v] = u;

            queue[tail++] = v;
        }
    }

    return result;
}