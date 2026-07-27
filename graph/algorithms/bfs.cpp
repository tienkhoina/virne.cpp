#include "bfs.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

template <typename GraphType>
BFSResult bfs_impl(
    const GraphType& g,
    Vertex source,
    const SearchMask* mask)
{
    const size_t n =
        g.num_nodes();

    if (source >= n)
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }

    if (mask != nullptr &&
        !mask->allows_node(source))
    {
        throw std::out_of_range(
            "source vertex is filtered out");
    }

    constexpr size_t INF =
        std::numeric_limits<size_t>::max();

    BFSResult result;

    result.distance.assign(
        n,
        INF);

    result.predecessor.assign(
        n,
        Vertex(-1));

    result.discovery_order.reserve(n);

    auto* dist =
        result.distance.data();

    auto* pred =
        result.predecessor.data();

    size_t head = 0;

    dist[source] = 0;
    pred[source] = source;

    result.discovery_order.push_back(source);

    while (head < result.discovery_order.size())
    {
        const Vertex u =
            result.discovery_order[head++];

        const size_t next_dist =
            dist[u] + 1;

        const auto& out =
            g.neighbors_fast(u);

        for (const auto& e : out)
        {
            const Vertex v =
                e.get_target();

            if (mask != nullptr &&
                !mask->allows(
                    u,
                    v,
                    e.get_property().edge_id))
            {
                continue;
            }

            if (dist[v] != INF)
            {
                continue;
            }

            dist[v] = next_dist;
            pred[v] = u;

            result.discovery_order.push_back(v);
        }
    }

    return result;
}

} // namespace

BFSResult bfs(
    const Graph& g,
    Vertex source)
{
    return bfs_impl(
        g,
        source,
        nullptr);
}

BFSResult bfs(
    const DiGraph& g,
    Vertex source)
{
    return bfs_impl(
        g,
        source,
        nullptr);
}

BFSResult bfs(
    const Graph& g,
    Vertex source,
    const SearchMask& mask)
{
    return bfs_impl(
        g,
        source,
        &mask);
}

BFSResult bfs(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask)
{
    return bfs_impl(
        g,
        source,
        &mask);
}
