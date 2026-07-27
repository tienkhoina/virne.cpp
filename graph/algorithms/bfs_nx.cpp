
#include "bfs_nx.h"

#include <stdexcept>
#include <vector>

namespace
{

template <typename GraphType>
nx::OrderedVertexMap<size_t>
bfs_nx_impl(
    const GraphType& g,
    Vertex source)
{
    nx::OrderedVertexMap<size_t> result;

    const size_t n =
        g.num_nodes();

    if (source >= n)
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }

    result.reserve(
        n);

    std::vector<uint8_t>
        seen(
            n,
            0);

    std::vector<Vertex>
        nextlevel;

    nextlevel.push_back(
        source);

    seen[source] = 1;

    result[source] = 0;

    size_t level = 0;

    size_t num_seen = 1;

    while (
        !nextlevel.empty())
    {
        ++level;

        std::vector<Vertex>
            thislevel;

        thislevel.swap(
            nextlevel);

        for (Vertex v :
             thislevel)
        {
            const auto& out =
                g.neighbors_fast(
                    v);

            for (const auto& e :
                 out)
            {
                const Vertex w =
                    e.get_target();

                if (seen[w])
                {
                    continue;
                }

                seen[w] = 1;

                ++num_seen;

                nextlevel.push_back(
                    w);

                result[w] =
                    level;
            }

            if (
                num_seen == n)
            {
                return result;
            }
        }
    }

    return result;
}

} // namespace

nx::OrderedVertexMap<size_t>
bfs_nx(
    const Graph& g,
    Vertex source)
{
    return bfs_nx_impl(g, source);
}

nx::OrderedVertexMap<size_t>
bfs_nx(
    const DiGraph& g,
    Vertex source)
{
    return bfs_nx_impl(g, source);
}
