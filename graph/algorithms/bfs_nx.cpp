
#include "bfs_nx.h"

#include <vector>

std::unordered_map<
    Vertex,
    size_t>
bfs_nx(
    const Graph& g,
    Vertex source)
{
    std::unordered_map<
        Vertex,
        size_t>
        result;

    const size_t n =
        g.num_nodes();

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

