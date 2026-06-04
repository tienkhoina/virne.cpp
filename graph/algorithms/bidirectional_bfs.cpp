#include "bidirectional_bfs.h"

#include <algorithm>
#include <limits>
#include <queue>

BidirectionalBFSResult
bidirectional_bfs(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    BidirectionalBFSResult out;

    const size_t n =
        g.num_nodes();

    if (source >= n
        ||
        target >= n)
    {
        return out;
    }

    if (source == target)
    {
        out.found = true;
        out.distance = 0;
        out.path = {source};

        return out;
    }

    std::vector<char>
        seen_f(n, 0),
        seen_b(n, 0);

    std::vector<Vertex>
        parent_f(n),
        parent_b(n);

    std::queue<Vertex>
        qf,
        qb;

    seen_f[source] = 1;
    seen_b[target] = 1;

    parent_f[source] = source;
    parent_b[target] = target;

    qf.push(source);
    qb.push(target);

    Vertex meet =
        source;

    bool found =
        false;

    while (!qf.empty()
           &&
           !qb.empty()
           &&
           !found)
    {
        const size_t forward_size =
            qf.size();

        for (size_t i = 0;
             i < forward_size;
             ++i)
        {
            Vertex u =
                qf.front();

            qf.pop();

            const auto& out_edges =
                g.neighbors_fast(u);

            for (const auto& edge :
                 out_edges)
            {
                Vertex v =
                    edge.get_target();

                if (seen_f[v])
                {
                    continue;
                }

                seen_f[v] = 1;
                parent_f[v] = u;

                if (seen_b[v])
                {
                    meet = v;
                    found = true;
                    break;
                }

                qf.push(v);
            }

            if (found)
            {
                break;
            }
        }

        if (found)
        {
            break;
        }

        const size_t backward_size =
            qb.size();

        for (size_t i = 0;
             i < backward_size;
             ++i)
        {
            Vertex u =
                qb.front();

            qb.pop();

            const auto& out_edges =
                g.neighbors_fast(u);

            for (const auto& edge :
                 out_edges)
            {
                Vertex v =
                    edge.get_target();

                if (seen_b[v])
                {
                    continue;
                }

                seen_b[v] = 1;
                parent_b[v] = u;

                if (seen_f[v])
                {
                    meet = v;
                    found = true;
                    break;
                }

                qb.push(v);
            }

            if (found)
            {
                break;
            }
        }
    }

    if (!found)
    {
        return out;
    }

    std::vector<Vertex> left;

    for (Vertex v = meet;;)
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

    for (Vertex v = meet;
         v != target;)
    {
        v = parent_b[v];

        right.push_back(v);
    }

    out.path =
        std::move(left);

    out.path.insert(
        out.path.end(),
        right.begin(),
        right.end());

    out.distance =
        out.path.size() - 1;

    out.found =
        true;

    return out;
}