#include "bidirectional_bfs.h"

#include <algorithm>
#include <limits>

namespace
{

const RawNeighborList& reverse_neighbors_fast(
    const Graph& g,
    Vertex v)
{
    return g.neighbors_fast(v);
}

const DiRawInNeighborList& reverse_neighbors_fast(
    const DiGraph& g,
    Vertex v)
{
    return g.raw().m_vertices[v].m_in_edges;
}

template <typename GraphType>
BidirectionalBFSResult
bidirectional_bfs_impl(
    const GraphType& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask)
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

    if (mask != nullptr &&
        (!mask->allows_node(source) ||
         !mask->allows_node(target)))
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

    seen_f[source] = 1;
    seen_b[target] = 1;

    parent_f[source] = source;
    parent_b[target] = target;

    std::vector<Vertex> forward_fringe{source};
    std::vector<Vertex> reverse_fringe{target};

    Vertex meet =
        source;

    bool found =
        false;

    while (!forward_fringe.empty() &&
           !reverse_fringe.empty() &&
           !found)
    {
        if (forward_fringe.size() <=
            reverse_fringe.size())
        {
            std::vector<Vertex> next_fringe;
            for (const Vertex u : forward_fringe)
            {
                for (const auto& edge :
                     g.neighbors_fast(u))
                {
                    const Vertex v =
                        edge.get_target();

                    if (mask != nullptr &&
                        !mask->allows(
                            u,
                            v,
                            edge.get_property().edge_id))
                    {
                        continue;
                    }

                    if (!seen_f[v])
                    {
                        seen_f[v] = 1;
                        parent_f[v] = u;
                        next_fringe.push_back(v);
                    }

                    if (seen_b[v])
                    {
                        meet = v;
                        found = true;
                        break;
                    }
                }

                if (found)
                {
                    break;
                }
            }
            forward_fringe =
                std::move(next_fringe);
        }
        else
        {
            std::vector<Vertex> next_fringe;
            for (const Vertex u : reverse_fringe)
            {
                const auto& in_edges =
                    reverse_neighbors_fast(g, u);

                for (const auto& edge : in_edges)
                {
                    const Vertex v =
                        edge.get_target();

                    if (mask != nullptr &&
                        !mask->allows(
                            v,
                            u,
                            edge.get_property().edge_id))
                    {
                        continue;
                    }

                    if (!seen_b[v])
                    {
                        seen_b[v] = 1;
                        parent_b[v] = u;
                        next_fringe.push_back(v);
                    }

                    if (seen_f[v])
                    {
                        meet = v;
                        found = true;
                        break;
                    }
                }

                if (found)
                {
                    break;
                }
            }
            reverse_fringe =
                std::move(next_fringe);
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

} // namespace

BidirectionalBFSResult
bidirectional_bfs(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    return bidirectional_bfs_impl(
        g,
        source,
        target,
        nullptr);
}

BidirectionalBFSResult
bidirectional_bfs(
    const DiGraph& g,
    Vertex source,
    Vertex target)
{
    return bidirectional_bfs_impl(
        g,
        source,
        target,
        nullptr);
}

BidirectionalBFSResult
bidirectional_bfs(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask)
{
    return bidirectional_bfs_impl(
        g,
        source,
        target,
        &mask);
}

BidirectionalBFSResult
bidirectional_bfs(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask)
{
    return bidirectional_bfs_impl(
        g,
        source,
        target,
        &mask);
}
