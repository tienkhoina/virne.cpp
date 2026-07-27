#include "connectivity.h"

#include <boost/graph/connected_components.hpp>

#include <cstdint>
#include <queue>
#include <vector>

namespace nx
{

bool is_connected(
    const Graph& g)
{
    if (g.num_nodes() == 0)
    {
        return true;
    }

    std::vector<int> component(g.num_nodes());
    const int num_components =
        boost::connected_components(
            g.raw(),
            component.data());

    return num_components == 1;
}

bool is_connected(
    const DiGraph& g)
{
    // For a directed graph this existing API uses weak connectivity:
    // every directed edge may be traversed in either direction for the
    // purpose of the predicate.  This corresponds to NetworkX's
    // `is_weakly_connected` while preserving the C++ API name.
    const size_t n = g.num_nodes();
    if (n == 0)
    {
        return true;
    }

    std::vector<uint8_t> seen(n, 0);
    std::queue<Vertex> queue;
    queue.push(0);
    seen[0] = 1;
    size_t count = 1;

    while (!queue.empty())
    {
        const Vertex u = queue.front();
        queue.pop();

        for (const auto& edge : g.neighbors_fast(u))
        {
            const Vertex v = edge.get_target();
            if (!seen[v])
            {
                seen[v] = 1;
                ++count;
                queue.push(v);
            }
        }

        // The reverse incidence list is needed because weak connectivity
        // treats an incoming directed edge as an undirected adjacency.
        for (const auto& edge : g.raw().m_vertices[u].m_in_edges)
        {
            const Vertex v = edge.get_target();
            if (!seen[v])
            {
                seen[v] = 1;
                ++count;
                queue.push(v);
            }
        }
    }

    return count == n;
}

} // namespace nx
