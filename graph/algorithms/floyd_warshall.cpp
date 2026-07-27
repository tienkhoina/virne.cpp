#include "floyd_warshall.h"

#include <limits>
#include <stdexcept>

namespace
{

template <typename NeighborType>
inline double fast_edge_weight(
    const NeighborType& edge,
    AttrId weight_attr_id)
{
    const auto& attrs =
        edge.get_property().attrs;

    const AttrValue* value =
        attrs.find(weight_attr_id);

    if (value == nullptr)
    {
        return 1.0;
    }

    return attr_to_double(*value);
}

template <typename GraphType>
DistanceMatrix floyd_warshall_impl(
    const GraphType& g,
    const std::string& weight_attr)
{
    constexpr double INF =
        std::numeric_limits<double>::infinity();

    const size_t n =
        g.num_nodes();

    const AttrId weight_attr_id =
        g.attr_id(weight_attr);

    DistanceMatrix dist(
        n);

    for (size_t i = 0;
         i < n;
         ++i)
    {
        for (size_t j = 0;
             j < n;
             ++j)
        {
            dist(i, j) =
                (i == j)
                ? 0.0
                : INF;
        }
    }

    for (Vertex u = 0;
         u < n;
         ++u)
    {
        const auto& out =
            g.neighbors_fast(
                u);

        for (const auto& edge :
             out)
        {
            const Vertex v =
                edge.get_target();

            const double w =
                fast_edge_weight(
                    edge,
                    weight_attr_id);

            if (w < dist(u, v))
            {
                dist(u, v) = w;
            }
        }
    }

    for (size_t k = 0;
         k < n;
         ++k)
    {
        for (size_t i = 0;
             i < n;
             ++i)
        {
            const double dik =
                dist(i, k);

            if (dik == INF)
            {
                continue;
            }

            for (size_t j = 0;
                 j < n;
                 ++j)
            {
                const double dkj =
                    dist(k, j);

                if (dkj == INF)
                {
                    continue;
                }

                const double candidate =
                    dik + dkj;

                if (candidate <
                    dist(i, j))
                {
                    dist(i, j) =
                        candidate;
                }
            }
        }
    }

    return dist;
}

} // namespace

DistanceMatrix floyd_warshall(
    const Graph& g,
    const std::string& weight_attr)
{
    return floyd_warshall_impl(
        g,
        weight_attr);
}

DistanceMatrix floyd_warshall(
    const DiGraph& g,
    const std::string& weight_attr)
{
    return floyd_warshall_impl(
        g,
        weight_attr);
}
