#include "subgraph.h"

namespace
{

template <typename GraphType,
          typename EdgeType,
          typename Predicate>
SearchMask build_mask(
    const GraphType& graph,
    const nx::NodeFilter& filter_node,
    const Predicate& filter_edge)
{
    SearchMask mask(
        graph.num_nodes(),
        graph.edge_id_capacity(),
        false);

    for (Vertex v = 0;
         v < graph.num_nodes();
         ++v)
    {
        mask.set_node(
            v,
            !filter_node ||
                filter_node(v));
    }

    auto [it, end] = graph.edges();
    for (; it != end; ++it)
    {
        const EdgeType edge = *it;
        const Vertex u = graph.source(edge);
        const Vertex v = graph.target(edge);
        const uint32_t id = graph.edge_id(edge);

        const bool endpoints_allowed =
            mask.allows_node(u) &&
            mask.allows_node(v);

        mask.set_edge(
            id,
            endpoints_allowed &&
                (!filter_edge ||
                 filter_edge(u, v, id)));
    }

    return mask;
}

template <typename GraphType,
          typename EdgeType>
nx::FilteredGraphView<GraphType>
subgraph_view_impl(
    const GraphType& graph,
    const nx::NodeFilter& filter_node,
    const nx::EdgeFilter& filter_edge)
{
    nx::EdgeIdFilter adapted;
    if (filter_edge)
    {
        adapted =
            [filter_edge](
                Vertex u,
                Vertex v,
                uint32_t)
            {
                return filter_edge(u, v);
            };
    }

    using View =
        nx::FilteredGraphView<GraphType>;
    typename View::MaskBuilder builder =
        [filter_node, adapted](
            const GraphType& current)
        {
            return build_mask<
                GraphType,
                EdgeType>(
                    current,
                    filter_node,
                    adapted);
        };
    return View(
        graph,
        std::move(builder));
}

template <typename GraphType,
          typename EdgeType>
nx::FilteredGraphView<GraphType>
subgraph_view_by_id_impl(
    const GraphType& graph,
    const nx::NodeFilter& filter_node,
    const nx::EdgeIdFilter& filter_edge)
{
    using View =
        nx::FilteredGraphView<GraphType>;
    typename View::MaskBuilder builder =
        [filter_node, filter_edge](
            const GraphType& current)
        {
            return build_mask<
                GraphType,
                EdgeType>(
                    current,
                    filter_node,
                    filter_edge);
        };
    return View(
        graph,
        std::move(builder));
}

} // namespace

namespace nx
{

GraphView subgraph_view(
    const Graph& graph,
    NodeFilter filter_node,
    EdgeFilter filter_edge)
{
    return subgraph_view_impl<Graph, Edge>(
        graph,
        filter_node,
        filter_edge);
}

DiGraphView subgraph_view(
    const DiGraph& graph,
    NodeFilter filter_node,
    EdgeFilter filter_edge)
{
    return subgraph_view_impl<DiGraph, DiEdge>(
        graph,
        filter_node,
        filter_edge);
}

GraphView subgraph_view_by_id(
    const Graph& graph,
    NodeFilter filter_node,
    EdgeIdFilter filter_edge)
{
    return subgraph_view_by_id_impl<Graph, Edge>(
        graph,
        filter_node,
        filter_edge);
}

DiGraphView subgraph_view_by_id(
    const DiGraph& graph,
    NodeFilter filter_node,
    EdgeIdFilter filter_edge)
{
    return subgraph_view_by_id_impl<DiGraph, DiEdge>(
        graph,
        filter_node,
        filter_edge);
}

GraphView subgraph_view(
    const Graph& graph,
    SearchMask mask)
{
    return GraphView(
        graph,
        std::move(mask));
}

DiGraphView subgraph_view(
    const DiGraph& graph,
    SearchMask mask)
{
    return DiGraphView(
        graph,
        std::move(mask));
}

GraphView subgraph(
    const Graph& graph,
    const std::vector<Vertex>& nodes)
{
    std::vector<uint8_t> included(
        graph.num_nodes(),
        uint8_t{0});
    for (const Vertex node : nodes)
    {
        if (node < included.size())
        {
            included[node] = 1;
        }
    }
    return subgraph_view(
        graph,
        [included = std::move(included)](
            Vertex node)
        {
            return node < included.size() &&
                   included[node] != 0;
        });
}

DiGraphView subgraph(
    const DiGraph& graph,
    const std::vector<Vertex>& nodes)
{
    std::vector<uint8_t> included(
        graph.num_nodes(),
        uint8_t{0});
    for (const Vertex node : nodes)
    {
        if (node < included.size())
        {
            included[node] = 1;
        }
    }
    return subgraph_view(
        graph,
        [included = std::move(included)](
            Vertex node)
        {
            return node < included.size() &&
                   included[node] != 0;
        });
}

} // namespace nx
