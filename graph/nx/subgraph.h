#pragma once

#include "../algorithms/search_mask.h"
#include "../graph.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace nx
{

using NodeFilter =
    std::function<bool(Vertex)>;

using EdgeFilter =
    std::function<bool(Vertex, Vertex)>;

using EdgeIdFilter =
    std::function<
        bool(Vertex, Vertex, uint32_t)>;

template <typename GraphType>
class FilteredGraphView
{
public:
    using MaskBuilder =
        std::function<
            SearchMask(const GraphType&)>;

    FilteredGraphView(
        const GraphType& graph,
        SearchMask mask)
        :
        graph_(&graph),
        state_(
            std::make_shared<State>(
                State{
                    std::move(mask),
                    MaskBuilder{}}))
    {
    }

    FilteredGraphView(
        const GraphType& graph,
        MaskBuilder builder)
        :
        graph_(&graph),
        state_(
            std::make_shared<State>(
                State{
                    SearchMask{},
                    std::move(builder)}))
    {
    }

    const GraphType& graph() const noexcept
    {
        return *graph_;
    }

    const SearchMask& mask() const
    {
        if (state_->builder)
        {
            state_->mask =
                state_->builder(*graph_);
        }
        return state_->mask;
    }

    bool contains_node(
        Vertex v) const
    {
        const SearchMask& current = mask();
        return v < graph_->num_nodes() &&
               current.allows_node(v);
    }

    bool contains_edge(
        Vertex u,
        Vertex v) const
    {
        const SearchMask& current = mask();
        if (u >= graph_->num_nodes() ||
            v >= graph_->num_nodes() ||
            !current.allows_node(u) ||
            !current.allows_node(v) ||
            !graph_->has_edge(u, v))
        {
            return false;
        }

        return current.allows_edge(
            graph_->edge_id(
                graph_->edge(u, v)));
    }

private:
    struct State
    {
        SearchMask mask;
        MaskBuilder builder;
    };

    const GraphType* graph_;
    std::shared_ptr<State> state_;
};

using GraphView =
    FilteredGraphView<Graph>;

using DiGraphView =
    FilteredGraphView<DiGraph>;

GraphView subgraph_view(
    const Graph& graph,
    NodeFilter filter_node = {},
    EdgeFilter filter_edge = {});

DiGraphView subgraph_view(
    const DiGraph& graph,
    NodeFilter filter_node = {},
    EdgeFilter filter_edge = {});

GraphView subgraph_view_by_id(
    const Graph& graph,
    NodeFilter filter_node,
    EdgeIdFilter filter_edge);

DiGraphView subgraph_view_by_id(
    const DiGraph& graph,
    NodeFilter filter_node,
    EdgeIdFilter filter_edge);

GraphView subgraph_view(
    const Graph& graph,
    SearchMask mask);

DiGraphView subgraph_view(
    const DiGraph& graph,
    SearchMask mask);

// Induced live views. Membership is fixed to nodes that existed and were in
// `nodes` at construction; current edges between those nodes are re-evaluated
// every time mask() is requested.
GraphView subgraph(
    const Graph& graph,
    const std::vector<Vertex>& nodes);

DiGraphView subgraph(
    const DiGraph& graph,
    const std::vector<Vertex>& nodes);

} // namespace nx
