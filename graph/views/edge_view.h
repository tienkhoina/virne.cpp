#pragma once

#include "../graph_types.h"
#include "node_view.h"

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

class Graph;
class DiGraph;

namespace graph_views
{

template <typename GraphType>
using BareGraph =
    std::remove_const_t<GraphType>;

template <typename GraphType>
inline constexpr bool is_undirected_graph_v =
    std::is_same_v<BareGraph<GraphType>, Graph>;

template <typename GraphType>
using ViewEdge =
    std::conditional_t<
        is_undirected_graph_v<GraphType>,
        Edge,
        DiEdge>;

template <typename GraphType>
using ViewBGLGraph =
    std::conditional_t<
        is_undirected_graph_v<GraphType>,
        BGLGraph,
        BGLDiGraph>;

template <typename GraphType>
using GlobalEdgeIterator =
    typename boost::graph_traits<
        ViewBGLGraph<GraphType>>::edge_iterator;

template <typename GraphType>
using RawOutEdgeIterator =
    typename boost::graph_traits<
        ViewBGLGraph<GraphType>>::out_edge_iterator;

// Iterates one node's pinned Boost incidence storage without allocation.
// Graph collapses Boost's duplicate undirected self-loop record at insertion
// and copy boundaries, so this is the same single increment as the raw path.
template <typename GraphType>
class IncidentEdgeIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type = ViewEdge<GraphType>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    IncidentEdgeIterator() = default;

    IncidentEdgeIterator(
        GraphType& graph,
        RawOutEdgeIterator<GraphType> current,
        RawOutEdgeIterator<GraphType> end)
        : graph_(&graph),
          current_(current),
          end_(end)
    {
    }

    value_type operator*() const
    {
        return *current_;
    }

    IncidentEdgeIterator& operator++()
    {
        ++current_;
        return *this;
    }

    IncidentEdgeIterator operator++(int)
    {
        IncidentEdgeIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const IncidentEdgeIterator& lhs,
        const IncidentEdgeIterator& rhs)
    {
        return lhs.graph_ == rhs.graph_ &&
               lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        const IncidentEdgeIterator& lhs,
        const IncidentEdgeIterator& rhs)
    {
        return !(lhs == rhs);
    }

private:
    GraphType* graph_ = nullptr;
    RawOutEdgeIterator<GraphType> current_{};
    RawOutEdgeIterator<GraphType> end_{};
};

template <typename DescriptorIterator>
class EdgeEndpointIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type =
        std::pair<Vertex, Vertex>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    EdgeEndpointIterator() = default;

    explicit EdgeEndpointIterator(
        DescriptorIterator current)
        : current_(current)
    {
    }

    value_type operator*() const
    {
        const auto edge = *current_;
        return {
            edge.m_source,
            edge.m_target};
    }

    EdgeEndpointIterator& operator++()
    {
        ++current_;
        return *this;
    }

    EdgeEndpointIterator operator++(int)
    {
        EdgeEndpointIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const EdgeEndpointIterator& lhs,
        const EdgeEndpointIterator& rhs)
    {
        return lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        const EdgeEndpointIterator& lhs,
        const EdgeEndpointIterator& rhs)
    {
        return !(lhs == rhs);
    }

private:
    DescriptorIterator current_{};
};

template <typename Iterator>
class IteratorRange
{
public:
    IteratorRange(
        Iterator begin,
        Iterator end)
        : begin_(begin),
          end_(end)
    {
    }

    Iterator begin() const
    {
        return begin_;
    }

    Iterator end() const
    {
        return end_;
    }

private:
    Iterator begin_;
    Iterator end_;
};

template <typename GraphType,
          typename EdgeIterator>
struct EdgeDataRef
{
    Vertex source;
    Vertex target;
    ViewEdge<GraphType> edge;
    ViewAttrMap<GraphType>& attrs;
};

template <typename GraphType,
          typename EdgeIterator>
class EdgeDataIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type =
        EdgeDataRef<GraphType, EdgeIterator>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    EdgeDataIterator() = default;

    EdgeDataIterator(
        GraphType& graph,
        EdgeIterator current)
        : graph_(&graph),
          current_(current)
    {
    }

    value_type operator*() const
    {
        ViewEdge<GraphType> edge =
            *current_;
        ViewAttrMap<GraphType>* attrs = nullptr;
        if constexpr (std::is_const_v<GraphType>)
        {
            attrs = &static_cast<const EdgeProperty*>(
                         edge.get_property())
                         ->attrs;
        }
        else
        {
            attrs = &static_cast<EdgeProperty*>(
                         edge.get_property())
                         ->attrs;
        }
        return {
            edge.m_source,
            edge.m_target,
            edge,
            *attrs};
    }

    EdgeDataIterator& operator++()
    {
        ++current_;
        return *this;
    }

    EdgeDataIterator operator++(int)
    {
        EdgeDataIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const EdgeDataIterator& lhs,
        const EdgeDataIterator& rhs)
    {
        return lhs.graph_ == rhs.graph_ &&
               lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        const EdgeDataIterator& lhs,
        const EdgeDataIterator& rhs)
    {
        return !(lhs == rhs);
    }

private:
    GraphType* graph_ = nullptr;
    EdgeIterator current_{};
};

template <typename GraphType,
          typename EdgeIterator>
class EdgeDataRange
{
public:
    EdgeDataRange(
        GraphType& graph,
        EdgeIterator begin,
        EdgeIterator end)
        : graph_(&graph),
          begin_(begin),
          end_(end)
    {
    }

    EdgeDataIterator<GraphType, EdgeIterator>
    begin() const
    {
        return {
            *graph_,
            begin_};
    }

    EdgeDataIterator<GraphType, EdgeIterator>
    end() const
    {
        return {
            *graph_,
            end_};
    }

private:
    GraphType* graph_;
    EdgeIterator begin_;
    EdgeIterator end_;
};

template <typename GraphType>
class IncidentEdgeRange
{
public:
    using DescriptorIterator =
        IncidentEdgeIterator<GraphType>;
    using iterator =
        EdgeEndpointIterator<
            DescriptorIterator>;

    IncidentEdgeRange(
        GraphType& graph,
        Vertex node)
        : graph_(&graph),
          node_(node)
    {
        if (node >= graph.num_nodes())
        {
            throw std::out_of_range(
                "Incident edge node is out of range");
        }
    }

    DescriptorIterator descriptor_begin() const
    {
        auto [begin, end] =
            boost::out_edges(
                node_,
                graph_->raw());
        return {
            *graph_, begin, end};
    }

    DescriptorIterator descriptor_end() const
    {
        auto [begin, end] =
            boost::out_edges(
                node_,
                graph_->raw());
        static_cast<void>(begin);
        return {
            *graph_, end, end};
    }

    iterator begin() const
    {
        return iterator(
            descriptor_begin());
    }

    iterator end() const
    {
        return iterator(
            descriptor_end());
    }

    IteratorRange<DescriptorIterator>
    descriptors() const
    {
        return {
            descriptor_begin(),
            descriptor_end()};
    }

    bool empty() const
    {
        return begin() == end();
    }

    size_t size() const
    {
        return static_cast<size_t>(
            std::distance(begin(), end()));
    }

    auto data() const
    {
        return EdgeDataRange<
            GraphType,
            DescriptorIterator>(
                *graph_,
                descriptor_begin(),
                descriptor_end());
    }

    auto items() const
    {
        return data();
    }

private:
    GraphType* graph_;
    Vertex node_;
};

// Whole-graph EdgeView plus endpoint lookup. Public iteration returns
// NetworkX-shaped `(u, v)` endpoint pairs in graph edge order. descriptors()
// is the explicit zero-overhead escape hatch for internal descriptor loops;
// data() adds live, zero-copy AttrMap references.
template <typename GraphType>
class EdgeView
{
public:
    using DescriptorIterator =
        GlobalEdgeIterator<GraphType>;
    using iterator =
        EdgeEndpointIterator<
            DescriptorIterator>;

    explicit EdgeView(GraphType& graph)
        : graph_(&graph)
    {
    }

    DescriptorIterator descriptor_begin() const
    {
        return graph_->edges().first;
    }

    DescriptorIterator descriptor_end() const
    {
        return graph_->edges().second;
    }

    iterator begin() const
    {
        return iterator(
            descriptor_begin());
    }

    iterator end() const
    {
        return iterator(
            descriptor_end());
    }

    IteratorRange<DescriptorIterator>
    descriptors() const
    {
        const auto [begin, end] =
            graph_->edges();
        return {begin, end};
    }

    size_t size() const noexcept
    {
        return graph_->num_edges();
    }

    bool empty() const noexcept
    {
        return size() == 0;
    }

    bool contains(
        Vertex u,
        Vertex v) const
    {
        return u < graph_->num_nodes() &&
               v < graph_->num_nodes() &&
               graph_->has_edge(u, v);
    }

    bool contains(
        const std::pair<Vertex, Vertex>& endpoints) const
    {
        return contains(
            endpoints.first,
            endpoints.second);
    }

    ViewAttrMap<GraphType>& at(
        Vertex u,
        Vertex v) const
    {
        if (!contains(u, v))
        {
            throw std::out_of_range(
                "EdgeView edge is not present");
        }
        return graph_->edge_attrs(
            graph_->edge(u, v));
    }

    ViewAttrMap<GraphType>& operator[](
        const std::pair<Vertex, Vertex>& endpoints) const
    {
        return at(
            endpoints.first,
            endpoints.second);
    }

    auto data() const
    {
        const auto [begin, end] =
            graph_->edges();
        return EdgeDataRange<
            GraphType,
            GlobalEdgeIterator<GraphType>>(
                *graph_,
                begin,
                end);
    }

    auto items() const
    {
        return data();
    }

    IncidentEdgeRange<GraphType>
    for_node(Vertex node) const
    {
        return {
            *graph_, node};
    }

    IncidentEdgeRange<GraphType>
    operator()(Vertex node) const
    {
        return for_node(node);
    }

    template <
        typename T = BareGraph<GraphType>,
        std::enable_if_t<
            std::is_same_v<T, Graph>,
            int> = 0>
    IncidentEdgeRange<GraphType>
    incident(Vertex node) const
    {
        return for_node(node);
    }

    template <
        typename T = BareGraph<GraphType>,
        std::enable_if_t<
            std::is_same_v<T, DiGraph>,
            int> = 0>
    IncidentEdgeRange<GraphType>
    outgoing(Vertex node) const
    {
        return for_node(node);
    }

private:
    GraphType* graph_;
};

} // namespace graph_views
