#pragma once

#include "edge_view.h"
#include "node_view.h"

#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace graph_views
{

template <typename GraphType>
using ViewRawNeighborList =
    std::conditional_t<
        is_undirected_graph_v<GraphType>,
        RawNeighborList,
        DiRawNeighborList>;

template <typename GraphType>
using ViewRawNeighbor =
    typename ViewRawNeighborList<GraphType>::value_type;

template <typename GraphType>
class AdjacencyDataIterator;

template <typename GraphType>
class AdjacencyView;

template <typename GraphType>
class NeighborIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type = Vertex;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = Vertex;

    NeighborIterator() = default;

    explicit NeighborIterator(
        const ViewRawNeighbor<GraphType>* current)
        : current_(current)
    {
    }

    Vertex operator*() const
    {
        return current_->get_target();
    }

    NeighborIterator& operator++()
    {
        ++current_;
        return *this;
    }

    NeighborIterator operator++(int)
    {
        NeighborIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const NeighborIterator& lhs,
        const NeighborIterator& rhs)
    {
        return lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        const NeighborIterator& lhs,
        const NeighborIterator& rhs)
    {
        return !(lhs == rhs);
    }

    ViewEdge<GraphType> edge() const
    {
        // listS stores a stable iterator to the global edge record in each
        // incidence entry.  Rebuild the descriptor directly instead of doing
        // a second adjacency lookup; this also keeps the iterator itself to
        // one pointer for the plain-neighbor hot loop.
        const auto stored = current_->get_iter();
        return ViewEdge<GraphType>(
            stored->m_source,
            stored->m_target,
            &current_->get_property());
    }

    ViewAttrMap<GraphType>& attrs() const
    {
        // Direct Boost stored-edge property access is the accepted pinned
        // hot-path hack. The public mutable view is safe because GraphType is
        // non-const even though neighbors_fast exposes traversal as const.
        if constexpr (std::is_const_v<GraphType>)
        {
            return current_->get_property().attrs;
        }
        else
        {
            return const_cast<EdgeProperty&>(
                current_->get_property()).attrs;
        }
    }

private:
    const ViewRawNeighbor<GraphType>*
        current_ = nullptr;
};

static_assert(
    sizeof(NeighborIterator<Graph>) == sizeof(void*) &&
        sizeof(NeighborIterator<DiGraph>) == sizeof(void*),
    "Pinned Boost neighbor iterators must stay pointer-sized");

template <typename GraphType>
struct NeighborDataRef
{
    Vertex neighbor;
    ViewAttrMap<GraphType>& attrs;
};

template <typename GraphType>
class NeighborDataIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type = NeighborDataRef<GraphType>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    NeighborDataIterator() = default;

    NeighborDataIterator(
        GraphType& graph,
        NeighborIterator<GraphType> current)
        : graph_(&graph),
          current_(current)
    {
    }

    value_type operator*() const
    {
        return {
            *current_,
            current_.attrs()};
    }

    NeighborDataIterator& operator++()
    {
        ++current_;
        return *this;
    }

    NeighborDataIterator operator++(int)
    {
        NeighborDataIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const NeighborDataIterator& lhs,
        const NeighborDataIterator& rhs)
    {
        return lhs.graph_ == rhs.graph_ &&
               lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        const NeighborDataIterator& lhs,
        const NeighborDataIterator& rhs)
    {
        return !(lhs == rhs);
    }

private:
    GraphType* graph_ = nullptr;
    NeighborIterator<GraphType> current_{};
};

template <typename GraphType>
class NeighborDataRange
{
public:
    NeighborDataRange(
        GraphType& graph,
        NeighborIterator<GraphType> begin,
        NeighborIterator<GraphType> end)
        : graph_(&graph),
          begin_(begin),
          end_(end)
    {
    }

    NeighborDataIterator<GraphType> begin() const
    {
        return {*graph_, begin_};
    }

    NeighborDataIterator<GraphType> end() const
    {
        return {*graph_, end_};
    }

private:
    GraphType* graph_;
    NeighborIterator<GraphType> begin_;
    NeighborIterator<GraphType> end_;
};

template <typename GraphType>
class NeighborRange
{
public:
    NeighborRange(
        GraphType& graph,
        Vertex node)
        : graph_(&graph),
          node_(node)
    {
        if (node >= graph.num_nodes())
        {
            throw std::out_of_range(
                "Neighbor node is out of range");
        }
        bind_storage();
    }

    NeighborIterator<GraphType> begin() const
    {
        return NeighborIterator<GraphType>(begin_);
    }

    NeighborIterator<GraphType> end() const
    {
        return NeighborIterator<GraphType>(end_);
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

    bool contains(Vertex neighbor) const
    {
        return neighbor < graph_->num_nodes() &&
               graph_->has_edge(node_, neighbor);
    }

    ViewAttrMap<GraphType>& at(
        Vertex neighbor) const
    {
        if (!contains(neighbor))
        {
            throw std::out_of_range(
                "Neighbor is not present in adjacency view");
        }
        return graph_->edge_attrs(
            graph_->edge(node_, neighbor));
    }

    ViewAttrMap<GraphType>& operator[](
        Vertex neighbor) const
    {
        return at(neighbor);
    }

    NeighborRange keys() const
    {
        return *this;
    }

    NeighborDataRange<GraphType> data() const
    {
        return {
            *graph_, begin(), end()};
    }

    NeighborDataRange<GraphType> items() const
    {
        return data();
    }

private:
    struct UncheckedTag
    {
    };

    NeighborRange(
        GraphType& graph,
        Vertex node,
        UncheckedTag)
        : graph_(&graph),
          node_(node)
    {
        bind_storage();
    }

    void bind_storage()
    {
        const auto& edges =
            graph_->neighbors_fast(node_);
        begin_ = edges.data();
        end_ = edges.empty()
            ? begin_
            : begin_ + edges.size();
    }

    template <typename>
    friend class AdjacencyDataIterator;

    template <typename>
    friend class AdjacencyView;

    GraphType* graph_;
    Vertex node_;
    const ViewRawNeighbor<GraphType>* begin_ = nullptr;
    const ViewRawNeighbor<GraphType>* end_ = nullptr;
};

template <typename GraphType>
struct AdjacencyDataRef
{
    Vertex node;
    NeighborRange<GraphType> neighbors;
};

template <typename GraphType>
class AdjacencyDataIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type =
        AdjacencyDataRef<GraphType>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    AdjacencyDataIterator() = default;

    AdjacencyDataIterator(
        GraphType& graph,
        Vertex current)
        : graph_(&graph),
          current_(current)
    {
    }

    value_type operator*() const
    {
        return {
            current_,
            NeighborRange<GraphType>(
                *graph_,
                current_,
                typename NeighborRange<
                    GraphType>::UncheckedTag{})};
    }

    AdjacencyDataIterator& operator++() noexcept
    {
        ++current_;
        return *this;
    }

    AdjacencyDataIterator operator++(int) noexcept
    {
        AdjacencyDataIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const AdjacencyDataIterator& lhs,
        const AdjacencyDataIterator& rhs) noexcept
    {
        return lhs.graph_ == rhs.graph_ &&
               lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        const AdjacencyDataIterator& lhs,
        const AdjacencyDataIterator& rhs) noexcept
    {
        return !(lhs == rhs);
    }

private:
    GraphType* graph_ = nullptr;
    Vertex current_ = 0;
};

// NetworkX-style adjacency facade. For Graph it exposes all incident
// neighbors; for DiGraph it exposes successors. Both preserve insertion order
// and return direct edge-attribute references from data().
template <typename GraphType>
class AdjacencyView
{
public:
    explicit AdjacencyView(GraphType& graph)
        : graph_(&graph)
    {
    }

    NodeIterator begin() const noexcept
    {
        return NodeIterator(0);
    }

    NodeIterator end() const noexcept
    {
        return NodeIterator(
            static_cast<Vertex>(
                graph_->num_nodes()));
    }

    size_t size() const noexcept
    {
        return graph_->num_nodes();
    }

    bool contains(Vertex node) const noexcept
    {
        return node < graph_->num_nodes();
    }

    NeighborRange<GraphType> at(
        Vertex node) const
    {
        return {
            *graph_, node};
    }

    NeighborRange<GraphType> operator[](
        Vertex node) const
    {
        // Deliberately mirrors std::vector::operator[]: the caller validates
        // a dense index once before a hot loop. Use at() at API boundaries.
        return NeighborRange<GraphType>(
            *graph_,
            node,
            typename NeighborRange<
                GraphType>::UncheckedTag{});
    }

    class DataRange
    {
    public:
        explicit DataRange(GraphType& graph)
            : graph_(&graph)
        {
        }

        AdjacencyDataIterator<GraphType>
        begin() const
        {
            return {*graph_, 0};
        }

        AdjacencyDataIterator<GraphType>
        end() const
        {
            return {
                *graph_,
                static_cast<Vertex>(
                    graph_->num_nodes())};
        }

    private:
        GraphType* graph_;
    };

    DataRange data() const
    {
        return DataRange(*graph_);
    }

    DataRange items() const
    {
        return data();
    }

    NodeView<GraphType> keys() const
    {
        return NodeView<GraphType>(*graph_);
    }

private:
    GraphType* graph_;
};

} // namespace graph_views
