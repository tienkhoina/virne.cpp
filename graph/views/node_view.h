#pragma once

#include "../graph_types.h"

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace graph_views
{

class NodeIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type = Vertex;
    using difference_type = std::ptrdiff_t;
    using pointer = const Vertex*;
    using reference = Vertex;

    NodeIterator() = default;

    explicit NodeIterator(Vertex current)
        : current_(current)
    {
    }

    Vertex operator*() const noexcept
    {
        return current_;
    }

    NodeIterator& operator++() noexcept
    {
        ++current_;
        return *this;
    }

    NodeIterator operator++(int) noexcept
    {
        NodeIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        NodeIterator lhs,
        NodeIterator rhs) noexcept
    {
        return lhs.current_ == rhs.current_;
    }

    friend bool operator!=(
        NodeIterator lhs,
        NodeIterator rhs) noexcept
    {
        return !(lhs == rhs);
    }

private:
    Vertex current_ = 0;
};

template <typename GraphType>
using ViewAttrMap =
    std::conditional_t<
        std::is_const_v<GraphType>,
        const AttrMap,
        AttrMap>;

template <typename GraphType>
struct NodeDataRef
{
    Vertex node;
    ViewAttrMap<GraphType>& attrs;
};

template <typename GraphType>
class NodeDataIterator
{
public:
    using iterator_category =
        std::forward_iterator_tag;
    using value_type = NodeDataRef<GraphType>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;
    using StorageIterator = decltype(
        std::declval<GraphType&>()
            .raw()
            .m_vertices.begin());

    NodeDataIterator() = default;

    NodeDataIterator(
        StorageIterator storage,
        Vertex node)
        : storage_(storage),
          node_(node)
    {
    }

    value_type operator*() const
    {
        // Pinned Boost 1.85 layout fast path: bypass bundled-property maps and
        // return the AttrMap stored directly in vecS vertex storage.
        return {
            node_,
            storage_
                ->m_property
                .attrs};
    }

    NodeDataIterator& operator++() noexcept
    {
        ++storage_;
        ++node_;
        return *this;
    }

    NodeDataIterator operator++(int) noexcept
    {
        NodeDataIterator copy = *this;
        ++*this;
        return copy;
    }

    friend bool operator==(
        const NodeDataIterator& lhs,
        const NodeDataIterator& rhs) noexcept
    {
        return lhs.storage_ == rhs.storage_;
    }

    friend bool operator!=(
        const NodeDataIterator& lhs,
        const NodeDataIterator& rhs) noexcept
    {
        return !(lhs == rhs);
    }

private:
    StorageIterator storage_{};
    Vertex node_ = 0;
};

template <typename GraphType>
class NodeDataRange
{
public:
    explicit NodeDataRange(GraphType& graph)
        : begin_(graph.raw().m_vertices.begin()),
          end_(graph.raw().m_vertices.end()),
          size_(static_cast<Vertex>(
              graph.num_nodes()))
    {
    }

    NodeDataIterator<GraphType> begin() const
    {
        return NodeDataIterator<GraphType>(
            begin_, 0);
    }

    NodeDataIterator<GraphType> end() const
    {
        return NodeDataIterator<GraphType>(
            end_, size_);
    }

private:
    typename NodeDataIterator<GraphType>::StorageIterator
        begin_;
    typename NodeDataIterator<GraphType>::StorageIterator
        end_;
    Vertex size_ = 0;
};

// Pointer-sized facade over contiguous graph node indices. The view owns no
// data; attribute references point directly into the graph.
template <typename GraphType>
class NodeView
{
public:
    explicit NodeView(GraphType& graph)
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

    bool empty() const noexcept
    {
        return size() == 0;
    }

    bool contains(Vertex node) const noexcept
    {
        return node < graph_->num_nodes();
    }

    ViewAttrMap<GraphType>& at(Vertex node) const
    {
        if (!contains(node))
        {
            throw std::out_of_range(
                "NodeView node is out of range");
        }
        return graph_->node_attrs(node);
    }

    ViewAttrMap<GraphType>& operator[](
        Vertex node) const
    {
        return at(node);
    }

    NodeDataRange<GraphType> data() const
    {
        return NodeDataRange<GraphType>(
            *graph_);
    }

    NodeDataRange<GraphType> items() const
    {
        return data();
    }

    NodeView keys() const noexcept
    {
        return *this;
    }

private:
    GraphType* graph_;
};

} // namespace graph_views
