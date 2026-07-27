#pragma once

#include "../graph_types.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nx
{

// Dict-like result container specialized for Virne's contiguous Vertex IDs.
// Values iterate in insertion order, while lookup uses a dense Vertex ->
// insertion-position table and therefore stays O(1) without hashing.
template <typename T>
class OrderedVertexMap
{
public:
    using key_type = Vertex;
    using mapped_type = T;
    using value_type = std::pair<const Vertex, T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator =
        typename std::vector<value_type>::const_iterator;

    OrderedVertexMap() = default;

    OrderedVertexMap(const OrderedVertexMap&) = default;
    OrderedVertexMap(OrderedVertexMap&&) noexcept = default;

    OrderedVertexMap& operator=(
        const OrderedVertexMap& other)
    {
        if (this != &other)
        {
            OrderedVertexMap copy(other);
            swap(copy);
        }
        return *this;
    }

    OrderedVertexMap& operator=(
        OrderedVertexMap&&) noexcept = default;

    explicit OrderedVertexMap(
        size_type vertex_capacity)
        :
        positions_(vertex_capacity, npos)
    {
        entries_.reserve(vertex_capacity);
    }

    void swap(OrderedVertexMap& other) noexcept
    {
        entries_.swap(other.entries_);
        positions_.swap(other.positions_);
    }

    bool empty() const noexcept
    {
        return entries_.empty();
    }

    size_type size() const noexcept
    {
        return entries_.size();
    }

    size_type capacity() const noexcept
    {
        return entries_.capacity();
    }

    void reserve(size_type vertex_capacity)
    {
        entries_.reserve(vertex_capacity);
        if (positions_.size() < vertex_capacity)
        {
            positions_.resize(vertex_capacity, npos);
        }
    }

    void clear() noexcept
    {
        entries_.clear();
        std::fill(
            positions_.begin(),
            positions_.end(),
            npos);
    }

    iterator begin() noexcept
    {
        return entries_.begin();
    }

    const_iterator begin() const noexcept
    {
        return entries_.begin();
    }

    const_iterator cbegin() const noexcept
    {
        return entries_.cbegin();
    }

    iterator end() noexcept
    {
        return entries_.end();
    }

    const_iterator end() const noexcept
    {
        return entries_.end();
    }

    const_iterator cend() const noexcept
    {
        return entries_.cend();
    }

    iterator find(Vertex key) noexcept
    {
        const size_type position = position_of(key);
        return position == npos
            ? entries_.end()
            : entries_.begin() +
                  static_cast<difference_type>(position);
    }

    const_iterator find(Vertex key) const noexcept
    {
        const size_type position = position_of(key);
        return position == npos
            ? entries_.end()
            : entries_.begin() +
                  static_cast<difference_type>(position);
    }

    bool contains(Vertex key) const noexcept
    {
        return position_of(key) != npos;
    }

    size_type count(Vertex key) const noexcept
    {
        return contains(key) ? 1 : 0;
    }

    std::vector<T> values() const
    {
        std::vector<T> result;
        result.reserve(entries_.size());
        for (const auto& entry : entries_)
        {
            result.push_back(entry.second);
        }
        return result;
    }

    const std::vector<value_type>& items() const noexcept
    {
        return entries_;
    }

    T& at(Vertex key)
    {
        const size_type position = position_of(key);
        if (position == npos)
        {
            throw std::out_of_range(
                "OrderedVertexMap key is not present");
        }
        return entries_[position].second;
    }

    const T& at(Vertex key) const
    {
        const size_type position = position_of(key);
        if (position == npos)
        {
            throw std::out_of_range(
                "OrderedVertexMap key is not present");
        }
        return entries_[position].second;
    }

    T& operator[](Vertex key)
    {
        return try_emplace(key).first->second;
    }

    template <typename... Args>
    std::pair<iterator, bool> try_emplace(
        Vertex key,
        Args&&... args)
    {
        ensure_key_capacity(key);
        const size_type existing = positions_[key];
        if (existing != npos)
        {
            return {
                entries_.begin() +
                    static_cast<difference_type>(existing),
                false};
        }

        const size_type position = entries_.size();
        entries_.emplace_back(
            key,
            T(std::forward<Args>(args)...));
        positions_[key] = position;
        return {entries_.begin() +
                    static_cast<difference_type>(position),
                true};
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(
        Vertex key,
        Args&&... args)
    {
        return try_emplace(
            key,
            std::forward<Args>(args)...);
    }

    std::pair<iterator, bool> insert(
        const value_type& value)
    {
        return try_emplace(
            value.first,
            value.second);
    }

    std::pair<iterator, bool> insert(
        value_type&& value)
    {
        return try_emplace(
            value.first,
            std::move(value.second));
    }

    template <typename M>
    std::pair<iterator, bool> insert_or_assign(
        Vertex key,
        M&& value)
    {
        auto [it, inserted] = try_emplace(
            key,
            std::forward<M>(value));
        if (!inserted)
        {
            it->second = std::forward<M>(value);
        }
        return {it, inserted};
    }

private:
    static constexpr size_type npos =
        std::numeric_limits<size_type>::max();

    size_type position_of(Vertex key) const noexcept
    {
        return key < positions_.size()
            ? positions_[key]
            : npos;
    }

    void ensure_key_capacity(Vertex key)
    {
        if (key == std::numeric_limits<Vertex>::max())
        {
            throw std::overflow_error(
                "OrderedVertexMap key cannot be represented densely");
        }
        if (key >= positions_.size())
        {
            positions_.resize(
                static_cast<size_type>(key) + size_type{1},
                npos);
        }
    }

    std::vector<value_type> entries_;
    std::vector<size_type> positions_;
};

} // namespace nx
