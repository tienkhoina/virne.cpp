#pragma once

#include "../graph.h"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nx
{

// A dual-purpose edge key: `.first/.second` expose the NetworkX endpoint
// tuple, while the implicit integer conversion keeps old Virne loops that
// used stable edge IDs source-compatible.  New code should use the endpoint
// fields; the ID is only a compatibility accelerator.
struct EdgeAttributeKey
{
    Vertex first = 0;
    Vertex second = 0;
    uint32_t id = 0;

    EdgeAttributeKey() = default;

    EdgeAttributeKey(EdgeEndpoints endpoints)
        : first(endpoints.first),
          second(endpoints.second)
    {
    }

    EdgeAttributeKey(
        EdgeEndpoints endpoints,
        uint32_t edge_id)
        : first(endpoints.first),
          second(endpoints.second),
          id(edge_id)
    {
    }

    operator EdgeEndpoints() const noexcept
    {
        return {first, second};
    }

    operator uint32_t() const noexcept
    {
        return id;
    }

    friend bool operator==(
        const EdgeAttributeKey& lhs,
        const EdgeAttributeKey& rhs) noexcept
    {
        return lhs.first == rhs.first &&
               lhs.second == rhs.second;
    }
};

// A dict-like result whose iteration order is the graph's node/edge view
// order, matching Python dict insertion semantics.  Hash lookup remains O(1),
// while values() gives a safe, alignment-preserving column for feature code.
template <typename Key, typename Hash = std::hash<Key>>
class OrderedAttributeMap
{
public:
    using value_type = std::pair<Key, AttrValue>;
    using const_iterator = typename std::vector<value_type>::const_iterator;
    using iterator = const_iterator;

    void reserve(size_t size)
    {
        entries_.reserve(size);
        indices_.reserve(size);
    }

    std::pair<AttrValue*, bool> emplace(
        Key key,
        AttrValue value)
    {
        const auto found = indices_.find(key);
        if (found != indices_.end())
        {
            return {
                &entries_[found->second].second,
                false};
        }
        const size_t index = entries_.size();
        entries_.emplace_back(key, std::move(value));
        indices_.emplace(key, index);
        return {&entries_[index].second, true};
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    std::pair<AttrValue*, bool> emplace_with_id(
        EdgeEndpoints endpoints,
        uint32_t id,
        AttrValue value)
    {
        auto result = emplace(
            Key{endpoints, id},
            std::move(value));
        const auto found = indices_.find(Key{endpoints, id});
        const size_t index = found->second;
        compatibility_indices_[id] = index;
        entries_[index].first.id = id;
        return result;
    }

    AttrValue& operator[](Key key)
    {
        const auto found = indices_.find(key);
        if (found != indices_.end())
        {
            return entries_[found->second].second;
        }
        return *emplace(key, int64_t{0}).first;
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    AttrValue& operator[](uint32_t id)
    {
        const auto found = compatibility_indices_.find(id);
        if (found == compatibility_indices_.end())
        {
            throw std::out_of_range(
                "Edge attribute ID not found in endpoint map");
        }
        return entries_[found->second].second;
    }

    AttrValue& at(Key key)
    {
        return entries_[index_or_throw(key)].second;
    }

    const AttrValue& at(Key key) const
    {
        return entries_[index_or_throw(key)].second;
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    AttrValue& at(uint32_t id)
    {
        const auto found = compatibility_indices_.find(id);
        if (found == compatibility_indices_.end())
        {
            throw std::out_of_range("Edge attribute ID not found");
        }
        return entries_[found->second].second;
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    const AttrValue& at(uint32_t id) const
    {
        const auto found = compatibility_indices_.find(id);
        if (found == compatibility_indices_.end())
        {
            throw std::out_of_range("Edge attribute ID not found");
        }
        return entries_[found->second].second;
    }

    const_iterator find(Key key)
    {
        const auto found = indices_.find(key);
        return found == indices_.end()
            ? entries_.cend()
            : entries_.cbegin() +
                  static_cast<std::ptrdiff_t>(found->second);
    }

    const_iterator find(Key key) const
    {
        const auto found = indices_.find(key);
        return found == indices_.end()
            ? entries_.end()
            : entries_.begin() +
                  static_cast<std::ptrdiff_t>(found->second);
    }

    bool contains(Key key) const
    {
        return indices_.find(key) != indices_.end();
    }

    size_t count(Key key) const
    {
        return contains(key) ? 1U : 0U;
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    const_iterator find(uint32_t id) const
    {
        const auto found = compatibility_indices_.find(id);
        return found == compatibility_indices_.end()
            ? entries_.cend()
            : entries_.cbegin() +
                  static_cast<std::ptrdiff_t>(found->second);
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    bool contains(uint32_t id) const
    {
        return compatibility_indices_.find(id) !=
               compatibility_indices_.end();
    }

    template <typename K = Key,
              std::enable_if_t<
                  std::is_same_v<K, EdgeAttributeKey>,
                  int> = 0>
    size_t count(uint32_t id) const
    {
        return contains(id) ? 1U : 0U;
    }

    bool empty() const noexcept
    {
        return entries_.empty();
    }

    size_t size() const noexcept
    {
        return entries_.size();
    }

    const_iterator begin() noexcept { return entries_.cbegin(); }
    const_iterator end() noexcept { return entries_.cend(); }
    const_iterator begin() const noexcept { return entries_.begin(); }
    const_iterator end() const noexcept { return entries_.end(); }
    const_iterator cbegin() const noexcept { return entries_.cbegin(); }
    const_iterator cend() const noexcept { return entries_.cend(); }

    std::vector<AttrValue> values() const
    {
        std::vector<AttrValue> result;
        result.reserve(entries_.size());
        for (const auto& [key, value] : entries_)
        {
            static_cast<void>(key);
            result.push_back(value);
        }
        return result;
    }

    const std::vector<value_type>& items() const noexcept
    {
        return entries_;
    }

    friend bool operator==(
        const OrderedAttributeMap& lhs,
        const OrderedAttributeMap& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }
        for (const auto& [key, value] : lhs)
        {
            const auto found = rhs.find(key);
            if (found == rhs.end() ||
                !attr_value_equal(value, found->second))
            {
                return false;
            }
        }
        return true;
    }

    friend bool operator!=(
        const OrderedAttributeMap& lhs,
        const OrderedAttributeMap& rhs)
    {
        return !(lhs == rhs);
    }

private:
    std::vector<value_type> entries_;
    std::unordered_map<Key, size_t, Hash> indices_;
    std::unordered_map<uint32_t, size_t> compatibility_indices_;

    size_t index_or_throw(Key key) const
    {
        const auto found = indices_.find(key);
        if (found == indices_.end())
        {
            throw std::out_of_range("Attribute key not found");
        }
        return found->second;
    }
};

using NodeAttributeMap = OrderedAttributeMap<Vertex>;
struct EdgeEndpointHash
{
    size_t operator()(const EdgeEndpoints& edge) const noexcept
    {
        const size_t first = std::hash<Vertex>{}(edge.first);
        const size_t second = std::hash<Vertex>{}(edge.second);
        return first ^
            (second + size_t{0x9e3779b9} +
             (first << 6) + (first >> 2));
    }

    size_t operator()(const EdgeAttributeKey& edge) const noexcept
    {
        return (*this)(EdgeEndpoints{edge.first, edge.second});
    }
};

using EdgeAttributeMap =
    OrderedAttributeMap<EdgeAttributeKey, EdgeEndpointHash>;
// Integer-ID indexing is retained as a compatibility accessor on the same
// endpoint-keyed map; NetworkX-shaped iteration still exposes `(u, v)` keys.
using EdgeIdAttributeMap = EdgeAttributeMap;

NodeAttributeMap
get_node_attributes(
    const Graph& g,
    AttrId attr_id);

NodeAttributeMap
get_node_attributes(
    const DiGraph& g,
    AttrId attr_id);

NodeAttributeMap
get_node_attributes(
    const Graph& g,
    std::string_view name);

NodeAttributeMap
get_node_attributes(
    const DiGraph& g,
    std::string_view name);

EdgeIdAttributeMap
get_edge_attributes_by_id(
    const Graph& g,
    AttrId attr_id);

EdgeIdAttributeMap
get_edge_attributes_by_id(
    const DiGraph& g,
    AttrId attr_id);

// Compatibility spelling for callers that already resolved an AttrId. Both
// overloads iterate with NetworkX endpoint keys; the returned map also keeps
// an O(1) edge-ID side index for integer hot loops.
EdgeIdAttributeMap
get_edge_attributes(
    const Graph& g,
    AttrId attr_id);

EdgeIdAttributeMap
get_edge_attributes(
    const DiGraph& g,
    AttrId attr_id);

EdgeAttributeMap
get_edge_attributes(
    const Graph& g,
    std::string_view name);

EdgeAttributeMap
get_edge_attributes(
    const DiGraph& g,
    std::string_view name);

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    AttrId attr_id);

void set_node_attributes(
    DiGraph& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    AttrId attr_id);

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    std::string_view name);

void set_node_attributes(
    DiGraph& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    std::string_view name);

void set_edge_attributes_by_id(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    AttrId attr_id);

void set_edge_attributes_by_id(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id);

void set_edge_attributes_by_id(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    std::string_view name);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id);

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name);

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name);

void set_edge_attributes_by_id(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    AttrId attr_id);

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    AttrId attr_id);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    std::string_view name);

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    std::string_view name);

}
