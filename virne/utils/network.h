#pragma once

#include "graph.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace virne::utils
{

// Python dictionaries permit non-string scalar keys.  The production Virne
// settings use strings, but flatten_dict_list_for_gml explicitly stringifies
// every key, so retain the observable scalar-key behavior at this boundary.
struct DynamicKey
{
    using Storage = std::variant<
        std::monostate,
        bool,
        std::int64_t,
        double,
        std::string>;

    DynamicKey() = default;
    DynamicKey(std::nullptr_t) noexcept;
    DynamicKey(bool value) noexcept;
    DynamicKey(int value) noexcept;
    DynamicKey(std::int64_t value) noexcept;
    DynamicKey(double value) noexcept;
    DynamicKey(const char* value);
    DynamicKey(std::string value);

    template <typename T>
    bool is() const noexcept
    {
        return std::holds_alternative<T>(data);
    }

    template <typename T>
    const T& as() const
    {
        return std::get<T>(data);
    }

    bool operator==(const DynamicKey& other) const
    {
        return data == other.data;
    }

    Storage data{};
};

// C++ representation of the small recursive value surface accepted by
// virne.utils.network.  Dict is a vector instead of an unordered container so
// Python insertion order is preserved for differential tests and GML output.
struct DynamicValue
{
    using List = std::vector<DynamicValue>;
    using Dict =
        std::vector<std::pair<DynamicKey, DynamicValue>>;
    using Storage = std::variant<
        std::monostate,
        bool,
        std::int64_t,
        double,
        std::string,
        List,
        Dict>;

    DynamicValue() = default;
    DynamicValue(std::nullptr_t) noexcept;
    DynamicValue(bool value) noexcept;
    DynamicValue(int value) noexcept;
    DynamicValue(std::int64_t value) noexcept;
    DynamicValue(double value) noexcept;
    DynamicValue(const char* value);
    DynamicValue(std::string value);
    DynamicValue(List value);
    DynamicValue(Dict value);

    template <typename T>
    bool is() const noexcept
    {
        return std::holds_alternative<T>(data);
    }

    template <typename T>
    T& as()
    {
        return std::get<T>(data);
    }

    template <typename T>
    const T& as() const
    {
        return std::get<T>(data);
    }

    bool operator==(const DynamicValue& other) const
    {
        return data == other.data;
    }

    Storage data{};
};

using PathLinks =
    std::vector<std::pair<Vertex, Vertex>>;
using BfsTreeLevels =
    std::vector<std::vector<Vertex>>;
using DynamicDictList =
    std::vector<DynamicValue::Dict>;

// Python: path_to_links(path)
// worker_count=0 selects the measured-fast sequential implementation. An
// explicit count remains available for unusually large caller-owned paths.
PathLinks path_to_links(
    const std::vector<Vertex>& path,
    std::size_t worker_count = 0);

// Python: get_bfs_tree_level(network, source)
BfsTreeLevels get_bfs_tree_level(
    const Graph& network,
    Vertex source);

BfsTreeLevels get_bfs_tree_level(
    const DiGraph& network,
    Vertex source);

// Deterministic C++ extension: independent sources are evaluated in parallel,
// while the returned element at index i always belongs to sources[i]. Zero
// selects the tuned automatic width (currently at most eight workers, bounded
// by process CPU affinity where the platform exposes it).
std::vector<BfsTreeLevels> get_bfs_tree_levels(
    const Graph& network,
    const std::vector<Vertex>& sources,
    std::size_t worker_count = 0);

std::vector<BfsTreeLevels> get_bfs_tree_levels(
    const DiGraph& network,
    const std::vector<Vertex>& sources,
    std::size_t worker_count = 0);

// Python: flatten_recurrent_dict(recurrent_dict).  The public entry value must
// be a Dict or List. Null leaves match the Python ValueError behavior.
std::vector<DynamicValue> flatten_recurrent_dict(
    const DynamicValue& recurrent_value);

// Python: flatten_dict_list_for_gml(dicts)
// Zero selects the same tuned automatic width; dict and entry order are stable.
DynamicDictList flatten_dict_list_for_gml(
    const DynamicDictList& dicts,
    std::size_t worker_count = 0);

// Python: sanitize_attr_setting(attrs). Mutates and returns the same object.
DynamicDictList& sanitize_attr_setting(
    DynamicDictList& attrs);

} // namespace virne::utils
