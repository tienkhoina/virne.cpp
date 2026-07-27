#pragma once

#include "../graph_types.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Dense, edge-id-indexed filtering for traversal hot paths.  An empty side
// means "allow all".  A materialized side is intentionally a byte vector:
// lookups stay branch-light and never hash or invoke a callback in the inner
// neighbor loop.
class SearchMask
{
public:
    SearchMask() = default;

    SearchMask(
        size_t node_count,
        size_t edge_id_capacity,
        bool allowed = true)
        :
        nodes_(
            node_count,
            static_cast<uint8_t>(allowed)),
        edges_(
            edge_id_capacity,
            static_cast<uint8_t>(allowed))
    {
    }

    bool allows_node(
        Vertex v) const noexcept
    {
        return nodes_.empty() ||
               (v < nodes_.size() && nodes_[v] != 0);
    }

    bool allows_edge(
        uint32_t edge_id) const noexcept
    {
        return edges_.empty() ||
               (edge_id < edges_.size() &&
                edges_[edge_id] != 0);
    }

    bool allows(
        Vertex u,
        Vertex v,
        uint32_t edge_id) const noexcept
    {
        return allows_node(u) &&
               allows_node(v) &&
               allows_edge(edge_id);
    }

    void set_node(
        Vertex v,
        bool allowed)
    {
        if (v >= nodes_.size())
        {
            throw std::out_of_range(
                "SearchMask node index is out of range");
        }

        nodes_[v] =
            static_cast<uint8_t>(allowed);
    }

    void set_edge(
        uint32_t edge_id,
        bool allowed)
    {
        if (edge_id >= edges_.size())
        {
            throw std::out_of_range(
                "SearchMask edge id is out of range");
        }

        edges_[edge_id] =
            static_cast<uint8_t>(allowed);
    }

    const std::vector<uint8_t>&
    node_flags() const noexcept
    {
        return nodes_;
    }

    const std::vector<uint8_t>&
    edge_flags() const noexcept
    {
        return edges_;
    }

    bool filters_nodes() const noexcept
    {
        return !nodes_.empty();
    }

    bool filters_edges() const noexcept
    {
        return !edges_.empty();
    }

private:
    std::vector<uint8_t> nodes_;
    std::vector<uint8_t> edges_;
};
