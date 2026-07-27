#pragma once

#include "../graph.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class WeightCache
{
public:

    explicit WeightCache(
        const Graph& g);

    AttrId attribute_id(
        const std::string& name) const;

    double value(
        const RawNeighbor& edge,
        AttrId attr_id) const
    {
        return values_[
            static_cast<size_t>(
                edge.get_property().edge_id) *
                attr_count_ +
            attr_id];
    }

    double value(
        Vertex u,
        Vertex v,
        AttrId attr_id) const;

private:

    const Graph* graph_ =
        nullptr;

    size_t edge_capacity_ = 0;

    size_t attr_count_ = 0;

    std::vector<double>
        values_;
};

// Directed counterpart with the same snapshot semantics and O(1) hot-path
// lookup.  It is separate because Boost's raw directed edge record is a
// different implementation type from RawNeighbor.
class DiWeightCache
{
public:
    explicit DiWeightCache(
        const DiGraph& g);

    AttrId attribute_id(
        const std::string& name) const;

    double value(
        const DiRawNeighbor& edge,
        AttrId attr_id) const
    {
        return values_[
            static_cast<size_t>(
                edge.get_property().edge_id) *
                attr_count_ +
            attr_id];
    }

    double value(
        Vertex u,
        Vertex v,
        AttrId attr_id) const;

private:
    const DiGraph* graph_ = nullptr;
    size_t edge_capacity_ = 0;
    size_t attr_count_ = 0;
    std::vector<double> values_;
};
