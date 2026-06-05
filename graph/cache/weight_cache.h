#pragma once

#include "../graph.h"

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
        AttrId attr_id) const;

    double value(
        Vertex u,
        Vertex v,
        AttrId attr_id) const;

private:

    const Graph* graph_ =
        nullptr;

    size_t edge_count_ = 0;

    size_t attr_count_ = 0;

    std::vector<double>
        values_;
};