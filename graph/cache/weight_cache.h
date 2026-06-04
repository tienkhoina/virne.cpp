#pragma once

#include "../graph.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class WeightCache
{
public:

    explicit WeightCache(
        Graph& g);

    std::size_t attribute_id(
        const std::string& name) const;

    double value(
        const RawNeighbor& edge,
        std::size_t attr_id) const;

    double value(
        Vertex u,
        Vertex v,
        std::size_t attr_id) const;

private:

    std::size_t edge_count_ = 0;

    std::unordered_map<
        std::string,
        std::size_t> attr_ids_;

    std::vector<double> values_;
};