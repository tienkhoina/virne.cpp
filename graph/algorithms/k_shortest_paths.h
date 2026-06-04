// graph/algorithms/k_shortest_paths.h
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../graph.h"

struct PathResult
{
    std::vector<Vertex> path;
    double cost = 0.0;
};

struct CandidateCompare
{
    bool operator()(
        const PathResult& a,
        const PathResult& b) const
    {
        if (a.cost != b.cost)
        {
            return a.cost > b.cost;
        }

        return a.path > b.path;
    }
};

std::vector<Vertex>
join_paths(
    const std::vector<Vertex>& root,
    const std::vector<Vertex>& spur);

std::vector<PathResult>
yen_k_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "weight");

std::vector<PathResult>
generate_candidates(
    const Graph& g,
    const PathResult& shortest,
    Vertex target,
    const std::string& weight_attr =
        "weight");