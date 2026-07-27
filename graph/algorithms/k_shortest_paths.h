// graph/algorithms/k_shortest_paths.h
#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../graph.h"
#include "search_mask.h"

struct PathResult
{
    std::vector<Vertex> path;
    double cost = 0.0;
    uint64_t insertion_order = 0;
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

        return a.insertion_order >
               b.insertion_order;
    }
};

struct ShortestSimplePathOptions
{
    std::optional<size_t> max_paths;
    std::optional<size_t> max_hops;
    double max_cost =
        std::numeric_limits<double>::infinity();
    // Empty is the C++ sentinel for NetworkX weight=None (unit weights).
    std::string weight_attr;
};

// Stateful Yen enumerator. next() performs only enough work to produce the
// following accepted path; callers may stop early without materializing the
// remainder. The object references the graph, which must outlive it.
class ShortestSimplePathGenerator
{
public:
    ShortestSimplePathGenerator(
        const Graph& graph,
        Vertex source,
        Vertex target,
        ShortestSimplePathOptions options = {});

    ShortestSimplePathGenerator(
        const DiGraph& graph,
        Vertex source,
        Vertex target,
        ShortestSimplePathOptions options = {});

    ShortestSimplePathGenerator(
        const Graph& graph,
        Vertex source,
        Vertex target,
        const SearchMask& mask,
        ShortestSimplePathOptions options = {});

    ShortestSimplePathGenerator(
        const DiGraph& graph,
        Vertex source,
        Vertex target,
        const SearchMask& mask,
        ShortestSimplePathOptions options = {});

    ~ShortestSimplePathGenerator();

    ShortestSimplePathGenerator(
        ShortestSimplePathGenerator&&) noexcept;

    ShortestSimplePathGenerator& operator=(
        ShortestSimplePathGenerator&&) noexcept;

    ShortestSimplePathGenerator(
        const ShortestSimplePathGenerator&) = delete;

    ShortestSimplePathGenerator& operator=(
        const ShortestSimplePathGenerator&) = delete;

    std::optional<PathResult> next();

    size_t yielded() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
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
yen_k_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    size_t k,
    const std::string& weight_attr =
        "weight");

std::vector<PathResult>
yen_k_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    size_t k,
    const std::string& weight_attr =
        "weight");

std::vector<PathResult>
yen_k_shortest_paths(
    const DiGraph& g,
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

std::vector<PathResult>
generate_candidates(
    const DiGraph& g,
    const PathResult& shortest,
    Vertex target,
    const std::string& weight_attr =
        "weight");
