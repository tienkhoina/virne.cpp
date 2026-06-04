// graph/algorithms/k_shortest_paths.cpp
#include "k_shortest_paths.h"

#include "bidirectional_dijkstra.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <queue>

namespace
{

struct PathHash
{
    std::size_t operator()(
        const std::vector<Vertex>& path) const noexcept
    {
        std::size_t h = 0;
        for (Vertex v : path)
        {
            std::size_t x = std::hash<Vertex>{}(v);
            h ^= x + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using PathSet = std::unordered_set<
    std::vector<Vertex>,
    PathHash>;

static bool
is_simple_path(
    const std::vector<Vertex>& path)
{
    std::unordered_set<Vertex> seen;
    seen.reserve(path.size());

    for (auto v : path)
    {
        if (!seen.insert(v).second)
        {
            return false;
        }
    }

    return true;
}

static bool
same_root(
    const std::vector<Vertex>& path,
    const std::vector<Vertex>& root)
{
    if (root.size() > path.size())
    {
        return false;
    }

    for (size_t i = 0; i < root.size(); ++i)
    {
        if (path[i] != root[i])
        {
            return false;
        }
    }

    return true;
}

static std::vector<Vertex>
path_prefix(
    const std::vector<Vertex>& path,
    size_t end_exclusive)
{
    return std::vector<Vertex>(
        path.begin(),
        path.begin() + end_exclusive);
}

static std::vector<PathResult>
build_candidates_from_base_path(
    const Graph& g,
    const std::vector<PathResult>& accepted_paths,
    const std::vector<Vertex>& base_path,
    Vertex target,
    const std::string& weight_attr)
{
    std::vector<PathResult> candidates;

    if (base_path.size() < 2)
    {
        return candidates;
    }

    const auto prefix_costs =
        path_prefix_costs(
            g,
            base_path,
            weight_attr);

    PathSet local_seen;
    local_seen.reserve(base_path.size());

    for (size_t spur_idx = 0; spur_idx + 1 < base_path.size(); ++spur_idx)
    {
        Vertex spur_node = base_path[spur_idx];

        auto root_path =
            path_prefix(
                base_path,
                spur_idx + 1);

        VertexSet banned_vertices;
        banned_vertices.reserve(spur_idx);

        for (size_t j = 0; j < spur_idx; ++j)
        {
            banned_vertices.insert(base_path[j]);
        }

        EdgeSet banned_edges;
        banned_edges.reserve(accepted_paths.size());

        for (const auto& accepted : accepted_paths)
        {
            if (!same_root(accepted.path, root_path))
            {
                continue;
            }

            if (accepted.path.size() <= spur_idx + 1)
            {
                continue;
            }

            banned_edges.insert(
                normalize_edge_key(
                    accepted.path[spur_idx],
                    accepted.path[spur_idx + 1]));
        }

        BidirectionalPathResult spur_result =
            bidirectional_dijkstra(
                g,
                spur_node,
                target,
                banned_vertices,
                banned_edges,
                weight_attr);

        if (!spur_result.found)
        {
            continue;
        }

        auto full_path =
            join_paths(
                root_path,
                spur_result.path);

        if (!is_simple_path(full_path))
        {
            continue;
        }

        if (!local_seen.insert(full_path).second)
        {
            continue;
        }

        PathResult candidate;
        candidate.path = std::move(full_path);
        candidate.cost =
            prefix_costs[spur_idx] +
            spur_result.cost;

        candidates.push_back(std::move(candidate));
    }

    return candidates;
}

} // namespace

std::vector<Vertex>
join_paths(
    const std::vector<Vertex>& root,
    const std::vector<Vertex>& spur)
{
    if (root.empty())
    {
        return spur;
    }

    if (spur.empty())
    {
        return root;
    }

    if (root.back() != spur.front())
    {
        throw std::runtime_error(
            "join_paths: root.back() != spur.front()");
    }

    std::vector<Vertex> result;
    result.reserve(root.size() + spur.size() - 1);

    result.insert(
        result.end(),
        root.begin(),
        root.end());

    result.insert(
        result.end(),
        spur.begin() + 1,
        spur.end());

    return result;
}

std::vector<PathResult>
generate_candidates(
    const Graph& g,
    const PathResult& shortest,
    Vertex target,
    const std::string& weight_attr)
{
    std::vector<PathResult> accepted;
    accepted.push_back(shortest);

    return build_candidates_from_base_path(
        g,
        accepted,
        shortest.path,
        target,
        weight_attr);
}

std::vector<PathResult>
yen_k_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    std::vector<PathResult> A;

    if (k == 0)
    {
        return A;
    }

    BidirectionalPathResult first_result =
        bidirectional_dijkstra(
            g,
            source,
            target,
            VertexSet{},
            EdgeSet{},
            weight_attr);

    if (!first_result.found)
    {
        return A;
    }

    PathResult first;
    first.path = std::move(first_result.path);
    first.cost = first_result.cost;

    A.push_back(std::move(first));

    std::priority_queue<
        PathResult,
        std::vector<PathResult>,
        CandidateCompare
    > B;

    PathSet seen;
    seen.insert(A.front().path);

    while (A.size() < k)
    {
        const auto& prev = A.back();

        auto candidates =
            build_candidates_from_base_path(
                g,
                A,
                prev.path,
                target,
                weight_attr);

        for (auto& candidate : candidates)
        {
            if (!seen.insert(candidate.path).second)
            {
                continue;
            }

            B.push(std::move(candidate));
        }

        if (B.empty())
        {
            break;
        }

        A.push_back(B.top());
        B.pop();
    }

    return A;
}