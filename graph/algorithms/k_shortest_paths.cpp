#include "k_shortest_paths.h"

#include "bidirectional_bfs.h"
#include "detail/bidirectional_dijkstra_by_id.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>

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
            std::size_t x =
                std::hash<Vertex>{}(v);
            h ^= x + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

using PathSet = std::unordered_set<
    std::vector<Vertex>,
    PathHash>;

static bool is_simple_path(
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

static bool same_root(
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

static std::vector<Vertex> path_prefix(
    const std::vector<Vertex>& path,
    size_t end_exclusive)
{
    return std::vector<Vertex>(
        path.begin(),
        path.begin() + end_exclusive);
}

template <typename GraphType>
static double edge_cost_by_id(
    const GraphType& g,
    Vertex u,
    Vertex v,
    AttrId weight_attr_id)
{
    const auto e =
        g.edge(u, v);

    const auto& attrs =
        g.edge_attrs(e);

    const AttrValue* value =
        attrs.find(weight_attr_id);

    if (value == nullptr)
    {
        return 1.0;
    }

    return attr_to_double(*value);
}

template <typename GraphType>
static std::vector<double> path_prefix_costs_by_id(
    const GraphType& g,
    const std::vector<Vertex>& path,
    AttrId weight_attr_id,
    bool unweighted)
{
    std::vector<double> prefix;
    prefix.reserve(path.size());

    if (path.empty())
    {
        return prefix;
    }

    prefix.push_back(0.0);

    double running = 0.0;

    for (std::size_t i = 0;
         i + 1 < path.size();
         ++i)
    {
        running += unweighted
            ? 1.0
            : edge_cost_by_id(
                  g,
                  path[i],
                  path[i + 1],
                  weight_attr_id);

        prefix.push_back(running);
    }

    return prefix;
}

EdgeKey blocked_edge_key(
    const Graph&,
    Vertex u,
    Vertex v)
{
    return normalize_edge_key(u, v);
}

EdgeKey blocked_edge_key(
    const DiGraph&,
    Vertex u,
    Vertex v)
{
    return {u, v};
}

template <typename GraphType>
static SearchMask combined_search_mask(
    const GraphType& g,
    const SearchMask* base_mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges)
{
    SearchMask combined(
        g.num_nodes(),
        g.edge_id_capacity(),
        true);

    for (Vertex v = 0;
         v < g.num_nodes();
         ++v)
    {
        if ((base_mask != nullptr &&
             !base_mask->allows_node(v)) ||
            banned_vertices.find(v) !=
                banned_vertices.end())
        {
            combined.set_node(v, false);
        }
    }

    const auto [edge_begin, edge_end] =
        g.edges();

    for (auto it = edge_begin;
         it != edge_end;
         ++it)
    {
        const auto edge = *it;
        const Vertex u = g.source(edge);
        const Vertex v = g.target(edge);
        const uint32_t edge_id =
            g.edge_id(edge);

        bool allowed =
            base_mask == nullptr ||
            base_mask->allows(u, v, edge_id);

        if (allowed && !banned_edges.empty())
        {
            allowed =
                banned_edges.find(
                    blocked_edge_key(g, u, v)) ==
                banned_edges.end();
        }

        if (!allowed)
        {
            combined.set_edge(edge_id, false);
        }
    }

    return combined;
}

template <typename GraphType>
static BidirectionalPathResult shortest_path_with_bans(
    const GraphType& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    AttrId weight_attr_id,
    bool weighted)
{
    if (weighted)
    {
        return graph_detail::bidirectional_dijkstra_by_id(
            g,
            source,
            target,
            mask,
            banned_vertices,
            banned_edges,
            weight_attr_id);
    }

    const SearchMask combined =
        combined_search_mask(
            g,
            mask,
            banned_vertices,
            banned_edges);

    const BidirectionalBFSResult bfs =
        bidirectional_bfs(
            g,
            source,
            target,
            combined);

    BidirectionalPathResult out;
    out.found = bfs.found;
    out.cost = bfs.found
        ? static_cast<double>(bfs.distance)
        : std::numeric_limits<double>::infinity();
    out.path = bfs.path;
    return out;
}

template <typename GraphType>
static std::vector<PathResult>
build_candidates_from_base_path(
    const GraphType& g,
    const SearchMask* mask,
    const std::vector<PathResult>& accepted_paths,
    const std::vector<Vertex>& base_path,
    Vertex target,
    AttrId weight_attr_id,
    bool weighted)
{
    std::vector<PathResult> candidates;

    if (base_path.size() < 2)
    {
        return candidates;
    }

    const auto prefix_costs =
        path_prefix_costs_by_id(
            g,
            base_path,
            weight_attr_id,
            !weighted);

    PathSet local_seen;
    local_seen.reserve(base_path.size());

    for (size_t spur_idx = 0;
         spur_idx + 1 < base_path.size();
         ++spur_idx)
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
            banned_vertices.insert(
                base_path[j]);
        }

        EdgeSet banned_edges;
        banned_edges.reserve(
            accepted_paths.size());

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
                blocked_edge_key(
                    g,
                    accepted.path[spur_idx],
                    accepted.path[spur_idx + 1]));
        }

        const BidirectionalPathResult spur_result =
            shortest_path_with_bans(
                g,
                spur_node,
                target,
                mask,
                banned_vertices,
                banned_edges,
                weight_attr_id,
                weighted);

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

        candidates.push_back(
            std::move(candidate));
    }

    return candidates;
}

template <typename GraphType>
std::vector<PathResult> generate_candidates_impl(
    const GraphType& g,
    const PathResult& shortest,
    Vertex target,
    const std::string& weight_attr)
{
    std::vector<PathResult> accepted;
    accepted.push_back(shortest);

    const bool weighted =
        !weight_attr.empty();
    const AttrId weight_attr_id =
        !weighted
            ? AttrId{0}
            : g.attr_id(weight_attr);

    return build_candidates_from_base_path(
        g,
        nullptr,
        accepted,
        shortest.path,
        target,
        weight_attr_id,
        weighted);
}

template <typename GraphType>
std::vector<PathResult> yen_k_shortest_paths_impl(
    const GraphType& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    size_t k,
    const std::string& weight_attr)
{
    std::vector<PathResult> accepted;

    if (k == 0)
    {
        return accepted;
    }

    const bool weighted =
        !weight_attr.empty();
    const AttrId weight_attr_id =
        !weighted
            ? AttrId{0}
            : g.attr_id(weight_attr);

    const BidirectionalPathResult first_result =
        shortest_path_with_bans(
            g,
            source,
            target,
            mask,
            VertexSet{},
            EdgeSet{},
            weight_attr_id,
            weighted);

    if (!first_result.found)
    {
        return accepted;
    }

    PathResult first;
    first.path = std::move(first_result.path);
    first.cost = first_result.cost;

    accepted.push_back(std::move(first));

    std::priority_queue<
        PathResult,
        std::vector<PathResult>,
        CandidateCompare>
        candidates_queue;

    PathSet seen;
    seen.insert(accepted.front().path);
    uint64_t next_insertion_order = 0;

    while (accepted.size() < k)
    {
        const auto& previous =
            accepted.back();

        auto candidates =
            build_candidates_from_base_path(
                g,
                mask,
                accepted,
                previous.path,
                target,
                weight_attr_id,
                weighted);

        for (auto& candidate : candidates)
        {
            if (!seen.insert(candidate.path).second)
            {
                continue;
            }

            candidate.insertion_order =
                next_insertion_order++;

            candidates_queue.push(
                std::move(candidate));
        }

        if (candidates_queue.empty())
        {
            break;
        }

        accepted.push_back(
            candidates_queue.top());
        candidates_queue.pop();
    }

    return accepted;
}

} // namespace

struct ShortestSimplePathGenerator::Impl
{
    std::variant<
        const Graph*,
        const DiGraph*>
        graph;

    Vertex source;
    Vertex target;
    ShortestSimplePathOptions options;
    AttrId weight_attr_id;
    std::optional<SearchMask> mask;

    std::vector<PathResult> accepted;

    std::priority_queue<
        PathResult,
        std::vector<PathResult>,
        CandidateCompare>
        candidates;

    PathSet seen;
    size_t yielded_count = 0;
    uint64_t next_insertion_order = 0;
    bool exhausted = false;

    Impl(
        const Graph* graph_value,
        Vertex source_value,
        Vertex target_value,
        ShortestSimplePathOptions options_value,
        std::optional<SearchMask> mask_value)
        :
        graph(graph_value),
        source(source_value),
        target(target_value),
        options(std::move(options_value)),
        weight_attr_id(
            options.weight_attr.empty()
                ? AttrId{0}
                : graph_value->attr_id(
                      options.weight_attr)),
        mask(std::move(mask_value))
    {
    }

    Impl(
        const DiGraph* graph_value,
        Vertex source_value,
        Vertex target_value,
        ShortestSimplePathOptions options_value,
        std::optional<SearchMask> mask_value)
        :
        graph(graph_value),
        source(source_value),
        target(target_value),
        options(std::move(options_value)),
        weight_attr_id(
            options.weight_attr.empty()
                ? AttrId{0}
                : graph_value->attr_id(
                      options.weight_attr)),
        mask(std::move(mask_value))
    {
    }

    template <typename GraphType>
    std::optional<PathResult> next_for(
        const GraphType& graph_value)
    {
        if (exhausted ||
            (options.max_paths.has_value() &&
             yielded_count >= *options.max_paths))
        {
            return std::nullopt;
        }

        const SearchMask* mask_ptr =
            mask.has_value()
                ? &*mask
                : nullptr;
        const bool weighted =
            !options.weight_attr.empty();

        while (true)
        {
            PathResult next_path;

            if (accepted.empty())
            {
                const BidirectionalPathResult first =
                    shortest_path_with_bans(
                        graph_value,
                        source,
                        target,
                        mask_ptr,
                        VertexSet{},
                        EdgeSet{},
                        weight_attr_id,
                        weighted);

                if (!first.found)
                {
                    exhausted = true;
                    return std::nullopt;
                }

                next_path.path =
                    std::move(first.path);
                next_path.cost = first.cost;

                accepted.push_back(next_path);
                seen.insert(next_path.path);
            }
            else
            {
                auto generated =
                    build_candidates_from_base_path(
                        graph_value,
                        mask_ptr,
                        accepted,
                        accepted.back().path,
                        target,
                        weight_attr_id,
                        weighted);

                for (auto& candidate : generated)
                {
                    if (!seen.insert(
                            candidate.path).second)
                    {
                        continue;
                    }

                    candidate.insertion_order =
                        next_insertion_order++;

                    candidates.push(
                        std::move(candidate));
                }

                if (candidates.empty())
                {
                    exhausted = true;
                    return std::nullopt;
                }

                next_path = candidates.top();
                candidates.pop();
                accepted.push_back(next_path);
            }

            if (next_path.cost > options.max_cost)
            {
                // Yen emits non-decreasing costs, so no later path can pass
                // this bound.
                exhausted = true;
                return std::nullopt;
            }

            const size_t hops =
                next_path.path.empty()
                    ? 0
                    : next_path.path.size() - 1;

            if (options.max_hops.has_value() &&
                hops > *options.max_hops)
            {
                // Hop count is not monotonic under weighted ordering. Keep
                // the path internally for Yen expansion, but do not yield it.
                continue;
            }

            ++yielded_count;
            return next_path;
        }
    }

    std::optional<PathResult> next()
    {
        return std::visit(
            [this](const auto* graph_value)
            {
                return next_for(*graph_value);
            },
            graph);
    }
};

ShortestSimplePathGenerator::ShortestSimplePathGenerator(
    const Graph& graph,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options)
    :
    impl_(std::make_unique<Impl>(
        &graph,
        source,
        target,
        std::move(options),
        std::nullopt))
{
}

ShortestSimplePathGenerator::ShortestSimplePathGenerator(
    const DiGraph& graph,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options)
    :
    impl_(std::make_unique<Impl>(
        &graph,
        source,
        target,
        std::move(options),
        std::nullopt))
{
}

ShortestSimplePathGenerator::ShortestSimplePathGenerator(
    const Graph& graph,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    ShortestSimplePathOptions options)
    :
    impl_(std::make_unique<Impl>(
        &graph,
        source,
        target,
        std::move(options),
        mask))
{
}

ShortestSimplePathGenerator::ShortestSimplePathGenerator(
    const DiGraph& graph,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    ShortestSimplePathOptions options)
    :
    impl_(std::make_unique<Impl>(
        &graph,
        source,
        target,
        std::move(options),
        mask))
{
}

ShortestSimplePathGenerator::~ShortestSimplePathGenerator() = default;

ShortestSimplePathGenerator::ShortestSimplePathGenerator(
    ShortestSimplePathGenerator&&) noexcept = default;

ShortestSimplePathGenerator&
ShortestSimplePathGenerator::operator=(
    ShortestSimplePathGenerator&&) noexcept = default;

std::optional<PathResult>
ShortestSimplePathGenerator::next()
{
    if (!impl_)
    {
        return std::nullopt;
    }

    return impl_->next();
}

size_t ShortestSimplePathGenerator::yielded() const noexcept
{
    return impl_
        ? impl_->yielded_count
        : 0;
}

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
    return generate_candidates_impl(
        g,
        shortest,
        target,
        weight_attr);
}

std::vector<PathResult>
generate_candidates(
    const DiGraph& g,
    const PathResult& shortest,
    Vertex target,
    const std::string& weight_attr)
{
    return generate_candidates_impl(
        g,
        shortest,
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
    return yen_k_shortest_paths_impl(
        g,
        source,
        target,
        nullptr,
        k,
        weight_attr);
}

std::vector<PathResult>
yen_k_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    return yen_k_shortest_paths_impl(
        g,
        source,
        target,
        nullptr,
        k,
        weight_attr);
}

std::vector<PathResult>
yen_k_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    size_t k,
    const std::string& weight_attr)
{
    return yen_k_shortest_paths_impl(
        g,
        source,
        target,
        &mask,
        k,
        weight_attr);
}

std::vector<PathResult>
yen_k_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    const SearchMask& mask,
    size_t k,
    const std::string& weight_attr)
{
    return yen_k_shortest_paths_impl(
        g,
        source,
        target,
        &mask,
        k,
        weight_attr);
}
