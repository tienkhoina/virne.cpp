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
        path.begin() +
            static_cast<
                std::vector<Vertex>::difference_type>(
                    end_exclusive));
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

const RawNeighborList&
yen_reverse_neighbors_fast(
    const Graph& g,
    Vertex v)
{
    return g.neighbors_fast(v);
}

const DiRawInNeighborList&
yen_reverse_neighbors_fast(
    const DiGraph& g,
    Vertex v)
{
    return g.raw().m_vertices[v].m_in_edges;
}

// One enumerator owns one workspace. Generation stamps make every spur reset
// proportional to its root/blocked-edge set instead of to the whole graph.
// The traversal itself uses only dense vertices and stable edge IDs.
class YenSearchWorkspace
{
public:
    YenSearchWorkspace(
        size_t node_count,
        size_t edge_id_capacity)
        :
        seen_forward_(node_count, 0),
        seen_reverse_(node_count, 0),
        banned_nodes_(node_count, 0),
        banned_edges_(edge_id_capacity, 0),
        parent_forward_(node_count),
        parent_reverse_(node_count)
    {
        forward_fringe_.reserve(node_count);
        reverse_fringe_.reserve(node_count);
        next_fringe_.reserve(node_count);
    }

    template <typename GraphType>
    BidirectionalPathResult shortest_path(
        const GraphType& g,
        Vertex source,
        Vertex target,
        const SearchMask* base_mask,
        const VertexSet& banned_vertices,
        const EdgeSet& banned_edges)
    {
        BidirectionalPathResult out;
        const size_t node_count = g.num_nodes();

        if (source >= node_count || target >= node_count)
        {
            return out;
        }

        ensure_capacity(
            node_count,
            g.edge_id_capacity());

        const uint32_t generation = begin_search();

        for (const Vertex vertex : banned_vertices)
        {
            if (vertex < banned_nodes_.size())
            {
                banned_nodes_[vertex] = generation;
            }
        }

        for (const EdgeKey& endpoints : banned_edges)
        {
            const auto edge =
                g.edge(endpoints.first, endpoints.second);
            const uint32_t edge_id = g.edge_id(edge);
            banned_edges_[edge_id] = generation;
        }

        if ((base_mask != nullptr &&
             (!base_mask->allows_node(source) ||
              !base_mask->allows_node(target))) ||
            banned_nodes_[source] == generation ||
            banned_nodes_[target] == generation)
        {
            return out;
        }

        if (source == target)
        {
            out.found = true;
            out.cost = 0.0;
            out.path = {source};
            return out;
        }

        seen_forward_[source] = generation;
        seen_reverse_[target] = generation;
        parent_forward_[source] = source;
        parent_reverse_[target] = target;

        forward_fringe_.clear();
        reverse_fringe_.clear();
        next_fringe_.clear();
        forward_fringe_.push_back(source);
        reverse_fringe_.push_back(target);

        Vertex meet = source;
        bool found = false;

        while (!forward_fringe_.empty() &&
               !reverse_fringe_.empty() &&
               !found)
        {
            if (forward_fringe_.size() <=
                reverse_fringe_.size())
            {
                next_fringe_.clear();

                for (const Vertex u : forward_fringe_)
                {
                    for (const auto& edge :
                         g.neighbors_fast(u))
                    {
                        const Vertex v = edge.get_target();
                        const uint32_t edge_id =
                            edge.get_property().edge_id;

                        if (!allows(
                                base_mask,
                                generation,
                                u,
                                v,
                                edge_id))
                        {
                            continue;
                        }

                        if (seen_forward_[v] != generation)
                        {
                            seen_forward_[v] = generation;
                            parent_forward_[v] = u;
                            next_fringe_.push_back(v);
                        }

                        if (seen_reverse_[v] == generation)
                        {
                            meet = v;
                            found = true;
                            break;
                        }
                    }

                    if (found)
                    {
                        break;
                    }
                }

                forward_fringe_.swap(next_fringe_);
            }
            else
            {
                next_fringe_.clear();

                for (const Vertex u : reverse_fringe_)
                {
                    const auto& in_edges =
                        yen_reverse_neighbors_fast(g, u);

                    for (const auto& edge : in_edges)
                    {
                        const Vertex v = edge.get_target();
                        const uint32_t edge_id =
                            edge.get_property().edge_id;

                        if (!allows(
                                base_mask,
                                generation,
                                v,
                                u,
                                edge_id))
                        {
                            continue;
                        }

                        if (seen_reverse_[v] != generation)
                        {
                            seen_reverse_[v] = generation;
                            parent_reverse_[v] = u;
                            next_fringe_.push_back(v);
                        }

                        if (seen_forward_[v] == generation)
                        {
                            meet = v;
                            found = true;
                            break;
                        }
                    }

                    if (found)
                    {
                        break;
                    }
                }

                reverse_fringe_.swap(next_fringe_);
            }
        }

        if (!found)
        {
            return out;
        }

        for (Vertex vertex = meet;;)
        {
            out.path.push_back(vertex);

            if (vertex == source)
            {
                break;
            }

            vertex = parent_forward_[vertex];
        }

        std::reverse(out.path.begin(), out.path.end());

        for (Vertex vertex = meet;
             vertex != target;)
        {
            vertex = parent_reverse_[vertex];
            out.path.push_back(vertex);
        }

        out.found = true;
        out.cost = static_cast<double>(
            out.path.size() - 1);
        return out;
    }

private:
    void ensure_capacity(
        size_t node_count,
        size_t edge_id_capacity)
    {
        if (seen_forward_.size() < node_count)
        {
            seen_forward_.resize(node_count, 0);
            seen_reverse_.resize(node_count, 0);
            banned_nodes_.resize(node_count, 0);
            parent_forward_.resize(node_count);
            parent_reverse_.resize(node_count);
            forward_fringe_.reserve(node_count);
            reverse_fringe_.reserve(node_count);
            next_fringe_.reserve(node_count);
        }

        if (banned_edges_.size() < edge_id_capacity)
        {
            banned_edges_.resize(
                edge_id_capacity,
                uint32_t{0});
        }
    }

    uint32_t begin_search()
    {
        if (generation_ ==
            std::numeric_limits<uint32_t>::max())
        {
            std::fill(
                seen_forward_.begin(),
                seen_forward_.end(),
                uint32_t{0});
            std::fill(
                seen_reverse_.begin(),
                seen_reverse_.end(),
                uint32_t{0});
            std::fill(
                banned_nodes_.begin(),
                banned_nodes_.end(),
                uint32_t{0});
            std::fill(
                banned_edges_.begin(),
                banned_edges_.end(),
                uint32_t{0});
            generation_ = 1;
        }
        else
        {
            ++generation_;
        }

        return generation_;
    }

    bool allows(
        const SearchMask* base_mask,
        uint32_t generation,
        Vertex u,
        Vertex v,
        uint32_t edge_id) const noexcept
    {
        if (banned_nodes_[u] == generation ||
            banned_nodes_[v] == generation ||
            banned_edges_[edge_id] == generation)
        {
            return false;
        }

        return base_mask == nullptr ||
               base_mask->allows(u, v, edge_id);
    }

    std::vector<uint32_t> seen_forward_;
    std::vector<uint32_t> seen_reverse_;
    std::vector<uint32_t> banned_nodes_;
    std::vector<uint32_t> banned_edges_;
    std::vector<Vertex> parent_forward_;
    std::vector<Vertex> parent_reverse_;
    std::vector<Vertex> forward_fringe_;
    std::vector<Vertex> reverse_fringe_;
    std::vector<Vertex> next_fringe_;
    uint32_t generation_ = 0;
};

template <typename GraphType>
static BidirectionalPathResult shortest_path_with_bans(
    const GraphType& g,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    AttrId weight_attr_id,
    bool weighted,
    YenSearchWorkspace* workspace)
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

    if (workspace == nullptr)
    {
        throw std::logic_error(
            "unweighted Yen search requires a workspace");
    }

    return workspace->shortest_path(
        g,
        source,
        target,
        mask,
        banned_vertices,
        banned_edges);
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
    bool weighted,
    YenSearchWorkspace* workspace)
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
                weighted,
                workspace);

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

    std::optional<YenSearchWorkspace> workspace;
    if (!weighted)
    {
        workspace.emplace(
            g.num_nodes(),
            g.edge_id_capacity());
    }

    return build_candidates_from_base_path(
        g,
        nullptr,
        accepted,
        shortest.path,
        target,
        weight_attr_id,
        weighted,
        workspace.has_value()
            ? &*workspace
            : nullptr);
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

    std::optional<YenSearchWorkspace> workspace;
    if (!weighted)
    {
        workspace.emplace(
            g.num_nodes(),
            g.edge_id_capacity());
    }

    YenSearchWorkspace* workspace_ptr =
        workspace.has_value()
            ? &*workspace
            : nullptr;

    const BidirectionalPathResult first_result =
        shortest_path_with_bans(
            g,
            source,
            target,
            mask,
            VertexSet{},
            EdgeSet{},
            weight_attr_id,
            weighted,
            workspace_ptr);

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
                weighted,
                workspace_ptr);

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
    bool weighted;
    AttrId weight_attr_id;
    std::optional<SearchMask> mask;
    std::optional<YenSearchWorkspace> workspace;

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
        weighted(!options.weight_attr.empty()),
        weight_attr_id(
            weighted
                ? graph_value->attr_id(
                      options.weight_attr)
                : AttrId{0}),
        mask(std::move(mask_value))
    {
        if (!weighted)
        {
            workspace.emplace(
                graph_value->num_nodes(),
                graph_value->edge_id_capacity());
        }
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
        weighted(!options.weight_attr.empty()),
        weight_attr_id(
            weighted
                ? graph_value->attr_id(
                      options.weight_attr)
                : AttrId{0}),
        mask(std::move(mask_value))
    {
        if (!weighted)
        {
            workspace.emplace(
                graph_value->num_nodes(),
                graph_value->edge_id_capacity());
        }
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
        YenSearchWorkspace* workspace_ptr =
            workspace.has_value()
                ? &*workspace
                : nullptr;

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
                        weighted,
                        workspace_ptr);

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
                        weighted,
                        workspace_ptr);

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
