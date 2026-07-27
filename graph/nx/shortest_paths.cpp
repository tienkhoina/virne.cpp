#include "shortest_paths.h"

#include "../algorithms/bfs.h"
#include "../algorithms/bidirectional_bfs.h"
#include "../algorithms/bidirectional_dijkstra.h"
#include "../algorithms/dijkstra.h"
#include "../algorithms/floyd_warshall.h"


#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include "../algorithms/k_shortest_paths.h"
#include "../algorithms/bfs_nx.h"

namespace
{

template <typename GraphType>
BFSResult run_bfs(
    const GraphType& graph,
    Vertex source,
    const SearchMask* mask)
{
    return mask == nullptr
        ? bfs(graph, source)
        : bfs(graph, source, *mask);
}

template <typename GraphType>
nx::SingleSourcePathLengths
single_source_lengths_impl(
    const GraphType& graph,
    Vertex source,
    const SearchMask* mask,
    std::optional<double> cutoff = std::nullopt)
{
    if (source >= graph.num_nodes())
    {
        throw std::out_of_range(
            "source vertex is out of range");
    }
    if (mask != nullptr && !mask->allows_node(source))
    {
        throw std::out_of_range(
            "source vertex is filtered out");
    }

    if (cutoff)
    {
        nx::SingleSourcePathLengths lengths;
        lengths.reserve(graph.num_nodes());
        lengths.emplace(source, size_t{0});

        std::vector<unsigned char> seen(
            graph.num_nodes(), 0);
        seen[source] = 1;
        std::vector<Vertex> current{source};
        std::vector<Vertex> next;
        next.reserve(graph.num_nodes());
        size_t level = 0;

        // This deliberately mirrors NetworkX 3.4.2's
        // `_single_shortest_path_length`: the comparison happens before the
        // level increment. Integer cutoffs therefore include exactly that
        // depth, while a positive fractional cutoff includes its ceiling.
        while (!current.empty() && *cutoff > static_cast<double>(level))
        {
            ++level;
            next.clear();
            for (const Vertex u : current)
            {
                for (const auto& edge : graph.neighbors_fast(u))
                {
                    const Vertex v = edge.get_target();
                    if (mask != nullptr &&
                        !mask->allows(
                            u, v, edge.get_property().edge_id))
                    {
                        continue;
                    }
                    if (seen[v] != 0)
                    {
                        continue;
                    }
                    seen[v] = 1;
                    next.push_back(v);
                    lengths.emplace(v, level);
                }
            }
            current.swap(next);
        }
        return lengths;
    }

    const BFSResult result =
        run_bfs(graph, source, mask);

    nx::SingleSourcePathLengths lengths;
    lengths.reserve(graph.num_nodes());

    for (const Vertex v : result.discovery_order)
    {
        lengths.emplace(
            v,
            result.distance[v]);
    }

    return lengths;
}

std::vector<Vertex> build_unweighted_path(
    const BFSResult& result,
    Vertex source,
    Vertex target)
{
    if (source >= result.predecessor.size() ||
        target >= result.predecessor.size() ||
        result.distance[target] ==
            std::numeric_limits<size_t>::max())
    {
        return {};
    }

    std::vector<Vertex> path;
    for (Vertex current = target;;)
    {
        path.push_back(current);
        if (current == source)
        {
            break;
        }

        const Vertex parent =
            result.predecessor[current];

        if (parent == Vertex(-1) ||
            parent == current)
        {
            return {};
        }

        current = parent;
    }

    std::reverse(
        path.begin(),
        path.end());

    return path;
}

template <typename GraphType>
nx::SingleSourcePaths
single_source_paths_impl(
    const GraphType& graph,
    Vertex source,
    const SearchMask* mask)
{
    const BFSResult result =
        run_bfs(graph, source, mask);

    nx::SingleSourcePaths paths;
    paths.reserve(graph.num_nodes());

    for (const Vertex v : result.discovery_order)
    {
        auto path =
            build_unweighted_path(
                result,
                source,
                v);

        if (!path.empty())
        {
            paths.emplace(
                v,
                std::move(path));
        }
    }

    return paths;
}

template <typename GraphType>
nx::AllPairsPathLengths
all_pairs_lengths_impl(
    const GraphType& graph,
    const SearchMask* mask)
{
    nx::AllPairsPathLengths result;
    result.reserve(graph.num_nodes());

    for (Vertex source = 0;
         source < graph.num_nodes();
         ++source)
    {
        if (mask != nullptr &&
            !mask->allows_node(source))
        {
            continue;
        }

        result.emplace(
            source,
            single_source_lengths_impl(
                graph,
                source,
                mask));
    }

    return result;
}

template <typename GraphType>
nx::AllPairsPaths
all_pairs_paths_impl(
    const GraphType& graph,
    const SearchMask* mask)
{
    nx::AllPairsPaths result;
    result.reserve(graph.num_nodes());

    for (Vertex source = 0;
         source < graph.num_nodes();
         ++source)
    {
        if (mask != nullptr &&
            !mask->allows_node(source))
        {
            continue;
        }

        result.emplace(
            source,
            single_source_paths_impl(
                graph,
                source,
                mask));
    }

    return result;
}

template <typename GraphType>
std::vector<std::vector<Vertex>>
build_all_shortest_paths(
    const GraphType& graph,
    Vertex source,
    Vertex target,
    const SearchMask* mask,
    const std::string* weight_attr)
{
    const size_t n = graph.num_nodes();
    if (source >= n || target >= n ||
        (mask != nullptr &&
         (!mask->allows_node(source) ||
          !mask->allows_node(target))))
    {
        throw std::runtime_error(
            "No path exists");
    }

    std::vector<std::vector<Vertex>> predecessors(n);

    if (weight_attr == nullptr)
    {
        const size_t unreachable =
            std::numeric_limits<size_t>::max();
        std::vector<size_t> distance(n, unreachable);
        std::vector<Vertex> queue(n);
        size_t head = 0;
        size_t tail = 0;

        distance[source] = 0;
        queue[tail++] = source;

        while (head < tail)
        {
            const Vertex u = queue[head++];
            const size_t next_distance =
                distance[u] + 1;

            for (const auto& edge :
                 graph.neighbors_fast(u))
            {
                const Vertex v = edge.get_target();
                const uint32_t edge_id =
                    edge.get_property().edge_id;

                if (mask != nullptr &&
                    !mask->allows(u, v, edge_id))
                {
                    continue;
                }

                if (distance[v] == unreachable)
                {
                    distance[v] = next_distance;
                    predecessors[v].push_back(u);
                    queue[tail++] = v;
                }
                else if (distance[v] == next_distance)
                {
                    predecessors[v].push_back(u);
                }
            }
        }

        if (distance[target] == unreachable)
        {
            throw std::runtime_error(
                "No path exists");
        }
    }
    else
    {
        struct State
        {
            double distance;
            uint64_t order;
            Vertex vertex;
        };

        struct Greater
        {
            bool operator()(
                const State& lhs,
                const State& rhs) const noexcept
            {
                if (lhs.distance != rhs.distance)
                {
                    return lhs.distance > rhs.distance;
                }
                return lhs.order > rhs.order;
            }
        };

        const AttrId weight_id =
            graph.attr_id(*weight_attr);
        const double unreachable =
            std::numeric_limits<double>::infinity();
        std::vector<double> distance(n, unreachable);
        std::priority_queue<
            State,
            std::vector<State>,
            Greater>
            queue;
        uint64_t push_order = 0;

        distance[source] = 0.0;
        queue.push({0.0, push_order++, source});

        while (!queue.empty())
        {
            const State state = queue.top();
            queue.pop();

            const Vertex u = state.vertex;
            if (state.distance != distance[u])
            {
                continue;
            }

            for (const auto& edge :
                 graph.neighbors_fast(u))
            {
                const Vertex v = edge.get_target();
                const uint32_t edge_id =
                    edge.get_property().edge_id;

                if (mask != nullptr &&
                    !mask->allows(u, v, edge_id))
                {
                    continue;
                }

                const AttrValue* value =
                    edge.get_property().attrs.find(
                        weight_id);
                const double weight = value == nullptr
                    ? 1.0
                    : attr_to_double(*value);

                if (weight < 0.0)
                {
                    throw std::runtime_error(
                        "Negative edge weights are not supported");
                }

                const double candidate =
                    state.distance + weight;

                if (candidate < distance[v])
                {
                    distance[v] = candidate;
                    predecessors[v].clear();
                    predecessors[v].push_back(u);
                    queue.push({
                        candidate,
                        push_order++,
                        v});
                }
                else if (candidate == distance[v])
                {
                    predecessors[v].push_back(u);
                }
            }
        }

        if (distance[target] == unreachable)
        {
            throw std::runtime_error(
                "No path exists");
        }
    }

    std::vector<std::vector<Vertex>> paths;
    std::vector<Vertex> reverse_path{target};
    std::vector<uint8_t> on_path(n, 0);
    on_path[target] = 1;

    std::function<void(Vertex)> visit =
        [&](Vertex current)
        {
            if (current == source)
            {
                paths.emplace_back(
                    reverse_path.rbegin(),
                    reverse_path.rend());
                return;
            }

            for (const Vertex predecessor :
                 predecessors[current])
            {
                if (on_path[predecessor])
                {
                    continue;
                }

                on_path[predecessor] = 1;
                reverse_path.push_back(predecessor);
                visit(predecessor);
                reverse_path.pop_back();
                on_path[predecessor] = 0;
            }
        };

    visit(target);
    return paths;
}

template <typename GraphType>
nx::SingleSourceDijkstraPathLengths
single_source_dijkstra_lengths_impl(
    const GraphType& graph,
    Vertex source,
    const SearchMask* mask,
    std::optional<double> cutoff,
    std::optional<std::string_view> weight)
{
    nx::SingleSourceDijkstraPathLengths distances;
    distances.reserve(graph.num_nodes());

    if (!weight)
    {
        const BFSResult result =
            run_bfs(graph, source, mask);
        for (const Vertex v : result.discovery_order)
        {
            const double distance =
                static_cast<double>(result.distance[v]);
            if (v == source || !cutoff || distance <= *cutoff)
            {
                distances.emplace(v, distance);
            }
        }
        return distances;
    }

    const std::string weight_attr(*weight);
    const DijkstraResult result = mask == nullptr
        ? dijkstra_with_cutoff(
              graph, source, cutoff, weight_attr)
        : dijkstra_with_cutoff(
              graph, source, *mask, cutoff, weight_attr);

    for (const Vertex v : result.settled_order)
    {
        if (v == source || !cutoff || result.distance[v] <= *cutoff)
        {
            distances.emplace(v, result.distance[v]);
        }
    }

    return distances;
}

template <typename Results>
std::vector<std::vector<Vertex>>
take_paths(Results results)
{
    std::vector<std::vector<Vertex>> paths;
    paths.reserve(results.size());
    for (auto& result : results)
    {
        paths.push_back(
            std::move(result.path));
    }
    return paths;
}

} // namespace

namespace nx
{

size_t shortest_path_length(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    auto result =
        bidirectional_bfs(
            g,
            source,
            target);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return result.distance;
}

size_t shortest_path_length(
    const DiGraph& g,
    Vertex source,
    Vertex target)
{
    auto result = bidirectional_bfs(g, source, target);
    if (!result.found)
    {
        throw std::runtime_error("No path exists");
    }
    return result.distance;
}

std::vector<Vertex> shortest_path(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    auto result =
        bidirectional_bfs(
            g,
            source,
            target);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return std::move(
        result.path);
}

std::vector<Vertex> shortest_path(
    const DiGraph& g,
    Vertex source,
    Vertex target)
{
    auto result = bidirectional_bfs(g, source, target);
    if (!result.found)
    {
        throw std::runtime_error("No path exists");
    }
    return std::move(result.path);
}

SingleSourcePathLengths
single_source_shortest_path_length(
    const Graph& g,
    Vertex source,
    std::optional<double> cutoff)
{
    return single_source_lengths_impl(
        g, source, nullptr, cutoff);
}

SingleSourcePathLengths
single_source_shortest_path_length(
    const DiGraph& g,
    Vertex source,
    std::optional<double> cutoff)
{
    return single_source_lengths_impl(
        g, source, nullptr, cutoff);
}

std::vector<Vertex> dijkstra_path(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return shortest_path(g, source, target);
    }
    const std::string weight_attr(*weight);
    auto result =
        bidirectional_dijkstra(
            g,
            source,
            target,
            VertexSet{},
            EdgeSet{},
            weight_attr);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return std::move(
        result.path);
}

std::vector<Vertex> dijkstra_path(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return shortest_path(g, source, target);
    }
    const std::string weight_attr(*weight);
    auto result = bidirectional_dijkstra(
        g, source, target, VertexSet{}, EdgeSet{}, weight_attr);
    if (!result.found)
    {
        throw std::runtime_error("No path exists");
    }
    return std::move(result.path);
}

double dijkstra_path_length(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return static_cast<double>(
            shortest_path_length(g, source, target));
    }
    const std::string weight_attr(*weight);
    auto result =
        bidirectional_dijkstra(
            g,
            source,
            target,
            VertexSet{},
            EdgeSet{},
            weight_attr);

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }

    return result.cost;
}

double dijkstra_path_length(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return static_cast<double>(
            shortest_path_length(g, source, target));
    }
    const std::string weight_attr(*weight);
    auto result = bidirectional_dijkstra(
        g, source, target, VertexSet{}, EdgeSet{}, weight_attr);
    if (!result.found)
    {
        throw std::runtime_error("No path exists");
    }
    return result.cost;
}

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const Graph& g,
    Vertex source,
    std::optional<double> cutoff,
    std::optional<std::string_view> weight)
{
    return single_source_dijkstra_lengths_impl(
        g,
        source,
        nullptr,
        cutoff,
        weight);
}

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const DiGraph& g,
    Vertex source,
    std::optional<double> cutoff,
    std::optional<std::string_view> weight)
{
    return single_source_dijkstra_lengths_impl(
        g,
        source,
        nullptr,
        cutoff,
        weight);
}

DistanceMatrix floyd_warshall(
    const Graph& g,
    const std::string& weight_attr)
{
    return ::floyd_warshall(
        g,
        weight_attr);
}

DistanceMatrix floyd_warshall(
    const DiGraph& g,
    const std::string& weight_attr)
{
    return ::floyd_warshall(g, weight_attr);
}

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    auto results =
        yen_k_shortest_paths(
            g,
            source,
            target,
            k,
            weight_attr);

    std::vector<
        std::vector<Vertex>
    > paths;

    paths.reserve(
        results.size());

    for (auto& r : results)
    {
        paths.push_back(
            std::move(
                r.path));
    }

    return paths;
}

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    auto results = yen_k_shortest_paths(
        g, source, target, k, weight_attr);
    std::vector<std::vector<Vertex>> paths;
    paths.reserve(results.size());
    for (auto& result : results)
    {
        paths.push_back(std::move(result.path));
    }
    return paths;
}

size_t shortest_path_length(
    const GraphView& view,
    Vertex source,
    Vertex target)
{
    const auto result = bidirectional_bfs(
        view.graph(),
        source,
        target,
        view.mask());

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return result.distance;
}

size_t shortest_path_length(
    const DiGraphView& view,
    Vertex source,
    Vertex target)
{
    const auto result = bidirectional_bfs(
        view.graph(),
        source,
        target,
        view.mask());

    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return result.distance;
}

SingleSourcePathLengths shortest_path_length(
    const Graph& g,
    Vertex source)
{
    return single_source_lengths_impl(
        g,
        source,
        nullptr);
}

SingleSourcePathLengths shortest_path_length(
    const DiGraph& g,
    Vertex source)
{
    return single_source_lengths_impl(
        g,
        source,
        nullptr);
}

SingleSourcePathLengths shortest_path_length(
    const GraphView& view,
    Vertex source)
{
    return single_source_lengths_impl(
        view.graph(),
        source,
        &view.mask());
}

SingleSourcePathLengths shortest_path_length(
    const DiGraphView& view,
    Vertex source)
{
    return single_source_lengths_impl(
        view.graph(),
        source,
        &view.mask());
}

AllPairsPathLengths shortest_path_length(
    const Graph& g)
{
    return all_pairs_lengths_impl(
        g,
        nullptr);
}

AllPairsPathLengths shortest_path_length(
    const DiGraph& g)
{
    return all_pairs_lengths_impl(
        g,
        nullptr);
}

AllPairsPathLengths shortest_path_length(
    const GraphView& view)
{
    return all_pairs_lengths_impl(
        view.graph(),
        &view.mask());
}

AllPairsPathLengths shortest_path_length(
    const DiGraphView& view)
{
    return all_pairs_lengths_impl(
        view.graph(),
        &view.mask());
}

std::vector<Vertex> shortest_path(
    const GraphView& view,
    Vertex source,
    Vertex target)
{
    auto result = bidirectional_bfs(
        view.graph(),
        source,
        target,
        view.mask());
    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return std::move(result.path);
}

std::vector<Vertex> shortest_path(
    const DiGraphView& view,
    Vertex source,
    Vertex target)
{
    auto result = bidirectional_bfs(
        view.graph(),
        source,
        target,
        view.mask());
    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return std::move(result.path);
}

SingleSourcePaths shortest_path(
    const Graph& g,
    Vertex source)
{
    return single_source_paths_impl(
        g,
        source,
        nullptr);
}

SingleSourcePaths shortest_path(
    const DiGraph& g,
    Vertex source)
{
    return single_source_paths_impl(
        g,
        source,
        nullptr);
}

SingleSourcePaths shortest_path(
    const GraphView& view,
    Vertex source)
{
    return single_source_paths_impl(
        view.graph(),
        source,
        &view.mask());
}

SingleSourcePaths shortest_path(
    const DiGraphView& view,
    Vertex source)
{
    return single_source_paths_impl(
        view.graph(),
        source,
        &view.mask());
}

AllPairsPaths shortest_path(
    const Graph& g)
{
    return all_pairs_paths_impl(
        g,
        nullptr);
}

AllPairsPaths shortest_path(
    const DiGraph& g)
{
    return all_pairs_paths_impl(
        g,
        nullptr);
}

AllPairsPaths shortest_path(
    const GraphView& view)
{
    return all_pairs_paths_impl(
        view.graph(),
        &view.mask());
}

AllPairsPaths shortest_path(
    const DiGraphView& view)
{
    return all_pairs_paths_impl(
        view.graph(),
        &view.mask());
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target)
{
    return build_all_shortest_paths(
        g,
        source,
        target,
        nullptr,
        nullptr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target)
{
    return build_all_shortest_paths(
        g,
        source,
        target,
        nullptr,
        nullptr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const GraphView& view,
    Vertex source,
    Vertex target)
{
    return build_all_shortest_paths(
        view.graph(),
        source,
        target,
        &view.mask(),
        nullptr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target)
{
    return build_all_shortest_paths(
        view.graph(),
        source,
        target,
        &view.mask(),
        nullptr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return build_all_shortest_paths(
            g, source, target, nullptr, nullptr);
    }
    const std::string weight_attr(*weight);
    return build_all_shortest_paths(
        g,
        source,
        target,
        nullptr,
        &weight_attr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return build_all_shortest_paths(
            g, source, target, nullptr, nullptr);
    }
    const std::string weight_attr(*weight);
    return build_all_shortest_paths(
        g,
        source,
        target,
        nullptr,
        &weight_attr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return build_all_shortest_paths(
            view.graph(), source, target,
            &view.mask(), nullptr);
    }
    const std::string weight_attr(*weight);
    return build_all_shortest_paths(
        view.graph(),
        source,
        target,
        &view.mask(),
        &weight_attr);
}

std::vector<std::vector<Vertex>>
all_shortest_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return build_all_shortest_paths(
            view.graph(), source, target,
            &view.mask(), nullptr);
    }
    const std::string weight_attr(*weight);
    return build_all_shortest_paths(
        view.graph(),
        source,
        target,
        &view.mask(),
        &weight_attr);
}

SingleSourcePathLengths
single_source_shortest_path_length(
    const GraphView& view,
    Vertex source,
    std::optional<double> cutoff)
{
    return single_source_lengths_impl(
        view.graph(),
        source,
        &view.mask(),
        cutoff);
}

SingleSourcePathLengths
single_source_shortest_path_length(
    const DiGraphView& view,
    Vertex source,
    std::optional<double> cutoff)
{
    return single_source_lengths_impl(
        view.graph(),
        source,
        &view.mask(),
        cutoff);
}

std::vector<Vertex> dijkstra_path(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return shortest_path(view, source, target);
    }
    const std::string weight_attr(*weight);
    auto result = bidirectional_dijkstra(
        view.graph(), source, target,
        view.mask(), VertexSet{}, EdgeSet{},
        weight_attr);
    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return std::move(result.path);
}

std::vector<Vertex> dijkstra_path(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return shortest_path(view, source, target);
    }
    const std::string weight_attr(*weight);
    auto result = bidirectional_dijkstra(
        view.graph(), source, target,
        view.mask(), VertexSet{}, EdgeSet{},
        weight_attr);
    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return std::move(result.path);
}

double dijkstra_path_length(
    const GraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return static_cast<double>(
            shortest_path_length(view, source, target));
    }
    const std::string weight_attr(*weight);
    const auto result = bidirectional_dijkstra(
        view.graph(), source, target,
        view.mask(), VertexSet{}, EdgeSet{},
        weight_attr);
    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return result.cost;
}

double dijkstra_path_length(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    std::optional<std::string_view> weight)
{
    if (!weight)
    {
        return static_cast<double>(
            shortest_path_length(view, source, target));
    }
    const std::string weight_attr(*weight);
    const auto result = bidirectional_dijkstra(
        view.graph(), source, target,
        view.mask(), VertexSet{}, EdgeSet{},
        weight_attr);
    if (!result.found)
    {
        throw std::runtime_error(
            "No path exists");
    }
    return result.cost;
}

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const GraphView& view,
    Vertex source,
    std::optional<double> cutoff,
    std::optional<std::string_view> weight)
{
    return single_source_dijkstra_lengths_impl(
        view.graph(), source,
        &view.mask(), cutoff, weight);
}

SingleSourceDijkstraPathLengths
single_source_dijkstra_path_length(
    const DiGraphView& view,
    Vertex source,
    std::optional<double> cutoff,
    std::optional<std::string_view> weight)
{
    return single_source_dijkstra_lengths_impl(
        view.graph(), source,
        &view.mask(), cutoff, weight);
}

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    return take_paths(
        yen_k_shortest_paths(
            view.graph(), source, target,
            view.mask(), k, weight_attr));
}

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    return take_paths(
        yen_k_shortest_paths(
            view.graph(), source, target,
            view.mask(), k, weight_attr));
}

ShortestSimplePathGenerator
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options)
{
    return ShortestSimplePathGenerator(
        g, source, target,
        std::move(options));
}

ShortestSimplePathGenerator
shortest_simple_paths(
    const DiGraph& g,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options)
{
    return ShortestSimplePathGenerator(
        g, source, target,
        std::move(options));
}

ShortestSimplePathGenerator
shortest_simple_paths(
    const GraphView& view,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options)
{
    return ShortestSimplePathGenerator(
        view.graph(), source, target,
        view.mask(), std::move(options));
}

ShortestSimplePathGenerator
shortest_simple_paths(
    const DiGraphView& view,
    Vertex source,
    Vertex target,
    ShortestSimplePathOptions options)
{
    return ShortestSimplePathGenerator(
        view.graph(), source, target,
        view.mask(), std::move(options));
}

}
