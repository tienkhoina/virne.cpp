#include "topology_analyzer.h"

#include "../../utils/deterministic_executor.h"
#include "algorithms/bfs.h"
#include "nx/shortest_paths.h"

#include <algorithm>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <variant>

namespace virne::core::controller
{
namespace
{

struct PhysicalEdge
{
    ConstraintLink endpoints;
    std::uint32_t id = 0U;
};

template <typename Function>
void parallel_indexed(
    std::size_t count,
    std::size_t requested_workers,
    Function&& function)
{
    if (requested_workers <= 1U || count <= 1U)
    {
        for (std::size_t index = 0U; index < count; ++index)
        {
            function(index);
        }
        return;
    }

    std::vector<std::exception_ptr> failures(count);
    virne::utils::deterministic_parallel_blocks(
        count,
        requested_workers,
        1U,
        [&](std::size_t begin, std::size_t end)
        {
            for (std::size_t index = begin; index < end; ++index)
            {
                try
                {
                    function(index);
                }
                catch (...)
                {
                    failures[index] = std::current_exception();
                    break;
                }
            }
        });

    for (const std::exception_ptr& failure : failures)
    {
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }
}

void validate_method(
    ShortestPathMethod method,
    TopologyAnalyzerOperation operation,
    std::optional<std::size_t> request_index = std::nullopt)
{
    switch (method)
    {
    case ShortestPathMethod::first_shortest:
    case ShortestPathMethod::k_shortest:
    case ShortestPathMethod::k_shortest_length:
    case ShortestPathMethod::all_shortest:
    case ShortestPathMethod::bfs_shortest:
    case ShortestPathMethod::available_shortest:
        return;
    }

    throw TopologyAnalyzerException(
        TopologyAnalyzerErrorCode::invalid_method,
        operation,
        "invalid shortest-path method",
        request_index);
}

std::vector<PhysicalEdge> collect_edges(const Graph& graph)
{
    std::vector<PhysicalEdge> edges;
    edges.reserve(graph.num_edges());
    const auto range = graph.edges();
    for (auto iterator = range.first; iterator != range.second; ++iterator)
    {
        const auto edge = *iterator;
        edges.push_back(
            {{graph.source(edge), graph.target(edge)}, graph.edge_id(edge)});
    }
    return edges;
}

std::vector<Vertex> ordered_unweighted_path(
    const Graph& graph,
    Vertex source,
    Vertex target,
    const SearchMask* mask = nullptr)
{
    if (target >= graph.num_nodes())
    {
        throw std::out_of_range("target vertex is out of range");
    }
    const BFSResult search = mask == nullptr
        ? bfs(graph, source)
        : bfs(graph, source, *mask);
    if (search.distance[target] == std::numeric_limits<std::size_t>::max())
    {
        throw std::runtime_error("no path between source and target");
    }

    std::vector<Vertex> path;
    for (Vertex vertex = target;; vertex = search.predecessor[vertex])
    {
        path.push_back(vertex);
        if (vertex == source)
        {
            break;
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

double numeric_value(
    const AttrValue& value,
    std::size_t item_index,
    ConstraintId resource_id)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return static_cast<double>(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value))
    {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return static_cast<double>(*boolean);
    }
    throw TopologyAnalyzerException(
        TopologyAnalyzerErrorCode::non_numeric_resource,
        TopologyAnalyzerOperation::create_pruned_mask,
        "virtual link resource value is not numeric",
        std::nullopt,
        item_index,
        resource_id);
}

} // namespace

TopologyAnalyzerException::TopologyAnalyzerException(
    TopologyAnalyzerErrorCode code,
    TopologyAnalyzerOperation operation,
    std::string message,
    std::optional<std::size_t> request_index,
    std::optional<std::size_t> item_index,
    std::optional<ConstraintId> resource_id)
    :
    std::runtime_error(std::move(message)),
    code_(code),
    operation_(operation),
    request_index_(request_index),
    item_index_(item_index),
    resource_id_(resource_id)
{
}

TopologyAnalyzerErrorCode TopologyAnalyzerException::code() const noexcept
{
    return code_;
}

TopologyAnalyzerOperation TopologyAnalyzerException::operation() const noexcept
{
    return operation_;
}

const std::optional<std::size_t>&
TopologyAnalyzerException::request_index() const noexcept
{
    return request_index_;
}

const std::optional<std::size_t>&
TopologyAnalyzerException::item_index() const noexcept
{
    return item_index_;
}

const std::optional<ConstraintId>&
TopologyAnalyzerException::resource_id() const noexcept
{
    return resource_id_;
}

TopologyAnalyzer::TopologyAnalyzer(TopologyAnalyzerSelection selection)
    :
    selection_(std::move(selection)),
    checker_(selection_.constraints)
{
}

const TopologyAnalyzerSelection& TopologyAnalyzer::selection() const noexcept
{
    return selection_;
}

PreparedTopologyAnalyzer TopologyAnalyzer::prepare(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network) const
{
    const auto& virtual_registry = virtual_network.link_attributes();
    const auto& physical_registry = physical_network.link_attributes();
    std::vector<std::optional<PreparedTopologyAnalyzer::PreparedLinkResource>>
        resolved(virtual_registry.size());
    std::vector<PreparedTopologyAnalyzer::PreparedLinkResource> ordered;
    ordered.reserve(selection_.link_resources.size());

    for (std::size_t index = 0U;
         index < selection_.link_resources.size();
         ++index)
    {
        const ConstraintId resource_id = selection_.link_resources[index];
        if (resource_id >= virtual_registry.size())
        {
            throw TopologyAnalyzerException(
                TopologyAnalyzerErrorCode::invalid_link_resource_selection,
                TopologyAnalyzerOperation::prepare,
                "link resource selection is out of range",
                std::nullopt,
                index,
                resource_id);
        }

        auto& prepared = resolved[resource_id];
        if (!prepared)
        {
            const auto* resource = dynamic_cast<
                const network::attribute::LinkResourceAttribute*>(
                    &virtual_registry.at(resource_id));
            if (resource == nullptr)
            {
                throw TopologyAnalyzerException(
                    TopologyAnalyzerErrorCode::invalid_link_resource_selection,
                    TopologyAnalyzerOperation::prepare,
                    "selected link attribute is not a resource",
                    std::nullopt,
                    index,
                    resource_id);
            }

            const std::string& name = resource->spec().name;
            const auto virtual_binding =
                virtual_network.bind_link_attribute(name);
            const auto physical_binding =
                physical_network.bind_link_attribute(name);
            if (!virtual_binding)
            {
                throw TopologyAnalyzerException(
                    TopologyAnalyzerErrorCode::invalid_link_resource_selection,
                    TopologyAnalyzerOperation::prepare,
                    "virtual link resource value is not bound",
                    std::nullopt,
                    index,
                    resource_id);
            }
            if (!physical_binding ||
                dynamic_cast<
                    const network::attribute::LinkResourceAttribute*>(
                        &physical_registry.at(
                            physical_binding->registry_id)) == nullptr)
            {
                throw TopologyAnalyzerException(
                    TopologyAnalyzerErrorCode::missing_physical_link_resource,
                    TopologyAnalyzerOperation::prepare,
                    "matching physical link resource is missing",
                    std::nullopt,
                    index,
                    resource_id);
            }

            prepared.emplace(
                PreparedTopologyAnalyzer::PreparedLinkResource{
                    resource_id,
                    resource,
                    virtual_binding->value_id,
                    physical_binding->value_id});
        }
        ordered.push_back(*prepared);
    }

    return PreparedTopologyAnalyzer(
        virtual_network,
        physical_network,
        checker_.prepare(virtual_network, physical_network),
        std::move(ordered));
}

PreparedTopologyAnalyzer::PreparedTopologyAnalyzer(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    PreparedConstraintChecker checker,
    std::vector<PreparedLinkResource> link_resources)
    :
    virtual_network_(&virtual_network),
    physical_network_(&physical_network),
    checker_(std::move(checker)),
    link_resources_(std::move(link_resources))
{
}

PreparedTopologyAnalyzer::Paths
PreparedTopologyAnalyzer::find_shortest_paths(
    const TopologyPathRequest& request) const
{
    validate_method(
        request.options.method,
        TopologyAnalyzerOperation::find_paths);

    Paths paths;
    const Graph& graph = physical_network_->graph();
    const Vertex source = request.physical_pair.source;
    const Vertex target = request.physical_pair.target;
    try
    {
        switch (request.options.method)
        {
        case ShortestPathMethod::first_shortest:
            paths.push_back(ordered_unweighted_path(graph, source, target));
            break;

        case ShortestPathMethod::k_shortest:
            if (request.options.k > 0)
            {
                auto generator = ::nx::shortest_simple_paths(
                    graph,
                    source,
                    target,
                    std::optional<std::string_view>{});
                for (std::int64_t index = 0;
                     index < request.options.k;
                     ++index)
                {
                    auto next = generator.next();
                    if (!next)
                    {
                        break;
                    }
                    paths.push_back(std::move(next->path));
                }
            }
            break;

        case ShortestPathMethod::k_shortest_length:
            if (request.options.k > 0)
            {
                auto generator = ::nx::shortest_simple_paths(
                    graph,
                    source,
                    target,
                    std::optional<std::string_view>{});
                while (auto next = generator.next())
                {
                    if (next->path.size() >
                        static_cast<std::uint64_t>(request.options.k))
                    {
                        break;
                    }
                    paths.push_back(std::move(next->path));
                }
            }
            break;

        case ShortestPathMethod::all_shortest:
            paths = ::nx::all_shortest_paths(
                graph,
                source,
                target,
                std::optional<std::string_view>{});
            break;

        case ShortestPathMethod::bfs_shortest:
        {
            auto path = find_bfs_shortest_path(
                request.virtual_link,
                source,
                target);
            if (path)
            {
                paths.push_back(std::move(*path));
            }
            break;
        }

        case ShortestPathMethod::available_shortest:
        {
            SearchMask mask = create_available_mask(
                request.virtual_link,
                request.options.constraint_workers);
            paths.push_back(ordered_unweighted_path(
                graph,
                source,
                target,
                &mask));
            break;
        }
        }
    }
    catch (const std::exception&)
    {
        paths.clear();
    }

    if (!paths.empty() &&
        static_cast<double>(paths.front().size()) >
            request.options.max_path_nodes)
    {
        paths.clear();
    }
    return paths;
}

std::optional<std::vector<Vertex>>
PreparedTopologyAnalyzer::find_bfs_shortest_path(
    ConstraintLink virtual_link,
    Vertex source,
    Vertex target) const
{
    const Graph& graph = physical_network_->graph();
    const std::size_t node_count = graph.num_nodes();
    if (source >= node_count || target >= node_count)
    {
        throw std::out_of_range("BFS endpoint is out of range");
    }
    if (source == target)
    {
        return std::vector<Vertex>{source};
    }

    const Vertex missing = std::numeric_limits<Vertex>::max();
    std::vector<Vertex> predecessor(node_count, missing);
    std::vector<std::uint8_t> visited(node_count, 0U);
    std::deque<Vertex> queue;
    visited[source] = 1U;
    predecessor[source] = source;
    queue.push_back(source);

    while (!queue.empty())
    {
        const Vertex current = queue.front();
        queue.pop_front();
        const auto neighbors = graph.neighbors(current);
        for (auto iterator = neighbors.first;
             iterator != neighbors.second;
             ++iterator)
        {
            const Vertex neighbor = *iterator;
            const ConstraintCheckResult check =
                checker_.check_link_level_constraints(
                    virtual_link,
                    {current, neighbor});
            if (!check.feasible)
            {
                continue;
            }

            if (neighbor == target)
            {
                predecessor[target] = current;
                std::vector<Vertex> path;
                for (Vertex vertex = target;; vertex = predecessor[vertex])
                {
                    path.push_back(vertex);
                    if (vertex == source)
                    {
                        break;
                    }
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            if (visited[neighbor] == 0U)
            {
                visited[neighbor] = 1U;
                predecessor[neighbor] = current;
                queue.push_back(neighbor);
            }
        }
    }
    return std::nullopt;
}

SearchMask PreparedTopologyAnalyzer::create_available_mask(
    ConstraintLink virtual_link,
    std::size_t workers) const
{
    const Graph& graph = physical_network_->graph();
    const std::vector<PhysicalEdge> edges = collect_edges(graph);
    std::vector<LinkConstraintRequest> requests;
    requests.reserve(edges.size());
    for (const PhysicalEdge& edge : edges)
    {
        requests.push_back({virtual_link, edge.endpoints});
    }

    const std::vector<ConstraintCheckResult> checks =
        checker_.check_link_level_constraints_batch(requests, workers);
    SearchMask mask(graph.num_nodes(), graph.edge_id_capacity(), true);
    for (std::size_t index = 0U; index < edges.size(); ++index)
    {
        mask.set_edge(edges[index].id, checks[index].feasible);
    }
    return mask;
}

::nx::GraphView PreparedTopologyAnalyzer::create_available_network(
    ConstraintLink virtual_link) const
{
    PreparedConstraintChecker checker = checker_;
    return ::nx::subgraph_view_by_id(
        physical_network_->graph(),
        ::nx::NodeFilter{},
        [checker = std::move(checker), virtual_link](
            Vertex source,
            Vertex target,
            std::uint32_t) mutable
        {
            return checker.check_link_level_constraints(
                virtual_link,
                {source, target}).feasible;
        });
}

AttrMap PreparedTopologyAnalyzer::adjusted_virtual_link(
    ConstraintLink virtual_link,
    double ratio,
    double div) const
{
    const Graph& graph = virtual_network_->graph();
    if (!graph.has_edge(virtual_link.source, virtual_link.target))
    {
        throw TopologyAnalyzerException(
            TopologyAnalyzerErrorCode::virtual_link_not_found,
            TopologyAnalyzerOperation::create_pruned_mask,
            "virtual link was not found");
    }

    const auto edge = graph.edge(
        virtual_link.source,
        virtual_link.target);
    AttrMap adjusted = graph.edge_attrs(edge);
    for (std::size_t index = 0U; index < link_resources_.size(); ++index)
    {
        const PreparedLinkResource& resource = link_resources_[index];
        const AttrValue* value = adjusted.find(resource.virtual_value_id);
        if (value == nullptr)
        {
            throw TopologyAnalyzerException(
                TopologyAnalyzerErrorCode::missing_resource_value,
                TopologyAnalyzerOperation::create_pruned_mask,
                "virtual link resource value is missing",
                std::nullopt,
                index,
                resource.resource_id);
        }
        const double scaled =
            numeric_value(*value, index, resource.resource_id) * ratio - div;
        adjusted.set(resource.virtual_value_id, scaled);
    }
    return adjusted;
}

bool PreparedTopologyAnalyzer::pruned_edge_is_available(
    const std::vector<PreparedLinkResource>& resources,
    const AttrMap& adjusted_virtual_link,
    const AttrMap& physical_link)
{
    bool feasible = true;
    for (const PreparedLinkResource& resource : resources)
    {
        const auto result = resource.attribute->check_constraint_satisfiability(
            adjusted_virtual_link,
            resource.virtual_value_id,
            physical_link,
            resource.physical_value_id);
        feasible = result.flag && feasible;
    }
    return feasible;
}

SearchMask PreparedTopologyAnalyzer::create_pruned_mask(
    ConstraintLink virtual_link,
    double ratio,
    double div,
    std::size_t workers) const
{
    const Graph& graph = physical_network_->graph();
    const AttrMap adjusted = adjusted_virtual_link(virtual_link, ratio, div);
    const std::vector<PhysicalEdge> edges = collect_edges(graph);
    std::vector<std::uint8_t> allowed(edges.size(), 0U);
    parallel_indexed(
        edges.size(),
        workers,
        [&](std::size_t index)
        {
            const auto edge = graph.edge_by_id(edges[index].id);
            allowed[index] = static_cast<std::uint8_t>(
                pruned_edge_is_available(
                    link_resources_,
                    adjusted,
                    graph.edge_attrs(edge)));
        });

    SearchMask mask(graph.num_nodes(), graph.edge_id_capacity(), true);
    for (std::size_t index = 0U; index < edges.size(); ++index)
    {
        mask.set_edge(edges[index].id, allowed[index] != 0U);
    }
    return mask;
}

::nx::GraphView PreparedTopologyAnalyzer::create_pruned_network(
    ConstraintLink virtual_link,
    double ratio,
    double div) const
{
    auto adjusted = std::make_shared<const AttrMap>(
        adjusted_virtual_link(virtual_link, ratio, div));
    std::vector<PreparedLinkResource> resources = link_resources_;
    const Graph* graph = &physical_network_->graph();
    return ::nx::subgraph_view_by_id(
        *graph,
        ::nx::NodeFilter{},
        [graph,
         adjusted = std::move(adjusted),
         resources = std::move(resources)](
            Vertex,
            Vertex,
            std::uint32_t edge_id)
        {
            const auto edge = graph->edge_by_id(edge_id);
            return PreparedTopologyAnalyzer::pruned_edge_is_available(
                resources,
                *adjusted,
                graph->edge_attrs(edge));
        });
}

std::vector<PreparedTopologyAnalyzer::Paths>
PreparedTopologyAnalyzer::find_shortest_paths_batch(
    const std::vector<TopologyPathRequest>& requests,
    std::size_t workers) const
{
    for (std::size_t index = 0U; index < requests.size(); ++index)
    {
        validate_method(
            requests[index].options.method,
            TopologyAnalyzerOperation::find_paths_batch,
            index);
    }

    std::vector<Paths> results(requests.size());
    parallel_indexed(
        requests.size(),
        workers,
        [&](std::size_t index)
        {
            results[index] = find_shortest_paths(requests[index]);
        });
    return results;
}

} // namespace virne::core::controller
