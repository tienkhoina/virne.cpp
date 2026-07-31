#include "../algorithms/k_shortest_paths.h"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using Paths = std::vector<std::vector<Vertex>>;

void require(
    bool condition,
    const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

Paths take_paths(
    const std::vector<PathResult>& results)
{
    Paths paths;
    paths.reserve(results.size());

    for (const PathResult& result : results)
    {
        paths.push_back(result.path);
    }

    return paths;
}

void test_directed_mask_and_workspace_reset()
{
    const std::vector<std::pair<Vertex, Vertex>> edges{
        {0, 3}, {0, 2}, {0, 1},
        {3, 5}, {3, 4},
        {2, 5}, {2, 4},
        {1, 5}, {1, 4},
        {5, 6}, {4, 6}};

    DiGraph graph;
    for (size_t index = 0; index < 7; ++index)
    {
        graph.add_node();
    }

    for (const auto& [u, v] : edges)
    {
        graph.add_edge(u, v);
    }

    // Retain a removed-edge hole: workspace edge slots must use stable edge
    // ID capacity, not the live edge count.
    const auto removed = graph.add_edge(6, 0);
    const uint32_t removed_id = graph.edge_id(removed);
    require(graph.remove_edge(6, 0), "temporary edge removal failed");
    require(
        removed_id < graph.edge_id_capacity(),
        "removed edge did not retain an ID hole");

    SearchMask mask(
        graph.num_nodes(),
        graph.edge_id_capacity(),
        true);
    mask.set_edge(
        graph.edge_id(graph.edge(2, 5)),
        false);

    const Paths expected{
        {0, 3, 5, 6},
        {0, 2, 4, 6},
        {0, 3, 4, 6},
        {0, 1, 5, 6},
        {0, 1, 4, 6}};

    const auto eager = yen_k_shortest_paths(
        graph, 0, 6, mask, 10, "");
    require(
        take_paths(eager) == expected,
        "directed masked Yen order changed");

    for (const PathResult& result : eager)
    {
        require(
            result.cost == 3.0,
            "unweighted Yen cost changed");
    }

    ShortestSimplePathGenerator lazy(
        graph, 0, 6, mask);
    Paths lazy_paths;
    while (auto result = lazy.next())
    {
        lazy_paths.push_back(std::move(result->path));
    }
    require(
        lazy_paths == expected,
        "lazy workspace state leaked between spur searches");

    require(
        take_paths(yen_k_shortest_paths(
            graph, 0, 6, mask, 10, "")) == expected,
        "workspace state leaked between enumerators");
}

void test_undirected_and_weighted_paths()
{
    Graph graph;
    graph.add_edges_from(
        std::vector<EdgeEndpoints>{
            {0, 2}, {0, 1}, {2, 3}, {1, 3}});

    const Paths expected_unweighted{
        {0, 2, 3},
        {0, 1, 3}};
    require(
        take_paths(yen_k_shortest_paths(
            graph, 0, 3, 4, "")) == expected_unweighted,
        "undirected workspace changed path order");

    const AttrId weight = graph.attr_id("weight");
    graph.edge_attrs(graph.edge(0, 2)).set(weight, 10.0);
    graph.edge_attrs(graph.edge(2, 3)).set(weight, 10.0);
    graph.edge_attrs(graph.edge(0, 1)).set(weight, 1.0);
    graph.edge_attrs(graph.edge(1, 3)).set(weight, 1.0);

    const Paths expected_weighted{
        {0, 1, 3},
        {0, 2, 3}};
    require(
        take_paths(yen_k_shortest_paths(
            graph, 0, 3, 4, "weight")) == expected_weighted,
        "weighted Yen fallback changed path order");
}

} // namespace

int main()
{
    try
    {
        test_directed_mask_and_workspace_reset();
        test_undirected_and_weighted_paths();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "k-shortest workspace tests passed\n";
    return 0;
}
