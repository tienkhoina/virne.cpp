#include "algorithms/k_shortest_paths.h"
#include "algorithms/floyd_warshall.h"
#include "generators/topology_generators.h"
#include "generators/waxman_generator.h"
#include "graph.h"
#include "nx/shortest_paths.h"
#include "nx/attributes.h"
#include "nx/centrality.h"
#include "nx/sparse.h"
#include "nx/subgraph.h"
#include "py_random.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define CHECK(condition)                                                   \
    do                                                                     \
    {                                                                      \
        if (!(condition))                                                  \
        {                                                                  \
            throw std::runtime_error(                                      \
                std::string("CHECK failed: ") + #condition);              \
        }                                                                  \
    } while (false)

namespace
{

bool close(
    double lhs,
    double rhs,
    double tolerance = 1e-12)
{
    return std::abs(lhs - rhs) <= tolerance;
}

template <typename GraphType>
std::set<std::pair<Vertex, Vertex>> edge_set(
    const GraphType& graph)
{
    std::set<std::pair<Vertex, Vertex>> result;
    auto [it, end] = graph.edges();
    for (; it != end; ++it)
    {
        Vertex u = graph.source(*it);
        Vertex v = graph.target(*it);
        result.emplace(u, v);
    }
    return result;
}

template <typename Map>
std::vector<Vertex> ordered_keys(
    const Map& map)
{
    std::vector<Vertex> keys;
    keys.reserve(map.size());
    for (const auto& [key, value] : map)
    {
        (void)value;
        keys.push_back(key);
    }
    return keys;
}

template <typename Map>
auto ordered_items(const Map& map)
{
    using Value = typename Map::mapped_type;
    std::vector<std::pair<Vertex, Value>> items;
    items.reserve(map.size());
    for (const auto& item : map)
    {
        items.push_back(item);
    }
    return items;
}

template <typename GraphType>
GraphType ordered_result_fixture()
{
    GraphType graph;
    for (size_t i = 0; i < 7; ++i)
    {
        graph.add_node();
    }
    const AttrId weight = graph.attr_id("weight");
    for (const auto [u, v, value] : {
             std::tuple<Vertex, Vertex, double>{0, 4, 2.0},
             {0, 2, 1.0}, {0, 5, 1.0},
             {4, 3, 1.0}, {2, 3, 2.0},
             {5, 6, 1.0}, {3, 1, 1.0},
             {6, 1, 1.0}})
    {
        graph.edge_attrs(graph.add_edge(u, v)).set(
            weight, value);
    }
    return graph;
}

double sparse_value(
    const SparseMatrix& matrix,
    size_t row,
    size_t column)
{
    double result = 0.0;
    for (size_t i = 0;
         i < matrix.nnz();
         ++i)
    {
        if (matrix.row[i] == row &&
            matrix.col[i] == column)
        {
            result += matrix.value[i];
        }
    }
    return result;
}

Graph diamond_graph()
{
    Graph graph;
    for (size_t i = 0; i < 4; ++i)
    {
        graph.add_node();
    }

    const AttrId weight = graph.attr_id("weight");
    for (const auto [u, v] : {
             std::pair<Vertex, Vertex>{0, 1},
             {0, 2},
             {1, 3},
             {2, 3}})
    {
        graph.edge_attrs(
            graph.add_edge(u, v)).set(
                weight,
                1.0);
    }
    return graph;
}

void test_digraph_access_and_conversion()
{
    DiGraph graph;
    for (size_t i = 0; i < 3; ++i)
    {
        graph.add_node();
    }
    const DiEdge edge01 = graph.add_edge(0, 1);
    const DiEdge edge21 = graph.add_edge(2, 1);
    graph.add_edge(1, 2);

    CHECK(graph.in_degree(1) == 2);
    CHECK(graph.out_degree(1) == 1);
    CHECK(graph.degree(1) == 3);

    std::vector<Vertex> predecessors;
    auto [pred, pred_end] = graph.predecessors(1);
    for (; pred != pred_end; ++pred)
    {
        predecessors.push_back(*pred);
    }
    CHECK(predecessors ==
          std::vector<Vertex>({0, 2}));

    std::vector<Vertex> fast_predecessors;
    for (const auto& edge :
         graph.predecessors_fast(1))
    {
        fast_predecessors.push_back(
            edge.get_target());
    }
    CHECK(fast_predecessors == predecessors);
    CHECK(graph.in_edges_fast(1).size() == 2);
    CHECK(graph.out_edges_fast(1).size() == 1);

    size_t in_count = 0;
    auto [in, in_end] = graph.in_edges(1);
    for (; in != in_end; ++in)
    {
        ++in_count;
    }
    CHECK(in_count == 2);

    size_t out_count = 0;
    auto [out, out_end] = graph.out_edges(1);
    for (; out != out_end; ++out)
    {
        ++out_count;
    }
    CHECK(out_count == 1);
    CHECK(graph.edge_id_capacity() == 3);
    CHECK(graph.edge_id(edge01) !=
          graph.edge_id(edge21));

    Graph undirected;
    undirected.add_node();
    undirected.add_node();
    undirected.graph_attrs()["name"] =
        std::string("physical");
    undirected.node_attrs(0)["cpu"] =
        int64_t{10};
    const Edge edge = undirected.add_edge(0, 1);
    undirected.edge_attrs(edge)["bw"] =
        12.5;

    const AttrId cpu_id = undirected.attr_id("cpu");
    DiGraph directed = undirected.to_directed();
    CHECK(directed.num_nodes() == 2);
    CHECK(directed.num_edges() == 2);
    CHECK(directed.has_edge(0, 1));
    CHECK(directed.has_edge(1, 0));
    CHECK(directed.attr_id("cpu") == cpu_id);
    CHECK(std::get<int64_t>(
              directed.node_attrs(0).at("cpu")) == 10);
    CHECK(close(attr_to_double(
              directed.edge_attrs(
                  directed.edge(1, 0)).at("bw")),
          12.5));
    CHECK(std::get<std::string>(
              directed.graph_attrs().at("name")) ==
          "physical");
}

void test_shortest_path_surface()
{
    Graph graph = diamond_graph();

    Graph disconnected;
    disconnected.add_node();
    disconnected.add_node();
    const DistanceMatrix disconnected_distances =
        floyd_warshall(disconnected);
    CHECK(std::isinf(disconnected_distances(0, 1)));

    auto all = nx::all_shortest_paths(
        graph, 0, 3);
    std::set<std::vector<Vertex>> all_set(
        all.begin(), all.end());
    CHECK(all_set ==
          std::set<std::vector<Vertex>>({
              {0, 1, 3},
              {0, 2, 3}}));

    auto weighted = nx::all_shortest_paths(
        graph, 0, 3, "weight");
    CHECK(std::set<std::vector<Vertex>>(
              weighted.begin(), weighted.end()) ==
          all_set);
    const auto explicit_unweighted =
        nx::all_shortest_paths(
            graph, 0, 3, std::nullopt);
    CHECK(std::set<std::vector<Vertex>>(
              explicit_unweighted.begin(),
              explicit_unweighted.end()) == all_set);

    const auto lengths =
        nx::shortest_path_length(graph);
    CHECK(lengths.at(0).at(3) == 2);
    CHECK(lengths.at(3).at(0) == 2);

    const auto paths = nx::shortest_path(graph);
    CHECK(paths.at(0).at(3).front() == 0);
    CHECK(paths.at(0).at(3).back() == 3);
    CHECK(paths.at(0).at(3).size() == 3);

    auto lazy = nx::shortest_simple_paths(
        graph, 0, 3);
    std::vector<PathResult> enumerated;
    while (auto path = lazy.next())
    {
        enumerated.push_back(std::move(*path));
    }
    CHECK(enumerated.size() == 2);
    CHECK(lazy.yielded() == 2);

    ShortestSimplePathOptions one_path;
    one_path.max_paths = 1;
    auto bounded_count = nx::shortest_simple_paths(
        graph, 0, 3, one_path);
    CHECK(bounded_count.next().has_value());
    CHECK(!bounded_count.next().has_value());

    ShortestSimplePathOptions bounded_hops;
    bounded_hops.max_hops = 1;
    auto none = nx::shortest_simple_paths(
        graph, 0, 3, bounded_hops);
    CHECK(!none.next().has_value());

    Graph weighted_conflict;
    for (size_t i = 0; i < 5; ++i)
    {
        weighted_conflict.add_node();
    }
    const AttrId weight =
        weighted_conflict.attr_id("weight");
    auto add_weighted =
        [&](Vertex u, Vertex v, double value)
        {
            weighted_conflict.edge_attrs(
                weighted_conflict.add_edge(u, v)).set(
                    weight, value);
        };
    add_weighted(0, 1, 100.0);
    add_weighted(1, 3, 100.0);
    add_weighted(0, 2, 1.0);
    add_weighted(2, 4, 1.0);
    add_weighted(4, 3, 1.0);

    CHECK(nx::shortest_simple_paths(
              weighted_conflict, 0, 3, 1).front() ==
          std::vector<Vertex>({0, 1, 3}));
    CHECK(nx::shortest_simple_paths(
              weighted_conflict, 0, 3, 1,
              "weight").front() ==
          std::vector<Vertex>({0, 2, 4, 3}));

    auto lazy_unweighted = nx::shortest_simple_paths(
        weighted_conflict, 0, 3);
    CHECK(lazy_unweighted.next()->path ==
          std::vector<Vertex>({0, 1, 3}));

    ShortestSimplePathOptions weighted_options;
    weighted_options.weight_attr = "weight";
    auto lazy_weighted = nx::shortest_simple_paths(
        weighted_conflict, 0, 3,
        weighted_options);
    CHECK(lazy_weighted.next()->path ==
          std::vector<Vertex>({0, 2, 4, 3}));
}

void test_networkx_tie_order_and_defaults()
{
    DiGraph graph;
    for (size_t i = 0; i < 5; ++i)
    {
        graph.add_node();
    }
    const AttrId weight = graph.attr_id("weight");
    for (const auto [u, v] : {
             std::pair<Vertex, Vertex>{0, 2},
             {0, 1},
             {2, 4},
             {1, 4}})
    {
        graph.edge_attrs(
            graph.add_edge(u, v)).set(
                weight, 1.0);
    }

    const std::vector<Vertex> expected{0, 2, 4};
    CHECK(nx::shortest_path(graph, 0, 4) == expected);
    CHECK(nx::dijkstra_path(
              graph, 0, 4, "weight") == expected);

    const nx::DiGraphView view = nx::subgraph_view(
        graph,
        nx::NodeFilter{},
        [](Vertex, Vertex)
        {
            return true;
        });
    CHECK(nx::shortest_path(view, 0, 4) == expected);
    CHECK(nx::dijkstra_path(
              view, 0, 4, "weight") == expected);

    const std::vector<std::vector<Vertex>>
        expected_all{
            {0, 2, 4},
            {0, 1, 4}};
    CHECK(nx::all_shortest_paths(
              graph, 0, 4) == expected_all);
    CHECK(nx::all_shortest_paths(
              graph, 0, 4, "weight") ==
          expected_all);
    CHECK(nx::all_shortest_paths(
              view, 0, 4) == expected_all);
    CHECK(nx::all_shortest_paths(
              view, 0, 4, "weight") ==
          expected_all);

    Graph undirected_ties;
    for (size_t i = 0; i < 5; ++i)
    {
        undirected_ties.add_node();
    }
    const AttrId undirected_weight =
        undirected_ties.attr_id("weight");
    for (const auto [u, v] : {
             std::pair<Vertex, Vertex>{0, 2},
             {0, 1}, {2, 4}, {1, 4}})
    {
        undirected_ties.edge_attrs(
            undirected_ties.add_edge(u, v)).set(
                undirected_weight, 1.0);
    }
    CHECK(nx::all_shortest_paths(
              undirected_ties, 0, 4) ==
          expected_all);
    CHECK(nx::all_shortest_paths(
              undirected_ties, 0, 4,
              "weight") == expected_all);
    const nx::GraphView undirected_view =
        nx::subgraph_view(
            undirected_ties,
            nx::NodeFilter{},
            [](Vertex, Vertex)
            {
                return true;
            });
    CHECK(nx::all_shortest_paths(
              undirected_view, 0, 4) ==
          expected_all);
    CHECK(nx::all_shortest_paths(
              undirected_view, 0, 4,
              "weight") == expected_all);

    Graph centrality_graph;
    for (size_t i = 0; i < 3; ++i)
    {
        centrality_graph.add_node();
    }
    const AttrId centrality_weight =
        centrality_graph.attr_id("weight");
    auto add =
        [&](Vertex u, Vertex v, double value)
        {
            centrality_graph.edge_attrs(
                centrality_graph.add_edge(u, v)).set(
                    centrality_weight, value);
        };
    add(0, 2, 10.0);
    add(0, 1, 1.0);
    add(1, 2, 1.0);

    const auto unweighted =
        nx::betweenness_centrality(
            centrality_graph);
    const auto weighted =
        nx::betweenness_centrality(
            centrality_graph,
            "weight");
    CHECK(close(unweighted[1], 0.0));
    CHECK(close(weighted[1], 1.0));

    const std::vector<std::pair<Vertex, Vertex>>
        reverse_layer_edges{
            {0, 3}, {0, 2}, {0, 1},
            {3, 5}, {3, 4},
            {2, 5}, {2, 4},
            {1, 5}, {1, 4},
            {5, 6}, {4, 6}};

    Graph undirected_paths;
    DiGraph directed_paths;
    for (size_t i = 0; i < 7; ++i)
    {
        undirected_paths.add_node();
        directed_paths.add_node();
    }
    for (const auto [u, v] : reverse_layer_edges)
    {
        undirected_paths.add_edge(u, v);
        directed_paths.add_edge(u, v);
    }

    auto collect = [](auto generator)
    {
        std::vector<std::vector<Vertex>> paths;
        while (auto path = generator.next())
        {
            paths.push_back(std::move(path->path));
        }
        return paths;
    };

    const std::vector<std::vector<Vertex>>
        expected_directed{
            {0, 3, 5, 6},
            {0, 2, 5, 6},
            {0, 3, 4, 6},
            {0, 1, 5, 6},
            {0, 2, 4, 6},
            {0, 1, 4, 6}};

    CHECK(collect(nx::shortest_simple_paths(
              directed_paths, 0, 6)) ==
          expected_directed);

    const std::vector<std::vector<Vertex>>
        expected_undirected{
            {0, 3, 5, 6},
            {0, 2, 5, 6},
            {0, 3, 4, 6},
            {0, 1, 5, 6},
            {0, 2, 4, 6},
            {0, 1, 4, 6},
            {0, 3, 5, 2, 4, 6},
            {0, 2, 5, 3, 4, 6},
            {0, 3, 4, 2, 5, 6},
            {0, 1, 5, 3, 4, 6},
            {0, 2, 4, 3, 5, 6},
            {0, 1, 4, 3, 5, 6},
            {0, 3, 5, 1, 4, 6},
            {0, 2, 5, 1, 4, 6},
            {0, 3, 4, 1, 5, 6},
            {0, 1, 5, 2, 4, 6},
            {0, 2, 4, 1, 5, 6},
            {0, 1, 4, 2, 5, 6}};

    CHECK(collect(nx::shortest_simple_paths(
              undirected_paths, 0, 6)) ==
          expected_undirected);

    const nx::DiGraphView path_view =
        nx::subgraph_view(
            directed_paths,
            nx::NodeFilter{},
            [](Vertex u, Vertex v)
            {
                return !(u == 2 && v == 5);
            });
    const std::vector<std::vector<Vertex>>
        expected_view{
            {0, 3, 5, 6},
            {0, 2, 4, 6},
            {0, 3, 4, 6},
            {0, 1, 5, 6},
            {0, 1, 4, 6}};
    CHECK(collect(nx::shortest_simple_paths(
              path_view, 0, 6)) ==
          expected_view);
}

void test_ordered_result_maps()
{
    nx::OrderedVertexMap<int> map(7);
    CHECK(map.empty());
    CHECK(map.emplace(4, 40).second);
    map[2] = 20;
    CHECK(!map.emplace(4, 99).second);
    CHECK(map.insert_or_assign(4, 41).first->second == 41);
    CHECK(map.size() == 2);
    CHECK(map.contains(4));
    CHECK(map.count(3) == 0);
    CHECK(map.find(2) != map.end());
    CHECK(map.at(4) == 41);
    CHECK(ordered_keys(map) ==
          std::vector<Vertex>({4, 2}));
    CHECK(map.values() ==
          std::vector<int>({41, 20}));
    CHECK(map.items().size() == 2);
    CHECK(map.items().front().first == 4);
    bool maximum_key_rejected = false;
    try
    {
        map.try_emplace(
            std::numeric_limits<Vertex>::max(),
            1);
    }
    catch (const std::overflow_error&)
    {
        maximum_key_rejected = true;
    }
    CHECK(maximum_key_rejected);

    const std::vector<Vertex> bfs_order{
        0, 4, 2, 5, 3, 6, 1};
    const std::vector<Vertex> weighted_order{
        0, 2, 5, 4, 6, 3, 1};
    const std::vector<Vertex> view_bfs_order{
        0, 2, 5, 3, 6, 1};
    const std::vector<Vertex> view_weighted_order{
        0, 2, 5, 6, 3, 1};
    const std::vector<Vertex> view_outer_order{
        0, 1, 2, 3, 5, 6};

    const std::vector<std::pair<Vertex, size_t>>
        expected_bfs{
            {0, 0}, {4, 1}, {2, 1}, {5, 1},
            {3, 2}, {6, 2}, {1, 3}};
    const std::vector<std::pair<Vertex, double>>
        expected_weighted{
            {0, 0.0}, {2, 1.0}, {5, 1.0},
            {4, 2.0}, {6, 2.0}, {3, 3.0},
            {1, 3.0}};

    auto verify_graph =
        [&](const auto& graph)
        {
            const auto lengths =
                nx::shortest_path_length(graph, 0);
            CHECK(ordered_keys(lengths) == bfs_order);
            CHECK(ordered_items(lengths) == expected_bfs);

            const auto single_lengths =
                nx::single_source_shortest_path_length(
                    graph, 0);
            CHECK(ordered_keys(single_lengths) ==
                  bfs_order);
            CHECK(ordered_items(single_lengths) ==
                  expected_bfs);

            const auto paths = nx::shortest_path(graph, 0);
            CHECK(ordered_keys(paths) == bfs_order);
            CHECK(paths.at(3) ==
                  std::vector<Vertex>({0, 4, 3}));

            const auto weighted =
                nx::single_source_dijkstra_path_length(
                    graph, 0, "weight");
            CHECK(ordered_keys(weighted) ==
                  weighted_order);
            CHECK(ordered_items(weighted) ==
                  expected_weighted);

            const auto all_lengths =
                nx::shortest_path_length(graph);
            CHECK(ordered_keys(all_lengths) ==
                  std::vector<Vertex>({
                      0, 1, 2, 3, 4, 5, 6}));
            CHECK(ordered_keys(all_lengths.at(0)) ==
                  bfs_order);

            const auto all_paths =
                nx::shortest_path(graph);
            CHECK(ordered_keys(all_paths) ==
                  std::vector<Vertex>({
                      0, 1, 2, 3, 4, 5, 6}));
            CHECK(ordered_keys(all_paths.at(0)) ==
                  bfs_order);
        };

    const Graph graph =
        ordered_result_fixture<Graph>();
    const DiGraph digraph =
        ordered_result_fixture<DiGraph>();
    verify_graph(graph);
    verify_graph(digraph);

    auto verify_view =
        [&](const auto& view)
        {
            const auto lengths =
                nx::shortest_path_length(view, 0);
            CHECK(ordered_keys(lengths) ==
                  view_bfs_order);

            const auto single_lengths =
                nx::single_source_shortest_path_length(
                    view, 0);
            CHECK(ordered_keys(single_lengths) ==
                  view_bfs_order);

            const auto paths = nx::shortest_path(view, 0);
            CHECK(ordered_keys(paths) ==
                  view_bfs_order);
            CHECK(paths.at(3) ==
                  std::vector<Vertex>({0, 2, 3}));

            const auto weighted =
                nx::single_source_dijkstra_path_length(
                    view, 0, "weight");
            CHECK(ordered_keys(weighted) ==
                  view_weighted_order);

            const auto all_lengths =
                nx::shortest_path_length(view);
            CHECK(ordered_keys(all_lengths) ==
                  view_outer_order);
            CHECK(ordered_keys(all_lengths.at(0)) ==
                  view_bfs_order);

            const auto all_paths =
                nx::shortest_path(view);
            CHECK(ordered_keys(all_paths) ==
                  view_outer_order);
            CHECK(ordered_keys(all_paths.at(0)) ==
                  view_bfs_order);
        };

    const nx::GraphView graph_view =
        nx::subgraph_view(
            graph,
            [](Vertex v)
            {
                return v != 4;
            });
    const nx::DiGraphView digraph_view =
        nx::subgraph_view(
            digraph,
            [](Vertex v)
            {
                return v != 4;
            });
    verify_view(graph_view);
    verify_view(digraph_view);
}

void test_filtered_view()
{
    Graph graph = diamond_graph();
    size_t predicate_calls = 0;
    const nx::GraphView view = nx::subgraph_view(
        graph,
        nx::NodeFilter{},
        [&](Vertex u, Vertex v)
        {
            ++predicate_calls;
            return !((u == 0 && v == 1) ||
                     (u == 1 && v == 0));
        });

    // Predicate-backed views are live: construction is allocation-light and
    // each public algorithm/materialization rebuilds the dense mask from the
    // graph's current attributes/topology.
    CHECK(predicate_calls == 0);
    CHECK(!view.contains_edge(0, 1));
    CHECK(view.contains_edge(0, 2));
    const size_t calls_after_queries =
        predicate_calls;
    CHECK(calls_after_queries >=
          graph.num_edges() * 2);
    CHECK(nx::shortest_path(view, 0, 3) ==
          std::vector<Vertex>({0, 2, 3}));
    CHECK(nx::dijkstra_path(view, 0, 3) ==
          std::vector<Vertex>({0, 2, 3}));
    CHECK(predicate_calls > calls_after_queries);

    const auto all = nx::all_shortest_paths(
        view, 0, 3);
    CHECK(all == std::vector<std::vector<Vertex>>({
                     {0, 2, 3}}));

    const auto lengths =
        nx::shortest_path_length(view);
    CHECK(lengths.at(0).at(3) == 2);

    auto lazy = nx::shortest_simple_paths(
        view, 0, 3);
    const auto path = lazy.next();
    CHECK(path.has_value());
    CHECK(path->path ==
          std::vector<Vertex>({0, 2, 3}));
    CHECK(!lazy.next().has_value());
}

void test_generators()
{
    const Graph path = nx::path_graph(5);
    CHECK(path.num_nodes() == 5);
    CHECK(path.num_edges() == 4);

    const Graph star = nx::star_graph(4);
    CHECK(star.num_nodes() == 5);
    CHECK(star.num_edges() == 4);
    CHECK(star.degree(0) == 4);

    const Graph grid = nx::grid_2d_graph(2, 3);
    CHECK(grid.num_nodes() == 6);
    CHECK(grid.num_edges() == 7);
    CHECK(grid.has_edge(0, 3));
    CHECK(grid.has_edge(1, 0));

    const Graph random =
        nx::erdos_renyi_graph(6, 0.35, 42);
    CHECK((edge_set(random) ==
          std::set<std::pair<Vertex, Vertex>>({
              {0, 2}, {0, 3}, {0, 4}, {1, 4},
              {2, 3}, {2, 4}, {3, 4}, {3, 5}})));

    const DiGraph directed =
        nx::erdos_renyi_digraph(6, 0.35, 42);
    CHECK((edge_set(directed) ==
          std::set<std::pair<Vertex, Vertex>>({
              {0, 2}, {0, 3}, {0, 4}, {1, 3}, {1, 5},
              {2, 0}, {2, 3}, {2, 4}, {3, 1}, {3, 5},
              {4, 2}, {4, 3}, {5, 0}, {5, 1}, {5, 2}})));

    const Graph connected =
        nx::connected_erdos_renyi_graph(
            8, 1.0, 7, 1);
    CHECK(connected.num_edges() == 28);

    WaxmanConfig waxman;
    waxman.num_nodes = 8;
    waxman.seed = 11;
    const Graph waxman_graph =
        WaxmanGenerator::generate(waxman);
    const AttrValue& pos =
        waxman_graph.node_attrs(0).at("pos");
    CHECK(attr_list(pos) != nullptr);
    CHECK(attr_list(pos)->values.size() == 2);

    PyRandom singleton_stream(42);
    PyRandom singleton_reference(42);
    singleton_reference.uniform(0.0, 1.0);
    singleton_reference.uniform(0.0, 1.0);
    bool singleton_waxman_rejected = false;
    try
    {
        static_cast<void>(nx::waxman_graph(
            1, 0.4, 0.1, singleton_stream));
    }
    catch (const std::invalid_argument&)
    {
        singleton_waxman_rejected = true;
    }
    CHECK(singleton_waxman_rejected);
    CHECK(singleton_stream.random() ==
          singleton_reference.random());

    PyRandom zero_alpha_stream(42);
    PyRandom zero_alpha_reference(42);
    for (size_t draw = 0; draw < 5; ++draw)
    {
        zero_alpha_reference.uniform(0.0, 1.0);
    }
    bool zero_alpha_rejected = false;
    try
    {
        static_cast<void>(nx::waxman_graph(
            2, 0.4, 0.0, zero_alpha_stream));
    }
    catch (const std::invalid_argument&)
    {
        zero_alpha_rejected = true;
    }
    CHECK(zero_alpha_rejected);
    CHECK(zero_alpha_stream.random() ==
          zero_alpha_reference.random());
}

void test_sparse_order_and_normalization()
{
    Graph graph;
    for (size_t i = 0; i < 3; ++i)
    {
        graph.add_node();
    }
    const AttrId capacity =
        graph.attr_id("capacity");
    graph.edge_attrs(
        graph.add_edge(0, 1)).set(
            capacity, 2.0);
    graph.edge_attrs(
        graph.add_edge(1, 2)).set(
            capacity, 1.0);

    const SparseMatrix matrix =
        nx::attr_sparse_matrix(
            graph,
            capacity,
            true,
            {2, 0, 1});

    CHECK(close(sparse_value(matrix, 0, 2), 1.0));
    CHECK(close(sparse_value(matrix, 1, 2), 1.0));
    CHECK(close(sparse_value(matrix, 2, 0), 1.0 / 3.0));
    CHECK(close(sparse_value(matrix, 2, 1), 2.0 / 3.0));

    bool rejected = false;
    try
    {
        static_cast<void>(
            nx::attr_sparse_matrix(
                graph,
                capacity,
                false,
                {0, 0, 2}));
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    CHECK(rejected);

    Graph missing_custom;
    const AttrId custom = missing_custom.attr_id("custom");
    missing_custom.edge_attrs(
        missing_custom.add_edge(0, 1)).set(custom, 2.0);
    missing_custom.add_edge(1, 2);
    bool missing_custom_rejected = false;
    try
    {
        static_cast<void>(
            nx::attr_sparse_matrix(missing_custom, custom));
    }
    catch (const std::out_of_range&)
    {
        missing_custom_rejected = true;
    }
    CHECK(missing_custom_rejected);

    Graph missing_weight;
    const AttrId weight = missing_weight.attr_id("weight");
    missing_weight.edge_attrs(
        missing_weight.add_edge(0, 1)).set(weight, 2.0);
    missing_weight.add_edge(1, 2);
    const SparseMatrix default_weight =
        nx::attr_sparse_matrix(missing_weight, weight);
    CHECK(close(sparse_value(default_weight, 0, 1), 2.0));
    CHECK(close(sparse_value(default_weight, 1, 2), 1.0));

    DiGraph zero_sum;
    const AttrId signed_weight = zero_sum.attr_id("capacity");
    zero_sum.edge_attrs(
        zero_sum.add_edge(0, 1)).set(signed_weight, 1.0);
    zero_sum.edge_attrs(
        zero_sum.add_edge(0, 2)).set(signed_weight, -1.0);
    const SparseMatrix divided_by_zero =
        nx::attr_sparse_matrix(zero_sum, signed_weight, true);
    const double positive = sparse_value(divided_by_zero, 0, 1);
    const double negative = sparse_value(divided_by_zero, 0, 2);
    CHECK(std::isinf(positive) && !std::signbit(positive));
    CHECK(std::isinf(negative) && std::signbit(negative));

    Graph zero_undirected;
    const AttrId zero_undirected_attr =
        zero_undirected.attr_id("capacity");
    zero_undirected.edge_attrs(
        zero_undirected.add_edge(0, 1)).set(
            zero_undirected_attr, 0.0);
    CHECK(nx::attr_sparse_matrix(
              zero_undirected,
              zero_undirected_attr,
              false).nnz() == 0);
    CHECK(nx::attr_sparse_matrix(
              zero_undirected,
              zero_undirected_attr,
              true).nnz() == 0);

    DiGraph zero_directed;
    const AttrId zero_directed_attr =
        zero_directed.attr_id("capacity");
    zero_directed.edge_attrs(
        zero_directed.add_edge(0, 1)).set(
            zero_directed_attr, 0.0);
    CHECK(nx::attr_sparse_matrix(
              zero_directed,
              zero_directed_attr,
              false).nnz() == 0);
    CHECK(nx::attr_sparse_matrix(
              zero_directed,
              zero_directed_attr,
              true).nnz() == 0);

    // SciPy LIL sums a row in sorted-column order.  In insertion order the
    // following values sum to 1, while column order rounds to exactly zero.
    DiGraph ordered_sum;
    const AttrId ordered_attr = ordered_sum.attr_id("capacity");
    ordered_sum.edge_attrs(
        ordered_sum.add_edge(0, 1)).set(ordered_attr, 1e16);
    ordered_sum.edge_attrs(
        ordered_sum.add_edge(0, 3)).set(ordered_attr, -1e16);
    ordered_sum.edge_attrs(
        ordered_sum.add_edge(0, 2)).set(ordered_attr, 1.0);
    const SparseMatrix ordered_normalized =
        nx::attr_sparse_matrix(ordered_sum, ordered_attr, true);
    CHECK(ordered_normalized.row ==
          std::vector<size_t>({0, 0, 0}));
    CHECK(ordered_normalized.col ==
          std::vector<size_t>({1, 2, 3}));
    CHECK(std::isinf(ordered_normalized.value[0]) &&
          !std::signbit(ordered_normalized.value[0]));
    CHECK(std::isinf(ordered_normalized.value[1]) &&
          !std::signbit(ordered_normalized.value[1]));
    CHECK(std::isinf(ordered_normalized.value[2]) &&
          std::signbit(ordered_normalized.value[2]));

    DiGraph reciprocal_rounding;
    const AttrId reciprocal_attr =
        reciprocal_rounding.attr_id("capacity");
    reciprocal_rounding.edge_attrs(
        reciprocal_rounding.add_edge(0, 1)).set(
            reciprocal_attr, 3.0);
    reciprocal_rounding.edge_attrs(
        reciprocal_rounding.add_edge(0, 2)).set(
            reciprocal_attr, 2.0);
    const SparseMatrix reciprocal_normalized =
        nx::attr_sparse_matrix(
            reciprocal_rounding,
            reciprocal_attr,
            true);
    CHECK(reciprocal_normalized.value[0] ==
          3.0 * (1.0 / 5.0));
    CHECK(reciprocal_normalized.value[1] ==
          2.0 * (1.0 / 5.0));

    bool empty_adjacency_rejected = false;
    try
    {
        static_cast<void>(nx::adjacency_matrix(Graph{}));
    }
    catch (const std::invalid_argument&)
    {
        empty_adjacency_rejected = true;
    }
    CHECK(empty_adjacency_rejected);
}

void test_dense_constructor_and_public_facades()
{
    const std::vector<std::vector<double>> matrix{
        {0.0, 2.0, 0.0, 0.0},
        {3.0, 0.0, 4.0, 0.0},
        {0.0, 0.0, 5.0, 6.0},
        {0.0, 7.0, 0.0, 0.0}};

    Graph graph(matrix);
    CHECK(graph.number_of_nodes() == 4);
    CHECK(graph.number_of_edges() == 5);

    std::vector<EdgeEndpoints> graph_edges;
    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        graph_edges.emplace_back(
            graph.source(*edge), graph.target(*edge));
    }
    CHECK(graph_edges == std::vector<EdgeEndpoints>({
        {0, 1}, {1, 2}, {1, 3}, {2, 2}, {2, 3}}));
    CHECK(close(attr_to_double(
                    graph.edge_attrs(graph.edge(0, 1)).at("weight")),
                3.0));
    CHECK(close(attr_to_double(
                    graph.edge_attrs(graph.edge(1, 3)).at("weight")),
                7.0));

    DiGraph digraph(matrix);
    std::vector<EdgeEndpoints> arcs;
    auto [arc, arc_end] = digraph.edges();
    for (; arc != arc_end; ++arc)
    {
        arcs.emplace_back(
            digraph.source(*arc), digraph.target(*arc));
    }
    CHECK(arcs == std::vector<EdgeEndpoints>({
        {0, 1}, {1, 0}, {1, 2}, {2, 2}, {2, 3}, {3, 1}}));

    const auto expect_out_of_range = [](auto&& operation)
    {
        bool rejected = false;
        try
        {
            operation();
        }
        catch (const std::out_of_range&)
        {
            rejected = true;
        }
        CHECK(rejected);
    };

    Graph checked_graph;
    checked_graph.add_node();
    expect_out_of_range([&]
    {
        static_cast<void>(checked_graph.degree(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_graph.neighbors(1));
    });

    DiGraph checked_digraph;
    checked_digraph.add_node();
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.degree(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.in_degree(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.out_degree(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.neighbors(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.successors(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.predecessors(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.out_edges(1));
    });
    expect_out_of_range([&]
    {
        static_cast<void>(checked_digraph.in_edges(1));
    });

    bool nonsquare_rejected = false;
    try
    {
        static_cast<void>(Graph(
            std::vector<std::vector<double>>{{0.0, 1.0}, {1.0}}));
    }
    catch (const std::invalid_argument&)
    {
        nonsquare_rejected = true;
    }
    CHECK(nonsquare_rejected);

    DiGraph sparse_labels;
    sparse_labels.add_edges_from(
        std::vector<EdgeEndpoints>{{5, 6}, {6, 8}});
    CHECK(sparse_labels.number_of_nodes() == 9);
    CHECK(sparse_labels.number_of_edges() == 2);
    CHECK(sparse_labels.has_edge(5, 6));
    CHECK(sparse_labels.has_edge(6, 8));

    AttrObject attrs;
    attrs.entries.emplace_back("capacity", 4.5);
    Graph attributed;
    const Edge attributed_edge = attributed.add_edge(2, 4, attrs);
    CHECK(attributed.number_of_nodes() == 5);
    CHECK(close(attr_to_double(
                    attributed.edge_attrs(attributed_edge).at("capacity")),
                4.5));

    std::unordered_map<
        EdgeEndpoints,
        AttrValue,
        nx::EdgeEndpointHash> values;
    values.emplace(EdgeEndpoints{4, 2}, 9.0);
    values.emplace(EdgeEndpoints{0, 1}, 123.0); // absent: NetworkX ignores it
    values.emplace(EdgeEndpoints{100, 101}, 456.0); // out of range: ignored
    nx::set_edge_attributes(attributed, values, "capacity");
    const auto endpoint_values =
        nx::get_edge_attributes(attributed, "capacity");
    static_assert(std::is_const_v<std::remove_reference_t<
        decltype((endpoint_values.begin()->first))>>);
    CHECK(endpoint_values.size() == 1);
    CHECK(endpoint_values.begin()->first.first == 2);
    CHECK(endpoint_values.begin()->first.second == 4);
    CHECK(close(attr_to_double(
                    endpoint_values.at(EdgeEndpoints{2, 4})),
                9.0));
    const uint32_t attributed_id = attributed.edge_id(attributed_edge);
    CHECK(endpoint_values.contains(attributed_id));
    CHECK(endpoint_values.count(attributed_id) == 1);
    CHECK(endpoint_values.find(attributed_id) != endpoint_values.end());
    CHECK(!attributed.has_edge(100, 101));
    CHECK(!attributed.remove_edge(100, 101));
    bool endpoint_out_of_range = false;
    try
    {
        static_cast<void>(attributed.edge(100, 101));
    }
    catch (const std::out_of_range&)
    {
        endpoint_out_of_range = true;
    }
    CHECK(endpoint_out_of_range);

    nx::set_node_attributes(
        attributed,
        {{Vertex{0}, int64_t{8}},
         {Vertex{100}, int64_t{9}}},
        "score");
    const auto valid_node_values =
        nx::get_node_attributes(attributed, "score");
    CHECK(valid_node_values.size() == 1);
    CHECK(std::get<int64_t>(valid_node_values.at(0)) == 8);

    DiGraph directed_attributes;
    directed_attributes.add_node();
    nx::set_node_attributes(
        directed_attributes,
        {{Vertex{0}, int64_t{4}},
         {Vertex{99}, int64_t{5}}},
        "score");
    const auto directed_node_values =
        nx::get_node_attributes(directed_attributes, "score");
    CHECK(directed_node_values.size() == 1);
    CHECK(std::get<int64_t>(directed_node_values.at(0)) == 4);

    AttrMap copied;
    copied.bind(attributed.node_attrs(0).registry());
    copied["x"] = int64_t{3};
    CHECK(copied.get("x").has_value());
    CHECK(std::get<int64_t>(*copied.get("x")) == 3);
    CHECK(std::get<int64_t>(copied.get("missing", int64_t{7})) == 7);
    CHECK(copied.keys().size() == 1);
    CHECK(copied.items().size() == 1);
    const size_t copied_slots = copied.slots().size();
    const AttrId invalid_attr =
        std::numeric_limits<AttrId>::max();
    expect_out_of_range([&]
    {
        static_cast<void>(copied.at(invalid_attr));
    });
    CHECK(copied.slots().size() == copied_slots);
    expect_out_of_range([&]
    {
        copied.set(invalid_attr, int64_t{1});
    });
    CHECK(copied.slots().size() == copied_slots);

    const auto source_registry =
        std::make_shared<AttributeRegistry>();
    const auto target_registry =
        std::make_shared<AttributeRegistry>();
    AttrMap source_attributes(source_registry);
    AttrMap target_attributes(target_registry);
    source_attributes["source_name"] = int64_t{11};
    target_attributes["target_name"] = int64_t{22};
    target_attributes.update(source_attributes);
    CHECK(std::get<int64_t>(
              target_attributes.at("source_name")) == 11);
    CHECK(std::get<int64_t>(
              target_attributes.at("target_name")) == 22);

    Graph stable_names;
    const AttrId retained_id = stable_names.attr_id("short");
    const std::string_view retained_name =
        stable_names.attr_name(retained_id);
    for (size_t index = 0; index < 4096; ++index)
    {
        stable_names.attr_id(
            "later_" + std::to_string(index));
    }
    CHECK(retained_name == "short");

    Graph paths;
    paths.add_edges_from(
        std::vector<EdgeEndpoints>{{0, 1}, {1, 3}, {0, 2}, {2, 3}});
    paths.edge_attrs(paths.edge(0, 1))["weight"] = 100.0;
    paths.edge_attrs(paths.edge(1, 3))["weight"] = 100.0;
    paths.edge_attrs(paths.edge(0, 2))["weight"] = 1.0;
    paths.edge_attrs(paths.edge(2, 3))["weight"] = 1.0;
    CHECK(nx::dijkstra_path(paths, 0, 3, std::nullopt) ==
          std::vector<Vertex>({0, 1, 3}));
    auto weighted_simple = nx::shortest_simple_paths(
        paths,
        0,
        3,
        std::optional<std::string_view>{"weight"});
    CHECK(weighted_simple.next()->path ==
          std::vector<Vertex>({0, 2, 3}));
    auto unweighted_simple = nx::shortest_simple_paths(
        paths, 0, 3, std::nullopt);
    CHECK(unweighted_simple.next()->path ==
          std::vector<Vertex>({0, 1, 3}));
    const auto hop_cutoff =
        nx::single_source_shortest_path_length(paths, 0, 0.1);
    CHECK(hop_cutoff.size() == 3);
    CHECK(hop_cutoff.at(0) == 0);
    CHECK(hop_cutoff.at(1) == 1);
    CHECK(hop_cutoff.at(2) == 1);
    const auto negative_hop_cutoff =
        nx::single_source_shortest_path_length(paths, 0, -1.0);
    CHECK(negative_hop_cutoff.size() == 1);
    CHECK(negative_hop_cutoff.at(0) == 0);
    const auto cutoff = nx::single_source_dijkstra_path_length(
        paths, 0, 1.5, std::string_view{"weight"});
    CHECK(cutoff.contains(0));
    CHECK(cutoff.contains(2));
    CHECK(!cutoff.contains(3));
    const auto negative_cutoff = nx::single_source_dijkstra_path_length(
        paths, 0, -1.0, std::string_view{"weight"});
    CHECK(negative_cutoff.size() == 1);
    CHECK(close(negative_cutoff.at(0), 0.0));
    const auto negative_unweighted = nx::single_source_dijkstra_path_length(
        paths, 0, -1.0, std::nullopt);
    CHECK(negative_unweighted.size() == 1);
    CHECK(close(negative_unweighted.at(0), 0.0));
}

} // namespace

int main()
{
    test_digraph_access_and_conversion();
    test_shortest_path_surface();
    test_networkx_tie_order_and_defaults();
    test_ordered_result_maps();
    test_filtered_view();
    test_generators();
    test_sparse_order_and_normalization();
    test_dense_constructor_and_public_facades();
    std::cout << "graph_foundation_test: PASS\n";
    return 0;
}
