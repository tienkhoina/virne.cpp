#include "graph/algorithms/bfs.h"
#include "graph/algorithms/bfs_nx.h"
#include "graph/algorithms/bfs_stats.h"
#include "graph/algorithms/bidirectional_bfs.h"
#include "graph/algorithms/bidirectional_dijkstra.h"
#include "graph/algorithms/dijkstra.h"
#include "graph/algorithms/floyd_warshall.h"
#include "graph/algorithms/k_shortest_paths.h"
#include "graph/cache/weight_cache.h"
#include "graph/generators/gml_loader.h"
#include "graph/generators/topology_generators.h"
#include "graph/graph.h"
#include "graph/io/graph_saver.h"
#include "graph/nx/attributes.h"
#include "graph/nx/centrality.h"
#include "graph/nx/connectivity.h"
#include "graph/nx/relabel.h"
#include "graph/nx/shortest_paths.h"
#include "graph/nx/sparse.h"
#include "graph/nx/subgraph.h"
#include "graph/nx/views.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

[[noreturn]] void fail(
    const char* expression,
    const char* file,
    int line)
{
    std::ostringstream message;
    message << file << ':' << line
            << ": check failed: " << expression;
    throw std::runtime_error(message.str());
}

#define CHECK(expression)                                      \
    do                                                         \
    {                                                          \
        if (!(expression))                                     \
        {                                                      \
            fail(#expression, __FILE__, __LINE__);             \
        }                                                      \
    } while (false)

bool close(
    double actual,
    double expected,
    double tolerance = 1e-9)
{
    return std::abs(actual - expected) <= tolerance;
}

template <typename T>
bool vector_equal(
    const std::vector<T>& actual,
    std::initializer_list<T> expected)
{
    return actual == std::vector<T>(expected);
}

void check_scores(
    const nx::NodeScores& actual,
    std::initializer_list<double> expected,
    double tolerance)
{
    CHECK(actual.size() == expected.size());
    size_t i = 0;
    for (double value : expected)
    {
        CHECK(close(actual[i], value, tolerance));
        ++i;
    }
}

double sparse_value(
    const SparseMatrix& matrix,
    size_t row,
    size_t col)
{
    double result = 0.0;
    for (size_t i = 0; i < matrix.nnz(); ++i)
    {
        if (matrix.row[i] == row && matrix.col[i] == col)
        {
            result += matrix.value[i];
        }
    }
    return result;
}

bool is_valid_directed_path(
    const DiGraph& graph,
    const std::vector<Vertex>& path,
    Vertex source,
    Vertex target)
{
    if (path.empty() || path.front() != source || path.back() != target)
    {
        return false;
    }
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        if (!graph.has_edge(path[i], path[i + 1]))
        {
            return false;
        }
    }
    return true;
}

struct Fixture
{
    DiGraph graph;
    AttrId weight_id = 0;
    std::vector<uint32_t> edge_ids;

    Fixture()
    {
        for (size_t i = 0; i < 5; ++i)
        {
            CHECK(graph.add_node() == i);
        }

        weight_id = graph.attr_id("weight");
        const struct Arc
        {
            Vertex u;
            Vertex v;
            int64_t weight;
        } arcs[] = {
            {0, 1, 1},
            {0, 2, 4},
            {1, 2, 2},
            {1, 3, 5},
            {2, 3, 1},
            {3, 1, 1},
            {3, 4, 3},
            {4, 0, 7},
        };

        for (const Arc& arc : arcs)
        {
            const DiEdge edge = graph.add_edge(arc.u, arc.v);
            graph.edge_attrs(edge).set(weight_id, arc.weight);
            edge_ids.push_back(graph.edge_id(edge));
        }
    }
};

void test_core_api(Fixture& fixture)
{
    DiGraph& graph = fixture.graph;
    const DiGraph& const_graph = graph;

    CHECK(graph.num_nodes() == 5);
    CHECK(graph.num_edges() == 8);
    CHECK(boost::num_vertices(graph.raw()) == 5);
    CHECK(boost::num_edges(const_graph.raw()) == 8);
    CHECK(graph.attr_name(fixture.weight_id) == "weight");
    CHECK(graph.attribute_registry().find("weight") == fixture.weight_id);

    CHECK(graph.has_edge(0, 1));
    CHECK(!graph.has_edge(1, 0));
    const DiEdge edge01 = graph.edge(0, 1);
    CHECK(graph.source(edge01) == 0);
    CHECK(graph.target(edge01) == 1);
    CHECK(graph.edge_id(edge01) == fixture.edge_ids[0]);
    CHECK(graph.edge_id(graph.edge_by_id(fixture.edge_ids[0])) == fixture.edge_ids[0]);
    CHECK(graph.edge_endpoints(fixture.edge_ids[0]) == std::make_pair(Vertex{0}, Vertex{1}));

    const DiEdge duplicate = graph.add_edge(0, 1);
    CHECK(graph.edge_id(duplicate) == fixture.edge_ids[0]);
    CHECK(graph.num_edges() == 8);

    std::vector<Vertex> neighbors;
    auto [neighbor, neighbor_end] = graph.neighbors(0);
    for (; neighbor != neighbor_end; ++neighbor)
    {
        neighbors.push_back(*neighbor);
    }
    CHECK(vector_equal(neighbors, {Vertex{1}, Vertex{2}}));

    std::vector<Vertex> fast_neighbors;
    for (const DiRawNeighbor& raw_edge : graph.neighbors_fast(0))
    {
        fast_neighbors.push_back(raw_edge.get_target());
    }
    CHECK(fast_neighbors == neighbors);
    CHECK(graph.raw().m_vertices[0].m_in_edges.size() == 1);
    CHECK(graph.raw().m_vertices[0].m_in_edges.front().get_target() == 4);

    size_t node_count = 0;
    auto [node, node_end] = graph.nodes();
    for (; node != node_end; ++node)
    {
        CHECK(*node == node_count);
        ++node_count;
    }
    CHECK(node_count == 5);

    size_t edge_count = 0;
    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        CHECK(graph.source(*edge) < graph.num_nodes());
        CHECK(graph.target(*edge) < graph.num_nodes());
        ++edge_count;
    }
    CHECK(edge_count == 8);

    CHECK(graph.degree(0) == 3);
    CHECK(graph.degree(1) == 4);
    CHECK(graph.degree(2) == 3);
    CHECK(graph.degree(3) == 4);
    CHECK(graph.degree(4) == 2);

    graph.node_attrs(0)["kind"] = std::string("source");
    CHECK(std::get<std::string>(const_graph.node_attrs(0).at("kind")) == "source");
    CHECK(attr_to_double(const_graph.edge_attrs(edge01).at(fixture.weight_id)) == 1.0);

    DiGraph removal;
    removal.add_node();
    removal.add_node();
    const DiEdge forward = removal.add_edge(0, 1);
    const uint32_t removed_id = removal.edge_id(forward);
    const DiEdge reverse = removal.add_edge(1, 0);
    CHECK(removal.edge_id(reverse) != removed_id);
    CHECK(removal.remove_edge(0, 1));
    CHECK(!removal.has_edge(0, 1));
    CHECK(removal.has_edge(1, 0));
    CHECK(!removal.remove_edge(0, 1));
    bool stale_id_rejected = false;
    try
    {
        static_cast<void>(removal.edge_by_id(removed_id));
    }
    catch (const std::runtime_error&)
    {
        stale_id_rejected = true;
    }
    CHECK(stale_id_rejected);
}

void test_attributes_and_sparse(Fixture& fixture)
{
    DiGraph& graph = fixture.graph;
    const AttrId score_id = graph.attr_id("score");
    nx::set_node_attributes(
        graph,
        {{0, int64_t{11}}, {2, int64_t{13}}},
        score_id);
    nx::set_node_attributes(
        graph,
        {{1, std::string("middle")}},
        "label");

    const auto score_by_id = nx::get_node_attributes(graph, score_id);
    const auto score_by_name = nx::get_node_attributes(graph, "score");
    const auto labels = nx::get_node_attributes(graph, "label");
    CHECK(score_by_id == score_by_name);
    CHECK(std::get<int64_t>(score_by_id.at(0)) == 11);
    CHECK(std::get<int64_t>(score_by_id.at(2)) == 13);
    CHECK(std::get<std::string>(labels.at(1)) == "middle");
    const auto ordered_scores = score_by_id.values();
    CHECK(ordered_scores.size() == 2);
    CHECK(std::get<int64_t>(ordered_scores[0]) == 11);
    CHECK(std::get<int64_t>(ordered_scores[1]) == 13);

    const AttrId capacity_id = graph.attr_id("capacity");
    nx::set_edge_attributes(
        graph,
        {{fixture.edge_ids[0], double{10.5}},
         {fixture.edge_ids[1], int64_t{20}}},
        capacity_id);
    nx::set_edge_attributes(
        graph,
        {{fixture.edge_ids[2], double{30.5}}},
        "capacity");

    const auto capacity_by_id = nx::get_edge_attributes(graph, capacity_id);
    const auto capacity_by_name = nx::get_edge_attributes(graph, "capacity");
    CHECK(capacity_by_id == capacity_by_name);
    CHECK(close(attr_to_double(capacity_by_id.at(fixture.edge_ids[0])), 10.5));
    CHECK(close(attr_to_double(capacity_by_id.at(fixture.edge_ids[1])), 20.0));
    CHECK(close(attr_to_double(capacity_by_id.at(fixture.edge_ids[2])), 30.5));
    std::vector<uint32_t> expected_capacity_order;
    auto [capacity_edge, capacity_edge_end] = graph.edges();
    for (; capacity_edge != capacity_edge_end; ++capacity_edge)
    {
        if (graph.edge_attrs(*capacity_edge).contains(capacity_id))
        {
            expected_capacity_order.push_back(
                graph.edge_id(*capacity_edge));
        }
    }
    std::vector<uint32_t> actual_capacity_order;
    for (const auto& [edge_id, value] : capacity_by_id)
    {
        static_cast<void>(value);
        actual_capacity_order.push_back(edge_id);
    }
    CHECK(actual_capacity_order == expected_capacity_order);

    const SparseMatrix adjacency = nx::adjacency_matrix(graph);
    CHECK(adjacency.rows == 5);
    CHECK(adjacency.cols == 5);
    CHECK(adjacency.nnz() == 8);
    CHECK(close(sparse_value(adjacency, 0, 1), 1.0));
    CHECK(close(sparse_value(adjacency, 1, 0), 0.0));
    CHECK(close(sparse_value(adjacency, 4, 0), 7.0));

    bool missing_capacity_rejected = false;
    try
    {
        static_cast<void>(
            nx::attr_sparse_matrix(graph, capacity_id));
    }
    catch (const std::out_of_range&)
    {
        missing_capacity_rejected = true;
    }
    CHECK(missing_capacity_rejected);

    auto [capacity_fill, capacity_fill_end] = graph.edges();
    for (; capacity_fill != capacity_fill_end; ++capacity_fill)
    {
        AttrMap& attrs = graph.edge_attrs(*capacity_fill);
        if (!attrs.contains(capacity_id))
        {
            attrs.set(capacity_id, 1.0);
        }
    }

    const SparseMatrix capacity_by_id_matrix =
        nx::attr_sparse_matrix(graph, capacity_id);
    const SparseMatrix capacity_by_name_matrix =
        nx::attr_sparse_matrix(graph, "capacity");
    CHECK(capacity_by_id_matrix.nnz() == 8);
    CHECK(capacity_by_name_matrix.nnz() == 8);
    CHECK(close(sparse_value(capacity_by_id_matrix, 0, 1), 10.5));
    CHECK(close(sparse_value(capacity_by_name_matrix, 1, 2), 30.5));
    CHECK(close(sparse_value(capacity_by_name_matrix, 1, 0), 0.0));
}

void test_unweighted_paths(const Fixture& fixture)
{
    const DiGraph& graph = fixture.graph;
    const std::vector<size_t> expected = {0, 1, 1, 2, 3};

    const BFSResult result = bfs(graph, 0);
    CHECK(result.distance == expected);
    CHECK(is_valid_directed_path(graph, build_path(
        DijkstraResult{
            std::vector<double>(graph.num_nodes(), 0.0),
            result.predecessor},
        0,
        4), 0, 4));

    const auto map_result = bfs_nx(graph, 0);
    CHECK(map_result.size() == expected.size());
    for (Vertex v = 0; v < expected.size(); ++v)
    {
        CHECK(map_result.at(v) == expected[v]);
    }

    BFSWorkspace workspace(graph.num_nodes());
    const BFSStats stats = bfs_stats(graph, 0, workspace);
    CHECK(stats.reachable == 4);
    CHECK(close(stats.sum_dist, 7.0));
    CHECK(workspace.dist == expected);

    const BidirectionalBFSResult bidirectional =
        bidirectional_bfs(graph, 0, 4);
    CHECK(bidirectional.found);
    CHECK(bidirectional.distance == 3);
    CHECK(is_valid_directed_path(graph, bidirectional.path, 0, 4));

    DiGraph one_way;
    one_way.add_node();
    one_way.add_node();
    one_way.add_edge(1, 0);
    CHECK(!bidirectional_bfs(one_way, 0, 1).found);
    CHECK(bidirectional_bfs(one_way, 1, 0).found);
}

void test_weighted_paths(const Fixture& fixture)
{
    const DiGraph& graph = fixture.graph;
    const std::vector<double> expected = {0.0, 1.0, 3.0, 4.0, 7.0};

    const DijkstraResult result = dijkstra(graph, 0);
    CHECK(result.distance.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        CHECK(close(result.distance[i], expected[i]));
    }
    CHECK(vector_equal(build_path(result, 0, 4),
                       {Vertex{0}, Vertex{1}, Vertex{2}, Vertex{3}, Vertex{4}}));

    const DijkstraResult banned_vertex =
        dijkstra(graph, 0, VertexSet{2}, EdgeSet{});
    CHECK(close(banned_vertex.distance[3], 6.0));
    CHECK(close(banned_vertex.distance[4], 9.0));

    const DijkstraResult banned_arc =
        dijkstra(graph, 0, VertexSet{}, EdgeSet{{1, 2}});
    CHECK(close(banned_arc.distance[3], 5.0));
    CHECK(close(banned_arc.distance[4], 8.0));

    CHECK(close(edge_cost(graph, 1, 2), 2.0));
    const std::vector<Vertex> best = {0, 1, 2, 3, 4};
    CHECK(close(path_cost(graph, best), 7.0));
    const std::vector<double> prefixes = path_prefix_costs(graph, best);
    CHECK(prefixes == std::vector<double>({0.0, 1.0, 3.0, 4.0, 7.0}));

    const BidirectionalPathResult bidirectional =
        bidirectional_dijkstra(graph, 0, 4);
    CHECK(bidirectional.found);
    CHECK(close(bidirectional.cost, 7.0));
    CHECK(is_valid_directed_path(graph, bidirectional.path, 0, 4));
    CHECK(close(path_cost(graph, bidirectional.path), 7.0));

    const BidirectionalPathResult masked_bidirectional =
        bidirectional_dijkstra(graph, 0, 4, VertexSet{2});
    CHECK(masked_bidirectional.found);
    CHECK(close(masked_bidirectional.cost, 9.0));

    const std::vector<PathResult> yen =
        yen_k_shortest_paths(graph, 0, 4, 3);
    CHECK(yen.size() == 3);
    CHECK(vector_equal(yen[0].path, {Vertex{0}, Vertex{1}, Vertex{2}, Vertex{3}, Vertex{4}}));
    CHECK(vector_equal(yen[1].path, {Vertex{0}, Vertex{2}, Vertex{3}, Vertex{4}}));
    CHECK(vector_equal(yen[2].path, {Vertex{0}, Vertex{1}, Vertex{3}, Vertex{4}}));
    CHECK(close(yen[0].cost, 7.0));
    CHECK(close(yen[1].cost, 8.0));
    CHECK(close(yen[2].cost, 9.0));

    const std::vector<PathResult> candidates =
        generate_candidates(graph, yen.front(), 4);
    CHECK(candidates.size() == 2);
    std::vector<double> candidate_costs;
    for (const PathResult& candidate : candidates)
    {
        CHECK(is_valid_directed_path(graph, candidate.path, 0, 4));
        candidate_costs.push_back(candidate.cost);
    }
    std::sort(candidate_costs.begin(), candidate_costs.end());
    CHECK(candidate_costs == std::vector<double>({8.0, 9.0}));

    CHECK(vector_equal(join_paths({Vertex{0}, Vertex{1}},
                                  {Vertex{1}, Vertex{2}, Vertex{3}}),
                       {Vertex{0}, Vertex{1}, Vertex{2}, Vertex{3}}));

    const double rows[5][5] = {
        {0, 1, 3, 4, 7},
        {13, 0, 2, 3, 6},
        {11, 2, 0, 1, 4},
        {10, 1, 3, 0, 3},
        {7, 8, 10, 11, 0},
    };
    const DistanceMatrix matrix = ::floyd_warshall(graph);
    CHECK(matrix.rows() == 5 && matrix.cols() == 5);
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            CHECK(close(matrix(i, j), rows[i][j]));
        }
    }
}

void test_nx_api(const Fixture& fixture)
{
    const DiGraph& graph = fixture.graph;
    CHECK(nx::is_connected(graph));

    DiGraph disconnected;
    disconnected.add_node();
    disconnected.add_node();
    CHECK(!nx::is_connected(disconnected));

    CHECK(nx::shortest_path_length(graph, 0, 4) == 3);
    const auto shortest = nx::shortest_path(graph, 0, 4);
    CHECK(is_valid_directed_path(graph, shortest, 0, 4));
    const auto single_unweighted =
        nx::single_source_shortest_path_length(graph, 0);
    CHECK(single_unweighted.at(4) == 3);

    CHECK(close(nx::dijkstra_path_length(graph, 0, 4), 7.0));
    const auto weighted_path = nx::dijkstra_path(graph, 0, 4);
    CHECK(is_valid_directed_path(graph, weighted_path, 0, 4));
    CHECK(close(path_cost(graph, weighted_path), 7.0));
    const auto single_weighted =
        nx::single_source_dijkstra_path_length(graph, 0);
    CHECK(single_weighted.size() == 5);
    CHECK(close(single_weighted.at(3), 4.0));

    const DistanceMatrix matrix = nx::floyd_warshall(graph);
    CHECK(close(matrix(4, 3), 11.0));

    const auto simple_paths =
        nx::shortest_simple_paths(
            graph, 0, 4, 3, "weight");
    CHECK(simple_paths.size() == 3);
    CHECK(vector_equal(simple_paths[0],
                       {Vertex{0}, Vertex{1}, Vertex{2}, Vertex{3}, Vertex{4}}));
    CHECK(vector_equal(simple_paths[1],
                       {Vertex{0}, Vertex{2}, Vertex{3}, Vertex{4}}));
    CHECK(vector_equal(simple_paths[2],
                       {Vertex{0}, Vertex{1}, Vertex{3}, Vertex{4}}));

    check_scores(nx::degree_centrality(graph),
                 {0.75, 1.0, 0.75, 1.0, 0.5}, 1e-12);
    check_scores(nx::eigenvector_centrality(graph),
                 {0.2266173743694314,
                  0.5067318539714362,
                  0.45323474873886277,
                  0.5932919885386007,
                  0.3666746141691693},
                 2e-5);
    check_scores(nx::closeness_centrality(graph),
                 {0.4444444444444444,
                  0.6666666666666666,
                  0.6666666666666666,
                  0.5714285714285714,
                  0.5},
                 1e-12);
    check_scores(nx::betweenness_centrality(
                     graph, "weight"),
                 {0.25, 0.5, 0.5, 0.5, 0.25},
                 1e-12);
}

void test_weight_cache(const Fixture& fixture)
{
    const DiGraph& graph = fixture.graph;
    DiWeightCache cache(graph);
    const AttrId weight_id = cache.attribute_id("weight");
    CHECK(weight_id == fixture.weight_id);
    CHECK(close(cache.value(0, 1, weight_id), 1.0));
    CHECK(close(cache.value(4, 0, weight_id), 7.0));

    bool found = false;
    for (const DiRawNeighbor& edge : graph.neighbors_fast(0))
    {
        if (edge.get_target() == 2)
        {
            CHECK(close(cache.value(edge, weight_id), 4.0));
            found = true;
        }
    }
    CHECK(found);
}

void test_masked_and_indexed_surface(const Fixture& fixture)
{
    const DiGraph& graph = fixture.graph;
    SearchMask mask(
        graph.num_nodes(),
        graph.edge_id_capacity(),
        true);
    mask.set_node(2, false);

    const BFSResult masked_bfs =
        bfs(graph, 0, mask);
    CHECK(masked_bfs.distance[2] ==
          std::numeric_limits<size_t>::max());
    CHECK(masked_bfs.distance[4] == 3);

    const BidirectionalBFSResult masked_two_way_bfs =
        bidirectional_bfs(graph, 0, 4, mask);
    CHECK(masked_two_way_bfs.found);
    CHECK(masked_two_way_bfs.distance == 3);

    const DijkstraResult masked_dijkstra =
        dijkstra(graph, 0, mask);
    CHECK(close(masked_dijkstra.distance[4], 9.0));

    const DijkstraResult cutoff =
        dijkstra_with_cutoff(graph, 0, mask, 5.0);
    CHECK(cutoff.distance[3] ==
          std::numeric_limits<double>::max());

    const DijkstraResult masked_and_banned =
        dijkstra(
            graph,
            0,
            mask,
            VertexSet{},
            EdgeSet{{1, 3}});
    CHECK(masked_and_banned.distance[4] ==
          std::numeric_limits<double>::max());

    const BidirectionalPathResult masked_two_way_dijkstra =
        bidirectional_dijkstra(graph, 0, 4, mask);
    CHECK(masked_two_way_dijkstra.found);
    CHECK(close(masked_two_way_dijkstra.cost, 9.0));

    const auto masked_yen =
        yen_k_shortest_paths(graph, 0, 4, mask, 3);
    CHECK(masked_yen.size() == 1);
    CHECK(vector_equal(
        masked_yen.front().path,
        {Vertex{0}, Vertex{1}, Vertex{3}, Vertex{4}}));

    const uint32_t rejected_edge =
        graph.edge_id(graph.edge(0, 2));
    const nx::DiGraphView indexed_view =
        nx::subgraph_view_by_id(
            graph,
            nx::NodeFilter{},
            [rejected_edge](Vertex, Vertex, uint32_t edge_id)
            {
                return edge_id != rejected_edge;
            });
    CHECK(indexed_view.contains_edge(0, 1));
    CHECK(!indexed_view.contains_edge(0, 2));

    const DiGraph relabelled =
        nx::convert_node_labels_to_integers(
            graph,
            0,
            "default",
            "old_label");
    const AttrId old_label_id =
        relabelled.attr_id("old_label");
    CHECK(std::get<int64_t>(
              relabelled.node_attrs(4).at(old_label_id)) == 4);
    CHECK(relabelled.has_edge(4, 0));

    const auto [neighbor, neighbor_end] =
        nx::neighbors(graph, 0);
    CHECK(neighbor != neighbor_end && *neighbor == 1);
    CHECK(nx::degree(graph) == graph.degree());
    CHECK(nx::in_degree(graph) == graph.in_degree());
    CHECK(nx::out_degree(graph) == graph.out_degree());

    const DiGraph complete =
        nx::erdos_renyi_digraph(4, 1.0, uint64_t{42});
    CHECK(complete.num_nodes() == 4);
    CHECK(complete.num_edges() == 12);
}

void test_gml_roundtrip(const Fixture& fixture)
{
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("virne-digraph-" + std::to_string(nonce) + ".gml");

    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};

    GraphSaver::save_gml(fixture.graph, path.string());
    DiGraph loaded = GmlLoader::load_directed(path.string());
    CHECK(loaded.num_nodes() == fixture.graph.num_nodes());
    CHECK(loaded.num_edges() == fixture.graph.num_edges());
    CHECK(loaded.has_edge(0, 1));
    CHECK(!loaded.has_edge(1, 0));
    CHECK(loaded.has_edge(4, 0));
    CHECK(close(attr_to_double(loaded.edge_attrs(loaded.edge(0, 2)).at("weight")), 4.0));
    CHECK(std::get<std::string>(loaded.node_attrs(0).at("kind")) == "source");

    // The documented NetworkX-style writer is a real exported API, not a
    // translation-unit-only inline helper.
    nx::write_gml(fixture.graph, path.string());
    DiGraph loaded_via_nx = GmlLoader::load_directed(path.string());
    CHECK(loaded_via_nx.num_edges() == fixture.graph.num_edges());

    bool undirected_loader_rejected = false;
    try
    {
        static_cast<void>(GmlLoader::load(path.string()));
    }
    catch (const std::runtime_error&)
    {
        undirected_loader_rejected = true;
    }
    CHECK(undirected_loader_rejected);
}

void test_graph_regressions()
{
    DiGraph directed_loop;
    directed_loop.add_node();
    directed_loop.add_node();
    const DiEdge directed_self_edge = directed_loop.add_edge(0, 0);
    directed_loop.edge_attrs(directed_self_edge)["weight"] = 3.0;
    CHECK(directed_loop.degree(0) == 2);
    CHECK(close(nx::degree_centrality(directed_loop)[0], 2.0));
    const SparseMatrix directed_loop_matrix =
        nx::adjacency_matrix(directed_loop);
    CHECK(directed_loop_matrix.nnz() == 1);
    CHECK(close(sparse_value(directed_loop_matrix, 0, 0), 3.0));

    DiGraph diamond;
    for (size_t i = 0; i < 4; ++i)
    {
        diamond.add_node();
    }
    for (const auto [u, v] : {
             std::pair<Vertex, Vertex>{0, 1},
             {0, 2},
             {1, 3},
             {2, 3}})
    {
        diamond.edge_attrs(diamond.add_edge(u, v))["weight"] = 1.0;
    }
    check_scores(
        nx::betweenness_centrality(diamond),
        {0.0, 1.0 / 12.0, 1.0 / 12.0, 0.0},
        1e-12);
    check_scores(
        nx::closeness_centrality(diamond),
        {0.0, 1.0 / 3.0, 1.0 / 3.0, 0.75},
        1e-12);

    Graph singleton;
    singleton.add_node();
    CHECK(nx::degree_centrality(singleton) == nx::NodeScores({1.0}));

    const AttrId weight_id = singleton.attr_id("weight");
    const Edge loop = singleton.add_edge(0, 0);
    singleton.edge_attrs(loop).set(weight_id, 2.0);
    const SparseMatrix loop_matrix = nx::adjacency_matrix(singleton);
    CHECK(loop_matrix.nnz() == 1);
    CHECK(close(sparse_value(loop_matrix, 0, 0), 2.0));

    Graph path;
    path.add_node();
    path.add_node();
    path.add_node();
    const AttrId path_weight = path.attr_id("weight");
    path.edge_attrs(path.add_edge(0, 1)).set(path_weight, 1.0);
    const Edge removed = path.add_edge(0, 2);
    path.edge_attrs(removed).set(path_weight, 8.0);
    path.edge_attrs(path.add_edge(1, 2)).set(path_weight, 1.0);
    CHECK(path.remove_edge(0, 2));

    check_scores(
        nx::betweenness_centrality(path),
        {0.0, 1.0, 0.0},
        1e-12);

    // Edge IDs are monotonic, so a cache built after removal must tolerate a
    // hole instead of sizing itself from num_edges().
    WeightCache cache(path);
    CHECK(close(cache.value(1, 2, cache.attribute_id("weight")), 1.0));
}

void test_deterministic_differential()
{
    constexpr size_t graph_count = 12;
    constexpr size_t node_count = 10;
    for (uint32_t seed = 1; seed <= graph_count; ++seed)
    {
        DiGraph graph;
        for (size_t i = 0; i < node_count; ++i)
        {
            graph.add_node();
        }
        const AttrId weight_id = graph.attr_id("weight");
        uint32_t state = seed * 747796405u + 2891336453u;
        for (Vertex u = 0; u < node_count; ++u)
        {
            for (Vertex v = 0; v < node_count; ++v)
            {
                if (u == v)
                {
                    continue;
                }
                state = state * 1664525u + 1013904223u;
                if (state % 5u != 0u)
                {
                    continue;
                }
                const DiEdge edge = graph.add_edge(u, v);
                graph.edge_attrs(edge).set(
                    weight_id,
                    static_cast<int64_t>(1u + (state % 23u)));
            }
        }

        const DistanceMatrix all_pairs = floyd_warshall(graph);
        for (Vertex source = 0; source < node_count; ++source)
        {
            const BFSResult unweighted = bfs(graph, source);
            const DijkstraResult weighted = dijkstra(graph, source);
            const VertexSet banned_vertices = {
                static_cast<Vertex>((source + 3) % node_count)};
            const EdgeSet banned_edges = {{
                source,
                static_cast<Vertex>((source + 1) % node_count)}};
            const DijkstraResult masked = dijkstra(
                graph,
                source,
                banned_vertices,
                banned_edges);
            for (Vertex target = 0; target < node_count; ++target)
            {
                const BidirectionalBFSResult two_way_bfs =
                    bidirectional_bfs(graph, source, target);
                const bool bfs_reachable =
                    unweighted.distance[target] !=
                    std::numeric_limits<size_t>::max();
                CHECK(two_way_bfs.found == bfs_reachable);
                if (bfs_reachable)
                {
                    CHECK(two_way_bfs.distance == unweighted.distance[target]);
                    CHECK(is_valid_directed_path(
                        graph, two_way_bfs.path, source, target));
                }

                const BidirectionalPathResult two_way_dijkstra =
                    bidirectional_dijkstra(graph, source, target);
                const bool dijkstra_reachable =
                    weighted.distance[target] !=
                    std::numeric_limits<double>::max();
                CHECK(two_way_dijkstra.found == dijkstra_reachable);
                if (dijkstra_reachable)
                {
                    CHECK(close(
                        two_way_dijkstra.cost,
                        weighted.distance[target],
                        1e-12));
                    CHECK(close(
                        all_pairs(source, target),
                        weighted.distance[target],
                        1e-12));
                    CHECK(is_valid_directed_path(
                        graph, two_way_dijkstra.path, source, target));
                    CHECK(close(
                        path_cost(graph, two_way_dijkstra.path),
                        weighted.distance[target],
                        1e-12));
                }

                const BidirectionalPathResult two_way_masked =
                    bidirectional_dijkstra(
                        graph,
                        source,
                        target,
                        banned_vertices,
                        banned_edges);
                const bool masked_reachable =
                    masked.distance[target] !=
                    std::numeric_limits<double>::max();
                CHECK(two_way_masked.found == masked_reachable);
                if (masked_reachable)
                {
                    CHECK(close(
                        two_way_masked.cost,
                        masked.distance[target],
                        1e-12));
                    CHECK(is_valid_directed_path(
                        graph, two_way_masked.path, source, target));
                }
            }
        }
    }
}

} // namespace

int main()
{
    try
    {
        Fixture fixture;
        test_core_api(fixture);
        test_attributes_and_sparse(fixture);
        test_unweighted_paths(fixture);
        test_weighted_paths(fixture);
        test_nx_api(fixture);
        test_weight_cache(fixture);
        test_masked_and_indexed_surface(fixture);
        test_gml_roundtrip(fixture);
        test_graph_regressions();
        test_deterministic_differential();
        std::cout << "test_digraph: ALL TESTS PASSED\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_digraph: " << error.what() << '\n';
        return 1;
    }
}
