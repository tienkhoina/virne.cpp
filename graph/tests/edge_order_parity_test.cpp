#include "graph.h"

#include <cstdint>
#include <stdexcept>
#include <string>
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

template <typename GraphType>
std::vector<std::pair<Vertex, Vertex>>
edge_sequence(const GraphType& graph)
{
    std::vector<std::pair<Vertex, Vertex>> result;
    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        result.emplace_back(
            graph.source(*edge),
            graph.target(*edge));
    }
    return result;
}

template <typename GraphType>
std::vector<Vertex> neighbor_sequence(
    const GraphType& graph,
    Vertex node)
{
    std::vector<Vertex> result;
    auto [neighbor, neighbor_end] =
        graph.neighbors(node);
    for (; neighbor != neighbor_end; ++neighbor)
    {
        result.push_back(*neighbor);
    }
    return result;
}

std::vector<Vertex> predecessor_sequence(
    const DiGraph& graph,
    Vertex node)
{
    std::vector<Vertex> result;
    auto [predecessor, predecessor_end] =
        graph.predecessors(node);
    for (; predecessor != predecessor_end;
         ++predecessor)
    {
        result.push_back(*predecessor);
    }
    return result;
}

void add_nodes(Graph& graph)
{
    for (size_t i = 0; i < 5; ++i)
    {
        CHECK(graph.add_node() == i);
    }
}

void add_nodes(DiGraph& graph)
{
    for (size_t i = 0; i < 5; ++i)
    {
        CHECK(graph.add_node() == i);
    }
}

void test_graph_edgeview_order()
{
    Graph graph;
    add_nodes(graph);

    const AttrId sequence =
        graph.attr_id("sequence");
    const std::vector<std::pair<Vertex, Vertex>> inserted{
        {2, 3}, {0, 2}, {4, 1},
        {3, 0}, {2, 2}, {0, 4}};
    for (size_t i = 0; i < inserted.size(); ++i)
    {
        const auto [u, v] = inserted[i];
        graph.edge_attrs(
            graph.add_edge(u, v)).set(
                sequence,
                static_cast<int64_t>(i));
    }

    // Oracle: NetworkX 3.4.2 Graph.edges() for the insertion fixture above.
    const std::vector<std::pair<Vertex, Vertex>> expected{
        {0, 2}, {0, 3}, {0, 4},
        {1, 4}, {2, 3}, {2, 2}};
    CHECK(edge_sequence(graph) == expected);
    CHECK(edge_sequence(graph) == expected);

    // The reverse duplicate neither changes order nor creates a new ID.
    const uint32_t existing_id =
        graph.edge_id(graph.edge(0, 3));
    CHECK(graph.edge_id(graph.add_edge(3, 0)) ==
          existing_id);
    CHECK(edge_sequence(graph) == expected);

    // Stable list relinking must preserve the edge properties/descriptors.
    CHECK(std::get<int64_t>(
              graph.edge_attrs(
                  graph.edge(2, 3)).at(sequence)) == 0);
    CHECK(std::get<int64_t>(
              graph.edge_attrs(
                  graph.edge(0, 3)).at(sequence)) == 3);

    // Calling edges() relinks only the global list. A subsequent graph copy
    // must still retain each node's original adjacency insertion order.
    const Graph copied = graph;
    CHECK(neighbor_sequence(copied, 2) ==
          neighbor_sequence(graph, 2));
    const std::vector<std::pair<Vertex, Vertex>> copied_directed{
        {0, 2}, {0, 3}, {0, 4}, {1, 4},
        {2, 3}, {2, 0}, {2, 2},
        {3, 2}, {3, 0}, {4, 1}, {4, 0}};
    CHECK(edge_sequence(copied.to_directed()) ==
          copied_directed);

    // Boost adjacency_list 1.85 has no real move operation.  A defaulted
    // Graph move would therefore copy from the normalized global list and
    // reorder node 2's incidence list from {3, 0, ...} to {0, 3, ...}.
    const auto expected_neighbors =
        neighbor_sequence(graph, 2);
    Graph move_source = graph;
    const Edge moved_descriptor =
        move_source.edge(2, 3);
    const void* moved_property =
        moved_descriptor.get_property();
    Graph moved(std::move(move_source));
    CHECK(neighbor_sequence(moved, 2) ==
          expected_neighbors);
    CHECK(edge_sequence(moved) == expected);
    CHECK(moved_descriptor.get_property() ==
          moved_property);
    CHECK(moved.edge_id(moved_descriptor) ==
          graph.edge_id(graph.edge(2, 3)));

    Graph assignment_source = graph;
    const Edge assigned_descriptor =
        assignment_source.edge(0, 3);
    const void* assigned_property =
        assigned_descriptor.get_property();
    Graph move_assigned;
    move_assigned.add_node();
    move_assigned = std::move(assignment_source);
    CHECK(neighbor_sequence(move_assigned, 2) ==
          expected_neighbors);
    CHECK(edge_sequence(move_assigned) == expected);
    CHECK(assigned_descriptor.get_property() ==
          assigned_property);
    CHECK(move_assigned.edge_id(assigned_descriptor) ==
          graph.edge_id(graph.edge(0, 3)));

    // A reverse endpoint insertion is canonical in EdgeView and ID metadata.
    const uint32_t edge14 =
        graph.edge_id(graph.edge(1, 4));
    CHECK(graph.edge_endpoints(edge14) ==
          std::make_pair(Vertex{1}, Vertex{4}));

    CHECK(graph.remove_edge(0, 2));
    const uint32_t replacement_id =
        graph.edge_id(graph.add_edge(2, 0));
    CHECK(replacement_id != existing_id);

    // Re-add appends inside node 0's adjacency group, as in NetworkX.
    const std::vector<std::pair<Vertex, Vertex>> after_readd{
        {0, 3}, {0, 4}, {0, 2},
        {1, 4}, {2, 3}, {2, 2}};
    CHECK(edge_sequence(graph) == after_readd);
}

void test_digraph_edgeview_order()
{
    DiGraph graph;
    add_nodes(graph);
    for (const auto [u, v] :
         std::vector<std::pair<Vertex, Vertex>>{
             {2, 3}, {0, 2}, {4, 1}, {3, 0},
             {2, 2}, {0, 4}, {2, 1}})
    {
        graph.add_edge(u, v);
    }

    const std::vector<std::pair<Vertex, Vertex>> expected{
        {0, 2}, {0, 4}, {2, 3}, {2, 2},
        {2, 1}, {3, 0}, {4, 1}};
    CHECK(edge_sequence(graph) == expected);

    const DiGraph copied = graph;
    CHECK(predecessor_sequence(copied, 1) ==
          std::vector<Vertex>({4, 2}));

    // Sorting the global list by source must not change predecessor insertion
    // order when a DiGraph is moved or move-assigned.
    const auto expected_predecessors =
        predecessor_sequence(graph, 1);
    DiGraph move_source = graph;
    const DiEdge moved_descriptor =
        move_source.edge(4, 1);
    const void* moved_property =
        moved_descriptor.get_property();
    DiGraph moved(std::move(move_source));
    CHECK(predecessor_sequence(moved, 1) ==
          expected_predecessors);
    CHECK(edge_sequence(moved) == expected);
    CHECK(moved_descriptor.get_property() ==
          moved_property);

    DiGraph assignment_source = graph;
    const DiEdge assigned_descriptor =
        assignment_source.edge(2, 1);
    const void* assigned_property =
        assigned_descriptor.get_property();
    DiGraph move_assigned;
    move_assigned.add_node();
    move_assigned = std::move(assignment_source);
    CHECK(predecessor_sequence(move_assigned, 1) ==
          expected_predecessors);
    CHECK(edge_sequence(move_assigned) == expected);
    CHECK(assigned_descriptor.get_property() ==
          assigned_property);

    CHECK(graph.remove_edge(2, 3));
    graph.add_edge(2, 3);
    const std::vector<std::pair<Vertex, Vertex>> after_readd{
        {0, 2}, {0, 4}, {2, 2}, {2, 1},
        {2, 3}, {3, 0}, {4, 1}};
    CHECK(edge_sequence(graph) == after_readd);
}

void test_to_directed_adjacency_order()
{
    Graph graph;
    add_nodes(graph);
    const AttrId sequence =
        graph.attr_id("sequence");
    const std::vector<std::pair<Vertex, Vertex>> inserted{
        {2, 3}, {0, 2}, {4, 1},
        {3, 0}, {2, 2}, {0, 4}};
    for (size_t i = 0; i < inserted.size(); ++i)
    {
        graph.edge_attrs(
            graph.add_edge(
                inserted[i].first,
                inserted[i].second)).set(
                    sequence,
                    static_cast<int64_t>(i));
    }

    const DiGraph directed = graph.to_directed();
    const std::vector<std::pair<Vertex, Vertex>> expected{
        {0, 2}, {0, 3}, {0, 4}, {1, 4},
        {2, 3}, {2, 0}, {2, 2},
        {3, 2}, {3, 0}, {4, 1}, {4, 0}};
    CHECK(edge_sequence(directed) == expected);
    CHECK(directed.num_edges() == expected.size());

    for (const auto [u, v] : expected)
    {
        CHECK(std::get<int64_t>(
                  directed.edge_attrs(
                      directed.edge(u, v)).at(sequence)) ==
              std::get<int64_t>(
                  graph.edge_attrs(
                      graph.edge(u, v)).at(sequence)));
    }
}

void test_const_lazy_normalization()
{
    Graph graph;
    add_nodes(graph);
    graph.add_edge(3, 4);
    graph.add_edge(0, 3);
    const Graph constant(std::move(graph));
    CHECK((edge_sequence(constant) ==
           std::vector<std::pair<Vertex, Vertex>>({
               {0, 3}, {3, 4}})));

    DiGraph digraph;
    add_nodes(digraph);
    digraph.add_edge(3, 4);
    digraph.add_edge(0, 4);
    const DiGraph directed_constant(
        std::move(digraph));
    CHECK((edge_sequence(directed_constant) ==
           std::vector<std::pair<Vertex, Vertex>>({
               {0, 4}, {3, 4}})));
}

} // namespace

int main()
{
    test_graph_edgeview_order();
    test_digraph_edgeview_order();
    test_to_directed_adjacency_order();
    test_const_lazy_normalization();
    return 0;
}
