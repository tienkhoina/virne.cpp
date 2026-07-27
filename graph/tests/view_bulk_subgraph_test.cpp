#include "graph.h"
#include "nx/relabel.h"
#include "nx/sparse.h"
#include "nx/subgraph.h"
#include "nx/views.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
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

template <typename Range,
          typename GraphType>
std::vector<EdgeEndpoints> endpoints(
    const Range& range,
    const GraphType& graph)
{
    std::vector<EdgeEndpoints> result;
    for (const auto value : range)
    {
        if constexpr (std::is_same_v<
                          std::decay_t<
                              decltype(value)>,
                          EdgeEndpoints>)
        {
            result.push_back(value);
        }
        else
        {
            result.emplace_back(
                graph.source(value),
                graph.target(value));
        }
    }
    return result;
}

template <typename Range>
std::vector<Vertex> vertices(
    const Range& range)
{
    return std::vector<Vertex>(
        range.begin(),
        range.end());
}

template <typename IncidenceList>
size_t count_target(
    const IncidenceList& incidence,
    Vertex target)
{
    size_t count = 0;
    for (const auto& edge : incidence)
    {
        count += static_cast<size_t>(
            edge.get_target() == target);
    }
    return count;
}

AttrObject attrs(
    std::initializer_list<
        std::pair<std::string, AttrValue>> values)
{
    AttrObject result;
    result.entries.assign(
        values.begin(),
        values.end());
    return result;
}

Graph order_fixture()
{
    return Graph(
        std::vector<EdgeEndpoints>{
            {2, 3}, {0, 2}, {4, 1},
            {3, 0}, {2, 2}, {0, 4}});
}

void test_degree_and_incident_order()
{
    Graph graph = order_fixture();
    CHECK(graph.degree() == DegreeItems({
        {0, 3}, {1, 1}, {2, 4},
        {3, 2}, {4, 2}}));
    CHECK(nx::degree(graph) == graph.degree());
    CHECK(endpoints(graph.edges(2), graph) ==
          std::vector<EdgeEndpoints>({
              {2, 3}, {2, 0}, {2, 2}}));
    CHECK(endpoints(
              graph.edge_view().incident(2),
              graph) ==
          std::vector<EdgeEndpoints>({
              {2, 3}, {2, 0}, {2, 2}}));
    CHECK(endpoints(
              graph.edge_view()
                  .incident(2)
                  .descriptors(),
              graph) ==
          std::vector<EdgeEndpoints>({
              {2, 3}, {2, 0}, {2, 2}}));

    DiGraph directed(
        std::vector<EdgeEndpoints>{
            {2, 3}, {0, 2}, {4, 1},
            {3, 0}, {2, 2}, {0, 4},
            {2, 1}});
    CHECK(directed.degree() == DegreeItems({
        {0, 3}, {1, 2}, {2, 5},
        {3, 2}, {4, 2}}));
    CHECK(directed.in_degree() == DegreeItems({
        {0, 1}, {1, 2}, {2, 2},
        {3, 1}, {4, 1}}));
    CHECK(directed.out_degree() == DegreeItems({
        {0, 2}, {1, 0}, {2, 3},
        {3, 1}, {4, 1}}));
    CHECK(nx::degree(directed) ==
          directed.degree());
    CHECK(nx::in_degree(directed) ==
          directed.in_degree());
    CHECK(nx::out_degree(directed) ==
          directed.out_degree());
    CHECK(endpoints(
              directed.edges(2),
              directed) ==
          std::vector<EdgeEndpoints>({
              {2, 3}, {2, 2}, {2, 1}}));
    CHECK(endpoints(
              directed.edge_view().outgoing(2),
              directed) ==
          std::vector<EdgeEndpoints>({
              {2, 3}, {2, 2}, {2, 1}}));
    const auto directed_neighbor =
        directed.adjacency_view()[2].begin();
    CHECK(directed.source(
              directed_neighbor.edge()) == 2);
    CHECK(directed.target(
              directed_neighbor.edge()) == 3);
}

void test_zero_copy_views()
{
    Graph graph = order_fixture();
    graph.node_attrs(2)["cpu"] =
        int64_t{7};
    graph.edge_attrs(
        graph.edge(2, 3))["bw"] =
            int64_t{11};

    auto nodes = graph.node_view();
    CHECK(vertices(nodes) ==
          std::vector<Vertex>({0, 1, 2, 3, 4}));
    CHECK(nodes.contains(2));
    CHECK(!nodes.contains(5));
    nodes[2]["cpu"] = int64_t{9};
    CHECK(std::get<int64_t>(
              graph.node_attrs(2).at("cpu")) == 9);

    size_t node_data_count = 0;
    for (auto item : nodes.data())
    {
        CHECK(item.node == node_data_count);
        item.attrs["seen"] = true;
        ++node_data_count;
    }
    CHECK(node_data_count == 5);
    CHECK(std::get<bool>(
        graph.node_attrs(4).at("seen")));

    auto edges = graph.edge_view();
    static_assert(std::is_same_v<
        decltype(*edges.begin()),
        EdgeEndpoints>);
    CHECK(edges.contains(3, 2));
    CHECK(!edges.contains(1, 3));
    edges[{3, 2}]["bw"] = int64_t{13};
    CHECK(std::get<int64_t>(
              graph.edge_attrs(
                  graph.edge(2, 3)).at("bw")) == 13);

    std::vector<EdgeEndpoints> data_order;
    for (auto item : edges.data())
    {
        data_order.emplace_back(
            item.source,
            item.target);
        item.attrs["visited"] = true;
    }
    CHECK(data_order ==
          std::vector<EdgeEndpoints>({
              {0, 2}, {0, 3}, {0, 4},
              {1, 4}, {2, 3}, {2, 2}}));
    CHECK(endpoints(
              edges.descriptors(),
              graph) == data_order);

    auto adjacency = graph.adjacency_view();
    CHECK(vertices(adjacency[2]) ==
          std::vector<Vertex>({3, 0, 2}));
    const auto first_neighbor =
        adjacency[2].begin();
    CHECK(first_neighbor.edge().get_property() ==
          graph.edge(2, 3).get_property());
    for (auto item : adjacency[2].data())
    {
        item.attrs["from_two"] = true;
    }
    CHECK(std::get<bool>(
        graph.edge_attrs(
            graph.edge(2, 0)).at("from_two")));

    size_t adjacency_nodes = 0;
    for (auto item : adjacency.data())
    {
        CHECK(item.node == adjacency_nodes);
        ++adjacency_nodes;
    }
    CHECK(adjacency_nodes == 5);

    const Graph& constant = graph;
    static_assert(std::is_same_v<
        decltype(constant.node_view().at(0)),
        const AttrMap&>);
    static_assert(std::is_same_v<
        decltype(constant.edge_view().at(0, 2)),
        const AttrMap&>);
    CHECK(std::get<int64_t>(
              constant.node_view()[2].at("cpu")) == 9);
    CHECK(constant.adjacency_view()[2]
              .begin()
              .edge()
              .get_property() ==
          constant.edge(2, 3).get_property());

    bool rejected_node = false;
    try
    {
        static_cast<void>(
            constant.adjacency_view().at(5));
    }
    catch (const std::out_of_range&)
    {
        rejected_node = true;
    }
    CHECK(rejected_node);
}

void test_self_loop_storage_lifecycle()
{
    Graph graph;
    graph.add_nodes_from(
        std::vector<Vertex>{0, 1, 2});
    graph.add_edge(1, 2);
    Edge loop = graph.add_edge(1, 1);
    graph.edge_attrs(loop)["weight"] = 7.0;
    const uint32_t first_loop_id =
        graph.edge_id(loop);

    CHECK(count_target(
              graph.neighbors_fast(1), 1) == 1);
    CHECK(graph.degree(1) == 3);
    CHECK(vertices(graph.adjacency_view()[1]) ==
          std::vector<Vertex>({2, 1}));

    Graph copied = graph;
    CHECK(count_target(
              copied.neighbors_fast(1), 1) == 1);
    CHECK(copied.degree(1) == 3);
    CHECK(std::get<double>(
              copied.edge_attrs(
                  copied.edge(1, 1)).at(
                  "weight")) == 7.0);

    const Edge moved_descriptor =
        copied.edge(1, 1);
    const void* moved_property =
        moved_descriptor.get_property();
    Graph moved(std::move(copied));
    CHECK(count_target(
              moved.neighbors_fast(1), 1) == 1);
    CHECK(moved_descriptor.get_property() ==
          moved_property);
    CHECK(moved.degree(1) == 3);

    CHECK(moved.remove_edge(1, 1));
    CHECK(!moved.has_edge(1, 1));
    CHECK(count_target(
              moved.neighbors_fast(1), 1) == 0);
    CHECK(moved.degree(1) == 1);

    loop = moved.add_edge(1, 1);
    CHECK(moved.edge_id(loop) !=
          first_loop_id);
    moved.edge_attrs(loop)["weight"] = 9.0;
    CHECK(count_target(
              moved.neighbors_fast(1), 1) == 1);
    CHECK(moved.degree(1) == 3);
    const auto graph_dense =
        nx::adjacency_matrix(moved).toarray();
    CHECK(graph_dense[1][1] == 9.0);

    DiGraph directed;
    directed.add_nodes_from(
        std::vector<Vertex>{0, 1});
    directed.add_edge(0, 1);
    DiEdge directed_loop =
        directed.add_edge(1, 1);
    directed.edge_attrs(directed_loop)[
        "weight"] = 4.0;
    const uint32_t first_directed_id =
        directed.edge_id(directed_loop);
    CHECK(count_target(
              directed.successors_fast(1), 1) == 1);
    CHECK(count_target(
              directed.predecessors_fast(1), 1) == 1);
    CHECK(directed.degree(1) == 3);
    CHECK(vertices(
              directed.adjacency_view()[1]) ==
          std::vector<Vertex>({1}));

    DiGraph directed_copy = directed;
    DiGraph directed_move(
        std::move(directed_copy));
    CHECK(count_target(
              directed_move.successors_fast(1), 1) == 1);
    CHECK(count_target(
              directed_move.predecessors_fast(1), 1) == 1);
    CHECK(directed_move.degree(1) == 3);
    CHECK(directed_move.remove_edge(1, 1));
    CHECK(directed_move.degree(1) == 1);

    directed_loop =
        directed_move.add_edge(1, 1);
    CHECK(directed_move.edge_id(
              directed_loop) !=
          first_directed_id);
    directed_move.edge_attrs(directed_loop)[
        "weight"] = 6.0;
    const auto directed_dense =
        nx::adjacency_matrix(
            directed_move).toarray();
    CHECK(directed_dense[1][1] == 6.0);
}

void test_bulk_attrs_duplicates_and_validation()
{
    AttrValue nested = make_attr_object({
        {"value", int64_t{41}}});
    Graph graph;
    graph.add_nodes_from(
        std::vector<NodeWithAttrs>{
            {2, attrs({{"name", std::string("two")}})},
            {0, attrs({{"a", int64_t{1}}})},
            {1, attrs({})},
            {0, attrs({{"b", int64_t{2}}})}});
    CHECK(graph.num_nodes() == 3);
    CHECK(graph.node_attrs(0).contains("a"));
    CHECK(graph.node_attrs(0).contains("b"));

    graph.add_edges_from(
        std::vector<EdgeWithAttrs>{
            {0, 1, attrs({
                {"x", int64_t{1}},
                {"nested", nested}})},
            {1, 0, attrs({{"y", int64_t{2}}})},
            {0, 1, attrs({{"x", int64_t{3}}})}});
    CHECK(graph.num_edges() == 1);
    const AttrMap& edge_attrs =
        graph.edge_attrs(graph.edge(0, 1));
    CHECK(std::get<int64_t>(edge_attrs.at("x")) == 3);
    CHECK(std::get<int64_t>(edge_attrs.at("y")) == 2);
    attr_object(nested)->set(
        "value",
        int64_t{99});
    CHECK(std::get<int64_t>(
        *attr_object(
            edge_attrs.at("nested"))->find("value")) == 41);

    Graph inferred(
        std::vector<EdgeEndpoints>{
            {2, 0}, {1, 2}});
    CHECK(inferred.num_nodes() == 3);
    CHECK(inferred.num_edges() == 2);

    Graph with_isolate(
        4,
        std::vector<EdgeEndpoints>{{1, 2}});
    CHECK(with_isolate.num_nodes() == 4);
    CHECK(with_isolate.num_edges() == 1);

    bool rejected_gap = false;
    Graph invalid;
    try
    {
        invalid.add_nodes_from(
            std::vector<Vertex>{0, 2});
    }
    catch (const std::invalid_argument&)
    {
        rejected_gap = true;
    }
    CHECK(rejected_gap);
    CHECK(invalid.num_nodes() == 0);

    Graph gap(
        std::vector<EdgeEndpoints>{{1, 2}});
    CHECK(gap.num_nodes() == 3);
    CHECK(gap.num_edges() == 1);
    CHECK(gap.has_edge(1, 2));

    DiGraph directed;
    directed.add_edges_from(
        std::vector<EdgeWithAttrs>{
            {0, 1, attrs({{"x", int64_t{1}}})},
            {1, 0, attrs({{"x", int64_t{2}}})},
            {0, 1, attrs({{"y", int64_t{3}}})}});
    CHECK(directed.num_edges() == 2);
    CHECK(std::get<int64_t>(
              directed.edge_attrs(
                  directed.edge(0, 1)).at("x")) == 1);
    CHECK(std::get<int64_t>(
              directed.edge_attrs(
                  directed.edge(0, 1)).at("y")) == 3);
}

void test_live_subgraph_views()
{
    Graph graph(
        3,
        std::vector<EdgeEndpoints>{
            {0, 1}, {1, 2}});
    for (Vertex node = 0; node < 3; ++node)
    {
        graph.node_attrs(node)["enabled"] = true;
    }
    graph.edge_attrs(
        graph.edge(0, 1))["capacity"] =
            int64_t{1};
    graph.edge_attrs(
        graph.edge(1, 2))["capacity"] =
            int64_t{0};

    const AttrId enabled_id =
        graph.attr_id("enabled");
    const AttrId capacity_id =
        graph.attr_id("capacity");

    const nx::GraphView live =
        nx::subgraph_view(
            graph,
            [&graph, enabled_id](Vertex node)
            {
                return std::get<bool>(
                    graph.node_attrs(node).at(enabled_id));
            },
            [&graph, capacity_id](Vertex u, Vertex v)
            {
                const AttrValue* capacity =
                    graph.edge_attrs(
                        graph.edge(u, v)).find(
                            capacity_id);
                return capacity != nullptr &&
                       attr_to_double(*capacity) > 0.0;
            });

    CHECK(live.contains_edge(0, 1));
    CHECK(!live.contains_edge(1, 2));
    graph.node_attrs(1)["enabled"] = false;
    CHECK(!live.contains_edge(0, 1));
    graph.node_attrs(1)["enabled"] = true;
    graph.edge_attrs(
        graph.edge(1, 2))["capacity"] =
            int64_t{2};
    CHECK(live.contains_edge(1, 2));

    CHECK(graph.remove_edge(0, 1));
    CHECK(!live.contains_edge(0, 1));
    graph.edge_attrs(
        graph.add_edge(0, 1))["capacity"] =
            int64_t{4};
    CHECK(live.contains_edge(0, 1));

    graph.add_node();
    graph.node_attrs(3)["enabled"] = true;
    graph.edge_attrs(
        graph.add_edge(2, 3))["capacity"] =
            int64_t{1};
    CHECK(live.contains_node(3));
    CHECK(live.contains_edge(2, 3));

    const nx::GraphView induced =
        nx::subgraph(
            graph,
            std::vector<Vertex>{2, 0, 1, 99});
    CHECK(induced.contains_node(0));
    CHECK(induced.contains_node(2));
    CHECK(!induced.contains_node(3));
    CHECK(induced.contains_edge(0, 1));
    CHECK(induced.contains_edge(1, 2));
    CHECK(!induced.contains_edge(2, 3));
}

void test_convert_labels()
{
    Graph graph = order_fixture();
    graph.graph_attrs()["name"] =
        std::string("fixture");
    for (Vertex node = 0;
         node < graph.num_nodes();
         ++node)
    {
        graph.node_attrs(node)["source"] =
            static_cast<int64_t>(node + 10);
    }
    graph.edge_attrs(
        graph.edge(0, 2))["weight"] =
            2.5;

    const Graph identity =
        nx::convert_node_labels_to_integers(
            graph,
            0,
            "default",
            "old_label");
    CHECK(identity.num_nodes() == graph.num_nodes());
    CHECK(identity.num_edges() == graph.num_edges());
    CHECK(std::get<int64_t>(
              identity.node_attrs(3).at(
                  "old_label")) == 3);
    CHECK(std::get<std::string>(
              identity.graph_attrs().at(
                  "name")) == "fixture");
    CHECK(std::get<double>(
              identity.edge_attrs(
                  identity.edge(0, 2)).at(
                      "weight")) == 2.5);

    const Graph degree_order =
        nx::convert_node_labels_to_integers(
            graph,
            0,
            "increasing degree",
            "old_label");
    // NetworkX ordering pairs are (degree, node): 1, 3, 4, 0, 2.
    CHECK(std::get<int64_t>(
              degree_order.node_attrs(0).at(
                  "old_label")) == 1);
    CHECK(std::get<int64_t>(
              degree_order.node_attrs(1).at(
                  "old_label")) == 3);
    CHECK(std::get<int64_t>(
              degree_order.node_attrs(2).at(
                  "old_label")) == 4);
    CHECK(degree_order.has_edge(3, 4));

    bool rejected_offset = false;
    try
    {
        static_cast<void>(
            nx::convert_node_labels_to_integers(
                graph, 1));
    }
    catch (const std::invalid_argument&)
    {
        rejected_offset = true;
    }
    CHECK(rejected_offset);

    DiGraph directed(
        3,
        std::vector<EdgeEndpoints>{
            {2, 0}, {0, 1}});
    const DiGraph converted =
        nx::convert_node_labels_to_integers(
            directed);
    CHECK(endpoints(
              converted.edge_view(),
              converted) ==
          std::vector<EdgeEndpoints>({
              {0, 1}, {2, 0}}));
}

} // namespace

int main()
{
    test_degree_and_incident_order();
    test_zero_copy_views();
    test_self_loop_storage_lifecycle();
    test_bulk_attrs_duplicates_and_validation();
    test_live_subgraph_views();
    test_convert_labels();
    return 0;
}
