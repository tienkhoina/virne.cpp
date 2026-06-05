#include <cassert>
#include <iostream>

#include "graph/graph.h"
#include "graph/nx/connectivity.h"

namespace
{

void test_node_attributes()
{
    Graph g;

    auto v = g.add_node();

    g.node_attrs(v)["cpu"] = int64_t(100);
    g.node_attrs(v)["gpu"] = int64_t(8);

    assert(
        std::get<int64_t>(
            g.node_attrs(v).at("cpu"))
        == 100);

    assert(
        std::get<int64_t>(
            g.node_attrs(v).at("gpu"))
        == 8);
}

void test_edge_attributes()
{
    Graph g;

    auto u = g.add_node();
    auto v = g.add_node();

    auto e =
        g.add_edge(u, v);

    g.edge_attrs(e)["weight"] =
        3.5;

    assert(
        std::get<double>(
            g.edge_attrs(e).at("weight"))
        == 3.5);
}

void test_edge_ids()
{
    Graph g;

    auto v0 = g.add_node();
    auto v1 = g.add_node();
    auto v2 = g.add_node();

    auto e0 =
        g.add_edge(v0, v1);

    auto e1 =
        g.add_edge(v1, v2);

    uint32_t id0 =
        g.edge_id(e0);

    uint32_t id1 =
        g.edge_id(e1);

    assert(id0 != id1);

    Edge x0 =
        g.edge_by_id(id0);

    Edge x1 =
        g.edge_by_id(id1);

    assert(
        g.edge_id(x0)
        ==
        id0);

    assert(
        g.edge_id(x1)
        ==
        id1);
}

void test_remove_edge()
{
    Graph g;

    auto v0 = g.add_node();
    auto v1 = g.add_node();

    auto e =
        g.add_edge(v0, v1);

    uint32_t id =
        g.edge_id(e);

    assert(
        g.has_edge(v0, v1));

    g.remove_edge(v0, v1);

    assert(
        !g.has_edge(v0, v1));

    bool thrown = false;

    try
    {
        g.edge_by_id(id);
    }
    catch (...)
    {
        thrown = true;
    }

    assert(thrown);
}

void test_connectivity()
{
    Graph g;

    auto v0 = g.add_node();
    auto v1 = g.add_node();
    auto v2 = g.add_node();

    g.add_edge(v0, v1);

    assert(
        !nx::is_connected(g));

    g.add_edge(v1, v2);

    assert(
        nx::is_connected(g));
}

void test_edge_endpoint_lookup()
{
    Graph g;

    auto v0 = g.add_node();
    auto v1 = g.add_node();

    auto e =
        g.add_edge(v0, v1);

    auto id =
        g.edge_id(e);

    auto [u, v] =
        g.edge_endpoints(id);

    assert(
        (u == v0 && v == v1)
        ||
        (u == v1 && v == v0));
}

void test_attribute_registry()
{
    Graph g;

    auto v =
        g.add_node();

    g.node_attrs(v)["cpu"] =
        int64_t(10);

    g.node_attrs(v)["gpu"] =
        int64_t(20);

    AttrId cpu =
        g.attr_id("cpu");

    AttrId gpu =
        g.attr_id("gpu");

    assert(cpu != gpu);

    assert(
        g.attr_name(cpu)
        ==
        "cpu");

    assert(
        g.attr_name(gpu)
        ==
        "gpu");
}

}

int main()
{
    test_node_attributes();
    test_edge_attributes();
    test_edge_ids();
    test_remove_edge();
    test_connectivity();
    test_edge_endpoint_lookup();
    test_attribute_registry();

    std::cout
        << "ALL GRAPH TESTS PASSED\n";

    return 0;
}