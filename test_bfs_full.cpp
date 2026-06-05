#include <cassert>
#include <cmath>
#include <iostream>

#include "graph/graph.h"
#include "graph/generators/gml_loader.h"
#include "graph/io/graph_saver.h"

static void test_roundtrip()
{
    Graph g;

    for (int i = 0; i < 5; ++i)
    {
        g.add_node();
    }

    auto e01 =
        g.add_edge(0, 1);

    auto e12 =
        g.add_edge(1, 2);

    auto e23 =
        g.add_edge(2, 3);

    g.node_attrs(0)["cpu"] =
        int64_t(100);

    g.node_attrs(1)["cpu"] =
        int64_t(200);

    g.node_attrs(0)["name"] =
        std::string("node0");

    g.node_attrs(1)["active"] =
        true;

    g.edge_attrs(e01)["weight"] =
        10.5;

    g.edge_attrs(e12)["weight"] =
        20.5;

    g.edge_attrs(e23)["bw"] =
        int64_t(300);

    GraphSaver::save_gml(
        g,
        "roundtrip.gml");

    Graph g2 =
        GmlLoader::load(
            "roundtrip.gml");

    assert(
        g.num_nodes() ==
        g2.num_nodes());

    assert(
        g.num_edges() ==
        g2.num_edges());

    //
    // node attrs
    //

    assert(
        std::get<int64_t>(
            g2.node_attrs(0)
                .at("cpu")) ==
        100);

    assert(
        std::get<int64_t>(
            g2.node_attrs(1)
                .at("cpu")) ==
        200);

    assert(
        std::get<std::string>(
            g2.node_attrs(0)
                .at("name")) ==
        "node0");

    assert(
        std::get<bool>(
            g2.node_attrs(1)
                .at("active")));

    //
    // edge attrs
    //

    auto e =
        g2.edge(
            0,
            1);

    assert(
        std::fabs(
            std::get<double>(
                g2.edge_attrs(e)
                    .at("weight"))
            - 10.5) <
        1e-9);

    auto e2 =
        g2.edge(
            1,
            2);

    assert(
        std::fabs(
            std::get<double>(
                g2.edge_attrs(e2)
                    .at("weight"))
            - 20.5) <
        1e-9);

    auto e3 =
        g2.edge(
            2,
            3);

    assert(
        std::get<int64_t>(
            g2.edge_attrs(e3)
                .at("bw")) ==
        300);

    std::cout
        << "ROUNDTRIP OK\n";
}

int main()
{
    test_roundtrip();

    std::cout
        << "ALL TESTS PASSED\n";

    return 0;
}