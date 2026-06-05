#include "waxman_generator.h"

#include "../../random/py_random.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
struct Point
{
    double x;
    double y;
};
}

Graph
WaxmanGenerator::generate(
    const WaxmanConfig& cfg)
{
    Graph g;

    PyRandom rng(
        cfg.seed);

    const AttrId x_attr =
        g.attr_id("x");
    const AttrId y_attr =
        g.attr_id("y");
    const AttrId distance_attr =
        g.attr_id("distance");

    std::vector<Vertex> vertices;
    std::vector<Point> points;

    vertices.reserve(
        cfg.num_nodes);

    points.reserve(
        cfg.num_nodes);

    //
    // create nodes
    //
    for (size_t i = 0;
         i < cfg.num_nodes;
         ++i)
    {
        auto v =
            g.add_node();

        double x =
            rng.uniform(
                0.0,
                1.0);

        double y =
            rng.uniform(
                0.0,
                1.0);

        g.node_attrs(v).set(
            x_attr,
            x);

        g.node_attrs(v).set(
            y_attr,
            y);

        vertices.push_back(v);
        points.push_back(
            {x, y});
    }

    //
    // compute L
    //
    double L = 0.0;

    for (size_t i = 0;
         i < points.size();
         ++i)
    {
        for (size_t j = i + 1;
             j < points.size();
             ++j)
        {
            double dx =
                points[i].x -
                points[j].x;

            double dy =
                points[i].y -
                points[j].y;

            double d =
                std::sqrt(
                    dx * dx +
                    dy * dy);

            L =
                std::max(
                    L,
                    d);
        }
    }

    if (L == 0.0)
    {
        return g;
    }

    //
    // generate edges
    //
    for (size_t i = 0;
         i < points.size();
         ++i)
    {
        for (size_t j = i + 1;
             j < points.size();
             ++j)
        {
            double dx =
                points[i].x -
                points[j].x;

            double dy =
                points[i].y -
                points[j].y;

            double d =
                std::sqrt(
                    dx * dx +
                    dy * dy);

            double p =
                cfg.beta *
                std::exp(
                    -d /
                    (cfg.alpha * L));

            if (rng.random() < p)
            {
                auto e =
                    g.add_edge(
                        vertices[i],
                        vertices[j]);

                g.edge_attrs(e).set(
                    distance_attr,
                    d);
            }
        }
    }

    return g;
}