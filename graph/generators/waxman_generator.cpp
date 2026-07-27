#include "waxman_generator.h"

#include "../../random/py_random.h"
#include "../../random/random_context.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
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
    PyRandom rng(
        cfg.seed);

    return generate(
        cfg,
        rng);
}

Graph
WaxmanGenerator::generate(
    const WaxmanConfig& cfg,
    PyRandom& rng)
{
    Graph g;

    const AttrId pos_attr =
        g.attr_id("pos");

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
            pos_attr,
            make_attr_list({
                AttrValue{x},
                AttrValue{y}}));

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

    if (points.size() < 2)
    {
        // NetworkX computes max(pairwise distances) when L is omitted.
        // For fewer than two positions that max is empty.
        throw std::invalid_argument(
            "waxman_graph requires at least two nodes when L is inferred");
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

            // NetworkX evaluates seed.random() before the probability
            // expression. Preserve that state consumption even when a zero
            // denominator makes the Python expression raise.
            const double draw = rng.random();
            const double denominator = cfg.alpha * L;
            if (denominator == 0.0)
            {
                throw std::invalid_argument(
                    "waxman_graph probability denominator is zero");
            }

            double p =
                cfg.beta *
                std::exp(
                    -d /
                    denominator);

            if (draw < p)
            {
                g.add_edge(
                    vertices[i],
                    vertices[j]);
            }
        }
    }

    return g;
}

namespace nx
{

Graph waxman_graph(
    size_t num_nodes,
    double beta,
    double alpha)
{
    return waxman_graph(
        num_nodes,
        beta,
        alpha,
        global_py_random());
}

Graph waxman_graph(
    size_t num_nodes,
    double beta,
    double alpha,
    PyRandom& random)
{
    WaxmanConfig config;
    config.num_nodes = num_nodes;
    config.beta = beta;
    config.alpha = alpha;
    return WaxmanGenerator::generate(
        config,
        random);
}

} // namespace nx
