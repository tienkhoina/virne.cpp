#pragma once

#include "../graph.h"

#include <cstddef>
#include <cstdint>

class PyRandom;

struct WaxmanConfig
{
    size_t num_nodes = 100;

    double beta = 0.4;
    double alpha = 0.1;

    uint64_t seed = 42;
};

class WaxmanGenerator
{
public:

    static Graph generate(
        const WaxmanConfig& cfg);

    // Uses the caller-owned Python-compatible stream. This overload is what
    // connected retry loops need: failed candidates consume randomness and
    // the next attempt continues from that state, like NetworkX seed=None.
    static Graph generate(
        const WaxmanConfig& cfg,
        PyRandom& random);
};

namespace nx
{

Graph waxman_graph(
    size_t num_nodes,
    double beta = 0.4,
    double alpha = 0.1);

Graph waxman_graph(
    size_t num_nodes,
    double beta,
    double alpha,
    PyRandom& random);

} // namespace nx
