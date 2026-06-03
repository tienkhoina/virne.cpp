#pragma once

#include "../graph.h"

#include <cstddef>
#include <cstdint>

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
};