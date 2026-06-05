#pragma once

#include "../graph.h"
#include "../distance_matrix.h"

#include <string>

DistanceMatrix floyd_warshall(
    const Graph& g,
    const std::string& weight_attr =
        "weight");