#pragma once

#include "../graph.h"
#include "k_shortest_paths.h"

#include <string>
#include <vector>

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "weight");