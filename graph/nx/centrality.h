#pragma once

#include "../graph.h"

#include <string>
#include <vector>

namespace nx
{

using NodeScores =
    std::vector<double>;

//
// Degree
//

NodeScores
degree_centrality(
    const Graph& g);

//
// Eigenvector
//

NodeScores
eigenvector_centrality(
    const Graph& g,
    size_t max_iter = 10000,
    double tol = 1e-6);

//
// Closeness
//

NodeScores
closeness_centrality(
    const Graph& g);

//
// Betweenness (Brandes)
//

NodeScores
betweenness_centrality(
    const Graph& g,
    const std::string& weight_attr =
        "weight");

} // namespace nx