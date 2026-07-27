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

NodeScores degree_centrality(
    const DiGraph& g);

//
// Eigenvector
//

NodeScores
eigenvector_centrality(
    const Graph& g,
    size_t max_iter = 100,
    double tol = 1e-6);

NodeScores eigenvector_centrality(
    const DiGraph& g,
    size_t max_iter = 100,
    double tol = 1e-6);

//
// Closeness
//

NodeScores
closeness_centrality(
    const Graph& g);

NodeScores closeness_centrality(
    const DiGraph& g);

//
// Betweenness (Brandes)
//

NodeScores
betweenness_centrality(
    const Graph& g,
    const std::string& weight_attr =
        "");

NodeScores betweenness_centrality(
    const DiGraph& g,
    const std::string& weight_attr =
        "");

} // namespace nx
