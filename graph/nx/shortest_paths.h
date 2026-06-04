#pragma once

#include "../graph.h"
#include "../sparse_matrix.h"

#include <unordered_map>
#include <vector>
#include <string>


namespace nx
{

size_t shortest_path_length(
    const Graph& g,
    Vertex source,
    Vertex target);

std::vector<Vertex> shortest_path(
    const Graph& g,
    Vertex source,
    Vertex target);

std::unordered_map<Vertex, size_t>
single_source_shortest_path_length(
    const Graph& g,
    Vertex source);


std::vector<Vertex> dijkstra_path(
    const Graph& g,
    Vertex source,
    Vertex target,
    const std::string& weight_attr =
        "weight");

double dijkstra_path_length(
    const Graph& g,
    Vertex source,
    Vertex target,
    const std::string& weight_attr =
        "weight");

std::unordered_map<
    Vertex,
    double>
single_source_dijkstra_path_length(
    const Graph& g,
    Vertex source,
    const std::string& weight_attr =
        "weight");

DistanceMatrix floyd_warshall(
    const Graph& g,
    const std::string& weight_attr =
        "weight");

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "weight");


}
