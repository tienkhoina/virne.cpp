#pragma once

#include "../graph.h"
#include "waxman_generator.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nx
{

Graph path_graph(
    size_t num_nodes);

// Matches NetworkX's integer signature: star_graph(n) has n + 1 nodes,
// with vertex 0 as the hub.
Graph star_graph(
    size_t outer_nodes);

// Tuple labels from NetworkX are relabelled to contiguous row-major indices:
// vertex(i, j) == i * columns + j.
Graph grid_2d_graph(
    size_t rows,
    size_t columns,
    bool periodic = false);

// seed=None compatibility: consume the process-wide Python-compatible
// stream, just like NetworkX consumes Python's module-global random state.
Graph erdos_renyi_graph(
    size_t num_nodes,
    double probability);

Graph erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    uint64_t seed);

Graph erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    PyRandom& random);

DiGraph erdos_renyi_digraph(
    size_t num_nodes,
    double probability);

DiGraph erdos_renyi_digraph(
    size_t num_nodes,
    double probability,
    uint64_t seed);

DiGraph erdos_renyi_digraph(
    size_t num_nodes,
    double probability,
    PyRandom& random);

// Retry without copying an existing graph. The callback receives the
// zero-based attempt and returns a fresh candidate.
Graph connected_retry(
    const std::function<Graph(size_t)>& generator,
    size_t max_attempts = 10000);

Graph connected_erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    uint64_t seed,
    size_t max_attempts = 10000);

Graph connected_erdos_renyi_graph(
    size_t num_nodes,
    double probability,
    PyRandom& random,
    size_t max_attempts = 10000);

Graph connected_waxman_graph(
    const WaxmanConfig& config,
    size_t max_attempts = 10000);

Graph connected_waxman_graph(
    const WaxmanConfig& config,
    PyRandom& random,
    size_t max_attempts = 10000);

} // namespace nx
