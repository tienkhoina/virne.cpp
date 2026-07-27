#pragma once

#include "../graph.h"
#include "search_mask.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

struct DijkstraResult
{
    std::vector<double> distance;
    std::vector<Vertex> predecessor;

    // Vertices in final heap-settlement order. Equal distances retain the
    // neighbor-discovery insertion order used by NetworkX's heap counter.
    std::vector<Vertex> settled_order;
};

using EdgeKey = std::pair<Vertex, Vertex>;

struct EdgeKeyHash
{
    std::size_t operator()(const EdgeKey& p) const noexcept
    {
        std::size_t h1 = std::hash<Vertex>{}(p.first);
        std::size_t h2 = std::hash<Vertex>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

using VertexSet = std::unordered_set<Vertex>;
using EdgeSet = std::unordered_set<EdgeKey, EdgeKeyHash>;

inline EdgeKey normalize_edge_key(
    Vertex u,
    Vertex v) noexcept
{
    return (u < v) ? EdgeKey{u, v} : EdgeKey{v, u};
}

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra_with_cutoff(
    const Graph& g,
    Vertex source,
    std::optional<double> cutoff,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra_with_cutoff(
    const DiGraph& g,
    Vertex source,
    std::optional<double> cutoff,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra_with_cutoff(
    const Graph& g,
    Vertex source,
    const SearchMask& mask,
    std::optional<double> cutoff,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra_with_cutoff(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask,
    std::optional<double> cutoff,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const SearchMask& mask,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const Graph& g,
    Vertex source,
    const SearchMask& mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");

DijkstraResult dijkstra(
    const DiGraph& g,
    Vertex source,
    const SearchMask& mask,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr = "weight");

double edge_cost(
    const Graph& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr = "weight");

double edge_cost(
    const DiGraph& g,
    Vertex u,
    Vertex v,
    const std::string& weight_attr = "weight");

double path_cost(
    const Graph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr = "weight");

double path_cost(
    const DiGraph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr = "weight");

std::vector<double> path_prefix_costs(
    const Graph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr = "weight");

std::vector<double> path_prefix_costs(
    const DiGraph& g,
    const std::vector<Vertex>& path,
    const std::string& weight_attr = "weight");

std::vector<Vertex> build_path(
    const DijkstraResult& result,
    Vertex source,
    Vertex target);
