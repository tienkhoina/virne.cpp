#pragma once

#include "../graph.h"

#include <string_view>
#include <unordered_map>

namespace nx
{

std::unordered_map<
    Vertex,
    AttrValue>
get_node_attributes(
    const Graph& g,
    AttrId attr_id);

std::unordered_map<
    Vertex,
    AttrValue>
get_node_attributes(
    const Graph& g,
    std::string_view name);

std::unordered_map<
    uint32_t,
    AttrValue>
get_edge_attributes(
    const Graph& g,
    AttrId attr_id);

std::unordered_map<
    uint32_t,
    AttrValue>
get_edge_attributes(
    const Graph& g,
    std::string_view name);

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    AttrId attr_id);

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    std::string_view name);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    AttrId attr_id);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    std::string_view name);

}

