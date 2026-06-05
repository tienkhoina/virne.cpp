#pragma once

#include "../graph.h"

#include <unordered_map>

namespace nx
{

std::unordered_map<
    Vertex,
    AttrValue>
get_node_attributes(
    const Graph& g,
    const std::string& name);

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    const std::string& name);


std::unordered_map<
    uint32_t,
    AttrValue>
get_edge_attributes(
    const Graph& g,
    const std::string& name);

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    const std::string& name);

}

