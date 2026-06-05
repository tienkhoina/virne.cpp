#pragma once

#include <string>

class Graph;

namespace GraphSaver
{
    void save_gml(
        const Graph& graph,
        const std::string& path);
}