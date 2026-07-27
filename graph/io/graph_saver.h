#pragma once

#include <string>

class Graph;
class DiGraph;

namespace GraphSaver
{
    void save_gml(
        const Graph& graph,
        const std::string& path);

    void save_gml(
        const DiGraph& graph,
        const std::string& path);
}

namespace nx
{
    void write_gml(
        const Graph& graph,
        const std::string& path);

    void write_gml(
        const DiGraph& graph,
        const std::string& path);
}
