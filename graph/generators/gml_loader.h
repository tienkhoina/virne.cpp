#pragma once

#include "../graph.h"

#include <string>
#include <variant>

using LoadedGraph =
    std::variant<Graph, DiGraph>;

class GmlLoader
{
public:

    static Graph load(
        const std::string& path,
        const std::string& label = "id");

    // GML has a directed flag but C++ cannot overload load() by return type.
    // Keep load() source-compatible for Graph and expose the matching
    // directed constructor under an explicit name.
    static DiGraph load_directed(
        const std::string& path,
        const std::string& label = "id");

    static LoadedGraph load_auto(
        const std::string& path,
        const std::string& label = "id");
};

namespace nx
{

Graph read_gml(
    const std::string& path,
    const std::string& label = "id");

DiGraph read_gml_directed(
    const std::string& path,
    const std::string& label = "id");

LoadedGraph read_gml_auto(
    const std::string& path,
    const std::string& label = "id");

}
