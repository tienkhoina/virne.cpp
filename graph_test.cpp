#include <iostream>

#include "graph/generators/gml_loader.h"

int main()
{
    auto g =
        GmlLoader::load(
            "/workspace/C++/data/Brain.gml");

    std::cout
        << "nodes = "
        << g.num_nodes()
        << '\n';

    std::cout
        << "edges = "
        << g.num_edges()
        << '\n';
}