#include "gml_loader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

Graph
GmlLoader::load(
    const std::string& path)
{
    Graph g;

    std::ifstream fin(path);

    if (!fin)
    {
        throw std::runtime_error(
            "cannot open gml file");
    }

    std::unordered_map<
        int,
        Vertex> id_to_vertex;

    std::string line;

    while (std::getline(fin, line))
    {
        //
        // node
        //
        if (line.find("node") !=
            std::string::npos)
        {
            int node_id = -1;

            while (std::getline(fin, line))
            {
                if (line.find(']') !=
                    std::string::npos)
                {
                    break;
                }

                std::stringstream ss(line);

                std::string key;

                ss >> key;

                if (key == "id")
                {
                    ss >> node_id;
                }
            }

            if (node_id >= 0)
            {
                auto v =
                    g.add_node();

                id_to_vertex[
                    node_id] = v;
            }
        }

        //
        // edge
        //
        else if (
            line.find("edge") !=
            std::string::npos)
        {
            int source = -1;
            int target = -1;

            while (std::getline(fin, line))
            {
                if (line.find(']') !=
                    std::string::npos)
                {
                    break;
                }

                std::stringstream ss(line);

                std::string key;

                ss >> key;

                if (key == "source")
                {
                    ss >> source;
                }
                else if (
                    key == "target")
                {
                    ss >> target;
                }
            }

            if (source >= 0 &&
                target >= 0)
            {
                g.add_edge(
                    id_to_vertex.at(
                        source),
                    id_to_vertex.at(
                        target));
            }
        }
    }

    return g;
}