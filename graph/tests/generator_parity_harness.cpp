#include "generators/topology_generators.h"
#include "generators/waxman_generator.h"
#include "graph.h"
#include "py_random.h"
#include "random_context.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

template <typename GraphType>
void dump_graph(
    const GraphType& graph)
{
    std::cout << "NODES";
    auto [node, node_end] = graph.nodes();
    for (; node != node_end; ++node)
    {
        std::cout << ' ' << *node;
    }
    std::cout << '\n';

    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        std::cout
            << "EDGE "
            << graph.source(*edge) << ' '
            << graph.target(*edge) << '\n';
    }

    for (Vertex v = 0;
         v < graph.num_nodes();
         ++v)
    {
        if (const AttrValue* value =
                graph.node_attrs(v).find("pos"))
        {
            const AttrList* position =
                attr_list(*value);
            if (position == nullptr ||
                position->values.size() != 2)
            {
                throw std::runtime_error(
                    "Invalid Waxman pos attribute");
            }

            std::cout
                << std::setprecision(17)
                << "POS " << v << ' '
                << attr_to_double(position->values[0]) << ' '
                << attr_to_double(position->values[1]) << '\n';
        }
    }

    size_t node_attributes = 0;
    size_t edge_attributes = 0;
    for (Vertex v = 0;
         v < graph.num_nodes();
         ++v)
    {
        node_attributes +=
            graph.node_attrs(v).size();
    }
    auto [attribute_edge, attribute_edge_end] =
        graph.edges();
    for (; attribute_edge != attribute_edge_end;
         ++attribute_edge)
    {
        edge_attributes +=
            graph.edge_attrs(*attribute_edge).size();
    }

    std::cout
        << "ATTR_COUNTS "
        << node_attributes << ' '
        << edge_attributes << '\n';
}

size_t parse_size(
    const char* value)
{
    return static_cast<size_t>(
        std::stoull(value));
}

uint64_t parse_seed(
    const char* value)
{
    return static_cast<uint64_t>(
        std::stoull(value));
}

} // namespace

int main(
    int argc,
    char** argv)
{
    try
    {
        if (argc < 2)
        {
            throw std::invalid_argument(
                "generator name is required");
        }

        const std::string name = argv[1];
        if ((name == "erdos_global" ||
             name == "erdos_di_global") &&
            argc == 5)
        {
            const size_t n = parse_size(argv[2]);
            const double p = std::stod(argv[3]);
            global_py_random().seed(parse_seed(argv[4]));
            if (name == "erdos_global")
            {
                dump_graph(nx::erdos_renyi_graph(n, p));
                std::cout << "NEXT\n";
                dump_graph(nx::erdos_renyi_graph(n, p));
            }
            else
            {
                dump_graph(nx::erdos_renyi_digraph(n, p));
                std::cout << "NEXT\n";
                dump_graph(nx::erdos_renyi_digraph(n, p));
            }
        }
        else if ((name == "erdos_stream" ||
             name == "erdos_di_stream") &&
            argc == 5)
        {
            const size_t n = parse_size(argv[2]);
            const double p = std::stod(argv[3]);
            PyRandom random(parse_seed(argv[4]));
            if (name == "erdos_stream")
            {
                dump_graph(nx::erdos_renyi_graph(n, p, random));
                std::cout << "NEXT\n";
                dump_graph(nx::erdos_renyi_graph(n, p, random));
            }
            else
            {
                dump_graph(nx::erdos_renyi_digraph(n, p, random));
                std::cout << "NEXT\n";
                dump_graph(nx::erdos_renyi_digraph(n, p, random));
            }
        }
        else if (name == "connected_erdos_stream" &&
                 argc == 6)
        {
            PyRandom random(parse_seed(argv[4]));
            dump_graph(nx::connected_erdos_renyi_graph(
                parse_size(argv[2]), std::stod(argv[3]),
                random, parse_size(argv[5])));
            std::cout << "NEXT\n";
            dump_graph(nx::connected_erdos_renyi_graph(
                parse_size(argv[2]), std::stod(argv[3]),
                random, parse_size(argv[5])));
        }
        else if ((name == "waxman_stream" && argc == 6) ||
                 (name == "connected_waxman_stream" &&
                  argc == 7))
        {
            WaxmanConfig config;
            config.num_nodes = parse_size(argv[2]);
            config.alpha = std::stod(argv[3]);
            config.beta = std::stod(argv[4]);
            PyRandom random(parse_seed(argv[5]));

            if (name == "waxman_stream")
            {
                dump_graph(nx::waxman_graph(
                    config.num_nodes,
                    config.beta,
                    config.alpha,
                    random));
                std::cout << "NEXT\n";
                dump_graph(nx::waxman_graph(
                    config.num_nodes,
                    config.beta,
                    config.alpha,
                    random));
            }
            else
            {
                const size_t attempts = parse_size(argv[6]);
                dump_graph(nx::connected_waxman_graph(
                    config, random, attempts));
                std::cout << "NEXT\n";
                dump_graph(nx::connected_waxman_graph(
                    config, random, attempts));
            }
        }
        else if (name == "path" && argc == 3)
        {
            dump_graph(nx::path_graph(
                parse_size(argv[2])));
        }
        else if (name == "star" && argc == 3)
        {
            dump_graph(nx::star_graph(
                parse_size(argv[2])));
        }
        else if ((name == "grid" ||
                  name == "grid_periodic") &&
                 argc == 4)
        {
            if (name == "grid")
            {
                dump_graph(nx::grid_2d_graph(
                    parse_size(argv[2]),
                    parse_size(argv[3])));
            }
            else
            {
                dump_graph(nx::grid_2d_graph(
                    parse_size(argv[2]),
                    parse_size(argv[3]),
                    true));
            }
        }
        else if (name == "waxman_global" &&
                 argc == 4)
        {
            global_py_random().seed(parse_seed(argv[3]));
            dump_graph(nx::waxman_graph(
                parse_size(argv[2])));
            std::cout << "NEXT\n";
            dump_graph(nx::waxman_graph(
                parse_size(argv[2])));
        }
        else if ((name == "erdos" ||
                  name == "erdos_di") &&
                 argc == 5)
        {
            const size_t n = parse_size(argv[2]);
            const double p = std::stod(argv[3]);
            const uint64_t seed = parse_seed(argv[4]);
            if (name == "erdos")
            {
                dump_graph(
                    nx::erdos_renyi_graph(
                        n, p, seed));
            }
            else
            {
                dump_graph(
                    nx::erdos_renyi_digraph(
                        n, p, seed));
            }
        }
        else if (name == "connected_erdos" &&
                 argc == 6)
        {
            dump_graph(
                nx::connected_erdos_renyi_graph(
                    parse_size(argv[2]),
                    std::stod(argv[3]),
                    parse_seed(argv[4]),
                    parse_size(argv[5])));
        }
        else if ((name == "waxman" && argc == 6) ||
                 (name == "connected_waxman" &&
                  argc == 7))
        {
            WaxmanConfig config;
            config.num_nodes = parse_size(argv[2]);
            config.alpha = std::stod(argv[3]);
            config.beta = std::stod(argv[4]);
            config.seed = parse_seed(argv[5]);

            if (name == "waxman")
            {
                dump_graph(
                    WaxmanGenerator::generate(config));
            }
            else
            {
                dump_graph(
                    nx::connected_waxman_graph(
                        config,
                        parse_size(argv[6])));
            }
        }
        else
        {
            throw std::invalid_argument(
                "invalid generator arguments");
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }

    return 0;
}
