#include <iostream>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

struct NodeData
{
    int cpu = 0;
    int ram = 0;
};

struct EdgeData
{
    double bw = 0.0;
    double weight = 1.0;
};

using Graph =
    boost::adjacency_list<
        boost::vecS,
        boost::vecS,
        boost::undirectedS,
        NodeData,
        EdgeData>;

int main()
{
    Graph g;

    auto n0 = boost::add_vertex(g);
    auto n1 = boost::add_vertex(g);
    auto n2 = boost::add_vertex(g);
    auto n3 = boost::add_vertex(g);

    g[n0].cpu = 100;
    g[n1].cpu = 80;
    g[n2].cpu = 60;
    g[n3].cpu = 40;

    auto [e01, ok01] = boost::add_edge(n0, n1, g);
    auto [e12, ok12] = boost::add_edge(n1, n2, g);
    auto [e23, ok23] = boost::add_edge(n2, n3, g);
    auto [e03, ok03] = boost::add_edge(n0, n3, g);

    g[e01].weight = 1.0;
    g[e12].weight = 2.0;
    g[e23].weight = 1.0;
    g[e03].weight = 10.0;

    g[e01].bw = 100;
    g[e12].bw = 80;
    g[e23].bw = 60;
    g[e03].bw = 20;

    std::cout << "Vertices: "
              << boost::num_vertices(g)
              << '\n';

    std::cout << "Edges: "
              << boost::num_edges(g)
              << '\n';

    std::vector<double> distances(
        boost::num_vertices(g));

    boost::dijkstra_shortest_paths(
        g,
        n0,
        boost::distance_map(
            distances.data())
        .weight_map(
            boost::get(
                &EdgeData::weight,
                g)));

    std::cout << "\nDistances from node 0\n";

    for (size_t i = 0;
         i < distances.size();
         ++i)
    {
        std::cout
            << "0 -> "
            << i
            << " = "
            << distances[i]
            << '\n';
    }

    std::cout
        << "\nNode 0 CPU = "
        << g[n0].cpu
        << '\n';

    std::cout
        << "Edge(0,1) BW = "
        << g[e01].bw
        << '\n';

    return 0;
}