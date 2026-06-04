#include "shortest_simple_paths.h"

std::vector<std::vector<Vertex>>
shortest_simple_paths(
    const Graph& g,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr)
{
    auto result =
        yen_k_shortest_paths(
            g,
            source,
            target,
            k,
            weight_attr);

    std::vector<
        std::vector<Vertex>
    > paths;

    paths.reserve(
        result.size());

    for (const auto& p : result)
    {
        paths.push_back(
            p.path);
    }

    return paths;
}