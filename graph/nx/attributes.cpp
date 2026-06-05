#include "attributes.h"

namespace nx
{

std::unordered_map<
    Vertex,
    AttrValue>
get_node_attributes(
    const Graph& g,
    const std::string& name)
{
    std::unordered_map<
        Vertex,
        AttrValue> result;

    auto [begin, end] =
        g.nodes();

    for (auto it = begin;
         it != end;
         ++it)
    {
        Vertex v = *it;

        const auto& attrs =
            g.node_attrs(v);

        auto attr_it =
            attrs.find(name);

        if (attr_it != attrs.end())
        {
            result.emplace(
                v,
                attr_it->second);
        }
    }

    return result;
}

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    const std::string& name)
{
    for (const auto& kv : values)
    {
        g.node_attrs(
            kv.first)[name] =
                kv.second;
    }
}

}