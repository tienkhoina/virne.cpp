#include "attributes.h"

namespace nx
{

std::unordered_map<
    Vertex,
    AttrValue>
get_node_attributes(
    const Graph& g,
    AttrId attr_id)
{
    std::unordered_map<
        Vertex,
        AttrValue> result;

    auto [it, end] =
        g.nodes();

    for (; it != end; ++it)
    {
        const Vertex v =
            *it;

        const AttrValue* value =
            g.node_attrs(v)
                .find(attr_id);

        if (value)
        {
            result.emplace(
                v,
                *value);
        }
    }

    return result;
}

std::unordered_map<
    Vertex,
    AttrValue>
get_node_attributes(
    const Graph& g,
    std::string_view name)
{
    const auto id =
        g.attribute_registry()
            .find(name);

    if (!id.has_value())
    {
        return {};
    }

    return get_node_attributes(
        g,
        *id);
}

std::unordered_map<
    uint32_t,
    AttrValue>
get_edge_attributes(
    const Graph& g,
    AttrId attr_id)
{
    std::unordered_map<
        uint32_t,
        AttrValue> result;

    auto [it, end] =
        g.edges();

    for (; it != end; ++it)
    {
        const Edge e =
            *it;

        const AttrValue* value =
            g.edge_attrs(e)
                .find(attr_id);

        if (value)
        {
            result.emplace(
                g.edge_id(e),
                *value);
        }
    }

    return result;
}

std::unordered_map<
    uint32_t,
    AttrValue>
get_edge_attributes(
    const Graph& g,
    std::string_view name)
{
    const auto id =
        g.attribute_registry()
            .find(name);

    if (!id.has_value())
    {
        return {};
    }

    return get_edge_attributes(
        g,
        *id);
}

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    AttrId attr_id)
{
    for (const auto& kv : values)
    {
        g.node_attrs(
            kv.first)
            .set(
                attr_id,
                kv.second);
    }
}

void set_node_attributes(
    Graph& g,
    const std::unordered_map<
        Vertex,
        AttrValue>& values,
    std::string_view name)
{
    const AttrId attr_id =
        g.attr_id(name);

    set_node_attributes(
        g,
        values,
        attr_id);
}

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    AttrId attr_id)
{
    for (const auto& kv : values)
    {
        const Edge e =
            g.edge_by_id(
                kv.first);

        g.edge_attrs(e)
            .set(
                attr_id,
                kv.second);
    }
}

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<
        uint32_t,
        AttrValue>& values,
    std::string_view name)
{
    const AttrId attr_id =
        g.attr_id(name);

    set_edge_attributes(
        g,
        values,
        attr_id);
}

}