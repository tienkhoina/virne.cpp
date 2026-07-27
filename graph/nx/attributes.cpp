#include "attributes.h"

namespace
{

template <typename GraphType>
nx::NodeAttributeMap
get_node_attributes_impl(
    const GraphType& g,
    AttrId attr_id)
{
    nx::NodeAttributeMap result;
    result.reserve(g.num_nodes());

    auto [it, end] = g.nodes();
    for (; it != end; ++it)
    {
        const Vertex v = *it;
        if (const AttrValue* value = g.node_attrs(v).find(attr_id))
        {
            result.emplace(v, *value);
        }
    }
    return result;
}

template <typename GraphType>
nx::EdgeIdAttributeMap
get_edge_attributes_impl(
    const GraphType& g,
    AttrId attr_id)
{
    nx::EdgeIdAttributeMap result;
    result.reserve(g.num_edges());

    auto [it, end] = g.edges();
    for (; it != end; ++it)
    {
        const auto e = *it;
        if (const AttrValue* value = g.edge_attrs(e).find(attr_id))
        {
            result.emplace_with_id(
                EdgeEndpoints{g.source(e), g.target(e)},
                g.edge_id(e),
                *value);
        }
    }
    return result;
}

template <typename GraphType>
nx::EdgeAttributeMap
get_edge_attributes_endpoint_impl(
    const GraphType& g,
    AttrId attr_id)
{
    nx::EdgeAttributeMap result;
    result.reserve(g.num_edges());

    auto [it, end] = g.edges();
    for (; it != end; ++it)
    {
        const auto e = *it;
        if (const AttrValue* value = g.edge_attrs(e).find(attr_id))
        {
            result.emplace_with_id(
                EdgeEndpoints{g.source(e), g.target(e)},
                g.edge_id(e),
                *value);
        }
    }
    return result;
}

template <typename GraphType>
void set_node_attributes_impl(
    GraphType& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    AttrId attr_id)
{
    for (const auto& [v, value] : values)
    {
        // NetworkX ignores attribute entries for nodes that are not present;
        // checking here also prevents an out-of-range vecS access.
        if (v < g.num_nodes())
        {
            g.node_attrs(v).set(attr_id, value);
        }
    }
}

template <typename GraphType>
void set_edge_attributes_impl(
    GraphType& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id)
{
    for (const auto& [edge_id, value] : values)
    {
        const auto e = g.edge_by_id(edge_id);
        g.edge_attrs(e).set(attr_id, value);
    }
}

template <typename GraphType>
void set_edge_attributes_endpoint_impl(
    GraphType& g,
    const std::unordered_map<
        EdgeEndpoints,
        AttrValue,
        nx::EdgeEndpointHash>& values,
    AttrId attr_id)
{
    for (const auto& [endpoints, value] : values)
    {
        const auto [u, v] = endpoints;
        if (g.has_edge(u, v))
        {
            g.edge_attrs(g.edge(u, v)).set(attr_id, value);
        }
    }
}

} // namespace

namespace nx
{

NodeAttributeMap
get_node_attributes(const Graph& g, AttrId attr_id)
{
    return get_node_attributes_impl(g, attr_id);
}

NodeAttributeMap
get_node_attributes(const DiGraph& g, AttrId attr_id)
{
    return get_node_attributes_impl(g, attr_id);
}

NodeAttributeMap
get_node_attributes(const Graph& g, std::string_view name)
{
    const auto id = g.attribute_registry().find(name);
    return id ? get_node_attributes(g, *id)
              : NodeAttributeMap{};
}

NodeAttributeMap
get_node_attributes(const DiGraph& g, std::string_view name)
{
    const auto id = g.attribute_registry().find(name);
    return id ? get_node_attributes(g, *id)
              : NodeAttributeMap{};
}

EdgeIdAttributeMap
get_edge_attributes_by_id(const Graph& g, AttrId attr_id)
{
    return get_edge_attributes_impl(g, attr_id);
}

EdgeIdAttributeMap
get_edge_attributes_by_id(const DiGraph& g, AttrId attr_id)
{
    return get_edge_attributes_impl(g, attr_id);
}

EdgeIdAttributeMap
get_edge_attributes(const Graph& g, AttrId attr_id)
{
    return get_edge_attributes_by_id(g, attr_id);
}

EdgeIdAttributeMap
get_edge_attributes(const DiGraph& g, AttrId attr_id)
{
    return get_edge_attributes_by_id(g, attr_id);
}

EdgeAttributeMap
get_edge_attributes(const Graph& g, std::string_view name)
{
    const auto id = g.attribute_registry().find(name);
    return id ? get_edge_attributes_endpoint_impl(g, *id)
              : EdgeAttributeMap{};
}

EdgeAttributeMap
get_edge_attributes(const DiGraph& g, std::string_view name)
{
    const auto id = g.attribute_registry().find(name);
    return id ? get_edge_attributes_endpoint_impl(g, *id)
              : EdgeAttributeMap{};
}

void set_node_attributes(
    Graph& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    AttrId attr_id)
{
    set_node_attributes_impl(g, values, attr_id);
}

void set_node_attributes(
    DiGraph& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    AttrId attr_id)
{
    set_node_attributes_impl(g, values, attr_id);
}

void set_node_attributes(
    Graph& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    std::string_view name)
{
    set_node_attributes(g, values, g.attr_id(name));
}

void set_node_attributes(
    DiGraph& g,
    const std::unordered_map<Vertex, AttrValue>& values,
    std::string_view name)
{
    set_node_attributes(g, values, g.attr_id(name));
}

void set_edge_attributes_by_id(
    Graph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id)
{
    set_edge_attributes_impl(g, values, attr_id);
}

void set_edge_attributes_by_id(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id)
{
    set_edge_attributes_impl(g, values, attr_id);
}

void set_edge_attributes_by_id(
    Graph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name)
{
    set_edge_attributes_by_id(g, values, g.attr_id(name));
}

void set_edge_attributes_by_id(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name)
{
    set_edge_attributes_by_id(g, values, g.attr_id(name));
}

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id)
{
    set_edge_attributes_by_id(g, values, attr_id);
}

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    AttrId attr_id)
{
    set_edge_attributes_by_id(g, values, attr_id);
}

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name)
{
    set_edge_attributes_by_id(g, values, name);
}

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<uint32_t, AttrValue>& values,
    std::string_view name)
{
    set_edge_attributes_by_id(g, values, name);
}

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    AttrId attr_id)
{
    set_edge_attributes_endpoint_impl(g, values, attr_id);
}

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    AttrId attr_id)
{
    set_edge_attributes_endpoint_impl(g, values, attr_id);
}

void set_edge_attributes(
    Graph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    std::string_view name)
{
    set_edge_attributes(g, values, g.attr_id(name));
}

void set_edge_attributes(
    DiGraph& g,
    const std::unordered_map<EdgeEndpoints, AttrValue, EdgeEndpointHash>& values,
    std::string_view name)
{
    set_edge_attributes(g, values, g.attr_id(name));
}

} // namespace nx
