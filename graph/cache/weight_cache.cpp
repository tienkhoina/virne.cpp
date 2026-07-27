#include "weight_cache.h"

#include <algorithm>
#include <stdexcept>

namespace
{

double to_double(const AttrValue& value)
{
    return attr_to_double(value);
}

template <typename GraphType>
size_t edge_capacity(const GraphType& graph)
{
    size_t capacity = 0;
    auto [it, end] = graph.edges();
    for (; it != end; ++it)
    {
        capacity = std::max(
            capacity,
            static_cast<size_t>(graph.edge_id(*it)) + 1);
    }
    return capacity;
}

template <typename GraphType>
std::vector<double> make_values(
    const GraphType& graph,
    size_t edge_capacity_value,
    size_t attr_count)
{
    std::vector<double> values(edge_capacity_value * attr_count, 0.0);
    if (attr_count == 0)
    {
        return values;
    }

    auto [it, end] = graph.edges();
    for (; it != end; ++it)
    {
        const auto e = *it;
        const size_t edge_id = graph.edge_id(e);
        const auto& attrs = graph.edge_attrs(e);
        for (AttrId attr_id : attrs.attribute_ids())
        {
            const AttrValue* value = attrs.find(attr_id);
            if (value != nullptr)
            {
                values[edge_id * attr_count + attr_id] = to_double(*value);
            }
        }
    }
    return values;
}

double checked_value(
    const std::vector<double>& values,
    size_t edge_capacity_value,
    size_t attr_count,
    uint32_t edge_id,
    AttrId attr_id)
{
    if (edge_id >= edge_capacity_value || attr_id >= attr_count)
    {
        throw std::out_of_range(
            "WeightCache is a snapshot; rebuild it after graph/attribute changes");
    }
    return values[static_cast<size_t>(edge_id) * attr_count + attr_id];
}

} // namespace

WeightCache::WeightCache(const Graph& g)
    : graph_(&g),
      edge_capacity_(edge_capacity(g)),
      attr_count_(g.attribute_registry().size()),
      values_(make_values(g, edge_capacity_, attr_count_))
{
}

AttrId WeightCache::attribute_id(const std::string& name) const
{
    const auto id = graph_->attribute_registry().find(name);
    if (!id || *id >= attr_count_)
    {
        throw std::out_of_range("attribute is not present in WeightCache snapshot");
    }
    return *id;
}

double WeightCache::value(Vertex u, Vertex v, AttrId attr_id) const
{
    const Edge e = graph_->edge(u, v);
    return checked_value(
        values_, edge_capacity_, attr_count_, graph_->edge_id(e), attr_id);
}

DiWeightCache::DiWeightCache(const DiGraph& g)
    : graph_(&g),
      edge_capacity_(edge_capacity(g)),
      attr_count_(g.attribute_registry().size()),
      values_(make_values(g, edge_capacity_, attr_count_))
{
}

AttrId DiWeightCache::attribute_id(const std::string& name) const
{
    const auto id = graph_->attribute_registry().find(name);
    if (!id || *id >= attr_count_)
    {
        throw std::out_of_range("attribute is not present in DiWeightCache snapshot");
    }
    return *id;
}

double DiWeightCache::value(Vertex u, Vertex v, AttrId attr_id) const
{
    const DiEdge e = graph_->edge(u, v);
    return checked_value(
        values_,
        edge_capacity_,
        attr_count_,
        graph_->edge_id(e),
        attr_id);
}
