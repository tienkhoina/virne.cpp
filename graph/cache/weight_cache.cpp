#include "weight_cache.h"

#include <stdexcept>

namespace
{

double to_double(
    const AttrValue& value)
{
    if (std::holds_alternative<double>(
            value))
    {
        return std::get<double>(
            value);
    }

    if (std::holds_alternative<int64_t>(
            value))
    {
        return static_cast<double>(
            std::get<int64_t>(
                value));
    }

    return 0.0;
}

} // namespace

WeightCache::WeightCache(
    const Graph& g)
    :
    graph_(&g),
    edge_count_(
        g.num_edges()),
    attr_count_(
        g.attribute_registry()
            .size())
{
    values_.assign(
        edge_count_ *
            attr_count_,
        0.0);

    auto [eit, eend] =
        g.edges();

    for (; eit != eend; ++eit)
    {
        const Edge e =
            *eit;

        const uint32_t edge_id =
            g.edge_id(e);

        const auto& attrs =
            g.edge_attrs(e);

        for (const AttrId attr_id :
             attrs.attribute_ids())
        {
            const AttrValue* value =
                attrs.find(attr_id);

            if (value == nullptr)
            {
                continue;
            }

            values_[
                static_cast<size_t>(
                    edge_id)
                    *
                    attr_count_
                +
                attr_id] =
                to_double(
                    *value);
        }
    }
}

AttrId WeightCache::attribute_id(
    const std::string& name) const
{
    return graph_->attr_id(
        name);
}

double WeightCache::value(
    const RawNeighbor& edge,
    AttrId attr_id) const
{
    const uint32_t edge_id =
        edge.get_property()
            .edge_id;

    return values_[
        static_cast<size_t>(
            edge_id)
            *
            attr_count_
        +
        attr_id];
}

double WeightCache::value(
    Vertex u,
    Vertex v,
    AttrId attr_id) const
{
    const Edge e =
        graph_->edge(
            u,
            v);

    const uint32_t edge_id =
        graph_->edge_id(
            e);

    return values_[
        static_cast<size_t>(
            edge_id)
            *
            attr_count_
        +
        attr_id];
}