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
    Graph& g)
{
    //
    // build attribute registry
    //

    auto [eit, eend] =
        g.edges();

    for (; eit != eend; ++eit)
    {
        const auto& attrs =
            g.edge_attrs(
                *eit);

        for (const auto& kv :
             attrs)
        {
            if (attr_ids_.find(
                    kv.first)
                ==
                attr_ids_.end())
            {
                attr_ids_[
                    kv.first] =
                    attr_ids_.size();
            }
        }
    }

    const std::size_t attr_count =
        attr_ids_.size();

    edge_count_ =
        g.num_edges();

    values_.assign(
        edge_count_ *
            attr_count,
        0.0);

    //
    // assign edge ids
    //

    uint32_t edge_id = 0;

    std::tie(
        eit,
        eend) =
        g.edges();

    for (; eit != eend; ++eit)
    {
        Edge e =
            *eit;

        auto& prop =
            g.raw()[e];

        prop.edge_id =
            edge_id;

        const auto& attrs =
            prop.attrs;

        for (const auto& kv :
             attrs)
        {
            const std::size_t attr_id =
                attr_ids_.at(
                    kv.first);

            values_[
                edge_id *
                    attr_count
                +
                attr_id] =
                to_double(
                    kv.second);
        }

        ++edge_id;
    }
}

std::size_t WeightCache::attribute_id(
    const std::string& name) const
{
    auto it =
        attr_ids_.find(
            name);

    if (it == attr_ids_.end())
    {
        throw std::runtime_error(
            "Attribute not found");
    }

    return it->second;
}

double WeightCache::value(
    const RawNeighbor& edge,
    std::size_t attr_id) const
{
    const auto edge_id =
        edge.get_property()
            .edge_id;

    const std::size_t attr_count =
        attr_ids_.size();

    return values_[
        edge_id *
            attr_count
        +
        attr_id];
}

double WeightCache::value(
    Vertex u,
    Vertex v,
    std::size_t attr_id) const
{
    (void)u;
    (void)v;
    (void)attr_id;

    throw std::runtime_error(
        "Vertex lookup not implemented");
}