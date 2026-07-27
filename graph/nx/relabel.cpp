#include "relabel.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

void copy_attributes(
    const AttrMap& source,
    AttrMap& target)
{
    for (const AttrId id :
         source.attribute_ids())
    {
        target.set(
            id,
            clone_attr_value(source.at(id)));
    }
}

template <typename GraphType>
std::vector<Vertex> make_mapping(
    const GraphType& graph,
    const std::string& ordering)
{
    std::vector<Vertex> old_nodes(
        graph.num_nodes());
    std::iota(
        old_nodes.begin(),
        old_nodes.end(),
        Vertex{0});

    if (ordering == "default" ||
        ordering == "sorted")
    {
        // Node insertion order and sorted order are both 0..N-1 for the
        // contiguous-index core.
    }
    else if (ordering ==
             "increasing degree" ||
             ordering ==
             "decreasing degree")
    {
        std::vector<
            std::pair<size_t, Vertex>>
            degree_nodes;
        degree_nodes.reserve(
            graph.num_nodes());
        for (const Vertex node : old_nodes)
        {
            degree_nodes.emplace_back(
                graph.degree(node),
                node);
        }
        std::sort(
            degree_nodes.begin(),
            degree_nodes.end());
        if (ordering ==
            "decreasing degree")
        {
            std::reverse(
                degree_nodes.begin(),
                degree_nodes.end());
        }
        for (size_t index = 0;
             index < degree_nodes.size();
             ++index)
        {
            old_nodes[index] =
                degree_nodes[index].second;
        }
    }
    else
    {
        throw std::invalid_argument(
            "Unknown node ordering: " + ordering);
    }

    std::vector<Vertex> old_to_new(
        graph.num_nodes());
    for (Vertex new_node = 0;
         new_node < old_nodes.size();
         ++new_node)
    {
        old_to_new[old_nodes[new_node]] =
            new_node;
    }
    return old_to_new;
}

template <typename GraphType>
GraphType convert_impl(
    const GraphType& graph,
    int64_t first_label,
    const std::string& ordering,
    const std::optional<std::string>&
        label_attribute)
{
    if (first_label != 0)
    {
        throw std::invalid_argument(
            "The contiguous-index graph core requires first_label == 0");
    }

    const std::vector<Vertex> mapping =
        make_mapping(graph, ordering);
    std::vector<Vertex> inverse(
        graph.num_nodes());
    for (Vertex old_node = 0;
         old_node < mapping.size();
         ++old_node)
    {
        inverse[mapping[old_node]] =
            old_node;
    }

    GraphType result;
    for (AttrId id = 0;
         id < graph.attribute_registry().size();
         ++id)
    {
        static_cast<void>(
            result.attr_id(
                graph.attr_name(id)));
    }
    copy_attributes(
        graph.graph_attrs(),
        result.graph_attrs());

    const std::optional<AttrId> label_attribute_id =
        label_attribute
        ? std::optional<AttrId>{
              result.attr_id(*label_attribute)}
        : std::nullopt;

    for (Vertex new_node = 0;
         new_node < inverse.size();
         ++new_node)
    {
        result.add_node();
        const Vertex old_node =
            inverse[new_node];
        copy_attributes(
            graph.node_attrs(old_node),
            result.node_attrs(new_node));
        if (label_attribute_id)
        {
            result.node_attrs(new_node).set(
                *label_attribute_id,
                static_cast<int64_t>(
                    old_node));
        }
    }

    auto [edge, edge_end] = graph.edges();
    for (; edge != edge_end; ++edge)
    {
        const auto source_edge = *edge;
        const Vertex u =
            mapping[graph.source(source_edge)];
        const Vertex v =
            mapping[graph.target(source_edge)];
        const auto target_edge =
            result.add_edge(u, v);
        copy_attributes(
            graph.edge_attrs(source_edge),
            result.edge_attrs(target_edge));
    }
    return result;
}

} // namespace

namespace nx
{

Graph convert_node_labels_to_integers(
    const Graph& graph,
    int64_t first_label,
    const std::string& ordering,
    std::optional<std::string> label_attribute)
{
    return convert_impl(
        graph,
        first_label,
        ordering,
        label_attribute);
}

DiGraph convert_node_labels_to_integers(
    const DiGraph& graph,
    int64_t first_label,
    const std::string& ordering,
    std::optional<std::string> label_attribute)
{
    return convert_impl(
        graph,
        first_label,
        ordering,
        label_attribute);
}

} // namespace nx
