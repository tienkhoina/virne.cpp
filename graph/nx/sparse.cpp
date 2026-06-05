#include "sparse.h"

namespace nx
{

SparseMatrix
adjacency_matrix(
    const Graph& g,
    std::string_view weight_attr)
{
    SparseMatrix A(
        g.num_nodes(),
        g.num_nodes());

    A.reserve(
        g.num_edges() * 2);

    const AttrId weight_id =
        g.attr_id(
            weight_attr);

    auto [it, end] =
        g.edges();

    for (; it != end; ++it)
    {
        const Edge e =
            *it;

        double w = 1.0;

        if (const AttrValue* value =
                g.edge_attrs(e)
                    .find(weight_id))
        {
            w =
                attr_to_double(
                    *value);
        }

        const Vertex u =
            g.source(e);

        const Vertex v =
            g.target(e);

        A.add(
            u,
            v,
            w);

        A.add(
            v,
            u,
            w);
    }

    return A;
}

SparseMatrix
attr_sparse_matrix(
    const Graph& g,
    AttrId attr_id)
{
    SparseMatrix A(
        g.num_nodes(),
        g.num_nodes());

    A.reserve(
        g.num_edges() * 2);

    auto [it, end] =
        g.edges();

    for (; it != end; ++it)
    {
        const Edge e =
            *it;

        const AttrValue* value =
            g.edge_attrs(e)
                .find(attr_id);

        if (!value)
        {
            continue;
        }

        const double w =
            attr_to_double(
                *value);

        const Vertex u =
            g.source(e);

        const Vertex v =
            g.target(e);

        A.add(
            u,
            v,
            w);

        A.add(
            v,
            u,
            w);
    }

    return A;
}

SparseMatrix
attr_sparse_matrix(
    const Graph& g,
    std::string_view name)
{
    return
        attr_sparse_matrix(
            g,
            g.attr_id(name));
}



}