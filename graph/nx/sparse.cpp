#include "sparse.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace
{

void sort_coo_like_scipy(
    SparseMatrix& matrix)
{
    std::vector<size_t> order(
        matrix.nnz());
    std::iota(
        order.begin(),
        order.end(),
        size_t{0});

    std::stable_sort(
        order.begin(),
        order.end(),
        [&matrix](size_t lhs, size_t rhs)
        {
            if (matrix.row[lhs] != matrix.row[rhs])
            {
                return matrix.row[lhs] < matrix.row[rhs];
            }
            return matrix.col[lhs] < matrix.col[rhs];
        });

    std::vector<size_t> row;
    std::vector<size_t> col;
    std::vector<double> value;
    row.reserve(order.size());
    col.reserve(order.size());
    value.reserve(order.size());

    for (size_t index : order)
    {
        row.push_back(matrix.row[index]);
        col.push_back(matrix.col[index]);
        value.push_back(matrix.value[index]);
    }

    matrix.row = std::move(row);
    matrix.col = std::move(col);
    matrix.value = std::move(value);
}

} // namespace

namespace
{

template <typename GraphType, typename EdgeType>
SparseMatrix adjacency_matrix_impl(
    const GraphType& g,
    std::string_view weight_attr,
    bool directed)
{
    if (g.num_nodes() == 0)
    {
        throw std::invalid_argument(
            "adjacency_matrix is undefined for an empty graph");
    }
    SparseMatrix matrix(g.num_nodes(), g.num_nodes());
    matrix.reserve(g.num_edges() * (directed ? 1 : 2));

    const AttrId weight_id = g.attr_id(weight_attr);
    auto [it, end] = g.edges();

    for (; it != end; ++it)
    {
        const EdgeType e = *it;
        double weight = 1.0;

        if (const AttrValue* value =
                g.edge_attrs(e).find(weight_id))
        {
            weight = attr_to_double(*value);
        }

        const Vertex u = g.source(e);
        const Vertex v = g.target(e);
        matrix.add(u, v, weight);

        if (!directed && u != v)
        {
            matrix.add(v, u, weight);
        }
    }

    // NetworkX first builds a SciPy CSR array.  Its public .tocoo() view is
    // row-major with sorted column indices, not global Boost edge order.
    sort_coo_like_scipy(matrix);
    return matrix;
}

template <typename GraphType, typename EdgeType>
SparseMatrix attr_sparse_matrix_impl(
    const GraphType& g,
    AttrId attr_id,
    bool directed,
    bool normalized,
    const std::vector<Vertex>& rc_order)
{
    const size_t n = g.num_nodes();
    const bool missing_defaults_to_one =
        g.attr_name(attr_id) == "weight";

    std::vector<size_t> position(
        n,
        std::numeric_limits<size_t>::max());

    if (rc_order.empty())
    {
        for (Vertex v = 0; v < n; ++v)
        {
            position[v] = v;
        }
    }
    else
    {
        if (rc_order.size() != n)
        {
            throw std::invalid_argument(
                "rc_order must contain every graph vertex exactly once");
        }

        for (size_t row = 0;
             row < rc_order.size();
             ++row)
        {
            const Vertex v = rc_order[row];
            if (v >= n ||
                position[v] !=
                    std::numeric_limits<size_t>::max())
            {
                throw std::invalid_argument(
                    "rc_order must be a permutation of graph vertices");
            }
            position[v] = row;
        }
    }

    SparseMatrix matrix(n, n);
    matrix.reserve(g.num_edges() * (directed ? 1 : 2));

    auto [it, end] = g.edges();
    for (; it != end; ++it)
    {
        const EdgeType e = *it;
        const AttrValue* value =
            g.edge_attrs(e).find(attr_id);

        if (value == nullptr)
        {
            if (!missing_defaults_to_one)
            {
                throw std::out_of_range(
                    "Requested edge attribute is missing");
            }
        }

        const double weight = value == nullptr
            ? 1.0
            : attr_to_double(*value);
        // NetworkX accumulates into a SciPy LIL array. Adding an exact zero
        // does not create a stored coordinate, which is observable during
        // row normalization: an all-zero row stays sparse zero instead of
        // producing an explicit 0/0 NaN.
        if (weight == 0.0)
        {
            continue;
        }
        const Vertex u = g.source(e);
        const Vertex v = g.target(e);
        matrix.add(
            position[u],
            position[v],
            weight);

        if (!directed && u != v)
        {
            matrix.add(
                position[v],
                position[u],
                weight);
        }
    }

    // NetworkX accumulates through SciPy LIL.  LIL rows keep column indices
    // sorted, so the observable floating-point row sum is in column order,
    // not Boost's edge-insertion order.  Sort before summing as well as before
    // exposing the COO arrays.
    sort_coo_like_scipy(matrix);

    if (normalized)
    {
        std::vector<double> row_sum(n, 0.0);
        for (size_t i = 0;
             i < matrix.nnz();
             ++i)
        {
            row_sum[matrix.row[i]] +=
                matrix.value[i];
        }

        std::vector<double> row_scale(n, 0.0);
        for (size_t row = 0;
             row < n;
             ++row)
        {
            // Match NetworkX/SciPy's two-step floating-point operation:
            // scale = 1 / row_sum, followed by element-wise multiply.
            row_scale[row] = 1.0 / row_sum[row];
        }

        for (size_t i = 0;
             i < matrix.nnz();
             ++i)
        {
            matrix.value[i] *=
                row_scale[matrix.row[i]];
        }
    }

    return matrix;
}

} // namespace

namespace nx
{

SparseMatrix adjacency_matrix(
    const Graph& g,
    std::string_view weight_attr)
{
    return adjacency_matrix_impl<Graph, Edge>(
        g, weight_attr, false);
}

SparseMatrix adjacency_matrix(
    const DiGraph& g,
    std::string_view weight_attr)
{
    return adjacency_matrix_impl<DiGraph, DiEdge>(
        g, weight_attr, true);
}

CSRMatrix to_scipy_sparse_matrix(
    const Graph& g,
    std::string_view weight_attr,
    std::string_view format)
{
    if (format != "csr")
    {
        throw std::invalid_argument(
            "to_scipy_sparse_matrix supports only format='csr'");
    }
    return adjacency_matrix(g, weight_attr).to_csr();
}

CSRMatrix to_scipy_sparse_matrix(
    const DiGraph& g,
    std::string_view weight_attr,
    std::string_view format)
{
    if (format != "csr")
    {
        throw std::invalid_argument(
            "to_scipy_sparse_matrix supports only format='csr'");
    }
    return adjacency_matrix(g, weight_attr).to_csr();
}

SparseMatrix attr_sparse_matrix(
    const Graph& g,
    AttrId attr_id,
    bool normalized,
    const std::vector<Vertex>& rc_order)
{
    return attr_sparse_matrix_impl<Graph, Edge>(
        g, attr_id, false,
        normalized, rc_order);
}

SparseMatrix attr_sparse_matrix(
    const DiGraph& g,
    AttrId attr_id,
    bool normalized,
    const std::vector<Vertex>& rc_order)
{
    return attr_sparse_matrix_impl<DiGraph, DiEdge>(
        g, attr_id, true,
        normalized, rc_order);
}

SparseMatrix attr_sparse_matrix(
    const Graph& g,
    std::string_view name,
    bool normalized,
    const std::vector<Vertex>& rc_order)
{
    return attr_sparse_matrix(
        g, g.attr_id(name),
        normalized, rc_order);
}

SparseMatrix attr_sparse_matrix(
    const DiGraph& g,
    std::string_view name,
    bool normalized,
    const std::vector<Vertex>& rc_order)
{
    return attr_sparse_matrix(
        g, g.attr_id(name),
        normalized, rc_order);
}

} // namespace nx
