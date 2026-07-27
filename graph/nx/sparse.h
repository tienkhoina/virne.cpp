#pragma once

#include <string_view>
#include <vector>
#include "../graph.h"
#include "../sparse_matrix.h"

namespace nx
{

SparseMatrix
adjacency_matrix(
    const Graph& g,
    std::string_view weight_attr =
        "weight");

SparseMatrix adjacency_matrix(
    const DiGraph& g,
    std::string_view weight_attr =
        "weight");

// Compatibility name used by the original Virne BaseNetwork. The Python
// call requests format="csr"; expose that exact useful mode without carrying
// SciPy dtype/nodelist dynamics into the C++ hot path.
CSRMatrix to_scipy_sparse_matrix(
    const Graph& g,
    std::string_view weight_attr = "weight",
    std::string_view format = "csr");

CSRMatrix to_scipy_sparse_matrix(
    const DiGraph& g,
    std::string_view weight_attr = "weight",
    std::string_view format = "csr");

SparseMatrix
attr_sparse_matrix(
    const Graph& g,
    AttrId attr_id,
    bool normalized = false,
    const std::vector<Vertex>& rc_order = {});

SparseMatrix attr_sparse_matrix(
    const DiGraph& g,
    AttrId attr_id,
    bool normalized = false,
    const std::vector<Vertex>& rc_order = {});

SparseMatrix
attr_sparse_matrix(
    const Graph& g,
    std::string_view name,
    bool normalized = false,
    const std::vector<Vertex>& rc_order = {});

SparseMatrix attr_sparse_matrix(
    const DiGraph& g,
    std::string_view name,
    bool normalized = false,
    const std::vector<Vertex>& rc_order = {});

}
