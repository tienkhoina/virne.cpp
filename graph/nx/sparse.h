#pragma once

#include <string_view>
#include "../graph.h"
#include "../sparse_matrix.h"

namespace nx
{

SparseMatrix
adjacency_matrix(
    const Graph& g,
    std::string_view weight_attr =
        "weight");

SparseMatrix
attr_sparse_matrix(
    const Graph& g,
    AttrId attr_id);

SparseMatrix
attr_sparse_matrix(
    const Graph& g,
    std::string_view name);

}

