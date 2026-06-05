#include "sparse_matrix.h"

#include <cassert>

CSRMatrix
SparseMatrix::to_csr() const
{
    assert(
        row.size()
        ==
        col.size());

    assert(
        row.size()
        ==
        value.size());

    CSRMatrix csr(
        rows,
        cols);

    const size_t nnz_ =
        nnz();

    csr.row_ptr.assign(
        rows + 1,
        0);

    csr.col_idx.resize(
        nnz_);

    csr.values.resize(
        nnz_);

    //
    // count entries per row
    //

    for (size_t r : row)
    {
        ++csr.row_ptr[r + 1];
    }

    //
    // prefix sum
    //

    for (size_t i = 0;
         i < rows;
         ++i)
    {
        csr.row_ptr[i + 1]
            +=
            csr.row_ptr[i];
    }

    //
    // scatter
    //

    std::vector<size_t>
        next =
            csr.row_ptr;

    for (size_t i = 0;
         i < nnz_;
         ++i)
    {
        const size_t r =
            row[i];

        const size_t pos =
            next[r]++;

        csr.col_idx[pos] =
            col[i];

        csr.values[pos] =
            value[i];
    }

    return csr;
}