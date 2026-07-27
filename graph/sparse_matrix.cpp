#include "sparse_matrix.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace
{

void validate_coo(const SparseMatrix& matrix)
{
    if (matrix.row.size() != matrix.col.size() ||
        matrix.row.size() != matrix.value.size())
    {
        throw std::invalid_argument(
            "SparseMatrix COO arrays must have equal lengths");
    }

    for (size_t i = 0; i < matrix.value.size(); ++i)
    {
        if (matrix.row[i] >= matrix.rows ||
            matrix.col[i] >= matrix.cols)
        {
            throw std::out_of_range(
                "SparseMatrix COO coordinate is outside shape");
        }
    }
}

} // namespace

std::vector<std::vector<double>>
SparseMatrix::toarray() const
{
    validate_coo(*this);

    std::vector<std::vector<double>> dense(
        rows,
        std::vector<double>(cols, 0.0));

    // Match scipy.sparse COO conversion: duplicate coordinates are summed.
    for (size_t i = 0; i < value.size(); ++i)
    {
        dense[row[i]][col[i]] += value[i];
    }

    return dense;
}

std::pair<std::vector<size_t>, std::vector<size_t>>
SparseMatrix::nonzero() const
{
    validate_coo(*this);

    std::vector<size_t> nonzero_rows;
    std::vector<size_t> nonzero_cols;
    nonzero_rows.reserve(value.size());
    nonzero_cols.reserve(value.size());

    for (size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] != 0.0)
        {
            nonzero_rows.push_back(row[i]);
            nonzero_cols.push_back(col[i]);
        }
    }

    return {std::move(nonzero_rows), std::move(nonzero_cols)};
}

CSRMatrix
SparseMatrix::to_csr() const
{
    validate_coo(*this);

    CSRMatrix csr(
        rows,
        cols);

    csr.row_ptr.assign(
        rows + 1,
        0);

    std::vector<size_t> order(nnz());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(
        order.begin(),
        order.end(),
        [this](size_t lhs, size_t rhs)
        {
            if (row[lhs] != row[rhs])
            {
                return row[lhs] < row[rhs];
            }
            return col[lhs] < col[rhs];
        });

    csr.col_idx.reserve(order.size());
    csr.values.reserve(order.size());

    size_t previous_row = rows;
    size_t previous_col = cols;
    for (size_t index : order)
    {
        const size_t current_row = row[index];
        const size_t current_col = col[index];
        if (current_row == previous_row &&
            current_col == previous_col)
        {
            // scipy.sparse.coo_matrix.tocsr() coalesces duplicates while
            // preserving explicit zero values.
            csr.values.back() += value[index];
            continue;
        }

        csr.col_idx.push_back(current_col);
        csr.values.push_back(value[index]);
        ++csr.row_ptr[current_row + 1];
        previous_row = current_row;
        previous_col = current_col;
    }

    for (size_t i = 0; i < rows; ++i)
    {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    return csr;
}
