#pragma once

#include <cstddef>
#include <vector>

struct CSRMatrix
{
    size_t rows = 0;
    size_t cols = 0;

    std::vector<size_t>
        row_ptr;

    std::vector<size_t>
        col_idx;

    std::vector<double>
        values;

    CSRMatrix() = default;

    CSRMatrix(
        size_t rows_,
        size_t cols_)
        :
        rows(rows_),
        cols(cols_)
    {
    }

    size_t nnz() const noexcept
    {
        return values.size();
    }

    bool empty() const noexcept
    {
        return values.empty();
    }

    size_t rows_count() const noexcept
    {
        return rows;
    }

    size_t cols_count() const noexcept
    {
        return cols;
    }

    const size_t*
    row_ptr_data() const noexcept
    {
        return row_ptr.data();
    }

    const size_t*
    col_idx_data() const noexcept
    {
        return col_idx.data();
    }

    const double*
    values_data() const noexcept
    {
        return values.data();
    }
};