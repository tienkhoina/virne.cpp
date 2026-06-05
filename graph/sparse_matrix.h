#pragma once

#include "csr_matrix.h"

#include <cstddef>
#include <vector>

struct SparseMatrix
{
    size_t rows = 0;
    size_t cols = 0;

    std::vector<size_t>
        row;

    std::vector<size_t>
        col;

    std::vector<double>
        value;

    SparseMatrix() = default;

    SparseMatrix(
        size_t rows_,
        size_t cols_)
        :
        rows(rows_),
        cols(cols_)
    {
    }

    void reserve(
        size_t nnz)
    {
        row.reserve(nnz);
        col.reserve(nnz);
        value.reserve(nnz);
    }

    void add(
        size_t r,
        size_t c,
        double v)
    {
        row.push_back(r);
        col.push_back(c);
        value.push_back(v);
    }

    size_t nnz() const noexcept
    {
        return value.size();
    }

    bool empty() const noexcept
    {
        return value.empty();
    }

    CSRMatrix
    to_csr() const;
};