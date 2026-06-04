#pragma once

#include <cstddef>
#include <vector>

struct DistanceMatrix
{
    size_t n = 0;

    std::vector<double> data;

    DistanceMatrix() = default;

    explicit DistanceMatrix(
        size_t n_)
        :
        n(n_),
        data(n_ * n_)
    {
    }

    double&
    operator()(
        size_t i,
        size_t j)
    {
        return data[i * n + j];
    }

    const double&
    operator()(
        size_t i,
        size_t j) const
    {
        return data[i * n + j];
    }
};