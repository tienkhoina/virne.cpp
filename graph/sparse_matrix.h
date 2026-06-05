#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

struct DistanceMatrix
{
    size_t n = 0;

    std::vector<double>
        data;

    DistanceMatrix() = default;

    explicit DistanceMatrix(
        size_t n_)
        :
        n(n_),
        data(
            n_ * n_)
    {
    }

    double&
    operator()(
        size_t i,
        size_t j)
    {
        return data[
            i * n + j];
    }

    const double&
    operator()(
        size_t i,
        size_t j) const
    {
        return data[
            i * n + j];
    }

    //
    // NEW
    //

    size_t rows() const noexcept
    {
        return n;
    }

    size_t cols() const noexcept
    {
        return n;
    }

    size_t size() const noexcept
    {
        return data.size();
    }

    bool empty() const noexcept
    {
        return data.empty();
    }

    double* raw_data() noexcept
    {
        return data.data();
    }

    const double* raw_data() const noexcept
    {
        return data.data();
    }

    void fill(
        double value)
    {
        std::fill(
            data.begin(),
            data.end(),
            value);
    }
};