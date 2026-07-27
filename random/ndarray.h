#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// A small owning, dependency-free ndarray value. Elements are always stored
// contiguously in NumPy C order (the final axis changes fastest). This type is
// deliberately about shape and storage only; it does not attempt to emulate
// NumPy dtypes, broadcasting, views, or strides.
template<typename T>
class NdArray
{
public:
    using value_type = T;
    using Shape = std::vector<std::size_t>;
    using Storage = std::vector<T>;
    using iterator = typename Storage::iterator;
    using const_iterator = typename Storage::const_iterator;

    NdArray()
        : shape_{0}
    {
    }

    explicit NdArray(
        Shape shape)
        : shape_(std::move(shape)),
          values_(checked_size(shape_))
    {
    }

    NdArray(
        Shape shape,
        Storage values)
        : shape_(std::move(shape)),
          values_(std::move(values))
    {
        if (values_.size() != checked_size(shape_))
        {
            throw std::invalid_argument(
                "ndarray storage size does not match its shape");
        }
    }

    static std::size_t checked_size(
        const Shape& shape)
    {
        // Like a NumPy 0-D array, an empty shape owns one scalar element.
        std::size_t count = 1;
        for (const std::size_t dimension : shape)
        {
            if (dimension == 0)
            {
                count = 0;
                continue;
            }

            // Once an earlier axis is zero, the mathematical product remains
            // zero and no later dimension can overflow it.
            if (count == 0)
            {
                continue;
            }

            if (count
                > std::numeric_limits<std::size_t>::max()
                    / dimension)
            {
                throw std::overflow_error(
                    "ndarray shape product overflows size_t");
            }
            count *= dimension;
        }
        return count;
    }

    const Shape& shape() const noexcept
    {
        return shape_;
    }

    std::size_t ndim() const noexcept
    {
        return shape_.size();
    }

    std::size_t size() const noexcept
    {
        return values_.size();
    }

    bool empty() const noexcept
    {
        return values_.empty();
    }

    Storage& flat() noexcept
    {
        return values_;
    }

    const Storage& flat() const noexcept
    {
        return values_;
    }

    T* data() noexcept
    {
        return values_.data();
    }

    const T* data() const noexcept
    {
        return values_.data();
    }

    T& operator[](
        std::size_t index) noexcept
    {
        return values_[index];
    }

    const T& operator[](
        std::size_t index) const noexcept
    {
        return values_[index];
    }

    T& at(
        std::size_t index)
    {
        return values_.at(index);
    }

    const T& at(
        std::size_t index) const
    {
        return values_.at(index);
    }

    T& at(
        const Shape& indices)
    {
        return values_.at(flat_index(indices));
    }

    const T& at(
        const Shape& indices) const
    {
        return values_.at(flat_index(indices));
    }

    iterator begin() noexcept
    {
        return values_.begin();
    }

    const_iterator begin() const noexcept
    {
        return values_.begin();
    }

    const_iterator cbegin() const noexcept
    {
        return values_.cbegin();
    }

    iterator end() noexcept
    {
        return values_.end();
    }

    const_iterator end() const noexcept
    {
        return values_.end();
    }

    const_iterator cend() const noexcept
    {
        return values_.cend();
    }

private:
    std::size_t flat_index(
        const Shape& indices) const
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument(
                "ndarray index rank does not match its shape");
        }

        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < shape_.size(); ++axis)
        {
            if (indices[axis] >= shape_[axis])
            {
                throw std::out_of_range(
                    "ndarray index is outside its shape");
            }
            offset = offset * shape_[axis] + indices[axis];
        }
        return offset;
    }

    Shape shape_;
    Storage values_;
};
