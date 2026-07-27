#pragma once

#include "ndarray.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// A dependency-free implementation of the NumPy 1.26 RandomState algorithms
// used by Virne.  This is intentionally a separate stream from PyRandom:
// NumPy legacy seeding and distribution state consumption are different from
// CPython random.Random even though both are based on MT19937.
class NumpyRandomState
{
public:
    using Shape = NdArray<double>::Shape;

private:
    static constexpr std::size_t StateSize = 624;
    static constexpr std::size_t StateMiddle = 397;

    std::array<std::uint32_t, StateSize> state_{};
    // A direct, contiguous cache of the tempered output words for the current
    // block. Keeping generation state and output state separate lets Release
    // builds temper all 624 words with one SIMD loop, then distributions read
    // the cached addresses without repeating the tempering pipeline.
    std::array<std::uint32_t, StateSize> tempered_{};
    std::size_t position_ = StateSize;

    bool has_gauss_ = false;
    double gauss_ = 0.0;

    void twist();

    // Fill an already allocated contiguous result buffer. This keeps the
    // legacy MT state and output addresses in registers and crosses the
    // 624-word boundary explicitly instead of paying two public calls and
    // two state-boundary checks for every double.
    void fill_random(
        double* output,
        std::size_t size);

    std::uint64_t next_uint64();

    std::uint64_t interval(
        std::uint64_t maximum);

    std::uint64_t interval_masked(
        std::uint64_t maximum,
        std::uint64_t mask);

    void fill_randint32_fast(
        std::int32_t low,
        std::uint32_t maximum,
        std::uint32_t mask,
        std::int64_t* output,
        std::size_t size);

    void fill_randint32_pipeline(
        std::int32_t low,
        std::uint32_t maximum,
        std::uint32_t mask,
        std::int64_t* output,
        std::size_t size);

    double standard_normal();

    static double log_gamma(
        double value);

    std::int64_t poisson_unchecked(
        double lambda);

    static void validate_probabilities(
        const std::vector<double>& probabilities,
        std::size_t population_size);

    std::vector<std::size_t> choice_indices(
        std::size_t population_size,
        std::size_t size,
        const std::vector<double>* probabilities,
        bool replace);

public:
    explicit NumpyRandomState(
        std::uint32_t seed_value = 0);

    void seed(
        std::uint32_t seed_value);

    std::uint32_t next_uint32();

    double random();

    std::vector<double> random(
        std::size_t size);

    NdArray<double> random(
        const Shape& shape);

    // RandomState.rand is an alias for random_sample with dimensions supplied
    // as positional arguments rather than one size tuple.
    double rand();

    NdArray<double> rand(
        const Shape& dimensions);

    template<typename... Dimensions,
             typename = std::enable_if_t<
                 (sizeof...(Dimensions) > 0)
                 && (std::is_integral_v<Dimensions> && ...)>>
    NdArray<double> rand(
        Dimensions... dimensions)
    {
        Shape shape;
        shape.reserve(sizeof...(Dimensions));
        (shape.push_back(
             checked_dimension(dimensions)), ...);
        return random(shape);
    }

    double uniform(
        double low = 0.0,
        double high = 1.0);

    std::vector<double> uniform(
        double low,
        double high,
        std::size_t size);

    NdArray<double> uniform(
        double low,
        double high,
        const Shape& shape);

    std::int64_t randint(
        std::int64_t high);

    std::int64_t randint(
        std::int64_t low,
        std::int64_t high);

    std::vector<std::int64_t> randint(
        std::int64_t low,
        std::int64_t high,
        std::size_t size);

    NdArray<std::int64_t> randint(
        std::int64_t high,
        const Shape& shape);

    NdArray<std::int64_t> randint(
        std::int64_t low,
        std::int64_t high,
        const Shape& shape);

    std::vector<std::int64_t> randints(
        std::int64_t high,
        std::size_t size);

    double normal(
        double location = 0.0,
        double scale = 1.0);

    std::vector<double> normal(
        double location,
        double scale,
        std::size_t size);

    NdArray<double> normal(
        double location,
        double scale,
        const Shape& shape);

    double exponential(
        double scale = 1.0);

    std::vector<double> exponential(
        double scale,
        std::size_t size);

    NdArray<double> exponential(
        double scale,
        const Shape& shape);

    std::int64_t poisson(
        double lambda = 1.0);

    std::vector<std::int64_t> poisson(
        double lambda,
        std::size_t size);

    NdArray<std::int64_t> poisson(
        double lambda,
        const Shape& shape);

    // NumPy also accepts an integer population and samples from
    // arange(population_size) without materializing that array.
    std::int64_t choice(
        std::int64_t population_size);

    std::int64_t choice(
        std::int64_t population_size,
        const std::vector<double>& probabilities);

    std::vector<std::int64_t> choice(
        std::int64_t population_size,
        std::size_t size,
        bool replace = true);

    std::vector<std::int64_t> choice(
        std::int64_t population_size,
        std::size_t size,
        const std::vector<double>& probabilities,
        bool replace = true);

    // Positional NumPy spelling (a, size, replace, p), retained alongside
    // the repository's existing (population, size, p, replace) convention.
    std::vector<std::int64_t> choice(
        std::int64_t population_size,
        std::size_t size,
        bool replace,
        const std::vector<double>& probabilities);

    template<typename T>
    const T& choice(
        const std::vector<T>& population)
    {
        if (population.empty())
        {
            throw std::out_of_range(
                "choice cannot sample an empty population");
        }

        const auto indices =
            choice_indices(
                population.size(),
                1,
                nullptr,
                true);

        return population[indices.front()];
    }

    template<typename T>
    const T& choice(
        const std::vector<T>& population,
        const std::vector<double>& probabilities)
    {
        if (population.empty())
        {
            throw std::out_of_range(
                "choice cannot sample an empty population");
        }

        const auto indices =
            choice_indices(
                population.size(),
                1,
                &probabilities,
                true);

        return population[indices.front()];
    }

    template<typename T>
    std::vector<T> choice(
        const std::vector<T>& population,
        std::size_t size,
        bool replace = true)
    {
        const auto indices =
            choice_indices(
                population.size(),
                size,
                nullptr,
                replace);

        std::vector<T> result;
        result.reserve(indices.size());
        for (std::size_t index : indices)
        {
            result.push_back(population[index]);
        }
        return result;
    }

    template<typename T>
    std::vector<T> choice(
        const std::vector<T>& population,
        std::size_t size,
        const std::vector<double>& probabilities,
        bool replace = true)
    {
        const auto indices =
            choice_indices(
                population.size(),
                size,
                &probabilities,
                replace);

        std::vector<T> result;
        result.reserve(indices.size());
        for (std::size_t index : indices)
        {
            result.push_back(population[index]);
        }
        return result;
    }

    template<typename T>
    void shuffle(
        std::vector<T>& values)
    {
        for (std::size_t i = values.size(); i > 1; --i)
        {
            const std::size_t j =
                static_cast<std::size_t>(
                    interval(i - 1));

            std::swap(
                values[i - 1],
                values[j]);
        }
    }

    // NumPy shuffles an ndarray along axis 0. Every trailing C-order block is
    // kept intact, so a non-empty 2-D matrix is permuted by rows and consumes
    // the same bounded-integer draws as permutation(matrix). Empty storage is
    // a state-consumption exception documented below.
    template<typename T>
    void shuffle(
        NdArray<T>& values)
    {
        if (values.ndim() == 0)
        {
            throw std::invalid_argument(
                "cannot shuffle a zero-dimensional ndarray");
        }

        // RandomState.shuffle returns before generating axis-0 swaps when
        // the ndarray has no elements, even if shape[0] is non-zero.
        if (values.empty())
        {
            return;
        }

        shuffle_axis_zero(values);
    }

    template<typename T>
    std::vector<T> permutation(
        const std::vector<T>& values)
    {
        std::vector<T> result = values;
        shuffle(result);
        return result;
    }

    template<typename T>
    NdArray<T> permutation(
        const NdArray<T>& values)
    {
        NdArray<T> result = values;
        if (result.ndim() == 0)
        {
            throw std::invalid_argument(
                "cannot permute a zero-dimensional ndarray");
        }

        // Unlike shuffle(), NumPy permutation(array) first permutes the
        // axis-0 indices.  It therefore consumes those draws even when a
        // trailing zero dimension makes the result storage empty.
        shuffle_axis_zero(result);
        return result;
    }

    std::vector<std::int64_t> permutation(
        std::size_t size);

private:
    template<typename T>
    void shuffle_axis_zero(
        NdArray<T>& values)
    {
        const std::size_t rows =
            values.shape().front();
        const std::size_t row_width =
            rows == 0 ? 0 : values.size() / rows;

        for (std::size_t i = rows; i > 1; --i)
        {
            const std::size_t j =
                static_cast<std::size_t>(
                    interval(i - 1));

            if (j == i - 1)
            {
                continue;
            }

            const std::size_t left =
                (i - 1) * row_width;
            const std::size_t right =
                j * row_width;
            for (std::size_t column = 0;
                 column < row_width;
                 ++column)
            {
                std::swap(
                    values[left + column],
                    values[right + column]);
            }
        }
    }

    template<typename Dimension>
    static std::size_t checked_dimension(
        Dimension dimension)
    {
        static_assert(
            std::is_integral_v<Dimension>,
            "rand dimensions must be integral");

        if constexpr (std::is_signed_v<Dimension>)
        {
            if (dimension < 0)
            {
                throw std::invalid_argument(
                    "rand dimensions must be non-negative");
            }
        }

        using Unsigned =
            std::make_unsigned_t<Dimension>;
        const auto unsigned_dimension =
            static_cast<Unsigned>(dimension);

        if constexpr (sizeof(Unsigned) > sizeof(std::size_t))
        {
            if (unsigned_dimension
                > static_cast<Unsigned>(
                    std::numeric_limits<std::size_t>::max()))
            {
                throw std::overflow_error(
                    "rand dimension does not fit size_t");
            }
        }

        return static_cast<std::size_t>(
            unsigned_dimension);
    }
};
