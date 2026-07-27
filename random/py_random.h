#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

class PyRandom
{
private:

    static constexpr int N = 624;
    static constexpr int M = 397;

    static constexpr uint32_t MATRIX_A =
        0x9908b0dfU;

    static constexpr uint32_t UPPER_MASK =
        0x80000000U;

    static constexpr uint32_t LOWER_MASK =
        0x7fffffffU;

    std::array<uint32_t, N> mt{};

    int index_ = N + 1;

private:

    void init_genrand(
        uint32_t s);

    void init_by_array(
        const uint32_t* init_key,
        size_t key_length);

public:

    explicit PyRandom(
        uint64_t seed);

    void seed(
        uint64_t value);

    uint32_t genrand_uint32();

    double random();

    double uniform(
        double a,
        double b);

    uint32_t getrandbits32();

    uint64_t getrandbits(
        int k);

    uint64_t randrange(
        uint64_t stop);

    int64_t randrange(
        int64_t start,
        int64_t stop);

    int64_t randrange(
        int64_t start,
        int64_t stop,
        int64_t step);

    int64_t randint(
        int64_t a,
        int64_t b);

    template<typename T>
    T& choice(
        std::vector<T>& v)
    {
        if (v.empty())
        {
            throw std::out_of_range(
                "Cannot choose from an empty sequence");
        }

        return v[
            randrange(
                v.size())];
    }

    template<typename T>
    const T& choice(
        const std::vector<T>& v)
    {
        if (v.empty())
        {
            throw std::out_of_range(
                "Cannot choose from an empty sequence");
        }

        return v[
            randrange(
                v.size())];
    }

    template<typename T>
    void shuffle(
        std::vector<T>& v)
    {
        for(size_t i=v.size(); i>1; --i)
        {
            size_t j =
                randrange(i);

            std::swap(
                v[i-1],
                v[j]);
        }
    }

    // CPython deliberately implements choices() using random()*n rather than
    // choice()/_randbelow().  Keep this separate algorithm so mixed calls
    // consume exactly the same MT19937 state as random.Random.
    template<typename T>
    std::vector<T> choices(
        const std::vector<T>& population,
        size_t k)
    {
        if (population.empty() && k != 0)
        {
            // CPython evaluates random() * len(population) before the empty
            // population subscript raises.  Preserve that one draw because
            // the state observed by the next call is part of this contract.
            static_cast<void>(random());
            throw std::out_of_range(
                "Cannot choose from an empty population");
        }

        std::vector<T> result;
        result.reserve(k);

        const double n =
            static_cast<double>(
                population.size());

        for (size_t i = 0; i < k; ++i)
        {
            const size_t index =
                static_cast<size_t>(
                    std::floor(
                        random() * n));

            result.push_back(
                population[index]);
        }

        return result;
    }

    template<typename T>
    std::vector<T> choices(
        const std::vector<T>& population,
        const std::vector<double>& weights,
        size_t k)
    {
        return choices_weights(
            population,
            weights,
            k);
    }

    // Explicitly named alternatives avoid the positional ambiguity between
    // Python's weights= and cum_weights= keyword-only inputs.
    template<typename T>
    std::vector<T> choices_weights(
        const std::vector<T>& population,
        const std::vector<double>& weights,
        size_t k)
    {
        if (population.size() != weights.size())
        {
            throw std::invalid_argument(
                "The number of weights does not match the population");
        }

        if (weights.empty())
        {
            throw std::invalid_argument(
                "The number of weights must be greater than zero");
        }

        std::vector<double> cumulative;
        cumulative.reserve(weights.size());

        double total = 0.0;
        for (double weight : weights)
        {
            total += weight;
            cumulative.push_back(total);
        }

        // This is the same validation performed by CPython 3.10 after its
        // itertools.accumulate pass.  In particular, negative individual
        // weights are not rejected independently by that implementation.
        if (!(total > 0.0) || !std::isfinite(total))
        {
            throw std::invalid_argument(
                "Total of weights must be greater than zero and finite");
        }

        std::vector<T> result;
        result.reserve(k);

        const auto search_end =
            cumulative.end() - 1;

        for (size_t i = 0; i < k; ++i)
        {
            const double target =
                random() * total;

            // random.choices calls bisect(..., hi=n-1), deliberately leaving
            // the final cumulative element as a rounding-error catch-all.
            size_t lower = 0;
            size_t upper =
                static_cast<size_t>(
                    search_end - cumulative.begin());

            // Spell out bisect_right instead of std::upper_bound.  CPython
            // historically permits negative individual weights, for which
            // the cumulative array need not be sorted; std::upper_bound would
            // then have a violated precondition, while bisect still has a
            // defined (if unusual) result.
            while (lower < upper)
            {
                const size_t middle =
                    (lower + upper) / 2;

                if (target < cumulative[middle])
                {
                    upper = middle;
                }
                else
                {
                    lower = middle + 1;
                }
            }

            result.push_back(
                population[lower]);
        }

        return result;
    }

    template<typename T>
    std::vector<T> choices_cum_weights(
        const std::vector<T>& population,
        const std::vector<double>& cumulative_weights,
        size_t k)
    {
        if (population.size()
            != cumulative_weights.size())
        {
            throw std::invalid_argument(
                "The number of cumulative weights does not match the population");
        }

        if (cumulative_weights.empty())
        {
            throw std::invalid_argument(
                "The number of cumulative weights must be greater than zero");
        }

        // CPython deliberately validates only the final cumulative value.
        // Individual entries need not be positive or monotonic; bisect still
        // has deterministic behavior for such unusual input.
        const double total =
            cumulative_weights.back();
        if (!(total > 0.0) || !std::isfinite(total))
        {
            throw std::invalid_argument(
                "Total of cumulative weights must be greater than zero and finite");
        }

        std::vector<T> result;
        result.reserve(k);

        // random.choices uses bisect(..., hi=n-1), keeping the last entry as
        // a rounding catch-all and never inspecting it during the search.
        const size_t search_size =
            cumulative_weights.size() - 1;

        for (size_t i = 0; i < k; ++i)
        {
            const double target =
                random() * total;
            size_t lower = 0;
            size_t upper = search_size;

            while (lower < upper)
            {
                const size_t middle =
                    (lower + upper) / 2;
                if (target < cumulative_weights[middle])
                {
                    upper = middle;
                }
                else
                {
                    lower = middle + 1;
                }
            }

            result.push_back(
                population[lower]);
        }

        return result;
    }
};
