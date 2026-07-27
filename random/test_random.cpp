#include "py_random.h"
#include "numpy_random_state.h"
#include "random_context.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Fn>
void check_throws(Fn&& fn, const char* message)
{
    bool threw = false;
    try
    {
        fn();
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    check(threw, message);
}

} // namespace

int main()
{
    try
    {
        {
            PyRandom rng(42);
            constexpr std::array<std::uint64_t, 5> expected = {
                0x3fe4762f307200c5ULL,
                0x3f999c6b5eeb2060ULL,
                0x3fd19a1491f589dcULL,
                0x3fcc922b623b8c1cULL,
                0x3fe7912c1468f47eULL,
            };
            for (std::uint64_t bits : expected)
            {
                check(double_bits(rng.random()) == bits,
                      "random() differs from Python 3.10");
            }

            rng.seed(42);
            check(double_bits(rng.random()) == expected.front(),
                  "seed() did not restore Python state");
        }

        {
            PyRandom rng(0);
            constexpr std::array<std::uint64_t, 3> expected = {
                0x3feb0580f98a7dbeULL,
                0x3fe84129978f9c1aULL,
                0x3fdaeaa51052e978ULL,
            };
            for (std::uint64_t bits : expected)
            {
                check(double_bits(rng.random()) == bits,
                      "zero-seed parity failed");
            }
        }

        {
            PyRandom rng(UINT64_MAX);
            constexpr std::array<std::uint64_t, 3> expected = {
                0x3f9659799fd7f980ULL,
                0x3fd5a35a94f333d8ULL,
                0x3fcb21c0274c09a4ULL,
            };
            for (std::uint64_t bits : expected)
            {
                check(double_bits(rng.random()) == bits,
                      "64-bit seed parity failed");
            }
        }

        {
            PyRandom rng(42);
            check(rng.getrandbits32() == 2746317213U,
                  "getrandbits32() differs from Python getrandbits(32)");

            rng.seed(42);
            check(rng.genrand_uint32() == 2746317213U,
                  "genrand_uint32() differs from the CPython MT word");
        }

        {
            PyRandom rng(42);
            constexpr std::array<int, 8> widths = {0, 1, 5, 31, 32, 33, 63, 64};
            constexpr std::array<std::uint64_t, 8> expected = {
                0ULL,
                1ULL,
                3ULL,
                53710184ULL,
                3184935163ULL,
                1181241943ULL,
                1287010195568088798ULL,
                1890702223848595625ULL,
            };
            for (std::size_t i = 0; i < widths.size(); ++i)
            {
                check(rng.getrandbits(widths[i]) == expected[i],
                      "getrandbits() differs from Python 3.10");
            }
            check_throws([&] { (void)rng.getrandbits(-1); },
                         "negative getrandbits width must throw");
            check_throws([&] { (void)rng.getrandbits(65); },
                         "unsupported getrandbits width must throw");
        }

        {
            PyRandom rng(42);
            constexpr std::array<std::uint64_t, 8> stops = {
                1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 16ULL, 17ULL, 1000000000ULL};
            constexpr std::array<std::uint64_t, 8> expected = {
                0ULL, 0ULL, 2ULL, 2ULL, 1ULL, 7ULL, 4ULL, 790779946ULL};
            for (std::size_t i = 0; i < stops.size(); ++i)
            {
                check(rng.randrange(stops[i]) == expected[i],
                      "randrange() differs from Python 3.10");
            }
            check_throws([&] { (void)rng.randrange(0); },
                         "empty randrange must throw");
        }

        {
            PyRandom rng(42);
            std::vector<int> values(10);
            for (int i = 0; i < 10; ++i)
            {
                values[static_cast<std::size_t>(i)] = i;
            }
            constexpr std::array<int, 8> expected = {1, 0, 4, 3, 3, 2, 1, 8};
            for (int value : expected)
            {
                check(rng.choice(values) == value,
                      "choice() differs from Python 3.10");
            }

            const std::vector<int> const_values = values;
            rng.seed(42);
            check(rng.choice(const_values) == 1,
                  "const choice() differs from Python 3.10");

            std::vector<int> empty;
            check_throws([&] { (void)rng.choice(empty); },
                         "choice(empty) must throw");
        }

        {
            PyRandom rng(42);
            std::vector<int> values(10);
            for (int i = 0; i < 10; ++i)
            {
                values[static_cast<std::size_t>(i)] = i;
            }
            rng.shuffle(values);
            check(values == std::vector<int>({7, 3, 2, 8, 5, 6, 9, 4, 0, 1}),
                  "shuffle() differs from Python 3.10");
        }

        {
            PyRandom rng(42);
            check(double_bits(rng.uniform(0.2, 0.7)) == 0x3fe0a17dfe9f66c8ULL,
                  "first uniform() differs from Python 3.10");
            check(double_bits(rng.uniform(0.2, 0.9)) == 0x3fcbd749651af93cULL,
                  "second uniform() differs from Python 3.10");
        }

        {
            PyRandom rng(42);
            constexpr std::array<std::int64_t, 10> expected = {
                10, -7, -10, -2, -3, -3, -6, -7, 7, -8};
            for (std::int64_t value : expected)
            {
                check(rng.randint(-10, 10) == value,
                      "randint() differs from Python 3.10");
            }

            rng.seed(42);
            check(rng.randrange(-20, 31, 7) == -13,
                  "positive-step randrange differs from Python 3.10");
            check(rng.randrange(20, -31, -7) == 20,
                  "negative-step randrange differs from Python 3.10");
            check(rng.randrange(-5, 8) == 6,
                  "two-bound randrange differs from Python 3.10");
            check(rng.randint(-8, 13) == 0,
                  "mixed randint state differs from Python 3.10");

            check_throws([&] { (void)rng.randrange(1, 1); },
                         "empty signed randrange must throw");
            check_throws([&] { (void)rng.randrange(1, 5, 0); },
                         "zero-step randrange must throw");
            check_throws([&] { (void)rng.randint(5, 4); },
                         "empty randint must throw");
        }

        {
            const std::vector<int> population = {0, 1, 2, 3, 4, 5, 6};
            PyRandom rng(42);
            check(rng.choices(population, 12) ==
                      std::vector<int>({4, 0, 1, 1, 5, 4, 6, 0, 2, 0, 1, 3}),
                  "unweighted choices differs from Python 3.10");

            rng.seed(42);
            const std::vector<int> weighted_population = {0, 1, 2, 3};
            const std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
            check(rng.choices(weighted_population, weights, 12) ==
                      std::vector<int>({3, 0, 1, 1, 3, 3, 3, 0, 2, 0, 1, 2}),
                  "weighted choices differs from Python 3.10");
            check(double_bits(rng.random()) == 0x3f9b2c3ec7d6e0c0ULL,
                  "weighted choices consumed a different Python state");

            rng.seed(42);
            check(rng.choices_weights(weighted_population, weights, 12) ==
                      std::vector<int>({3, 0, 1, 1, 3, 3, 3, 0, 2, 0, 1, 2}),
                  "explicit choices_weights differs from Python 3.10");

            rng.seed(42);
            const std::vector<double> cumulative_weights =
                {1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0};
            check(rng.choices_cum_weights(
                      population,
                      cumulative_weights,
                      12) ==
                      std::vector<int>({5, 0, 3, 3, 5, 5, 6, 1, 4, 0, 3, 4}),
                  "cum_weights choices differs from Python 3.10");
            check(double_bits(rng.random()) == 0x3f9b2c3ec7d6e0c0ULL,
                  "cum_weights choices consumed a different Python state");

            const std::vector<int> empty_population;
            rng.seed(42);
            check(rng.choices(empty_population, 0).empty(),
                  "empty choices with k=0 must return without consuming state");
            check(double_bits(rng.random()) == 0x3fe4762f307200c5ULL,
                  "empty choices with k=0 consumed Python state");

            rng.seed(42);
            check_throws(
                [&] {
                    (void)rng.choices(empty_population, 5);
                },
                "empty choices with k>0 must throw");
            check(double_bits(rng.random()) == 0x3f999c6b5eeb2060ULL,
                  "empty choices must consume one CPython random draw before throwing");

            rng.seed(42);
            check_throws(
                [&] {
                    (void)rng.choices_cum_weights(
                        population,
                        std::vector<double>({1.0, 2.0}),
                        1);
                },
                "mismatched cumulative weights must throw");
            check_throws(
                [&] {
                    (void)rng.choices_cum_weights(
                        population,
                        std::vector<double>(population.size(), 0.0),
                        1);
                },
                "non-positive cumulative total must throw");
            check(double_bits(rng.random()) == 0x3fe4762f307200c5ULL,
                  "invalid cumulative weights must not consume state");
        }

        {
            NumpyRandomState rng(42);
            check(rng.next_uint32() == 1608637542U,
                  "next_uint32() differs from the NumPy MT word");
            rng.seed(42);
            constexpr std::array<std::uint64_t, 5> expected = {
                0x3fd7f8771e5f51ecULL,
                0x3fee6c4068bbd654ULL,
                0x3fe76c7e8f1e6751ULL,
                0x3fe32835d6632cb0ULL,
                0x3fc3f86b37222184ULL,
            };
            for (std::uint64_t bits : expected)
            {
                check(double_bits(rng.random()) == bits,
                      "RandomState.random differs from NumPy 1.26.4");
            }

            rng.seed(0);
            constexpr std::array<std::uint64_t, 3> zero_seed = {
                0x3fe18fe1565f12a8ULL,
                0x3fe6e2d4cf608733ULL,
                0x3fe349d66b6e894bULL,
            };
            for (std::uint64_t bits : zero_seed)
            {
                check(double_bits(rng.random()) == bits,
                      "RandomState zero-seed parity failed");
            }

            rng.seed(UINT32_MAX);
            constexpr std::array<std::uint64_t, 3> max_seed = {
                0x3fb8fe69a3924820ULL,
                0x3fed323d824032b1ULL,
                0x3fe93fc6f61af3f9ULL,
            };
            for (std::uint64_t bits : max_seed)
            {
                check(double_bits(rng.random()) == bits,
                      "RandomState UINT32_MAX seed parity failed");
            }
        }

        {
            NumpyRandomState default_uniform(42);
            NumpyRandomState explicit_uniform(42);
            check(double_bits(default_uniform.uniform()) ==
                      double_bits(explicit_uniform.uniform(0.0, 1.0)),
                  "default uniform parameters differ");

            NumpyRandomState default_normal(42);
            NumpyRandomState explicit_normal(42);
            check(double_bits(default_normal.normal()) ==
                      double_bits(explicit_normal.normal(0.0, 1.0)),
                  "default normal parameters differ");

            NumpyRandomState default_exponential(42);
            NumpyRandomState explicit_exponential(42);
            check(double_bits(default_exponential.exponential()) ==
                      double_bits(explicit_exponential.exponential(1.0)),
                  "default exponential parameter differs");

            NumpyRandomState default_poisson(42);
            NumpyRandomState explicit_poisson(42);
            check(default_poisson.poisson() ==
                      explicit_poisson.poisson(1.0),
                  "default poisson parameter differs");
        }

        {
            NumpyRandomState rng(42);
            const auto values = rng.uniform(-3.25, 8.75, 7);
            constexpr std::array<std::uint64_t, 7> expected = {
                0x3ff3e9655b1df5c4ULL,
                0x402051304e8ce0bfULL,
                0x401622bdd6ad9afaULL,
                0x400f78a183298610ULL,
                0xbff60b5f2d4ccdbaULL,
                0xbff60c8ead99c10eULL,
                0xc0046c89827d2ffbULL,
            };
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                check(double_bits(values[i]) == expected[i],
                      "RandomState.uniform differs from NumPy 1.26.4");
            }
        }

        {
            NumpyRandomState rng(42);
            const auto values = rng.randint(-100, 101, 20);
            check(values == std::vector<std::int64_t>({
                      2, 79, -8, -86, 6, -29, 88, -80, 2, 21,
                      -26, -13, 16, -1, 3, 51, 30, 49, -48, -99}),
                  "RandomState.randint vector differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fe71a9d2b546bd7ULL,
                  "RandomState.randint consumed a different NumPy state");

            NumpyRandomState named_high_only(42);
            NumpyRandomState explicit_bounds(42);
            check(named_high_only.randints(101, 20) ==
                      explicit_bounds.randint(0, 101, 20),
                  "randints(high, size) differs from randint(0, high, size)");

            NumpyRandomState scalar_high_only(42);
            NumpyRandomState scalar_explicit(42);
            check(scalar_high_only.randint(101) ==
                      scalar_explicit.randint(0, 101),
                  "randint(high) differs from randint(0, high)");
        }

        {
            std::vector<int> population(10);
            for (int i = 0; i < 10; ++i)
            {
                population[static_cast<std::size_t>(i)] = i;
            }

            NumpyRandomState rng(42);
            check(rng.choice(population, 15) ==
                      std::vector<int>({6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7, 2, 5}),
                  "uniform RandomState.choice differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3face1fa7e1300c0ULL,
                  "uniform RandomState.choice consumed a different state");

            rng.seed(42);
            const std::vector<int> weighted_population = {0, 1, 2, 3, 4};
            const std::vector<double> probabilities = {0.05, 0.1, 0.2, 0.25, 0.4};
            check(rng.choice(weighted_population, 15, probabilities) ==
                      std::vector<int>({3, 4, 4, 3, 2, 2, 1, 4, 4, 4, 0, 4, 4, 2, 2}),
                  "weighted RandomState.choice differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fc779cc89e13448ULL,
                  "weighted RandomState.choice consumed a different state");

            rng.seed(42);
            check(rng.choice(population, 6, false) ==
                      std::vector<int>({8, 1, 5, 0, 7, 2}),
                  "no-replacement choice differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fc3f7a0e1997f4cULL,
                  "no-replacement choice consumed a different state");

            rng.seed(42);
            const std::vector<double> no_replace_probabilities = {
                1.0 / 55.0, 2.0 / 55.0, 3.0 / 55.0, 4.0 / 55.0,
                5.0 / 55.0, 6.0 / 55.0, 7.0 / 55.0, 8.0 / 55.0,
                9.0 / 55.0, 10.0 / 55.0};
            check(rng.choice(population, 6, no_replace_probabilities, false) ==
                      std::vector<int>({5, 9, 8, 7, 3, 1}),
                  "weighted no-replacement choice differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3febb7b70955b7b5ULL,
                  "weighted no-replacement choice consumed a different state");

            NumpyRandomState positional(42);
            NumpyRandomState named_order(42);
            check(positional.choice(
                      std::int64_t{10},
                      6,
                      false,
                      no_replace_probabilities) ==
                      named_order.choice(
                          std::int64_t{10},
                          6,
                          no_replace_probabilities,
                          false),
                  "positional choice(a, size, replace, p) differs");
        }

        {
            NumpyRandomState rng(42);
            check(rng.choice(std::int64_t{10}) == 6,
                  "integer-population scalar choice differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fe97d47b7cd880dULL,
                  "integer-population scalar choice consumed a different state");

            rng.seed(42);
            check(rng.choice(std::int64_t{10}, 15) ==
                      std::vector<std::int64_t>({
                          6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7, 2, 5}),
                  "integer-population choice differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3face1fa7e1300c0ULL,
                  "integer-population choice consumed a different state");

            const std::vector<double> probabilities =
                {0.05, 0.1, 0.2, 0.25, 0.4};
            rng.seed(42);
            check(rng.choice(std::int64_t{5}, probabilities) == 3,
                  "weighted integer-population scalar choice differs from NumPy");
            check(double_bits(rng.random()) == 0x3fee6c4068bbd654ULL,
                  "weighted integer scalar choice consumed a different state");

            rng.seed(42);
            check(rng.choice(std::int64_t{10}, 6, false) ==
                      std::vector<std::int64_t>({8, 1, 5, 0, 7, 2}),
                  "integer-population no-replacement choice differs from NumPy");
            check(double_bits(rng.random()) == 0x3fc3f7a0e1997f4cULL,
                  "integer no-replacement choice consumed a different state");

            check(rng.choice(std::int64_t{0}, 0).empty(),
                  "zero population must allow an empty choice");
            check(rng.choice(std::int64_t{-1}, 0).empty(),
                  "negative population must allow an empty replacement choice");
            rng.seed(42);
            check(rng.choice(std::int64_t{10}, 0, false).empty(),
                  "empty no-replacement choice must return no values");
            check(double_bits(rng.random()) == 0x3fd7f8771e5f51ecULL,
                  "empty no-replacement choice must not consume state");
            check_throws(
                [&] { (void)rng.choice(std::int64_t{0}); },
                "zero integer population scalar choice must throw");
            check_throws(
                [&] { (void)rng.choice(std::int64_t{-1}, 0, false); },
                "negative population without replacement must throw");
            check_throws(
                [&] {
                    (void)rng.choice(
                        std::int64_t{5},
                        1,
                        std::vector<double>({0.5, 0.5}));
                },
                "integer population and probabilities size mismatch must throw");
        }

        {
            NumpyRandomState rng(42);
            std::vector<int> values(12);
            for (int i = 0; i < 12; ++i)
            {
                values[static_cast<std::size_t>(i)] = i;
            }
            rng.shuffle(values);
            check(values == std::vector<int>({10, 9, 0, 8, 5, 2, 1, 11, 4, 7, 3, 6}),
                  "RandomState.shuffle differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fc24975bb5443e8ULL,
                  "RandomState.shuffle consumed a different state");

            rng.seed(42);
            check(rng.permutation(12) ==
                      std::vector<std::int64_t>({10, 9, 0, 8, 5, 2, 1, 11, 4, 7, 3, 6}),
                  "RandomState.permutation differs from NumPy 1.26.4");

            rng.seed(42);
            std::vector<std::int64_t> matrix_values(21);
            for (std::size_t i = 0; i < matrix_values.size(); ++i)
            {
                matrix_values[i] = static_cast<std::int64_t>(i);
            }
            const NdArray<std::int64_t> matrix(
                {7, 3},
                matrix_values);
            const auto permuted = rng.permutation(matrix);
            check(permuted.shape() == NumpyRandomState::Shape({7, 3}),
                  "matrix permutation did not preserve shape");
            check(permuted.flat() == std::vector<std::int64_t>({
                      0, 1, 2, 3, 4, 5, 15, 16, 17, 6, 7, 8,
                      12, 13, 14, 9, 10, 11, 18, 19, 20}),
                  "matrix permutation must shuffle complete axis-0 rows");
            check(double_bits(rng.random()) == 0x3fe32835d6632cb0ULL,
                  "matrix permutation consumed a different NumPy state");

            rng.seed(42);
            NdArray<std::int64_t> empty_rows(
                {3, 0},
                std::vector<std::int64_t>{});
            rng.shuffle(empty_rows);
            check(double_bits(rng.random()) == 0x3fd7f8771e5f51ecULL,
                  "shuffle of an empty ndarray must not consume NumPy state");

            rng.seed(42);
            const auto empty_permutation =
                rng.permutation(empty_rows);
            check(empty_permutation.shape() ==
                      NumpyRandomState::Shape({3, 0}) &&
                      empty_permutation.empty(),
                  "empty ndarray permutation must preserve shape");
            check(double_bits(rng.random()) == 0x3fee6c4068bbd654ULL,
                  "empty ndarray permutation must still permute axis-0 indices");
        }

        {
            NumpyRandomState rng(42);
            const auto values =
                rng.random(NumpyRandomState::Shape({2, 3}));
            check(values.shape() == NumpyRandomState::Shape({2, 3}),
                  "random(shape) did not preserve shape");
            check(values.size() == 6,
                  "random(shape) has the wrong checked size");
            constexpr std::array<std::uint64_t, 6> expected = {
                0x3fd7f8771e5f51ecULL,
                0x3fee6c4068bbd654ULL,
                0x3fe76c7e8f1e6751ULL,
                0x3fe32835d6632cb0ULL,
                0x3fc3f86b37222184ULL,
                0x3fc3f7a0e1997f4cULL,
            };
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                check(double_bits(values[i]) == expected[i],
                      "random(shape) differs from NumPy C-order output");
            }
            check(double_bits(values.at({1, 2})) == expected.back(),
                  "ndarray C-order indexing is incorrect");
            check(double_bits(rng.random()) == 0x3fadbd229d645570ULL,
                  "random(shape) consumed a different NumPy state");

            rng.seed(42);
            const auto scalar_array =
                rng.random(NumpyRandomState::Shape{});
            check(scalar_array.ndim() == 0 && scalar_array.size() == 1,
                  "empty shape must represent a NumPy 0-D scalar array");
            check(double_bits(scalar_array[0]) == expected.front(),
                  "0-D random array differs from NumPy");

            rng.seed(42);
            const auto empty =
                rng.random(NumpyRandomState::Shape({2, 0, 3}));
            check(empty.shape() == NumpyRandomState::Shape({2, 0, 3})
                      && empty.empty(),
                  "zero-sized ndarray shape is incorrect");
            check(double_bits(rng.random()) == expected.front(),
                  "zero-sized random array must not consume state");
        }

        {
            NumpyRandomState shaped(999);
            NumpyRandomState flat(999);

            check(double_bits(shaped.rand()) == double_bits(flat.random()),
                  "scalar rand alias differs from random_sample");
            check(shaped.rand(2, 3).flat() == flat.random(6),
                  "positional rand dimensions differ from random_sample");
            check(shaped.rand(NumpyRandomState::Shape({2, 2, 2})).flat()
                      == flat.random(8),
                  "vector rand dimensions differ from random_sample");
            check(shaped.uniform(-2.0, 4.0, {2, 3}).flat()
                      == flat.uniform(-2.0, 4.0, 6),
                  "uniform(shape) differs from flat NumPy order");
            check(shaped.randint(-7, 12, {2, 2, 3}).flat()
                      == flat.randint(-7, 12, 12),
                  "randint(shape) differs from flat NumPy order");
            check(shaped.randint(17, {2, 3}).flat()
                      == flat.randint(0, 17, 6),
                  "high-only randint(shape) differs from NumPy");
            check(shaped.normal(-1.5, 0.75, {2, 3}).flat()
                      == flat.normal(-1.5, 0.75, 6),
                  "normal(shape) differs from flat NumPy order");
            check(shaped.exponential(1.25, {2, 2, 2}).flat()
                      == flat.exponential(1.25, 8),
                  "exponential(shape) differs from flat NumPy order");
            check(shaped.poisson(13.5, {2, 3}).flat()
                      == flat.poisson(13.5, 6),
                  "poisson(shape) differs from flat NumPy order");
            check(double_bits(shaped.random()) == double_bits(flat.random()),
                  "shape overloads consumed a different mixed NumPy state");
        }

        {
            NumpyRandomState rng(42);
            const auto values = rng.normal(2.5, 1.75, 8);
            constexpr std::array<std::uint64_t, 8> expected = {
                0x400af43938e5029bULL,
                0x40021075f35bb76fULL,
                0x400d1150d30fe94eULL,
                0x4014a944fe20ba73ULL,
                0x4000b8cb579269d1ULL,
                0x4000b8da67cba806ULL,
                0x40150df3099c32c4ULL,
                0x400ebe7c6f095e7aULL,
            };
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                check(double_bits(values[i]) == expected[i],
                      "RandomState.normal differs from NumPy 1.26.4");
            }
            check(double_bits(rng.normal(2.5, 1.75)) == 0x3ffadacebefacaa3ULL,
                  "RandomState cached normal differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fc7460a5fe01938ULL,
                  "RandomState.normal consumed a different state");
        }

        {
            NumpyRandomState rng(42);
            const auto values = rng.exponential(3.5, 8);
            constexpr std::array<std::uint64_t, 8> expected = {
                0x3ffa476d66046448ULL,
                0x402512233a0774d2ULL,
                0x40126f3770d8f8bbULL,
                0x40098ff8e3c4a4f3ULL,
                0x3fe2ff7bfa2916ecULL,
                0x3fe2feaa35b129bcULL,
                0x3fcacec9e7dcd2c6ULL,
                0x401c28405cf30f61ULL,
            };
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                check(double_bits(values[i]) == expected[i],
                      "RandomState.exponential differs from NumPy 1.26.4");
            }
            check(double_bits(rng.random()) == 0x3fe33c558c924badULL,
                  "RandomState.exponential consumed a different state");
        }

        {
            NumpyRandomState rng(42);
            check(rng.poisson(0.5, 12) ==
                      std::vector<std::int64_t>({0, 2, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0}),
                  "small-lambda RandomState.poisson differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fdba4fcb82f3b5aULL,
                  "small-lambda poisson consumed a different state");

            rng.seed(42);
            check(rng.poisson(10.0, 12) ==
                      std::vector<std::int64_t>({12, 6, 11, 14, 7, 8, 9, 11, 8, 10, 7, 11}),
                  "PTRS RandomState.poisson differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fe37101e6b8a28aULL,
                  "PTRS poisson consumed a different state");

            rng.seed(42);
            check(rng.poisson(1000.0, 12) ==
                      std::vector<std::int64_t>({988, 1022, 963, 1009, 1035, 967,
                                                 982, 994, 1010, 980, 996, 970}),
                  "large-lambda RandomState.poisson differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fe2f50f65ddf717ULL,
                  "large-lambda poisson consumed a different state");

            rng.seed(42);
            check(rng.poisson(9.0e18) == 9000000002087106560LL,
                  "near-int64 scalar poisson differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fc3f86b37222184ULL,
                  "near-int64 scalar poisson consumed a different state");

            rng.seed(42);
            check(rng.poisson(9.0e18, 12) ==
                      std::vector<std::int64_t>({
                          9000000002087106560LL,
                          8999999996558170112LL,
                          8999999993778146304LL,
                          9000000000863009792LL,
                          8999999987582735360LL,
                          9000000003271067648LL,
                          8999999996925480960LL,
                          8999999998271678464LL,
                          8999999999424076800LL,
                          9000000000957581312LL,
                          8999999998154207232LL,
                          8999999999630265344LL}),
                  "near-int64 vector poisson differs from NumPy 1.26.4");
            check(double_bits(rng.random()) == 0x3fc98ee9161e9074ULL,
                  "near-int64 vector poisson consumed a different state");
        }

        {
            NumpyRandomState rng(123456789);
            check(double_bits(rng.random()) == 0x3fe10cf7d7f66cbaULL,
                  "mixed RandomState initial random differs");
            check(rng.randint(-1000, 1001, 5) ==
                      std::vector<std::int64_t>({330, -654, -203, -508, -225}),
                  "mixed RandomState randint differs");

            const auto normal = rng.normal(-2.0, 0.25, 3);
            constexpr std::array<std::uint64_t, 3> normal_expected = {
                0xbffc906bd33f45d8ULL,
                0xc001a6eba008fe61ULL,
                0xbffb5ef47a28229bULL,
            };
            for (std::size_t i = 0; i < normal.size(); ++i)
            {
                check(double_bits(normal[i]) == normal_expected[i],
                      "mixed RandomState normal differs");
            }

            const auto exponential = rng.exponential(0.75, 3);
            constexpr std::array<std::uint64_t, 3> exponential_expected = {
                0x3feedd3629efaf35ULL,
                0x3ff09312434bb215ULL,
                0x3ff47b2b4ceda0ceULL,
            };
            for (std::size_t i = 0; i < exponential.size(); ++i)
            {
                check(double_bits(exponential[i]) == exponential_expected[i],
                      "mixed RandomState exponential differs");
            }

            check(rng.poisson(13.5, 5) ==
                      std::vector<std::int64_t>({8, 10, 12, 17, 19}),
                  "mixed RandomState poisson differs");

            const std::vector<int> population = {0, 1, 2, 3, 4, 5};
            const std::vector<double> probabilities = {0.1, 0.05, 0.2, 0.15, 0.3, 0.2};
            check(rng.choice(population, 7, probabilities) ==
                      std::vector<int>({3, 5, 4, 2, 2, 4, 1}),
                  "mixed RandomState choice differs");

            std::vector<int> shuffled = {0, 1, 2, 3, 4, 5, 6, 7, 8};
            rng.shuffle(shuffled);
            check(shuffled == std::vector<int>({1, 5, 7, 3, 6, 2, 4, 0, 8}),
                  "mixed RandomState shuffle differs");
            check(double_bits(rng.random()) == 0x3fe66f87c3eb7d61ULL,
                  "mixed RandomState final state differs");
        }

        {
            NumpyRandomState rng(42);
            check_throws([&] { (void)rng.randint(3, 3); },
                         "empty NumPy randint must throw");
            check_throws([&] { (void)rng.normal(0.0, -1.0); },
                         "negative normal scale must throw");
            check_throws([&] { (void)rng.exponential(-1.0); },
                         "negative exponential scale must throw");
            check_throws([&] { (void)rng.poisson(-1.0); },
                         "negative poisson lambda must throw");

            const NumpyRandomState::Shape overflowing_shape = {
                std::numeric_limits<std::size_t>::max(),
                2};
            check_throws(
                [&] { (void)rng.random(overflowing_shape); },
                "overflowing ndarray shape must throw");
            check_throws(
                [&] { (void)rng.rand(-1, 2); },
                "negative rand dimension must throw");

            check_throws(
                [] {
                    (void)NdArray<int>(
                        {2, 2},
                        std::vector<int>({1, 2, 3}));
                },
                "ndarray storage/shape mismatch must throw");

            NdArray<int> matrix(
                {2, 2},
                std::vector<int>({0, 1, 2, 3}));
            check(matrix.data() == matrix.flat().data(),
                  "ndarray data/flat storage differs");
            check(std::vector<int>(matrix.cbegin(), matrix.cend()) ==
                      matrix.flat(),
                  "ndarray const iterator range differs");
            matrix.at(std::size_t{0}) = 7;
            check(matrix[0] == 7,
                  "ndarray checked flat access differs from operator[]");
            check_throws(
                [&] {
                    (void)matrix.at(
                        NdArray<int>::Shape({0}));
                },
                "ndarray rank mismatch must throw");
            check_throws(
                [&] { (void)matrix.at({2, 0}); },
                "ndarray out-of-range index must throw");

            NdArray<int> scalar(
                {},
                std::vector<int>({7}));
            check_throws(
                [&] { rng.shuffle(scalar); },
                "zero-dimensional shuffle must throw");

            rng.seed(42);
            check_throws(
                [&] { (void)rng.random(overflowing_shape); },
                "overflowing shape must fail before allocation");
            check(double_bits(rng.random()) == 0x3fd7f8771e5f51ecULL,
                  "shape validation failure must not consume NumPy state");
        }

        {
            NumpyRandomState empty(42);
            NumpyRandomState untouched(42);
            check(empty.random(std::size_t{0}).empty(),
                  "zero-size random must be empty");
            check(empty.uniform(-1.0, 2.0, std::size_t{0}).empty(),
                  "zero-size uniform must be empty");
            check(empty.randint(-5, 6, std::size_t{0}).empty(),
                  "zero-size randint must be empty");
            check(empty.normal(1.0, 2.0, std::size_t{0}).empty(),
                  "zero-size normal must be empty");
            check(empty.exponential(2.0, std::size_t{0}).empty(),
                  "zero-size exponential must be empty");
            check(empty.poisson(3.0, std::size_t{0}).empty(),
                  "zero-size poisson must be empty");
            check(double_bits(empty.random()) ==
                      double_bits(untouched.random()),
                  "zero-size vector calls must not consume state");
        }

        {
            // This narrow interval is deliberately above int32, has a low
            // masked-rejection acceptance ratio, and lands exactly on the
            // two-pass threshold rather than the int32/pipeline fast paths.
            constexpr std::size_t count = 262144;
            constexpr std::int64_t low =
                static_cast<std::int64_t>(
                    std::numeric_limits<std::int32_t>::max()) + 1024;
            constexpr std::int64_t high = low + 17;
            NumpyRandomState bulk(42);
            NumpyRandomState scalar(42);
            const auto values = bulk.randint(low, high, count);
            check(values.size() == count,
                  "two-pass randint returned the wrong size");
            for (std::size_t index = 0; index < count; ++index)
            {
                check(values[index] == scalar.randint(low, high),
                      "two-pass randint differs from scalar state order");
            }
            check(double_bits(bulk.random()) == double_bits(scalar.random()),
                  "two-pass randint consumed a different next state");
        }

        {
            // Cross the large-vector pipeline threshold and one full internal
            // chunk. Compare every value and the following state against the
            // scalar masked-rejection contract, not only a checksum.
            constexpr std::size_t count = (1U << 20) + 17;
            NumpyRandomState bulk(42);
            NumpyRandomState scalar(42);
            const auto values = bulk.randint(-1000, 1001, count);
            check(values.size() == count,
                  "large randint pipeline returned the wrong size");
            for (std::size_t index = 0; index < count; ++index)
            {
                check(values[index] == scalar.randint(-1000, 1001),
                      "large randint pipeline differs from scalar state order");
            }
            check(double_bits(bulk.random()) == double_bits(scalar.random()),
                  "large randint pipeline consumed a different next state");
        }

        {
            RandomContext default_context;
            check(double_bits(default_context.python().random())
                      == 0x3feb0580f98a7dbeULL,
                  "default RandomContext CPython stream must use seed zero");
            check(double_bits(default_context.numpy().random())
                      == 0x3fe18fe1565f12a8ULL,
                  "default RandomContext NumPy stream must use seed zero");

            NumpyRandomState default_numpy;
            check(double_bits(default_numpy.random())
                      == 0x3fe18fe1565f12a8ULL,
                  "default NumpyRandomState must use seed zero");

            RandomContext accessor_context(42);
            const RandomContext& const_context = accessor_context;
            check(&const_context.python() == &accessor_context.python(),
                  "const RandomContext python accessor differs");
            check(&const_context.numpy() == &accessor_context.numpy(),
                  "const RandomContext NumPy accessor differs");

            RandomContext context(42);
            check(double_bits(context.python().random())
                      == 0x3fe4762f307200c5ULL,
                  "RandomContext did not seed the CPython stream");
            check(double_bits(context.numpy().random())
                      == 0x3fd7f8771e5f51ecULL,
                  "RandomContext did not seed the NumPy stream");

            context.set_seed();
            check(double_bits(context.python().random())
                      == 0x3f999c6b5eeb2060ULL,
                  "set_seed(nullopt) rewound the CPython stream");
            check(double_bits(context.numpy().random())
                      == 0x3fee6c4068bbd654ULL,
                  "set_seed(nullopt) rewound the NumPy stream");

            context.set_seed(42);
            check(double_bits(context.python().random())
                      == 0x3fe4762f307200c5ULL,
                  "RandomContext reseed failed for CPython");
            check(double_bits(context.numpy().random())
                      == 0x3fd7f8771e5f51ecULL,
                  "RandomContext reseed failed for NumPy");

            set_seed(42);
            check(double_bits(global_py_random().random())
                      == 0x3fe4762f307200c5ULL,
                  "global set_seed failed for CPython");
            check(double_bits(global_numpy_random().random())
                      == 0x3fd7f8771e5f51ecULL,
                  "global set_seed failed for NumPy");
            set_seed();
            check(double_bits(global_py_random().random())
                      == 0x3f999c6b5eeb2060ULL,
                  "global set_seed(None) rewound CPython");
            check(double_bits(global_numpy_random().random())
                      == 0x3fee6c4068bbd654ULL,
                  "global set_seed(None) rewound NumPy");
        }

        std::cout << "test_random: ALL TESTS PASSED\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_random: " << error.what() << '\n';
        return 1;
    }
}
