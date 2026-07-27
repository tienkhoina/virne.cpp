#include "numpy_random_state.h"
#include "py_random.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template<typename T>
void emit(
    const std::string& key,
    const std::vector<T>& values)
{
    std::cout << key << '\t';
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ',';
        }
        std::cout << values[i];
    }
    std::cout << '\n';
}

void emit_doubles(
    const std::string& key,
    const std::vector<double>& values)
{
    std::cout << key << '\t';
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ',';
        }
        std::cout
            << std::hex
            << std::setw(16)
            << std::setfill('0')
            << double_bits(values[i])
            << std::dec;
    }
    std::cout << '\n';
}

void run_python_random(
    std::uint64_t seed)
{
    PyRandom rng(seed);

    std::vector<double> random_values(8);
    for (double& value : random_values)
    {
        value = rng.random();
    }
    emit_doubles("random", random_values);

    std::vector<std::int64_t> integers(20);
    for (std::int64_t& value : integers)
    {
        value = rng.randint(-1000, 1000);
    }
    emit("randint", integers);

    std::vector<std::int64_t> positive_ranges(15);
    for (std::int64_t& value : positive_ranges)
    {
        value = rng.randrange(-500, 701, 13);
    }
    emit("randrange_positive", positive_ranges);

    std::vector<std::int64_t> negative_ranges(15);
    for (std::int64_t& value : negative_ranges)
    {
        value = rng.randrange(700, -501, -17);
    }
    emit("randrange_negative", negative_ranges);

    std::vector<int> population(11);
    std::iota(population.begin(), population.end(), 0);
    emit("choices_uniform", rng.choices(population, 20));

    const std::vector<double> weights =
        {1.0, 3.0, 2.0, 7.0, 4.0, 9.0, 5.0, 6.0, 8.0, 10.0, 11.0};
    emit("choices_weighted", rng.choices(population, weights, 20));

    const std::vector<double> cumulative_weights =
        {1.0, 4.0, 6.0, 13.0, 17.0, 26.0,
         31.0, 37.0, 45.0, 55.0, 66.0};
    emit(
        "choices_cum_weights",
        rng.choices_cum_weights(
            population,
            cumulative_weights,
            20));

    std::vector<int> shuffled(20);
    std::iota(shuffled.begin(), shuffled.end(), 0);
    rng.shuffle(shuffled);
    emit("shuffle", shuffled);

    bool empty_choices_error = false;
    try
    {
        static_cast<void>(
            rng.choices(
                std::vector<int>{},
                5));
    }
    catch (const std::out_of_range&)
    {
        empty_choices_error = true;
    }
    emit(
        "choices_empty_error",
        std::vector<int>{empty_choices_error ? 1 : 0});

    emit_doubles("next", {rng.random()});
}

void run_numpy_random_state(
    std::uint32_t seed)
{
    NumpyRandomState rng(seed);

    emit_doubles("random", rng.random(8));
    emit_doubles("uniform", rng.uniform(-17.25, 93.5, 10));
    emit("randint", rng.randint(-200, 301, 20));

    const auto random_shape =
        rng.random(NumpyRandomState::Shape({2, 3, 2}));
    emit("random_shape_shape", random_shape.shape());
    emit_doubles("random_shape", random_shape.flat());

    emit_doubles("rand_scalar", {rng.rand()});
    const auto rand_dimensions = rng.rand(2, 3, 2);
    emit("rand_dimensions_shape", rand_dimensions.shape());
    emit_doubles("rand_dimensions", rand_dimensions.flat());

    const auto uniform_shape =
        rng.uniform(-4.5, 7.25, {2, 2, 3});
    emit_doubles("uniform_shape", uniform_shape.flat());

    const auto randint_shape =
        rng.randint(-37, 52, {2, 3, 2});
    emit("randint_shape", randint_shape.flat());

    const auto normal_shape =
        rng.normal(1.25, 0.75, {2, 3});
    emit_doubles("normal_shape", normal_shape.flat());

    const auto exponential_shape =
        rng.exponential(2.5, {2, 2, 2});
    emit_doubles(
        "exponential_shape",
        exponential_shape.flat());

    const auto poisson_shape =
        rng.poisson(12.5, {2, 3});
    emit("poisson_shape", poisson_shape.flat());

    std::vector<int> population(12);
    std::iota(population.begin(), population.end(), 0);
    emit("choice_uniform", rng.choice(population, 15));

    const std::vector<double> probabilities =
        {0.02, 0.03, 0.05, 0.07, 0.08, 0.10,
         0.11, 0.12, 0.10, 0.09, 0.13, 0.10};
    emit("choice_weighted", rng.choice(population, 15, probabilities));

    emit("choice_uniform_no_replace", rng.choice(population, 6, false));

    std::vector<double> no_replace_probabilities(12);
    for (std::size_t i = 0; i < no_replace_probabilities.size(); ++i)
    {
        no_replace_probabilities[i] =
            static_cast<double>(i + 1) / 78.0;
    }
    emit(
        "choice_weighted_no_replace",
        rng.choice(population, 6, no_replace_probabilities, false));

    emit(
        "choice_integer_scalar",
        std::vector<std::int64_t>({
            rng.choice(std::int64_t{12})}));
    emit(
        "choice_integer_uniform",
        rng.choice(std::int64_t{12}, 15));
    emit(
        "choice_integer_weighted",
        rng.choice(
            std::int64_t{12},
            15,
            probabilities));
    emit(
        "choice_integer_uniform_no_replace",
        rng.choice(std::int64_t{12}, 6, false));
    emit(
        "choice_integer_weighted_no_replace",
        rng.choice(
            std::int64_t{12},
            6,
            no_replace_probabilities,
            false));

    emit_doubles("normal", rng.normal(-3.0, 2.25, 11));
    emit_doubles("exponential", rng.exponential(1.75, 9));
    emit("poisson_small", rng.poisson(0.75, 10));
    emit("poisson_threshold", rng.poisson(10.0, 10));
    emit("poisson_large", rng.poisson(250.0, 10));

    std::vector<int> shuffled(20);
    std::iota(shuffled.begin(), shuffled.end(), 0);
    rng.shuffle(shuffled);
    emit("shuffle", shuffled);

    std::vector<std::int64_t> matrix_values(21);
    std::iota(
        matrix_values.begin(),
        matrix_values.end(),
        std::int64_t{0});
    const NdArray<std::int64_t> matrix(
        {7, 3},
        matrix_values);
    const auto matrix_permutation =
        rng.permutation(matrix);
    emit(
        "matrix_permutation_shape",
        matrix_permutation.shape());
    emit(
        "matrix_permutation",
        matrix_permutation.flat());

    NdArray<std::int64_t> empty_matrix(
        {3, 0},
        std::vector<std::int64_t>{});
    rng.shuffle(empty_matrix);
    emit(
        "empty_shuffle_shape",
        empty_matrix.shape());
    emit(
        "empty_shuffle",
        empty_matrix.flat());

    const auto empty_permutation =
        rng.permutation(empty_matrix);
    emit(
        "empty_permutation_shape",
        empty_permutation.shape());
    emit(
        "empty_permutation",
        empty_permutation.flat());

    emit_doubles("next", {rng.random()});
}

void run_numpy_large(
    std::uint32_t seed)
{
    NumpyRandomState rng(seed);
    emit(
        "randint_large",
        rng.randint(-17, 1000003, 262144));
    emit_doubles("next", {rng.random()});
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3)
        {
            throw std::invalid_argument(
                "usage: random_differential_harness <python|numpy|numpy-large> <seed>");
        }

        const std::string mode = argv[1];
        const std::uint64_t seed =
            std::stoull(argv[2]);

        if (mode == "python")
        {
            run_python_random(seed);
        }
        else if (mode == "numpy" ||
                 mode == "numpy-large")
        {
            if (seed > UINT32_MAX)
            {
                throw std::out_of_range(
                    "NumPy legacy scalar seed must fit uint32");
            }
            if (mode == "numpy")
            {
                run_numpy_random_state(
                    static_cast<std::uint32_t>(seed));
            }
            else
            {
                run_numpy_large(
                    static_cast<std::uint32_t>(seed));
            }
        }
        else
        {
            throw std::invalid_argument(
                "mode must be python, numpy, or numpy-large");
        }

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "random_differential_harness: "
            << error.what()
            << '\n';
        return 1;
    }
}
