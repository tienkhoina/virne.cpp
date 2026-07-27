#include "numpy_random_state.h"
#include "py_random.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

template<typename Function>
void benchmark(
    const std::string& name,
    std::size_t samples,
    Function&& function)
{
    volatile double sink = 0.0;

    const auto begin =
        std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < samples; ++i)
    {
        sink += static_cast<double>(
            function());
    }

    const auto end =
        std::chrono::steady_clock::now();

    const double seconds =
        std::chrono::duration<double>(
            end - begin).count();

    const double ns_per_sample =
        seconds * 1.0e9
        / static_cast<double>(samples);
    const double million_per_second =
        static_cast<double>(samples)
        / seconds
        / 1.0e6;

    std::cout
        << "| " << name
        << " | " << samples
        << " | " << std::fixed << std::setprecision(2)
        << ns_per_sample
        << " | " << million_per_second
        << " |\n";

    if (sink == -1.0)
    {
        std::cerr << "unreachable: " << sink << '\n';
    }
}

template<typename Function>
void benchmark_vector(
    const std::string& name,
    std::size_t samples,
    Function&& function)
{
    volatile double sink = 0.0;
    // The Python oracle process reuses its allocator and already receives an
    // untimed API warm-up. Each C++ sample runs in a fresh subprocess, so do
    // the equivalent warm-up here; the callable resets its RNG before each
    // invocation. This keeps page faults/first malloc arena setup out of both
    // steady-state medians without taking the old best-of shortcut.
    {
        const auto warm = function();
        if (warm.size() != samples)
        {
            throw std::runtime_error(
                "benchmark warm-up returned an unexpected size");
        }
    }
    const auto begin =
        std::chrono::steady_clock::now();
    const auto values = function();
    const auto end =
        std::chrono::steady_clock::now();

    if (values.size() != samples)
    {
        throw std::runtime_error(
            "benchmark vector returned an unexpected size");
    }
    if (!values.empty())
    {
        sink +=
            static_cast<double>(values.front())
            + static_cast<double>(values.back());
    }

    const double seconds =
        std::chrono::duration<double>(
            end - begin).count();

    const double ns_per_sample =
        seconds * 1.0e9
        / static_cast<double>(samples);
    const double million_per_second =
        static_cast<double>(samples)
        / seconds
        / 1.0e6;

    std::cout
        << "| " << name
        << " | " << samples
        << " | " << std::fixed << std::setprecision(2)
        << ns_per_sample
        << " | " << million_per_second
        << " |\n";

    if (sink == -1.0)
    {
        std::cerr << "unreachable: " << sink << '\n';
    }
}

} // namespace

int main(
    int argc,
    char** argv)
{
    constexpr std::size_t FastSamples = 5'000'000;
    constexpr std::size_t DistributionSamples = 1'000'000;

    if (argc == 2 &&
        std::string(argv[1]) == "--raw-uint32")
    {
        NumpyRandomState raw(42);
        benchmark(
            "NumpyRandomState.next_uint32",
            FastSamples * 4,
            [&] { return raw.next_uint32(); });
        return 0;
    }

    std::string selected_api;
    if (argc == 3 &&
        std::string(argv[1]) == "--api")
    {
        selected_api = argv[2];
    }
    else if (argc != 1)
    {
        throw std::invalid_argument(
            "usage: benchmark_random [--raw-uint32 | --api API]");
    }

    bool matched = selected_api.empty();
    const auto selected =
        [&](const std::string& name)
        {
            if (selected_api.empty() || selected_api == name)
            {
                matched = true;
                return true;
            }
            return false;
        };

    std::cout
        << "| API | samples | ns/sample | million samples/s |\n"
        << "|---|---:|---:|---:|\n";

    PyRandom python(42);
    if (selected("PyRandom.random"))
    {
        benchmark(
            "PyRandom.random",
            FastSamples,
            [&] { return python.random(); });
    }

    python.seed(42);
    if (selected("PyRandom.randint(-1000,1000)"))
    {
        benchmark(
            "PyRandom.randint(-1000,1000)",
            FastSamples,
            [&] { return python.randint(-1000, 1000); });
    }

    NumpyRandomState numpy(42);
    if (selected("NumpyRandomState.random"))
    {
        benchmark_vector(
            "NumpyRandomState.random",
            FastSamples,
            [&]
            {
                numpy.seed(42);
                return numpy.random(FastSamples);
            });
    }

    numpy.seed(42);
    if (selected("NumpyRandomState.randint(-1000,1001)"))
    {
        benchmark_vector(
            "NumpyRandomState.randint(-1000,1001)",
            FastSamples,
            [&]
            {
                numpy.seed(42);
                return numpy.randint(
                    -1000,
                    1001,
                    FastSamples);
            });
    }

    numpy.seed(42);
    if (selected("NumpyRandomState.normal(0,1)"))
    {
        benchmark_vector(
            "NumpyRandomState.normal(0,1)",
            DistributionSamples,
            [&]
            {
                numpy.seed(42);
                return numpy.normal(
                    0.0,
                    1.0,
                    DistributionSamples);
            });
    }

    numpy.seed(42);
    if (selected("NumpyRandomState.exponential(1)"))
    {
        benchmark_vector(
            "NumpyRandomState.exponential(1)",
            DistributionSamples,
            [&]
            {
                numpy.seed(42);
                return numpy.exponential(
                    1.0,
                    DistributionSamples);
            });
    }

    numpy.seed(42);
    if (selected("NumpyRandomState.poisson(20)"))
    {
        benchmark_vector(
            "NumpyRandomState.poisson(20)",
            DistributionSamples,
            [&]
            {
                numpy.seed(42);
                return numpy.poisson(
                    20.0,
                    DistributionSamples);
            });
    }

    if (!matched)
    {
        throw std::invalid_argument(
            "unknown benchmark API: " + selected_api);
    }
}
