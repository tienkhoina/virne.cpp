#include "dataset.h"

#include "numpy_random_state.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace virne::utils
{
namespace
{

[[noreturn]] void throw_parameter_error(std::string message)
{
    throw DatasetException(
        DatasetErrorCode::invalid_parameter,
        DatasetOperation::generate_values,
        std::move(message));
}

const DatasetScalar& require_rng_parameter(
    const std::optional<DatasetScalar>& value)
{
    if (!value || std::holds_alternative<std::monostate>(*value))
    {
        throw DatasetException(
            DatasetErrorCode::missing_parameter,
            DatasetOperation::generate_values,
            "missing distribution parameter");
    }
    return *value;
}

double scalar_to_double(const DatasetScalar& value)
{
    if (const auto* floating = std::get_if<double>(&value))
    {
        return *floating;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return static_cast<double>(*integer);
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? 1.0 : 0.0;
    }
    throw_parameter_error("distribution parameter is not numeric");
}

std::int64_t scalar_to_integer(const DatasetScalar& value)
{
    if (const auto* integer = std::get_if<std::int64_t>(&value))
    {
        return *integer;
    }
    if (const auto* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? 1 : 0;
    }
    throw_parameter_error("integer uniform bounds require integer values");
}

std::uint64_t inclusive_bit_mask(std::uint64_t maximum) noexcept
{
    maximum |= maximum >> 1U;
    maximum |= maximum >> 2U;
    maximum |= maximum >> 4U;
    maximum |= maximum >> 8U;
    maximum |= maximum >> 16U;
    maximum |= maximum >> 32U;
    return maximum;
}

std::uint64_t draw_interval(
    NumpyRandomState& rng,
    std::uint64_t maximum,
    std::uint64_t mask)
{
    if (maximum == 0)
    {
        return 0;
    }

    std::uint64_t value = 0;
    do
    {
        if (maximum <= std::numeric_limits<std::uint32_t>::max())
        {
            value = static_cast<std::uint64_t>(rng.next_uint32()) & mask;
        }
        else
        {
            value =
                (static_cast<std::uint64_t>(rng.next_uint32()) << 32U) |
                static_cast<std::uint64_t>(rng.next_uint32());
            value &= mask;
        }
    }
    while (value > maximum);
    return value;
}

std::vector<std::int64_t> randint_through_int64_max(
    NumpyRandomState& rng,
    std::int64_t low,
    std::size_t count)
{
    // NumPy accepts an exclusive Python-integer bound of 2^63, while the
    // frozen signed C++ RNG API cannot represent it.  Reproduce its legacy
    // masked-interval path through the documented raw-word hook.  The
    // unsigned subtraction is the exact inclusive range size minus one,
    // including the complete signed int64 domain.
    const std::uint64_t maximum =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()) -
        static_cast<std::uint64_t>(low);
    const std::uint64_t mask = inclusive_bit_mask(maximum);

    std::vector<std::int64_t> values(count);
    for (std::int64_t& value : values)
    {
        const std::uint64_t offset = draw_interval(rng, maximum, mask);
        const std::uint64_t result_bits =
            static_cast<std::uint64_t>(low) + offset;
        value = static_cast<std::int64_t>(result_bits);
    }
    return values;
}

void validate_discriminants(const DistributionRequest& request)
{
    switch (request.distribution.kind)
    {
    case DistributionKind::uniform:
    case DistributionKind::normal:
    case DistributionKind::exponential:
    case DistributionKind::poisson:
        break;
    case DistributionKind::none:
    case DistributionKind::customized:
        throw DatasetException(
            DatasetErrorCode::invalid_distribution,
            DatasetOperation::resolve_distribution,
            "unsupported generation distribution");
    default:
        throw DatasetException(
            DatasetErrorCode::invalid_distribution,
            DatasetOperation::resolve_distribution,
            "invalid distribution enum");
    }

    switch (request.value_kind)
    {
    case DatasetValueKind::integer:
    case DatasetValueKind::floating:
    case DatasetValueKind::boolean:
        return;
    }
    throw DatasetException(
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values,
        "invalid value kind enum");
}

template <typename Callable>
auto invoke_rng(Callable&& callable) -> decltype(callable())
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw DatasetException(
            DatasetErrorCode::rng_backend_failure,
            DatasetOperation::generate_values,
            error.what());
    }
}

std::size_t effective_cast_workers(
    std::size_t requested,
    std::size_t count,
    std::size_t automatic_workers)
{
    if (count < 2 || requested == 1)
    {
        return std::min<std::size_t>(count, 1);
    }
    std::size_t workers = requested;
    if (workers == 0)
    {
        workers = automatic_workers;
    }
    if (workers <= 1)
    {
        return 1;
    }
    std::size_t available = 0;
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        const int affinity_count = CPU_COUNT(&affinity);
        if (affinity_count > 0)
        {
            available = static_cast<std::size_t>(affinity_count);
        }
    }
#endif
    if (available == 0)
    {
        available =
            static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (available == 0)
        {
            available = 1;
        }
    }
    return std::min({workers, count, available});
}

std::size_t automatic_cast_workers(
    DistributionKind distribution,
    DatasetValueKind output_kind,
    std::size_t count) noexcept
{
    constexpr std::size_t medium_threshold = 131072;
    constexpr std::size_t large_threshold = 262144;
    if (distribution != DistributionKind::exponential ||
        output_kind == DatasetValueKind::floating ||
        count < medium_threshold)
    {
        return 1;
    }
    // Canonical 1..8 sweeps at 32k, 64k, 96k, 128k, 192k, 300k, and 600k show a
    // repeatable win only for the fused exponential transform+cast pass.  The
    // other conversions are generator- or bandwidth-bound and thread startup
    // makes their end-to-end median worse, so automatic mode intentionally
    // remains sequential for them. Explicit widths remain available.
    if (count < large_threshold)
    {
        return output_kind == DatasetValueKind::integer ? 3 : 7;
    }
    return 7;
}

template <typename Input, typename Output, typename Converter>
std::vector<Output> cast_values(
    const std::vector<Input>& input,
    std::size_t requested_workers,
    std::size_t automatic_workers,
    Converter&& converter)
{
    std::vector<Output> output(input.size());
    const std::size_t worker_count =
        effective_cast_workers(
            requested_workers, input.size(), automatic_workers);
    if (worker_count <= 1)
    {
        for (std::size_t index = 0; index < input.size(); ++index)
        {
            output[index] = converter(input[index]);
        }
        return output;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count - 1);
    try
    {
        for (std::size_t worker = 1; worker < worker_count; ++worker)
        {
            threads.emplace_back(
                [&, worker]
                {
                    const std::size_t begin =
                        input.size() * worker / worker_count;
                    const std::size_t end =
                        input.size() * (worker + 1) / worker_count;
                    for (std::size_t index = begin; index < end; ++index)
                    {
                        output[index] = converter(input[index]);
                    }
                });
        }
    }
    catch (...)
    {
        for (std::thread& thread : threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        throw;
    }

    const std::size_t caller_end = input.size() / worker_count;
    for (std::size_t index = 0; index < caller_end; ++index)
    {
        output[index] = converter(input[index]);
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    return output;
}

std::int64_t numpy_double_to_int64(double value) noexcept
{
    constexpr double lower = -9223372036854775808.0;
    constexpr double upper = 9223372036854775808.0;
    if (!std::isfinite(value) || value < lower || value >= upper)
    {
        return std::numeric_limits<std::int64_t>::min();
    }
    return static_cast<std::int64_t>(value);
}

GeneratedData cast_floating_result(
    std::vector<double> values,
    DatasetValueKind kind,
    std::size_t workers,
    std::size_t automatic_workers)
{
    switch (kind)
    {
    case DatasetValueKind::floating:
        return {kind, std::move(values)};
    case DatasetValueKind::integer:
        return {
            kind,
            cast_values<double, std::int64_t>(
                values, workers, automatic_workers, numpy_double_to_int64)};
    case DatasetValueKind::boolean:
        return {
            kind,
            cast_values<double, std::uint8_t>(
                values,
                workers,
                automatic_workers,
                [](double value) noexcept
                {
                    return static_cast<std::uint8_t>(value != 0.0);
                })};
    }
    throw DatasetException(
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values,
        "invalid value kind enum");
}

GeneratedData cast_integer_result(
    std::vector<std::int64_t> values,
    DatasetValueKind kind,
    std::size_t workers,
    std::size_t automatic_workers)
{
    switch (kind)
    {
    case DatasetValueKind::integer:
        return {kind, std::move(values)};
    case DatasetValueKind::floating:
        return {
            kind,
            cast_values<std::int64_t, double>(
                values,
                workers,
                automatic_workers,
                [](std::int64_t value) noexcept
                {
                    return static_cast<double>(value);
                })};
    case DatasetValueKind::boolean:
        return {
            kind,
            cast_values<std::int64_t, std::uint8_t>(
                values,
                workers,
                automatic_workers,
                [](std::int64_t value) noexcept
                {
                    return static_cast<std::uint8_t>(value != 0);
                })};
    }
    throw DatasetException(
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values,
        "invalid value kind enum");
}

GeneratedData cast_exponential_randoms(
    std::vector<double> random_values,
    double scale,
    DatasetValueKind kind,
    std::size_t workers,
    std::size_t automatic_workers)
{
    switch (kind)
    {
    case DatasetValueKind::integer:
        return {
            kind,
            cast_values<double, std::int64_t>(
                random_values,
                workers,
                automatic_workers,
                [scale](double value) noexcept
                {
                    const double exponential =
                        scale * -std::log(1.0 - value);
                    return numpy_double_to_int64(exponential);
                })};
    case DatasetValueKind::boolean:
        return {
            kind,
            cast_values<double, std::uint8_t>(
                random_values,
                workers,
                automatic_workers,
                [scale](double value) noexcept
                {
                    const double exponential =
                        scale * -std::log(1.0 - value);
                    return static_cast<std::uint8_t>(exponential != 0.0);
                })};
    case DatasetValueKind::floating:
        break;
    }
    throw DatasetException(
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values,
        "invalid exponential cast kind");
}

} // namespace

DatasetValueKind dataset_value_kind_from_string(std::string_view value)
{
    if (value == "int")
    {
        return DatasetValueKind::integer;
    }
    if (value == "float")
    {
        return DatasetValueKind::floating;
    }
    if (value == "bool")
    {
        return DatasetValueKind::boolean;
    }
    throw DatasetException(
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values,
        "unsupported value kind");
}

GeneratedData generate_data_with_distribution(
    const DistributionRequest& request,
    NumpyRandomState& rng,
    std::size_t cast_workers)
{
    validate_discriminants(request);
    const DistributionSpec& distribution = request.distribution;
    const std::size_t automatic_workers = automatic_cast_workers(
        distribution.kind, request.value_kind, request.count);
    switch (distribution.kind)
    {
    case DistributionKind::normal:
    {
        const double location = distribution.loc
            ? scalar_to_double(*distribution.loc)
            : 0.0;
        const double scale = distribution.scale
            ? scalar_to_double(*distribution.scale)
            : 1.0;
        return cast_floating_result(
            invoke_rng(
                [&] { return rng.normal(location, scale, request.count); }),
            request.value_kind,
            cast_workers,
            automatic_workers);
    }
    case DistributionKind::uniform:
    {
        if (request.value_kind == DatasetValueKind::boolean)
        {
            throw DatasetException(
                DatasetErrorCode::uniform_boolean_uninitialized,
                DatasetOperation::generate_values,
                "Python uniform+bool leaves data uninitialized");
        }
        const DatasetScalar& low_value = require_rng_parameter(distribution.low);
        const DatasetScalar& high_value = require_rng_parameter(distribution.high);
        if (request.value_kind == DatasetValueKind::integer)
        {
            const std::int64_t low = scalar_to_integer(low_value);
            const std::int64_t high = scalar_to_integer(high_value);
            if (high == std::numeric_limits<std::int64_t>::max())
            {
                return cast_integer_result(
                    randint_through_int64_max(rng, low, request.count),
                    request.value_kind,
                    cast_workers,
                    automatic_workers);
            }
            return cast_integer_result(
                invoke_rng(
                    [&] { return rng.randint(low, high + 1, request.count); }),
                request.value_kind,
                cast_workers,
                automatic_workers);
        }
        const double low = scalar_to_double(low_value);
        const double high = scalar_to_double(high_value);
        return cast_floating_result(
            invoke_rng(
                [&] { return rng.uniform(low, high, request.count); }),
            request.value_kind,
            cast_workers,
            automatic_workers);
    }
    case DistributionKind::exponential:
    {
        const double scale = scalar_to_double(
            require_rng_parameter(distribution.scale));
        if (request.value_kind != DatasetValueKind::floating)
        {
            std::vector<double> random_values = invoke_rng(
                [&]
                {
                    if (scale < 0.0)
                    {
                        throw std::invalid_argument(
                            "exponential scale must be non-negative");
                    }
                    return rng.random(request.count);
                });
            return cast_exponential_randoms(
                std::move(random_values),
                scale,
                request.value_kind,
                cast_workers,
                automatic_workers);
        }
        return cast_floating_result(
            invoke_rng([&] { return rng.exponential(scale, request.count); }),
            request.value_kind,
            cast_workers,
            automatic_workers);
    }
    case DistributionKind::poisson:
    {
        double lambda = scalar_to_double(
            require_rng_parameter(distribution.lambda));
        if (distribution.reciprocal)
        {
            if (lambda == 0.0)
            {
                throw_parameter_error("poisson reciprocal divides by zero");
            }
            lambda = 1.0 / lambda;
        }
        return cast_integer_result(
            invoke_rng([&] { return rng.poisson(lambda, request.count); }),
            request.value_kind,
            cast_workers,
            automatic_workers);
    }
    case DistributionKind::none:
    case DistributionKind::customized:
        break;
    }
    throw DatasetException(
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution,
        "unsupported generation distribution");
}

} // namespace virne::utils
