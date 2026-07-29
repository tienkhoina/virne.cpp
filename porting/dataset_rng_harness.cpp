#include "dataset.h"
#include "numpy_random_state.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using virne::utils::DatasetErrorCode;
using virne::utils::DatasetException;
using virne::utils::DatasetOperation;
using virne::utils::DatasetScalar;
using virne::utils::DatasetValueKind;
using virne::utils::DistributionKind;
using virne::utils::DistributionRequest;
using virne::utils::GeneratedData;

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct Case
{
    std::string name;
    std::uint32_t seed = 0;
    DistributionRequest request;
    std::size_t workers = 1;
};

enum class BenchmarkKind : std::uint8_t
{
    normal_float,
    uniform_float,
    uniform_int,
    uniform_int64_max,
    exponential_float,
    poisson_int,
    normal_int,
    normal_bool,
    exponential_int,
    exponential_bool,
    poisson_float,
    poisson_bool,
};

DatasetScalar integer(std::int64_t value)
{
    return DatasetScalar{value};
}

DistributionRequest request(
    DistributionKind distribution,
    DatasetValueKind value_kind,
    std::size_t count)
{
    DistributionRequest result;
    result.count = count;
    result.value_kind = value_kind;
    result.distribution.kind = distribution;
    return result;
}

std::string hex_encode(std::string_view value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2, '0');
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2] = digits[byte >> 4U];
        result[index * 2 + 1] = digits[byte & 0x0FU];
    }
    return result;
}

std::string float_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << bits;
    return output.str();
}

void fnv_byte(std::uint64_t& checksum, std::uint8_t value)
{
    checksum ^= value;
    checksum *= kFnvPrime;
}

template <typename Integer>
void fnv_integer(std::uint64_t& checksum, Integer value)
{
    using Unsigned = std::make_unsigned_t<Integer>;
    Unsigned bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(Integer); ++index)
    {
        fnv_byte(checksum, static_cast<std::uint8_t>(bits & 0xFFU));
        bits = static_cast<Unsigned>(bits >> 8U);
    }
}

std::string serialize_data(const GeneratedData& data)
{
    return std::visit(
        [&](const auto& values)
        {
            using Vector = std::decay_t<decltype(values)>;
            using Value = typename Vector::value_type;
            const char kind = std::is_same_v<Value, double>
                ? 'f'
                : (std::is_same_v<Value, std::int64_t> ? 'i' : 'b');
            if (values.size() <= 1024)
            {
                std::string result;
                result += kind;
                result.push_back(':');
                for (std::size_t index = 0; index < values.size(); ++index)
                {
                    if (index != 0)
                    {
                        result.push_back(',');
                    }
                    if constexpr (std::is_same_v<Value, double>)
                    {
                        result += float_bits(values[index]);
                    }
                    else
                    {
                        result += std::to_string(values[index]);
                    }
                }
                return result;
            }

            std::uint64_t checksum = kFnvOffset;
            for (const Value value : values)
            {
                if constexpr (std::is_same_v<Value, double>)
                {
                    std::uint64_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    fnv_integer(checksum, bits);
                }
                else
                {
                    fnv_integer(checksum, value);
                }
            }
            return std::string("summary:") + kind + ':' +
                std::to_string(values.size()) + ':' +
                std::to_string(checksum) + ':' +
                std::to_string(values.size() * sizeof(Value));
        },
        data.values);
}

std::string continuation(NumpyRandomState& rng)
{
    // Sequence the stateful calls explicitly.  Operand evaluation order in a
    // chained string concatenation is not the RNG contract and GCC may evaluate
    // the normal draw before the random draw.
    const double random_value = rng.random();
    const double normal_value = rng.normal();
    return "random=" + float_bits(random_value) +
        ";normal=" + float_bits(normal_value);
}

BenchmarkKind benchmark_kind_from_string(std::string_view value)
{
    if (value == "normal_float")
    {
        return BenchmarkKind::normal_float;
    }
    if (value == "uniform_float")
    {
        return BenchmarkKind::uniform_float;
    }
    if (value == "uniform_int")
    {
        return BenchmarkKind::uniform_int;
    }
    if (value == "uniform_int64_max")
    {
        return BenchmarkKind::uniform_int64_max;
    }
    if (value == "exponential_float")
    {
        return BenchmarkKind::exponential_float;
    }
    if (value == "poisson_int")
    {
        return BenchmarkKind::poisson_int;
    }
    if (value == "normal_int")
    {
        return BenchmarkKind::normal_int;
    }
    if (value == "normal_bool")
    {
        return BenchmarkKind::normal_bool;
    }
    if (value == "exponential_int")
    {
        return BenchmarkKind::exponential_int;
    }
    if (value == "exponential_bool")
    {
        return BenchmarkKind::exponential_bool;
    }
    if (value == "poisson_float")
    {
        return BenchmarkKind::poisson_float;
    }
    if (value == "poisson_bool")
    {
        return BenchmarkKind::poisson_bool;
    }
    throw std::invalid_argument("unsupported dataset RNG benchmark kind");
}

DistributionRequest benchmark_request(BenchmarkKind kind, std::size_t count)
{
    DistributionRequest result;
    result.count = count;
    switch (kind)
    {
    case BenchmarkKind::normal_float:
    case BenchmarkKind::normal_int:
    case BenchmarkKind::normal_bool:
        result.distribution.kind = DistributionKind::normal;
        result.distribution.loc = DatasetScalar{-1.25};
        result.distribution.scale = DatasetScalar{4.5};
        result.value_kind = kind == BenchmarkKind::normal_float
            ? DatasetValueKind::floating
            : (kind == BenchmarkKind::normal_int
                   ? DatasetValueKind::integer
                   : DatasetValueKind::boolean);
        return result;
    case BenchmarkKind::uniform_float:
        result.distribution.kind = DistributionKind::uniform;
        result.distribution.low = integer(-2);
        result.distribution.high = DatasetScalar{7.5};
        result.value_kind = DatasetValueKind::floating;
        return result;
    case BenchmarkKind::uniform_int:
        result.distribution.kind = DistributionKind::uniform;
        result.distribution.low = integer(-1000);
        result.distribution.high = integer(1000);
        result.value_kind = DatasetValueKind::integer;
        return result;
    case BenchmarkKind::uniform_int64_max:
        result.distribution.kind = DistributionKind::uniform;
        result.distribution.low = integer(0);
        result.distribution.high =
            integer(std::numeric_limits<std::int64_t>::max());
        result.value_kind = DatasetValueKind::integer;
        return result;
    case BenchmarkKind::exponential_float:
    case BenchmarkKind::exponential_int:
    case BenchmarkKind::exponential_bool:
        result.distribution.kind = DistributionKind::exponential;
        result.distribution.scale = DatasetScalar{0.5};
        result.value_kind = kind == BenchmarkKind::exponential_float
            ? DatasetValueKind::floating
            : (kind == BenchmarkKind::exponential_int
                   ? DatasetValueKind::integer
                   : DatasetValueKind::boolean);
        return result;
    case BenchmarkKind::poisson_int:
    case BenchmarkKind::poisson_float:
    case BenchmarkKind::poisson_bool:
        result.distribution.kind = DistributionKind::poisson;
        result.distribution.lambda = DatasetScalar{20.0};
        result.value_kind = kind == BenchmarkKind::poisson_int
            ? DatasetValueKind::integer
            : (kind == BenchmarkKind::poisson_float
                   ? DatasetValueKind::floating
                   : DatasetValueKind::boolean);
        return result;
    }
    throw std::invalid_argument("invalid dataset RNG benchmark enum");
}

std::pair<std::uint32_t, std::size_t> adler32_summary(
    const GeneratedData& data)
{
    constexpr std::uint32_t modulus = 65521U;
    return std::visit(
        [](const auto& values)
        {
            const auto* bytes = reinterpret_cast<const unsigned char*>(
                values.data());
            const std::size_t byte_count =
                values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type);
            std::uint32_t first = 1;
            std::uint32_t second = 0;
            std::size_t offset = 0;
            while (offset < byte_count)
            {
                const std::size_t end =
                    std::min<std::size_t>(offset + 5552U, byte_count);
                while (offset < end)
                {
                    first += bytes[offset++];
                    second += first;
                }
                first %= modulus;
                second %= modulus;
            }
            return std::pair<std::uint32_t, std::size_t>{
                (second << 16U) | first,
                byte_count};
        },
        data.values);
}

struct BenchmarkResult
{
    std::uint64_t elapsed_ns = 0;
    std::uint32_t checksum = 0;
    std::size_t output_bytes = 0;
    std::string random_bits;
    std::string normal_bits;
};

BenchmarkResult run_benchmark(
    BenchmarkKind kind,
    std::size_t count,
    std::size_t workers,
    std::uint32_t seed)
{
    using Clock = std::chrono::steady_clock;
    const DistributionRequest value = benchmark_request(kind, count);
    NumpyRandomState rng(seed);
    const auto start = Clock::now();
    GeneratedData generated =
        virne::utils::generate_data_with_distribution(value, rng, workers);
    const auto end = Clock::now();

    const auto summary = adler32_summary(generated);
    const double random_value = rng.random();
    const double normal_value = rng.normal();
    return {
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count()),
        summary.first,
        summary.second,
        float_bits(random_value),
        float_bits(normal_value)};
}

std::string_view error_name(DatasetErrorCode code)
{
    switch (code)
    {
    case DatasetErrorCode::invalid_distribution:
        return "invalid_distribution";
    case DatasetErrorCode::invalid_value_kind:
        return "invalid_value_kind";
    case DatasetErrorCode::invalid_topology:
        return "invalid_topology";
    case DatasetErrorCode::missing_parameter:
        return "missing_parameter";
    case DatasetErrorCode::invalid_parameter:
        return "invalid_parameter";
    case DatasetErrorCode::uniform_boolean_uninitialized:
        return "uniform_boolean_uninitialized";
    case DatasetErrorCode::unsupported_parameter_distribution:
        return "unsupported_parameter_distribution";
    case DatasetErrorCode::rng_backend_failure:
        return "rng_backend_failure";
    case DatasetErrorCode::xml_parse_failure:
        return "xml_parse_failure";
    case DatasetErrorCode::xml_schema_failure:
        return "xml_schema_failure";
    case DatasetErrorCode::unknown_endpoint:
        return "unknown_endpoint";
    case DatasetErrorCode::graph_materialization_failure:
        return "graph_materialization_failure";
    case DatasetErrorCode::gml_write_failure:
        return "gml_write_failure";
    }
    return "invalid_error_enum";
}

std::string_view operation_name(DatasetOperation operation)
{
    switch (operation)
    {
    case DatasetOperation::resolve_distribution:
        return "resolve_distribution";
    case DatasetOperation::resolve_topology:
        return "resolve_topology";
    case DatasetOperation::generate_values:
        return "generate_values";
    case DatasetOperation::cast_values:
        return "cast_values";
    case DatasetOperation::format_parameters:
        return "format_parameters";
    case DatasetOperation::format_file_name:
        return "format_file_name";
    case DatasetOperation::build_physical_path:
        return "build_physical_path";
    case DatasetOperation::build_virtual_path:
        return "build_virtual_path";
    case DatasetOperation::parse_xml:
        return "parse_xml";
    case DatasetOperation::materialize_graph:
        return "materialize_graph";
    case DatasetOperation::write_gml:
        return "write_gml";
    }
    return "invalid_operation_enum";
}

std::vector<Case> build_cases()
{
    std::vector<Case> cases;
    for (const std::size_t size : {0U, 1U, 2U, 7U, 257U})
    {
        cases.push_back({
            "normal_default_float_n" + std::to_string(size),
            17,
            request(DistributionKind::normal, DatasetValueKind::floating, size),
            1});
    }
    DistributionRequest normal_int = request(
        DistributionKind::normal, DatasetValueKind::integer, 257);
    normal_int.distribution.loc = DatasetScalar{-1.25};
    normal_int.distribution.scale = DatasetScalar{4.5};
    cases.push_back({"normal_custom_int", 123, normal_int, 1});
    DistributionRequest normal_bool = normal_int;
    normal_bool.value_kind = DatasetValueKind::boolean;
    cases.push_back({"normal_custom_bool", 123, normal_bool, 1});
    DistributionRequest normal_nan = request(
        DistributionKind::normal, DatasetValueKind::integer, 7);
    normal_nan.distribution.loc =
        DatasetScalar{std::numeric_limits<double>::quiet_NaN()};
    normal_nan.distribution.scale = DatasetScalar{0.0};
    cases.push_back({"normal_nan_int", 123, normal_nan, 1});
    DistributionRequest normal_infinity = normal_nan;
    normal_infinity.distribution.loc =
        DatasetScalar{std::numeric_limits<double>::infinity()};
    cases.push_back({"normal_positive_inf_int", 123, normal_infinity, 1});
    DistributionRequest normal_huge = normal_nan;
    normal_huge.distribution.loc = DatasetScalar{1.0e300};
    cases.push_back({"normal_huge_int", 123, normal_huge, 1});
    DistributionRequest normal_zero_scale = normal_nan;
    normal_zero_scale.value_kind = DatasetValueKind::floating;
    normal_zero_scale.distribution.loc = DatasetScalar{1.5};
    cases.push_back({"normal_zero_scale_float", 123, normal_zero_scale, 1});
    const double int64_upper = 9223372036854775808.0;
    const double int64_lower = -9223372036854775808.0;
    DistributionRequest normal_boundary = request(
        DistributionKind::normal, DatasetValueKind::integer, 3);
    normal_boundary.distribution.scale = DatasetScalar{0.0};
    normal_boundary.distribution.loc = DatasetScalar{int64_upper};
    cases.push_back({"normal_int_upper_limit", 123, normal_boundary, 1});
    normal_boundary.distribution.loc =
        DatasetScalar{std::nextafter(int64_upper, 0.0)};
    cases.push_back({"normal_int_upper_inside", 123, normal_boundary, 1});
    normal_boundary.distribution.loc = DatasetScalar{int64_lower};
    cases.push_back({"normal_int_lower_limit", 123, normal_boundary, 1});
    normal_boundary.distribution.loc = DatasetScalar{std::nextafter(
        int64_lower, -std::numeric_limits<double>::infinity())};
    cases.push_back({"normal_int_lower_outside", 123, normal_boundary, 1});
    DistributionRequest normal_special_bool = request(
        DistributionKind::normal, DatasetValueKind::boolean, 3);
    normal_special_bool.distribution.scale = DatasetScalar{0.0};
    normal_special_bool.distribution.loc =
        DatasetScalar{std::numeric_limits<double>::quiet_NaN()};
    cases.push_back({"normal_nan_bool", 123, normal_special_bool, 1});
    normal_special_bool.distribution.loc =
        DatasetScalar{std::numeric_limits<double>::infinity()};
    cases.push_back({"normal_positive_inf_bool", 123, normal_special_bool, 1});
    normal_special_bool.distribution.loc = DatasetScalar{-0.0};
    cases.push_back({"normal_negative_zero_bool", 123, normal_special_bool, 1});

    DistributionRequest uniform_int = request(
        DistributionKind::uniform, DatasetValueKind::integer, 513);
    uniform_int.distribution.low = integer(-4);
    uniform_int.distribution.high = integer(7);
    cases.push_back({"uniform_int_inclusive", 91, uniform_int, 1});
    DistributionRequest uniform_float = request(
        DistributionKind::uniform, DatasetValueKind::floating, 513);
    uniform_float.distribution.low = integer(-2);
    uniform_float.distribution.high = DatasetScalar{7.5};
    cases.push_back({"uniform_float", 3, uniform_float, 1});

    const std::int64_t int64_max =
        std::numeric_limits<std::int64_t>::max();
    const std::int64_t int64_min =
        std::numeric_limits<std::int64_t>::min();
    DistributionRequest uniform_to_max = request(
        DistributionKind::uniform, DatasetValueKind::integer, 9);
    uniform_to_max.distribution.low = integer(0);
    uniform_to_max.distribution.high = integer(int64_max);
    cases.push_back({"uniform_int64_max_from_zero", 99, uniform_to_max, 1});
    DistributionRequest uniform_near_max = request(
        DistributionKind::uniform, DatasetValueKind::integer, 17);
    uniform_near_max.distribution.low = integer(int64_max - 5);
    uniform_near_max.distribution.high = integer(int64_max);
    cases.push_back({"uniform_int64_max_near", 99, uniform_near_max, 1});
    DistributionRequest uniform_signed_to_max = request(
        DistributionKind::uniform, DatasetValueKind::integer, 17);
    uniform_signed_to_max.distribution.low = integer(-4);
    uniform_signed_to_max.distribution.high = integer(int64_max);
    cases.push_back({"uniform_int64_max_signed", 99, uniform_signed_to_max, 1});
    DistributionRequest uniform_full_int64 = request(
        DistributionKind::uniform, DatasetValueKind::integer, 17);
    uniform_full_int64.distribution.low = integer(int64_min);
    uniform_full_int64.distribution.high = integer(int64_max);
    cases.push_back({"uniform_full_int64", 99, uniform_full_int64, 1});
    DistributionRequest uniform_max_singleton = request(
        DistributionKind::uniform, DatasetValueKind::integer, 4);
    uniform_max_singleton.distribution.low = integer(int64_max);
    uniform_max_singleton.distribution.high = integer(int64_max);
    cases.push_back({"uniform_int64_max_singleton", 99, uniform_max_singleton, 1});

    for (const DatasetValueKind kind : {
             DatasetValueKind::floating,
             DatasetValueKind::integer,
             DatasetValueKind::boolean})
    {
        DistributionRequest exponential = request(
            DistributionKind::exponential, kind, 513);
        exponential.distribution.scale = DatasetScalar{0.5};
        const std::string suffix = kind == DatasetValueKind::floating
            ? "float"
            : (kind == DatasetValueKind::integer ? "int" : "bool");
        cases.push_back({"exponential_" + suffix, 42, exponential, 1});

        DistributionRequest poisson = request(
            DistributionKind::poisson, kind, 777);
        poisson.distribution.lambda = DatasetScalar{20.0};
        cases.push_back({"poisson_" + suffix, 12, poisson, 1});
    }
    DistributionRequest exponential_zero = request(
        DistributionKind::exponential, DatasetValueKind::floating, 7);
    exponential_zero.distribution.scale = DatasetScalar{0.0};
    cases.push_back({"exponential_zero_float", 42, exponential_zero, 1});
    exponential_zero.value_kind = DatasetValueKind::integer;
    cases.push_back({"exponential_zero_int", 42, exponential_zero, 1});
    DistributionRequest exponential_special = exponential_zero;
    exponential_special.distribution.scale =
        DatasetScalar{std::numeric_limits<double>::quiet_NaN()};
    cases.push_back({"exponential_nan_int", 42, exponential_special, 1});
    exponential_special.distribution.scale =
        DatasetScalar{std::numeric_limits<double>::infinity()};
    cases.push_back({"exponential_positive_inf_int", 42, exponential_special, 1});
    for (const DatasetValueKind kind : {
             DatasetValueKind::floating,
             DatasetValueKind::integer,
             DatasetValueKind::boolean})
    {
        DistributionRequest poisson_zero = request(
            DistributionKind::poisson, kind, 7);
        poisson_zero.distribution.lambda = DatasetScalar{0.0};
        const std::string suffix = kind == DatasetValueKind::floating
            ? "float"
            : (kind == DatasetValueKind::integer ? "int" : "bool");
        cases.push_back({"poisson_zero_" + suffix, 12, poisson_zero, 1});
    }
    DistributionRequest poisson_large = request(
        DistributionKind::poisson, DatasetValueKind::floating, 5);
    poisson_large.distribution.lambda = DatasetScalar{1.0e16};
    cases.push_back({"poisson_large_float_rounding", 12, poisson_large, 1});
    DistributionRequest reciprocal = request(
        DistributionKind::poisson, DatasetValueKind::integer, 777);
    reciprocal.distribution.lambda = DatasetScalar{0.04};
    reciprocal.distribution.reciprocal = true;
    cases.push_back({"poisson_reciprocal", 12, reciprocal, 1});

    DistributionRequest invalid_distribution = request(
        DistributionKind::customized, DatasetValueKind::floating, 1);
    cases.push_back({"error_invalid_distribution", 99, invalid_distribution, 1});
    DistributionRequest invalid_value = request(
        DistributionKind::normal, static_cast<DatasetValueKind>(255), 1);
    cases.push_back({"error_invalid_value_kind", 99, invalid_value, 1});
    DistributionRequest uniform_bool = request(
        DistributionKind::uniform, DatasetValueKind::boolean, 10);
    uniform_bool.distribution.low = integer(0);
    uniform_bool.distribution.high = integer(1);
    cases.push_back({"error_uniform_bool", 99, uniform_bool, 1});
    DistributionRequest uniform_bool_missing = request(
        DistributionKind::uniform, DatasetValueKind::boolean, 1);
    cases.push_back({"error_uniform_bool_missing", 99, uniform_bool_missing, 1});
    DistributionRequest uniform_bool_low_only = uniform_bool_missing;
    uniform_bool_low_only.distribution.low = integer(0);
    cases.push_back({"error_uniform_bool_low_only", 99, uniform_bool_low_only, 1});
    DistributionRequest uniform_bool_high_only = uniform_bool_missing;
    uniform_bool_high_only.distribution.high = integer(1);
    cases.push_back({"error_uniform_bool_high_only", 99, uniform_bool_high_only, 1});
    DistributionRequest uniform_bool_strings = uniform_bool_missing;
    uniform_bool_strings.distribution.low = DatasetScalar{std::string("x")};
    uniform_bool_strings.distribution.high = DatasetScalar{std::string("y")};
    cases.push_back({"error_uniform_bool_strings", 99, uniform_bool_strings, 1});
    DistributionRequest uniform_missing = request(
        DistributionKind::uniform, DatasetValueKind::integer, 1);
    cases.push_back({"error_uniform_missing_bounds", 99, uniform_missing, 1});
    DistributionRequest uniform_missing_high = uniform_missing;
    uniform_missing_high.distribution.low = integer(0);
    cases.push_back({"error_uniform_missing_high", 99, uniform_missing_high, 1});
    DistributionRequest uniform_explicit_none = uniform_missing;
    uniform_explicit_none.distribution.low = DatasetScalar{std::monostate{}};
    uniform_explicit_none.distribution.high = integer(1);
    cases.push_back({"error_uniform_explicit_none", 99, uniform_explicit_none, 1});
    DistributionRequest uniform_missing_n0 = uniform_missing;
    uniform_missing_n0.count = 0;
    cases.push_back({"error_uniform_missing_bounds_n0", 99, uniform_missing_n0, 1});
    DistributionRequest missing = request(
        DistributionKind::exponential, DatasetValueKind::floating, 1);
    cases.push_back({"error_missing_scale", 99, missing, 1});
    DistributionRequest explicit_none_scale = missing;
    explicit_none_scale.distribution.scale = DatasetScalar{std::monostate{}};
    cases.push_back({"error_explicit_none_scale", 99, explicit_none_scale, 1});
    DistributionRequest missing_scale_n0 = missing;
    missing_scale_n0.count = 0;
    cases.push_back({"error_missing_scale_n0", 99, missing_scale_n0, 1});
    DistributionRequest missing_lambda = request(
        DistributionKind::poisson, DatasetValueKind::integer, 1);
    cases.push_back({"error_missing_lambda", 99, missing_lambda, 1});
    DistributionRequest explicit_none_lambda = missing_lambda;
    explicit_none_lambda.distribution.lambda = DatasetScalar{std::monostate{}};
    cases.push_back({"error_explicit_none_lambda", 99, explicit_none_lambda, 1});
    DistributionRequest missing_lambda_n0 = missing_lambda;
    missing_lambda_n0.count = 0;
    cases.push_back({"error_missing_lambda_n0", 99, missing_lambda_n0, 1});
    DistributionRequest reciprocal_zero = request(
        DistributionKind::poisson, DatasetValueKind::integer, 1);
    reciprocal_zero.distribution.lambda = DatasetScalar{0.0};
    reciprocal_zero.distribution.reciprocal = true;
    cases.push_back({"error_reciprocal_zero", 99, reciprocal_zero, 1});
    DistributionRequest negative_scale = request(
        DistributionKind::normal, DatasetValueKind::floating, 10);
    negative_scale.distribution.scale = DatasetScalar{-1.0};
    cases.push_back({"error_negative_scale", 99, negative_scale, 1});
    DistributionRequest negative_scale_n0 = negative_scale;
    negative_scale_n0.count = 0;
    cases.push_back({"error_negative_scale_n0", 99, negative_scale_n0, 1});
    DistributionRequest negative_exponential = request(
        DistributionKind::exponential, DatasetValueKind::floating, 1);
    negative_exponential.distribution.scale = DatasetScalar{-1.0};
    cases.push_back({"error_negative_exponential", 99, negative_exponential, 1});
    DistributionRequest negative_exponential_n0 = negative_exponential;
    negative_exponential_n0.count = 0;
    cases.push_back({"error_negative_exponential_n0", 99, negative_exponential_n0, 1});
    DistributionRequest negative_poisson = request(
        DistributionKind::poisson, DatasetValueKind::integer, 1);
    negative_poisson.distribution.lambda = DatasetScalar{-1.0};
    cases.push_back({"error_negative_poisson", 99, negative_poisson, 1});
    DistributionRequest negative_poisson_n0 = negative_poisson;
    negative_poisson_n0.count = 0;
    cases.push_back({"error_negative_poisson_n0", 99, negative_poisson_n0, 1});
    DistributionRequest string_location = request(
        DistributionKind::normal, DatasetValueKind::floating, 1);
    string_location.distribution.loc = DatasetScalar{std::string("x")};
    cases.push_back({"error_string_location", 99, string_location, 1});
    DistributionRequest explicit_none_normal = request(
        DistributionKind::normal, DatasetValueKind::floating, 1);
    explicit_none_normal.distribution.loc = DatasetScalar{std::monostate{}};
    cases.push_back({"error_explicit_none_normal", 99, explicit_none_normal, 1});
    for (const std::size_t workers : {1U, 2U, 4U, 8U, 0U})
    {
        DistributionRequest large_int = request(
            DistributionKind::normal, DatasetValueKind::integer, 300000);
        large_int.distribution.loc = DatasetScalar{-1.25};
        large_int.distribution.scale = DatasetScalar{4.5};
        const std::string worker_text = workers == 0
            ? "auto"
            : std::to_string(workers);
        cases.push_back({"large_normal_int_w" + worker_text, 123, large_int, workers});
        DistributionRequest large_bool = large_int;
        large_bool.value_kind = DatasetValueKind::boolean;
        cases.push_back({"large_normal_bool_w" + worker_text, 123, large_bool, workers});
    }
    for (const std::size_t workers : {1U, 0U})
    {
        DistributionRequest medium_exponential = request(
            DistributionKind::exponential, DatasetValueKind::integer, 192000);
        medium_exponential.distribution.scale = DatasetScalar{0.5};
        const std::string worker_text = workers == 0
            ? "auto"
            : std::to_string(workers);
        cases.push_back({
            "medium_exponential_int_w" + worker_text,
            42,
            medium_exponential,
            workers});
        DistributionRequest medium_exponential_bool = medium_exponential;
        medium_exponential_bool.value_kind = DatasetValueKind::boolean;
        cases.push_back({
            "medium_exponential_bool_w" + worker_text,
            42,
            medium_exponential_bool,
            workers});
    }
    for (const std::size_t workers : {1U, 2U, 3U, 4U, 8U, 0U})
    {
        DistributionRequest large_exponential = request(
            DistributionKind::exponential, DatasetValueKind::integer, 600000);
        large_exponential.distribution.scale = DatasetScalar{0.5};
        const std::string worker_text = workers == 0
            ? "auto"
            : std::to_string(workers);
        cases.push_back({
            "large_exponential_int_w" + worker_text,
            42,
            large_exponential,
            workers});
        DistributionRequest large_exponential_bool = large_exponential;
        large_exponential_bool.value_kind = DatasetValueKind::boolean;
        cases.push_back({
            "large_exponential_bool_w" + worker_text,
            42,
            large_exponential_bool,
            workers});
    }
    return cases;
}

void run_cases()
{
    std::cout << "dataset_rng_harness_version=1\n";
    for (const Case& item : build_cases())
    {
        NumpyRandomState rng(item.seed);
        try
        {
            const GeneratedData result =
                virne::utils::generate_data_with_distribution(
                    item.request, rng, item.workers);
            const std::string payload = serialize_data(result) + '|' + continuation(rng);
            std::cout << "case=" << item.name << "|ok|" << hex_encode(payload) << '\n';
        }
        catch (const DatasetException& error)
        {
            std::cout << "case=" << item.name << "|error|"
                      << error_name(error.code()) << '|'
                      << operation_name(error.operation()) << '|'
                      << hex_encode(continuation(rng)) << '\n';
        }
    }
    std::cout << "status=PASS\n";
}

std::uint64_t parse_unsigned(const char* text)
{
    const std::string value(text);
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size())
    {
        throw std::invalid_argument("invalid unsigned integer");
    }
    return static_cast<std::uint64_t>(parsed);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "cases")
        {
            run_cases();
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "benchmark")
        {
            const std::string_view kind_text(argv[2]);
            const BenchmarkKind kind = benchmark_kind_from_string(kind_text);
            const std::uint64_t count_value = parse_unsigned(argv[3]);
            const std::uint64_t worker_value = parse_unsigned(argv[4]);
            const std::uint64_t seed_value = parse_unsigned(argv[5]);
            if (count_value > std::numeric_limits<std::size_t>::max() ||
                worker_value > std::numeric_limits<std::size_t>::max() ||
                seed_value > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::out_of_range("dataset RNG benchmark argument overflow");
            }
            const BenchmarkResult result = run_benchmark(
                kind,
                static_cast<std::size_t>(count_value),
                static_cast<std::size_t>(worker_value),
                static_cast<std::uint32_t>(seed_value));
            std::cout << "benchmark_version=1\n"
                      << "kind=" << kind_text << '\n'
                      << "workers=" << worker_value << '\n'
                      << "count=" << count_value << '\n'
                      << "elapsed_ns=" << result.elapsed_ns << '\n'
                      << "checksum=" << result.checksum << '\n'
                      << "output_bytes=" << result.output_bytes << '\n'
                      << "random_bits=" << result.random_bits << '\n'
                      << "normal_bits=" << result.normal_bits << '\n'
                      << "status=PASS\n";
            return 0;
        }
        throw std::invalid_argument("invalid dataset RNG harness command");
    }
    catch (const std::exception& error)
    {
        std::cerr << "dataset RNG harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
