#include "dataset.h"
#include "numpy_random_state.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
DatasetException expect_dataset_exception(
    Callable&& callable,
    DatasetErrorCode code,
    DatasetOperation operation)
{
    try
    {
        callable();
    }
    catch (const DatasetException& error)
    {
        expect(error.code() == code, "dataset RNG error code mismatch");
        expect(error.operation() == operation, "dataset RNG operation mismatch");
        return error;
    }
    throw std::runtime_error("expected DatasetException");
}

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

template <typename T>
const std::vector<T>& values(const GeneratedData& result)
{
    const auto* output = std::get_if<std::vector<T>>(&result.values);
    expect(output != nullptr, "generated data variant mismatch");
    return *output;
}

void expect_same_bits(const std::vector<double>& left, const std::vector<double>& right)
{
    expect(left.size() == right.size(), "floating output size mismatch");
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        expect(
            std::memcmp(&left[index], &right[index], sizeof(double)) == 0,
            "floating output bit mismatch");
    }
}

void test_value_kind_boundary()
{
    expect(
        virne::utils::dataset_value_kind_from_string("int") ==
            DatasetValueKind::integer,
        "int value kind mismatch");
    expect(
        virne::utils::dataset_value_kind_from_string("float") ==
            DatasetValueKind::floating,
        "float value kind mismatch");
    expect(
        virne::utils::dataset_value_kind_from_string("bool") ==
            DatasetValueKind::boolean,
        "bool value kind mismatch");
    expect_dataset_exception(
        [] { (void)virne::utils::dataset_value_kind_from_string("bad"); },
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values);
}

void test_direct_rng_dispatch_and_continuation()
{
    {
        DistributionRequest value = request(
            DistributionKind::normal, DatasetValueKind::floating, 257);
        value.distribution.loc = DatasetScalar{2.0};
        value.distribution.scale = DatasetScalar{3.0};
        NumpyRandomState wrapped(17);
        NumpyRandomState direct(17);
        const GeneratedData result =
            virne::utils::generate_data_with_distribution(value, wrapped, 1);
        expect(result.value_kind == DatasetValueKind::floating,
               "normal GeneratedData value kind mismatch");
        expect_same_bits(values<double>(result), direct.normal(2.0, 3.0, 257));
        expect(wrapped.next_uint32() == direct.next_uint32(),
               "normal continuation mismatch");
    }
    {
        DistributionRequest value = request(
            DistributionKind::uniform, DatasetValueKind::integer, 1000);
        value.distribution.low = integer(-4);
        value.distribution.high = integer(7);
        NumpyRandomState wrapped(91);
        NumpyRandomState direct(91);
        const GeneratedData result =
            virne::utils::generate_data_with_distribution(value, wrapped, 1);
        expect(
            values<std::int64_t>(result) == direct.randint(-4, 8, 1000),
            "inclusive integer uniform mismatch");
        expect(wrapped.next_uint32() == direct.next_uint32(),
               "integer uniform continuation mismatch");
    }
    {
        DistributionRequest value = request(
            DistributionKind::uniform, DatasetValueKind::floating, 511);
        value.distribution.low = integer(-2);
        value.distribution.high = DatasetScalar{7.5};
        NumpyRandomState wrapped(3);
        NumpyRandomState direct(3);
        const GeneratedData result =
            virne::utils::generate_data_with_distribution(value, wrapped, 1);
        expect_same_bits(values<double>(result), direct.uniform(-2.0, 7.5, 511));
        expect(wrapped.next_uint32() == direct.next_uint32(),
               "floating uniform continuation mismatch");
    }
    {
        DistributionRequest value = request(
            DistributionKind::exponential, DatasetValueKind::floating, 513);
        value.distribution.scale = DatasetScalar{0.5};
        NumpyRandomState wrapped(42);
        NumpyRandomState direct(42);
        const GeneratedData result =
            virne::utils::generate_data_with_distribution(value, wrapped, 1);
        expect_same_bits(values<double>(result), direct.exponential(0.5, 513));
        expect(wrapped.next_uint32() == direct.next_uint32(),
               "exponential continuation mismatch");
    }
    {
        DistributionRequest value = request(
            DistributionKind::poisson, DatasetValueKind::integer, 777);
        value.distribution.lambda = DatasetScalar{0.04};
        value.distribution.reciprocal = true;
        NumpyRandomState wrapped(12);
        NumpyRandomState direct(12);
        const GeneratedData result =
            virne::utils::generate_data_with_distribution(value, wrapped, 1);
        expect(values<std::int64_t>(result) == direct.poisson(25.0, 777),
               "reciprocal poisson mismatch");
        expect(wrapped.next_uint32() == direct.next_uint32(),
               "poisson continuation mismatch");
    }
}

void test_casts_and_worker_invariance()
{
    DistributionRequest integer_cast = request(
        DistributionKind::normal, DatasetValueKind::integer, 300000);
    integer_cast.distribution.loc = DatasetScalar{-1.25};
    integer_cast.distribution.scale = DatasetScalar{4.5};
    DistributionRequest boolean_cast = integer_cast;
    boolean_cast.value_kind = DatasetValueKind::boolean;

    std::vector<std::int64_t> integer_baseline;
    std::vector<std::uint8_t> boolean_baseline;
    for (const std::size_t workers : {1U, 2U, 4U, 8U, 0U})
    {
        NumpyRandomState integer_rng(123);
        const GeneratedData integers = virne::utils::generate_data_with_distribution(
            integer_cast, integer_rng, workers);
        if (integer_baseline.empty())
        {
            integer_baseline = values<std::int64_t>(integers);
        }
        else
        {
            expect(values<std::int64_t>(integers) == integer_baseline,
                   "integer cast worker output drift");
        }
        const double integer_next = integer_rng.normal();

        NumpyRandomState boolean_rng(123);
        const GeneratedData booleans = virne::utils::generate_data_with_distribution(
            boolean_cast, boolean_rng, workers);
        if (boolean_baseline.empty())
        {
            boolean_baseline = values<std::uint8_t>(booleans);
        }
        else
        {
            expect(values<std::uint8_t>(booleans) == boolean_baseline,
                   "boolean cast worker output drift");
        }
        expect(boolean_rng.normal() == integer_next,
               "cast worker changed RNG continuation");
    }

    NumpyRandomState capped_rng(123);
    const GeneratedData capped = virne::utils::generate_data_with_distribution(
        integer_cast,
        capped_rng,
        std::numeric_limits<std::size_t>::max());
    expect(values<std::int64_t>(capped) == integer_baseline,
           "oversized explicit worker request changed output");
    NumpyRandomState capped_reference(123);
    (void)capped_reference.normal(-1.25, 4.5, integer_cast.count);
    expect(capped_rng.next_uint32() == capped_reference.next_uint32(),
           "oversized explicit worker request changed continuation");

    DistributionRequest poisson_float = request(
        DistributionKind::poisson, DatasetValueKind::floating, 4096);
    poisson_float.distribution.lambda = DatasetScalar{20.0};
    NumpyRandomState wrapped(8);
    NumpyRandomState direct(8);
    const GeneratedData result =
        virne::utils::generate_data_with_distribution(poisson_float, wrapped, 4);
    const std::vector<std::int64_t> source = direct.poisson(20.0, 4096);
    const std::vector<double>& output = values<double>(result);
    expect(output.size() == source.size(), "poisson float size mismatch");
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        expect(output[index] == static_cast<double>(source[index]),
               "poisson float cast mismatch");
    }
    expect(wrapped.next_uint32() == direct.next_uint32(),
           "poisson float continuation mismatch");

    DistributionRequest exponential_cast = request(
        DistributionKind::exponential, DatasetValueKind::integer, 600000);
    exponential_cast.distribution.scale = DatasetScalar{0.5};
    NumpyRandomState exponential_direct(42);
    const std::vector<double> exponential_source =
        exponential_direct.exponential(0.5, exponential_cast.count);
    std::vector<std::int64_t> exponential_expected(exponential_source.size());
    std::vector<std::uint8_t> exponential_bool_expected(exponential_source.size());
    for (std::size_t index = 0; index < exponential_source.size(); ++index)
    {
        exponential_expected[index] =
            static_cast<std::int64_t>(exponential_source[index]);
        exponential_bool_expected[index] = static_cast<std::uint8_t>(
            exponential_source[index] != 0.0);
    }
    const std::uint32_t exponential_continuation =
        exponential_direct.next_uint32();
    DistributionRequest exponential_bool = exponential_cast;
    exponential_bool.value_kind = DatasetValueKind::boolean;
    for (const std::size_t workers : {1U, 2U, 3U, 4U, 8U, 0U})
    {
        NumpyRandomState exponential_rng(42);
        const GeneratedData generated =
            virne::utils::generate_data_with_distribution(
                exponential_cast, exponential_rng, workers);
        expect(values<std::int64_t>(generated) == exponential_expected,
               "fused exponential integer output drift");
        expect(exponential_rng.next_uint32() == exponential_continuation,
               "fused exponential integer continuation drift");

        NumpyRandomState exponential_bool_rng(42);
        const GeneratedData generated_bool =
            virne::utils::generate_data_with_distribution(
                exponential_bool, exponential_bool_rng, workers);
        expect(values<std::uint8_t>(generated_bool) == exponential_bool_expected,
               "fused exponential boolean output drift");
        expect(exponential_bool_rng.next_uint32() == exponential_continuation,
               "fused exponential boolean continuation drift");
    }
}

void test_special_numeric_casts()
{
    for (const double location : {
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             1.0e300,
             -1.0e300})
    {
        DistributionRequest value = request(
            DistributionKind::normal, DatasetValueKind::integer, 7);
        value.distribution.loc = DatasetScalar{location};
        value.distribution.scale = DatasetScalar{0.0};
        NumpyRandomState rng(123);
        const GeneratedData result =
            virne::utils::generate_data_with_distribution(value, rng, 4);
        expect(result.value_kind == DatasetValueKind::integer,
               "special cast value kind mismatch");
        expect(
            values<std::int64_t>(result) == std::vector<std::int64_t>(
                7, std::numeric_limits<std::int64_t>::min()),
            "NumPy non-finite/out-of-range integer cast mismatch");
    }

    DistributionRequest zero_bool = request(
        DistributionKind::normal, DatasetValueKind::boolean, 7);
    zero_bool.distribution.loc = DatasetScalar{0.0};
    zero_bool.distribution.scale = DatasetScalar{0.0};
    NumpyRandomState zero_bool_rng(9);
    const GeneratedData booleans =
        virne::utils::generate_data_with_distribution(zero_bool, zero_bool_rng, 8);
    expect(values<std::uint8_t>(booleans) == std::vector<std::uint8_t>(7, 0),
           "zero floating values did not cast to false");

    DistributionRequest zero_poisson = request(
        DistributionKind::poisson, DatasetValueKind::floating, 17);
    zero_poisson.distribution.lambda = DatasetScalar{0.0};
    NumpyRandomState zero_poisson_rng(9);
    NumpyRandomState untouched_rng(9);
    const GeneratedData floating = virne::utils::generate_data_with_distribution(
        zero_poisson, zero_poisson_rng, 8);
    expect(values<double>(floating) == std::vector<double>(17, 0.0),
           "zero Poisson float cast mismatch");
    expect(zero_poisson_rng.next_uint32() == untouched_rng.next_uint32(),
           "zero Poisson consumed RNG state");
}

void test_int64_max_uniform_boundary()
{
    DistributionRequest broad = request(
        DistributionKind::uniform, DatasetValueKind::integer, 5);
    broad.distribution.low = integer(0);
    broad.distribution.high =
        integer(std::numeric_limits<std::int64_t>::max());
    NumpyRandomState broad_rng(99);
    const GeneratedData broad_result =
        virne::utils::generate_data_with_distribution(broad, broad_rng, 1);
    const std::vector<std::int64_t> broad_expected = {
        3177978388542938147LL,
        9003457315302182056LL,
        6004326113586415336LL,
        580083448559260740LL,
        5682518831843309492LL};
    expect(values<std::int64_t>(broad_result) == broad_expected,
           "INT64_MAX uniform legacy interval mismatch");

    NumpyRandomState raw_rng(99);
    for (std::size_t index = 0; index < 10; ++index)
    {
        (void)raw_rng.next_uint32();
    }
    expect(broad_rng.next_uint32() == raw_rng.next_uint32(),
           "INT64_MAX uniform continuation mismatch");

    DistributionRequest singleton = request(
        DistributionKind::uniform, DatasetValueKind::integer, 4);
    singleton.distribution.low =
        integer(std::numeric_limits<std::int64_t>::max());
    singleton.distribution.high =
        integer(std::numeric_limits<std::int64_t>::max());
    NumpyRandomState singleton_rng(99);
    NumpyRandomState untouched_rng(99);
    const GeneratedData singleton_result =
        virne::utils::generate_data_with_distribution(singleton, singleton_rng, 8);
    expect(
        values<std::int64_t>(singleton_result) ==
            std::vector<std::int64_t>(
                4, std::numeric_limits<std::int64_t>::max()),
        "INT64_MAX singleton uniform mismatch");
    expect(singleton_rng.next_uint32() == untouched_rng.next_uint32(),
           "singleton uniform consumed RNG state");
}

void test_errors_and_state_consumption()
{
    const auto expect_unchanged = [](
        DistributionRequest value,
        DatasetErrorCode code,
        DatasetOperation operation)
    {
        NumpyRandomState wrapped(99);
        NumpyRandomState untouched(99);
        expect_dataset_exception(
            [&]
            {
                (void)virne::utils::generate_data_with_distribution(
                    value, wrapped, 8);
            },
            code,
            operation);
        expect(wrapped.next_uint32() == untouched.next_uint32(),
               "pre-draw error consumed RNG state");
    };

    expect_unchanged(
        request(DistributionKind::none, DatasetValueKind::floating, 1),
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution);
    expect_unchanged(
        request(DistributionKind::customized, DatasetValueKind::floating, 1),
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution);
    expect_unchanged(
        request(
            static_cast<DistributionKind>(255),
            DatasetValueKind::floating,
            1),
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution);
    DistributionRequest invalid_kind = request(
        DistributionKind::normal,
        static_cast<DatasetValueKind>(255),
        1);
    expect_unchanged(
        invalid_kind,
        DatasetErrorCode::invalid_value_kind,
        DatasetOperation::cast_values);
    DistributionRequest uniform_bool = request(
        DistributionKind::uniform, DatasetValueKind::boolean, 100);
    uniform_bool.distribution.low = integer(0);
    uniform_bool.distribution.high = integer(1);
    expect_unchanged(
        uniform_bool,
        DatasetErrorCode::uniform_boolean_uninitialized,
        DatasetOperation::generate_values);
    DistributionRequest uniform_bool_missing = request(
        DistributionKind::uniform, DatasetValueKind::boolean, 100);
    expect_unchanged(
        uniform_bool_missing,
        DatasetErrorCode::uniform_boolean_uninitialized,
        DatasetOperation::generate_values);
    uniform_bool_missing.distribution.low = DatasetScalar{std::string("x")};
    uniform_bool_missing.distribution.high = DatasetScalar{std::string("y")};
    expect_unchanged(
        uniform_bool_missing,
        DatasetErrorCode::uniform_boolean_uninitialized,
        DatasetOperation::generate_values);
    DistributionRequest uniform_missing = request(
        DistributionKind::uniform, DatasetValueKind::integer, 1);
    expect_unchanged(
        uniform_missing,
        DatasetErrorCode::missing_parameter,
        DatasetOperation::generate_values);
    uniform_missing.distribution.low = DatasetScalar{std::monostate{}};
    uniform_missing.distribution.high = integer(1);
    expect_unchanged(
        uniform_missing,
        DatasetErrorCode::missing_parameter,
        DatasetOperation::generate_values);
    DistributionRequest missing = request(
        DistributionKind::exponential, DatasetValueKind::floating, 1);
    expect_unchanged(
        missing,
        DatasetErrorCode::missing_parameter,
        DatasetOperation::generate_values);
    missing.distribution.scale = DatasetScalar{std::monostate{}};
    expect_unchanged(
        missing,
        DatasetErrorCode::missing_parameter,
        DatasetOperation::generate_values);
    DistributionRequest missing_lambda = request(
        DistributionKind::poisson, DatasetValueKind::integer, 1);
    expect_unchanged(
        missing_lambda,
        DatasetErrorCode::missing_parameter,
        DatasetOperation::generate_values);
    missing_lambda.distribution.lambda = DatasetScalar{std::monostate{}};
    expect_unchanged(
        missing_lambda,
        DatasetErrorCode::missing_parameter,
        DatasetOperation::generate_values);
    DistributionRequest reciprocal_zero = request(
        DistributionKind::poisson, DatasetValueKind::integer, 1);
    reciprocal_zero.distribution.lambda = DatasetScalar{0.0};
    reciprocal_zero.distribution.reciprocal = true;
    expect_unchanged(
        reciprocal_zero,
        DatasetErrorCode::invalid_parameter,
        DatasetOperation::generate_values);
    DistributionRequest negative_scale = request(
        DistributionKind::normal, DatasetValueKind::floating, 10);
    negative_scale.distribution.scale = DatasetScalar{-1.0};
    expect_unchanged(
        negative_scale,
        DatasetErrorCode::rng_backend_failure,
        DatasetOperation::generate_values);
    DistributionRequest explicit_none_normal = request(
        DistributionKind::normal, DatasetValueKind::floating, 1);
    explicit_none_normal.distribution.loc = DatasetScalar{std::monostate{}};
    expect_unchanged(
        explicit_none_normal,
        DatasetErrorCode::invalid_parameter,
        DatasetOperation::generate_values);

    DistributionRequest empty = request(
        DistributionKind::normal, DatasetValueKind::floating, 0);
    NumpyRandomState wrapped(55);
    NumpyRandomState untouched(55);
    const GeneratedData result =
        virne::utils::generate_data_with_distribution(empty, wrapped, 8);
    expect(values<double>(result).empty(), "zero-size output was not empty");
    expect(wrapped.next_uint32() == untouched.next_uint32(),
           "zero-size generation consumed RNG state");
}

} // namespace

int main()
{
    try
    {
        test_value_kind_boundary();
        test_direct_rng_dispatch_and_continuation();
        test_casts_and_worker_invariance();
        test_special_numeric_casts();
        test_int64_max_uniform_boundary();
        test_errors_and_state_consumption();
        std::cout << "dataset RNG unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "dataset RNG unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
