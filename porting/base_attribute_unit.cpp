#include "attribute/base_attribute.h"
#include "numpy_random_state.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::BaseAttribute;
using attribute::BaseAttributeErrorCode;
using attribute::BaseAttributeException;
using attribute::BaseAttributeOperation;
using attribute::BaseAttributeSnapshot;
using attribute::BaseAttributeSnapshotEntry;
using attribute::BaseAttributeSpec;
using attribute::NetworkCardinality;
using utils::DatasetErrorCode;
using utils::DatasetException;
using utils::DatasetScalar;
using utils::DatasetValueKind;
using utils::DistributionKind;
using utils::DistributionRequest;
using utils::DistributionSpec;
using utils::GeneratedData;

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        fail(std::string(message));
    }
}

std::uint64_t double_bits(double value) noexcept
{
    std::uint64_t result = 0U;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool scalar_equal(const DatasetScalar& left, const DatasetScalar& right)
{
    if (left.index() != right.index())
    {
        return false;
    }
    if (std::holds_alternative<std::monostate>(left))
    {
        return true;
    }
    if (const std::int64_t* value = std::get_if<std::int64_t>(&left))
    {
        return *value == std::get<std::int64_t>(right);
    }
    if (const double* value = std::get_if<double>(&left))
    {
        return double_bits(*value) == double_bits(std::get<double>(right));
    }
    if (const bool* value = std::get_if<bool>(&left))
    {
        return *value == std::get<bool>(right);
    }
    return std::get<std::string>(left) == std::get<std::string>(right);
}

void expect_snapshot_equal(
    const BaseAttributeSnapshot& actual,
    const BaseAttributeSnapshot& expected,
    std::string_view context)
{
    if (actual.size() != expected.size())
    {
        fail(std::string(context) + ": snapshot size drift");
    }
    for (std::size_t index = 0U; index < actual.size(); ++index)
    {
        if (actual[index].name != expected[index].name ||
            !scalar_equal(actual[index].value, expected[index].value))
        {
            fail(
                std::string(context) + ": snapshot entry drift at index " +
                std::to_string(index));
        }
    }
}

void expect_generated_equal(
    const GeneratedData& actual,
    const GeneratedData& expected,
    std::string_view context)
{
    if (actual.value_kind != expected.value_kind ||
        actual.values.index() != expected.values.index())
    {
        fail(std::string(context) + ": generated representation drift");
    }
    if (const auto* values =
            std::get_if<std::vector<std::int64_t>>(&actual.values))
    {
        expect(
            *values == std::get<std::vector<std::int64_t>>(expected.values),
            std::string(context) + ": integer values drift");
        return;
    }
    if (const auto* values = std::get_if<std::vector<double>>(&actual.values))
    {
        const auto& expected_values =
            std::get<std::vector<double>>(expected.values);
        if (values->size() != expected_values.size())
        {
            fail(std::string(context) + ": floating size drift");
        }
        for (std::size_t index = 0U; index < values->size(); ++index)
        {
            if (double_bits((*values)[index]) !=
                double_bits(expected_values[index]))
            {
                fail(
                    std::string(context) +
                    ": floating bit drift at index " +
                    std::to_string(index));
            }
        }
        return;
    }
    expect(
        std::get<std::vector<std::uint8_t>>(actual.values) ==
            std::get<std::vector<std::uint8_t>>(expected.values),
        std::string(context) + ": boolean values drift");
}

std::size_t generated_size(const GeneratedData& data)
{
    if (const auto* values =
            std::get_if<std::vector<std::int64_t>>(&data.values))
    {
        return values->size();
    }
    if (const auto* values = std::get_if<std::vector<double>>(&data.values))
    {
        return values->size();
    }
    return std::get<std::vector<std::uint8_t>>(data.values).size();
}

template <typename Callable>
void expect_base_exception(
    Callable&& callable,
    BaseAttributeErrorCode expected_code,
    BaseAttributeOperation expected_operation,
    std::string_view expected_message = {})
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const BaseAttributeException& error)
    {
        expect(error.code() == expected_code, "BaseAttribute error code drift");
        expect(
            error.operation() == expected_operation,
            "BaseAttribute error operation drift");
        if (!expected_message.empty())
        {
            expect(error.what() == expected_message, "BaseAttribute message drift");
        }
        return;
    }
    catch (const std::exception& error)
    {
        fail(std::string("unexpected exception type: ") + error.what());
    }
    fail("expected BaseAttributeException");
}

template <typename Callable>
void expect_dataset_exception(
    Callable&& callable,
    DatasetErrorCode expected_code)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const DatasetException& error)
    {
        if (error.code() != expected_code)
        {
            fail(
                "dataset error code drift: expected " +
                std::to_string(static_cast<unsigned int>(expected_code)) +
                ", actual " +
                std::to_string(static_cast<unsigned int>(error.code())));
        }
        return;
    }
    catch (const std::exception& error)
    {
        fail(std::string("unexpected dataset exception type: ") + error.what());
    }
    fail("expected DatasetException");
}

template <typename ErrorCheck>
void expect_error_without_rng_consumption(
    std::uint32_t seed,
    ErrorCheck&& check)
{
    NumpyRandomState actual_rng(seed);
    NumpyRandomState untouched_rng(seed);
    std::forward<ErrorCheck>(check)(actual_rng);
    expect(
        double_bits(actual_rng.random()) == double_bits(untouched_rng.random()),
        "error consumed RNG state");
}

DistributionSpec uniform_distribution(DatasetScalar low, DatasetScalar high)
{
    DistributionSpec result;
    result.kind = DistributionKind::uniform;
    result.low = std::move(low);
    result.high = std::move(high);
    return result;
}

DistributionSpec normal_distribution(double location, double scale)
{
    DistributionSpec result;
    result.kind = DistributionKind::normal;
    result.loc = DatasetScalar{location};
    result.scale = DatasetScalar{scale};
    return result;
}

DistributionSpec exponential_distribution(double scale)
{
    DistributionSpec result;
    result.kind = DistributionKind::exponential;
    result.scale = DatasetScalar{scale};
    return result;
}

DistributionSpec poisson_distribution(double lambda, bool reciprocal = false)
{
    DistributionSpec result;
    result.kind = DistributionKind::poisson;
    result.lambda = DatasetScalar{lambda};
    result.reciprocal = reciprocal;
    return result;
}

DistributionSpec customized_distribution(
    std::optional<DatasetScalar> minimum,
    std::optional<DatasetScalar> maximum)
{
    DistributionSpec result;
    result.kind = DistributionKind::customized;
    result.minimum = std::move(minimum);
    result.maximum = std::move(maximum);
    return result;
}

BaseAttributeSpec configured_spec(
    std::string name,
    AttributeOwner owner,
    const DistributionSpec& distribution,
    std::optional<DatasetValueKind> dtype = std::nullopt)
{
    BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.generative = true;
    result.distribution = distribution;
    result.dtype = dtype;
    return result;
}

std::size_t configured_count(
    AttributeOwner owner,
    const NetworkCardinality& network) noexcept
{
    return owner == AttributeOwner::node ? network.num_nodes : network.num_links;
}

double numeric_scalar(const DatasetScalar& value)
{
    if (const std::int64_t* integer = std::get_if<std::int64_t>(&value))
    {
        return static_cast<double>(*integer);
    }
    if (const double* floating = std::get_if<double>(&value))
    {
        return *floating;
    }
    if (const bool* boolean = std::get_if<bool>(&value))
    {
        return *boolean ? 1.0 : 0.0;
    }
    fail("non-numeric scalar in unit expected path");
}

GeneratedData customized_expected(
    std::size_t count,
    double minimum,
    double maximum,
    NumpyRandomState& rng)
{
    std::vector<double> values = rng.random(count);
    const double width = maximum - minimum;
    for (std::size_t index = 0U; index < values.size(); ++index)
    {
        values[index] = values[index] * width + minimum;
    }
    return GeneratedData{DatasetValueKind::floating, std::move(values)};
}

GeneratedData customized_expected_scalars(
    std::size_t count,
    const DatasetScalar& minimum,
    const DatasetScalar& maximum,
    NumpyRandomState& rng)
{
    const auto integral = [](const DatasetScalar& value) noexcept
    {
        return std::holds_alternative<std::int64_t>(value)
            || std::holds_alternative<bool>(value);
    };
    const auto integer = [](const DatasetScalar& value) noexcept
    {
        if (const auto* item = std::get_if<std::int64_t>(&value))
        {
            return *item;
        }
        return std::get<bool>(value) ? std::int64_t{1} : std::int64_t{0};
    };
    const double minimum_value = numeric_scalar(minimum);
    double width = numeric_scalar(maximum) - minimum_value;
    if (integral(minimum) && integral(maximum))
    {
        const std::uint64_t low = static_cast<std::uint64_t>(integer(minimum));
        const std::uint64_t high = static_cast<std::uint64_t>(integer(maximum));
        width = static_cast<double>(high - low);
    }
    std::vector<double> values = rng.random(count);
    for (double& value : values)
    {
        value = value * width + minimum_value;
    }
    return GeneratedData{DatasetValueKind::floating, std::move(values)};
}

void expect_standard_generation(
    std::string_view context,
    const DistributionSpec& distribution,
    std::optional<DatasetValueKind> dtype,
    AttributeOwner owner,
    const NetworkCardinality& network,
    std::uint32_t seed,
    std::size_t workers)
{
    const BaseAttribute value(configured_spec(
        std::string(context), owner, distribution, dtype));
    NumpyRandomState actual_rng(seed);
    NumpyRandomState expected_rng(seed);
    const GeneratedData actual =
        value.generate_configured_data(network, actual_rng, workers);

    DistributionRequest request;
    request.count = configured_count(owner, network);
    request.value_kind = dtype.value_or(DatasetValueKind::floating);
    request.distribution = distribution;
    request.distribution.reciprocal = false;
    const GeneratedData expected =
        utils::generate_data_with_distribution(request, expected_rng, 1U);
    expect_generated_equal(actual, expected, context);
    expect(
        double_bits(actual_rng.random()) == double_bits(expected_rng.random()),
        std::string(context) + ": RNG continuation drift");
}

void expect_customized_generation(
    std::string_view context,
    const DatasetScalar& minimum,
    const DatasetScalar& maximum,
    std::optional<DatasetValueKind> dtype,
    AttributeOwner owner,
    const NetworkCardinality& network,
    std::uint32_t seed,
    std::size_t workers)
{
    const DistributionSpec distribution = customized_distribution(
        DatasetScalar{minimum}, DatasetScalar{maximum});
    const BaseAttribute value(configured_spec(
        std::string(context), owner, distribution, dtype));
    NumpyRandomState actual_rng(seed);
    NumpyRandomState expected_rng(seed);
    const GeneratedData actual =
        value.generate_configured_data(network, actual_rng, workers);
    const GeneratedData expected = customized_expected_scalars(
        configured_count(owner, network),
        minimum,
        maximum,
        expected_rng);
    expect_generated_equal(actual, expected, context);
    expect(
        double_bits(actual_rng.random()) == double_bits(expected_rng.random()),
        std::string(context) + ": customized RNG continuation drift");
}

void test_resolvers()
{
    const std::array<std::pair<std::string_view, AttributeOwner>, 3U> owners = {{
        {"node", AttributeOwner::node},
        {"link", AttributeOwner::link},
        {"graph", AttributeOwner::graph},
    }};
    for (const auto& entry : owners)
    {
        expect(
            attribute::attribute_owner_from_string(entry.first) == entry.second,
            "owner resolver drift");
        expect(
            attribute::attribute_owner_name(entry.second) == entry.first,
            "owner name drift");
    }

    const std::array<std::pair<std::string_view, AttributeKind>, 5U> kinds = {{
        {"resource", AttributeKind::resource},
        {"extrema", AttributeKind::extrema},
        {"status", AttributeKind::status},
        {"position", AttributeKind::position},
        {"latency", AttributeKind::latency},
    }};
    for (const auto& entry : kinds)
    {
        expect(
            attribute::attribute_kind_from_string(entry.first) == entry.second,
            "kind resolver drift");
        expect(
            attribute::attribute_kind_name(entry.second) == entry.first,
            "kind name drift");
    }

    for (const std::string_view invalid : {"", "Node", "edge", " node"})
    {
        expect_base_exception(
            [invalid]
            {
                static_cast<void>(attribute::attribute_owner_from_string(invalid));
            },
            BaseAttributeErrorCode::invalid_owner,
            BaseAttributeOperation::resolve_owner);
    }
    for (const std::string_view invalid : {"", "Resource", "constraint", "status "})
    {
        expect_base_exception(
            [invalid]
            {
                static_cast<void>(attribute::attribute_kind_from_string(invalid));
            },
            BaseAttributeErrorCode::invalid_kind,
            BaseAttributeOperation::resolve_kind);
    }
}

void test_spec_snapshot_and_repr()
{
    BaseAttributeSpec input;
    input.name = "cpu";
    input.owner = AttributeOwner::link;
    input.kind = AttributeKind::resource;
    input.generative = true;
    input.distribution = uniform_distribution(
        DatasetScalar{std::int64_t{-3}}, DatasetScalar{7.5});
    input.dtype = DatasetValueKind::integer;
    input.originator = "capacity";
    input.is_constraint = true;

    const BaseAttribute value(input);
    input.name = "mutated";
    input.originator = "mutated";
    expect(value.spec().name == "cpu", "constructor did not own name");
    expect(value.spec().owner == AttributeOwner::link, "spec owner drift");
    expect(value.spec().kind == AttributeKind::resource, "spec kind drift");
    expect(value.spec().generative, "spec generative drift");
    expect(
        value.spec().distribution.kind == DistributionKind::uniform,
        "spec distribution drift");
    expect(
        value.spec().dtype == DatasetValueKind::integer,
        "spec dtype drift");
    expect(
        value.spec().originator == std::optional<std::string>{"capacity"},
        "spec originator ownership drift");
    expect(
        value.spec().is_constraint == std::optional<bool>{true},
        "spec constraint drift");
    expect(&value.spec() == &value.spec(), "spec reference is not stable");

    const BaseAttributeSnapshot expected = {
        {"name", DatasetScalar{std::string{"cpu"}}},
        {"owner", DatasetScalar{std::string{"link"}}},
        {"type", DatasetScalar{std::string{"resource"}}},
        {"generative", DatasetScalar{true}},
        {"distribution", DatasetScalar{std::string{"uniform"}}},
        {"dtype", DatasetScalar{std::string{"int"}}},
        {"low", DatasetScalar{std::int64_t{-3}}},
        {"high", DatasetScalar{7.5}},
        {"originator", DatasetScalar{std::string{"capacity"}}},
        {"is_constraint", DatasetScalar{true}},
    };
    expect_snapshot_equal(value.to_dict(), expected, "full BaseAttribute");
    const std::string expected_fields =
        "(name=cpu, owner=link, type=resource, generative=True, "
        "distribution=uniform, dtype=int, low=-3, high=7.5, "
        "originator=capacity, is_constraint=True)";
    expect(
        value.repr() == "BaseAttribute" + expected_fields,
        "default repr drift");
    expect(value.repr("Probe") == "Probe" + expected_fields, "custom repr drift");

    BaseAttributeSpec minimal_spec;
    minimal_spec.name = "status";
    const BaseAttribute minimal(std::move(minimal_spec));
    const BaseAttributeSnapshot minimal_expected = {
        {"name", DatasetScalar{std::string{"status"}}},
        {"owner", DatasetScalar{std::string{"node"}}},
        {"type", DatasetScalar{std::string{"status"}}},
        {"generative", DatasetScalar{false}},
    };
    expect_snapshot_equal(minimal.to_dict(), minimal_expected, "minimal BaseAttribute");
    expect(
        minimal.repr() ==
            "BaseAttribute(name=status, owner=node, type=status, generative=False)",
        "minimal repr drift");
}

void test_default_errors()
{
    BaseAttributeSpec spec;
    spec.name = "default";
    BaseAttribute value(std::move(spec));
    expect_base_exception(
        [&value]
        {
            static_cast<void>(value.generate_data());
        },
        BaseAttributeErrorCode::not_implemented,
        BaseAttributeOperation::generate_data,
        "Subclasses must implement this method.");
    expect_base_exception(
        [&value]
        {
            value.update_data();
        },
        BaseAttributeErrorCode::not_implemented,
        BaseAttributeOperation::update_data,
        "Subclasses must implement this method.");
}

void test_standard_distributions_and_sizing()
{
    const NetworkCardinality network{37U, 53U};
    expect_standard_generation(
        "uniform_int_node",
        uniform_distribution(
            DatasetScalar{std::int64_t{-7}}, DatasetScalar{std::int64_t{13}}),
        DatasetValueKind::integer,
        AttributeOwner::node,
        network,
        101U,
        1U);
    expect_standard_generation(
        "uniform_float_link",
        uniform_distribution(DatasetScalar{-2.25}, DatasetScalar{8.5}),
        DatasetValueKind::floating,
        AttributeOwner::link,
        network,
        102U,
        2U);

    for (const DatasetValueKind dtype : {
             DatasetValueKind::integer,
             DatasetValueKind::floating,
             DatasetValueKind::boolean})
    {
        expect_standard_generation(
            "normal_dtype",
            normal_distribution(-1.25, 2.5),
            dtype,
            AttributeOwner::graph,
            network,
            103U,
            1U);
        expect_standard_generation(
            "exponential_dtype",
            exponential_distribution(2.75),
            dtype,
            AttributeOwner::node,
            network,
            104U,
            2U);
        expect_standard_generation(
            "poisson_dtype",
            poisson_distribution(4.25),
            dtype,
            AttributeOwner::link,
            network,
            105U,
            8U);
    }

    expect_standard_generation(
        "absent_dtype_defaults_float",
        normal_distribution(0.0, 1.0),
        std::nullopt,
        AttributeOwner::node,
        network,
        106U,
        0U);
    expect_standard_generation(
        "base_ignores_dataset_reciprocal_flag",
        poisson_distribution(0.0, true),
        DatasetValueKind::integer,
        AttributeOwner::node,
        network,
        108U,
        8U);

    const DistributionSpec zero_distribution = normal_distribution(0.0, 1.0);
    const NetworkCardinality zero_network{};
    for (const AttributeOwner owner : {
             AttributeOwner::node,
             AttributeOwner::link,
             AttributeOwner::graph})
    {
        const BaseAttribute value(configured_spec(
            "zero", owner, zero_distribution, DatasetValueKind::floating));
        NumpyRandomState actual_rng(107U);
        NumpyRandomState untouched_rng(107U);
        const GeneratedData output =
            value.generate_configured_data(zero_network, actual_rng, 8U);
        expect(generated_size(output) == 0U, "zero cardinality produced values");
        expect(
            double_bits(actual_rng.random()) == double_bits(untouched_rng.random()),
            "zero cardinality consumed RNG state");
    }
}

void test_customized_generation()
{
    const NetworkCardinality network{41U, 59U};
    expect_customized_generation(
        "custom_bool_bounds",
        DatasetScalar{false},
        DatasetScalar{true},
        DatasetValueKind::integer,
        AttributeOwner::node,
        network,
        201U,
        1U);
    expect_customized_generation(
        "custom_mixed_bounds",
        DatasetScalar{std::int64_t{-7}},
        DatasetScalar{4.5},
        DatasetValueKind::boolean,
        AttributeOwner::graph,
        network,
        202U,
        2U);
    expect_customized_generation(
        "custom_above_2pow53",
        DatasetScalar{std::int64_t{9'007'199'254'740'993LL}},
        DatasetScalar{9'007'199'254'741'000.0},
        std::nullopt,
        AttributeOwner::link,
        network,
        203U,
        8U);
    expect_customized_generation(
        "custom_integral_span_before_float_conversion",
        DatasetScalar{std::int64_t{4'611'686'018'427'387'904LL}},
        DatasetScalar{std::int64_t{4'611'686'018'427'388'929LL}},
        std::nullopt,
        AttributeOwner::node,
        NetworkCardinality{10'000U, 0U},
        7U,
        8U);
    expect_customized_generation(
        "custom_negative_infinity",
        DatasetScalar{-std::numeric_limits<double>::infinity()},
        DatasetScalar{1.0},
        DatasetValueKind::floating,
        AttributeOwner::node,
        network,
        205U,
        2U);
    expect_customized_generation(
        "custom_positive_infinity",
        DatasetScalar{0.0},
        DatasetScalar{std::numeric_limits<double>::infinity()},
        DatasetValueKind::floating,
        AttributeOwner::link,
        network,
        206U,
        8U);

    const DatasetScalar minimum{std::int64_t{-11}};
    const DatasetScalar maximum{6.25};
    const DistributionSpec distribution =
        customized_distribution(DatasetScalar{minimum}, DatasetScalar{maximum});
    std::optional<GeneratedData> canonical;
    for (const std::optional<DatasetValueKind> dtype : {
             std::optional<DatasetValueKind>{},
             std::optional<DatasetValueKind>{DatasetValueKind::integer},
             std::optional<DatasetValueKind>{DatasetValueKind::floating},
             std::optional<DatasetValueKind>{DatasetValueKind::boolean}})
    {
        const BaseAttribute value(configured_spec(
            "dtype_ignored", AttributeOwner::node, distribution, dtype));
        NumpyRandomState rng(204U);
        const GeneratedData output = value.generate_configured_data(network, rng, 2U);
        expect(
            output.value_kind == DatasetValueKind::floating &&
                std::holds_alternative<std::vector<double>>(output.values),
            "customized dtype was not ignored");
        if (!canonical.has_value())
        {
            canonical = output;
        }
        else
        {
            expect_generated_equal(output, *canonical, "customized dtype invariance");
        }
    }
}

void test_invalid_generation_without_rng_consumption()
{
    const NetworkCardinality network{19U, 23U};

    BaseAttributeSpec non_generative_spec;
    non_generative_spec.name = "non_generative";
    non_generative_spec.distribution.kind = DistributionKind::customized;
    const BaseAttribute non_generative(std::move(non_generative_spec));
    expect_error_without_rng_consumption(
        301U,
        [&non_generative, &network](NumpyRandomState& rng)
        {
            expect_base_exception(
                [&non_generative, &network, &rng]
                {
                    static_cast<void>(non_generative.generate_configured_data(
                        network, rng, 8U));
                },
                BaseAttributeErrorCode::not_generative,
                BaseAttributeOperation::generate_configured_data);
        });

    for (const DistributionKind kind : {
             DistributionKind::none,
             static_cast<DistributionKind>(255U)})
    {
        DistributionSpec distribution;
        distribution.kind = kind;
        const BaseAttribute value(configured_spec(
            "unsupported", AttributeOwner::node, distribution));
        expect_error_without_rng_consumption(
            302U,
            [&value, &network](NumpyRandomState& rng)
            {
                expect_base_exception(
                    [&value, &network, &rng]
                    {
                        static_cast<void>(value.generate_configured_data(
                            network, rng, 1U));
                    },
                    BaseAttributeErrorCode::unsupported_distribution,
                    BaseAttributeOperation::generate_configured_data);
            });
    }

    struct InvalidCustomCase {
        std::optional<DatasetScalar> minimum;
        std::optional<DatasetScalar> maximum;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const std::vector<InvalidCustomCase> invalid_custom = {
        {std::nullopt, DatasetScalar{1.0}},
        {DatasetScalar{0.0}, std::nullopt},
        {DatasetScalar{std::monostate{}}, DatasetScalar{1.0}},
        {DatasetScalar{std::string{"0"}}, DatasetScalar{1.0}},
        {DatasetScalar{nan}, DatasetScalar{1.0}},
        {DatasetScalar{0.0}, DatasetScalar{nan}},
        {DatasetScalar{infinity}, DatasetScalar{infinity}},
        {DatasetScalar{3.0}, DatasetScalar{3.0}},
        {DatasetScalar{4.0}, DatasetScalar{-2.0}},
    };
    for (std::size_t index = 0U; index < invalid_custom.size(); ++index)
    {
        const BaseAttribute value(configured_spec(
            "invalid_custom",
            AttributeOwner::node,
            customized_distribution(
                invalid_custom[index].minimum,
                invalid_custom[index].maximum),
            DatasetValueKind::floating));
        expect_error_without_rng_consumption(
            static_cast<std::uint32_t>(310U + index),
            [&value, &network](NumpyRandomState& rng)
            {
                expect_base_exception(
                    [&value, &network, &rng]
                    {
                        static_cast<void>(value.generate_configured_data(
                            network, rng, 8U));
                    },
                    BaseAttributeErrorCode::invalid_custom_range,
                    BaseAttributeOperation::generate_configured_data);
            });
    }

    auto expect_standard_error = [&network](
                                     const DistributionSpec& distribution,
                                     DatasetValueKind dtype,
                                     DatasetErrorCode code,
                                     std::uint32_t seed)
    {
        const BaseAttribute value(configured_spec(
            "invalid_standard", AttributeOwner::node, distribution, dtype));
        expect_error_without_rng_consumption(
            seed,
            [&value, &network, code](NumpyRandomState& rng)
            {
                expect_dataset_exception(
                    [&value, &network, &rng]
                    {
                        static_cast<void>(value.generate_configured_data(
                            network, rng, 8U));
                    },
                    code);
            });
    };

    DistributionSpec missing_uniform;
    missing_uniform.kind = DistributionKind::uniform;
    missing_uniform.high = DatasetScalar{1.0};
    expect_standard_error(
        missing_uniform,
        DatasetValueKind::floating,
        DatasetErrorCode::missing_parameter,
        401U);
    expect_standard_error(
        normal_distribution(0.0, -1.0),
        DatasetValueKind::floating,
        DatasetErrorCode::rng_backend_failure,
        402U);
    expect_standard_error(
        exponential_distribution(-1.0),
        DatasetValueKind::floating,
        DatasetErrorCode::rng_backend_failure,
        403U);
    expect_standard_error(
        poisson_distribution(-1.0),
        DatasetValueKind::integer,
        DatasetErrorCode::rng_backend_failure,
        404U);
    DistributionSpec missing_normal_location;
    missing_normal_location.kind = DistributionKind::normal;
    missing_normal_location.scale = DatasetScalar{1.0};
    expect_standard_error(
        missing_normal_location,
        DatasetValueKind::floating,
        DatasetErrorCode::invalid_parameter,
        405U);
    expect_standard_error(
        uniform_distribution(DatasetScalar{false}, DatasetScalar{true}),
        DatasetValueKind::boolean,
        DatasetErrorCode::uniform_boolean_uninitialized,
        406U);
    expect_standard_error(
        normal_distribution(0.0, 1.0),
        static_cast<DatasetValueKind>(255U),
        DatasetErrorCode::invalid_value_kind,
        407U);

    const NetworkCardinality zero_network{};
    const BaseAttribute invalid_zero(configured_spec(
        "invalid_zero",
        AttributeOwner::node,
        customized_distribution(DatasetScalar{2.0}, DatasetScalar{1.0})));
    expect_error_without_rng_consumption(
        408U,
        [&invalid_zero, &zero_network](NumpyRandomState& rng)
        {
            expect_base_exception(
                [&invalid_zero, &zero_network, &rng]
                {
                    static_cast<void>(invalid_zero.generate_configured_data(
                        zero_network, rng, 1U));
                },
                BaseAttributeErrorCode::invalid_custom_range,
                BaseAttributeOperation::generate_configured_data);
        });
}

void test_worker_invariance()
{
    const std::array<std::size_t, 4U> workers = {0U, 1U, 2U, 8U};

    const NetworkCardinality standard_network{131'089U, 0U};
    const DistributionSpec standard_distribution = exponential_distribution(2.5);
    const BaseAttribute standard(configured_spec(
        "worker_standard",
        AttributeOwner::node,
        standard_distribution,
        DatasetValueKind::integer));
    DistributionRequest request;
    request.count = standard_network.num_nodes;
    request.value_kind = DatasetValueKind::integer;
    request.distribution = standard_distribution;
    NumpyRandomState standard_expected_rng(501U);
    const GeneratedData standard_expected = utils::generate_data_with_distribution(
        request, standard_expected_rng, 1U);
    const std::uint64_t standard_next =
        double_bits(standard_expected_rng.random());
    for (const std::size_t width : workers)
    {
        NumpyRandomState rng(501U);
        const GeneratedData output =
            standard.generate_configured_data(standard_network, rng, width);
        expect_generated_equal(output, standard_expected, "standard worker invariance");
        expect(
            double_bits(rng.random()) == standard_next,
            "standard worker RNG continuation drift");
    }

    const NetworkCardinality custom_network{65'537U, 0U};
    const double custom_minimum = -3.5;
    const double custom_maximum = 9.25;
    const BaseAttribute customized(configured_spec(
        "worker_custom",
        AttributeOwner::node,
        customized_distribution(
            DatasetScalar{custom_minimum}, DatasetScalar{custom_maximum}),
        DatasetValueKind::boolean));
    NumpyRandomState custom_expected_rng(502U);
    const GeneratedData custom_expected = customized_expected(
        custom_network.num_nodes,
        custom_minimum,
        custom_maximum,
        custom_expected_rng);
    const std::uint64_t custom_next = double_bits(custom_expected_rng.random());
    for (const std::size_t width : workers)
    {
        NumpyRandomState rng(502U);
        const GeneratedData output =
            customized.generate_configured_data(custom_network, rng, width);
        expect_generated_equal(output, custom_expected, "custom worker invariance");
        expect(
            double_bits(rng.random()) == custom_next,
            "custom worker RNG continuation drift");
    }
}

void test_concurrent_independent_callers()
{
    constexpr std::size_t caller_count = 4U;
    const std::array<std::size_t, caller_count> workers = {0U, 1U, 2U, 8U};
    const std::array<std::uint32_t, caller_count> seeds = {
        601U, 602U, 603U, 604U};
    const NetworkCardinality network{32'777U, 0U};
    const double minimum = -5.0;
    const double maximum = 11.5;
    const BaseAttribute shared(configured_spec(
        "concurrent",
        AttributeOwner::node,
        customized_distribution(DatasetScalar{minimum}, DatasetScalar{maximum}),
        DatasetValueKind::integer));

    std::array<GeneratedData, caller_count> expected;
    std::array<std::uint64_t, caller_count> expected_next{};
    for (std::size_t caller = 0U; caller < caller_count; ++caller)
    {
        NumpyRandomState rng(seeds[caller]);
        expected[caller] = customized_expected(
            network.num_nodes, minimum, maximum, rng);
        expected_next[caller] = double_bits(rng.random());
    }

    std::array<GeneratedData, caller_count> actual;
    std::array<std::uint64_t, caller_count> actual_next{};
    std::array<std::exception_ptr, caller_count> errors{};
    std::vector<std::thread> callers;
    callers.reserve(caller_count);
    for (std::size_t caller = 0U; caller < caller_count; ++caller)
    {
        callers.emplace_back(
            [&, caller]
            {
                try
                {
                    NumpyRandomState rng(seeds[caller]);
                    actual[caller] = shared.generate_configured_data(
                        network, rng, workers[caller]);
                    actual_next[caller] = double_bits(rng.random());
                }
                catch (...)
                {
                    errors[caller] = std::current_exception();
                }
            });
    }
    for (std::thread& caller : callers)
    {
        caller.join();
    }
    for (std::size_t caller = 0U; caller < caller_count; ++caller)
    {
        if (errors[caller])
        {
            std::rethrow_exception(errors[caller]);
        }
        expect_generated_equal(
            actual[caller], expected[caller], "concurrent independent caller");
        expect(
            actual_next[caller] == expected_next[caller],
            "concurrent caller RNG continuation drift");
    }
}

}  // namespace

int main()
{
    try
    {
        test_resolvers();
        test_spec_snapshot_and_repr();
        test_default_errors();
        test_standard_distributions_and_sizing();
        test_customized_generation();
        test_invalid_generation_without_rng_consumption();
        test_worker_invariance();
        test_concurrent_independent_callers();
        std::cout << "base_attribute_unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "base_attribute_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
