#include "attribute/attribute_method.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;

using attribute::AttributeMethodErrorCode;
using attribute::AttributeMethodException;
using attribute::AttributeMethodOperation;
using attribute::AttributeNumber;
using attribute::ComparisonOperation;
using attribute::ConstraintRestriction;
using attribute::ResourceUpdateOperation;
using attribute::SatisfiabilityResult;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

struct CapturedError
{
    AttributeMethodErrorCode code;
    AttributeMethodOperation operation;
    std::string message;
};

template <typename Callable>
CapturedError expect_attribute_exception(
    Callable&& callable,
    AttributeMethodErrorCode code,
    AttributeMethodOperation operation)
{
    try
    {
        callable();
    }
    catch (const AttributeMethodException& error)
    {
        expect(error.code() == code, "attribute-method error code mismatch");
        expect(
            error.operation() == operation,
            "attribute-method operation mismatch");
        return {error.code(), error.operation(), error.what()};
    }
    throw std::runtime_error("expected AttributeMethodException");
}

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void expect_boolean(const AttributeNumber& value, bool expected)
{
    const bool* actual = std::get_if<bool>(&value);
    expect(actual != nullptr, "expected boolean AttributeNumber lane");
    expect(*actual == expected, "boolean AttributeNumber value mismatch");
}

void expect_integer(const AttributeNumber& value, std::int64_t expected)
{
    const std::int64_t* actual = std::get_if<std::int64_t>(&value);
    expect(actual != nullptr, "expected integer AttributeNumber lane");
    expect(*actual == expected, "integer AttributeNumber value mismatch");
}

void expect_floating_bits(const AttributeNumber& value, double expected)
{
    const double* actual = std::get_if<double>(&value);
    expect(actual != nullptr, "expected floating AttributeNumber lane");
    expect(
        double_bits(*actual) == double_bits(expected),
        "floating AttributeNumber bit mismatch");
}

void test_specs_and_enums()
{
    attribute::ResourceMethodSpec resource;
    resource.definition_id = 17U;
    resource.generative = true;
    const attribute::ResourceMethodSpec copied = resource;
    expect(copied.definition_id == 17U, "resource spec copy ID mismatch");
    expect(copied.generative, "resource spec copy flag mismatch");

    attribute::ExtremaMethodSpec extrema;
    extrema.declared_owner = attribute::AttributeOwner::graph;
    extrema.origin_registry = attribute::ExtremaOriginRegistry::link;
    extrema.originator_id = 29U;
    const attribute::ExtremaMethodSpec moved = std::move(extrema);
    expect(
        moved.declared_owner == attribute::AttributeOwner::graph,
        "extrema declared owner mismatch");
    expect(
        moved.origin_registry == attribute::ExtremaOriginRegistry::link,
        "extrema origin registry mismatch");
    expect(moved.originator_id == 29U, "extrema originator ID mismatch");

    static_assert(!attribute::InformationMethodSpec::is_constraint);
    static_assert(attribute::ConstraintMethodSpec::is_constraint);
    attribute::ConstraintMethodSpec constraint;
    expect(
        constraint.restriction == ConstraintRestriction::hard,
        "constraint spec default mismatch");
    constraint.restriction = ConstraintRestriction::soft;
    const attribute::ConstraintMethodSpec constraint_copy = constraint;
    expect(
        constraint_copy.restriction == ConstraintRestriction::soft,
        "constraint spec copy mismatch");

    expect_boolean(AttributeNumber{false}, false);
    expect_integer(AttributeNumber{std::int64_t{7}}, 7);
    expect_floating_bits(AttributeNumber{-0.0}, -0.0);
    expect(
        attribute::attribute_number_kind(AttributeNumber{false}) ==
            attribute::AttributeNumberKind::boolean,
        "boolean number-kind mismatch");
    expect(
        attribute::attribute_number_kind(AttributeNumber{std::int64_t{7}}) ==
            attribute::AttributeNumberKind::integer,
        "integer number-kind mismatch");
    expect(
        attribute::attribute_number_kind(AttributeNumber{-0.0}) ==
            attribute::AttributeNumberKind::floating,
        "floating number-kind mismatch");
}

void test_resolvers()
{
    for (const std::string_view value : {"+", "add"})
    {
        expect(
            attribute::resource_update_operation_from_string(value) ==
                ResourceUpdateOperation::add,
            "resource add resolver mismatch");
    }
    for (const std::string_view value : {"-", "sub"})
    {
        expect(
            attribute::resource_update_operation_from_string(value) ==
                ResourceUpdateOperation::subtract,
            "resource subtract resolver mismatch");
    }
    for (const std::string_view value : {"", "Add", "subtract", "++", "="})
    {
        expect_attribute_exception(
            [value]
            {
                (void)attribute::resource_update_operation_from_string(value);
            },
            AttributeMethodErrorCode::unsupported_update_operation,
            AttributeMethodOperation::resolve_update);
    }

    for (const std::string_view value : {">=", "ge"})
    {
        expect(
            attribute::comparison_operation_from_string(value) ==
                ComparisonOperation::greater_equal,
            "greater-equal resolver mismatch");
    }
    for (const std::string_view value : {"<=", "le"})
    {
        expect(
            attribute::comparison_operation_from_string(value) ==
                ComparisonOperation::less_equal,
            "less-equal resolver mismatch");
    }
    expect(
        attribute::comparison_operation_from_string("eq") ==
            ComparisonOperation::equal,
        "equal resolver mismatch");
    for (const std::string_view value : {"", "==", "EQ", "gt", "<"})
    {
        expect_attribute_exception(
            [value]
            {
                (void)attribute::comparison_operation_from_string(value);
            },
            AttributeMethodErrorCode::unsupported_comparison,
            AttributeMethodOperation::resolve_comparison);
    }

    expect(
        attribute::constraint_restriction_from_string("hard") ==
            ConstraintRestriction::hard,
        "hard restriction resolver mismatch");
    expect(
        attribute::constraint_restriction_from_string("soft") ==
            ConstraintRestriction::soft,
        "soft restriction resolver mismatch");
    for (const std::string_view value : {"", "Hard", "SOFT", " hard"})
    {
        expect_attribute_exception(
            [value]
            {
                (void)attribute::constraint_restriction_from_string(value);
            },
            AttributeMethodErrorCode::invalid_restriction,
            AttributeMethodOperation::resolve_restriction);
    }
}

void test_resource_updates()
{
    AttributeNumber physical = std::int64_t{10};
    expect(
        attribute::update_resource_value(
            AttributeNumber{std::int64_t{3}},
            physical,
            ResourceUpdateOperation::add),
        "integer add return mismatch");
    expect_integer(physical, 13);

    expect(
        attribute::update_resource_value(
            AttributeNumber{std::int64_t{4}},
            physical,
            ResourceUpdateOperation::subtract,
            true,
            "cpu"),
        "integer subtract return mismatch");
    expect_integer(physical, 9);

    AttributeNumber guarded = std::int64_t{5};
    const CapturedError guard_error = expect_attribute_exception(
        [&guarded]
        {
            (void)attribute::update_resource_value(
                AttributeNumber{std::int64_t{6}},
                guarded,
                ResourceUpdateOperation::subtract,
                true,
                "cpu");
        },
        AttributeMethodErrorCode::insufficient_resource,
        AttributeMethodOperation::update_resource);
    expect(!guard_error.message.empty(), "guard error message is empty");
    expect_integer(guarded, 5);

    expect(
        attribute::update_resource_value(
            AttributeNumber{std::int64_t{6}},
            guarded,
            ResourceUpdateOperation::subtract,
            false,
            "cpu"),
        "unsafe subtract return mismatch");
    expect_integer(guarded, -1);

    AttributeNumber boolean = true;
    expect(
        attribute::update_resource_value(
            AttributeNumber{true},
            boolean,
            ResourceUpdateOperation::add),
        "boolean add return mismatch");
    expect_integer(boolean, 2);

    AttributeNumber mixed_integer = std::int64_t{7};
    (void)attribute::update_resource_value(
        AttributeNumber{true},
        mixed_integer,
        ResourceUpdateOperation::subtract,
        false);
    expect_integer(mixed_integer, 6);

    AttributeNumber mixed_floating = std::int64_t{2};
    (void)attribute::update_resource_value(
        AttributeNumber{0.5},
        mixed_floating,
        ResourceUpdateOperation::add);
    expect_floating_bits(mixed_floating, 2.5);

    AttributeNumber negative_zero = -0.0;
    (void)attribute::update_resource_value(
        AttributeNumber{std::int64_t{0}},
        negative_zero,
        ResourceUpdateOperation::subtract,
        false);
    expect_floating_bits(negative_zero, -0.0);

    const double quiet_nan = std::numeric_limits<double>::quiet_NaN();
    AttributeNumber nan_value = quiet_nan;
    (void)attribute::update_resource_value(
        AttributeNumber{1.0},
        nan_value,
        ResourceUpdateOperation::add);
    const double* nan_output = std::get_if<double>(&nan_value);
    expect(nan_output != nullptr && std::isnan(*nan_output), "NaN update mismatch");

    AttributeNumber infinity = std::numeric_limits<double>::infinity();
    (void)attribute::update_resource_value(
        AttributeNumber{1.0},
        infinity,
        ResourceUpdateOperation::subtract,
        true);
    const double* infinity_output = std::get_if<double>(&infinity);
    expect(
        infinity_output != nullptr && std::isinf(*infinity_output) &&
            *infinity_output > 0.0,
        "infinity update mismatch");

    AttributeNumber maximum = std::numeric_limits<std::int64_t>::max();
    expect_attribute_exception(
        [&maximum]
        {
            (void)attribute::update_resource_value(
                AttributeNumber{std::int64_t{1}},
                maximum,
                ResourceUpdateOperation::add);
        },
        AttributeMethodErrorCode::numeric_range,
        AttributeMethodOperation::update_resource);
    expect_integer(maximum, std::numeric_limits<std::int64_t>::max());

    AttributeNumber minimum = std::numeric_limits<std::int64_t>::min();
    expect_attribute_exception(
        [&minimum]
        {
            (void)attribute::update_resource_value(
                AttributeNumber{std::int64_t{1}},
                minimum,
                ResourceUpdateOperation::subtract,
                false);
        },
        AttributeMethodErrorCode::numeric_range,
        AttributeMethodOperation::update_resource);
    expect_integer(minimum, std::numeric_limits<std::int64_t>::min());

    AttributeNumber invalid_operation_value = std::int64_t{4};
    expect_attribute_exception(
        [&invalid_operation_value]
        {
            (void)attribute::update_resource_value(
                AttributeNumber{std::int64_t{1}},
                invalid_operation_value,
                static_cast<ResourceUpdateOperation>(255),
                false);
        },
        AttributeMethodErrorCode::unsupported_update_operation,
        AttributeMethodOperation::update_resource);
    expect_integer(invalid_operation_value, 4);
}

void test_satisfiability()
{
    SatisfiabilityResult result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::int64_t{5}},
        AttributeNumber{std::int64_t{3}},
        ComparisonOperation::greater_equal,
        ConstraintRestriction::hard);
    expect(result.flag, "hard ge flag mismatch");
    expect_integer(result.offset, -2);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::int64_t{2}},
        AttributeNumber{std::int64_t{3}},
        ComparisonOperation::greater_equal,
        ConstraintRestriction::hard);
    expect(!result.flag, "hard ge false flag mismatch");
    expect_integer(result.offset, 1);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::int64_t{-4}},
        AttributeNumber{std::int64_t{1}},
        ComparisonOperation::less_equal,
        ConstraintRestriction::hard);
    expect(result.flag, "hard le flag mismatch");
    expect_integer(result.offset, -5);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::int64_t{-4}},
        AttributeNumber{std::int64_t{1}},
        ComparisonOperation::equal,
        ConstraintRestriction::hard);
    expect(!result.flag, "hard eq flag mismatch");
    expect_integer(result.offset, 5);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{true},
        AttributeNumber{false},
        ComparisonOperation::greater_equal,
        ConstraintRestriction::hard);
    expect(result.flag, "boolean ge flag mismatch");
    expect_integer(result.offset, -1);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::int64_t{2}},
        AttributeNumber{2.5},
        ComparisonOperation::less_equal,
        ConstraintRestriction::hard);
    expect(result.flag, "mixed le flag mismatch");
    expect_floating_bits(result.offset, -0.5);

    constexpr std::int64_t beyond_binary64_integer =
        std::int64_t{9007199254740993LL};
    constexpr double rounded_binary64_integer = 9007199254740992.0;
    result = attribute::calculate_satisfiability_values(
        AttributeNumber{beyond_binary64_integer},
        AttributeNumber{rounded_binary64_integer},
        ComparisonOperation::equal,
        ConstraintRestriction::hard);
    expect(!result.flag, "mixed equality rounded an int64 through double");
    expect_floating_bits(result.offset, 0.0);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{rounded_binary64_integer},
        AttributeNumber{beyond_binary64_integer},
        ComparisonOperation::equal,
        ConstraintRestriction::hard);
    expect(!result.flag, "reverse mixed equality rounded int64 through double");
    expect_floating_bits(result.offset, 0.0);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::numeric_limits<std::int64_t>::max()},
        AttributeNumber{9223372036854775808.0},
        ComparisonOperation::greater_equal,
        ConstraintRestriction::hard);
    expect(!result.flag, "int64 maximum compared equal to +2^63");
    expect_floating_bits(result.offset, 0.0);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::numeric_limits<std::int64_t>::min()},
        AttributeNumber{-9223372036854775808.0},
        ComparisonOperation::equal,
        ConstraintRestriction::hard);
    expect(result.flag, "int64 minimum did not compare equal to exact -2^63");
    expect_floating_bits(result.offset, 0.0);

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{-0.0},
        AttributeNumber{0.0},
        ComparisonOperation::equal,
        ConstraintRestriction::hard);
    expect(result.flag, "signed-zero equality flag mismatch");
    expect_floating_bits(result.offset, 0.0);

    const double quiet_nan = std::numeric_limits<double>::quiet_NaN();
    result = attribute::calculate_satisfiability_values(
        AttributeNumber{quiet_nan},
        AttributeNumber{1.0},
        ComparisonOperation::less_equal,
        ConstraintRestriction::hard);
    expect(!result.flag, "NaN hard flag mismatch");
    const double* nan_offset = std::get_if<double>(&result.offset);
    expect(nan_offset != nullptr && std::isnan(*nan_offset), "NaN offset mismatch");

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{quiet_nan},
        AttributeNumber{1.0},
        ComparisonOperation::less_equal,
        ConstraintRestriction::soft);
    expect(result.flag, "soft restriction did not mask flag");
    nan_offset = std::get_if<double>(&result.offset);
    expect(
        nan_offset != nullptr && std::isnan(*nan_offset),
        "soft NaN offset mismatch");

    result = attribute::calculate_satisfiability_values(
        AttributeNumber{std::int64_t{8}},
        AttributeNumber{std::int64_t{2}},
        ComparisonOperation::less_equal,
        ConstraintRestriction::soft);
    expect(result.flag, "soft false comparison was not masked");
    expect_integer(result.offset, 6);

    expect_attribute_exception(
        []
        {
            (void)attribute::calculate_satisfiability_values(
                AttributeNumber{std::numeric_limits<std::int64_t>::min()},
                AttributeNumber{std::numeric_limits<std::int64_t>::max()},
                ComparisonOperation::greater_equal,
                ConstraintRestriction::hard);
        },
        AttributeMethodErrorCode::numeric_range,
        AttributeMethodOperation::calculate_satisfiability);

    expect_attribute_exception(
        []
        {
            (void)attribute::calculate_satisfiability_values(
                AttributeNumber{std::int64_t{1}},
                AttributeNumber{std::int64_t{2}},
                static_cast<ComparisonOperation>(255),
                ConstraintRestriction::hard);
        },
        AttributeMethodErrorCode::unsupported_comparison,
        AttributeMethodOperation::calculate_satisfiability);

    expect_attribute_exception(
        []
        {
            (void)attribute::calculate_satisfiability_values(
                AttributeNumber{std::int64_t{1}},
                AttributeNumber{std::int64_t{2}},
                ComparisonOperation::less_equal,
                static_cast<ConstraintRestriction>(255));
        },
        AttributeMethodErrorCode::invalid_restriction,
        AttributeMethodOperation::calculate_satisfiability);
}

void test_double_satisfiability_batch()
{
    const double quiet_nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> virtual_values = {
        -std::numeric_limits<double>::infinity(),
        -0.0,
        0.0,
        1.25,
        -3.5,
        std::numeric_limits<double>::infinity(),
        quiet_nan,
        7.0,
    };
    const std::vector<double> physical_values = {
        -std::numeric_limits<double>::infinity(),
        0.0,
        -0.0,
        1.0,
        -4.0,
        2.0,
        1.0,
        7.0,
    };
    const std::vector<ComparisonOperation> operations = {
        ComparisonOperation::greater_equal,
        ComparisonOperation::less_equal,
        ComparisonOperation::equal,
    };
    const std::vector<ConstraintRestriction> restrictions = {
        ConstraintRestriction::hard,
        ConstraintRestriction::soft,
    };

    expect(
        attribute::double_satisfiability_batch_worker_count(
            0U,
            8U) == 0U,
        "empty double batch selected workers");
    expect(
        attribute::double_satisfiability_batch_worker_count(
            1U,
            999U) == 1U,
        "single-item double batch did not cap workers");
    expect(
        attribute::double_satisfiability_batch_worker_count(
            virtual_values.size(),
            0U) == 1U,
        "zero-configured double batch must remain sequential");

    for (const ComparisonOperation operation : operations)
    {
        for (const ConstraintRestriction restriction : restrictions)
        {
            std::vector<std::uint8_t> expected_flags(virtual_values.size());
            std::vector<double> expected_offsets(virtual_values.size());
            for (std::size_t index = 0U; index < virtual_values.size(); ++index)
            {
                const SatisfiabilityResult scalar =
                    attribute::calculate_satisfiability_values(
                        AttributeNumber{virtual_values[index]},
                        AttributeNumber{physical_values[index]},
                        operation,
                        restriction);
                expected_flags[index] = static_cast<std::uint8_t>(scalar.flag);
                expected_offsets[index] = std::get<double>(scalar.offset);
            }

            for (std::size_t workers = 0U; workers <= 8U; ++workers)
            {
                std::vector<std::uint8_t> flags(virtual_values.size(), 0xA5U);
                std::vector<double> offsets(virtual_values.size(), 123.0);
                attribute::calculate_satisfiability_values_double_batch(
                    virtual_values,
                    physical_values,
                    operation,
                    restriction,
                    flags,
                    offsets,
                    workers);
                expect(flags == expected_flags, "double batch flag drift");
                for (std::size_t index = 0U; index < offsets.size(); ++index)
                {
                    expect(
                        double_bits(offsets[index])
                            == double_bits(expected_offsets[index]),
                        "double batch offset bit drift");
                }
            }

            for (std::size_t workers = 1U; workers <= 8U; ++workers)
            {
                std::vector<std::uint8_t> virtual_alias_flags(
                    virtual_values.size(), 0xA5U);
                std::vector<double> virtual_alias = virtual_values;
                attribute::calculate_satisfiability_values_double_batch(
                    virtual_alias,
                    physical_values,
                    operation,
                    restriction,
                    virtual_alias_flags,
                    virtual_alias,
                    workers);
                expect(
                    virtual_alias_flags == expected_flags,
                    "double batch virtual-alias flag drift");
                for (std::size_t index = 0U; index < virtual_alias.size(); ++index)
                {
                    expect(
                        double_bits(virtual_alias[index])
                            == double_bits(expected_offsets[index]),
                        "double batch offsets==virtual alias drift");
                }

                std::vector<std::uint8_t> physical_alias_flags(
                    physical_values.size(), 0xA5U);
                std::vector<double> physical_alias = physical_values;
                attribute::calculate_satisfiability_values_double_batch(
                    virtual_values,
                    physical_alias,
                    operation,
                    restriction,
                    physical_alias_flags,
                    physical_alias,
                    workers);
                expect(
                    physical_alias_flags == expected_flags,
                    "double batch physical-alias flag drift");
                for (std::size_t index = 0U; index < physical_alias.size(); ++index)
                {
                    expect(
                        double_bits(physical_alias[index])
                            == double_bits(expected_offsets[index]),
                        "double batch offsets==physical alias drift");
                }
            }
        }
    }

    std::vector<std::exception_ptr> concurrent_errors(8U);
    std::vector<std::thread> concurrent_callers;
    concurrent_callers.reserve(concurrent_errors.size());
    for (std::size_t caller = 0U; caller < concurrent_errors.size(); ++caller)
    {
        concurrent_callers.emplace_back(
            [&, caller]
            {
                try
                {
                    for (std::size_t iteration = 0U; iteration < 32U; ++iteration)
                    {
                        const ComparisonOperation operation =
                            operations[(caller + iteration) % operations.size()];
                        const ConstraintRestriction restriction =
                            restrictions[(caller + iteration) % restrictions.size()];
                        std::vector<std::uint8_t> flags(virtual_values.size());
                        std::vector<double> offsets(virtual_values.size());
                        attribute::calculate_satisfiability_values_double_batch(
                            virtual_values,
                            physical_values,
                            operation,
                            restriction,
                            flags,
                            offsets,
                            caller + 1U);
                        for (std::size_t index = 0U;
                             index < virtual_values.size();
                             ++index)
                        {
                            const SatisfiabilityResult scalar =
                                attribute::calculate_satisfiability_values(
                                    AttributeNumber{virtual_values[index]},
                                    AttributeNumber{physical_values[index]},
                                    operation,
                                    restriction);
                            expect(
                                flags[index]
                                    == static_cast<std::uint8_t>(scalar.flag),
                                "concurrent double batch flag drift");
                            expect(
                                double_bits(offsets[index])
                                    == double_bits(std::get<double>(scalar.offset)),
                                "concurrent double batch offset drift");
                        }
                    }
                }
                catch (...)
                {
                    concurrent_errors[caller] = std::current_exception();
                }
            });
    }
    for (std::thread& caller : concurrent_callers)
    {
        caller.join();
    }
    for (const std::exception_ptr& error : concurrent_errors)
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }

    std::vector<std::uint8_t> short_flags(virtual_values.size() - 1U);
    std::vector<double> offsets(virtual_values.size());
    expect_attribute_exception(
        [&]
        {
            attribute::calculate_satisfiability_values_double_batch(
                virtual_values,
                physical_values,
                ComparisonOperation::less_equal,
                ConstraintRestriction::hard,
                short_flags,
                offsets,
                1U);
        },
        AttributeMethodErrorCode::invalid_batch_shape,
        AttributeMethodOperation::calculate_satisfiability_batch);

    std::vector<std::uint8_t> flags(virtual_values.size());
    expect_attribute_exception(
        [&]
        {
            attribute::calculate_satisfiability_values_double_batch(
                virtual_values,
                physical_values,
                static_cast<ComparisonOperation>(255),
                ConstraintRestriction::hard,
                flags,
                offsets,
                1U);
        },
        AttributeMethodErrorCode::unsupported_comparison,
        AttributeMethodOperation::calculate_satisfiability_batch);
    expect_attribute_exception(
        [&]
        {
            attribute::calculate_satisfiability_values_double_batch(
                virtual_values,
                physical_values,
                ComparisonOperation::less_equal,
                static_cast<ConstraintRestriction>(255),
                flags,
                offsets,
                1U);
        },
        AttributeMethodErrorCode::invalid_restriction,
        AttributeMethodOperation::calculate_satisfiability_batch);
}

}  // namespace

int main()
{
    try
    {
        test_specs_and_enums();
        test_resolvers();
        test_resource_updates();
        test_satisfiability();
        test_double_satisfiability_batch();
        std::cout << "attribute_method_unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "attribute_method_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
