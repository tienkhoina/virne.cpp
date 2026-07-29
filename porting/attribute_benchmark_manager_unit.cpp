#include "attribute/attribute_benchmark_manager.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;

using attribute::AttributeBenchmarkDescriptor;
using attribute::AttributeBenchmarkEntry;
using attribute::AttributeBenchmarkErrorCode;
using attribute::AttributeBenchmarkException;
using attribute::AttributeBenchmarkId;
using attribute::AttributeBenchmarkMap;
using attribute::AttributeBenchmarkMatrix;
using attribute::AttributeBenchmarkOperation;
using attribute::AttributeBenchmarkRequest;
using attribute::AttributeBenchmarks;
using attribute::AttributeKind;
using attribute::PreparedAttributeBenchmarkData;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "unexpected binary64 size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(std::uint32_t bits) noexcept {
    float value = 0.0F;
    static_assert(sizeof(bits) == sizeof(value), "unexpected binary32 size");
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

std::uint64_t promoted_float_bits(std::uint32_t bits) noexcept {
    const std::uint32_t exponent = bits & 0x7f800000U;
    const std::uint32_t fraction = bits & 0x007fffffU;
    if (exponent == 0x7f800000U && fraction != 0U) {
        const std::uint64_t sign =
            static_cast<std::uint64_t>(bits & 0x80000000U) << 32U;
        const std::uint64_t quiet_fraction =
            static_cast<std::uint64_t>(fraction | 0x00400000U) << 29U;
        return sign | 0x7ff0000000000000ULL | quiet_fraction;
    }
    return double_bits(static_cast<double>(float_from_bits(bits)));
}

AttributeBenchmarkDescriptor descriptor(
    std::string name,
    AttributeKind kind = AttributeKind::status,
    std::optional<std::string> originator_name = std::nullopt,
    attribute::AttributeDefinitionId definition_id = 0U) {
    AttributeBenchmarkDescriptor result;
    result.definition_id = definition_id;
    result.kind = kind;
    result.name = std::move(name);
    result.originator_name = std::move(originator_name);
    return result;
}

PreparedAttributeBenchmarkData prepared(
    std::vector<AttributeBenchmarkDescriptor> attributes,
    std::size_t rows,
    std::size_t columns,
    std::vector<float> values,
    bool extrema_requested = false,
    std::size_t column_repetitions = 1U) {
    PreparedAttributeBenchmarkData result;
    result.attributes = std::move(attributes);
    result.matrix = AttributeBenchmarkMatrix{rows, columns, std::move(values)};
    result.extrema_requested = extrema_requested;
    result.column_repetitions = column_repetitions;
    return result;
}

template <typename Callable>
void expect_benchmark_error(
    Callable&& callable,
    AttributeBenchmarkErrorCode code,
    AttributeBenchmarkOperation operation,
    const std::string& context) {
    try {
        std::forward<Callable>(callable)();
    } catch (const AttributeBenchmarkException& error) {
        expect(error.code() == code, context + ": error code drift");
        expect(error.operation() == operation, context + ": operation drift");
        expect(std::string_view(error.what()).empty() == false,
               context + ": typed error must include a message");
        return;
    }
    fail(context + ": expected AttributeBenchmarkException");
}

void expect_entry(
    const AttributeBenchmarkMap& map,
    std::string_view name,
    std::uint64_t expected_bits,
    const std::string& context) {
    const std::optional<AttributeBenchmarkId> id = map.bind(name);
    expect(id.has_value(), context + ": missing compact ID");
    const AttributeBenchmarkEntry& entry = map.at(*id);
    expect(entry.name == name, context + ": bound entry name drift");
    expect(double_bits(entry.value) == expected_bits,
           context + ": bound entry value bits drift");
    const double* found = map.find(name);
    expect(found != nullptr, context + ": find failed for present name");
    expect(double_bits(*found) == expected_bits,
           context + ": find value bits drift");
}

void expect_maps_exact(
    const AttributeBenchmarkMap& actual,
    const AttributeBenchmarkMap& expected_map,
    const std::string& context) {
    const std::vector<AttributeBenchmarkEntry>& actual_entries = actual.entries();
    const std::vector<AttributeBenchmarkEntry>& expected_entries =
        expected_map.entries();
    expect(actual_entries.size() == expected_entries.size(),
           context + ": entry count drift");
    for (std::size_t index = 0U; index < actual_entries.size(); ++index) {
        expect(actual_entries[index].name == expected_entries[index].name,
               context + ": entry order/name drift at " +
                   std::to_string(index));
        expect(double_bits(actual_entries[index].value) ==
                   double_bits(expected_entries[index].value),
               context + ": entry value bits drift at " +
                   std::to_string(index));
        expect(index <=
                   static_cast<std::size_t>(
                       std::numeric_limits<AttributeBenchmarkId>::max()),
               context + ": test ID exceeds native ID domain");
        const AttributeBenchmarkId id =
            static_cast<AttributeBenchmarkId>(index);
        expect(actual.bind(actual_entries[index].name) ==
                   std::optional<AttributeBenchmarkId>{id},
               context + ": compact ID/order drift");
        expect(&actual.at(id) == &actual_entries[index],
               context + ": at(id) must reference stable entry storage");
    }
}

void test_matrix_validation_and_truncation() {
    const auto one = std::vector<AttributeBenchmarkDescriptor>{
        descriptor("one")};

    expect_benchmark_error(
        [&]() {
            static_cast<void>(attribute::get_attr_benchmarks(
                prepared(one, 1U, 2U, {1.0F})));
        },
        AttributeBenchmarkErrorCode::invalid_matrix_shape,
        AttributeBenchmarkOperation::validate_prepared_data,
        "short matrix storage");
    expect_benchmark_error(
        [&]() {
            static_cast<void>(attribute::get_attr_benchmarks(
                prepared(one, 1U, 1U, {1.0F, 2.0F})));
        },
        AttributeBenchmarkErrorCode::invalid_matrix_shape,
        AttributeBenchmarkOperation::validate_prepared_data,
        "long matrix storage");
    expect_benchmark_error(
        [&]() {
            static_cast<void>(attribute::get_attr_benchmarks(prepared(
                {},
                std::numeric_limits<std::size_t>::max(),
                2U,
                {})));
        },
        AttributeBenchmarkErrorCode::invalid_matrix_shape,
        AttributeBenchmarkOperation::validate_prepared_data,
        "matrix product overflow");
    expect_benchmark_error(
        [&]() {
            static_cast<void>(attribute::get_attr_benchmarks(
                prepared(one, 1U, 1U, {1.0F}, false, 0U)));
        },
        AttributeBenchmarkErrorCode::invalid_column_repetitions,
        AttributeBenchmarkOperation::validate_prepared_data,
        "zero virtual repetition");
    expect_benchmark_error(
        [&]() {
            static_cast<void>(attribute::get_attr_benchmarks(prepared(
                one,
                1U,
                2U,
                {1.0F, 2.0F},
                false,
                std::numeric_limits<std::size_t>::max())));
        },
        AttributeBenchmarkErrorCode::invalid_column_repetitions,
        AttributeBenchmarkOperation::validate_prepared_data,
        "virtual repeated-column product overflow");
    expect_benchmark_error(
        [&]() {
            static_cast<void>(attribute::get_attr_benchmarks(
                prepared(one, 1U, 0U, {})));
        },
        AttributeBenchmarkErrorCode::empty_retained_row,
        AttributeBenchmarkOperation::reduce_rows,
        "retained empty row");

    const AttributeBenchmarkMap no_attributes =
        attribute::get_attr_benchmarks(prepared({}, 3U, 0U, {}));
    expect(no_attributes.entries().empty(),
           "zero paired rows must tolerate empty physical rows");

    const AttributeBenchmarkMap zero_rows = attribute::get_attr_benchmarks(
        prepared({descriptor("ignored-a"), descriptor("ignored-b")},
                 0U,
                 std::numeric_limits<std::size_t>::max(),
                 {}));
    expect(zero_rows.entries().empty(),
           "zero matrix rows must zip-truncate all descriptors");

    const AttributeBenchmarkMap skipped_resource =
        attribute::get_attr_benchmarks(prepared(
            {descriptor("resource", AttributeKind::resource)},
            1U,
            0U,
            {},
            true));
    expect(skipped_resource.entries().empty(),
           "an extrema-mode resource row is skipped, not retained");

    const AttributeBenchmarkMap descriptors_short =
        attribute::get_attr_benchmarks(prepared(
            {descriptor("first")},
            3U,
            2U,
            {1.0F, 2.0F, 100.0F, 200.0F, 1000.0F, 2000.0F}));
    expect(descriptors_short.entries().size() == 1U,
           "extra matrix rows must be zip-truncated");
    expect_entry(descriptors_short,
                 "first",
                 double_bits(2.0),
                 "extra-row truncation");

    const AttributeBenchmarkMap rows_short = attribute::get_attr_benchmarks(
        prepared({descriptor("first"), descriptor("second"), descriptor("third")},
                 1U,
                 2U,
                 {-3.0F, -1.0F}));
    expect(rows_short.entries().size() == 1U,
           "extra descriptors must be zip-truncated");
    expect_entry(rows_short,
                 "first",
                 double_bits(-1.0),
                 "extra-descriptor truncation");
}

void test_branch_keys_duplicates_and_map_ids() {
    const std::vector<AttributeBenchmarkDescriptor> descriptors{
        descriptor("cpu", AttributeKind::resource, "ignored-resource", 11U),
        descriptor("cpu_max", AttributeKind::extrema, "cpu", 12U),
        descriptor("position", AttributeKind::position, std::nullopt, 13U),
        descriptor("latency-a", AttributeKind::latency, "shared", 14U),
        descriptor("latency-b", AttributeKind::status, "middle", 15U),
        descriptor("latency-c", AttributeKind::latency, "shared", 16U),
        descriptor("empty-origin", AttributeKind::status, std::string{}, 17U),
    };
    const std::vector<float> values{
        8.0F, 9.0F,
        2.0F, 5.0F,
        -8.0F, -3.0F,
        10.0F, 11.0F,
        20.0F, 21.0F,
        30.0F, 31.0F,
        40.0F, 41.0F,
    };

    const AttributeBenchmarkMap extrema = attribute::get_attr_benchmarks(
        prepared(descriptors, 7U, 2U, values, true),
        8U);
    const std::vector<AttributeBenchmarkEntry>& extrema_entries =
        extrema.entries();
    expect(extrema_entries.size() == 5U,
           "extrema mode must skip resource and merge duplicate originators");
    expect(extrema_entries[0U].name == "cpu",
           "originator key must occupy first retained position");
    expect(extrema_entries[1U].name == "position",
           "missing originator must fall back to descriptor name");
    expect(extrema_entries[2U].name == "shared",
           "first duplicate key position drift");
    expect(extrema_entries[3U].name == "middle",
           "middle key insertion order drift");
    expect(extrema_entries[4U].name.empty(),
           "present empty originator is a key, not a name fallback");
    expect_entry(extrema, "cpu", double_bits(5.0), "extrema originator");
    expect_entry(extrema, "position", double_bits(-3.0), "extrema fallback");
    expect_entry(extrema,
                 "shared",
                 double_bits(31.0),
                 "duplicate key last value");
    expect_entry(extrema, "middle", double_bits(21.0), "middle value");
    expect_entry(extrema, "", double_bits(41.0), "empty originator value");
    expect(!extrema.bind("cpu_max").has_value(),
           "extrema output must not expose descriptor name when originator exists");
    expect(extrema.find("missing") == nullptr,
           "find must return null for a missing dynamic name");
    expect(!extrema.bind("missing").has_value(),
           "bind must return nullopt for a missing dynamic name");

    expect_benchmark_error(
        [&]() {
            static_cast<void>(extrema.at(
                std::numeric_limits<AttributeBenchmarkId>::max()));
        },
        AttributeBenchmarkErrorCode::invalid_benchmark_id,
        AttributeBenchmarkOperation::access_map,
        "out-of-range compact map access");

    const AttributeBenchmarkMap ordinary = attribute::get_attr_benchmarks(
        prepared(descriptors, 7U, 2U, values, false),
        2U);
    expect(ordinary.entries().size() == descriptors.size(),
           "ordinary mode must retain resource and ignore originator keys");
    for (std::size_t index = 0U; index < descriptors.size(); ++index) {
        expect(ordinary.entries()[index].name == descriptors[index].name,
               "ordinary descriptor-name order drift at " +
                   std::to_string(index));
    }
    expect_entry(ordinary, "cpu", double_bits(9.0), "ordinary resource");
    expect_entry(ordinary,
                 "latency-c",
                 double_bits(31.0),
                 "ordinary originator ignored");

    const AttributeBenchmarkMap duplicate_names =
        attribute::get_attr_benchmarks(prepared(
            {descriptor("dup"), descriptor("middle"), descriptor("dup")},
            3U,
            1U,
            {1.0F, 2.0F, 3.0F}));
    expect(duplicate_names.entries().size() == 2U,
           "duplicate ordinary names must overwrite in place");
    expect(duplicate_names.entries()[0U].name == "dup" &&
               duplicate_names.entries()[1U].name == "middle",
           "duplicate overwrite must preserve first insertion order");
    expect_entry(duplicate_names,
                 "dup",
                 double_bits(3.0),
                 "duplicate ordinary last write");

    AttributeBenchmarkMap copy_assigned;
    {
        const AttributeBenchmarkMap source = duplicate_names;
        copy_assigned = source;
        expect_maps_exact(copy_assigned, source, "live copy assignment");
    }
    expect_entry(copy_assigned,
                 "dup",
                 double_bits(3.0),
                 "copy assignment index lifetime");
    const AttributeBenchmarkMap copy_constructed(copy_assigned);
    expect_maps_exact(copy_constructed,
                      copy_assigned,
                      "map copy constructor index rebuild");
    AttributeBenchmarkMap moved(std::move(copy_assigned));
    expect_entry(moved, "middle", double_bits(2.0), "map move index lifetime");
}

void test_float32_exactness() {
    const float positive_subnormal = float_from_bits(0x00000001U);
    const float negative_subnormal = float_from_bits(0x80000001U);
    const float rounded_down = static_cast<float>(16777217.0);
    const float rounded_up = static_cast<float>(16777218.0);

    const AttributeBenchmarkMap ordinary = attribute::get_attr_benchmarks(
        prepared(
            {descriptor("rounding"),
             descriptor("infinity"),
             descriptor("subnormal"),
             descriptor("negative-infinity")},
            4U,
            3U,
            {rounded_down,
             16777215.0F,
             rounded_up,
             -std::numeric_limits<float>::infinity(),
             7.0F,
             std::numeric_limits<float>::infinity(),
             negative_subnormal,
             0.0F,
             positive_subnormal,
             -std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity()}));
    expect_entry(ordinary,
                 "rounding",
                 double_bits(static_cast<double>(rounded_up)),
                 "binary32 rounding before maximum");
    expect_entry(ordinary,
                 "infinity",
                 double_bits(std::numeric_limits<double>::infinity()),
                 "positive infinity maximum");
    expect_entry(ordinary,
                 "subnormal",
                 double_bits(static_cast<double>(positive_subnormal)),
                 "binary32 subnormal maximum");
    expect_entry(ordinary,
                 "negative-infinity",
                 double_bits(-std::numeric_limits<double>::infinity()),
                 "negative infinity maximum");

    const AttributeBenchmarkMap zero_orders = attribute::get_attr_benchmarks(
        prepared(
            {descriptor("negative-positive"),
             descriptor("positive-negative"),
             descriptor("negative-negative"),
             descriptor("positive-positive")},
            4U,
            2U,
            {-0.0F, 0.0F, 0.0F, -0.0F, -0.0F, -0.0F, 0.0F, 0.0F}));
    expect_entry(zero_orders,
                 "negative-positive",
                 0x0000000000000000ULL,
                 "later positive signed zero");
    expect_entry(zero_orders,
                 "positive-negative",
                 0x8000000000000000ULL,
                 "later negative signed zero");
    expect_entry(zero_orders,
                 "negative-negative",
                 0x8000000000000000ULL,
                 "all negative signed zeros");
    expect_entry(zero_orders,
                 "positive-positive",
                 0x0000000000000000ULL,
                 "all positive signed zeros");

    constexpr std::uint32_t qnan_a = 0x7fc12345U;
    constexpr std::uint32_t qnan_b = 0xffc54321U;
    constexpr std::uint32_t snan_a = 0x7fa12345U;
    constexpr std::uint32_t snan_b = 0xffa54321U;

    const AttributeBenchmarkMap singleton_nans =
        attribute::get_attr_benchmarks(prepared(
            {descriptor("qnan-single"), descriptor("snan-single")},
            2U,
            1U,
            {float_from_bits(qnan_a), float_from_bits(snan_a)}));
    expect_entry(singleton_nans,
                 "qnan-single",
                 promoted_float_bits(qnan_a),
                 "singleton qNaN payload promotion");
    expect_entry(singleton_nans,
                 "snan-single",
                 promoted_float_bits(snan_a),
                 "singleton sNaN quieting and payload promotion");

    const AttributeBenchmarkMap ordered_nans = attribute::get_attr_benchmarks(
        prepared(
            {descriptor("qnan-first"),
             descriptor("qnan-later"),
             descriptor("snan-first"),
             descriptor("snan-later")},
            4U,
            3U,
            {float_from_bits(qnan_a),
             1.0F,
             2.0F,
             -3.0F,
             float_from_bits(qnan_b),
             float_from_bits(qnan_a),
             float_from_bits(snan_a),
             -1.0F,
             9.0F,
             -2.0F,
             float_from_bits(snan_b),
             float_from_bits(qnan_a)}));
    expect_entry(ordered_nans,
                 "qnan-first",
                 promoted_float_bits(0x7fc00000U),
                 "first qNaN canonicalization with later operand");
    expect_entry(ordered_nans,
                 "qnan-later",
                 promoted_float_bits(qnan_b),
                 "later qNaN payload wins and remains sticky");
    expect_entry(ordered_nans,
                 "snan-first",
                 promoted_float_bits(0x7fc00000U),
                 "first sNaN canonicalization with later operand");
    expect_entry(ordered_nans,
                 "snan-later",
                 promoted_float_bits(snan_b),
                 "later sNaN quieting preserves sign and payload");

    const AttributeBenchmarkMap repeated_nans =
        attribute::get_attr_benchmarks(prepared(
            {descriptor("qnan-repeated"),
             descriptor("snan-repeated"),
             descriptor("negative-zero-repeated")},
            3U,
            1U,
            {float_from_bits(qnan_a), float_from_bits(snan_a), -0.0F},
            false,
            2U));
    expect_entry(repeated_nans,
                 "qnan-repeated",
                 promoted_float_bits(0x7fc00000U),
                 "virtual direct-link qNaN repetition");
    expect_entry(repeated_nans,
                 "snan-repeated",
                 promoted_float_bits(0x7fc00000U),
                 "virtual direct-link sNaN repetition");
    expect_entry(repeated_nans,
                 "negative-zero-repeated",
                 0x8000000000000000ULL,
                 "virtual repetition preserves later negative zero");

    std::vector<float> vector_lane_values(3U * 17U, 0.0F);
    vector_lane_values[0U] = -1.0F;
    vector_lane_values[4U] = -0.0F;
    vector_lane_values[17U] = -1.0F;
    vector_lane_values[17U + 16U] = -0.0F;
    vector_lane_values[34U] = -1.0F;
    vector_lane_values[34U + 7U] = float_from_bits(qnan_b);
    const AttributeBenchmarkMap vector_lanes =
        attribute::get_attr_benchmarks(prepared(
            {descriptor("horizontal-lane-last"),
             descriptor("horizontal-source-last"),
             descriptor("vector-qnan")},
            3U,
            17U,
            std::move(vector_lane_values)));
    expect_entry(vector_lanes,
                 "horizontal-lane-last",
                 0x8000000000000000ULL,
                 "NumPy 16-lane horizontal signed-zero tie order");
    expect_entry(vector_lanes,
                 "horizontal-source-last",
                 0x0000000000000000ULL,
                 "NumPy lane order differs from source-order signed-zero tie");
    expect_entry(vector_lanes,
                 "vector-qnan",
                 promoted_float_bits(0x7fc00000U),
                 "qNaN inside a vectorized reduction block canonicalizes");

    std::vector<float> tail_nan_values(18U, -1.0F);
    tail_nan_values.back() = float_from_bits(qnan_b);
    const std::size_t tail_nan_columns = tail_nan_values.size();
    const AttributeBenchmarkMap tail_nan = attribute::get_attr_benchmarks(
        prepared({descriptor("tail-qnan")},
                 1U,
                 tail_nan_columns,
                 std::move(tail_nan_values)));
    expect_entry(tail_nan,
                 "tail-qnan",
                 promoted_float_bits(qnan_b),
                 "qNaN in the post-vector scalar tail preserves payload");
}

PreparedAttributeBenchmarkData make_parallel_fixture() {
    constexpr std::size_t row_count = 257U;
    constexpr std::size_t column_count = 19U;
    std::vector<AttributeBenchmarkDescriptor> attributes;
    std::vector<float> values;
    attributes.reserve(row_count);
    values.reserve(row_count * column_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        attributes.push_back(descriptor(
            "row-" + std::to_string(row),
            (row % 11U == 0U) ? AttributeKind::latency
                              : AttributeKind::status,
            std::nullopt,
            static_cast<attribute::AttributeDefinitionId>(row)));
        for (std::size_t column = 0U; column < column_count; ++column) {
            const std::int64_t encoded =
                static_cast<std::int64_t>((row * 37U + column * 19U) % 1009U) -
                504;
            values.push_back(static_cast<float>(encoded) / 8.0F);
        }
    }
    return prepared(
        std::move(attributes),
        row_count,
        column_count,
        std::move(values),
        false,
        2U);
}

void test_workers_and_concurrent_reductions() {
    const PreparedAttributeBenchmarkData fixture = make_parallel_fixture();
    const AttributeBenchmarkMap expected_map =
        attribute::get_attr_benchmarks(fixture, 0U);
    for (const std::size_t workers : {1U, 2U, 8U}) {
        expect_maps_exact(attribute::get_attr_benchmarks(fixture, workers),
                          expected_map,
                          "configured worker parity " +
                              std::to_string(workers));
    }

    constexpr std::size_t caller_count = 8U;
    constexpr std::size_t rounds = 6U;
    std::array<std::future<void>, caller_count> callers;
    const std::array<std::size_t, 4U> worker_counts{0U, 1U, 2U, 8U};
    for (std::size_t caller = 0U; caller < caller_count; ++caller) {
        callers[caller] = std::async(
            std::launch::async,
            [caller, &fixture, &expected_map, &worker_counts]() {
                const std::size_t workers =
                    worker_counts[caller % worker_counts.size()];
                for (std::size_t round = 0U; round < rounds; ++round) {
                    expect_maps_exact(
                        attribute::get_attr_benchmarks(fixture, workers),
                        expected_map,
                        "concurrent reduction caller " +
                            std::to_string(caller) + " round " +
                            std::to_string(round));
                }
            });
    }
    for (std::future<void>& caller : callers) {
        caller.get();
    }
}

PreparedAttributeBenchmarkData single_group_data(
    std::string name,
    float value) {
    return prepared(
        {descriptor(std::move(name))}, 1U, 1U, {value});
}

void expect_optional_maps_equal(
    const std::optional<AttributeBenchmarkMap>& actual,
    const std::optional<AttributeBenchmarkMap>& expected_map,
    const std::string& context) {
    expect(actual.has_value() == expected_map.has_value(),
           context + ": optional presence drift");
    if (actual.has_value()) {
        expect_maps_exact(*actual, *expected_map, context);
    }
}

void expect_benchmark_groups_equal(
    const AttributeBenchmarks& actual,
    const AttributeBenchmarks& expected_groups,
    const std::string& context) {
    expect_optional_maps_equal(actual.node_attr_benchmarks,
                               expected_groups.node_attr_benchmarks,
                               context + " node");
    expect_optional_maps_equal(actual.link_attr_benchmarks,
                               expected_groups.link_attr_benchmarks,
                               context + " link");
    expect_optional_maps_equal(actual.link_sum_attr_benchmarks,
                               expected_groups.link_sum_attr_benchmarks,
                               context + " link-sum");
}

void test_manager_groups_and_order() {
    AttributeBenchmarkRequest request;
    request.node = single_group_data("node", 3.0F);
    request.link = single_group_data("link", 5.0F);
    request.link->column_repetitions = 2U;
    request.link_sum = single_group_data("link-sum", 7.0F);
    request.workers = 8U;

    const AttributeBenchmarks static_groups =
        attribute::AttributeBenchmarkManager::get_benchmarks(request);
    expect(static_groups.node_attr_benchmarks.has_value(),
           "enabled node group missing");
    expect(static_groups.link_attr_benchmarks.has_value(),
           "enabled link group missing");
    expect(static_groups.link_sum_attr_benchmarks.has_value(),
           "enabled link-sum group missing");
    expect_entry(*static_groups.node_attr_benchmarks,
                 "node",
                 double_bits(3.0),
                 "manager node group");
    expect_entry(*static_groups.link_attr_benchmarks,
                 "link",
                 double_bits(5.0),
                 "manager direct-link group");
    expect_entry(*static_groups.link_sum_attr_benchmarks,
                 "link-sum",
                 double_bits(7.0),
                 "manager link-sum group");

    const attribute::AttributeBenchmarkManager manager(request);
    expect_benchmark_groups_equal(manager.benchmarks(),
                                  static_groups,
                                  "constructor/static parity");
    expect(&manager.benchmarks() == &manager.benchmarks(),
           "benchmarks accessor must return stable owned object identity");

    AttributeBenchmarkRequest only_link;
    only_link.link = single_group_data("only-link", 11.0F);
    only_link.workers = 0U;
    const AttributeBenchmarks partial =
        attribute::AttributeBenchmarkManager::get_benchmarks(only_link);
    expect(!partial.node_attr_benchmarks.has_value(),
           "disabled node group must remain nullopt");
    expect(partial.link_attr_benchmarks.has_value(),
           "enabled partial link group missing");
    expect(!partial.link_sum_attr_benchmarks.has_value(),
           "disabled link-sum group must remain nullopt");

    const AttributeBenchmarks empty =
        attribute::AttributeBenchmarkManager::get_benchmarks({});
    expect(!empty.node_attr_benchmarks.has_value() &&
               !empty.link_attr_benchmarks.has_value() &&
               !empty.link_sum_attr_benchmarks.has_value(),
           "empty request must preserve three disabled optionals");

    AttributeBenchmarkRequest node_first;
    node_first.node = single_group_data("bad-node", 1.0F);
    node_first.node->column_repetitions = 0U;
    node_first.link = prepared({}, 1U, 2U, {1.0F});
    expect_benchmark_error(
        [&]() {
            static_cast<void>(
                attribute::AttributeBenchmarkManager::get_benchmarks(
                    node_first));
        },
        AttributeBenchmarkErrorCode::invalid_column_repetitions,
        AttributeBenchmarkOperation::validate_prepared_data,
        "manager must validate node before link");

    AttributeBenchmarkRequest link_first;
    link_first.node = single_group_data("valid-node", 1.0F);
    link_first.link = single_group_data("bad-link", 2.0F);
    link_first.link->column_repetitions = 0U;
    link_first.link_sum = prepared({}, 1U, 2U, {1.0F});
    expect_benchmark_error(
        [&]() {
            static_cast<void>(
                attribute::AttributeBenchmarkManager::get_benchmarks(
                    link_first));
        },
        AttributeBenchmarkErrorCode::invalid_column_repetitions,
        AttributeBenchmarkOperation::validate_prepared_data,
        "manager must validate link before link-sum");
}

std::shared_ptr<AttributeBenchmarks> make_cached_groups(
    const std::string& name,
    float value) {
    AttributeBenchmarkRequest request;
    request.node = single_group_data(name, value);
    return std::make_shared<AttributeBenchmarks>(
        attribute::AttributeBenchmarkManager::get_benchmarks(request));
}

void test_cache_identity_overwrite_and_concurrency() {
    attribute::AttributeBenchmarkManager::clear_cache();
    expect(attribute::AttributeBenchmarkManager::cache_size() == 0U,
           "cache must begin empty after clear");
    expect(!attribute::AttributeBenchmarkManager::get_from_cache("missing"),
           "missing cache key must return null shared_ptr");

    const std::shared_ptr<AttributeBenchmarks> first =
        make_cached_groups("first", 1.0F);
    attribute::AttributeBenchmarkManager::add_to_cache(
        std::string("temporary-key"), first);
    expect(attribute::AttributeBenchmarkManager::cache_size() == 1U,
           "first cache insert size drift");
    expect(attribute::AttributeBenchmarkManager::get_from_cache("temporary-key") ==
               first,
           "cache lookup must retain exact shared object identity");

    const std::shared_ptr<AttributeBenchmarks> replacement =
        make_cached_groups("replacement", 2.0F);
    attribute::AttributeBenchmarkManager::add_to_cache("temporary-key",
                                                       replacement);
    expect(attribute::AttributeBenchmarkManager::cache_size() == 1U,
           "cache overwrite must not add a second key");
    expect(attribute::AttributeBenchmarkManager::get_from_cache("temporary-key") ==
               replacement,
           "cache overwrite must expose replacement identity");

    constexpr std::size_t item_count = 32U;
    std::array<std::shared_ptr<AttributeBenchmarks>, item_count> items;
    std::array<std::future<void>, item_count> writers;
    for (std::size_t index = 0U; index < item_count; ++index) {
        items[index] = make_cached_groups(
            "cached-" + std::to_string(index),
            static_cast<float>(index));
        writers[index] = std::async(
            std::launch::async,
            [index, &items]() {
                const std::string key = "parallel-" + std::to_string(index);
                attribute::AttributeBenchmarkManager::add_to_cache(
                    key, items[index]);
                const std::shared_ptr<AttributeBenchmarks> found =
                    attribute::AttributeBenchmarkManager::get_from_cache(key);
                expect(found == items[index],
                       "parallel cache insert/read identity drift for " + key);
            });
    }
    for (std::future<void>& writer : writers) {
        writer.get();
    }
    expect(attribute::AttributeBenchmarkManager::cache_size() == item_count + 1U,
           "parallel unique cache insert size drift");
    for (std::size_t index = 0U; index < item_count; ++index) {
        const std::string key = "parallel-" + std::to_string(index);
        expect(attribute::AttributeBenchmarkManager::get_from_cache(key) ==
                   items[index],
               "post-join cache identity drift for " + key);
    }

    constexpr std::size_t racing_count = 8U;
    std::array<std::shared_ptr<AttributeBenchmarks>, racing_count> racing_items;
    std::array<std::future<void>, racing_count> racers;
    for (std::size_t index = 0U; index < racing_count; ++index) {
        racing_items[index] = make_cached_groups(
            "racing-" + std::to_string(index),
            static_cast<float>(100U + index));
        racers[index] = std::async(
            std::launch::async,
            [index, &racing_items]() {
                for (std::size_t round = 0U; round < 16U; ++round) {
                    attribute::AttributeBenchmarkManager::add_to_cache(
                        "same-key", racing_items[index]);
                    static_cast<void>(
                        attribute::AttributeBenchmarkManager::get_from_cache(
                            "same-key"));
                }
            });
    }
    for (std::future<void>& racer : racers) {
        racer.get();
    }
    const std::shared_ptr<AttributeBenchmarks> racing_winner =
        attribute::AttributeBenchmarkManager::get_from_cache("same-key");
    expect(static_cast<bool>(racing_winner),
           "concurrent overwrite must leave a stored value");
    bool winner_has_known_identity = false;
    for (const std::shared_ptr<AttributeBenchmarks>& candidate : racing_items) {
        winner_has_known_identity =
            winner_has_known_identity || racing_winner == candidate;
    }
    expect(winner_has_known_identity,
           "concurrent overwrite produced an unknown object identity");
    expect(attribute::AttributeBenchmarkManager::cache_size() == item_count + 2U,
           "same-key concurrent overwrite size drift");

    attribute::AttributeBenchmarkManager::clear_cache();
    expect(attribute::AttributeBenchmarkManager::cache_size() == 0U,
           "final cache clear size drift");
    expect(!attribute::AttributeBenchmarkManager::get_from_cache("temporary-key") &&
               !attribute::AttributeBenchmarkManager::get_from_cache("same-key"),
           "clear must remove all cache entries");
}

}  // namespace

template <typename Callable>
void run_test(const char* name, Callable&& callable) {
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string(name) + ": " + error.what());
    }
}

int main() {
    try {
        run_test("matrix validation/truncation",
                 test_matrix_validation_and_truncation);
        run_test("branches/keys/map IDs",
                 test_branch_keys_duplicates_and_map_ids);
        run_test("float32 exactness", test_float32_exactness);
        run_test("workers/concurrent reductions",
                 test_workers_and_concurrent_reductions);
        run_test("manager groups/order", test_manager_groups_and_order);
        run_test("cache identity/concurrency",
                 test_cache_identity_overwrite_and_concurrency);
        std::cout << "attribute_benchmark_manager_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attribute_benchmark_manager_unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
