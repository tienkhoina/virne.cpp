#include "attribute/attribute_benchmark_manager.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(std::uint32_t bits) noexcept {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::string hex_bits(double value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16)
           << double_bits(value);
    return output.str();
}

std::string hex_encode(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2U, '0');
    for (std::size_t index = 0U; index < value.size(); ++index) {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2U] = digits[byte >> 4U];
        result[index * 2U + 1U] = digits[byte & 15U];
    }
    return result;
}

std::string encode_map(const attribute::AttributeBenchmarkMap& values) {
    std::string result;
    for (const auto& entry : values.entries()) {
        if (!result.empty()) {
            result.push_back(',');
        }
        result.append(hex_encode(entry.name));
        result.push_back('=');
        result.append(hex_bits(entry.value));
    }
    return result;
}

std::string error_name(
    attribute::AttributeBenchmarkErrorCode code) noexcept {
    using Code = attribute::AttributeBenchmarkErrorCode;
    switch (code) {
        case Code::invalid_matrix_shape: return "invalid_matrix_shape";
        case Code::invalid_column_repetitions:
            return "invalid_column_repetitions";
        case Code::empty_retained_row: return "empty_retained_row";
        case Code::benchmark_id_range: return "benchmark_id_range";
        case Code::invalid_benchmark_id: return "invalid_benchmark_id";
    }
    return "invalid_attribute_benchmark_error";
}

template <typename Callable>
void emit_case(std::string_view name, Callable&& callable) {
    try {
        const std::string payload = callable();
        std::cout << "case=" << name << "|ok|" << payload << '\n';
    } catch (const attribute::AttributeBenchmarkException& error) {
        std::cout << "case=" << name << "|error|"
                  << error_name(error.code()) << '\n';
    } catch (const std::exception& error) {
        std::cout << "case=" << name << "|error|dependency_error|"
                  << hex_encode(error.what()) << '\n';
    }
}

attribute::AttributeBenchmarkDescriptor descriptor(
    attribute::AttributeDefinitionId definition_id,
    attribute::AttributeKind kind,
    std::string name,
    std::optional<std::string> originator_name = std::nullopt) {
    return attribute::AttributeBenchmarkDescriptor{
        definition_id, kind, std::move(name), std::move(originator_name)};
}

attribute::PreparedAttributeBenchmarkData prepared(
    std::vector<attribute::AttributeBenchmarkDescriptor> attributes,
    std::size_t rows,
    std::size_t columns,
    std::vector<float> values,
    bool extrema_requested = false,
    std::size_t column_repetitions = 1U) {
    attribute::PreparedAttributeBenchmarkData result;
    result.attributes = std::move(attributes);
    result.matrix.rows = rows;
    result.matrix.columns = columns;
    result.matrix.values = std::move(values);
    result.extrema_requested = extrema_requested;
    result.column_repetitions = column_repetitions;
    return result;
}

void emit_reduction_cases() {
    emit_case("empty", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared({}, 0U, 0U, {}), 1U));
    });

    emit_case("truncated_rows", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "alpha"),
                 descriptor(1U, attribute::AttributeKind::status, "beta"),
                 descriptor(2U, attribute::AttributeKind::status, "omitted")},
                2U,
                3U,
                {1.0F, 3.0F, 2.0F, -4.0F, -1.0F, -2.0F}),
            1U));
    });

    emit_case("extra_rows", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "only")},
                3U,
                2U,
                {1.0F, 2.0F, 9.0F, 10.0F, 11.0F, 12.0F}),
            1U));
    });

    emit_case("extrema_resource_originator_duplicate", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::resource, "capacity"),
                 descriptor(
                     1U, attribute::AttributeKind::extrema, "latency_max",
                     std::string{"latency"}),
                 descriptor(2U, attribute::AttributeKind::status, "fallback"),
                 descriptor(
                     3U, attribute::AttributeKind::extrema, "latency_late",
                     std::string{"latency"})},
                4U,
                2U,
                {99.0F, 100.0F, -2.0F, -1.0F,
                 4.0F, 5.0F, 7.0F, 6.0F},
                true),
            2U));
    });

    emit_case("non_extrema_names_duplicate", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::resource, "same"),
                 descriptor(
                     1U, attribute::AttributeKind::extrema, "other",
                     std::string{"ignored_originator"}),
                 descriptor(2U, attribute::AttributeKind::status, "same")},
                3U,
                2U,
                {1.0F, 2.0F, 8.0F, 9.0F, 4.0F, 6.0F}),
            8U));
    });

    emit_case("numeric_basics", [] {
        const float denormal = std::numeric_limits<float>::denorm_min();
        const float infinity = std::numeric_limits<float>::infinity();
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "rounding"),
                 descriptor(1U, attribute::AttributeKind::status, "positive_inf"),
                 descriptor(2U, attribute::AttributeKind::status, "negative_inf"),
                 descriptor(3U, attribute::AttributeKind::status, "subnormal")},
                4U,
                2U,
                {16777217.0F, 16777215.0F,
                 -infinity, infinity,
                 -infinity, -5.0F,
                 -denormal, denormal}),
            1U));
    });

    emit_case("signed_zero_pairs", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "negative_positive"),
                 descriptor(1U, attribute::AttributeKind::status, "positive_negative")},
                2U,
                2U,
                {-0.0F, 0.0F, 0.0F, -0.0F}),
            1U));
    });

    emit_case("nan_singletons", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "qnan"),
                 descriptor(1U, attribute::AttributeKind::status, "snan")},
                2U,
                1U,
                {float_from_bits(UINT32_C(0x7fc12345)),
                 float_from_bits(UINT32_C(0xff812346))}),
            1U));
    });

    emit_case("nan_pairs", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "qnan_first"),
                 descriptor(1U, attribute::AttributeKind::status, "qnan_later"),
                 descriptor(2U, attribute::AttributeKind::status, "snan_first"),
                 descriptor(3U, attribute::AttributeKind::status, "snan_later")},
                4U,
                2U,
                {float_from_bits(UINT32_C(0x7fc12345)), 1.0F,
                 1.0F, float_from_bits(UINT32_C(0xffc23456)),
                 float_from_bits(UINT32_C(0x7f812345)), 1.0F,
                 1.0F, float_from_bits(UINT32_C(0xff812346))}),
            2U));
    });

    emit_case("nan_repetition", [] {
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "qnan_repeat"),
                 descriptor(1U, attribute::AttributeKind::status, "snan_repeat")},
                2U,
                1U,
                {float_from_bits(UINT32_C(0x7fc12345)),
                 float_from_bits(UINT32_C(0xff812346))},
                false,
                2U),
            1U));
    });

    emit_case("nan_simd_17", [] {
        std::vector<float> values(3U * 17U, -4.0F);
        values[0U] = float_from_bits(UINT32_C(0x7fc12345));
        values[17U + 8U] = float_from_bits(UINT32_C(0xffc23456));
        values[34U + 16U] = float_from_bits(UINT32_C(0x7f812345));
        return encode_map(attribute::get_attr_benchmarks(
            prepared(
                {descriptor(0U, attribute::AttributeKind::status, "nan_first"),
                 descriptor(1U, attribute::AttributeKind::status, "nan_middle"),
                 descriptor(2U, attribute::AttributeKind::status, "nan_last")},
                3U,
                17U,
                std::move(values)),
            8U));
    });
}

attribute::PreparedAttributeBenchmarkData worker_data() {
    std::vector<attribute::AttributeBenchmarkDescriptor> attributes;
    std::vector<float> values;
    attributes.reserve(12U);
    values.reserve(12U * 9U);
    for (std::size_t row = 0U; row < 12U; ++row) {
        attributes.push_back(descriptor(
            static_cast<attribute::AttributeDefinitionId>(row),
            attribute::AttributeKind::status,
            "worker_" + std::to_string(row)));
        for (std::size_t column = 0U; column < 9U; ++column) {
            const auto residue = static_cast<std::int64_t>(
                (row * 31U + column * 7U) % 101U);
            values.push_back(static_cast<float>(residue - 50) * 0.25F);
        }
    }
    return prepared(
        std::move(attributes), 12U, 9U, std::move(values), false, 2U);
}

std::string encode_groups(const attribute::AttributeBenchmarks& values) {
    const auto encode_optional = [](const auto& group) {
        return group.has_value() ? encode_map(*group) : std::string{"-"};
    };
    return "node=" + encode_optional(values.node_attr_benchmarks) +
           ";link=" + encode_optional(values.link_attr_benchmarks) +
           ";link_sum=" + encode_optional(values.link_sum_attr_benchmarks);
}

attribute::AttributeBenchmarkRequest group_request() {
    attribute::AttributeBenchmarkRequest request;
    request.node = prepared(
        {descriptor(
            0U, attribute::AttributeKind::extrema, "node_max",
            std::string{"node"})},
        1U, 3U, {1.0F, 3.0F, 2.0F}, true);
    request.link = prepared(
        {descriptor(
            1U, attribute::AttributeKind::extrema, "link_max",
            std::string{"link"})},
        1U, 3U, {-1.0F, 4.0F, 2.0F}, true, 2U);
    request.link_sum = prepared(
        {descriptor(
            1U, attribute::AttributeKind::extrema, "link_max",
            std::string{"link"})},
        1U, 2U, {7.0F, 6.0F}, true);
    request.workers = 2U;
    return request;
}

void emit_worker_manager_cache_cases() {
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        emit_case("workers_" + std::to_string(workers), [workers] {
            return encode_map(attribute::get_attr_benchmarks(
                worker_data(), workers));
        });
    }

    emit_case("optional_all_groups", [] {
        return encode_groups(
            attribute::AttributeBenchmarkManager::get_benchmarks(
                group_request()));
    });
    emit_case("optional_node_only", [] {
        auto request = group_request();
        request.link.reset();
        request.link_sum.reset();
        return encode_groups(
            attribute::AttributeBenchmarkManager::get_benchmarks(request));
    });
    emit_case("manager_constructor", [] {
        const attribute::AttributeBenchmarkManager manager(group_request());
        return encode_groups(manager.benchmarks());
    });

    emit_case("map_bind_access_copy", [] {
        const auto values = attribute::get_attr_benchmarks(prepared(
            {descriptor(0U, attribute::AttributeKind::status, "alpha"),
             descriptor(1U, attribute::AttributeKind::status, "beta")},
            2U, 2U, {1.0F, 3.0F, 5.0F, 4.0F}));
        const auto alpha = values.bind("alpha");
        const auto beta = values.bind("beta");
        const auto missing = values.bind("missing");
        if (!alpha || !beta) {
            throw std::runtime_error("known benchmark name failed to bind");
        }
        const attribute::AttributeBenchmarkMap copied(values);
        const auto copied_beta = copied.bind("beta");
        const double* found_alpha = copied.find("alpha");
        if (!copied_beta || found_alpha == nullptr) {
            throw std::runtime_error("copied benchmark index was not rebuilt");
        }
        return std::to_string(*alpha) + ',' + std::to_string(*beta) + ',' +
               (missing ? "0" : "1") + ',' +
               hex_bits(values.at(*alpha).value) + ',' +
               hex_bits(*found_alpha) + ',' + encode_map(copied);
    });

    emit_case("cache_semantics", [] {
        attribute::AttributeBenchmarkManager::clear_cache();
        const bool initially_missing =
            !attribute::AttributeBenchmarkManager::get_from_cache("same");
        auto first = std::make_shared<attribute::AttributeBenchmarks>();
        auto second = std::make_shared<attribute::AttributeBenchmarks>();
        attribute::AttributeBenchmarkManager::add_to_cache("same", first);
        const bool first_identity =
            attribute::AttributeBenchmarkManager::get_from_cache("same") == first;
        attribute::AttributeBenchmarkManager::add_to_cache("same", second);
        const bool second_identity =
            attribute::AttributeBenchmarkManager::get_from_cache("same") == second;
        const std::size_t overwritten_size =
            attribute::AttributeBenchmarkManager::cache_size();
        attribute::AttributeBenchmarkManager::add_to_cache("other", first);
        const std::size_t two_size =
            attribute::AttributeBenchmarkManager::cache_size();
        attribute::AttributeBenchmarkManager::clear_cache();
        const std::size_t cleared_size =
            attribute::AttributeBenchmarkManager::cache_size();
        const bool finally_missing =
            !attribute::AttributeBenchmarkManager::get_from_cache("same");
        return std::to_string(initially_missing) + ',' +
               std::to_string(first_identity) + ',' +
               std::to_string(second_identity) + ',' +
               std::to_string(overwritten_size) + ',' +
               std::to_string(two_size) + ',' +
               std::to_string(cleared_size) + ',' +
               std::to_string(finally_missing);
    });

    emit_case("concurrent_independent", [] {
        const auto data = worker_data();
        const std::string expected = encode_map(
            attribute::get_attr_benchmarks(data, 1U));
        std::atomic<bool> equal{true};
        std::vector<std::thread> threads;
        threads.reserve(4U);
        for (std::size_t index = 0U; index < 4U; ++index) {
            threads.emplace_back([&] {
                try {
                    if (encode_map(attribute::get_attr_benchmarks(data, 2U)) !=
                        expected) {
                        equal.store(false, std::memory_order_relaxed);
                    }
                } catch (...) {
                    equal.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return equal.load(std::memory_order_relaxed) ? std::string{"1"}
                                                     : std::string{"0"};
    });
}

void emit_error_cases() {
    emit_case("invalid_matrix_shape", [] {
        return encode_map(attribute::get_attr_benchmarks(prepared(
            {descriptor(0U, attribute::AttributeKind::status, "bad")},
            2U, 2U, {1.0F, 2.0F, 3.0F})));
    });
    emit_case("invalid_matrix_overflow", [] {
        return encode_map(attribute::get_attr_benchmarks(prepared(
            {}, std::numeric_limits<std::size_t>::max(), 2U, {})));
    });
    emit_case("zero_column_repetitions", [] {
        return encode_map(attribute::get_attr_benchmarks(prepared(
            {descriptor(0U, attribute::AttributeKind::status, "bad")},
            1U, 1U, {1.0F}, false, 0U)));
    });
    emit_case("empty_retained_row", [] {
        return encode_map(attribute::get_attr_benchmarks(prepared(
            {descriptor(0U, attribute::AttributeKind::status, "bad")},
            1U, 0U, {})));
    });
    emit_case("invalid_benchmark_id", [] {
        const auto values = attribute::get_attr_benchmarks(prepared(
            {descriptor(0U, attribute::AttributeKind::status, "only")},
            1U, 1U, {1.0F}));
        return hex_bits(values.at(1U).value);
    });
}

}  // namespace

int main() {
    try {
        emit_reduction_cases();
        emit_worker_manager_cache_cases();
        emit_error_cases();
        std::cout << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attribute_benchmark_manager_harness: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
