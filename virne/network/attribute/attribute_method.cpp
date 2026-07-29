#include "attribute_method.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <thread>
#include <utility>

#if (defined(__GNUC__) || defined(__clang__)) \
    && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define VIRNE_ATTRIBUTE_X86_SIMD 1
#else
#define VIRNE_ATTRIBUTE_X86_SIMD 0
#endif

#if defined(__linux__)
#include <sched.h>
#endif

namespace virne::network::attribute {
namespace {

enum class NumericOrdering : std::uint8_t {
    less,
    equal,
    greater,
    unordered,
};

constexpr double kTwoTo63 = 9223372036854775808.0;

bool is_integral_number(const AttributeNumber& value) noexcept {
    return !std::holds_alternative<double>(value);
}

std::int64_t integral_value(const AttributeNumber& value) noexcept {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    return std::get<bool>(value) ? std::int64_t{1} : std::int64_t{0};
}

double floating_value(const AttributeNumber& value) noexcept {
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    return static_cast<double>(integral_value(value));
}

NumericOrdering reverse_ordering(NumericOrdering ordering) noexcept {
    switch (ordering) {
        case NumericOrdering::less:
            return NumericOrdering::greater;
        case NumericOrdering::equal:
            return NumericOrdering::equal;
        case NumericOrdering::greater:
            return NumericOrdering::less;
        case NumericOrdering::unordered:
            return NumericOrdering::unordered;
    }
    return NumericOrdering::unordered;
}

NumericOrdering compare_integers(std::int64_t lhs, std::int64_t rhs) noexcept {
    if (lhs < rhs) {
        return NumericOrdering::less;
    }
    if (lhs > rhs) {
        return NumericOrdering::greater;
    }
    return NumericOrdering::equal;
}

NumericOrdering compare_floats(double lhs, double rhs) noexcept {
    if (std::isnan(lhs) || std::isnan(rhs)) {
        return NumericOrdering::unordered;
    }
    if (lhs < rhs) {
        return NumericOrdering::less;
    }
    if (lhs > rhs) {
        return NumericOrdering::greater;
    }
    return NumericOrdering::equal;
}

// Compare without rounding the integer through double. This mirrors Python's
// exact mixed int/float comparison for the complete signed-int64 domain.
NumericOrdering compare_integer_float(
    std::int64_t integer,
    double floating) noexcept {
    if (std::isnan(floating)) {
        return NumericOrdering::unordered;
    }
    if (floating >= kTwoTo63) {
        return NumericOrdering::less;
    }
    if (floating < -kTwoTo63) {
        return NumericOrdering::greater;
    }

    // The range checks make this conversion defined, including exactly -2^63.
    const auto truncated = static_cast<std::int64_t>(floating);
    const auto integer_ordering = compare_integers(integer, truncated);
    if (integer_ordering != NumericOrdering::equal) {
        return integer_ordering;
    }

    const double truncated_as_double = static_cast<double>(truncated);
    if (floating == truncated_as_double) {
        return NumericOrdering::equal;
    }
    return floating > truncated_as_double
        ? NumericOrdering::less
        : NumericOrdering::greater;
}

NumericOrdering compare_numbers(
    const AttributeNumber& lhs,
    const AttributeNumber& rhs) noexcept {
    const bool lhs_integral = is_integral_number(lhs);
    const bool rhs_integral = is_integral_number(rhs);
    if (lhs_integral && rhs_integral) {
        return compare_integers(integral_value(lhs), integral_value(rhs));
    }
    if (!lhs_integral && !rhs_integral) {
        return compare_floats(
            std::get<double>(lhs),
            std::get<double>(rhs));
    }
    if (lhs_integral) {
        return compare_integer_float(
            integral_value(lhs),
            std::get<double>(rhs));
    }
    return reverse_ordering(compare_integer_float(
        integral_value(rhs),
        std::get<double>(lhs)));
}

bool checked_add(
    std::int64_t lhs,
    std::int64_t rhs,
    std::int64_t& result) noexcept {
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((rhs > 0 && lhs > maximum - rhs)
        || (rhs < 0 && lhs < minimum - rhs)) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_subtract(
    std::int64_t lhs,
    std::int64_t rhs,
    std::int64_t& result) noexcept {
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((rhs > 0 && lhs < minimum + rhs)
        || (rhs < 0 && lhs > maximum + rhs)) {
        return false;
    }
    result = lhs - rhs;
    return true;
}

AttributeMethodException numeric_range_error(
    AttributeMethodOperation operation,
    const char* message) {
    return AttributeMethodException(
        AttributeMethodErrorCode::numeric_range,
        operation,
        message);
}

AttributeNumber add_numbers(
    const AttributeNumber& lhs,
    const AttributeNumber& rhs,
    AttributeMethodOperation operation) {
    if (!is_integral_number(lhs) || !is_integral_number(rhs)) {
        return floating_value(lhs) + floating_value(rhs);
    }

    std::int64_t result = 0;
    if (!checked_add(integral_value(lhs), integral_value(rhs), result)) {
        throw numeric_range_error(operation, "integer addition is outside int64");
    }
    return result;
}

AttributeNumber subtract_numbers(
    const AttributeNumber& lhs,
    const AttributeNumber& rhs,
    AttributeMethodOperation operation) {
    if (!is_integral_number(lhs) || !is_integral_number(rhs)) {
        return floating_value(lhs) - floating_value(rhs);
    }

    std::int64_t result = 0;
    if (!checked_subtract(integral_value(lhs), integral_value(rhs), result)) {
        throw numeric_range_error(
            operation,
            "integer subtraction is outside int64");
    }
    return result;
}

AttributeNumber absolute_number(
    AttributeNumber value,
    AttributeMethodOperation operation) {
    if (auto* floating = std::get_if<double>(&value)) {
        *floating = std::fabs(*floating);
        return value;
    }

    const std::int64_t integer = integral_value(value);
    if (integer == std::numeric_limits<std::int64_t>::min()) {
        throw numeric_range_error(
            operation,
            "integer absolute value is outside int64");
    }
    return integer < 0 ? -integer : integer;
}

std::string number_to_string(const AttributeNumber& value) {
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean ? "True" : "False";
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return std::to_string(*integer);
    }

    const double floating = std::get<double>(value);
    if (std::isnan(floating)) {
        return "nan";
    }
    if (std::isinf(floating)) {
        return std::signbit(floating) ? "-inf" : "inf";
    }

    std::array<char, 64> buffer{};
    const auto conversion = std::to_chars(
        buffer.data(),
        buffer.data() + buffer.size(),
        floating,
        std::chars_format::general);
    if (conversion.ec != std::errc{}) {
        return "<floating>";
    }

    std::string result(buffer.data(), conversion.ptr);
    if (result.find_first_of(".eE") == std::string::npos) {
        result += ".0";
    }
    return result;
}

std::string unsupported_value_message(
    std::string_view prefix,
    std::string_view value) {
    std::string message;
    message.reserve(prefix.size() + value.size());
    message.append(prefix);
    message.append(value);
    return message;
}

std::size_t available_cpu_count() noexcept {
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int count = CPU_COUNT(&affinity);
        if (count > 0) {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    const unsigned int reported = std::thread::hardware_concurrency();
    return reported == 0U ? 1U : static_cast<std::size_t>(reported);
}

template<ComparisonOperation Operation, bool Soft>
void calculate_double_range_scalar(
    const double* virtual_values,
    const double* physical_values,
    std::uint8_t* flags,
    double* offsets,
    std::size_t begin,
    std::size_t end) noexcept {
    for (std::size_t index = begin; index < end; ++index) {
        const double virtual_value = virtual_values[index];
        const double physical_value = physical_values[index];
        if constexpr (Soft) {
            flags[index] = std::uint8_t{1};
        } else if constexpr (Operation == ComparisonOperation::greater_equal) {
            flags[index] = static_cast<std::uint8_t>(
                virtual_value >= physical_value);
        } else if constexpr (Operation == ComparisonOperation::less_equal) {
            flags[index] = static_cast<std::uint8_t>(
                virtual_value <= physical_value);
        } else {
            flags[index] = static_cast<std::uint8_t>(
                virtual_value == physical_value);
        }

        if constexpr (Operation == ComparisonOperation::greater_equal) {
            offsets[index] = physical_value - virtual_value;
        } else if constexpr (Operation == ComparisonOperation::less_equal) {
            offsets[index] = virtual_value - physical_value;
        } else {
            offsets[index] = std::fabs(virtual_value - physical_value);
        }
    }
}

#if VIRNE_ATTRIBUTE_X86_SIMD

constexpr std::array<std::uint32_t, 16> make_four_flag_bytes() noexcept {
    std::array<std::uint32_t, 16> values{};
    for (std::uint32_t mask = 0U; mask < values.size(); ++mask) {
        std::uint32_t bytes = 0U;
        for (std::uint32_t lane = 0U; lane < 4U; ++lane) {
            bytes |= ((mask >> lane) & 1U) << (lane * 8U);
        }
        values[mask] = bytes;
    }
    return values;
}

constexpr auto kFourFlagBytes = make_four_flag_bytes();

bool cpu_supports_avx512_batch() noexcept {
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f")
            && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("avx512vl");
    }();
    return supported;
}

bool cpu_supports_avx2_batch() noexcept {
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return supported;
}

template<ComparisonOperation Operation, bool Soft>
__attribute__((target("avx512f,avx512bw,avx512vl")))
void calculate_double_range_avx512(
    const double* virtual_values,
    const double* physical_values,
    std::uint8_t* flags,
    double* offsets,
    std::size_t begin,
    std::size_t end) noexcept {
    constexpr std::size_t lanes = 8U;
    std::size_t index = begin;
    const std::size_t vector_end = begin + ((end - begin) / lanes) * lanes;
    const __m512i absolute_mask = _mm512_set1_epi64(
        static_cast<std::int64_t>(0x7FFFFFFFFFFFFFFFULL));
    const __m128i true_bytes = _mm_set1_epi8(1);
    for (; index < vector_end; index += lanes) {
        const __m512d virtual_value = _mm512_loadu_pd(virtual_values + index);
        const __m512d physical_value = _mm512_loadu_pd(physical_values + index);

        if constexpr (Soft) {
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(flags + index),
                true_bytes);
        } else {
            __mmask8 comparison = 0U;
            if constexpr (Operation == ComparisonOperation::greater_equal) {
                comparison = _mm512_cmp_pd_mask(
                    virtual_value,
                    physical_value,
                    _CMP_GE_OQ);
            } else if constexpr (Operation == ComparisonOperation::less_equal) {
                comparison = _mm512_cmp_pd_mask(
                    virtual_value,
                    physical_value,
                    _CMP_LE_OQ);
            } else {
                comparison = _mm512_cmp_pd_mask(
                    virtual_value,
                    physical_value,
                    _CMP_EQ_OQ);
            }
            const __m128i flag_bytes = _mm_maskz_set1_epi8(
                static_cast<__mmask16>(comparison),
                1);
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(flags + index),
                flag_bytes);
        }

        __m512d offset;
        if constexpr (Operation == ComparisonOperation::greater_equal) {
            offset = _mm512_sub_pd(physical_value, virtual_value);
        } else {
            offset = _mm512_sub_pd(virtual_value, physical_value);
            if constexpr (Operation == ComparisonOperation::equal) {
                offset = _mm512_castsi512_pd(_mm512_and_si512(
                    _mm512_castpd_si512(offset),
                    absolute_mask));
            }
        }
        _mm512_storeu_pd(offsets + index, offset);
    }
    calculate_double_range_scalar<Operation, Soft>(
        virtual_values,
        physical_values,
        flags,
        offsets,
        index,
        end);
}

template<ComparisonOperation Operation, bool Soft>
__attribute__((target("avx2")))
void calculate_double_range_avx2(
    const double* virtual_values,
    const double* physical_values,
    std::uint8_t* flags,
    double* offsets,
    std::size_t begin,
    std::size_t end) noexcept {
    constexpr std::size_t lanes = 4U;
    std::size_t index = begin;
    const std::size_t vector_end = begin + ((end - begin) / lanes) * lanes;
    const __m256i absolute_mask = _mm256_set1_epi64x(
        static_cast<std::int64_t>(0x7FFFFFFFFFFFFFFFULL));
    for (; index < vector_end; index += lanes) {
        const __m256d virtual_value = _mm256_loadu_pd(virtual_values + index);
        const __m256d physical_value = _mm256_loadu_pd(physical_values + index);

        std::uint32_t flag_bytes = 0x01010101U;
        if constexpr (!Soft) {
            __m256d comparison;
            if constexpr (Operation == ComparisonOperation::greater_equal) {
                comparison = _mm256_cmp_pd(
                    virtual_value,
                    physical_value,
                    _CMP_GE_OQ);
            } else if constexpr (Operation == ComparisonOperation::less_equal) {
                comparison = _mm256_cmp_pd(
                    virtual_value,
                    physical_value,
                    _CMP_LE_OQ);
            } else {
                comparison = _mm256_cmp_pd(
                    virtual_value,
                    physical_value,
                    _CMP_EQ_OQ);
            }
            flag_bytes = kFourFlagBytes[static_cast<std::size_t>(
                _mm256_movemask_pd(comparison))];
        }
        std::memcpy(flags + index, &flag_bytes, sizeof(flag_bytes));

        __m256d offset;
        if constexpr (Operation == ComparisonOperation::greater_equal) {
            offset = _mm256_sub_pd(physical_value, virtual_value);
        } else {
            offset = _mm256_sub_pd(virtual_value, physical_value);
            if constexpr (Operation == ComparisonOperation::equal) {
                offset = _mm256_castsi256_pd(_mm256_and_si256(
                    _mm256_castpd_si256(offset),
                    absolute_mask));
            }
        }
        _mm256_storeu_pd(offsets + index, offset);
    }
    calculate_double_range_scalar<Operation, Soft>(
        virtual_values,
        physical_values,
        flags,
        offsets,
        index,
        end);
}

#endif

template<ComparisonOperation Operation, bool Soft>
void calculate_double_range(
    const double* virtual_values,
    const double* physical_values,
    std::uint8_t* flags,
    double* offsets,
    std::size_t begin,
    std::size_t end) noexcept {
#if VIRNE_ATTRIBUTE_X86_SIMD
    if (cpu_supports_avx512_batch()) {
        calculate_double_range_avx512<Operation, Soft>(
            virtual_values,
            physical_values,
            flags,
            offsets,
            begin,
            end);
        return;
    }
    if (cpu_supports_avx2_batch()) {
        calculate_double_range_avx2<Operation, Soft>(
            virtual_values,
            physical_values,
            flags,
            offsets,
            begin,
            end);
        return;
    }
#endif
    calculate_double_range_scalar<Operation, Soft>(
        virtual_values,
        physical_values,
        flags,
        offsets,
        begin,
        end);
}

template<ComparisonOperation Operation, bool Soft>
void calculate_double_batch_impl(
    const std::vector<double>& virtual_values,
    const std::vector<double>& physical_values,
    std::vector<std::uint8_t>& flags,
    std::vector<double>& offsets,
    std::size_t width) {
    const std::size_t count = virtual_values.size();
    auto run_block = [&](std::size_t worker) noexcept {
        const std::size_t base = count / width;
        const std::size_t remainder = count % width;
        const std::size_t begin = worker * base + std::min(worker, remainder);
        const std::size_t end = begin + base + (worker < remainder ? 1U : 0U);
        calculate_double_range<Operation, Soft>(
            virtual_values.data(),
            physical_values.data(),
            flags.data(),
            offsets.data(),
            begin,
            end);
    };

    if (width <= 1U) {
        calculate_double_range<Operation, Soft>(
            virtual_values.data(),
            physical_values.data(),
            flags.data(),
            offsets.data(),
            0U,
            count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(width - 1U);
    try {
        for (std::size_t worker = 1U; worker < width; ++worker) {
            threads.emplace_back(run_block, worker);
        }
    } catch (...) {
        for (std::thread& thread : threads) {
            thread.join();
        }
        // Workers [1, threads.size()] completed. Run only block zero and the
        // blocks whose threads were not constructed. Recomputing completed
        // blocks would be incorrect when offsets aliases either input vector.
        run_block(0U);
        for (std::size_t worker = threads.size() + 1U;
             worker < width;
             ++worker) {
            run_block(worker);
        }
        return;
    }

    run_block(0U);
    for (std::thread& thread : threads) {
        thread.join();
    }
}

template<bool Soft>
void dispatch_double_batch_operation(
    const std::vector<double>& virtual_values,
    const std::vector<double>& physical_values,
    ComparisonOperation operation,
    std::vector<std::uint8_t>& flags,
    std::vector<double>& offsets,
    std::size_t width) {
    switch (operation) {
        case ComparisonOperation::greater_equal:
            calculate_double_batch_impl<
                ComparisonOperation::greater_equal,
                Soft>(
                    virtual_values,
                    physical_values,
                    flags,
                    offsets,
                    width);
            return;
        case ComparisonOperation::less_equal:
            calculate_double_batch_impl<ComparisonOperation::less_equal, Soft>(
                virtual_values,
                physical_values,
                flags,
                offsets,
                width);
            return;
        case ComparisonOperation::equal:
            calculate_double_batch_impl<ComparisonOperation::equal, Soft>(
                virtual_values,
                physical_values,
                flags,
                offsets,
                width);
            return;
    }
    throw AttributeMethodException(
        AttributeMethodErrorCode::unsupported_comparison,
        AttributeMethodOperation::calculate_satisfiability_batch,
        "unsupported comparison operation enum");
}

}  // namespace

AttributeMethodException::AttributeMethodException(
    AttributeMethodErrorCode code,
    AttributeMethodOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

AttributeMethodErrorCode AttributeMethodException::code() const noexcept {
    return code_;
}

AttributeMethodOperation AttributeMethodException::operation() const noexcept {
    return operation_;
}

ResourceUpdateOperation resource_update_operation_from_string(
    std::string_view value) {
    if (value == "+" || value == "add") {
        return ResourceUpdateOperation::add;
    }
    if (value == "-" || value == "sub") {
        return ResourceUpdateOperation::subtract;
    }
    throw AttributeMethodException(
        AttributeMethodErrorCode::unsupported_update_operation,
        AttributeMethodOperation::resolve_update,
        unsupported_value_message("unsupported resource update operation: ", value));
}

ComparisonOperation comparison_operation_from_string(std::string_view value) {
    if (value == ">=" || value == "ge") {
        return ComparisonOperation::greater_equal;
    }
    if (value == "<=" || value == "le") {
        return ComparisonOperation::less_equal;
    }
    if (value == "eq") {
        return ComparisonOperation::equal;
    }
    // "==" is intentionally rejected: the original validator makes its later
    // equality branch unreachable.
    throw AttributeMethodException(
        AttributeMethodErrorCode::unsupported_comparison,
        AttributeMethodOperation::resolve_comparison,
        unsupported_value_message("unsupported comparison operation: ", value));
}

ConstraintRestriction constraint_restriction_from_string(
    std::string_view value) {
    if (value == "hard") {
        return ConstraintRestriction::hard;
    }
    if (value == "soft") {
        return ConstraintRestriction::soft;
    }
    throw AttributeMethodException(
        AttributeMethodErrorCode::invalid_restriction,
        AttributeMethodOperation::resolve_restriction,
        unsupported_value_message("invalid constraint restriction: ", value));
}

AttributeNumberKind attribute_number_kind(const AttributeNumber& value) noexcept {
    if (std::holds_alternative<bool>(value)) {
        return AttributeNumberKind::boolean;
    }
    if (std::holds_alternative<std::int64_t>(value)) {
        return AttributeNumberKind::integer;
    }
    return AttributeNumberKind::floating;
}

bool update_resource_value(
    const AttributeNumber& virtual_value,
    AttributeNumber& physical_value,
    ResourceUpdateOperation operation,
    bool safe,
    std::string_view diagnostic_name) {
    switch (operation) {
        case ResourceUpdateOperation::add: {
            AttributeNumber result = add_numbers(
                physical_value,
                virtual_value,
                AttributeMethodOperation::update_resource);
            physical_value = std::move(result);
            return true;
        }
        case ResourceUpdateOperation::subtract:
            break;
        default:
            throw AttributeMethodException(
                AttributeMethodErrorCode::unsupported_update_operation,
                AttributeMethodOperation::update_resource,
                "unsupported resource update operation enum");
    }

    if (safe
        && compare_numbers(virtual_value, physical_value)
            == NumericOrdering::greater) {
        std::string message;
        const std::string virtual_text = number_to_string(virtual_value);
        const std::string physical_text = number_to_string(physical_value);
        message.reserve(
            diagnostic_name.size()
            + virtual_text.size()
            + physical_text.size()
            + 20);
        message.append(diagnostic_name);
        message.append(": (v = ");
        message.append(virtual_text);
        message.append(") > (p = ");
        message.append(physical_text);
        message.push_back(')');
        throw AttributeMethodException(
            AttributeMethodErrorCode::insufficient_resource,
            AttributeMethodOperation::update_resource,
            std::move(message));
    }

    AttributeNumber result = subtract_numbers(
        physical_value,
        virtual_value,
        AttributeMethodOperation::update_resource);
    physical_value = std::move(result);
    return true;
}

SatisfiabilityResult calculate_satisfiability_values(
    const AttributeNumber& virtual_value,
    const AttributeNumber& physical_value,
    ComparisonOperation operation,
    ConstraintRestriction restriction) {
    switch (operation) {
        case ComparisonOperation::greater_equal:
        case ComparisonOperation::less_equal:
        case ComparisonOperation::equal:
            break;
        default:
            throw AttributeMethodException(
                AttributeMethodErrorCode::unsupported_comparison,
                AttributeMethodOperation::calculate_satisfiability,
                "unsupported comparison operation enum");
    }

    const NumericOrdering ordering = compare_numbers(
        virtual_value,
        physical_value);

    bool flag = false;
    AttributeNumber offset;
    switch (operation) {
        case ComparisonOperation::greater_equal:
            flag = ordering == NumericOrdering::greater
                || ordering == NumericOrdering::equal;
            offset = subtract_numbers(
                physical_value,
                virtual_value,
                AttributeMethodOperation::calculate_satisfiability);
            break;
        case ComparisonOperation::less_equal:
            flag = ordering == NumericOrdering::less
                || ordering == NumericOrdering::equal;
            offset = subtract_numbers(
                virtual_value,
                physical_value,
                AttributeMethodOperation::calculate_satisfiability);
            break;
        case ComparisonOperation::equal:
            flag = ordering == NumericOrdering::equal;
            offset = absolute_number(
                subtract_numbers(
                    virtual_value,
                    physical_value,
                    AttributeMethodOperation::calculate_satisfiability),
                AttributeMethodOperation::calculate_satisfiability);
            break;
        default:
            // The operation was validated before touching either numeric value.
            break;
    }

    switch (restriction) {
        case ConstraintRestriction::hard:
            return {flag, std::move(offset)};
        case ConstraintRestriction::soft:
            return {true, std::move(offset)};
        default:
            throw AttributeMethodException(
                AttributeMethodErrorCode::invalid_restriction,
                AttributeMethodOperation::calculate_satisfiability,
                "invalid constraint restriction enum");
    }
}

std::size_t double_satisfiability_batch_worker_count(
    std::size_t count,
    std::size_t configured_workers) noexcept {
    if (count == 0U) {
        return 0U;
    }
    // Worker width is an explicit typed configuration input. Zero and one
    // both select the deterministic sequential route; wider values are capped
    // by the item count and CPUs available to this process.
    const std::size_t desired = configured_workers <= 1U
        ? 1U
        : configured_workers;
    if (desired <= 1U) {
        return 1U;
    }
    return std::max<std::size_t>(
        1U,
        std::min({desired, count, available_cpu_count()}));
}

void calculate_satisfiability_values_double_batch(
    const std::vector<double>& virtual_values,
    const std::vector<double>& physical_values,
    ComparisonOperation operation,
    ConstraintRestriction restriction,
    std::vector<std::uint8_t>& flags,
    std::vector<double>& offsets,
    std::size_t workers) {
    switch (operation) {
        case ComparisonOperation::greater_equal:
        case ComparisonOperation::less_equal:
        case ComparisonOperation::equal:
            break;
        default:
            throw AttributeMethodException(
                AttributeMethodErrorCode::unsupported_comparison,
                AttributeMethodOperation::calculate_satisfiability_batch,
                "unsupported comparison operation enum");
    }
    switch (restriction) {
        case ConstraintRestriction::hard:
        case ConstraintRestriction::soft:
            break;
        default:
            throw AttributeMethodException(
                AttributeMethodErrorCode::invalid_restriction,
                AttributeMethodOperation::calculate_satisfiability_batch,
                "invalid constraint restriction enum");
    }

    const std::size_t count = virtual_values.size();
    if (physical_values.size() != count
        || flags.size() != count
        || offsets.size() != count) {
        throw AttributeMethodException(
            AttributeMethodErrorCode::invalid_batch_shape,
            AttributeMethodOperation::calculate_satisfiability_batch,
            "double satisfiability batch vectors must have equal sizes");
    }
    if (count == 0U) {
        return;
    }

    const std::size_t width = double_satisfiability_batch_worker_count(
        count,
        workers);
    if (restriction == ConstraintRestriction::soft) {
        dispatch_double_batch_operation<true>(
            virtual_values,
            physical_values,
            operation,
            flags,
            offsets,
            width);
        return;
    }
    dispatch_double_batch_operation<false>(
        virtual_values,
        physical_values,
        operation,
        flags,
        offsets,
        width);
}

}  // namespace virne::network::attribute
