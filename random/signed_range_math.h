#pragma once

#include <cstdint>
#include <limits>

namespace random_detail {

// Precondition: lower < upper. Unsigned subtraction yields the exact positive
// distance across the complete signed int64 domain without signed overflow.
inline std::uint64_t ordered_distance(
    const std::int64_t lower,
    const std::int64_t upper) noexcept {
    return static_cast<std::uint64_t>(upper)
        - static_cast<std::uint64_t>(lower);
}

inline std::uint64_t unsigned_magnitude(
    const std::int64_t value) noexcept {
    return value >= 0
        ? static_cast<std::uint64_t>(value)
        : std::uint64_t{0} - static_cast<std::uint64_t>(value);
}

inline std::int64_t signed_from_twos_complement(
    const std::uint64_t bits) noexcept {
    constexpr auto sign_bit = std::uint64_t{1} << 63U;
    if (bits < sign_bit) {
        return static_cast<std::int64_t>(bits);
    }
    const std::uint64_t magnitude = (~bits) + 1U;
    if (magnitude == sign_bit) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return -static_cast<std::int64_t>(magnitude);
}

inline std::int64_t add_offset(
    const std::int64_t base,
    const std::uint64_t offset) noexcept {
    return signed_from_twos_complement(
        static_cast<std::uint64_t>(base) + offset);
}

inline std::int64_t add_scaled_offset(
    const std::int64_t base,
    const std::int64_t step,
    const std::uint64_t offset) noexcept {
    return signed_from_twos_complement(
        static_cast<std::uint64_t>(base)
        + static_cast<std::uint64_t>(step) * offset);
}

}  // namespace random_detail
