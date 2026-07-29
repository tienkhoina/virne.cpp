#include "../virne/solver/rank/python310_generic_timsort.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using virne::solver::rank::detail::python310_timsort;
using virne::solver::rank::detail::python310_timsort_reverse;

double double_from_bits(std::uint64_t bits) noexcept {
    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void fnv1a_mix(std::uint64_t value, std::uint64_t& hash) noexcept {
    constexpr std::uint64_t prime = UINT64_C(1099511628211);
    for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
        const std::uint64_t octet = (value >> shift) & UINT64_C(0xff);
        hash ^= octet;
        hash *= prime;
    }
}

struct ScalarEntry {
    std::uint64_t id = 0U;
    double value = 0.0;
};

void probe_scalar_reverse() {
    constexpr std::array<std::uint64_t, 15U> patterns{{
        UINT64_C(0x0000000000000000),  // +0
        UINT64_C(0x8000000000000000),  // -0
        UINT64_C(0x3ff0000000000000),  // +1
        UINT64_C(0xbff0000000000000),  // -1
        UINT64_C(0x4008000000000000),  // +3
        UINT64_C(0xc008000000000000),  // -3
        UINT64_C(0x7ff0000000000000),  // +inf
        UINT64_C(0xfff0000000000000),  // -inf
        UINT64_C(0x7ff8000000000000),  // quiet NaN; payload added below
        UINT64_C(0x7ff0000000000001),  // signaling NaN; payload added below
        UINT64_C(0x4000000000000000),  // +2
        UINT64_C(0xbfe0000000000000),  // -0.5
        UINT64_C(0x3fe0000000000000),  // +0.5
        UINT64_C(0x4014000000000000),  // +5
        UINT64_C(0xc014000000000000),  // -5
    }};

    std::vector<ScalarEntry> values;
    values.reserve(96U);
    for (std::size_t index = 0U; index < 96U; ++index) {
        const std::size_t pattern = index % patterns.size();
        std::uint64_t bits = patterns[pattern];
        if (pattern == 8U || pattern == 9U) {
            bits |= static_cast<std::uint64_t>(index);
        }
        values.push_back(ScalarEntry{
            static_cast<std::uint64_t>(index), double_from_bits(bits)});
    }

    python310_timsort_reverse(
        values,
        [](const ScalarEntry& left, const ScalarEntry& right) noexcept {
            return left.value < right.value;
        });

    // Pinned from the already-differential-tested CPython 3.10.20 LinkRank
    // oracle. Ninety-six entries force multiple natural runs and merge/gallop
    // paths; hashing both id and raw value bits also checks stable payload moves.
    constexpr std::uint64_t expected_hash = UINT64_C(0x43d00c07a46a6946);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const ScalarEntry& entry : values) {
        fnv1a_mix(entry.id, hash);
        fnv1a_mix(double_bits(entry.value), hash);
    }
    if (hash != expected_hash) {
        throw std::runtime_error("scalar reverse CPython order mismatch");
    }
}

struct NpsEntry {
    std::uint32_t id = 0U;
    double distance = 0.0;
    double nrm = 0.0;
};

using TracePair = std::pair<std::uint32_t, std::uint32_t>;

struct NpsLess {
    std::vector<TracePair>* trace = nullptr;

    bool operator()(const NpsEntry& left, const NpsEntry& right) const {
        trace->emplace_back(left.id, right.id);

        // Python tuple comparison for key=(distance, -nrm): each preceding
        // field is tested for equality before '<' is applied to the first
        // unequal field. A NaN distance is therefore unordered immediately.
        if (left.distance == right.distance) {
            return -left.nrm < -right.nrm;
        }
        return left.distance < right.distance;
    }
};

void probe_nps_composite_ascending() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<NpsEntry> values{
        {0U, 2.0, 4.0},
        {1U, 1.0, 10.0},
        {2U, 1.0, 3.0},
        {3U, nan, 5.0},
        {4U, 0.0, 8.0},
        {5U, 1.0, nan},
        {6U, 1.0, 3.0},
        {7U, nan, -2.0},
        {8U, 2.0, 7.0},
        {9U, 0.0, 8.0},
        {10U, 2.0, nan},
        {11U, -0.0, 9.0},
        {12U, 1.0, 10.0},
        {13U, 3.0, -1.0},
        {14U, std::numeric_limits<double>::infinity(), 2.0},
        {15U, -std::numeric_limits<double>::infinity(), 1.0},
    };

    std::vector<TracePair> trace;
    python310_timsort(values, NpsLess{&trace});

    constexpr std::array<std::uint32_t, 16U> expected_ids{{
        15U, 11U, 4U, 9U, 1U, 12U, 2U, 6U,
        8U, 0U, 3U, 5U, 7U, 10U, 13U, 14U,
    }};
    for (std::size_t index = 0U; index < expected_ids.size(); ++index) {
        if (values[index].id != expected_ids[index]) {
            throw std::runtime_error("NPS composite CPython order mismatch");
        }
    }

    // CPython 3.10.20 comparison trace. This catches an implementation that
    // happens to produce the same order but changes the non-SWO schedule.
    constexpr std::array<TracePair, 45U> expected_trace{{
        {1U, 0U}, {2U, 1U}, {2U, 0U}, {2U, 1U}, {3U, 2U},
        {3U, 0U}, {4U, 0U}, {4U, 2U}, {4U, 1U}, {5U, 2U},
        {5U, 3U}, {6U, 0U}, {6U, 1U}, {6U, 2U}, {7U, 6U},
        {7U, 3U}, {7U, 5U}, {8U, 0U}, {8U, 2U}, {8U, 6U},
        {9U, 8U}, {9U, 2U}, {9U, 1U}, {9U, 4U}, {10U, 8U},
        {10U, 5U}, {10U, 7U}, {11U, 8U}, {11U, 1U}, {11U, 9U},
        {11U, 4U}, {12U, 8U}, {12U, 1U}, {12U, 6U}, {12U, 2U},
        {13U, 6U}, {13U, 5U}, {13U, 10U}, {14U, 8U}, {14U, 7U},
        {14U, 13U}, {15U, 8U}, {15U, 1U}, {15U, 4U}, {15U, 11U},
    }};
    if (trace.size() != expected_trace.size()) {
        throw std::runtime_error("NPS composite comparison count mismatch");
    }
    for (std::size_t index = 0U; index < expected_trace.size(); ++index) {
        if (trace[index] != expected_trace[index]) {
            throw std::runtime_error("NPS composite comparison schedule mismatch");
        }
    }
}

}  // namespace

int main() {
    try {
        probe_scalar_reverse();
        probe_nps_composite_ascending();
        std::cout << "python310 generic timsort probe: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "python310 generic timsort probe: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
