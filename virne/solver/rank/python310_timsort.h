#pragma once

#include "link_rank.h"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace virne::solver::rank::detail {

namespace python310_timsort_detail {

// This is a comparison-schedule port of CPython 3.10.20's
// Objects/listobject.c listsort. It intentionally uses only score < score.
// In particular, NaN is unordered and must not be passed to an STL sorting
// algorithm whose comparator is required to be a strict weak ordering.
class ReverseLinkRankingTimSort {
public:
    explicit ReverseLinkRankingTimSort(LinkRanking& ranking)
        : ranking_(ranking), temporary_((ranking.size() + 1U) / 2U) {}

    void sort() {
        const std::size_t size = ranking_.size();
        if (size < 2U) {
            return;
        }

        // CPython implements reverse=True by reversing before its stable
        // ascending sort and reversing the completed result afterward.
        reverse_range(0U, size);

        const std::size_t minimum_run = compute_minimum_run(size);
        std::size_t remaining = size;
        std::size_t first = 0U;
        do {
            std::size_t run_length = count_run(first, first + remaining);
            if (run_length < minimum_run) {
                const std::size_t forced =
                    remaining <= minimum_run ? remaining : minimum_run;
                binary_sort(first, first + forced, first + run_length);
                run_length = forced;
            }

            push_run(first, run_length);
            merge_collapse();
            first += run_length;
            remaining -= run_length;
        } while (remaining != 0U);

        merge_force_collapse();
        reverse_range(0U, size);
    }

private:
    static constexpr std::size_t minimum_gallop = 7U;
    static constexpr std::size_t maximum_pending_runs = 85U;

    struct Run {
        std::size_t base = 0U;
        std::size_t length = 0U;
    };

    LinkRanking& ranking_;
    LinkRanking temporary_;
    std::array<Run, maximum_pending_runs> pending_{};
    std::size_t pending_count_ = 0U;
    std::size_t current_minimum_gallop_ = minimum_gallop;

    static bool less(
        const LinkRankEntry& left,
        const LinkRankEntry& right) noexcept {
        return left.value < right.value;
    }

    static std::size_t compute_minimum_run(std::size_t size) noexcept {
        std::size_t shifted_bit = 0U;
        while (size >= 64U) {
            shifted_bit |= size & 1U;
            size >>= 1U;
        }
        return size + shifted_bit;
    }

    void reverse_range(std::size_t first, std::size_t last) {
        while (first < last) {
            --last;
            if (first >= last) {
                break;
            }
            using std::swap;
            swap(ranking_[first], ranking_[last]);
            ++first;
        }
    }

    std::size_t count_run(std::size_t first, std::size_t last) {
        std::size_t cursor = first + 1U;
        if (cursor == last) {
            return 1U;
        }

        const bool descending = less(ranking_[cursor], ranking_[cursor - 1U]);
        ++cursor;
        if (descending) {
            while (cursor < last &&
                   less(ranking_[cursor], ranking_[cursor - 1U])) {
                ++cursor;
            }
            reverse_range(first, cursor);
        } else {
            while (cursor < last &&
                   !less(ranking_[cursor], ranking_[cursor - 1U])) {
                ++cursor;
            }
        }
        return cursor - first;
    }

    void binary_sort(
        std::size_t first,
        std::size_t last,
        std::size_t sorted_last) {
        if (sorted_last == first) {
            ++sorted_last;
        }
        for (; sorted_last < last; ++sorted_last) {
            LinkRankEntry pivot = ranking_[sorted_last];
            std::size_t left = first;
            std::size_t right = sorted_last;
            do {
                const std::size_t middle = left + ((right - left) >> 1U);
                if (less(pivot, ranking_[middle])) {
                    right = middle;
                } else {
                    left = middle + 1U;
                }
            } while (left < right);

            for (std::size_t cursor = sorted_last; cursor > left; --cursor) {
                ranking_[cursor] = ranking_[cursor - 1U];
            }
            ranking_[left] = pivot;
        }
    }

    static std::ptrdiff_t next_gallop_offset(
        std::ptrdiff_t offset,
        std::ptrdiff_t maximum) noexcept {
        const std::ptrdiff_t limit =
            (std::numeric_limits<std::ptrdiff_t>::max() - 1) / 2;
        if (offset > limit) {
            return maximum;
        }
        const std::ptrdiff_t next = (offset << 1) + 1;
        return next > maximum ? maximum : next;
    }

    static std::size_t gallop_left(
        const LinkRankEntry& key,
        const LinkRankEntry* values,
        std::size_t size,
        std::size_t hint) noexcept {
        using Difference = std::ptrdiff_t;
        const Difference count = static_cast<Difference>(size);
        const Difference initial_hint = static_cast<Difference>(hint);
        const LinkRankEntry* at_hint = values + initial_hint;
        Difference last_offset = 0;
        Difference offset = 1;

        if (less(*at_hint, key)) {
            const Difference maximum = count - initial_hint;
            while (offset < maximum && less(at_hint[offset], key)) {
                last_offset = offset;
                offset = next_gallop_offset(offset, maximum);
            }
            if (offset > maximum) {
                offset = maximum;
            }
            last_offset += initial_hint;
            offset += initial_hint;
        } else {
            const Difference maximum = initial_hint + 1;
            while (offset < maximum && !less(at_hint[-offset], key)) {
                last_offset = offset;
                offset = next_gallop_offset(offset, maximum);
            }
            if (offset > maximum) {
                offset = maximum;
            }
            const Difference previous = last_offset;
            last_offset = initial_hint - offset;
            offset = initial_hint - previous;
        }

        ++last_offset;
        while (last_offset < offset) {
            const Difference middle =
                last_offset + ((offset - last_offset) >> 1);
            if (less(values[middle], key)) {
                last_offset = middle + 1;
            } else {
                offset = middle;
            }
        }
        return static_cast<std::size_t>(offset);
    }

    static std::size_t gallop_right(
        const LinkRankEntry& key,
        const LinkRankEntry* values,
        std::size_t size,
        std::size_t hint) noexcept {
        using Difference = std::ptrdiff_t;
        const Difference count = static_cast<Difference>(size);
        const Difference initial_hint = static_cast<Difference>(hint);
        const LinkRankEntry* at_hint = values + initial_hint;
        Difference last_offset = 0;
        Difference offset = 1;

        if (less(key, *at_hint)) {
            const Difference maximum = initial_hint + 1;
            while (offset < maximum && less(key, at_hint[-offset])) {
                last_offset = offset;
                offset = next_gallop_offset(offset, maximum);
            }
            if (offset > maximum) {
                offset = maximum;
            }
            const Difference previous = last_offset;
            last_offset = initial_hint - offset;
            offset = initial_hint - previous;
        } else {
            const Difference maximum = count - initial_hint;
            while (offset < maximum && !less(key, at_hint[offset])) {
                last_offset = offset;
                offset = next_gallop_offset(offset, maximum);
            }
            if (offset > maximum) {
                offset = maximum;
            }
            last_offset += initial_hint;
            offset += initial_hint;
        }

        ++last_offset;
        while (last_offset < offset) {
            const Difference middle =
                last_offset + ((offset - last_offset) >> 1);
            if (less(key, values[middle])) {
                offset = middle;
            } else {
                last_offset = middle + 1;
            }
        }
        return static_cast<std::size_t>(offset);
    }

    void push_run(std::size_t base, std::size_t length) {
        if (pending_count_ == pending_.size()) {
            throw std::length_error("CPython 3.10 Timsort run stack overflow");
        }
        pending_[pending_count_++] = Run{base, length};
    }

    void merge_collapse() {
        while (pending_count_ > 1U) {
            std::size_t index = pending_count_ - 2U;
            if ((index > 0U &&
                 pending_[index - 1U].length <=
                     pending_[index].length + pending_[index + 1U].length) ||
                (index > 1U &&
                 pending_[index - 2U].length <=
                     pending_[index - 1U].length + pending_[index].length)) {
                if (pending_[index - 1U].length <
                    pending_[index + 1U].length) {
                    --index;
                }
                merge_at(index);
            } else if (pending_[index].length <=
                       pending_[index + 1U].length) {
                merge_at(index);
            } else {
                break;
            }
        }
    }

    void merge_force_collapse() {
        while (pending_count_ > 1U) {
            std::size_t index = pending_count_ - 2U;
            if (index > 0U && pending_[index - 1U].length <
                                  pending_[index + 1U].length) {
                --index;
            }
            merge_at(index);
        }
    }

    void merge_at(std::size_t index) {
        const std::size_t original_base_a = pending_[index].base;
        const std::size_t original_length_a = pending_[index].length;
        const std::size_t base_b = pending_[index + 1U].base;
        const std::size_t original_length_b = pending_[index + 1U].length;

        pending_[index].length = original_length_a + original_length_b;
        if (index == pending_count_ - 3U) {
            pending_[index + 1U] = pending_[index + 2U];
        }
        --pending_count_;

        std::size_t base_a = original_base_a;
        std::size_t length_a = original_length_a;
        std::size_t length_b = original_length_b;

        const std::size_t prefix = gallop_right(
            ranking_[base_b], ranking_.data() + base_a, length_a, 0U);
        base_a += prefix;
        length_a -= prefix;
        if (length_a == 0U) {
            return;
        }

        length_b = gallop_left(
            ranking_[base_a + length_a - 1U],
            ranking_.data() + base_b,
            length_b,
            length_b - 1U);
        if (length_b == 0U) {
            return;
        }

        if (length_a <= length_b) {
            merge_low(base_a, length_a, base_b, length_b);
        } else {
            merge_high(base_a, length_a, base_b, length_b);
        }
    }

    void copy_temporary_forward(
        std::size_t destination,
        std::size_t source,
        std::size_t count) {
        for (std::size_t offset = 0U; offset < count; ++offset) {
            ranking_[destination + offset] = temporary_[source + offset];
        }
    }

    void copy_ranking_forward(
        std::size_t destination,
        std::size_t source,
        std::size_t count) {
        for (std::size_t offset = 0U; offset < count; ++offset) {
            ranking_[destination + offset] = ranking_[source + offset];
        }
    }

    void copy_ranking_backward(
        std::size_t destination,
        std::size_t source,
        std::size_t count) {
        while (count != 0U) {
            --count;
            ranking_[destination + count] = ranking_[source + count];
        }
    }

    void finish_merge_low(
        std::size_t destination,
        std::size_t cursor_a,
        std::size_t length_a) {
        if (length_a != 0U) {
            copy_temporary_forward(destination, cursor_a, length_a);
        }
    }

    void copy_b_then_last_a(
        std::size_t destination,
        std::size_t cursor_a,
        std::size_t cursor_b,
        std::size_t length_b) {
        copy_ranking_forward(destination, cursor_b, length_b);
        ranking_[destination + length_b] = temporary_[cursor_a];
    }

    void merge_low(
        std::size_t base_a,
        std::size_t length_a,
        std::size_t base_b,
        std::size_t length_b) {
        for (std::size_t index = 0U; index < length_a; ++index) {
            temporary_[index] = ranking_[base_a + index];
        }

        std::size_t cursor_a = 0U;
        std::size_t cursor_b = base_b;
        std::size_t destination = base_a;
        ranking_[destination++] = ranking_[cursor_b++];
        --length_b;
        if (length_b == 0U) {
            finish_merge_low(destination, cursor_a, length_a);
            return;
        }
        if (length_a == 1U) {
            copy_b_then_last_a(destination, cursor_a, cursor_b, length_b);
            return;
        }

        std::size_t local_minimum_gallop = current_minimum_gallop_;
        for (;;) {
            std::size_t a_count = 0U;
            std::size_t b_count = 0U;
            for (;;) {
                if (less(ranking_[cursor_b], temporary_[cursor_a])) {
                    ranking_[destination++] = ranking_[cursor_b++];
                    ++b_count;
                    a_count = 0U;
                    --length_b;
                    if (length_b == 0U) {
                        finish_merge_low(destination, cursor_a, length_a);
                        return;
                    }
                    if (b_count >= local_minimum_gallop) {
                        break;
                    }
                } else {
                    ranking_[destination++] = temporary_[cursor_a++];
                    ++a_count;
                    b_count = 0U;
                    --length_a;
                    if (length_a == 1U) {
                        copy_b_then_last_a(
                            destination, cursor_a, cursor_b, length_b);
                        return;
                    }
                    if (a_count >= local_minimum_gallop) {
                        break;
                    }
                }
            }

            ++local_minimum_gallop;
            do {
                if (local_minimum_gallop > 1U) {
                    --local_minimum_gallop;
                }
                current_minimum_gallop_ = local_minimum_gallop;

                const std::size_t copied_a = gallop_right(
                    ranking_[cursor_b],
                    temporary_.data() + cursor_a,
                    length_a,
                    0U);
                a_count = copied_a;
                if (copied_a != 0U) {
                    copy_temporary_forward(destination, cursor_a, copied_a);
                    destination += copied_a;
                    cursor_a += copied_a;
                    length_a -= copied_a;
                    if (length_a == 1U) {
                        copy_b_then_last_a(
                            destination, cursor_a, cursor_b, length_b);
                        return;
                    }
                    if (length_a == 0U) {
                        return;
                    }
                }

                ranking_[destination++] = ranking_[cursor_b++];
                --length_b;
                if (length_b == 0U) {
                    finish_merge_low(destination, cursor_a, length_a);
                    return;
                }

                const std::size_t copied_b = gallop_left(
                    temporary_[cursor_a],
                    ranking_.data() + cursor_b,
                    length_b,
                    0U);
                b_count = copied_b;
                if (copied_b != 0U) {
                    copy_ranking_forward(destination, cursor_b, copied_b);
                    destination += copied_b;
                    cursor_b += copied_b;
                    length_b -= copied_b;
                    if (length_b == 0U) {
                        finish_merge_low(destination, cursor_a, length_a);
                        return;
                    }
                }

                ranking_[destination++] = temporary_[cursor_a++];
                --length_a;
                if (length_a == 1U) {
                    copy_b_then_last_a(
                        destination, cursor_a, cursor_b, length_b);
                    return;
                }
            } while (a_count >= minimum_gallop ||
                     b_count >= minimum_gallop);

            ++local_minimum_gallop;
            current_minimum_gallop_ = local_minimum_gallop;
        }
    }

    void finish_merge_high(
        std::size_t destination,
        std::size_t length_b) {
        if (length_b != 0U) {
            copy_temporary_forward(destination - length_b, 0U, length_b);
        }
    }

    void copy_a_then_first_b(
        std::size_t destination,
        std::size_t cursor_a,
        std::size_t length_a) {
        const std::size_t source = cursor_a - length_a;
        const std::size_t target = destination - length_a;
        copy_ranking_backward(target, source, length_a);
        ranking_[target - 1U] = temporary_[0U];
    }

    void merge_high(
        std::size_t base_a,
        std::size_t length_a,
        std::size_t base_b,
        std::size_t length_b) {
        for (std::size_t index = 0U; index < length_b; ++index) {
            temporary_[index] = ranking_[base_b + index];
        }

        std::size_t cursor_a = base_a + length_a;
        std::size_t cursor_b = length_b;
        std::size_t destination = base_b + length_b;
        ranking_[--destination] = ranking_[--cursor_a];
        --length_a;
        if (length_a == 0U) {
            finish_merge_high(destination, length_b);
            return;
        }
        if (length_b == 1U) {
            copy_a_then_first_b(destination, cursor_a, length_a);
            return;
        }

        std::size_t local_minimum_gallop = current_minimum_gallop_;
        for (;;) {
            std::size_t a_count = 0U;
            std::size_t b_count = 0U;
            for (;;) {
                if (less(temporary_[cursor_b - 1U],
                         ranking_[cursor_a - 1U])) {
                    ranking_[--destination] = ranking_[--cursor_a];
                    ++a_count;
                    b_count = 0U;
                    --length_a;
                    if (length_a == 0U) {
                        finish_merge_high(destination, length_b);
                        return;
                    }
                    if (a_count >= local_minimum_gallop) {
                        break;
                    }
                } else {
                    ranking_[--destination] = temporary_[--cursor_b];
                    ++b_count;
                    a_count = 0U;
                    --length_b;
                    if (length_b == 1U) {
                        copy_a_then_first_b(destination, cursor_a, length_a);
                        return;
                    }
                    if (b_count >= local_minimum_gallop) {
                        break;
                    }
                }
            }

            ++local_minimum_gallop;
            do {
                if (local_minimum_gallop > 1U) {
                    --local_minimum_gallop;
                }
                current_minimum_gallop_ = local_minimum_gallop;

                const std::size_t split_a = gallop_right(
                    temporary_[cursor_b - 1U],
                    ranking_.data() + base_a,
                    length_a,
                    length_a - 1U);
                const std::size_t copied_a = length_a - split_a;
                a_count = copied_a;
                if (copied_a != 0U) {
                    destination -= copied_a;
                    cursor_a -= copied_a;
                    copy_ranking_backward(
                        destination, cursor_a, copied_a);
                    length_a -= copied_a;
                    if (length_a == 0U) {
                        finish_merge_high(destination, length_b);
                        return;
                    }
                }

                ranking_[--destination] = temporary_[--cursor_b];
                --length_b;
                if (length_b == 1U) {
                    copy_a_then_first_b(destination, cursor_a, length_a);
                    return;
                }

                const std::size_t split_b = gallop_left(
                    ranking_[cursor_a - 1U],
                    temporary_.data(),
                    length_b,
                    length_b - 1U);
                const std::size_t copied_b = length_b - split_b;
                b_count = copied_b;
                if (copied_b != 0U) {
                    destination -= copied_b;
                    cursor_b -= copied_b;
                    copy_temporary_forward(
                        destination, cursor_b, copied_b);
                    length_b -= copied_b;
                    if (length_b == 1U) {
                        copy_a_then_first_b(
                            destination, cursor_a, length_a);
                        return;
                    }
                    if (length_b == 0U) {
                        return;
                    }
                }

                ranking_[--destination] = ranking_[--cursor_a];
                --length_a;
                if (length_a == 0U) {
                    finish_merge_high(destination, length_b);
                    return;
                }
            } while (a_count >= minimum_gallop ||
                     b_count >= minimum_gallop);

            ++local_minimum_gallop;
            current_minimum_gallop_ = local_minimum_gallop;
        }
    }
};

}  // namespace python310_timsort_detail

// Exact cold path for Python's sorted(..., key=entry.value, reverse=True).
// All allocations and the complete sort happen on a private copy; the caller's
// ranking changes only through the final noexcept vector swap.
inline void python310_timsort_reverse(LinkRanking& ranking) {
    if (ranking.size() < 2U) {
        return;
    }
    if (ranking.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("Link ranking exceeds CPython index range");
    }

    LinkRanking sorted = ranking;
    python310_timsort_detail::ReverseLinkRankingTimSort sorter(sorted);
    sorter.sort();
    ranking.swap(sorted);
}

}  // namespace virne::solver::rank::detail
