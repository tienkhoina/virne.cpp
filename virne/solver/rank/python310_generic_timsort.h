#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace virne::solver::rank::detail {

namespace python310_generic_timsort_detail {

// This is a comparison-schedule port of CPython 3.10.20's
// Objects/listobject.c listsort. It intentionally delegates every key
// comparison to Less and preserves the exact comparison schedule. In
// particular, a comparator using IEEE-754 '<' may be non-SWO in the presence
// of NaNs and therefore must not be passed to an STL sorting algorithm.
template <typename T, typename Less>
class Python310TimSort {
public:
    Python310TimSort(std::vector<T>& values, Less& less)
        : values_(values),
          temporary_((values.size() + 1U) / 2U, values.front()),
          less_(less) {}

    void sort(bool reverse) {
        const std::size_t size = values_.size();
        if (size < 2U) {
            return;
        }

        // CPython implements reverse=True by reversing before its stable
        // ascending sort and reversing the completed result afterward.
        if (reverse) {
            reverse_range(0U, size);
        }

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
        if (reverse) {
            reverse_range(0U, size);
        }
    }

private:
    static constexpr std::size_t minimum_gallop = 7U;
    static constexpr std::size_t maximum_pending_runs = 85U;

    struct Run {
        std::size_t base = 0U;
        std::size_t length = 0U;
    };

    std::vector<T>& values_;
    std::vector<T> temporary_;
    Less& less_;
    std::array<Run, maximum_pending_runs> pending_{};
    std::size_t pending_count_ = 0U;
    std::size_t current_minimum_gallop_ = minimum_gallop;

    bool less(const T& left, const T& right) {
        return static_cast<bool>(less_(left, right));
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
            swap(values_[first], values_[last]);
            ++first;
        }
    }

    std::size_t count_run(std::size_t first, std::size_t last) {
        std::size_t cursor = first + 1U;
        if (cursor == last) {
            return 1U;
        }

        const bool descending = less(values_[cursor], values_[cursor - 1U]);
        ++cursor;
        if (descending) {
            while (cursor < last &&
                   less(values_[cursor], values_[cursor - 1U])) {
                ++cursor;
            }
            reverse_range(first, cursor);
        } else {
            while (cursor < last &&
                   !less(values_[cursor], values_[cursor - 1U])) {
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
            T pivot = values_[sorted_last];
            std::size_t left = first;
            std::size_t right = sorted_last;
            do {
                const std::size_t middle = left + ((right - left) >> 1U);
                if (less(pivot, values_[middle])) {
                    right = middle;
                } else {
                    left = middle + 1U;
                }
            } while (left < right);

            for (std::size_t cursor = sorted_last; cursor > left; --cursor) {
                values_[cursor] = values_[cursor - 1U];
            }
            values_[left] = pivot;
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

    std::size_t gallop_left(
        const T& key,
        const T* values,
        std::size_t size,
        std::size_t hint) {
        using Difference = std::ptrdiff_t;
        const Difference count = static_cast<Difference>(size);
        const Difference initial_hint = static_cast<Difference>(hint);
        const T* at_hint = values + initial_hint;
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

    std::size_t gallop_right(
        const T& key,
        const T* values,
        std::size_t size,
        std::size_t hint) {
        using Difference = std::ptrdiff_t;
        const Difference count = static_cast<Difference>(size);
        const Difference initial_hint = static_cast<Difference>(hint);
        const T* at_hint = values + initial_hint;
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
            values_[base_b], values_.data() + base_a, length_a, 0U);
        base_a += prefix;
        length_a -= prefix;
        if (length_a == 0U) {
            return;
        }

        length_b = gallop_left(
            values_[base_a + length_a - 1U],
            values_.data() + base_b,
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
            values_[destination + offset] = temporary_[source + offset];
        }
    }

    void copy_values_forward(
        std::size_t destination,
        std::size_t source,
        std::size_t count) {
        for (std::size_t offset = 0U; offset < count; ++offset) {
            values_[destination + offset] = values_[source + offset];
        }
    }

    void copy_values_backward(
        std::size_t destination,
        std::size_t source,
        std::size_t count) {
        while (count != 0U) {
            --count;
            values_[destination + count] = values_[source + count];
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
        copy_values_forward(destination, cursor_b, length_b);
        values_[destination + length_b] = temporary_[cursor_a];
    }

    void merge_low(
        std::size_t base_a,
        std::size_t length_a,
        std::size_t base_b,
        std::size_t length_b) {
        for (std::size_t index = 0U; index < length_a; ++index) {
            temporary_[index] = values_[base_a + index];
        }

        std::size_t cursor_a = 0U;
        std::size_t cursor_b = base_b;
        std::size_t destination = base_a;
        values_[destination++] = values_[cursor_b++];
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
                if (less(values_[cursor_b], temporary_[cursor_a])) {
                    values_[destination++] = values_[cursor_b++];
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
                    values_[destination++] = temporary_[cursor_a++];
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
                    values_[cursor_b],
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

                values_[destination++] = values_[cursor_b++];
                --length_b;
                if (length_b == 0U) {
                    finish_merge_low(destination, cursor_a, length_a);
                    return;
                }

                const std::size_t copied_b = gallop_left(
                    temporary_[cursor_a],
                    values_.data() + cursor_b,
                    length_b,
                    0U);
                b_count = copied_b;
                if (copied_b != 0U) {
                    copy_values_forward(destination, cursor_b, copied_b);
                    destination += copied_b;
                    cursor_b += copied_b;
                    length_b -= copied_b;
                    if (length_b == 0U) {
                        finish_merge_low(destination, cursor_a, length_a);
                        return;
                    }
                }

                values_[destination++] = temporary_[cursor_a++];
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
        copy_values_backward(target, source, length_a);
        values_[target - 1U] = temporary_[0U];
    }

    void merge_high(
        std::size_t base_a,
        std::size_t length_a,
        std::size_t base_b,
        std::size_t length_b) {
        for (std::size_t index = 0U; index < length_b; ++index) {
            temporary_[index] = values_[base_b + index];
        }

        std::size_t cursor_a = base_a + length_a;
        std::size_t cursor_b = length_b;
        std::size_t destination = base_b + length_b;
        values_[--destination] = values_[--cursor_a];
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
                         values_[cursor_a - 1U])) {
                    values_[--destination] = values_[--cursor_a];
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
                    values_[--destination] = temporary_[--cursor_b];
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
                    values_.data() + base_a,
                    length_a,
                    length_a - 1U);
                const std::size_t copied_a = length_a - split_a;
                a_count = copied_a;
                if (copied_a != 0U) {
                    destination -= copied_a;
                    cursor_a -= copied_a;
                    copy_values_backward(
                        destination, cursor_a, copied_a);
                    length_a -= copied_a;
                    if (length_a == 0U) {
                        finish_merge_high(destination, length_b);
                        return;
                    }
                }

                values_[--destination] = temporary_[--cursor_b];
                --length_b;
                if (length_b == 1U) {
                    copy_a_then_first_b(destination, cursor_a, length_a);
                    return;
                }

                const std::size_t split_b = gallop_left(
                    values_[cursor_a - 1U],
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

                values_[--destination] = values_[--cursor_a];
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

}  // namespace python310_generic_timsort_detail

// Exact cold paths for CPython 3.10.20 list.sort/sorted. A private
// copy gives the caller a strong exception guarantee; only the final vector
// swap mutates values. Less may intentionally be a non-SWO comparator, such
// as IEEE-754 '<' over values containing NaNs.
template <typename T, typename Less>
void python310_timsort(std::vector<T>& values, Less less) {
    if (values.size() < 2U) {
        return;
    }
    if (values.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("Sequence exceeds CPython index range");
    }

    std::vector<T> sorted = values;
    python310_generic_timsort_detail::Python310TimSort<T, Less> sorter(
        sorted, less);
    sorter.sort(false);
    values.swap(sorted);
}

template <typename T, typename Less>
void python310_timsort_reverse(std::vector<T>& values, Less less) {
    if (values.size() < 2U) {
        return;
    }
    if (values.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("Sequence exceeds CPython index range");
    }

    std::vector<T> sorted = values;
    python310_generic_timsort_detail::Python310TimSort<T, Less> sorter(
        sorted, less);
    sorter.sort(true);
    values.swap(sorted);
}

}  // namespace virne::solver::rank::detail
