#include "python_int_set_order.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace virne::solver::heuristic::detail {
namespace {

constexpr std::size_t minimum_table_size = 8U;
constexpr std::size_t linear_probes = 9U;
constexpr std::size_t perturb_shift = 5U;
constexpr std::size_t python_hash_modulus =
    (std::size_t{1U} << 61U) - 1U;

enum class SlotState : std::uint8_t {
    empty,
    active,
    dummy,
};

struct Slot {
    Vertex key = 0U;
    std::size_t hash = 0U;
    SlotState state = SlotState::empty;
};

std::size_t integer_hash(Vertex key) noexcept {
    // Physical graph storage cannot approach this modulus in practice, but
    // retaining it makes the numeric seam match PyLong/np.int64 hashing.
    return static_cast<std::size_t>(key) % python_hash_modulus;
}

class Cpython310IntSet {
public:
    Cpython310IntSet()
        : table_(minimum_table_size) {}

    explicit Cpython310IntSet(const std::vector<Vertex>& input)
        : Cpython310IntSet() {
        for (const Vertex key : input) {
            add(key);
        }
    }

    std::size_t size() const noexcept {
        return used_;
    }

    std::vector<Vertex> ordered_keys() const {
        std::vector<Vertex> result;
        result.reserve(used_);
        for (const auto& slot : table_) {
            if (slot.state == SlotState::active) {
                result.push_back(slot.key);
            }
        }
        return result;
    }

    bool contains(Vertex key) const noexcept {
        return find_active(key, integer_hash(key)) != table_.size();
    }

    void add(Vertex key) {
        add_entry(key, integer_hash(key));
    }

    static Cpython310IntSet difference(
        const Cpython310IntSet& left,
        const Cpython310IntSet& right) {
        if ((left.used_ >> 2U) > right.used_) {
            Cpython310IntSet result = left.copy();
            result.difference_update(right);
            return result;
        }

        Cpython310IntSet result;
        for (const auto& slot : left.table_) {
            if (slot.state == SlotState::active &&
                !right.contains_entry(slot.key, slot.hash)) {
                result.add_entry(slot.key, slot.hash);
            }
        }
        return result;
    }

private:
    std::vector<Slot> table_;
    std::size_t fill_ = 0U;
    std::size_t used_ = 0U;

    std::size_t mask() const noexcept {
        return table_.size() - 1U;
    }

    bool contains_entry(Vertex key, std::size_t hash) const noexcept {
        return find_active(key, hash) != table_.size();
    }

    std::size_t find_active(Vertex key, std::size_t hash) const noexcept {
        const std::size_t table_mask = mask();
        std::size_t perturb = hash;
        std::size_t index = hash & table_mask;
        for (;;) {
            const std::size_t count =
                index + linear_probes <= table_mask
                    ? linear_probes + 1U
                    : 1U;
            for (std::size_t probe = 0U; probe < count; ++probe) {
                const std::size_t slot_index = index + probe;
                const auto& slot = table_[slot_index];
                if (slot.state == SlotState::empty) {
                    return table_.size();
                }
                if (slot.state == SlotState::active &&
                    slot.hash == hash && slot.key == key) {
                    return slot_index;
                }
            }
            perturb >>= perturb_shift;
            index = (index * 5U + 1U + perturb) & table_mask;
        }
    }

    void add_entry(Vertex key, std::size_t hash) {
        const std::size_t table_mask = mask();
        std::size_t perturb = hash;
        std::size_t index = hash & table_mask;
        std::size_t free_slot = table_.size();

        for (;;) {
            const std::size_t count =
                index + linear_probes <= table_mask
                    ? linear_probes + 1U
                    : 1U;
            for (std::size_t probe = 0U; probe < count; ++probe) {
                const std::size_t slot_index = index + probe;
                auto& slot = table_[slot_index];
                if (slot.state == SlotState::empty) {
                    if (free_slot != table_.size()) {
                        auto& destination = table_[free_slot];
                        destination.key = key;
                        destination.hash = hash;
                        destination.state = SlotState::active;
                        ++used_;
                        return;
                    }
                    slot.key = key;
                    slot.hash = hash;
                    slot.state = SlotState::active;
                    ++fill_;
                    ++used_;
                    if (fill_ * 5U >= table_mask * 3U) {
                        resize(used_ > 50000U ? used_ * 2U : used_ * 4U);
                    }
                    return;
                }
                if (slot.state == SlotState::active &&
                    slot.hash == hash && slot.key == key) {
                    return;
                }
                if (slot.state == SlotState::dummy) {
                    free_slot = slot_index;
                }
            }
            perturb >>= perturb_shift;
            index = (index * 5U + 1U + perturb) & table_mask;
        }
    }

    static void insert_clean(
        std::vector<Slot>& table,
        Vertex key,
        std::size_t hash) noexcept {
        const std::size_t table_mask = table.size() - 1U;
        std::size_t perturb = hash;
        std::size_t index = hash & table_mask;
        for (;;) {
            auto& first = table[index];
            if (first.state == SlotState::empty) {
                first = Slot{key, hash, SlotState::active};
                return;
            }
            if (index + linear_probes <= table_mask) {
                for (std::size_t probe = 1U;
                     probe <= linear_probes;
                     ++probe) {
                    auto& slot = table[index + probe];
                    if (slot.state == SlotState::empty) {
                        slot = Slot{key, hash, SlotState::active};
                        return;
                    }
                }
            }
            perturb >>= perturb_shift;
            index = (index * 5U + 1U + perturb) & table_mask;
        }
    }

    void resize(std::size_t minimum_used) {
        std::size_t new_size = minimum_table_size;
        while (new_size <= minimum_used) {
            if (new_size >
                std::numeric_limits<std::size_t>::max() / 2U) {
                throw std::length_error(
                    "CPython candidate set table size overflow");
            }
            new_size *= 2U;
        }

        std::vector<Slot> replacement(new_size);
        for (const auto& slot : table_) {
            if (slot.state == SlotState::active) {
                insert_clean(replacement, slot.key, slot.hash);
            }
        }
        table_ = std::move(replacement);
        fill_ = used_;
    }

    void merge(const Cpython310IntSet& other) {
        if (other.used_ == 0U) {
            return;
        }
        if ((fill_ + other.used_) * 5U >= mask() * 3U) {
            resize((used_ + other.used_) * 2U);
        }
        if (fill_ == 0U && mask() == other.mask() &&
            other.fill_ == other.used_) {
            table_ = other.table_;
            fill_ = other.fill_;
            used_ = other.used_;
            return;
        }
        if (fill_ == 0U) {
            fill_ = other.used_;
            used_ = other.used_;
            for (const auto& slot : other.table_) {
                if (slot.state == SlotState::active) {
                    insert_clean(table_, slot.key, slot.hash);
                }
            }
            return;
        }
        for (const auto& slot : other.table_) {
            if (slot.state == SlotState::active) {
                add_entry(slot.key, slot.hash);
            }
        }
    }

    Cpython310IntSet copy() const {
        Cpython310IntSet result;
        result.merge(*this);
        return result;
    }

    bool discard_entry(Vertex key, std::size_t hash) noexcept {
        const std::size_t slot_index = find_active(key, hash);
        if (slot_index == table_.size()) {
            return false;
        }
        table_[slot_index].state = SlotState::dummy;
        table_[slot_index].hash = std::numeric_limits<std::size_t>::max();
        --used_;
        return true;
    }

    static Cpython310IntSet intersection(
        const Cpython310IntSet& left,
        const Cpython310IntSet& right) {
        const Cpython310IntSet* membership = &left;
        const Cpython310IntSet* iteration = &right;
        if (right.used_ <= left.used_) {
            membership = &left;
            iteration = &right;
        } else {
            membership = &right;
            iteration = &left;
        }

        Cpython310IntSet result;
        for (const auto& slot : iteration->table_) {
            if (slot.state == SlotState::active &&
                membership->contains_entry(slot.key, slot.hash)) {
                result.add_entry(slot.key, slot.hash);
            }
        }
        return result;
    }

    void difference_update(const Cpython310IntSet& other) {
        Cpython310IntSet reduced;
        const Cpython310IntSet* removal = &other;
        if ((other.used_ >> 3U) > used_) {
            reduced = intersection(*this, other);
            removal = &reduced;
        }
        for (const auto& slot : removal->table_) {
            if (slot.state == SlotState::active) {
                (void)discard_entry(slot.key, slot.hash);
            }
        }
        if (fill_ - used_ > mask() / 4U) {
            resize(used_ > 50000U ? used_ * 2U : used_ * 4U);
        }
    }
};

} // namespace

std::vector<Vertex> cpython310_int_set_difference_order(
    const std::vector<Vertex>& left,
    const std::vector<Vertex>& right) {
    const Cpython310IntSet left_set(left);
    const Cpython310IntSet right_set(right);
    return Cpython310IntSet::difference(left_set, right_set).ordered_keys();
}

} // namespace virne::solver::heuristic::detail
