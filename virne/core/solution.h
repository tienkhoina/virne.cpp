#pragma once

#include "attribute/attribute_method.h"
#include "virtual_network.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace virne::core {

using SolutionNodeId = std::int64_t;

struct SolutionLink {
    SolutionNodeId source = 0;
    SolutionNodeId target = 0;

    friend bool operator==(
        const SolutionLink& left,
        const SolutionLink& right) noexcept {
        return left.source == right.source && left.target == right.target;
    }

    friend bool operator!=(
        const SolutionLink& left,
        const SolutionLink& right) noexcept {
        return !(left == right);
    }
};

struct SolutionLinkHash {
    std::size_t operator()(const SolutionLink& link) const noexcept {
        const std::size_t left = std::hash<SolutionNodeId>{}(link.source);
        const std::size_t right = std::hash<SolutionNodeId>{}(link.target);
        return left ^ (right + std::size_t{0x9e3779b9U} +
                       (left << 6U) + (left >> 2U));
    }
};

struct NodeSlotInfoKey {
    SolutionNodeId virtual_node = 0;
    SolutionNodeId physical_node = 0;

    friend bool operator==(
        const NodeSlotInfoKey& left,
        const NodeSlotInfoKey& right) noexcept {
        return left.virtual_node == right.virtual_node &&
               left.physical_node == right.physical_node;
    }
};

struct NodeSlotInfoKeyHash {
    std::size_t operator()(const NodeSlotInfoKey& key) const noexcept {
        return SolutionLinkHash{}(
            SolutionLink{key.virtual_node, key.physical_node});
    }
};

struct LinkPathInfoKey {
    SolutionLink virtual_link;
    SolutionLink physical_link;

    friend bool operator==(
        const LinkPathInfoKey& left,
        const LinkPathInfoKey& right) noexcept {
        return left.virtual_link == right.virtual_link &&
               left.physical_link == right.physical_link;
    }
};

struct LinkPathInfoKeyHash {
    std::size_t operator()(const LinkPathInfoKey& key) const noexcept {
        const std::size_t left = SolutionLinkHash{}(key.virtual_link);
        const std::size_t right = SolutionLinkHash{}(key.physical_link);
        return left ^ (right + std::size_t{0x9e3779b9U} +
                       (left << 6U) + (left >> 2U));
    }
};

template <typename Tag>
struct SolutionEntryId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    friend bool operator==(
        SolutionEntryId left,
        SolutionEntryId right) noexcept {
        return left.value == right.value;
    }
};

// Ordered compatibility maps retain Python insertion order. Numeric keys are
// resolved once to an object-local compact ID when repeated access is needed.
// Any structural erase/clear invalidates previously returned IDs.
template <
    typename Key,
    typename Value,
    typename Tag,
    typename Hash = std::hash<Key>>
class SolutionOrderedMap {
public:
    using EntryId = SolutionEntryId<Tag>;

    struct Entry {
        Key key;
        Value value;

        friend bool operator==(
            const Entry& left,
            const Entry& right) {
            return left.key == right.key && left.value == right.value;
        }
    };

    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }

    // Reserve both the ordered storage and its numeric index before a known
    // batch is populated. Solution maps remain dynamic compatibility fields,
    // but their cardinality is known at the solver boundary; reserving here
    // avoids repeated growth/rehash in candidate hot paths.
    void reserve(std::size_t count) {
        entries_.reserve(count);
        index_.reserve(count);
    }

    const std::vector<Entry>& entries() const noexcept { return entries_; }

    std::optional<EntryId> find_id(const Key& key) const {
        const auto found = index_.find(key);
        if (found == index_.end()) {
            return std::nullopt;
        }
        return EntryId{found->second};
    }

    bool contains(const Key& key) const { return index_.find(key) != index_.end(); }

    EntryId insert_or_assign(Key key, Value value) {
        const auto found = index_.find(key);
        if (found != index_.end()) {
            entries_[found->second].value = std::move(value);
            return EntryId{found->second};
        }
        if (entries_.size() >=
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::length_error("solution ordered map ID space exhausted");
        }
        entries_.push_back(Entry{std::move(key), std::move(value)});
        const auto id = static_cast<std::uint32_t>(entries_.size() - 1U);
        try {
            index_.emplace(entries_.back().key, id);
        } catch (...) {
            entries_.pop_back();
            throw;
        }
        return EntryId{id};
    }

    bool erase(const Key& key) {
        const auto found = index_.find(key);
        if (found == index_.end()) {
            return false;
        }
        const std::size_t erased = found->second;
        index_.erase(found);
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(erased));
        for (std::size_t index = erased; index < entries_.size(); ++index) {
            index_.at(entries_[index].key) = static_cast<std::uint32_t>(index);
        }
        return true;
    }

    void clear() noexcept {
        entries_.clear();
        index_.clear();
    }

    Value& at(EntryId id) { return checked_entry(id).value; }
    const Value& at(EntryId id) const { return checked_entry(id).value; }
    const Key& key(EntryId id) const { return checked_entry(id).key; }

    friend bool operator==(
        const SolutionOrderedMap& left,
        const SolutionOrderedMap& right) {
        return left.entries_ == right.entries_;
    }

private:
    Entry& checked_entry(EntryId id) {
        if (static_cast<std::size_t>(id.value) >= entries_.size()) {
            throw std::out_of_range("invalid solution entry ID");
        }
        return entries_[id.value];
    }

    const Entry& checked_entry(EntryId id) const {
        if (static_cast<std::size_t>(id.value) >= entries_.size()) {
            throw std::out_of_range("invalid solution entry ID");
        }
        return entries_[id.value];
    }

    std::vector<Entry> entries_;
    std::unordered_map<Key, std::uint32_t, Hash> index_;
};

class SolutionAttributeValues {
public:
    using AttributeNumber = network::attribute::AttributeNumber;
    using AttributeId = network::attribute::AttributeRegistryId;

    std::size_t slot_count() const noexcept { return values_.size(); }
    bool empty() const noexcept { return values_.empty(); }

    void set(AttributeId id, AttributeNumber value) {
        if (id >= values_.size()) {
            values_.resize(id + 1U);
        }
        values_[id] = std::move(value);
    }

    AttributeNumber* find(AttributeId id) noexcept {
        if (id >= values_.size() || !values_[id].has_value()) {
            return nullptr;
        }
        return &*values_[id];
    }

    const AttributeNumber* find(AttributeId id) const noexcept {
        if (id >= values_.size() || !values_[id].has_value()) {
            return nullptr;
        }
        return &*values_[id];
    }

    AttributeNumber& at(AttributeId id) {
        auto* value = find(id);
        if (value == nullptr) {
            throw std::out_of_range("missing solution attribute value");
        }
        return *value;
    }

    const AttributeNumber& at(AttributeId id) const {
        const auto* value = find(id);
        if (value == nullptr) {
            throw std::out_of_range("missing solution attribute value");
        }
        return *value;
    }

    void clear() noexcept { values_.clear(); }

    const std::vector<std::optional<AttributeNumber>>& slots() const noexcept {
        return values_;
    }

    friend bool operator==(
        const SolutionAttributeValues& left,
        const SolutionAttributeValues& right) {
        return left.values_ == right.values_;
    }

private:
    std::vector<std::optional<AttributeNumber>> values_;
};

struct NodeSlotsTag;
struct LinkPathsTag;
struct NodeSlotsInfoTag;
struct LinkPathsInfoTag;
struct NodeConstraintOffsetsTag;
struct LinkConstraintOffsetsTag;
struct PathConstraintOffsetsTag;
struct NodeConstraintViolationsTag;
struct LinkConstraintViolationsTag;
struct PathConstraintViolationsTag;

using NodeSlots = SolutionOrderedMap<
    SolutionNodeId, SolutionNodeId, NodeSlotsTag>;
using LinkPaths = SolutionOrderedMap<
    SolutionLink,
    std::vector<SolutionLink>,
    LinkPathsTag,
    SolutionLinkHash>;
using NodeSlotsInfo = SolutionOrderedMap<
    NodeSlotInfoKey,
    SolutionAttributeValues,
    NodeSlotsInfoTag,
    NodeSlotInfoKeyHash>;
using LinkPathsInfo = SolutionOrderedMap<
    LinkPathInfoKey,
    SolutionAttributeValues,
    LinkPathsInfoTag,
    LinkPathInfoKeyHash>;

using NodeConstraintTable = SolutionOrderedMap<
    SolutionNodeId,
    SolutionAttributeValues,
    NodeConstraintOffsetsTag>;
using LinkConstraintTable = SolutionOrderedMap<
    SolutionLink,
    SolutionAttributeValues,
    LinkConstraintOffsetsTag,
    SolutionLinkHash>;
using PathConstraintTable = SolutionOrderedMap<
    SolutionLink,
    SolutionAttributeValues,
    PathConstraintOffsetsTag,
    SolutionLinkHash>;
using NodeViolationTable = SolutionOrderedMap<
    SolutionNodeId,
    SolutionAttributeValues,
    NodeConstraintViolationsTag>;
using LinkViolationTable = SolutionOrderedMap<
    SolutionLink,
    SolutionAttributeValues,
    LinkConstraintViolationsTag,
    SolutionLinkHash>;
using PathViolationTable = SolutionOrderedMap<
    SolutionLink,
    SolutionAttributeValues,
    PathConstraintViolationsTag,
    SolutionLinkHash>;

struct SolutionStepConstraintValues {
    SolutionAttributeValues node_level;
    SolutionAttributeValues link_level;
    SolutionAttributeValues path_level;

    void clear() noexcept {
        node_level.clear();
        link_level.clear();
        path_level.clear();
    }
};

struct SolutionConstraintOffsets {
    NodeConstraintTable node_level;
    LinkConstraintTable link_level;
    PathConstraintTable path_level;

    void clear() noexcept {
        node_level.clear();
        link_level.clear();
        path_level.clear();
    }
};

struct SolutionConstraintViolations {
    NodeViolationTable node_level;
    LinkViolationTable link_level;
    PathViolationTable path_level;

    void clear() noexcept {
        node_level.clear();
        link_level.clear();
        path_level.clear();
    }
};

struct SolutionMetadata {
    std::int64_t v_net_id = 0;
    double v_net_lifetime = 0.0;
    double v_net_arrival_time = 0.0;
    std::size_t v_net_num_nodes = 0U;
    std::size_t v_net_num_edges = 0U;
};

enum class SolutionNetworkField : std::uint8_t {
    id,
    lifetime,
    arrival_time,
};

class SolutionException final : public std::runtime_error {
public:
    SolutionException(SolutionNetworkField field, std::string message);

    SolutionNetworkField field() const noexcept { return field_; }

private:
    SolutionNetworkField field_;
};

class Solution {
public:
    explicit Solution(SolutionMetadata metadata);

    static Solution from_v_net(const network::VirtualNetwork& virtual_network);
    static std::vector<Solution> from_metadata_batch(
        const std::vector<SolutionMetadata>& metadata,
        std::size_t workers = 1U);
    static void reset_batch(
        std::vector<Solution>& solutions,
        std::size_t workers = 1U);

    void reset();
    bool is_feasible() const noexcept;
    std::string repr() const;

    // Immutable-by-reset request facts.
    std::int64_t v_net_id = 0;
    double v_net_lifetime = 0.0;
    double v_net_arrival_time = 0.0;
    std::size_t v_net_num_nodes = 0U;
    std::size_t v_net_num_edges = 0U;

    // Fixed solution state.
    bool result = false;
    NodeSlots node_slots;
    LinkPaths link_paths;
    NodeSlotsInfo node_slots_info;
    LinkPathsInfo link_paths_info;

    double v_net_cost = 0.0;
    double v_net_revenue = 0.0;
    double v_net_demand = 0.0;
    double v_net_node_demand = 0.0;
    double v_net_link_demand = 0.0;
    double v_net_node_revenue = 0.0;
    double v_net_link_revenue = 0.0;
    double v_net_node_cost = 0.0;
    double v_net_link_cost = 0.0;
    double v_net_path_cost = 0.0;
    double v_net_r2c_ratio = 0.0;
    double v_net_time_cost = 0.0;
    double v_net_time_revenue = 0.0;
    double v_net_time_rc_ratio = 0.0;
    std::string description;

    double v_net_total_hard_constraint_violation = 0.0;
    SolutionStepConstraintValues v_net_single_step_constraint_offset;
    SolutionConstraintOffsets v_net_constraint_offsets;
    SolutionConstraintViolations v_net_constraint_violations;
    std::vector<network::attribute::AttributeNumber>
        v_net_single_step_violation_list;
    double v_net_single_step_hard_constraint_offset =
        -std::numeric_limits<double>::infinity();
    double v_net_max_single_step_hard_constraint_violation =
        -std::numeric_limits<double>::infinity();

    bool place_result = true;
    bool route_result = true;
    bool early_rejection = false;
    std::int64_t revoke_times = 0;
    std::vector<std::int64_t> selected_actions;
    std::int64_t num_interactions = 0;
    double v_net_reward = 0.0;

    // These are created later by Counter/Controller in Python and are not
    // removed by reset. Optional direct fields preserve that presence rule.
    std::optional<std::size_t> num_placed_nodes;
    std::optional<std::size_t> num_routed_links;
    std::optional<std::size_t> num_attempt_times;
};

} // namespace virne::core
