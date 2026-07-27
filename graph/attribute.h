#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

struct AttrList;
struct AttrObject;

using AttrListPtr =
    std::shared_ptr<AttrList>;

using AttrObjectPtr =
    std::shared_ptr<AttrObject>;

// Scalar values stay inline for the graph hot path. Structured values are
// heap-backed and are intended for graph metadata, GML repeated keys and
// occasional attributes such as a node position. Algorithms should continue
// resolving a scalar attribute to AttrId once before entering a hot loop.
using AttrValue =
    std::variant<
        int64_t,
        double,
        bool,
        std::string,
        AttrListPtr,
        AttrObjectPtr>;

struct AttrList
{
    std::vector<AttrValue> values;
};

struct AttrObject
{
    std::vector<
        std::pair<std::string, AttrValue>>
        entries;

    AttrValue* find(
        std::string_view name)
    {
        for (auto& [key, value] : entries)
        {
            if (key == name)
            {
                return &value;
            }
        }
        return nullptr;
    }

    const AttrValue* find(
        std::string_view name) const
    {
        for (const auto& [key, value] : entries)
        {
            if (key == name)
            {
                return &value;
            }
        }
        return nullptr;
    }

    void set(
        std::string name,
        AttrValue value)
    {
        if (AttrValue* current = find(name))
        {
            *current = std::move(value);
            return;
        }
        entries.emplace_back(
            std::move(name),
            std::move(value));
    }
};

inline AttrValue
make_attr_list(
    std::vector<AttrValue> values = {})
{
    return std::make_shared<AttrList>(
        AttrList{std::move(values)});
}

inline AttrValue
make_attr_object(
    std::vector<
        std::pair<std::string, AttrValue>>
        entries = {})
{
    return std::make_shared<AttrObject>(
        AttrObject{std::move(entries)});
}

inline const AttrList*
attr_list(
    const AttrValue& value) noexcept
{
    const auto* pointer =
        std::get_if<AttrListPtr>(&value);
    return pointer != nullptr && *pointer
        ? pointer->get()
        : nullptr;
}

inline AttrList*
attr_list(
    AttrValue& value) noexcept
{
    auto* pointer =
        std::get_if<AttrListPtr>(&value);
    return pointer != nullptr && *pointer
        ? pointer->get()
        : nullptr;
}

inline const AttrObject*
attr_object(
    const AttrValue& value) noexcept
{
    const auto* pointer =
        std::get_if<AttrObjectPtr>(&value);
    return pointer != nullptr && *pointer
        ? pointer->get()
        : nullptr;
}

inline AttrObject*
attr_object(
    AttrValue& value) noexcept
{
    auto* pointer =
        std::get_if<AttrObjectPtr>(&value);
    return pointer != nullptr && *pointer
        ? pointer->get()
        : nullptr;
}

inline bool
attr_value_equal_impl(
    const AttrValue& lhs,
    const AttrValue& rhs,
    size_t depth)
{
    if (depth > 256)
    {
        throw std::invalid_argument(
            "Attribute metadata is cyclic or exceeds depth 256");
    }
    if (lhs.index() != rhs.index())
    {
        return false;
    }

    if (const auto* value =
            std::get_if<int64_t>(&lhs))
    {
        return *value == std::get<int64_t>(rhs);
    }
    if (const auto* value =
            std::get_if<double>(&lhs))
    {
        return *value == std::get<double>(rhs);
    }
    if (const auto* value =
            std::get_if<bool>(&lhs))
    {
        return *value == std::get<bool>(rhs);
    }
    if (const auto* value =
            std::get_if<std::string>(&lhs))
    {
        return *value == std::get<std::string>(rhs);
    }
    if (std::holds_alternative<AttrListPtr>(lhs))
    {
        const AttrList* left = attr_list(lhs);
        const AttrList* right = attr_list(rhs);
        if (left == nullptr || right == nullptr)
        {
            return left == right;
        }
        if (right == nullptr ||
            left->values.size() != right->values.size())
        {
            return false;
        }
        for (size_t index = 0;
             index < left->values.size();
             ++index)
        {
            if (!attr_value_equal_impl(
                    left->values[index],
                    right->values[index],
                    depth + 1))
            {
                return false;
            }
        }
        return true;
    }
    if (std::holds_alternative<AttrObjectPtr>(lhs))
    {
        const AttrObject* left = attr_object(lhs);
        const AttrObject* right = attr_object(rhs);
        if (left == nullptr || right == nullptr)
        {
            return left == right;
        }
        if (right == nullptr ||
            left->entries.size() != right->entries.size())
        {
            return false;
        }
        for (size_t index = 0;
             index < left->entries.size();
             ++index)
        {
            if (left->entries[index].first !=
                    right->entries[index].first ||
                !attr_value_equal_impl(
                    left->entries[index].second,
                    right->entries[index].second,
                    depth + 1))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

inline bool
attr_value_equal(
    const AttrValue& lhs,
    const AttrValue& rhs)
{
    return attr_value_equal_impl(lhs, rhs, 0);
}

// AttrValue uses shared pointers only to keep scalar attributes compact and
// cheap in hot loops.  Copying a graph must still have ordinary value
// semantics, so structured metadata is cloned recursively at copy boundaries.
inline AttrValue
clone_attr_value_impl(
    const AttrValue& value,
    size_t depth)
{
    if (depth > 256)
    {
        throw std::invalid_argument(
            "Attribute metadata is cyclic or exceeds depth 256");
    }
    if (const AttrList* list = attr_list(value))
    {
        std::vector<AttrValue> copy;
        copy.reserve(list->values.size());
        for (const AttrValue& item : list->values)
        {
            copy.push_back(
                clone_attr_value_impl(item, depth + 1));
        }
        return make_attr_list(std::move(copy));
    }

    if (std::holds_alternative<AttrListPtr>(value))
    {
        return AttrListPtr{};
    }

    if (const AttrObject* object = attr_object(value))
    {
        std::vector<std::pair<std::string, AttrValue>> copy;
        copy.reserve(object->entries.size());
        for (const auto& [name, item] : object->entries)
        {
            copy.emplace_back(
                name,
                clone_attr_value_impl(item, depth + 1));
        }
        return make_attr_object(std::move(copy));
    }

    if (std::holds_alternative<AttrObjectPtr>(value))
    {
        return AttrObjectPtr{};
    }

    return value;
}

inline AttrValue
clone_attr_value(
    const AttrValue& value)
{
    return clone_attr_value_impl(value, 0);
}

using AttrId = uint32_t;

inline double
attr_to_double(
    const AttrValue& value)
{
    if (const auto* p =
            std::get_if<int64_t>(
                &value))
    {
        return static_cast<double>(
            *p);
    }

    if (const auto* p =
            std::get_if<double>(
                &value))
    {
        return *p;
    }

    if (const auto* p =
            std::get_if<bool>(
                &value))
    {
        return *p
            ? 1.0
            : 0.0;
    }

    throw std::runtime_error(
        "Attribute is not numeric");
}

struct TransparentStringHash
{
    using is_transparent = void;

    size_t operator()(
        std::string_view s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }

    size_t operator()(
        const std::string& s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }

    size_t operator()(
        const char* s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }
};

struct TransparentStringEq
{
    using is_transparent = void;

    bool operator()(
        std::string_view a,
        std::string_view b) const noexcept
    {
        return a == b;
    }
};

class AttributeRegistry
{
public:

    AttrId intern(
        std::string_view name)
    {
        auto it =
            name_to_id_.find(
                std::string(name));

        if (it != name_to_id_.end())
        {
            return it->second;
        }

        if (id_to_name_.size() >=
            static_cast<size_t>(
                std::numeric_limits<AttrId>::max()))
        {
            throw std::overflow_error(
                "Attribute ID space is exhausted");
        }

        const AttrId id =
            static_cast<AttrId>(
                id_to_name_.size());

        id_to_name_.emplace_back(
            name);

        name_to_id_.emplace(
            id_to_name_.back(),
            id);

        return id;
    }

    std::optional<AttrId> find(
        std::string_view name) const
    {
        auto it =
            name_to_id_.find(
                std::string(name));

        if (it == name_to_id_.end())
        {
            return std::nullopt;
        }

        return it->second;
    }

    std::string_view name(
        AttrId id) const
    {
        if (id >= id_to_name_.size())
        {
            throw std::out_of_range(
                "Attribute id not found");
        }

        return id_to_name_[id];
    }

    size_t size() const noexcept
    {
        return id_to_name_.size();
    }

private:

    std::unordered_map<
        std::string,
        AttrId,
        TransparentStringHash,
        TransparentStringEq>
        name_to_id_;

    // Public attr_name() returns string_view. deque keeps existing string
    // objects at stable addresses when later names are interned.
    std::deque<std::string>
        id_to_name_;
};

class AttrMap
{
public:

    AttrMap() = default;

    AttrMap(
        const AttrMap& other)
        :
        registry_(other.registry_),
        active_ids_(other.active_ids_)
    {
        slots_.reserve(other.slots_.size());
        for (const auto& slot : other.slots_)
        {
            if (slot.has_value())
            {
                slots_.emplace_back(
                    clone_attr_value(*slot));
            }
            else
            {
                slots_.emplace_back(std::nullopt);
            }
        }
    }

    AttrMap& operator=(
        const AttrMap& other)
    {
        if (this == &other)
        {
            return *this;
        }

        registry_ = other.registry_;
        active_ids_ = other.active_ids_;
        slots_.clear();
        slots_.reserve(other.slots_.size());
        for (const auto& slot : other.slots_)
        {
            if (slot.has_value())
            {
                slots_.emplace_back(
                    clone_attr_value(*slot));
            }
            else
            {
                slots_.emplace_back(std::nullopt);
            }
        }
        return *this;
    }

    AttrMap(AttrMap&&) noexcept = default;

    AttrMap& operator=(AttrMap&&) noexcept = default;

    explicit AttrMap(
        std::shared_ptr<
            AttributeRegistry> registry)
        :
        registry_(
            std::move(
                registry))
    {
    }

    void bind(
        std::shared_ptr<
            AttributeRegistry> registry)
    {
        registry_ =
            std::move(
                registry);
    }

    bool bound() const noexcept
    {
        return static_cast<bool>(
            registry_);
    }

    AttrValue& operator[](
        std::string_view name)
    {
        ensure_bound();

        const AttrId id =
            registry_->intern(
                name);

        ensure_slot(id);

        auto& slot =
            slots_[id];

        if (!slot.has_value())
        {
            slot.emplace(
                int64_t{0});

            active_ids_.push_back(
                id);
        }

        return *slot;
    }

    AttrValue& at(
        std::string_view name)
    {
        const AttrId id =
            resolve_id_or_throw(
                name);

        return at(id);
    }

    const AttrValue& at(
        std::string_view name) const
    {
        const AttrId id =
            resolve_id_or_throw(
                name);

        return at(id);
    }

    AttrValue* find(
        std::string_view name)
    {
        const auto id =
            resolve_id(
                name);

        if (!id.has_value())
        {
            return nullptr;
        }

        return find(
            *id);
    }

    const AttrValue* find(
        std::string_view name) const
    {
        const auto id =
            resolve_id(
                name);

        if (!id.has_value())
        {
            return nullptr;
        }

        return find(
            *id);
    }

    bool contains(
        std::string_view name) const
    {
        return
            find(name)
            !=
            nullptr;
    }

    AttrValue& at(
        AttrId id)
    {
        if (id >= slots_.size()
            ||
            !slots_[id].has_value())
        {
            throw std::out_of_range(
                "Attribute value not found");
        }

        return *slots_[id];
    }

    const AttrValue& at(
        AttrId id) const
    {
        if (id >= slots_.size()
            ||
            !slots_[id].has_value())
        {
            throw std::out_of_range(
                "Attribute value not found");
        }

        return *slots_[id];
    }

    AttrValue* find(
        AttrId id)
    {
        if (id >= slots_.size()
            ||
            !slots_[id].has_value())
        {
            return nullptr;
        }

        return &*slots_[id];
    }

    const AttrValue* find(
        AttrId id) const
    {
        if (id >= slots_.size()
            ||
            !slots_[id].has_value())
        {
            return nullptr;
        }

        return &*slots_[id];
    }

    bool contains(
        AttrId id) const
    {
        return
            find(id)
            !=
            nullptr;
    }

    // Small dict-like facade used by code ported from NetworkX.  The hot
    // path should still resolve an AttrId once and use find()/set(); these
    // helpers intentionally return value copies and are for boundary code.
    std::optional<AttrValue> get(
        std::string_view name) const
    {
        const AttrValue* value = find(name);
        return value == nullptr
            ? std::nullopt
            : std::optional<AttrValue>{clone_attr_value(*value)};
    }

    std::optional<AttrValue> get(
        AttrId id) const
    {
        const AttrValue* value = find(id);
        return value == nullptr
            ? std::nullopt
            : std::optional<AttrValue>{clone_attr_value(*value)};
    }

    AttrValue get(
        std::string_view name,
        AttrValue fallback) const
    {
        const AttrValue* value = find(name);
        return value == nullptr
            ? fallback
            : clone_attr_value(*value);
    }

    AttrValue get(
        AttrId id,
        AttrValue fallback) const
    {
        const AttrValue* value = find(id);
        return value == nullptr
            ? fallback
            : clone_attr_value(*value);
    }

    std::vector<AttrId> keys() const
    {
        return active_ids_;
    }

    std::vector<std::pair<AttrId, AttrValue>> items() const
    {
        std::vector<std::pair<AttrId, AttrValue>> result;
        result.reserve(active_ids_.size());
        for (const AttrId id : active_ids_)
        {
            result.emplace_back(id, clone_attr_value(at(id)));
        }
        return result;
    }

    void update(const AttrMap& other)
    {
        if (registry_ != other.registry_)
        {
            ensure_bound();
            other.ensure_bound();
            for (const AttrId id : other.attribute_ids())
            {
                (*this)[other.registry_->name(id)] =
                    clone_attr_value(other.at(id));
            }
            return;
        }

        for (const AttrId id : other.attribute_ids())
        {
            set(id, clone_attr_value(other.at(id)));
        }
    }

    void update(const AttrObject& other)
    {
        for (const auto& [name, value] : other.entries)
        {
            (*this)[name] = clone_attr_value(value);
        }
    }

    void set(
        AttrId id,
        AttrValue value)
    {
        ensure_slot(id);

        if (!slots_[id].has_value())
        {
            active_ids_.push_back(
                id);
        }

        slots_[id] =
            std::move(
                value);
    }

    bool erase(
        AttrId id)
    {
        if (id >= slots_.size() ||
            !slots_[id].has_value())
        {
            return false;
        }

        slots_[id].reset();
        const auto active =
            std::find(
                active_ids_.begin(),
                active_ids_.end(),
                id);
        if (active != active_ids_.end())
        {
            active_ids_.erase(active);
        }
        return true;
    }

    bool erase(
        std::string_view name)
    {
        const auto id = resolve_id(name);
        return id.has_value() && erase(*id);
    }

    size_t size() const noexcept
    {
        return active_ids_.size();
    }

    bool empty() const noexcept
    {
        return active_ids_.empty();
    }

    void clear()
    {
        slots_.clear();
        active_ids_.clear();
    }

    std::shared_ptr<
        AttributeRegistry>
    registry() const noexcept
    {
        return registry_;
    }

    const std::vector<
        std::optional<
            AttrValue>>&
    slots() const noexcept
    {
        return slots_;
    }

    const std::vector<
        AttrId>&
    attribute_ids() const noexcept
    {
        return active_ids_;
    }

private:

    std::shared_ptr<
        AttributeRegistry>
        registry_;

    std::vector<
        std::optional<
            AttrValue>>
        slots_;

    std::vector<
        AttrId>
        active_ids_;

    void ensure_bound() const
    {
        if (!registry_)
        {
            throw std::runtime_error(
                "AttrMap is not bound to an AttributeRegistry");
        }
    }

    void ensure_slot(
        AttrId id)
    {
        ensure_bound();
        const size_t index =
            static_cast<size_t>(id);
        if (index >= registry_->size())
        {
            throw std::out_of_range(
                "Attribute id is outside the bound registry");
        }

        if (index >= slots_.size())
        {
            slots_.resize(
                index + size_t{1});
        }
    }

    std::optional<AttrId>
    resolve_id(
        std::string_view name) const
    {
        ensure_bound();

        return registry_->find(
            name);
    }

    AttrId resolve_id_or_throw(
        std::string_view name) const
    {
        auto id =
            resolve_id(
                name);

        if (!id.has_value())
        {
            throw std::out_of_range(
                "Attribute not found");
        }

        return *id;
    }
};
