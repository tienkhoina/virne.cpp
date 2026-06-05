#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

using AttrValue =
    std::variant<
        int64_t,
        double,
        bool,
        std::string>;

using AttrId = uint32_t;

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

    std::vector<std::string>
        id_to_name_;
};

class AttrMap
{
public:

    AttrMap() = default;

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
        ensure_slot(id);

        auto& slot =
            slots_[id];

        if (!slot.has_value())
        {
            throw std::out_of_range(
                "Attribute value not found");
        }

        return *slot;
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

    std::vector<
        std::optional<
            AttrValue>>&
    slots() noexcept
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
        if (id >= slots_.size())
        {
            slots_.resize(
                id + 1);
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