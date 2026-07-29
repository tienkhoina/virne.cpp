#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace virne::utils
{

// IDs are object-local and remain stable across insert/overwrite operations
// because ClassDict never removes fields. Copy/move assignment and swap replace
// the schema and therefore invalidate IDs previously cached for that object.
// Resolve a dynamic name once, then carry this compact ID through hot paths
// instead of hashing the string again.
struct ClassFieldId
{
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    friend bool operator==(
        ClassFieldId left,
        ClassFieldId right) noexcept
    {
        return left.value == right.value;
    }

    friend bool operator!=(
        ClassFieldId left,
        ClassFieldId right) noexcept
    {
        return !(left == right);
    }
};

// The leaf Python class accepts Any. std::any keeps that open type boundary.
// These two recursive containers provide explicit deep-copy behavior for the
// list/mapping surface used by Virne solutions and the differential oracle.
using ClassAnyList = std::vector<std::any>;
using ClassAnyListPtr = std::shared_ptr<ClassAnyList>;

struct ClassMappingItem
{
    std::any key;
    std::any value;
};

struct ClassMapping
{
    std::vector<ClassMappingItem> items;
    bool ordered = true;
};

using ClassMappingPtr = std::shared_ptr<ClassMapping>;

struct ClassDictItem
{
    std::string key;
    std::any value;
};

using ClassDictSnapshot = std::vector<ClassDictItem>;

class ClassDict
{
public:
    ClassDict() = default;
    ClassDict(const ClassDict& other);
    ClassDict(ClassDict&& other) noexcept;
    ClassDict& operator=(const ClassDict& other);
    ClassDict& operator=(ClassDict&& other) noexcept;

    std::size_t size() const noexcept
    {
        return fields_.size();
    }

    bool empty() const noexcept
    {
        return fields_.empty();
    }

    // One boundary hash lookup. Missing names are represented by nullopt.
    std::optional<ClassFieldId> find_field_id(
        std::string_view key) const;

    // One boundary hash operation. A new field starts with Python None
    // (empty std::any); an existing field retains its value.
    ClassFieldId resolve_or_create(
        std::string_view key);

    // One boundary hash operation, returning the stable ID for subsequent
    // direct accesses.
    ClassFieldId set(
        std::string_view key,
        std::any value);

    template <typename T>
    ClassFieldId set_value(
        std::string_view key,
        T&& value)
    {
        return set(
            key,
            std::any(std::forward<T>(value)));
    }

    // ID-only accessors: no hashing, allocation, interning, or string compare.
    void set(
        ClassFieldId id,
        std::any value);

    std::any& at(ClassFieldId id);
    const std::any& at(ClassFieldId id) const;

    std::string_view field_name(ClassFieldId id) const;

    template <typename T>
    T& at_as(ClassFieldId id)
    {
        return std::any_cast<T&>(at(id));
    }

    template <typename T>
    const T& at_as(ClassFieldId id) const
    {
        return std::any_cast<const T&>(at(id));
    }

    // Python integer indexing, including negative indices, in insertion order.
    std::any& at_index(std::int64_t index);
    const std::any& at_index(std::int64_t index) const;

    // String-style Python lookup: missing returns nullptr (Python None).
    // Explicit None is distinguishable because the returned pointer is
    // non-null but points to an empty std::any.
    std::any* find(std::string_view key);
    const std::any* find(std::string_view key) const;

    // Python get(key, default): returns the stored object by identity or a
    // caller-owned default by identity. The caller must keep default_value
    // alive while using a returned reference to it.
    std::any& get_or(
        std::string_view key,
        std::any& default_value);

    const std::any& get_or(
        std::string_view key,
        const std::any& default_value) const;

    // Prevent a caller from keeping a reference to an already-destroyed
    // temporary default.  Lvalue defaults retain Python's identity behavior.
    const std::any& get_or(
        std::string_view key,
        const std::any&& default_value) const = delete;

    void update(const ClassDictSnapshot& values);
    void update(const ClassDict& values);

    static ClassDict from_dict(
        const ClassDictSnapshot& values);

    // Deep-copies the supported recursive value surface. The four historical
    // ordered-mapping fields are emitted as plain mappings, matching Python.
    ClassDictSnapshot to_dict() const;

    // Deterministic C++ extensions for independent objects. Output and error
    // order follow input order; zero selects the measured automatic width.
    static std::vector<ClassDict> from_dict_batch(
        const std::vector<ClassDictSnapshot>& values,
        std::size_t worker_count = 0);

    static std::vector<ClassDictSnapshot> to_dict_batch(
        const std::vector<ClassDict>& values,
        std::size_t worker_count = 0);

    void swap(ClassDict& other) noexcept;

private:
    struct Field
    {
        std::string key;
        std::any value;
        bool plain_mapping_on_snapshot = false;
    };

    using FieldIndex = std::unordered_map<
        std::string_view,
        ClassFieldId,
        std::hash<std::string_view>,
        std::equal_to<std::string_view>>;

    static bool is_plain_mapping_snapshot_field(
        std::string_view key) noexcept;

    std::size_t checked_index(ClassFieldId id) const;
    std::size_t checked_integer_index(std::int64_t index) const;
    void rebuild_index();

    std::deque<Field> fields_;
    FieldIndex field_index_;
};

void swap(ClassDict& left, ClassDict& right) noexcept;

} // namespace virne::utils
