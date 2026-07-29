#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace virne::utils
{

// Fixed discriminants are resolved once at the file boundary. No format or
// mode string is carried through parsing, serialization, or batch hot loops.
enum class SettingFormat : std::uint8_t
{
    json,
    yaml,
};

enum class SettingMode : std::uint8_t
{
    read,
    read_update,
    write,
    write_update,
    append,
    append_update,
    exclusive_create,
    exclusive_create_update,
};

enum class SettingValueKind : std::uint8_t
{
    null_value,
    boolean,
    integer,
    real,
    string,
    list,
    object,
};

enum class SettingErrorCode : std::uint8_t
{
    unsupported_format,
    invalid_mode,
    open_failed,
    read_failed,
    write_failed,
    parse_error,
    unsupported_yaml_feature,
    serialization_error,
};

class SettingException : public std::runtime_error
{
public:
    SettingException(SettingErrorCode code, std::string message);

    SettingErrorCode code() const noexcept
    {
        return code_;
    }

private:
    SettingErrorCode code_;
};

// IDs belong to one SettingObject. Insert/overwrite keeps them stable; copy or
// move assignment and swap replace the schema and invalidate cached IDs.
struct SettingKeyId
{
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    friend bool operator==(SettingKeyId left, SettingKeyId right) noexcept
    {
        return left.value == right.value;
    }

    friend bool operator!=(SettingKeyId left, SettingKeyId right) noexcept
    {
        return !(left == right);
    }
};

class SettingValue;
class SettingObject;

class SettingInteger
{
public:
    using BigInteger = boost::multiprecision::cpp_int;

    SettingInteger() noexcept = default;

    template <
        typename Integer,
        typename = std::enable_if_t<
            std::is_integral_v<std::remove_reference_t<Integer>> &&
            !std::is_same_v<
                std::remove_cv_t<std::remove_reference_t<Integer>>,
                bool>>>
    SettingInteger(Integer value) noexcept
    {
        static_assert(
            sizeof(Integer) <= sizeof(std::uint64_t),
            "SettingInteger integral constructor supports up to 64 bits");
        if constexpr (std::is_signed_v<Integer>)
        {
            value_ = static_cast<std::int64_t>(value);
        }
        else
        {
            value_ = static_cast<std::uint64_t>(value);
        }
    }

    SettingInteger(BigInteger value);
    explicit SettingInteger(std::string_view decimal);
    explicit SettingInteger(const char* decimal);

    bool is_big() const noexcept;

    template <typename T>
    T convert_to() const
    {
        return std::visit(
            [](const auto& value) -> T
            {
                using Value =
                    std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, std::string>)
                {
                    if constexpr (
                        std::is_same_v<Value, BigInteger>)
                    {
                        return value.template convert_to<std::string>();
                    }
                    else
                    {
                        return std::to_string(value);
                    }
                }
                else if constexpr (
                    std::is_same_v<Value, BigInteger>)
                {
                    return value.template convert_to<T>();
                }
                else
                {
                    return static_cast<T>(value);
                }
            },
            value_);
    }

    SettingInteger& operator*=(std::uint32_t multiplier);
    SettingInteger& operator+=(std::uint32_t addend);
    SettingInteger& operator+=(const SettingInteger& addend);

    friend SettingInteger operator-(const SettingInteger& value);
    friend bool operator==(
        const SettingInteger& left,
        const SettingInteger& right);
    friend bool operator!=(
        const SettingInteger& left,
        const SettingInteger& right)
    {
        return !(left == right);
    }
    friend bool operator>=(
        const SettingInteger& left,
        std::uint64_t right);

private:
    BigInteger to_big() const;
    void assign_compact(BigInteger value);

    std::variant<std::int64_t, std::uint64_t, BigInteger> value_{
        std::int64_t{0}};
};

using SettingList = std::vector<SettingValue>;
using SettingListPtr = std::shared_ptr<SettingList>;
using SettingObjectPtr = std::shared_ptr<SettingObject>;

class SettingValue
{
public:
    SettingValue() noexcept = default;
    SettingValue(std::nullptr_t) noexcept;
    SettingValue(bool value) noexcept;
    SettingValue(std::int64_t value);
    SettingValue(std::uint64_t value);
    SettingValue(SettingInteger value);
    SettingValue(double value) noexcept;
    SettingValue(std::string value);
    SettingValue(std::string_view value);
    SettingValue(const char* value);
    SettingValue(SettingListPtr value);
    SettingValue(SettingObjectPtr value);

    static SettingValue make_list();
    static SettingValue make_object();

    SettingValueKind kind() const noexcept;

    bool is_null() const noexcept;
    bool as_bool() const;
    const SettingInteger& as_integer() const;
    double as_real() const;
    const std::string& as_string() const;
    SettingList& as_list();
    const SettingList& as_list() const;
    SettingObject& as_object();
    const SettingObject& as_object() const;

    const SettingListPtr& list_ptr() const;
    const SettingObjectPtr& object_ptr() const;

private:
    using Storage = std::variant<
        std::monostate,
        bool,
        SettingInteger,
        double,
        std::string,
        SettingListPtr,
        SettingObjectPtr>;

    Storage value_;
};

class SettingObject
{
public:
    SettingObject() = default;
    SettingObject(const SettingObject& other);
    SettingObject(SettingObject&& other) noexcept;
    SettingObject& operator=(const SettingObject& other);
    SettingObject& operator=(SettingObject&& other) noexcept;

    std::size_t size() const noexcept
    {
        return entries_.size();
    }

    bool empty() const noexcept
    {
        return entries_.empty();
    }

    // Capacity hint only; IDs and logical ordering are unchanged.
    void reserve(std::size_t key_count);

    // Dynamic strings are accepted only at this boundary. Resolve once and
    // retain the compact ID for every repeated/hot access.
    std::optional<SettingKeyId> find_key_id(std::string_view key) const;
    SettingKeyId resolve_or_create(std::string_view key);
    SettingKeyId set(std::string_view key, SettingValue value);

    // Ownership-transfer boundary for parsers that already decoded a key.
    // This avoids allocating/copying the same dynamic key a second time.
    SettingKeyId set_owned(std::string key, SettingValue value);

    // ID access performs no hashing, allocation, or string comparison.
    void set(SettingKeyId id, SettingValue value);
    SettingValue& at(SettingKeyId id);
    const SettingValue& at(SettingKeyId id) const;
    std::string_view key_name(SettingKeyId id) const;

    // Cached lexicographic ID order required by PyYAML's sort_keys=True.
    // The returned view is invalidated by a new key or schema replacement.
    const std::vector<SettingKeyId>& sorted_key_ids() const;

    void swap(SettingObject& other) noexcept;

private:
    struct Entry
    {
        std::string_view key;
        SettingValue value;
    };

    // The index stores only non-owning views into key_storage_. A flat table
    // avoids one node allocation and pointer chase per dynamic key. IDs remain
    // the stable hot-loop representation; strings only cross this boundary.
    using KeyIndex = boost::unordered_flat_map<
        std::string_view,
        SettingKeyId,
        std::hash<std::string_view>,
        std::equal_to<std::string_view>>;

    std::size_t checked_index(SettingKeyId id) const;
    void rebuild_index();
    void rebuild_key_views_and_index();

    // Dynamic key bytes live in stable storage while the ID/value lane stays
    // contiguous for cache-efficient direct access and serialization.
    std::vector<std::string> key_storage_;
    std::vector<Entry> entries_;
    KeyIndex key_index_;
    mutable std::vector<SettingKeyId> sorted_key_ids_;
    mutable std::mutex sorted_key_ids_mutex_;
    mutable std::atomic<bool> sorted_key_ids_dirty_{true};
};

void swap(SettingObject& left, SettingObject& right) noexcept;

struct SettingDocument
{
    // The root is a fixed field. It is intentionally not hidden in a
    // string-keyed property bag.
    SettingValue root;
};

struct ReturnedSettingError
{
    SettingErrorCode code = SettingErrorCode::unsupported_format;
    std::string message;
};

using WriteSettingResult = std::variant<
    std::reference_wrapper<const SettingDocument>,
    ReturnedSettingError>;

std::optional<SettingFormat> setting_format_from_path(
    std::string_view path) noexcept;

SettingMode parse_setting_mode(std::string_view mode);

// In-memory leaf API used by isolated tests and high-throughput callers.
SettingDocument parse_setting(
    std::string_view bytes,
    SettingFormat format);

std::string dump_setting(
    const SettingDocument& document,
    SettingFormat format);

// Compatibility API. File opening happens before suffix dispatch, matching
// Python. Unsupported write suffixes are returned rather than thrown.
SettingDocument read_setting(
    const std::string& path,
    SettingMode mode = SettingMode::read);

WriteSettingResult write_setting(
    const SettingDocument& document,
    const std::string& path,
    SettingMode mode = SettingMode::write);

const SettingDocument& write_setting_strict(
    const SettingDocument& document,
    const std::string& path,
    SettingMode mode = SettingMode::write);

// Historical typo retained as the compatibility surface.
void conver_format(
    const std::string& source_path,
    const std::string& destination_path);

// Deterministic C++ extensions. Results and failures are observed in input
// order. worker_count==0 selects a measured automatic width.
std::vector<SettingDocument> parse_setting_batch(
    const std::vector<std::string>& inputs,
    SettingFormat format,
    std::size_t worker_count = 0);

std::vector<std::string> dump_setting_batch(
    const std::vector<SettingDocument>& documents,
    SettingFormat format,
    std::size_t worker_count = 0);

} // namespace virne::utils
