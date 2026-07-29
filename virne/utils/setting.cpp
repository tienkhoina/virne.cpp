#include "setting.h"

#include <boost/json/basic_parser_impl.hpp>
#include <boost/json/src.hpp>
#include <boost/system/error_code.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace virne::utils
{

SettingInteger::SettingInteger(BigInteger value)
{
    assign_compact(std::move(value));
}

SettingInteger::SettingInteger(std::string_view decimal)
{
    if (decimal.empty())
    {
        throw std::invalid_argument("empty SettingInteger decimal");
    }
    bool negative = false;
    std::size_t index = 0;
    if (decimal.front() == '+' || decimal.front() == '-')
    {
        negative = decimal.front() == '-';
        index = 1;
    }
    if (index == decimal.size())
    {
        throw std::invalid_argument(
            "SettingInteger decimal has no digits");
    }
    SettingInteger result{std::uint64_t{0}};
    for (; index < decimal.size(); ++index)
    {
        const char digit = decimal[index];
        if (digit < '0' || digit > '9')
        {
            throw std::invalid_argument(
                "invalid SettingInteger decimal");
        }
        result *= 10U;
        result += static_cast<std::uint32_t>(digit - '0');
    }
    *this = negative ? -result : std::move(result);
}

SettingInteger::SettingInteger(const char* decimal)
    : SettingInteger(std::string_view(
          decimal == nullptr ? "" : decimal))
{
}

bool SettingInteger::is_big() const noexcept
{
    return std::holds_alternative<BigInteger>(value_);
}

SettingInteger::BigInteger SettingInteger::to_big() const
{
    return std::visit(
        [](const auto& value) -> BigInteger
        {
            return BigInteger(value);
        },
        value_);
}

void SettingInteger::assign_compact(BigInteger value)
{
    static const BigInteger signed_minimum =
        std::numeric_limits<std::int64_t>::min();
    static const BigInteger signed_maximum =
        std::numeric_limits<std::int64_t>::max();
    static const BigInteger unsigned_maximum =
        std::numeric_limits<std::uint64_t>::max();
    if (value >= signed_minimum && value <= signed_maximum)
    {
        value_ = value.convert_to<std::int64_t>();
    }
    else if (value >= 0 && value <= unsigned_maximum)
    {
        value_ = value.convert_to<std::uint64_t>();
    }
    else
    {
        value_ = std::move(value);
    }
}

SettingInteger& SettingInteger::operator*=(
    std::uint32_t multiplier)
{
    if (auto* signed_value = std::get_if<std::int64_t>(&value_))
    {
        if (*signed_value >= 0 &&
            (multiplier == 0U ||
             static_cast<std::uint64_t>(*signed_value) <=
                 static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max()) /
                     multiplier))
        {
            *signed_value *= static_cast<std::int64_t>(multiplier);
            return *this;
        }
    }
    else if (auto* unsigned_value =
                 std::get_if<std::uint64_t>(&value_))
    {
        if (multiplier == 0U ||
            *unsigned_value <=
                std::numeric_limits<std::uint64_t>::max() /
                    multiplier)
        {
            *unsigned_value *= multiplier;
            return *this;
        }
    }
    BigInteger result = to_big();
    result *= multiplier;
    assign_compact(std::move(result));
    return *this;
}

SettingInteger& SettingInteger::operator+=(
    std::uint32_t addend)
{
    if (auto* signed_value = std::get_if<std::int64_t>(&value_))
    {
        if (*signed_value >= 0 &&
            *signed_value <=
                std::numeric_limits<std::int64_t>::max() -
                    static_cast<std::int64_t>(addend))
        {
            *signed_value += static_cast<std::int64_t>(addend);
            return *this;
        }
    }
    else if (auto* unsigned_value =
                 std::get_if<std::uint64_t>(&value_))
    {
        if (*unsigned_value <=
            std::numeric_limits<std::uint64_t>::max() - addend)
        {
            *unsigned_value += addend;
            return *this;
        }
    }
    BigInteger result = to_big();
    result += addend;
    assign_compact(std::move(result));
    return *this;
}

SettingInteger& SettingInteger::operator+=(
    const SettingInteger& addend)
{
    BigInteger result = to_big();
    result += addend.to_big();
    assign_compact(std::move(result));
    return *this;
}

SettingInteger operator-(const SettingInteger& value)
{
    if (const auto* signed_value =
            std::get_if<std::int64_t>(&value.value_))
    {
        if (*signed_value !=
            std::numeric_limits<std::int64_t>::min())
        {
            return SettingInteger(-*signed_value);
        }
        return SettingInteger(
            std::uint64_t{1} << 63U);
    }
    if (const auto* unsigned_value =
            std::get_if<std::uint64_t>(&value.value_))
    {
        const std::uint64_t signed_limit =
            std::uint64_t{1} << 63U;
        if (*unsigned_value < signed_limit)
        {
            return SettingInteger(
                -static_cast<std::int64_t>(*unsigned_value));
        }
        if (*unsigned_value == signed_limit)
        {
            return SettingInteger(
                std::numeric_limits<std::int64_t>::min());
        }
    }
    SettingInteger::BigInteger result = value.to_big();
    result = -result;
    return SettingInteger(std::move(result));
}

bool operator==(
    const SettingInteger& left,
    const SettingInteger& right)
{
    if (left.value_.index() == right.value_.index())
    {
        return left.value_ == right.value_;
    }
    return left.to_big() == right.to_big();
}

bool operator>=(
    const SettingInteger& left,
    std::uint64_t right)
{
    if (const auto* signed_value =
            std::get_if<std::int64_t>(&left.value_))
    {
        return *signed_value >= 0 &&
               static_cast<std::uint64_t>(*signed_value) >= right;
    }
    if (const auto* unsigned_value =
            std::get_if<std::uint64_t>(&left.value_))
    {
        return *unsigned_value >= right;
    }
    return std::get<SettingInteger::BigInteger>(left.value_) >= right;
}

SettingException::SettingException(
    SettingErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code)
{
}

SettingValue::SettingValue(std::nullptr_t) noexcept
{
}

SettingValue::SettingValue(bool value) noexcept
    : value_(value)
{
}

SettingValue::SettingValue(std::int64_t value)
    : value_(SettingInteger(value))
{
}

SettingValue::SettingValue(std::uint64_t value)
    : value_(SettingInteger(value))
{
}

SettingValue::SettingValue(SettingInteger value)
    : value_(std::move(value))
{
}

SettingValue::SettingValue(double value) noexcept
    : value_(value)
{
}

SettingValue::SettingValue(std::string value)
    : value_(std::move(value))
{
}

SettingValue::SettingValue(std::string_view value)
    : value_(std::string(value))
{
}

SettingValue::SettingValue(const char* value)
    : value_(std::string(value == nullptr ? "" : value))
{
}

SettingValue::SettingValue(SettingListPtr value)
    : value_(std::move(value))
{
    if (!std::get<SettingListPtr>(value_))
    {
        throw std::invalid_argument("SettingValue list pointer is null");
    }
}

SettingValue::SettingValue(SettingObjectPtr value)
    : value_(std::move(value))
{
    if (!std::get<SettingObjectPtr>(value_))
    {
        throw std::invalid_argument("SettingValue object pointer is null");
    }
}

SettingValue SettingValue::make_list()
{
    return SettingValue(std::make_shared<SettingList>());
}

SettingValue SettingValue::make_object()
{
    return SettingValue(std::make_shared<SettingObject>());
}

SettingValueKind SettingValue::kind() const noexcept
{
    return static_cast<SettingValueKind>(value_.index());
}

bool SettingValue::is_null() const noexcept
{
    return std::holds_alternative<std::monostate>(value_);
}

bool SettingValue::as_bool() const
{
    return std::get<bool>(value_);
}

const SettingInteger& SettingValue::as_integer() const
{
    return std::get<SettingInteger>(value_);
}

double SettingValue::as_real() const
{
    return std::get<double>(value_);
}

const std::string& SettingValue::as_string() const
{
    return std::get<std::string>(value_);
}

SettingList& SettingValue::as_list()
{
    return *std::get<SettingListPtr>(value_);
}

const SettingList& SettingValue::as_list() const
{
    return *std::get<SettingListPtr>(value_);
}

SettingObject& SettingValue::as_object()
{
    return *std::get<SettingObjectPtr>(value_);
}

const SettingObject& SettingValue::as_object() const
{
    return *std::get<SettingObjectPtr>(value_);
}

const SettingListPtr& SettingValue::list_ptr() const
{
    return std::get<SettingListPtr>(value_);
}

const SettingObjectPtr& SettingValue::object_ptr() const
{
    return std::get<SettingObjectPtr>(value_);
}

SettingObject::SettingObject(const SettingObject& other)
{
    key_storage_.reserve(other.key_storage_.size());
    entries_.reserve(other.entries_.size());
    for (const Entry& entry : other.entries_)
    {
        key_storage_.emplace_back(entry.key);
        entries_.push_back(
            {std::string_view(key_storage_.back()), entry.value});
    }
    rebuild_index();
    std::lock_guard<std::mutex> lock(other.sorted_key_ids_mutex_);
    sorted_key_ids_ = other.sorted_key_ids_;
    sorted_key_ids_dirty_.store(
        other.sorted_key_ids_dirty_.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
}

SettingObject::SettingObject(SettingObject&& other) noexcept
    : key_storage_(std::move(other.key_storage_)),
      entries_(std::move(other.entries_)),
      key_index_(std::move(other.key_index_)),
      sorted_key_ids_(std::move(other.sorted_key_ids_)),
      sorted_key_ids_dirty_(
          other.sorted_key_ids_dirty_.load(std::memory_order_relaxed))
{
    other.key_storage_.clear();
    other.entries_.clear();
    other.key_index_.clear();
    other.sorted_key_ids_.clear();
    other.sorted_key_ids_dirty_.store(true, std::memory_order_relaxed);
}

SettingObject& SettingObject::operator=(const SettingObject& other)
{
    if (this != &other)
    {
        SettingObject copy(other);
        swap(copy);
    }
    return *this;
}

SettingObject& SettingObject::operator=(SettingObject&& other) noexcept
{
    if (this != &other)
    {
        SettingObject moved(std::move(other));
        swap(moved);
    }
    return *this;
}

void SettingObject::reserve(std::size_t key_count)
{
    const std::string* const previous_keys = key_storage_.data();
    key_storage_.reserve(key_count);
    if (key_storage_.data() != previous_keys && !entries_.empty())
    {
        rebuild_key_views_and_index();
    }
    entries_.reserve(key_count);
    key_index_.reserve(key_count);
}

std::optional<SettingKeyId> SettingObject::find_key_id(
    std::string_view key) const
{
    const auto found = key_index_.find(key);
    if (found == key_index_.end())
    {
        return std::nullopt;
    }
    return found->second;
}

SettingKeyId SettingObject::resolve_or_create(std::string_view key)
{
    if (entries_.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::overflow_error("SettingObject key ID space exhausted");
    }

    const SettingKeyId candidate{
        static_cast<std::uint32_t>(entries_.size())};
    const std::string* const previous_keys = key_storage_.data();
    key_storage_.emplace_back(key);
    try
    {
        if (key_storage_.data() != previous_keys && !entries_.empty())
        {
            rebuild_key_views_and_index();
        }
        entries_.push_back(
            {std::string_view(key_storage_.back()), SettingValue{}});
        const auto inserted = key_index_.emplace(
            entries_.back().key, candidate);
        if (inserted.second)
        {
            sorted_key_ids_dirty_.store(
                true, std::memory_order_release);
            return candidate;
        }
        entries_.pop_back();
        key_storage_.pop_back();
        return inserted.first->second;
    }
    catch (...)
    {
        if (entries_.size() > candidate.value)
        {
            entries_.pop_back();
        }
        key_storage_.pop_back();
        throw;
    }
}

SettingKeyId SettingObject::set(
    std::string_view key,
    SettingValue value)
{
    return set_owned(std::string(key), std::move(value));
}

SettingKeyId SettingObject::set_owned(
    std::string key,
    SettingValue value)
{
    if (entries_.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::overflow_error("SettingObject key ID space exhausted");
    }

    const SettingKeyId candidate{
        static_cast<std::uint32_t>(entries_.size())};
    const std::string* const previous_keys = key_storage_.data();
    key_storage_.push_back(std::move(key));
    try
    {
        if (key_storage_.data() != previous_keys && !entries_.empty())
        {
            rebuild_key_views_and_index();
        }
        entries_.push_back(
            {std::string_view(key_storage_.back()), std::move(value)});
    }
    catch (...)
    {
        key_storage_.pop_back();
        throw;
    }
    const auto inserted = [&]
    {
        try
        {
            return key_index_.emplace(
                entries_.back().key, candidate);
        }
        catch (...)
        {
            entries_.pop_back();
            key_storage_.pop_back();
            throw;
        }
    }();
    if (inserted.second)
    {
        sorted_key_ids_dirty_.store(
            true, std::memory_order_release);
        return candidate;
    }

    SettingValue replacement = std::move(entries_.back().value);
    entries_.pop_back();
    key_storage_.pop_back();
    const SettingKeyId existing = inserted.first->second;
    entries_[existing.value].value = std::move(replacement);
    return existing;
}

void SettingObject::set(
    SettingKeyId id,
    SettingValue value)
{
    entries_[checked_index(id)].value = std::move(value);
}

SettingValue& SettingObject::at(SettingKeyId id)
{
    return entries_[checked_index(id)].value;
}

const SettingValue& SettingObject::at(SettingKeyId id) const
{
    return entries_[checked_index(id)].value;
}

std::string_view SettingObject::key_name(SettingKeyId id) const
{
    return entries_[checked_index(id)].key;
}

const std::vector<SettingKeyId>& SettingObject::sorted_key_ids() const
{
    if (!sorted_key_ids_dirty_.load(std::memory_order_acquire))
    {
        return sorted_key_ids_;
    }
    std::lock_guard<std::mutex> lock(sorted_key_ids_mutex_);
    if (!sorted_key_ids_dirty_.load(std::memory_order_relaxed))
    {
        return sorted_key_ids_;
    }
    sorted_key_ids_.clear();
    sorted_key_ids_.reserve(entries_.size());
    for (std::size_t index = 0; index < entries_.size(); ++index)
    {
        sorted_key_ids_.push_back(
            SettingKeyId{static_cast<std::uint32_t>(index)});
    }
    std::sort(
        sorted_key_ids_.begin(),
        sorted_key_ids_.end(),
        [&](SettingKeyId left, SettingKeyId right)
        {
            return key_name(left) < key_name(right);
        });
    sorted_key_ids_dirty_.store(false, std::memory_order_release);
    return sorted_key_ids_;
}

std::size_t SettingObject::checked_index(SettingKeyId id) const
{
    const std::size_t index = id.value;
    if (index >= entries_.size())
    {
        throw std::out_of_range("SettingObject key ID is out of range");
    }
    return index;
}

void SettingObject::rebuild_index()
{
    KeyIndex replacement;
    replacement.reserve(entries_.size());
    for (std::size_t index = 0; index < entries_.size(); ++index)
    {
        replacement.emplace(
            std::string_view(entries_[index].key),
            SettingKeyId{static_cast<std::uint32_t>(index)});
    }
    key_index_.swap(replacement);
}

void SettingObject::rebuild_key_views_and_index()
{
    KeyIndex replacement;
    replacement.reserve(entries_.size());
    for (std::size_t index = 0; index < entries_.size(); ++index)
    {
        const std::string_view key(key_storage_[index]);
        replacement.emplace(
            key,
            SettingKeyId{static_cast<std::uint32_t>(index)});
    }
    for (std::size_t index = 0; index < entries_.size(); ++index)
    {
        entries_[index].key = key_storage_[index];
    }
    key_index_.swap(replacement);
}

void SettingObject::swap(SettingObject& other) noexcept
{
    using std::swap;
    swap(key_storage_, other.key_storage_);
    swap(entries_, other.entries_);
    swap(key_index_, other.key_index_);
    swap(sorted_key_ids_, other.sorted_key_ids_);
    const bool left_dirty =
        sorted_key_ids_dirty_.load(std::memory_order_relaxed);
    const bool right_dirty =
        other.sorted_key_ids_dirty_.load(std::memory_order_relaxed);
    sorted_key_ids_dirty_.store(right_dirty, std::memory_order_relaxed);
    other.sorted_key_ids_dirty_.store(left_dirty, std::memory_order_relaxed);
}

void swap(SettingObject& left, SettingObject& right) noexcept
{
    left.swap(right);
}

} // namespace virne::utils

namespace
{

using virne::utils::SettingDocument;
using virne::utils::SettingErrorCode;
using virne::utils::SettingException;
using virne::utils::SettingFormat;
using virne::utils::SettingInteger;
using virne::utils::SettingKeyId;
using virne::utils::SettingList;
using virne::utils::SettingListPtr;
using virne::utils::SettingMode;
using virne::utils::SettingObject;
using virne::utils::SettingObjectPtr;
using virne::utils::SettingValue;
using virne::utils::SettingValueKind;

constexpr std::string_view unsupported_format_message =
    "Only supports settings files in yaml and json format!";

SettingInteger parse_decimal_integer(std::string_view text)
{
    if (text.empty())
    {
        throw SettingException(
            SettingErrorCode::parse_error,
            "empty integer token");
    }
    bool negative = false;
    std::size_t index = 0;
    if (text.front() == '-' || text.front() == '+')
    {
        negative = text.front() == '-';
        index = 1;
    }
    if (index == text.size())
    {
        throw SettingException(
            SettingErrorCode::parse_error,
            "integer token has no digits");
    }

    SettingInteger result = 0;
    for (; index < text.size(); ++index)
    {
        const char digit = text[index];
        if (digit == '_')
        {
            continue;
        }
        if (digit < '0' || digit > '9')
        {
            throw SettingException(
                SettingErrorCode::parse_error,
                "invalid decimal integer token");
        }
        result *= 10;
        result += static_cast<unsigned int>(digit - '0');
    }
    return negative ? -result : result;
}

struct JsonNumberClassification
{
    bool integer_lexeme = true;
    bool requires_precise_path = false;
};

JsonNumberClassification classify_json_number(
    std::string_view token) noexcept
{
    // Boost reports non-standard JSON constants through on_double(). Avoid
    // mistaking their dot/exponent-free spellings for arbitrary integers.
    if (token == "NaN" ||
        token == "Infinity" ||
        token == "-Infinity")
    {
        return {false, false};
    }

    JsonNumberClassification result;
    result.requires_precise_path = token.size() > 24;
    std::size_t significant_digits = 0;
    bool saw_nonzero = false;
    bool in_exponent = false;
    bool saw_exponent_digit = false;
    unsigned int exponent_magnitude = 0;
    for (std::size_t index = 0; index < token.size(); ++index)
    {
        const char character = token[index];
        if (!in_exponent)
        {
            if (character == '.')
            {
                result.integer_lexeme = false;
                continue;
            }
            if (character == 'e' || character == 'E')
            {
                result.integer_lexeme = false;
                in_exponent = true;
                if (index + 1 < token.size() &&
                    (token[index + 1] == '+' ||
                     token[index + 1] == '-'))
                {
                    ++index;
                }
                continue;
            }
            if (character >= '0' && character <= '9')
            {
                if (character != '0')
                {
                    saw_nonzero = true;
                }
                if (saw_nonzero)
                {
                    ++significant_digits;
                }
            }
            continue;
        }

        if (character >= '0' && character <= '9')
        {
            saw_exponent_digit = true;
            exponent_magnitude = std::min(
                1000U,
                exponent_magnitude * 10U +
                    static_cast<unsigned int>(character - '0'));
        }
        else
        {
            // The SAX parser rejects malformed numbers before this callback,
            // but route an unexpected spelling through the strict converter.
            result.requires_precise_path = true;
        }
    }
    if (significant_digits > 10)
    {
        result.requires_precise_path = true;
    }
    if (in_exponent &&
        (!saw_exponent_digit || exponent_magnitude > 22U))
    {
        result.requires_precise_path = true;
    }
    return result;
}

double precise_json_double(std::string_view token)
{
    double value = 0.0;
    const auto parsed =
        boost::json::detail::charconv::from_chars(
            token.data(), token.data() + token.size(), value);
    if (parsed.ptr != token.data() + token.size() ||
        parsed.ec == std::errc::invalid_argument)
    {
        throw SettingException(
            SettingErrorCode::parse_error,
            "JSON floating-point conversion failed");
    }
    return value;
}

class JsonSettingHandler
{
public:
    static constexpr std::size_t max_array_size =
        std::numeric_limits<std::size_t>::max();
    static constexpr std::size_t max_object_size =
        std::numeric_limits<std::size_t>::max();
    static constexpr std::size_t max_string_size =
        std::numeric_limits<std::size_t>::max();
    static constexpr std::size_t max_key_size =
        std::numeric_limits<std::size_t>::max();

    bool on_document_begin(boost::system::error_code&)
    {
        root_.reset();
        frames_.clear();
        text_buffer_.clear();
        number_buffer_.clear();
        return true;
    }

    bool on_document_end(boost::system::error_code&)
    {
        return root_.has_value() && frames_.empty();
    }

    bool on_array_begin(boost::system::error_code&)
    {
        SettingValue value = SettingValue::make_list();
        SettingList* const pointer = value.list_ptr().get();
        pointer->reserve(8);
        attach(std::move(value));
        frames_.push_back(Frame{pointer, nullptr, {}});
        return true;
    }

    bool on_array_end(std::size_t, boost::system::error_code&)
    {
        if (frames_.empty() ||
            frames_.back().list == nullptr)
        {
            return false;
        }
        frames_.pop_back();
        return true;
    }

    bool on_object_begin(boost::system::error_code&)
    {
        SettingValue value = SettingValue::make_object();
        SettingObject* const pointer = value.object_ptr().get();
        // Sixteen covers the common medium setting mappings and avoids the
        // key-view/index rebuild at entry nine. Larger maps retain geometric
        // vector growth without imposing a 32-entry footprint on every node.
        pointer->reserve(16);
        attach(std::move(value));
        frames_.push_back(Frame{nullptr, pointer, {}});
        return true;
    }

    bool on_object_end(std::size_t, boost::system::error_code&)
    {
        if (frames_.empty() ||
            frames_.back().object == nullptr)
        {
            return false;
        }
        frames_.pop_back();
        return true;
    }

    bool on_string_part(
        boost::json::string_view value,
        std::size_t,
        boost::system::error_code&)
    {
        text_buffer_.append(value.data(), value.size());
        return true;
    }

    bool on_string(
        boost::json::string_view value,
        std::size_t,
        boost::system::error_code&)
    {
        if (text_buffer_.empty())
        {
            attach(SettingValue(
                std::string(value.data(), value.size())));
            return true;
        }
        text_buffer_.append(value.data(), value.size());
        attach(SettingValue(std::move(text_buffer_)));
        text_buffer_.clear();
        return true;
    }

    bool on_key_part(
        boost::json::string_view value,
        std::size_t,
        boost::system::error_code&)
    {
        text_buffer_.append(value.data(), value.size());
        return true;
    }

    bool on_key(
        boost::json::string_view value,
        std::size_t,
        boost::system::error_code&)
    {
        if (frames_.empty() ||
            frames_.back().object == nullptr)
        {
            return false;
        }
        if (text_buffer_.empty())
        {
            frames_.back().pending_key.assign(
                value.data(), value.size());
            return true;
        }
        text_buffer_.append(value.data(), value.size());
        frames_.back().pending_key = std::move(text_buffer_);
        text_buffer_.clear();
        return true;
    }

    bool on_number_part(
        boost::json::string_view value,
        boost::system::error_code&)
    {
        number_buffer_.append(value.data(), value.size());
        return true;
    }

    bool on_int64(
        std::int64_t value,
        boost::json::string_view,
        boost::system::error_code&)
    {
        number_buffer_.clear();
        attach(SettingValue(value));
        return true;
    }

    bool on_uint64(
        std::uint64_t value,
        boost::json::string_view,
        boost::system::error_code&)
    {
        number_buffer_.clear();
        attach(SettingValue(value));
        return true;
    }

    bool on_double(
        double value,
        boost::json::string_view token,
        boost::system::error_code&)
    {
        // A complete one-buffer token is overwhelmingly common. Classify that
        // view in place instead of copying it into the retained chunk buffer.
        if (number_buffer_.empty())
        {
            return attach_double_token(
                value,
                std::string_view(token.data(), token.size()));
        }

        number_buffer_.append(token.data(), token.size());
        const bool result = attach_double_token(value, number_buffer_);
        number_buffer_.clear();
        return result;
    }

    bool on_bool(bool value, boost::system::error_code&)
    {
        attach(SettingValue(value));
        return true;
    }

    bool on_null(boost::system::error_code&)
    {
        attach(SettingValue{});
        return true;
    }

    bool on_comment_part(
        boost::json::string_view,
        boost::system::error_code&)
    {
        return false;
    }

    bool on_comment(
        boost::json::string_view,
        boost::system::error_code&)
    {
        return false;
    }

    SettingDocument release_document()
    {
        if (!root_)
        {
            throw SettingException(
                SettingErrorCode::parse_error,
                "JSON parser produced no root value");
        }
        return SettingDocument{std::move(*root_)};
    }

private:
    struct Frame
    {
        SettingList* list = nullptr;
        SettingObject* object = nullptr;
        std::string pending_key;
    };

    bool attach_double_token(
        double value,
        std::string_view token)
    {
        const JsonNumberClassification classification =
            classify_json_number(token);
        if (classification.integer_lexeme)
        {
            attach(SettingValue(parse_decimal_integer(token)));
            return true;
        }
        if (classification.requires_precise_path)
        {
            value = precise_json_double(token);
        }
        attach(SettingValue(value));
        return true;
    }

    void attach(SettingValue value)
    {
        if (frames_.empty())
        {
            if (root_)
            {
                throw SettingException(
                    SettingErrorCode::parse_error,
                    "JSON contains more than one root value");
            }
            root_ = std::move(value);
            return;
        }

        Frame& frame = frames_.back();
        if (frame.list != nullptr)
        {
            frame.list->push_back(std::move(value));
            return;
        }
        frame.object->set_owned(
            std::move(frame.pending_key), std::move(value));
    }

    std::optional<SettingValue> root_;
    std::vector<Frame> frames_;
    std::string text_buffer_;
    std::string number_buffer_;
};

SettingDocument parse_json(std::string_view input)
{
    const auto make_options = []
    {
        boost::json::parse_options options;
        options.max_depth = 1024;
        options.numbers = boost::json::number_precision::imprecise;
        options.allow_infinity_and_nan = true;
        return options;
    };
    // Boost's SAX parser is explicitly designed to reuse its internal stack.
    // One instance per thread removes allocator traffic without introducing
    // synchronization or changing deterministic batch behavior.
    thread_local boost::json::basic_parser<JsonSettingHandler> parser(
        make_options());
    parser.reset();
    boost::system::error_code error;
    const std::size_t consumed = parser.write_some(
        false, input.data(), input.size(), error);
    if (error)
    {
        throw SettingException(
            SettingErrorCode::parse_error,
            std::string("JSON parse failed: ") + error.message());
    }
    if (consumed != input.size())
    {
        throw SettingException(
            SettingErrorCode::parse_error,
            "JSON parse failed: extra data");
    }
    return parser.handler().release_document();
}

std::string ascii_lower(std::string_view value)
{
    std::string result(value);
    for (char& character : result)
    {
        if (character >= 'A' && character <= 'Z')
        {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    return result;
}

std::string without_underscores(std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    for (const char character : value)
    {
        if (character != '_')
        {
            result.push_back(character);
        }
    }
    return result;
}

std::size_t find_character(
    std::string_view value,
    char character,
    std::size_t begin = 0) noexcept
{
    for (std::size_t index = begin; index < value.size(); ++index)
    {
        if (value[index] == character)
        {
            return index;
        }
    }
    return std::string_view::npos;
}

bool all_digits_for_base(
    std::string_view value,
    std::size_t begin,
    unsigned int base)
{
    bool saw_digit = false;
    for (std::size_t index = begin; index < value.size(); ++index)
    {
        const char character = value[index];
        if (character == '_')
        {
            continue;
        }
        unsigned int digit = base;
        if (character >= '0' && character <= '9')
        {
            digit = static_cast<unsigned int>(character - '0');
        }
        else if (character >= 'a' && character <= 'f')
        {
            digit = 10U +
                    static_cast<unsigned int>(character - 'a');
        }
        else if (character >= 'A' && character <= 'F')
        {
            digit = 10U +
                    static_cast<unsigned int>(character - 'A');
        }
        if (digit >= base)
        {
            return false;
        }
        saw_digit = true;
    }
    return saw_digit;
}

SettingInteger parse_based_integer(
    std::string_view value,
    std::size_t begin,
    unsigned int base,
    bool negative)
{
    SettingInteger result = 0;
    for (std::size_t index = begin; index < value.size(); ++index)
    {
        const char character = value[index];
        if (character == '_')
        {
            continue;
        }
        unsigned int digit = 0;
        if (character >= '0' && character <= '9')
        {
            digit = static_cast<unsigned int>(character - '0');
        }
        else if (character >= 'a' && character <= 'f')
        {
            digit = 10U +
                    static_cast<unsigned int>(character - 'a');
        }
        else
        {
            digit = 10U +
                    static_cast<unsigned int>(character - 'A');
        }
        result *= base;
        result += digit;
    }
    return negative ? -result : result;
}

std::optional<SettingInteger> try_parse_yaml_integer(
    std::string_view token)
{
    if (token.empty())
    {
        return std::nullopt;
    }
    std::size_t begin = 0;
    bool negative = false;
    if (token.front() == '+' || token.front() == '-')
    {
        negative = token.front() == '-';
        begin = 1;
    }
    if (begin == token.size())
    {
        return std::nullopt;
    }

    const std::string_view body = token.substr(begin);
    if (find_character(body, ':') != std::string_view::npos)
    {
        if (body.front() < '1' || body.front() > '9')
        {
            return std::nullopt;
        }
        SettingInteger result = 0;
        std::size_t segment_begin = 0;
        while (segment_begin < body.size())
        {
            const std::size_t colon =
                find_character(body, ':', segment_begin);
            const std::size_t segment_end =
                colon == std::string_view::npos
                    ? body.size()
                    : colon;
            const std::string_view segment =
                body.substr(segment_begin, segment_end - segment_begin);
            if (!all_digits_for_base(segment, 0, 10))
            {
                return std::nullopt;
            }
            const SettingInteger part =
                parse_based_integer(segment, 0, 10, false);
            if (segment_begin != 0 && part >= 60)
            {
                return std::nullopt;
            }
            result *= 60;
            result += part;
            if (colon == std::string_view::npos)
            {
                break;
            }
            segment_begin = colon + 1;
        }
        return negative ? -result : result;
    }

    if (body.size() > 2 &&
        body[0] == '0' &&
        (body[1] == 'b' || body[1] == 'B') &&
        all_digits_for_base(body, 2, 2))
    {
        return parse_based_integer(body, 2, 2, negative);
    }
    if (body.size() > 2 &&
        body[0] == '0' &&
        (body[1] == 'x' || body[1] == 'X') &&
        all_digits_for_base(body, 2, 16))
    {
        return parse_based_integer(body, 2, 16, negative);
    }
    if (body.size() > 1 &&
        body.front() == '0' &&
        all_digits_for_base(body, 1, 8))
    {
        return parse_based_integer(body, 1, 8, negative);
    }
    if (all_digits_for_base(body, 0, 10) &&
        (body == "0" || body.front() != '0'))
    {
        return parse_based_integer(body, 0, 10, negative);
    }
    return std::nullopt;
}

std::optional<double> try_parse_yaml_float(
    std::string_view token,
    bool implicit_resolution = true)
{
    if (token.empty())
    {
        return std::nullopt;
    }
    const std::string lower = ascii_lower(token);
    if (lower == ".nan" || lower == "+.nan" || lower == "-.nan")
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (lower == ".inf" || lower == "+.inf")
    {
        return std::numeric_limits<double>::infinity();
    }
    if (lower == "-.inf")
    {
        return -std::numeric_limits<double>::infinity();
    }

    const bool has_decimal =
        find_character(token, '.') != std::string_view::npos;
    const std::size_t exponent_position = token.find_first_of("eE");
    const bool has_exponent =
        exponent_position != std::string_view::npos;
    const bool has_sexagesimal =
        find_character(token, ':') != std::string_view::npos;
    // PyYAML 6.0.1's implicit YAML 1.1 resolver does not classify an
    // exponent-only spelling such as 1e-5 as a float. An explicit !!float
    // tag still accepts it through the constructor.
    if (implicit_resolution && !has_decimal)
    {
        return std::nullopt;
    }
    if (!has_decimal && !has_exponent)
    {
        return std::nullopt;
    }
    if (has_exponent)
    {
        // PyYAML 6.0.1's YAML 1.1 resolver requires an exponent sign.
        if (exponent_position + 1 >= token.size() ||
            (token[exponent_position + 1] != '+' &&
             token[exponent_position + 1] != '-'))
        {
            return std::nullopt;
        }
    }

    if (has_sexagesimal)
    {
        std::string clean = without_underscores(token);
        bool negative = false;
        std::size_t begin = 0;
        if (clean.front() == '+' || clean.front() == '-')
        {
            negative = clean.front() == '-';
            begin = 1;
        }
        double result = 0.0;
        while (begin < clean.size())
        {
            const std::size_t colon = clean.find(':', begin);
            const std::size_t end =
                colon == std::string::npos ? clean.size() : colon;
            const std::string segment = clean.substr(begin, end - begin);
            char* parse_end = nullptr;
            const double part = std::strtod(segment.c_str(), &parse_end);
            if (parse_end != segment.c_str() + segment.size())
            {
                return std::nullopt;
            }
            result = result * 60.0 + part;
            if (colon == std::string::npos)
            {
                break;
            }
            begin = colon + 1;
        }
        return negative ? -result : result;
    }

    std::string clean = without_underscores(token);
    if (!clean.empty() && clean.front() == '.')
    {
        clean.insert(clean.begin(), '0');
    }
    else if (clean.size() >= 2 &&
             (clean[0] == '+' || clean[0] == '-') &&
             clean[1] == '.')
    {
        clean.insert(clean.begin() + 1, '0');
    }
    const std::size_t exponent = clean.find_first_of("eE");
    const std::size_t mantissa_end =
        exponent == std::string::npos ? clean.size() : exponent;
    if (mantissa_end > 0 && clean[mantissa_end - 1] == '.')
    {
        clean.insert(mantissa_end, 1, '0');
    }

    double result = 0.0;
    const auto parsed = std::from_chars(
        clean.data(),
        clean.data() + clean.size(),
        result,
        std::chars_format::general);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != clean.data() + clean.size())
    {
        return std::nullopt;
    }
    return result;
}

bool try_parse_yaml_bool(
    std::string_view token,
    bool& result)
{
    const std::string lower = ascii_lower(token);
    if (lower == "yes" || lower == "true" || lower == "on")
    {
        result = true;
        return true;
    }
    if (lower == "no" || lower == "false" || lower == "off")
    {
        result = false;
        return true;
    }
    return false;
}

bool is_yaml_null(std::string_view token)
{
    const std::string lower = ascii_lower(token);
    return token == "~" || lower == "null";
}

bool looks_like_yaml_timestamp(std::string_view token)
{
    if (token.size() < 10)
    {
        return false;
    }
    const auto digit = [&](std::size_t index)
    {
        return index < token.size() &&
               token[index] >= '0' &&
               token[index] <= '9';
    };
    return digit(0) && digit(1) && digit(2) && digit(3) &&
           token[4] == '-' &&
           digit(5) && digit(6) &&
           token[7] == '-' &&
           digit(8) && digit(9);
}

std::optional<SettingValue> resolve_plain_yaml_scalar(
    std::string_view token)
{
    if (is_yaml_null(token))
    {
        return SettingValue{};
    }
    bool boolean = false;
    if (try_parse_yaml_bool(token, boolean))
    {
        return SettingValue(boolean);
    }
    if (const auto integer = try_parse_yaml_integer(token))
    {
        return SettingValue(*integer);
    }
    if (const auto real = try_parse_yaml_float(token))
    {
        return SettingValue(*real);
    }
    return std::nullopt;
}

bool is_default_yaml_tag(std::string_view tag)
{
    return tag.empty() || tag == "?" || tag == "!";
}

bool is_core_yaml_tag(
    std::string_view tag,
    std::string_view name)
{
    return tag == std::string("tag:yaml.org,2002:") + std::string(name);
}

struct YamlMemoEntry
{
    YAML::Node node;
    SettingValue value;
};

class YamlSettingConverter
{
public:
    SettingDocument convert(std::string_view input)
    {
        std::vector<YAML::Node> documents;
        try
        {
            documents = YAML::LoadAll(std::string(input));
        }
        catch (const YAML::Exception& error)
        {
            throw SettingException(
                SettingErrorCode::parse_error,
                std::string("YAML parse failed: ") + error.what());
        }
        if (documents.empty())
        {
            return SettingDocument{SettingValue{}};
        }
        if (documents.size() != 1)
        {
            throw SettingException(
                SettingErrorCode::parse_error,
                "YAML parse failed: expected a single document");
        }
        return SettingDocument{convert_node(documents.front())};
    }

private:
    std::optional<SettingValue> find_memo(const YAML::Node& node) const
    {
        const auto bucket = memo_.find(node.Mark().pos);
        if (bucket == memo_.end())
        {
            return std::nullopt;
        }
        for (const YamlMemoEntry& entry : bucket->second)
        {
            if (node.is(entry.node))
            {
                return entry.value;
            }
        }
        return std::nullopt;
    }

    void validate_container_tag(
        const YAML::Node& node,
        std::string_view expected) const
    {
        const std::string tag = node.Tag();
        if (!is_default_yaml_tag(tag) &&
            !is_core_yaml_tag(tag, expected))
        {
            throw SettingException(
                SettingErrorCode::unsupported_yaml_feature,
                "unsupported YAML container tag: " + tag);
        }
    }

    SettingValue convert_node(const YAML::Node& node)
    {
        if (!node || node.IsNull())
        {
            return SettingValue{};
        }
        if (node.IsSequence())
        {
            validate_container_tag(node, "seq");
            if (const auto prior = find_memo(node))
            {
                return *prior;
            }
            SettingValue value = SettingValue::make_list();
            memo_[node.Mark().pos].push_back({node, value});
            SettingList& list = value.as_list();
            list.reserve(node.size());
            for (const YAML::Node& item : node)
            {
                list.push_back(convert_node(item));
            }
            return value;
        }
        if (node.IsMap())
        {
            validate_container_tag(node, "map");
            if (const auto prior = find_memo(node))
            {
                return *prior;
            }
            SettingValue value = SettingValue::make_object();
            memo_[node.Mark().pos].push_back({node, value});
            SettingObject& object = value.as_object();
            object.reserve(node.size());
            for (const auto& item : node)
            {
                if (!item.first.IsScalar())
                {
                    throw SettingException(
                        SettingErrorCode::unsupported_yaml_feature,
                        "YAML mapping keys must be strings");
                }
                const SettingValue key = convert_scalar(item.first);
                if (key.kind() != SettingValueKind::string)
                {
                    throw SettingException(
                        SettingErrorCode::unsupported_yaml_feature,
                        "YAML mapping keys must resolve to strings");
                }
                if (key.as_string() == "<<")
                {
                    throw SettingException(
                        SettingErrorCode::unsupported_yaml_feature,
                        "YAML merge keys are outside the setting profile");
                }
                object.set(key.as_string(), convert_node(item.second));
            }
            return value;
        }
        if (node.IsScalar())
        {
            return convert_scalar(node);
        }
        throw SettingException(
            SettingErrorCode::unsupported_yaml_feature,
            "unsupported YAML node kind");
    }

    SettingValue convert_scalar(const YAML::Node& node) const
    {
        const std::string scalar = node.Scalar();
        const std::string tag = node.Tag();
        if (tag == "!" || is_core_yaml_tag(tag, "str"))
        {
            return SettingValue(scalar);
        }
        if (is_core_yaml_tag(tag, "null"))
        {
            return SettingValue{};
        }
        if (is_core_yaml_tag(tag, "bool"))
        {
            bool result = false;
            if (!try_parse_yaml_bool(scalar, result))
            {
                throw SettingException(
                    SettingErrorCode::parse_error,
                    "invalid explicitly tagged YAML boolean");
            }
            return SettingValue(result);
        }
        if (is_core_yaml_tag(tag, "int"))
        {
            const auto result = try_parse_yaml_integer(scalar);
            if (!result)
            {
                throw SettingException(
                    SettingErrorCode::parse_error,
                    "invalid explicitly tagged YAML integer");
            }
            return SettingValue(*result);
        }
        if (is_core_yaml_tag(tag, "float"))
        {
            const auto result = try_parse_yaml_float(scalar, false);
            if (!result)
            {
                throw SettingException(
                    SettingErrorCode::parse_error,
                    "invalid explicitly tagged YAML float");
            }
            return SettingValue(*result);
        }
        if (!is_default_yaml_tag(tag))
        {
            throw SettingException(
                SettingErrorCode::unsupported_yaml_feature,
                "unsupported YAML scalar tag: " + tag);
        }
        if (looks_like_yaml_timestamp(scalar))
        {
            throw SettingException(
                SettingErrorCode::unsupported_yaml_feature,
                "YAML timestamps are outside the setting profile");
        }
        if (tag == "?")
        {
            if (const auto resolved =
                    resolve_plain_yaml_scalar(scalar))
            {
                return *resolved;
            }
        }
        return SettingValue(scalar);
    }

    // yaml-cpp aliases share the underlying node and therefore its source
    // mark. The mark index makes ordinary large event files O(n) while the
    // identity confirmation preserves correctness for rare mark collisions.
    std::unordered_map<int, std::vector<YamlMemoEntry>> memo_;
};

SettingDocument parse_yaml(std::string_view input)
{
    return YamlSettingConverter{}.convert(input);
}

struct ContainerKey
{
    const void* pointer = nullptr;
    SettingValueKind kind = SettingValueKind::null_value;

    friend bool operator==(
        const ContainerKey& left,
        const ContainerKey& right) noexcept
    {
        return left.pointer == right.pointer &&
               left.kind == right.kind;
    }
};

struct ContainerKeyHash
{
    std::size_t operator()(const ContainerKey& key) const noexcept
    {
        const std::size_t pointer_hash =
            std::hash<const void*>{}(key.pointer);
        const std::size_t kind_hash =
            static_cast<std::size_t>(key.kind);
        return pointer_hash ^
               (kind_hash + 0x9e3779b9U +
                (pointer_hash << 6U) +
                (pointer_hash >> 2U));
    }
};

ContainerKey container_key(const SettingValue& value)
{
    if (value.kind() == SettingValueKind::list)
    {
        return {value.list_ptr().get(), SettingValueKind::list};
    }
    if (value.kind() == SettingValueKind::object)
    {
        return {value.object_ptr().get(), SettingValueKind::object};
    }
    throw std::logic_error("scalar SettingValue has no container identity");
}

bool decode_utf8(
    std::string_view input,
    std::size_t& index,
    std::uint32_t& code_point)
{
    const auto byte = [&](std::size_t position)
    {
        return static_cast<unsigned char>(input[position]);
    };
    if (index >= input.size())
    {
        return false;
    }
    const unsigned char first = byte(index);
    if (first <= 0x7FU)
    {
        code_point = first;
        ++index;
        return true;
    }

    std::size_t length = 0;
    std::uint32_t minimum = 0;
    if ((first & 0xE0U) == 0xC0U)
    {
        length = 2;
        code_point = first & 0x1FU;
        minimum = 0x80U;
    }
    else if ((first & 0xF0U) == 0xE0U)
    {
        length = 3;
        code_point = first & 0x0FU;
        minimum = 0x800U;
    }
    else if ((first & 0xF8U) == 0xF0U)
    {
        length = 4;
        code_point = first & 0x07U;
        minimum = 0x10000U;
    }
    else
    {
        return false;
    }
    if (index + length > input.size())
    {
        return false;
    }
    for (std::size_t offset = 1; offset < length; ++offset)
    {
        const unsigned char continuation = byte(index + offset);
        if ((continuation & 0xC0U) != 0x80U)
        {
            return false;
        }
        code_point =
            (code_point << 6U) |
            static_cast<std::uint32_t>(continuation & 0x3FU);
    }
    if (code_point < minimum ||
        code_point > 0x10FFFFU ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU))
    {
        return false;
    }
    index += length;
    return true;
}

char lowercase_hex_digit(unsigned int value)
{
    return value < 10U
        ? static_cast<char>('0' + value)
        : static_cast<char>('a' + (value - 10U));
}

char uppercase_hex_digit(unsigned int value)
{
    return value < 10U
        ? static_cast<char>('0' + value)
        : static_cast<char>('A' + (value - 10U));
}

void append_hex_escape(
    std::string& output,
    std::uint32_t value,
    std::size_t digits,
    bool uppercase)
{
    for (std::size_t shift_index = digits; shift_index > 0; --shift_index)
    {
        const unsigned int nibble = static_cast<unsigned int>(
            (value >> ((shift_index - 1U) * 4U)) & 0xFU);
        output.push_back(
            uppercase
                ? uppercase_hex_digit(nibble)
                : lowercase_hex_digit(nibble));
    }
}

void append_json_string(
    std::string& output,
    std::string_view value)
{
    output.push_back('"');
    std::size_t index = 0;
    while (index < value.size())
    {
        const std::size_t byte_index = index;
        std::uint32_t code_point = 0;
        if (!decode_utf8(value, index, code_point))
        {
            throw SettingException(
                SettingErrorCode::serialization_error,
                "setting string is not valid UTF-8");
        }
        switch (code_point)
        {
        case '"':
            output += "\\\"";
            break;
        case '\\':
            output += "\\\\";
            break;
        case '\b':
            output += "\\b";
            break;
        case '\t':
            output += "\\t";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\f':
            output += "\\f";
            break;
        case '\r':
            output += "\\r";
            break;
        default:
            if (code_point < 0x20U)
            {
                output += "\\u";
                append_hex_escape(output, code_point, 4, false);
            }
            else if (code_point <= 0x7FU)
            {
                output.push_back(value[byte_index]);
            }
            else if (code_point <= 0xFFFFU)
            {
                output += "\\u";
                append_hex_escape(output, code_point, 4, false);
            }
            else
            {
                const std::uint32_t adjusted = code_point - 0x10000U;
                const std::uint32_t high =
                    0xD800U + (adjusted >> 10U);
                const std::uint32_t low =
                    0xDC00U + (adjusted & 0x3FFU);
                output += "\\u";
                append_hex_escape(output, high, 4, false);
                output += "\\u";
                append_hex_escape(output, low, 4, false);
            }
            break;
        }
    }
    output.push_back('"');
}

std::string python_float_repr(double value)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (std::isinf(value))
    {
        return std::signbit(value) ? "-inf" : "inf";
    }
    if (value == 0.0)
    {
        return std::signbit(value) ? "-0.0" : "0.0";
    }

    std::array<char, 128> buffer{};
    const auto converted = std::to_chars(
        buffer.data(),
        buffer.data() + buffer.size(),
        value,
        std::chars_format::general);
    if (converted.ec != std::errc{})
    {
        throw SettingException(
            SettingErrorCode::serialization_error,
            "failed to format floating-point setting value");
    }
    std::string raw(buffer.data(), converted.ptr);
    bool negative = false;
    if (!raw.empty() && raw.front() == '-')
    {
        negative = true;
        raw.erase(raw.begin());
    }

    int explicit_exponent = 0;
    const std::size_t exponent_position = raw.find_first_of("eE");
    std::string mantissa =
        exponent_position == std::string::npos
            ? raw
            : raw.substr(0, exponent_position);
    if (exponent_position != std::string::npos)
    {
        explicit_exponent = std::stoi(
            raw.substr(exponent_position + 1));
    }

    const std::size_t dot_position = mantissa.find('.');
    int decimal_position = static_cast<int>(
        dot_position == std::string::npos
            ? mantissa.size()
            : dot_position);
    std::string digits;
    digits.reserve(mantissa.size());
    for (const char character : mantissa)
    {
        if (character != '.')
        {
            digits.push_back(character);
        }
    }
    decimal_position += explicit_exponent;

    std::size_t leading = 0;
    while (leading < digits.size() && digits[leading] == '0')
    {
        ++leading;
    }
    if (leading == digits.size())
    {
        return negative ? "-0.0" : "0.0";
    }
    digits.erase(0, leading);
    decimal_position -= static_cast<int>(leading);
    while (digits.size() > 1 && digits.back() == '0')
    {
        digits.pop_back();
    }

    const int first_digit_exponent = decimal_position - 1;
    std::string result;
    if (negative)
    {
        result.push_back('-');
    }
    if (first_digit_exponent >= -4 &&
        first_digit_exponent < 16)
    {
        if (decimal_position <= 0)
        {
            result += "0.";
            result.append(
                static_cast<std::size_t>(-decimal_position), '0');
            result += digits;
        }
        else if (decimal_position >=
                 static_cast<int>(digits.size()))
        {
            result += digits;
            result.append(
                static_cast<std::size_t>(
                    decimal_position -
                    static_cast<int>(digits.size())),
                '0');
            result += ".0";
        }
        else
        {
            const std::size_t split =
                static_cast<std::size_t>(decimal_position);
            result.append(digits, 0, split);
            result.push_back('.');
            result.append(digits, split, std::string::npos);
        }
        return result;
    }

    result.push_back(digits.front());
    if (digits.size() > 1)
    {
        result.push_back('.');
        result.append(digits, 1, std::string::npos);
    }
    result.push_back('e');
    result.push_back(first_digit_exponent >= 0 ? '+' : '-');
    const unsigned int exponent_magnitude =
        static_cast<unsigned int>(
            first_digit_exponent >= 0
                ? first_digit_exponent
                : -first_digit_exponent);
    if (exponent_magnitude < 10U)
    {
        result.push_back('0');
    }
    result += std::to_string(exponent_magnitude);
    return result;
}

class JsonSettingEmitter
{
public:
    std::string emit(const SettingDocument& document)
    {
        output_.clear();
        active_.clear();
        emit_value(document.root);
        return output_;
    }

private:
    class ActiveScope
    {
    public:
        ActiveScope(
            std::unordered_set<ContainerKey, ContainerKeyHash>& active,
            ContainerKey key)
            : active_(active),
              key_(key)
        {
            if (!active_.insert(key_).second)
            {
                throw SettingException(
                    SettingErrorCode::serialization_error,
                    "circular setting value cannot be serialized as JSON");
            }
        }

        ~ActiveScope()
        {
            active_.erase(key_);
        }

        ActiveScope(const ActiveScope&) = delete;
        ActiveScope& operator=(const ActiveScope&) = delete;

    private:
        std::unordered_set<ContainerKey, ContainerKeyHash>& active_;
        ContainerKey key_;
    };

    void emit_value(const SettingValue& value)
    {
        switch (value.kind())
        {
        case SettingValueKind::null_value:
            output_ += "null";
            return;
        case SettingValueKind::boolean:
            output_ += value.as_bool() ? "true" : "false";
            return;
        case SettingValueKind::integer:
            output_ += value.as_integer().convert_to<std::string>();
            return;
        case SettingValueKind::real:
            emit_real(value.as_real());
            return;
        case SettingValueKind::string:
            append_json_string(output_, value.as_string());
            return;
        case SettingValueKind::list:
            emit_list(value);
            return;
        case SettingValueKind::object:
            emit_object(value);
            return;
        }
        throw std::logic_error("unknown SettingValueKind");
    }

    void emit_real(double value)
    {
        if (std::isnan(value))
        {
            output_ += "NaN";
        }
        else if (std::isinf(value))
        {
            output_ += std::signbit(value)
                ? "-Infinity"
                : "Infinity";
        }
        else
        {
            output_ += python_float_repr(value);
        }
    }

    void emit_list(const SettingValue& value)
    {
        ActiveScope scope(active_, container_key(value));
        output_.push_back('[');
        const SettingList& list = value.as_list();
        for (std::size_t index = 0; index < list.size(); ++index)
        {
            if (index != 0)
            {
                output_ += ", ";
            }
            emit_value(list[index]);
        }
        output_.push_back(']');
    }

    void emit_object(const SettingValue& value)
    {
        ActiveScope scope(active_, container_key(value));
        output_.push_back('{');
        const SettingObject& object = value.as_object();
        for (std::size_t index = 0; index < object.size(); ++index)
        {
            if (index != 0)
            {
                output_ += ", ";
            }
            const SettingKeyId id{
                static_cast<std::uint32_t>(index)};
            append_json_string(output_, object.key_name(id));
            output_ += ": ";
            emit_value(object.at(id));
        }
        output_.push_back('}');
    }

    std::string output_;
    std::unordered_set<ContainerKey, ContainerKeyHash> active_;
};

bool yaml_string_would_resolve(std::string_view value)
{
    return resolve_plain_yaml_scalar(value).has_value() ||
           looks_like_yaml_timestamp(value);
}

bool yaml_plain_string_allowed(std::string_view value)
{
    if (value.empty() ||
        value.front() == ' ' ||
        value.back() == ' ')
    {
        return false;
    }
    const unsigned char first =
        static_cast<unsigned char>(value.front());
    if (first < 0x20U || first >= 0x7FU)
    {
        return false;
    }
    const auto indicator = [](char character)
    {
        switch (character)
        {
        case ',':
        case '[':
        case ']':
        case '{':
        case '}':
        case '#':
        case '&':
        case '*':
        case '!':
        case '|':
        case '>':
        case '\'':
        case '"':
        case '%':
        case '@':
            return true;
        default:
            return static_cast<unsigned char>(character) == 0x60U;
        }
    };
    if (indicator(value.front()))
    {
        return false;
    }
    if ((value.front() == '-' ||
         value.front() == '?' ||
         value.front() == ':') &&
        (value.size() == 1 ||
         value[1] == ' ' ||
         value[1] == '\t'))
    {
        return false;
    }
    if ((value.rfind("---", 0) == 0 ||
         value.rfind("...", 0) == 0) &&
        (value.size() == 3 ||
         value[3] == ' ' ||
         value[3] == '\t'))
    {
        return false;
    }

    for (std::size_t index = 0; index < value.size(); ++index)
    {
        const unsigned char character =
            static_cast<unsigned char>(value[index]);
        if (character < 0x20U || character >= 0x7FU)
        {
            return false;
        }
        if (value[index] == ':' &&
            index + 1 < value.size() &&
            (value[index + 1] == ' ' ||
             value[index + 1] == '\t'))
        {
            return false;
        }
        if (value[index] == '#' &&
            index > 0 &&
            (value[index - 1] == ' ' ||
             value[index - 1] == '\t'))
        {
            return false;
        }
    }
    return !yaml_string_would_resolve(value);
}

void append_yaml_double_quoted(
    std::string& output,
    std::string_view value)
{
    output.push_back('"');
    std::size_t index = 0;
    while (index < value.size())
    {
        const std::size_t byte_index = index;
        std::uint32_t code_point = 0;
        if (!decode_utf8(value, index, code_point))
        {
            throw SettingException(
                SettingErrorCode::serialization_error,
                "setting string is not valid UTF-8");
        }
        switch (code_point)
        {
        case 0x00U:
            output += "\\0";
            break;
        case 0x07U:
            output += "\\a";
            break;
        case 0x08U:
            output += "\\b";
            break;
        case 0x09U:
            output += "\\t";
            break;
        case 0x0AU:
            output += "\\n";
            break;
        case 0x0BU:
            output += "\\v";
            break;
        case 0x0CU:
            output += "\\f";
            break;
        case 0x0DU:
            output += "\\r";
            break;
        case 0x1BU:
            output += "\\e";
            break;
        case 0x22U:
            output += "\\\"";
            break;
        case 0x5CU:
            output += "\\\\";
            break;
        case 0x85U:
            output += "\\N";
            break;
        case 0xA0U:
            output += "\\_";
            break;
        case 0x2028U:
            output += "\\L";
            break;
        case 0x2029U:
            output += "\\P";
            break;
        default:
            if (code_point >= 0x20U && code_point <= 0x7EU)
            {
                output.push_back(value[byte_index]);
            }
            else if (code_point <= 0xFFU)
            {
                output += "\\x";
                append_hex_escape(output, code_point, 2, true);
            }
            else if (code_point <= 0xFFFFU)
            {
                output += "\\u";
                append_hex_escape(output, code_point, 4, true);
            }
            else
            {
                output += "\\U";
                append_hex_escape(output, code_point, 8, true);
            }
            break;
        }
    }
    output.push_back('"');
}

std::string yaml_string_scalar(std::string_view value)
{
    if (value.empty())
    {
        return "''";
    }
    bool printable_ascii = true;
    for (const char raw_character : value)
    {
        const auto character =
            static_cast<unsigned char>(raw_character);
        if (character < 0x20U || character >= 0x7FU)
        {
            printable_ascii = false;
            break;
        }
    }
    if (yaml_plain_string_allowed(value))
    {
        return std::string(value);
    }
    if (!printable_ascii)
    {
        std::string result;
        append_yaml_double_quoted(result, value);
        return result;
    }
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('\'');
    for (const char character : value)
    {
        result.push_back(character);
        if (character == '\'')
        {
            result.push_back('\'');
        }
    }
    result.push_back('\'');
    return result;
}

std::string yaml_float_scalar(double value)
{
    if (std::isnan(value))
    {
        return ".nan";
    }
    if (std::isinf(value))
    {
        return std::signbit(value) ? "-.inf" : ".inf";
    }
    std::string result = ascii_lower(python_float_repr(value));
    const std::size_t exponent = result.find('e');
    if (exponent != std::string::npos &&
        result.substr(0, exponent).find('.') == std::string::npos)
    {
        result.insert(exponent, ".0");
    }
    return result;
}

const std::vector<SettingKeyId>& sorted_key_ids(
    const SettingObject& object)
{
    return object.sorted_key_ids();
}

bool is_nonempty_container(const SettingValue& value)
{
    if (value.kind() == SettingValueKind::list)
    {
        return !value.as_list().empty();
    }
    if (value.kind() == SettingValueKind::object)
    {
        return !value.as_object().empty();
    }
    return false;
}

class YamlSettingEmitter
{
public:
    std::string emit(const SettingDocument& document)
    {
        anchors_.clear();
        emitted_.clear();
        lines_.clear();
        next_anchor_ = 1;
        scan_for_anchors(document.root);

        const Claim root_claim = claim(document.root);
        if (document.root.kind() != SettingValueKind::list &&
            document.root.kind() != SettingValueKind::object)
        {
            lines_.push_back(scalar(document.root));
            return join_lines(root_is_plain_scalar(document.root));
        }
        if (!is_nonempty_container(document.root))
        {
            std::string line;
            if (!root_claim.anchor.empty())
            {
                line += root_claim.anchor;
                line.push_back(' ');
            }
            line += scalar(document.root);
            lines_.push_back(std::move(line));
            return join_lines(false);
        }
        if (!root_claim.anchor.empty())
        {
            lines_.push_back(root_claim.anchor);
        }
        emit_container_content(document.root, 0);
        return join_lines(false);
    }

private:
    struct Claim
    {
        std::string anchor;
        std::string alias;
    };

    void scan_for_anchors(const SettingValue& value)
    {
        if (value.kind() != SettingValueKind::list &&
            value.kind() != SettingValueKind::object)
        {
            return;
        }
        const ContainerKey key = container_key(value);
        const auto inserted = anchors_.emplace(key, 0U);
        if (!inserted.second)
        {
            if (inserted.first->second == 0U)
            {
                inserted.first->second = next_anchor_++;
            }
            return;
        }

        if (value.kind() == SettingValueKind::list)
        {
            for (const SettingValue& item : value.as_list())
            {
                scan_for_anchors(item);
            }
            return;
        }
        const SettingObject& object = value.as_object();
        for (const SettingKeyId id : sorted_key_ids(object))
        {
            scan_for_anchors(object.at(id));
        }
    }

    std::string anchor_name(unsigned int id) const
    {
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << "id" << std::setw(3) << std::setfill('0') << id;
        return stream.str();
    }

    Claim claim(const SettingValue& value)
    {
        if (value.kind() != SettingValueKind::list &&
            value.kind() != SettingValueKind::object)
        {
            return {};
        }
        const ContainerKey key = container_key(value);
        const auto found = anchors_.find(key);
        if (found == anchors_.end() || found->second == 0U)
        {
            return {};
        }
        const std::string name = anchor_name(found->second);
        if (!emitted_.insert(key).second)
        {
            return {"", "*" + name};
        }
        return {"&" + name, ""};
    }

    std::string scalar(const SettingValue& value) const
    {
        switch (value.kind())
        {
        case SettingValueKind::null_value:
            return "null";
        case SettingValueKind::boolean:
            return value.as_bool() ? "true" : "false";
        case SettingValueKind::integer:
            return value.as_integer().convert_to<std::string>();
        case SettingValueKind::real:
            return yaml_float_scalar(value.as_real());
        case SettingValueKind::string:
            return yaml_string_scalar(value.as_string());
        case SettingValueKind::list:
            if (value.as_list().empty())
            {
                return "[]";
            }
            break;
        case SettingValueKind::object:
            if (value.as_object().empty())
            {
                return "{}";
            }
            break;
        }
        throw std::logic_error(
            "nonempty container cannot be emitted as YAML scalar");
    }

    bool root_is_plain_scalar(const SettingValue& value) const
    {
        if (value.kind() == SettingValueKind::string)
        {
            const std::string rendered =
                yaml_string_scalar(value.as_string());
            return rendered.empty() ||
                   (rendered.front() != '\'' &&
                    rendered.front() != '"');
        }
        return value.kind() != SettingValueKind::list &&
               value.kind() != SettingValueKind::object;
    }

    void emit_container_content(
        const SettingValue& value,
        std::size_t indent)
    {
        if (value.kind() == SettingValueKind::list)
        {
            emit_list_content(value.as_list(), indent);
        }
        else
        {
            emit_object_content(value.as_object(), indent);
        }
    }

    std::string indentation(std::size_t count) const
    {
        return std::string(count, ' ');
    }

    void emit_object_content(
        const SettingObject& object,
        std::size_t indent)
    {
        for (const SettingKeyId id : sorted_key_ids(object))
        {
            const SettingValue& value = object.at(id);
            const Claim value_claim = claim(value);
            std::string prefix =
                indentation(indent) +
                yaml_string_scalar(object.key_name(id)) +
                ":";
            if (!value_claim.alias.empty())
            {
                lines_.push_back(
                    std::move(prefix) + " " + value_claim.alias);
                continue;
            }
            if (!is_nonempty_container(value))
            {
                prefix.push_back(' ');
                if (!value_claim.anchor.empty())
                {
                    prefix += value_claim.anchor;
                    prefix.push_back(' ');
                }
                prefix += scalar(value);
                lines_.push_back(std::move(prefix));
                continue;
            }

            if (!value_claim.anchor.empty())
            {
                prefix.push_back(' ');
                prefix += value_claim.anchor;
            }
            lines_.push_back(std::move(prefix));
            const std::size_t child_indent =
                value.kind() == SettingValueKind::list
                    ? indent
                    : indent + 2;
            emit_container_content(value, child_indent);
        }
    }

    void emit_list_content(
        const SettingList& list,
        std::size_t indent)
    {
        for (const SettingValue& value : list)
        {
            const Claim value_claim = claim(value);
            const std::string dash = indentation(indent) + "-";
            if (!value_claim.alias.empty())
            {
                lines_.push_back(dash + " " + value_claim.alias);
                continue;
            }
            if (!is_nonempty_container(value))
            {
                std::string line = dash + " ";
                if (!value_claim.anchor.empty())
                {
                    line += value_claim.anchor;
                    line.push_back(' ');
                }
                line += scalar(value);
                lines_.push_back(std::move(line));
                continue;
            }
            if (!value_claim.anchor.empty())
            {
                lines_.push_back(dash + " " + value_claim.anchor);
                emit_container_content(value, indent + 2);
                continue;
            }

            const std::size_t first_child_line = lines_.size();
            emit_container_content(value, indent + 2);
            if (lines_.size() == first_child_line)
            {
                throw std::logic_error(
                    "nonempty YAML container emitted no lines");
            }
            std::string& first = lines_[first_child_line];
            const std::size_t child_indent = indent + 2;
            first.replace(
                0,
                std::min(child_indent, first.size()),
                dash + " ");
        }
    }

    std::string join_lines(bool explicit_end)
    {
        std::size_t size = explicit_end ? 5U : 1U;
        for (const std::string& line : lines_)
        {
            size += line.size() + 1U;
        }
        std::string output;
        output.reserve(size);
        for (const std::string& line : lines_)
        {
            output += line;
            output.push_back('\n');
        }
        if (explicit_end)
        {
            output += "...\n";
        }
        return output;
    }

    std::unordered_map<ContainerKey, unsigned int, ContainerKeyHash>
        anchors_;
    std::unordered_set<ContainerKey, ContainerKeyHash> emitted_;
    std::vector<std::string> lines_;
    unsigned int next_anchor_ = 1;
};

struct ModeTraits
{
    std::ios::openmode flags{};
    bool readable = false;
    bool writable = false;
    bool append = false;
    bool exclusive = false;
};

ModeTraits mode_traits(SettingMode mode)
{
    const std::ios::openmode binary = std::ios::binary;
    switch (mode)
    {
    case SettingMode::read:
        return {binary | std::ios::in, true, false, false, false};
    case SettingMode::read_update:
        return {
            binary | std::ios::in | std::ios::out,
            true, true, false, false};
    case SettingMode::write:
        return {
            binary | std::ios::out | std::ios::trunc,
            false, true, false, false};
    case SettingMode::write_update:
        return {
            binary | std::ios::in | std::ios::out | std::ios::trunc,
            true, true, false, false};
    case SettingMode::append:
        return {
            binary | std::ios::out | std::ios::app,
            false, true, true, false};
    case SettingMode::append_update:
        return {
            binary | std::ios::in | std::ios::out | std::ios::app,
            true, true, true, false};
    case SettingMode::exclusive_create:
        return {
            binary | std::ios::out | std::ios::trunc,
            false, true, false, true};
    case SettingMode::exclusive_create_update:
        return {
            binary | std::ios::in | std::ios::out | std::ios::trunc,
            true, true, false, true};
    }
    throw SettingException(
        SettingErrorCode::invalid_mode,
        "unknown setting file mode");
}

std::fstream open_setting_stream(
    const std::string& path,
    const ModeTraits& traits)
{
    if (traits.exclusive &&
        std::filesystem::exists(std::filesystem::path(path)))
    {
        throw SettingException(
            SettingErrorCode::open_failed,
            "setting file already exists: " + path);
    }
    std::fstream stream(path, traits.flags);
    if (!stream.is_open())
    {
        throw SettingException(
            SettingErrorCode::open_failed,
            "failed to open setting file: " + path);
    }
    return stream;
}

std::string normalize_universal_newlines(std::string bytes)
{
    std::string normalized;
    normalized.reserve(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        if (bytes[index] == '\r')
        {
            if (index + 1 < bytes.size() && bytes[index + 1] == '\n')
            {
                ++index;
            }
            normalized.push_back('\n');
        }
        else
        {
            normalized.push_back(bytes[index]);
        }
    }
    return normalized;
}

std::string read_stream_bytes(
    std::fstream& stream,
    const ModeTraits& traits)
{
    if (!traits.readable)
    {
        throw SettingException(
            SettingErrorCode::read_failed,
            "setting file mode is not readable");
    }
    stream.clear();
    stream.seekg(
        0,
        traits.append ? std::ios::end : std::ios::beg);
    if (!stream)
    {
        throw SettingException(
            SettingErrorCode::read_failed,
            "failed to seek setting file");
    }
    std::string bytes{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
    if (stream.bad())
    {
        throw SettingException(
            SettingErrorCode::read_failed,
            "failed to read setting file");
    }
    return normalize_universal_newlines(std::move(bytes));
}

void write_stream_bytes(
    std::fstream& stream,
    const ModeTraits& traits,
    std::string_view bytes)
{
    if (!traits.writable)
    {
        throw SettingException(
            SettingErrorCode::write_failed,
            "setting file mode is not writable");
    }
    stream.clear();
    stream.seekp(0, traits.append ? std::ios::end : std::ios::beg);
    if (!stream)
    {
        throw SettingException(
            SettingErrorCode::write_failed,
            "failed to seek setting file");
    }
    stream.write(
        bytes.data(),
        static_cast<std::streamsize>(bytes.size()));
    stream.flush();
    if (!stream)
    {
        throw SettingException(
            SettingErrorCode::write_failed,
            "failed to write setting file");
    }
}

thread_local bool inside_setting_parallel_task = false;

class SettingParallelTaskScope
{
public:
    SettingParallelTaskScope() noexcept
        : previous_(inside_setting_parallel_task)
    {
        inside_setting_parallel_task = true;
    }

    ~SettingParallelTaskScope()
    {
        inside_setting_parallel_task = previous_;
    }

    SettingParallelTaskScope(const SettingParallelTaskScope&) = delete;
    SettingParallelTaskScope& operator=(
        const SettingParallelTaskScope&) = delete;

private:
    bool previous_;
};

class SettingExecutor
{
public:
    SettingExecutor() = default;
    SettingExecutor(const SettingExecutor&) = delete;
    SettingExecutor& operator=(const SettingExecutor&) = delete;

    ~SettingExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    template <typename Function>
    void run(std::size_t worker_count, Function&& function)
    {
        if (worker_count <= 1)
        {
            function(0);
            return;
        }
        std::lock_guard<std::mutex> execution_lock(execution_mutex_);
        ensure_workers(worker_count - 1);
        std::vector<std::exception_ptr> failures(worker_count);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            task_ = [
                callable = std::forward<Function>(function),
                &failures](std::size_t worker_index) mutable
            {
                try
                {
                    callable(worker_index);
                }
                catch (...)
                {
                    failures[worker_index] =
                        std::current_exception();
                }
            };
            active_background_workers_ = worker_count - 1;
            completed_background_workers_ = 0;
            ++generation_;
        }
        ready_.notify_all();
        task_(0);

        std::unique_lock<std::mutex> state_lock(state_mutex_);
        finished_.wait(state_lock, [this]
        {
            return completed_background_workers_ ==
                   active_background_workers_;
        });
        task_ = {};
        state_lock.unlock();
        for (const std::exception_ptr& failure : failures)
        {
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
    }

private:
    void ensure_workers(std::size_t count)
    {
        while (workers_.size() < count)
        {
            const std::size_t worker_index = workers_.size() + 1;
            std::size_t initial_generation = 0;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                initial_generation = generation_;
            }
            workers_.emplace_back(
                [this, worker_index, initial_generation]
                {
                    worker_loop(worker_index, initial_generation);
                });
        }
    }

    void worker_loop(
        std::size_t worker_index,
        std::size_t seen_generation)
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        for (;;)
        {
            ready_.wait(lock, [this, &seen_generation]
            {
                return generation_ != seen_generation || stopping_;
            });
            if (stopping_)
            {
                return;
            }
            seen_generation = generation_;
            if (worker_index > active_background_workers_)
            {
                continue;
            }
            const auto* task = &task_;
            lock.unlock();
            (*task)(worker_index);
            lock.lock();
            ++completed_background_workers_;
            if (completed_background_workers_ ==
                active_background_workers_)
            {
                finished_.notify_one();
            }
        }
    }

    std::mutex execution_mutex_;
    std::mutex state_mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::function<void(std::size_t)> task_;
    std::size_t generation_ = 0;
    std::size_t active_background_workers_ = 0;
    std::size_t completed_background_workers_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

SettingExecutor& setting_executor()
{
    static SettingExecutor executor;
    return executor;
}

std::size_t available_setting_workers()
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        const int count = CPU_COUNT(&affinity);
        if (count > 0)
        {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    return std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
}

std::size_t resolve_setting_worker_count(
    std::size_t requested,
    std::size_t item_count,
    std::size_t work_units,
    std::size_t auto_threshold)
{
    if (item_count == 0 || requested == 1)
    {
        return 1;
    }
    const std::size_t available = available_setting_workers();
    if (requested == 0)
    {
        if (work_units < auto_threshold)
        {
            return 1;
        }
        requested = std::min<std::size_t>(8, available);
    }
    return std::max<std::size_t>(
        1,
        std::min(
            std::min(requested, item_count),
            available));
}

template <typename Function>
void deterministic_setting_parallel_for(
    std::size_t item_count,
    std::size_t work_units,
    std::size_t auto_threshold,
    std::size_t requested_workers,
    Function&& function)
{
    const std::size_t worker_count = resolve_setting_worker_count(
        requested_workers,
        item_count,
        work_units,
        auto_threshold);
    if (worker_count == 1 || inside_setting_parallel_task)
    {
        for (std::size_t index = 0; index < item_count; ++index)
        {
            function(index);
        }
        return;
    }

    std::vector<std::exception_ptr> failures(item_count);
    setting_executor().run(worker_count, [&](std::size_t worker_index)
    {
        SettingParallelTaskScope scope;
        const std::size_t base = item_count / worker_count;
        const std::size_t remainder = item_count % worker_count;
        const std::size_t begin =
            worker_index * base +
            std::min(worker_index, remainder);
        const std::size_t end =
            begin + base +
            (worker_index < remainder ? 1U : 0U);
        for (std::size_t index = begin; index < end; ++index)
        {
            try
            {
                function(index);
            }
            catch (...)
            {
                failures[index] = std::current_exception();
            }
        }
    });

    for (const std::exception_ptr& failure : failures)
    {
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }
}

std::size_t saturating_add(
    std::size_t left,
    std::size_t right) noexcept
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        return std::numeric_limits<std::size_t>::max();
    }
    return left + right;
}

std::size_t setting_node_count_impl(
    const SettingValue& value,
    std::unordered_set<ContainerKey, ContainerKeyHash>& seen)
{
    std::size_t count = 1;
    if (value.kind() != SettingValueKind::list &&
        value.kind() != SettingValueKind::object)
    {
        return count;
    }
    if (!seen.insert(container_key(value)).second)
    {
        return count;
    }
    if (value.kind() == SettingValueKind::list)
    {
        for (const SettingValue& item : value.as_list())
        {
            count = saturating_add(
                count, setting_node_count_impl(item, seen));
        }
    }
    else
    {
        const SettingObject& object = value.as_object();
        for (std::size_t index = 0; index < object.size(); ++index)
        {
            count = saturating_add(
                count,
                setting_node_count_impl(
                    object.at(SettingKeyId{
                        static_cast<std::uint32_t>(index)}),
                    seen));
        }
    }
    return count;
}

std::size_t setting_node_count(const SettingDocument& document)
{
    std::unordered_set<ContainerKey, ContainerKeyHash> seen;
    return setting_node_count_impl(document.root, seen);
}

} // namespace

namespace virne::utils
{

std::optional<SettingFormat> setting_format_from_path(
    std::string_view path) noexcept
{
    if (path.size() < 4)
    {
        return std::nullopt;
    }
    const std::string_view suffix = path.substr(path.size() - 4);
    if (suffix == "json")
    {
        return SettingFormat::json;
    }
    if (suffix == "yaml")
    {
        return SettingFormat::yaml;
    }
    return std::nullopt;
}

SettingMode parse_setting_mode(std::string_view mode)
{
    if (mode == "r")
    {
        return SettingMode::read;
    }
    if (mode == "r+")
    {
        return SettingMode::read_update;
    }
    if (mode == "w")
    {
        return SettingMode::write;
    }
    if (mode == "w+")
    {
        return SettingMode::write_update;
    }
    if (mode == "a")
    {
        return SettingMode::append;
    }
    if (mode == "a+")
    {
        return SettingMode::append_update;
    }
    if (mode == "x")
    {
        return SettingMode::exclusive_create;
    }
    if (mode == "x+")
    {
        return SettingMode::exclusive_create_update;
    }
    throw SettingException(
        SettingErrorCode::invalid_mode,
        "unsupported setting file mode");
}

SettingDocument parse_setting(
    std::string_view bytes,
    SettingFormat format)
{
    if (format == SettingFormat::json)
    {
        return parse_json(bytes);
    }
    return parse_yaml(bytes);
}

std::string dump_setting(
    const SettingDocument& document,
    SettingFormat format)
{
    if (format == SettingFormat::json)
    {
        return JsonSettingEmitter{}.emit(document);
    }
    return YamlSettingEmitter{}.emit(document);
}

SettingDocument read_setting(
    const std::string& path,
    SettingMode mode)
{
    const ModeTraits traits = mode_traits(mode);
    std::fstream stream = open_setting_stream(path, traits);
    const std::optional<SettingFormat> format =
        setting_format_from_path(path);
    if (!format)
    {
        throw SettingException(
            SettingErrorCode::unsupported_format,
            std::string(unsupported_format_message));
    }
    const std::string bytes = read_stream_bytes(stream, traits);
    return parse_setting(bytes, *format);
}

WriteSettingResult write_setting(
    const SettingDocument& document,
    const std::string& path,
    SettingMode mode)
{
    const ModeTraits traits = mode_traits(mode);
    std::fstream stream = open_setting_stream(path, traits);
    const std::optional<SettingFormat> format =
        setting_format_from_path(path);
    if (!format)
    {
        return ReturnedSettingError{
            SettingErrorCode::unsupported_format,
            std::string(unsupported_format_message)};
    }
    const std::string bytes = dump_setting(document, *format);
    write_stream_bytes(stream, traits, bytes);
    return std::cref(document);
}

const SettingDocument& write_setting_strict(
    const SettingDocument& document,
    const std::string& path,
    SettingMode mode)
{
    WriteSettingResult result = write_setting(document, path, mode);
    if (const auto* error =
            std::get_if<ReturnedSettingError>(&result))
    {
        throw SettingException(error->code, error->message);
    }
    return std::get<std::reference_wrapper<const SettingDocument>>(
               result)
        .get();
}

void conver_format(
    const std::string& source_path,
    const std::string& destination_path)
{
    const SettingDocument document = read_setting(source_path);
    static_cast<void>(write_setting(document, destination_path));
}

std::vector<SettingDocument> parse_setting_batch(
    const std::vector<std::string>& inputs,
    SettingFormat format,
    std::size_t worker_count)
{
    std::size_t bytes = 0;
    for (const std::string& input : inputs)
    {
        bytes = saturating_add(bytes, input.size());
    }
    std::vector<SettingDocument> results(inputs.size());
    deterministic_setting_parallel_for(
        inputs.size(),
        bytes,
        256U * 1024U,
        worker_count,
        [&](std::size_t index)
        {
            results[index] = parse_setting(inputs[index], format);
        });
    return results;
}

std::vector<std::string> dump_setting_batch(
    const std::vector<SettingDocument>& documents,
    SettingFormat format,
    std::size_t worker_count)
{
    std::size_t nodes = 0;
    for (const SettingDocument& document : documents)
    {
        nodes = saturating_add(nodes, setting_node_count(document));
    }
    std::vector<std::string> results(documents.size());
    deterministic_setting_parallel_for(
        documents.size(),
        nodes,
        8192U,
        worker_count,
        [&](std::size_t index)
        {
            results[index] =
                dump_setting(documents[index], format);
        });
    return results;
}

} // namespace virne::utils
