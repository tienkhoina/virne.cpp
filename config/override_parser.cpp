#include "override_parser.h"

#include "config_loader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {

std::string trim_copy(
    const std::string& value)
{
    auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](unsigned char c) {
            return std::isspace(c) != 0;
        });
    auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](unsigned char c) {
            return std::isspace(c) != 0;
        }).base();

    if (first >= last)
        return {};

    return std::string(first, last);
}

std::string lower_copy(
    std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return value;
}

std::string remove_numeric_separators(
    std::string value)
{
    value.erase(
        std::remove(value.begin(), value.end(), '_'),
        value.end());
    return value;
}

bool parse_integer(
    const std::string& input,
    long long& result)
{
    auto value = remove_numeric_separators(input);

    if (value.empty())
        return false;

    bool negative = false;
    std::size_t offset = 0;

    if (value.front() == '+' || value.front() == '-')
    {
        negative = value.front() == '-';
        offset = 1;
    }

    if (offset == value.size())
        return false;

    int base = 10;

    if (value.size() >= offset + 2 && value[offset] == '0')
    {
        const char prefix =
            static_cast<char>(std::tolower(
                static_cast<unsigned char>(value[offset + 1])));

        if (prefix == 'x')
            base = 16;
        else if (prefix == 'o')
            base = 8;
        else if (prefix == 'b')
            base = 2;

        if (base != 10)
        {
            value.erase(offset, 2);

            if (value.size() == offset)
                return false;
        }
    }

    const auto digits = value.substr(offset);

    if (!std::all_of(
            digits.begin(),
            digits.end(),
            [base](unsigned char c) {
                if (std::isdigit(c))
                    return c - '0' < base;

                const auto lower =
                    static_cast<unsigned char>(std::tolower(c));
                return base == 16 && lower >= 'a' && lower <= 'f';
            }))
    {
        return false;
    }

    try
    {
        std::size_t consumed = 0;
        auto magnitude = std::stoull(digits, &consumed, base);

        if (consumed != digits.size())
            return false;

        const auto positive_limit =
            static_cast<unsigned long long>(
                std::numeric_limits<long long>::max());
        const auto negative_limit = positive_limit + 1ULL;

        if ((!negative && magnitude > positive_limit) ||
            (negative && magnitude > negative_limit))
        {
            return false;
        }

        if (negative && magnitude == negative_limit)
            result = std::numeric_limits<long long>::min();
        else
            result = negative
                ? -static_cast<long long>(magnitude)
                : static_cast<long long>(magnitude);

        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_floating(
    const std::string& input,
    double& result)
{
    auto value = remove_numeric_separators(input);

    if (value.empty())
        return false;

    const auto lower = lower_copy(value);
    const bool looks_float =
        value.find_first_of(".eE") != std::string::npos ||
        lower == "inf" || lower == "+inf" || lower == "-inf" ||
        lower == ".inf" || lower == "+.inf" || lower == "-.inf" ||
        lower == "nan" || lower == ".nan";

    if (!looks_float)
        return false;

    auto normalized = lower;

    if (normalized == ".inf")
        normalized = "inf";
    else if (normalized == "+.inf")
        normalized = "+inf";
    else if (normalized == "-.inf")
        normalized = "-inf";
    else if (normalized == ".nan")
        normalized = "nan";

    errno = 0;
    char* end = nullptr;
    result = std::strtod(normalized.c_str(), &end);

    return end == normalized.c_str() + normalized.size() &&
        errno != ERANGE;
}

YAML::Node parse_value(
    const std::string& input)
{
    const auto value = trim_copy(input);

    if (value.empty())
        return YAML::Node(std::string());

    const auto lower = lower_copy(value);

    if (lower == "true")
        return YAML::Node(true);

    if (lower == "false")
        return YAML::Node(false);

    if (lower == "null" || value == "~")
        return YAML::Node(YAML::NodeType::Null);

    if ((value.front() == '[' && value.back() == ']') ||
        (value.front() == '{' && value.back() == '}') ||
        (value.size() >= 2 &&
         ((value.front() == '\'' && value.back() == '\'') ||
          (value.front() == '"' && value.back() == '"'))))
    {
        try
        {
            auto parsed = YAML::Load(value);

            if (!parsed.IsDefined())
                return YAML::Node(std::string());

            return parsed;
        }
        catch (const YAML::Exception& error)
        {
            throw std::runtime_error(
                "Invalid YAML override value '" + value +
                "': " + error.what());
        }
    }

    long long integer = 0;

    if (parse_integer(value, integer))
        return YAML::Node(integer);

    double floating = 0.0;

    if (parse_floating(value, floating))
        return YAML::Node(floating);

    return YAML::Node(value);
}

std::string group_package(
    std::string group)
{
    while (!group.empty() && group.front() == '/')
        group.erase(group.begin());

    std::replace(group.begin(), group.end(), '/', '.');
    return group;
}

std::string scalar_string(
    const std::string& value)
{
    auto parsed = parse_value(value);

    if (!parsed.IsScalar())
        return {};

    return parsed.as<std::string>();
}

} // namespace

namespace override_parser {

void apply(
    Config& cfg,
    const std::string& expr)
{
    bool create_only = false;
    bool create_or_replace = false;
    std::string work = trim_copy(expr);

    if (work.rfind("++", 0) == 0)
    {
        create_or_replace = true;
        work = work.substr(2);
    }
    else if (work.rfind('+', 0) == 0)
    {
        create_only = true;
        work = work.substr(1);
    }

    const auto equals = work.find('=');

    if (equals == std::string::npos)
    {
        throw std::runtime_error(
            "Invalid override (expected key=value): " + expr);
    }

    const auto key = trim_copy(work.substr(0, equals));
    const auto value = work.substr(equals + 1);

    if (key.empty())
    {
        throw std::runtime_error(
            "Invalid override with an empty key: " + expr);
    }

    // A matching <config-root>/<group>/<option>.yaml takes precedence over a
    // scalar assignment. This is Hydra's config-group selection syntax and is
    // deliberately independent of the process working directory.
    const auto option = scalar_string(value);
    auto selected = option.empty()
        ? YAML::Node(YAML::NodeType::Undefined)
        : ConfigLoader::load_group(cfg.config_root_, key, option);

    auto clean_group_key = key;

    while (!clean_group_key.empty() && clean_group_key.front() == '/')
        clean_group_key.erase(clean_group_key.begin());

    auto group_directory = cfg.config_root_ / clean_group_key;
    const bool known_group =
        !cfg.config_root_.empty() &&
        std::filesystem::is_directory(group_directory);

    if (known_group && !selected.IsDefined())
    {
        throw std::runtime_error(
            "Config group option not found: " + key + "=" + option);
    }

    if (selected.IsDefined())
    {
        const auto package = group_package(key);
        const bool exists = cfg.contains(package);

        if (create_only && exists)
        {
            throw std::runtime_error(
                "Config group already exists: " + package);
        }

        if (!create_only && !create_or_replace && !exists)
        {
            throw std::runtime_error(
                "Unknown config group: " + package);
        }

        cfg.set(package, selected);
        return;
    }

    const bool exists = cfg.contains(key);

    if (create_only && exists)
    {
        throw std::runtime_error(
            "Key already exists: " + key);
    }

    if (!create_only && !create_or_replace && !exists)
    {
        throw std::runtime_error(
            "Unknown key: " + key);
    }

    cfg.set(key, parse_value(value));
}

} // namespace override_parser
