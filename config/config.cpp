#include "config.h"

#include "yaml_merge.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
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

std::vector<std::string> split_path(
    const std::string& path)
{
    std::vector<std::string> parts;
    std::string token;

    auto emit = [&]() {
        if (token.empty())
        {
            throw std::runtime_error(
                "Invalid empty config path component: " + path);
        }

        parts.push_back(token);
        token.clear();
    };

    for (std::size_t i = 0; i < path.size(); ++i)
    {
        const char c = path[i];

        if (c == '\\')
        {
            if (i + 1 >= path.size())
            {
                throw std::runtime_error(
                    "Invalid trailing escape in config path: " + path);
            }

            token.push_back(path[++i]);
            continue;
        }

        if (c == '.')
        {
            emit();
            continue;
        }

        if (c != '[')
        {
            token.push_back(c);
            continue;
        }

        if (!token.empty())
            emit();

        const auto close = path.find(']', i + 1);

        if (close == std::string::npos)
        {
            throw std::runtime_error(
                "Unclosed index in config path: " + path);
        }

        auto index = trim_copy(
            path.substr(i + 1, close - i - 1));

        if (index.size() >= 2 &&
            ((index.front() == '\'' && index.back() == '\'') ||
             (index.front() == '"' && index.back() == '"')))
        {
            index = index.substr(1, index.size() - 2);
        }

        if (index.empty())
        {
            throw std::runtime_error(
                "Empty index in config path: " + path);
        }

        parts.push_back(index);
        i = close;

        if (i + 1 < path.size() && path[i + 1] == '.')
            ++i;
    }

    if (!token.empty())
        parts.push_back(token);

    if (parts.empty() && !path.empty())
    {
        throw std::runtime_error(
            "Invalid config path: " + path);
    }

    return parts;
}

bool parse_index(
    const std::string& value,
    std::size_t& index)
{
    if (value.empty() ||
        !std::all_of(
            value.begin(),
            value.end(),
            [](unsigned char c) {
                return std::isdigit(c) != 0;
            }))
    {
        return false;
    }

    try
    {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);

        if (consumed != value.size() ||
            parsed > std::numeric_limits<std::size_t>::max())
        {
            return false;
        }

        index = static_cast<std::size_t>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

std::string join_path(
    const std::vector<std::string>& parts)
{
    std::string result;

    for (const auto& part : parts)
    {
        if (!result.empty())
            result.push_back('.');

        result += part;
    }

    return result;
}

std::string child_path(
    const std::string& parent,
    const std::string& child)
{
    if (parent.empty())
        return child;

    return parent + "." + child;
}

std::string scalar_to_string(
    const YAML::Node& node)
{
    if (!node.IsDefined() || node.IsNull())
        return "null";

    if (!node.IsScalar())
    {
        throw std::runtime_error(
            "A map or list interpolation cannot be embedded in a string");
    }

    return node.Scalar();
}

std::pair<std::string, std::string> split_resolver_arguments(
    const std::string& value)
{
    bool single_quote = false;
    bool double_quote = false;
    int square_depth = 0;
    int brace_depth = 0;

    for (std::size_t i = 0; i < value.size(); ++i)
    {
        const char c = value[i];

        if (c == '\\')
        {
            ++i;
            continue;
        }

        if (c == '\'' && !double_quote)
            single_quote = !single_quote;
        else if (c == '"' && !single_quote)
            double_quote = !double_quote;
        else if (!single_quote && !double_quote)
        {
            if (c == '[')
                ++square_depth;
            else if (c == ']')
                --square_depth;
            else if (c == '{')
                ++brace_depth;
            else if (c == '}')
                --brace_depth;
            else if (c == ',' && square_depth == 0 && brace_depth == 0)
            {
                return {
                    trim_copy(value.substr(0, i)),
                    trim_copy(value.substr(i + 1))};
            }
        }
    }

    return {trim_copy(value), {}};
}

std::string absolute_reference_path(
    const std::string& expression,
    const std::string& current_path)
{
    if (expression.empty() || expression.front() != '.')
        return expression;

    std::size_t dots = 0;

    while (dots < expression.size() && expression[dots] == '.')
        ++dots;

    auto base = split_path(current_path);

    if (!base.empty())
        base.pop_back();

    for (std::size_t i = 1; i < dots; ++i)
    {
        if (base.empty())
        {
            throw std::runtime_error(
                "Relative interpolation escapes config root: ${" +
                expression + "}");
        }

        base.pop_back();
    }

    const auto suffix = expression.substr(dots);

    if (!suffix.empty())
    {
        auto suffix_parts = split_path(suffix);
        base.insert(base.end(), suffix_parts.begin(), suffix_parts.end());
    }

    return join_path(base);
}

} // namespace

Config::Config()
    : root_(YAML::NodeType::Map),
      resolver_time_(std::time(nullptr))
{
}

Config::Config(
    const YAML::Node& root)
    : root_(YAML::Clone(root)),
      resolver_time_(std::time(nullptr))
{
}

Config::Config(
    const YAML::Node& root,
    const std::filesystem::path& config_root)
    : root_(YAML::Clone(root)),
      config_root_(config_root),
      resolver_time_(std::time(nullptr))
{
}

YAML::Node Config::root() const
{
    std::unordered_set<std::string> resolving;
    return resolve_node(root_, {}, resolving);
}

bool Config::contains(
    const std::string& path) const
{
    return get_raw_node(path).IsDefined();
}

YAML::Node Config::get_raw_node(
    const std::string& path) const
{
    if (path.empty())
        return YAML::Clone(root_);

    YAML::Node current = YAML::Clone(root_);

    for (const auto& token : split_path(path))
    {
        if (current.IsMap())
        {
            current = current[token];
        }
        else if (current.IsSequence())
        {
            std::size_t index = 0;

            if (!parse_index(token, index) || index >= current.size())
                return YAML::Node(YAML::NodeType::Undefined);

            current = current[index];
        }
        else
        {
            return YAML::Node(YAML::NodeType::Undefined);
        }

        if (!current.IsDefined())
            return YAML::Node(YAML::NodeType::Undefined);
    }

    return YAML::Clone(current);
}

YAML::Node Config::get_node(
    const std::string& path) const
{
    auto raw = get_raw_node(path);

    if (!raw.IsDefined())
        return raw;

    std::unordered_set<std::string> resolving;
    return resolve_node(raw, path, resolving);
}

YAML::Node Config::resolve_expression(
    const std::string& expression,
    const std::string& current_path,
    std::unordered_set<std::string>& resolving) const
{
    const auto work = trim_copy(expression);

    if (work.rfind("now:", 0) == 0)
    {
        const auto format = work.substr(4);
        std::tm tm{};

#if defined(_WIN32)
        localtime_s(&tm, &resolver_time_);
#else
        localtime_r(&resolver_time_, &tm);
#endif

        std::ostringstream output;
        output << std::put_time(&tm, format.c_str());

        if (output.fail())
        {
            throw std::runtime_error(
                "Invalid now resolver format: " + format);
        }

        return YAML::Node(output.str());
    }

    const bool env_resolver =
        work.rfind("oc.env:", 0) == 0 ||
        work.rfind("env:", 0) == 0;

    if (env_resolver)
    {
        const auto colon = work.find(':');
        const auto arguments = split_resolver_arguments(work.substr(colon + 1));

        if (arguments.first.empty())
        {
            throw std::runtime_error(
                "Environment resolver requires a variable name");
        }

        if (const char* value = std::getenv(arguments.first.c_str()))
            return YAML::Node(std::string(value));

        if (!arguments.second.empty())
        {
            try
            {
                return YAML::Load(arguments.second);
            }
            catch (const YAML::Exception& error)
            {
                throw std::runtime_error(
                    "Invalid oc.env default value: " +
                    arguments.second + " (" + error.what() + ")");
            }
        }

        throw std::runtime_error(
            "Environment variable is not set: " + arguments.first);
    }

    if (work.rfind("oc.select:", 0) == 0)
    {
        const auto arguments =
            split_resolver_arguments(work.substr(10));
        const auto selected_path =
            absolute_reference_path(arguments.first, current_path);
        auto selected = get_raw_node(selected_path);

        if (!selected.IsDefined())
        {
            if (arguments.second.empty())
                return YAML::Node(YAML::NodeType::Null);

            try
            {
                selected = YAML::Load(arguments.second);
            }
            catch (const YAML::Exception& error)
            {
                throw std::runtime_error(
                    "Invalid oc.select default value: " +
                    arguments.second + " (" + error.what() + ")");
            }

            // Resolver defaults are parsed as typed YAML values. Keeping the
            // fallback as a value (rather than resolving it again under the
            // caller's path) avoids falsely treating `${oc.select:...,42}` as
            // a self-reference.
            return selected;
        }

        return resolve_node(selected, selected_path, resolving);
    }

    if (work.find(':') != std::string::npos)
    {
        throw std::runtime_error(
            "Unknown interpolation resolver: ${" + work + "}");
    }

    const auto target_path =
        absolute_reference_path(work, current_path);
    auto target = get_raw_node(target_path);

    if (!target.IsDefined())
    {
        throw std::runtime_error(
            "Missing interpolation key: " + target_path);
    }

    return resolve_node(target, target_path, resolving);
}

YAML::Node Config::resolve_node(
    const YAML::Node& node,
    const std::string& current_path,
    std::unordered_set<std::string>& resolving) const
{
    if (!node.IsDefined())
        return YAML::Node(YAML::NodeType::Undefined);

    if (node.IsNull())
        return YAML::Node(YAML::NodeType::Null);

    if (node.IsMap())
    {
        YAML::Node result(YAML::NodeType::Map);

        for (auto it = node.begin(); it != node.end(); ++it)
        {
            const auto key = it->first.as<std::string>();
            result[key] = resolve_node(
                it->second,
                child_path(current_path, key),
                resolving);
        }

        return result;
    }

    if (node.IsSequence())
    {
        YAML::Node result(YAML::NodeType::Sequence);

        for (std::size_t i = 0; i < node.size(); ++i)
        {
            result.push_back(resolve_node(
                node[i],
                child_path(current_path, std::to_string(i)),
                resolving));
        }

        return result;
    }

    if (!node.IsScalar())
        return YAML::Clone(node);

    if (!resolving.insert(current_path).second)
    {
        throw std::runtime_error(
            "Interpolation cycle detected at: " + current_path);
    }

    try
    {
        const auto value = node.Scalar();
        const auto first = value.find("${");

        if (first == std::string::npos)
        {
            resolving.erase(current_path);
            return YAML::Clone(node);
        }

        auto find_end = [&](std::size_t begin) {
            int depth = 1;

            for (std::size_t i = begin + 2; i < value.size(); ++i)
            {
                if (value[i] == '$' &&
                    i + 1 < value.size() &&
                    value[i + 1] == '{')
                {
                    ++depth;
                    ++i;
                }
                else if (value[i] == '}' && --depth == 0)
                {
                    return i;
                }
            }

            return std::string::npos;
        };

        const auto first_end = find_end(first);

        if (first_end == std::string::npos)
        {
            throw std::runtime_error(
                "Unclosed interpolation at: " + current_path);
        }

        if (first == 0 && first_end + 1 == value.size())
        {
            auto result = resolve_expression(
                value.substr(2, first_end - 2),
                current_path,
                resolving);
            resolving.erase(current_path);
            return result;
        }

        std::string result;
        std::size_t cursor = 0;

        while (true)
        {
            const auto start = value.find("${", cursor);

            if (start == std::string::npos)
            {
                result += value.substr(cursor);
                break;
            }

            result += value.substr(cursor, start - cursor);
            const auto end = find_end(start);

            if (end == std::string::npos)
            {
                throw std::runtime_error(
                    "Unclosed interpolation at: " + current_path);
            }

            result += scalar_to_string(resolve_expression(
                value.substr(start + 2, end - start - 2),
                current_path,
                resolving));
            cursor = end + 1;
        }

        resolving.erase(current_path);
        return YAML::Node(result);
    }
    catch (...)
    {
        resolving.erase(current_path);
        throw;
    }
}

void Config::set(
    const std::string& path,
    const YAML::Node& value)
{
    const auto parts = split_path(path);

    if (parts.empty())
    {
        root_ = YAML::Clone(value);
        return;
    }

    root_ = set_node(root_, parts, 0, value);
}

YAML::Node Config::set_node(
    const YAML::Node& node,
    const std::vector<std::string>& parts,
    std::size_t index,
    const YAML::Node& value) const
{
    if (index == parts.size())
        return YAML::Clone(value);

    std::size_t sequence_index = 0;
    const bool use_sequence =
        node.IsSequence() ||
        ((!node.IsDefined() || node.IsNull() || node.IsScalar()) &&
         parse_index(parts[index], sequence_index));

    if (use_sequence)
    {
        if (!parse_index(parts[index], sequence_index))
        {
            throw std::runtime_error(
                "Expected list index, got: " + parts[index]);
        }

        YAML::Node result = node.IsSequence()
            ? YAML::Clone(node)
            : YAML::Node(YAML::NodeType::Sequence);

        while (result.size() <= sequence_index)
            result.push_back(YAML::Node(YAML::NodeType::Null));

        result[sequence_index] = set_node(
            result[sequence_index],
            parts,
            index + 1,
            value);
        return result;
    }

    YAML::Node result = node.IsMap()
        ? YAML::Clone(node)
        : YAML::Node(YAML::NodeType::Map);

    result[parts[index]] = set_node(
        result[parts[index]],
        parts,
        index + 1,
        value);
    return result;
}

void Config::save(
    const std::string& file) const
{
    std::ofstream output(file);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot open file: " + file);
    }

    output << root();
}

YAML::Node Config::get_raw(
    const std::string& path) const
{
    auto raw = get_raw_node(path);

    if (!raw.IsDefined())
        return raw;

    std::unordered_set<std::string> resolving;
    return resolve_node(raw, path, resolving);
}

void Config::merge(
    const YAML::Node& node)
{
    root_ = yaml_merge::merge(root_, node);
}
