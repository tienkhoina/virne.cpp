#include "config_loader.h"

#include "yaml_merge.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DefaultSpec
{
    std::string config_path;
    std::string package_override;
    bool optional = false;
    bool replace_package = false;
    bool self = false;
};

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

bool consume_prefix(
    std::string& value,
    const std::string& prefix)
{
    if (value.rfind(prefix, 0) != 0)
        return false;

    value = trim_copy(value.substr(prefix.size()));
    return true;
}

std::vector<std::string> package_parts(
    const std::string& package)
{
    std::vector<std::string> result;
    std::string token;

    for (char c : package)
    {
        if (c == '.' || c == '/')
        {
            if (!token.empty())
            {
                result.push_back(token);
                token.clear();
            }
        }
        else
        {
            token.push_back(c);
        }
    }

    if (!token.empty())
        result.push_back(token);

    return result;
}

std::string join_package(
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

YAML::Node wrap_package(
    const YAML::Node& node,
    const std::string& package)
{
    auto parts = package_parts(package);
    YAML::Node result = YAML::Clone(node);

    for (auto it = parts.rbegin(); it != parts.rend(); ++it)
    {
        YAML::Node wrapped(YAML::NodeType::Map);
        wrapped[*it] = result;
        result = wrapped;
    }

    return result;
}

YAML::Node read_package(
    const YAML::Node& node,
    const std::string& package)
{
    YAML::Node current = YAML::Clone(node);

    for (const auto& part : package_parts(package))
    {
        if (!current.IsMap() || !current[part].IsDefined())
            return YAML::Node(YAML::NodeType::Undefined);

        current = current[part];
    }

    return YAML::Clone(current);
}

void remove_package(
    YAML::Node& node,
    const std::string& package)
{
    const auto parts = package_parts(package);

    if (parts.empty())
    {
        node = YAML::Node(YAML::NodeType::Map);
        return;
    }

    YAML::Node current = node;

    for (std::size_t i = 0; i + 1 < parts.size(); ++i)
    {
        if (!current.IsMap() || !current[parts[i]].IsDefined())
            return;

        current = current[parts[i]];
    }

    if (current.IsMap())
        current.remove(parts.back());
}

fs::path normalize_file(
    fs::path file)
{
    if (file.extension().empty())
        file += ".yaml";

    std::error_code error;
    auto normalized = fs::weakly_canonical(file, error);

    if (error)
        return fs::absolute(file).lexically_normal();

    return normalized;
}

std::string derived_package(
    const fs::path& file,
    const fs::path& config_root)
{
    std::error_code error;
    const auto relative = fs::relative(file.parent_path(), config_root, error);

    if (error || relative.empty() || relative == ".")
        return {};

    std::vector<std::string> parts;

    for (const auto& part : relative)
    {
        if (part == "..")
        {
            throw std::runtime_error(
                "Config escapes its search root: " + file.string());
        }

        if (part != ".")
            parts.push_back(part.string());
    }

    return join_package(parts);
}

std::pair<std::string, std::string> split_package_override(
    const std::string& value)
{
    const auto at = value.find('@');

    if (at == std::string::npos)
        return {value, {}};

    return {
        trim_copy(value.substr(0, at)),
        trim_copy(value.substr(at + 1))};
}

DefaultSpec parse_scalar_default(
    std::string value)
{
    value = trim_copy(value);
    DefaultSpec spec;

    if (value == "_self_")
    {
        spec.self = true;
        return spec;
    }

    spec.optional = consume_prefix(value, "optional ");
    spec.replace_package = consume_prefix(value, "override ");

    const auto path_and_package = split_package_override(value);
    spec.config_path = path_and_package.first;
    spec.package_override = path_and_package.second;
    return spec;
}

DefaultSpec parse_mapping_default(
    std::string group,
    const std::string& option)
{
    group = trim_copy(group);
    DefaultSpec spec;
    spec.optional = consume_prefix(group, "optional ");
    spec.replace_package = consume_prefix(group, "override ");

    const auto group_and_package = split_package_override(group);
    group = group_and_package.first;
    spec.package_override = group_and_package.second;

    if (group.empty() || option.empty())
    {
        throw std::runtime_error(
            "Invalid defaults group selection: " + group + ": " + option);
    }

    spec.config_path = group;

    if (spec.config_path.back() != '/')
        spec.config_path.push_back('/');

    spec.config_path += option;
    return spec;
}

std::vector<DefaultSpec> parse_default_item(
    const YAML::Node& item)
{
    if (item.IsScalar())
        return {parse_scalar_default(item.as<std::string>())};

    if (!item.IsMap() || item.size() != 1)
    {
        throw std::runtime_error(
            "Each defaults entry must be a string or one-entry map");
    }

    const auto entry = item.begin();
    const auto group = entry->first.as<std::string>();
    // yaml-cpp's iterator proxy is temporary; retaining a reference to its
    // second node is a stack-use-after-scope under ASan.
    const YAML::Node selection = entry->second;

    if (!selection.IsDefined() || selection.IsNull())
        return {};

    std::vector<DefaultSpec> result;

    if (selection.IsSequence())
    {
        for (const auto& option : selection)
        {
            result.push_back(parse_mapping_default(
                group,
                option.as<std::string>()));
        }
    }
    else if (selection.IsScalar())
    {
        result.push_back(parse_mapping_default(
            group,
            selection.as<std::string>()));
    }
    else
    {
        throw std::runtime_error(
            "Defaults group option must be a scalar, list, or null");
    }

    return result;
}

fs::path resolve_default_file(
    const fs::path& including_file,
    const fs::path& config_root,
    std::string config_path)
{
    const bool absolute_group =
        !config_path.empty() && config_path.front() == '/';

    if (absolute_group)
        config_path.erase(config_path.begin());

    auto base = absolute_group
        ? config_root
        : including_file.parent_path();
    auto candidate = normalize_file(base / config_path);

    // Root configs commonly select `group/option`; nested configs commonly
    // select a sibling by bare name. If a relative group does not exist next
    // to the including file, fall back to the search root just like Hydra's
    // config search path.
    if (!fs::exists(candidate) && !absolute_group)
        candidate = normalize_file(config_root / config_path);

    return candidate;
}

std::string effective_package(
    const std::string& package_override,
    const std::string& current_package,
    const std::string& group_package)
{
    if (package_override.empty() || package_override == "_group_")
        return group_package;

    if (package_override == "_here_")
        return current_package;

    if (package_override == "_global_")
        return {};

    return package_override;
}

YAML::Node without_defaults(
    const YAML::Node& current)
{
    YAML::Node body(YAML::NodeType::Map);

    for (auto it = current.begin(); it != current.end(); ++it)
    {
        const auto key = it->first.as<std::string>();

        if (key != "defaults")
            body[key] = YAML::Clone(it->second);
    }

    return body;
}

} // namespace

Config ConfigLoader::load(
    const std::string& main_yaml)
{
    const auto main_file = normalize_file(fs::absolute(main_yaml));

    if (!fs::exists(main_file))
    {
        throw std::runtime_error(
            "Config file does not exist: " + main_file.string());
    }

    const auto config_root = main_file.parent_path();
    std::unordered_set<std::string> active;

    return Config(
        load_recursive(main_file, config_root, {}, active),
        config_root);
}

YAML::Node ConfigLoader::load_recursive(
    const fs::path& file,
    const fs::path& config_root,
    const std::string& package,
    std::unordered_set<std::string>& active)
{
    const auto normalized = normalize_file(file);
    const auto key = normalized.string();

    if (!fs::exists(normalized))
    {
        throw std::runtime_error(
            "Config file does not exist: " + key);
    }

    if (!active.insert(key).second)
    {
        throw std::runtime_error(
            "Config cycle detected: " + key);
    }

    try
    {
        YAML::Node current = YAML::LoadFile(key);

        if (!current.IsMap())
        {
            throw std::runtime_error(
                "Top-level config must be a map: " + key);
        }

        YAML::Node result(YAML::NodeType::Map);
        const auto defaults = current["defaults"];
        bool explicit_self = false;

        auto merge_self = [&]() {
            result = yaml_merge::merge(
                result,
                wrap_package(without_defaults(current), package));
        };

        if (defaults.IsDefined())
        {
            if (!defaults.IsSequence())
            {
                throw std::runtime_error(
                    "defaults must be a list in: " + key);
            }

            for (const auto& item : defaults)
            {
                for (const auto& spec : parse_default_item(item))
                {
                    if (spec.self)
                    {
                        if (explicit_self)
                        {
                            throw std::runtime_error(
                                "_self_ appears more than once in: " + key);
                        }

                        explicit_self = true;
                        merge_self();
                        continue;
                    }

                    const auto child_file = resolve_default_file(
                        normalized,
                        config_root,
                        spec.config_path);

                    if (!fs::exists(child_file))
                    {
                        if (spec.optional)
                            continue;

                        throw std::runtime_error(
                            "Defaults entry not found: " +
                            spec.config_path + " (from " + key + ")");
                    }

                    const auto group_package =
                        derived_package(child_file, config_root);
                    const auto child_package = effective_package(
                        spec.package_override,
                        package,
                        group_package);

                    if (spec.replace_package)
                        remove_package(result, child_package);

                    result = yaml_merge::merge(
                        result,
                        load_recursive(
                            child_file,
                            config_root,
                            child_package,
                            active));
                }
            }
        }

        if (!explicit_self)
            merge_self();

        active.erase(key);
        return result;
    }
    catch (...)
    {
        active.erase(key);
        throw;
    }
}

YAML::Node ConfigLoader::load_group(
    const fs::path& config_root,
    const std::string& group,
    const std::string& option)
{
    if (config_root.empty())
        return YAML::Node(YAML::NodeType::Undefined);

    auto clean_group = group;

    while (!clean_group.empty() && clean_group.front() == '/')
        clean_group.erase(clean_group.begin());

    const auto file = normalize_file(
        config_root / clean_group / option);

    if (!fs::exists(file))
        return YAML::Node(YAML::NodeType::Undefined);

    std::unordered_set<std::string> active;
    const auto package = join_package(package_parts(clean_group));
    auto composed = load_recursive(
        file,
        config_root,
        package,
        active);
    return read_package(composed, package);
}
