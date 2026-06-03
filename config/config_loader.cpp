#include "config_loader.h"
#include "yaml_merge.h"

namespace fs = std::filesystem;

Config ConfigLoader::load(
    const std::string& main_yaml)
{
    std::unordered_set<std::string> visited;

    return Config(
        load_recursive(
            fs::absolute(main_yaml),
            visited));
}

YAML::Node ConfigLoader::load_recursive(
    const fs::path& file,
    std::unordered_set<std::string>& visited)
{
    auto abs = fs::absolute(file);
    auto key = abs.string();

    if (visited.find(key) != visited.end())
    {
        throw std::runtime_error(
            "Config cycle detected: " + key);
    }

    visited.insert(key);

    YAML::Node current =
        YAML::LoadFile(key);

    YAML::Node result;

    auto defaults =
        current["defaults"];

    if (defaults)
    {
        for (const auto& item : defaults)
        {
            auto value =
                item.as<std::string>();

            if (value == "_self_")
            {
                YAML::Node filtered;

                for (auto it = current.begin();
                     it != current.end();
                     ++it)
                {
                    auto k =
                        it->first.as<std::string>();

                    if (k == "defaults")
                        continue;

                    filtered[k] =
                        it->second;
                }

                result =
                    yaml_merge::merge(
                        result,
                        filtered);

                continue;
            }

            auto child =
                file.parent_path() /
                (value + ".yaml");

            auto sub =
                load_recursive(
                    child,
                    visited);

            auto pos =
                value.find('/');

            if (pos != std::string::npos)
            {
                auto group =
                    value.substr(
                        0,
                        pos);

                YAML::Node wrapped;
                wrapped[group] = sub;

                result =
                    yaml_merge::merge(
                        result,
                        wrapped);
            }
            else
            {
                result =
                    yaml_merge::merge(
                        result,
                        sub);
            }
        }
    }
    else
    {
        result =
            yaml_merge::merge(
                result,
                current);
    }

    visited.erase(key);

    return result;
}