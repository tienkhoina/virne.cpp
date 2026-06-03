#include "override_parser.h"

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace override_parser {

static bool is_group_override(
    const std::string& key,
    const std::string& value)
{
    fs::path p =
        fs::path("setting")
        / key
        / (value + ".yaml");

    return fs::exists(p);
}

static YAML::Node parse_value(
    const std::string& value)
{
    if (value == "true" ||
        value == "True")
    {
        return YAML::Node(true);
    }

    if (value == "false" ||
        value == "False")
    {
        return YAML::Node(false);
    }

    if (value == "null")
    {
        return YAML::Node();
    }

    try
    {
        size_t pos;

        int v =
            std::stoi(
                value,
                &pos);

        if (pos == value.size())
        {
            return YAML::Node(v);
        }
    }
    catch (...)
    {
    }

    try
    {
        size_t pos;

        double v =
            std::stod(
                value,
                &pos);

        if (pos == value.size())
        {
            return YAML::Node(v);
        }
    }
    catch (...)
    {
    }

    return YAML::Node(value);
}

void apply(
    Config& cfg,
    const std::string& expr)
{
    bool create_only = false;
    bool create_or_replace = false;

    std::string work = expr;

    if (work.rfind("++", 0) == 0)
    {
        create_or_replace = true;
        work = work.substr(2);
    }
    else if (work.rfind("+", 0) == 0)
    {
        create_only = true;
        work = work.substr(1);
    }

    auto pos =
        work.find('=');

    if (pos == std::string::npos)
    {
        throw std::runtime_error(
            "Invalid override: " +
            expr);
    }

    std::string key =
        work.substr(
            0,
            pos);

    std::string value =
        work.substr(
            pos + 1);

    bool exists =
        cfg.contains(key);

    if (create_only &&
        is_group_override(
            key,
            value))
    {
        YAML::Node loaded =
            YAML::LoadFile(
                (
                    fs::path("setting")
                    / key
                    / (value + ".yaml")
                ).string());

        YAML::Node wrapped;
        wrapped[key] = loaded;

        cfg.merge(wrapped);

        return;
    }

    if (create_only && exists)
    {
        throw std::runtime_error(
            "Key already exists: " +
            key);
    }

    if (!create_only &&
        !create_or_replace &&
        !exists)
    {
        throw std::runtime_error(
            "Unknown key: " +
            key);
    }

    cfg.set(
        key,
        parse_value(value));
}

}