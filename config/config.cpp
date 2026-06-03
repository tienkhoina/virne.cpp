#include "config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include "yaml_merge.h"

Config::Config(
    const YAML::Node& root)
{
    root_ = YAML::Clone(root);
}

YAML::Node Config::root() const
{
    return YAML::Clone(root_);
}

bool Config::contains(
    const std::string& path) const
{
    YAML::Node node =
        get_node(path);

    return node.IsDefined()
        && !node.IsNull();
}

YAML::Node Config::get_node(
    const std::string& path) const
{
    if (path.empty())
    {
        return YAML::Clone(root_);
    }

    YAML::Node current =
        YAML::Clone(root_);

    std::stringstream ss(path);
    std::string token;

    while (std::getline(ss, token, '.'))
    {
        if (!current.IsMap())
        {
            return YAML::Node();
        }

        current = current[token];

        if (!current)
        {
            return YAML::Node();
        }
    }

    return YAML::Clone(current);
}

void Config::set(
    const std::string& path,
    const YAML::Node& value)
{
    std::stringstream ss(path);

    std::vector<std::string> parts;
    std::string token;

    while (std::getline(ss, token, '.'))
    {
        parts.push_back(token);
    }

    if (parts.empty())
    {
        return;
    }

    root_ =
        set_node(
            root_,
            parts,
            0,
            value);
}

YAML::Node Config::set_node(
    const YAML::Node& node,
    const std::vector<std::string>& parts,
    size_t index,
    const YAML::Node& value) const
{
    YAML::Node out;

    if (node)
    {
        out = YAML::Clone(node);
    }
    else
    {
        out =
            YAML::Node(
                YAML::NodeType::Map);
    }

    if (!out.IsMap())
    {
        out =
            YAML::Node(
                YAML::NodeType::Map);
    }

    if (index + 1 == parts.size())
    {
        out[parts[index]] =
            YAML::Clone(value);

        return out;
    }

    out[parts[index]] =
        set_node(
            out[parts[index]],
            parts,
            index + 1,
            value);

    return out;
}

void Config::save(
    const std::string& file) const
{
    std::ofstream ofs(file);

    if (!ofs)
    {
        throw std::runtime_error(
            "Cannot open file: " + file);
    }

    ofs << root_;
}

YAML::Node Config::get_raw(
    const std::string& path) const
{
    return get_node(path);
}

void Config::merge(
    const YAML::Node& node)
{
    root_ =
        yaml_merge::merge(
            root_,
            node);
}