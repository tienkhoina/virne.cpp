#pragma once

template<typename T>
T Config::get(
    const std::string& path) const
{
    auto node = get_node(path);

    if (!node)
    {
        throw std::runtime_error(
            "Missing config key: " + path);
    }

    return node.as<T>();
}

template<typename T>
T Config::get(
    const std::string& path,
    const T& default_value) const
{
    auto node = get_node(path);

    if (!node)
        return default_value;

    return node.as<T>();
}