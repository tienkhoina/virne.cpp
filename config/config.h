#pragma once

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

class Config
{
public:
    Config() = default;

    explicit Config(
        const YAML::Node& root);

    template<typename T>
    T get(
        const std::string& path) const;

    template<typename T>
    T get(
        const std::string& path,
        const T& default_value) const;

    bool contains(
        const std::string& path) const;

    void set(
        const std::string& path,
        const YAML::Node& value);

    void save(
        const std::string& file) const;

    void merge(
        const YAML::Node& node);

    YAML::Node root() const;

    YAML::Node get_raw(
        const std::string& path) const;

private:
    YAML::Node root_;

    YAML::Node get_node(
        const std::string& path) const;

    YAML::Node set_node(
        const YAML::Node& node,
        const std::vector<std::string>& parts,
        size_t index,
        const YAML::Node& value) const;
};

#include "config.tpp"