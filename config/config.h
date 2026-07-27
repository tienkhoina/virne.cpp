#pragma once

#include <yaml-cpp/yaml.h>

#include <ctime>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

class ConfigLoader;
class Config;

namespace override_parser {

void apply(
    Config& cfg,
    const std::string& expr);

}

class Config
{
public:
    Config();

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
    friend class ConfigLoader;
    friend void override_parser::apply(
        Config& cfg,
        const std::string& expr);

    YAML::Node root_;
    std::filesystem::path config_root_;
    std::time_t resolver_time_ = 0;

    Config(
        const YAML::Node& root,
        const std::filesystem::path& config_root);

    YAML::Node get_node(
        const std::string& path) const;

    YAML::Node get_raw_node(
        const std::string& path) const;

    YAML::Node resolve_node(
        const YAML::Node& node,
        const std::string& current_path,
        std::unordered_set<std::string>& resolving) const;

    YAML::Node resolve_expression(
        const std::string& expression,
        const std::string& current_path,
        std::unordered_set<std::string>& resolving) const;

    YAML::Node set_node(
        const YAML::Node& node,
        const std::vector<std::string>& parts,
        size_t index,
        const YAML::Node& value) const;
};

#include "config.tpp"
