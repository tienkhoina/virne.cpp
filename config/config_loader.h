#pragma once

#include "config.h"

#include <filesystem>
#include <unordered_set>

namespace override_parser {

void apply(
    Config& cfg,
    const std::string& expr);

}

class ConfigLoader {
public:
    static Config load(
        const std::string& main_yaml);

private:
    friend void override_parser::apply(
        Config& cfg,
        const std::string& expr);

    static YAML::Node load_recursive(
        const std::filesystem::path& file,
        const std::filesystem::path& config_root,
        const std::string& package,
        std::unordered_set<std::string>& active);

    static YAML::Node load_group(
        const std::filesystem::path& config_root,
        const std::string& group,
        const std::string& option);
};
