#pragma once

#include "config.h"

#include <filesystem>
#include <unordered_set>

class ConfigLoader {
public:
    static Config load(
        const std::string& main_yaml);

private:
    static YAML::Node load_recursive(
        const std::filesystem::path& file,
        std::unordered_set<std::string>& visited);
};