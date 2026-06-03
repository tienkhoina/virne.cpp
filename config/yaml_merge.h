#pragma once

#include <yaml-cpp/yaml.h>

namespace yaml_merge {

YAML::Node merge(
    const YAML::Node& base,
    const YAML::Node& override);

}