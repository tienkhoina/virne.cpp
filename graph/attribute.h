#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

using AttrValue =
    std::variant<
        int64_t,
        double,
        bool,
        std::string>;

using AttrMap =
    std::unordered_map<
        std::string,
        AttrValue>;