#pragma once

#include "../graph.h"

#include <string>

class GmlLoader
{
public:

    static Graph load(
        const std::string& path);
};