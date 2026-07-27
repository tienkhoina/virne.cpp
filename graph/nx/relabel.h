#pragma once

#include "../graph.h"

#include <cstdint>
#include <optional>
#include <string>

namespace nx
{

Graph convert_node_labels_to_integers(
    const Graph& graph,
    int64_t first_label = 0,
    const std::string& ordering = "default",
    std::optional<std::string> label_attribute =
        std::nullopt);

DiGraph convert_node_labels_to_integers(
    const DiGraph& graph,
    int64_t first_label = 0,
    const std::string& ordering = "default",
    std::optional<std::string> label_attribute =
        std::nullopt);

} // namespace nx
