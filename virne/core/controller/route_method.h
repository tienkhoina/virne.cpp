#pragma once
#include <cstdint>

namespace virne::core::controller {
enum class ShortestPathMethod : std::uint8_t {
    first_shortest, k_shortest, k_shortest_length, all_shortest, bfs_shortest, available_shortest,
};
}  // namespace virne::core::controller
