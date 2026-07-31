#pragma once

#include "../../../graph/graph_types.h"

#include <vector>

namespace virne::solver::heuristic::detail {

// Reproduces list(set(left).difference(set(right))) for non-negative dense
// graph vertex IDs on the pinned 64-bit CPython 3.10 runtime. This is a
// compatibility seam for two Python solver candidate lists; hot scoring and
// mapping loops continue to consume compact Vertex IDs only.
std::vector<Vertex> cpython310_int_set_difference_order(
    const std::vector<Vertex>& left,
    const std::vector<Vertex>& right);

} // namespace virne::solver::heuristic::detail
