#pragma once

#include "graph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class PyRandom;

namespace virne::network
{

enum class TopologyType : std::uint8_t
{
    Path,
    Star,
    Grid2D,
    Waxman,
    Random,
};

// Dynamic configuration strings are resolved once at the API boundary. Hot
// generation/batch paths carry TopologyType directly.
TopologyType topology_type_from_string(
    std::string_view type);

// C++ representation of the keyword arguments accepted by
// TopologyGenerator.generate in the Python implementation.  An unset
// max_attempts preserves Python's unbounded connected-graph retry.  Setting it
// enables the explicit C++ safety extension without changing the compatible
// default path.
struct TopologyOptions
{
    std::optional<std::int64_t> m;
    std::optional<std::int64_t> n;

    double wm_alpha = 0.5;
    double wm_beta = 0.2;
    double random_prob = 0.5;

    std::optional<std::size_t> max_attempts;
};

struct TopologyRequest
{
    TopologyType type = TopologyType::Path;
    std::int64_t num_nodes = 1;
    TopologyOptions options;
    std::uint64_t seed = 0;
};

class TopologyGenerator
{
public:
    // Fixed-field overloads avoid all string dispatch for native callers.
    static Graph generate(
        TopologyType type,
        std::int64_t num_nodes,
        const TopologyOptions& options = {});

    static Graph generate(
        TopologyType type,
        std::int64_t num_nodes,
        const TopologyOptions& options,
        PyRandom& random);

    // Uses Virne's process-wide Python-compatible random stream, matching
    // NetworkX seed=None.  The stream is mutable and callers must serialize
    // access when they use this overload from multiple threads.
    static Graph generate(
        std::string_view type,
        std::int64_t num_nodes,
        const TopologyOptions& options = {});

    // Explicit stream overload used by deterministic callers and the Python
    // differential oracle.  Failed connected candidates consume the stream
    // before the next retry, exactly as NetworkX does.
    static Graph generate(
        std::string_view type,
        std::int64_t num_nodes,
        const TopologyOptions& options,
        PyRandom& random);

    // Deterministic parallel extension.  Every request owns its seed, so
    // workers never share RNG state; result order and exception selection
    // follow input order regardless of scheduling. Zero selects the measured
    // family-specific width (five for homogeneous path/grid, six otherwise),
    // always bounded by request count and process CPU affinity.
    static std::vector<Graph> generate_batch(
        const std::vector<TopologyRequest>& requests,
        std::size_t worker_count = 0);
};

} // namespace virne::network
