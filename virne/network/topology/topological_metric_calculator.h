#pragma once

#include "graph.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace virne::network
{

// Fixed metric discriminants are carried as compact IDs. Production graph
// loops never compare or hash metric-name strings.
enum class TopologicalMetricKind : std::uint8_t
{
    Degree,
    Closeness,
    Eigenvector,
    Betweenness,
};

// NumPy exposes every result as an (N, 1) float32 array. The C++ equivalent
// stores that single column contiguously and makes the fixed second dimension
// explicit without adding per-row objects or indirection.
struct MetricColumn
{
    std::vector<float> values;

    std::size_t rows() const noexcept
    {
        return values.size();
    }

    static constexpr std::size_t columns() noexcept
    {
        return 1;
    }

    bool empty() const noexcept
    {
        return values.empty();
    }

    float& operator[](std::size_t row) noexcept
    {
        return values[row];
    }

    const float& operator[](std::size_t row) const noexcept
    {
        return values[row];
    }
};

struct TopologicalMetrics
{
    std::optional<MetricColumn> node_degree_centrality;
    std::optional<MetricColumn> node_closeness_centrality;
    std::optional<MetricColumn> node_eigenvector_centrality;
    std::optional<MetricColumn> node_betweenness_centrality;
};

// Static calculate() follows the Python defaults: all four metrics and
// float32 min/max normalization. worker_count=0 selects the measured automatic
// policy (eight workers for either expensive metric alone, seven when both are
// requested on the measured eight-CPU cpuset); one forces the deterministic
// sequential implementation.
struct TopologicalMetricOptions
{
    bool degree = true;
    bool closeness = true;
    bool eigenvector = true;
    bool betweenness = true;
    bool normalize = true;
    std::size_t worker_count = 0;

    // The Python constructor differs from calculate(): it enables degree only.
    static TopologicalMetricOptions degree_only() noexcept;
};

class TopologicalMetricCalculator
{
public:
    using CachedMetrics = std::shared_ptr<TopologicalMetrics>;

    explicit TopologicalMetricCalculator(
        const Graph& network,
        const TopologicalMetricOptions& options =
            TopologicalMetricOptions::degree_only());

    // These are fixed public fields in the Python class as well. The caller
    // must keep network alive for the lifetime of the calculator.
    const Graph& network;
    TopologicalMetrics metrics;

    static TopologicalMetrics calculate(
        const Graph& network,
        const TopologicalMetricOptions& options = {});

    // Cache keys are intentionally dynamic. Each function performs exactly
    // one hash-table operation at this boundary; no cache string reaches a
    // graph or metric loop. Shared ownership preserves Python-like identity.
    static void add_to_cache(
        std::string cache_key,
        CachedMetrics metrics);

    static CachedMetrics get_from_cache(
        std::string_view cache_key);

    static void clear_cache();
};

} // namespace virne::network
