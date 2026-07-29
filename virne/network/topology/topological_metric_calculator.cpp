#include "topological_metric_calculator.h"

#include "nx/centrality.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace
{

using virne::network::MetricColumn;
using virne::network::TopologicalMetricKind;
using virne::network::TopologicalMetrics;

constexpr std::size_t kEigenvectorMaxIterations = 100;
constexpr double kEigenvectorTolerance = 1.0e-6;

class DeterministicExecutor
{
public:
    DeterministicExecutor() = default;
    DeterministicExecutor(const DeterministicExecutor&) = delete;
    DeterministicExecutor& operator=(const DeterministicExecutor&) = delete;

    ~DeterministicExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    template <typename Function>
    void run(std::size_t worker_count, Function&& function)
    {
        if (worker_count <= 1)
        {
            function(0);
            return;
        }

        std::lock_guard<std::mutex> execution_lock(execution_mutex_);
        ensure_workers(worker_count - 1);
        std::vector<std::exception_ptr> failures(worker_count);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            task_ = [callable = std::forward<Function>(function), &failures](
                        std::size_t worker_index) mutable
            {
                try
                {
                    callable(worker_index);
                }
                catch (...)
                {
                    failures[worker_index] = std::current_exception();
                }
            };
            active_background_workers_ = worker_count - 1;
            completed_background_workers_ = 0;
            ++generation_;
        }
        ready_.notify_all();

        task_(0);

        std::unique_lock<std::mutex> state_lock(state_mutex_);
        finished_.wait(state_lock, [this]
        {
            return completed_background_workers_ ==
                   active_background_workers_;
        });
        task_ = {};
        state_lock.unlock();

        for (const auto& failure : failures)
        {
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
    }

private:
    void ensure_workers(std::size_t count)
    {
        while (workers_.size() < count)
        {
            const std::size_t worker_index = workers_.size() + 1;
            std::size_t initial_generation = 0;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                initial_generation = generation_;
            }
            workers_.emplace_back(
                [this, worker_index, initial_generation]
                {
                    worker_loop(worker_index, initial_generation);
                });
        }
    }

    void worker_loop(
        std::size_t worker_index,
        std::size_t seen_generation)
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        for (;;)
        {
            ready_.wait(lock, [this, &seen_generation]
            {
                return generation_ != seen_generation || stopping_;
            });
            if (stopping_)
            {
                return;
            }
            seen_generation = generation_;
            if (worker_index > active_background_workers_)
            {
                continue;
            }

            const auto* task = &task_;
            lock.unlock();
            (*task)(worker_index);
            lock.lock();
            ++completed_background_workers_;
            if (completed_background_workers_ ==
                active_background_workers_)
            {
                finished_.notify_one();
            }
        }
    }

    std::mutex execution_mutex_;
    std::mutex state_mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::function<void(std::size_t)> task_;
    std::size_t generation_ = 0;
    std::size_t active_background_workers_ = 0;
    std::size_t completed_background_workers_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

DeterministicExecutor& metric_executor()
{
    static DeterministicExecutor executor;
    return executor;
}

std::size_t available_worker_width()
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        std::size_t available = 0;
        for (std::size_t cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (CPU_ISSET(static_cast<int>(cpu), &affinity))
            {
                ++available;
            }
        }
        if (available != 0)
        {
            return available;
        }
    }
#endif
    return std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
}

std::size_t resolved_worker_count(
    std::size_t requested,
    std::size_t source_count,
    TopologicalMetricKind kind,
    bool combined_expensive_metrics)
{
    if (source_count == 0)
    {
        return 1;
    }

    if (requested == 0)
    {
        const std::size_t minimum_parallel_nodes =
            kind == TopologicalMetricKind::Betweenness ? 32 : 64;
        if (source_count < minimum_parallel_nodes)
        {
            return 1;
        }

        // The final five-round/31-sample sweep selected eight workers when
        // either source-parallel metric runs alone and seven when both run in
        // one calculate() call. Always respect process affinity.
        const std::size_t tuned_limit =
            combined_expensive_metrics ? 7 : 8;
        requested = std::min(tuned_limit, available_worker_width());
    }

    return std::max<std::size_t>(
        1, std::min(requested, source_count));
}

MetricColumn to_float32(const nx::NodeScores& scores)
{
    MetricColumn column;
    column.values.resize(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index)
    {
        column.values[index] = static_cast<float>(scores[index]);
    }
    return column;
}

void normalize_float32(MetricColumn& column)
{
    if (column.values.empty())
    {
        return;
    }

    float minimum = column.values.front();
    float maximum = column.values.front();
    bool has_nan = std::isnan(minimum);
    for (std::size_t index = 1; index < column.values.size(); ++index)
    {
        const float value = column.values[index];
        has_nan = has_nan || std::isnan(value);
        if (value < minimum)
        {
            minimum = value;
        }
        if (value > maximum)
        {
            maximum = value;
        }
    }

    if (has_nan)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::fill(column.values.begin(), column.values.end(), nan);
        return;
    }

    if (maximum == minimum)
    {
        std::fill(column.values.begin(), column.values.end(), 0.0F);
        return;
    }

    const float denominator = maximum - minimum;
    for (float& value : column.values)
    {
        value = (value - minimum) / denominator;
    }
}

// CPython 3.10's variadic math.hypot uses a scaled compensated sum. Matching
// that operation shape keeps NetworkX's power-iteration stopping point and
// float32 conversion stable without calling Python at runtime.
double python_hypot(const std::vector<double>& values)
{
    double maximum = 0.0;
    bool has_nan = false;
    for (double value : values)
    {
        value = std::abs(value);
        if (std::isinf(value))
        {
            return std::numeric_limits<double>::infinity();
        }
        has_nan = has_nan || std::isnan(value);
        if (value > maximum)
        {
            maximum = value;
        }
    }
    if (has_nan)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (maximum == 0.0 || values.size() <= 1)
    {
        return maximum;
    }

    double compensated_sum = 1.0;
    double fractional_error = 0.0;
    for (double value : values)
    {
        const double scaled = value / maximum;
        const double squared = scaled * scaled;
        const double previous = compensated_sum;
        compensated_sum += squared;
        fractional_error +=
            (previous - compensated_sum) + squared;
    }
    return maximum * std::sqrt(
        compensated_sum - 1.0 + fractional_error);
}

nx::NodeScores eigenvector_centrality_python(const Graph& graph)
{
    const std::size_t node_count = graph.num_nodes();
    if (node_count == 0)
    {
        throw std::domain_error(
            "cannot compute centrality for the null graph");
    }

    nx::NodeScores values(
        node_count,
        1.0 / static_cast<double>(node_count));
    for (std::size_t iteration = 0;
         iteration < kEigenvectorMaxIterations;
         ++iteration)
    {
        const nx::NodeScores previous = values;
        values = previous;

        // NetworkX iterates nodes and their adjacency dictionaries. This is
        // deliberately not a global EdgeView loop: addition order affects the
        // iterative floating-point result.
        for (Vertex node = 0; node < node_count; ++node)
        {
            const double contribution = previous[node];
            for (const auto& edge : graph.neighbors_fast(node))
            {
                values[edge.get_target()] += contribution;
            }
        }

        double norm = python_hypot(values);
        if (norm == 0.0)
        {
            norm = 1.0;
        }
        for (double& value : values)
        {
            value /= norm;
        }

        double error = 0.0;
        for (std::size_t node = 0; node < node_count; ++node)
        {
            error += std::abs(values[node] - previous[node]);
        }
        if (error <
            static_cast<double>(node_count) * kEigenvectorTolerance)
        {
            return values;
        }
    }

    throw std::runtime_error(
        "power iteration failed to converge within 100 iterations");
}

struct ClosenessWorkspace
{
    explicit ClosenessWorkspace(std::size_t node_count)
        : distance(node_count, -1), queue(node_count)
    {
    }

    std::vector<std::int64_t> distance;
    std::vector<Vertex> queue;
};

double closeness_for_source(
    const Graph& graph,
    Vertex source,
    ClosenessWorkspace& workspace)
{
    std::fill(workspace.distance.begin(), workspace.distance.end(), -1);
    std::size_t head = 0;
    std::size_t tail = 0;
    workspace.distance[source] = 0;
    workspace.queue[tail++] = source;

    while (head < tail)
    {
        const Vertex node = workspace.queue[head++];
        const std::int64_t next_distance = workspace.distance[node] + 1;
        for (const auto& edge : graph.neighbors_fast(node))
        {
            const Vertex neighbor = edge.get_target();
            if (workspace.distance[neighbor] < 0)
            {
                workspace.distance[neighbor] = next_distance;
                workspace.queue[tail++] = neighbor;
            }
        }
    }

    std::int64_t total_distance = 0;
    std::size_t reachable = 0;
    for (std::int64_t distance : workspace.distance)
    {
        if (distance >= 0)
        {
            ++reachable;
            total_distance += distance;
        }
    }

    const std::size_t node_count = graph.num_nodes();
    if (total_distance <= 0 || node_count <= 1)
    {
        return 0.0;
    }

    const double reachable_without_source =
        static_cast<double>(reachable - 1);
    double value = reachable_without_source /
                   static_cast<double>(total_distance);
    const double scale = reachable_without_source /
                         static_cast<double>(node_count - 1);
    value *= scale;
    return value;
}

nx::NodeScores closeness_centrality_python(
    const Graph& graph,
    std::size_t requested_workers,
    bool combined_expensive_metrics)
{
    const std::size_t node_count = graph.num_nodes();
    nx::NodeScores scores(node_count, 0.0);
    if (node_count == 0)
    {
        return scores;
    }

    const std::size_t worker_count = resolved_worker_count(
        requested_workers,
        node_count,
        TopologicalMetricKind::Closeness,
        combined_expensive_metrics);
    if (worker_count == 1)
    {
        ClosenessWorkspace workspace(node_count);
        for (Vertex source = 0; source < node_count; ++source)
        {
            scores[source] = closeness_for_source(
                graph, source, workspace);
        }
        return scores;
    }

    std::atomic<std::size_t> next_source{0};
    metric_executor().run(worker_count, [&](std::size_t)
    {
        ClosenessWorkspace workspace(node_count);
        for (;;)
        {
            const std::size_t source = next_source.fetch_add(
                1, std::memory_order_relaxed);
            if (source >= node_count)
            {
                return;
            }
            scores[source] = closeness_for_source(
                graph,
                static_cast<Vertex>(source),
                workspace);
        }
    });
    return scores;
}

struct BetweennessWorkspace
{
    explicit BetweennessWorkspace(std::size_t node_count)
        : distance(node_count, -1),
          sigma(node_count, 0.0),
          predecessors(node_count),
          stack(),
          queue(node_count),
          delta(node_count, 0.0)
    {
        stack.reserve(node_count);
    }

    std::vector<std::int64_t> distance;
    std::vector<double> sigma;
    std::vector<std::vector<Vertex>> predecessors;
    std::vector<Vertex> stack;
    std::vector<Vertex> queue;
    std::vector<double> delta;
};

void betweenness_for_source(
    const Graph& graph,
    Vertex source,
    BetweennessWorkspace& workspace)
{
    std::fill(workspace.distance.begin(), workspace.distance.end(), -1);
    std::fill(workspace.sigma.begin(), workspace.sigma.end(), 0.0);
    std::fill(workspace.delta.begin(), workspace.delta.end(), 0.0);
    for (auto& predecessors : workspace.predecessors)
    {
        predecessors.clear();
    }
    workspace.stack.clear();

    std::size_t head = 0;
    std::size_t tail = 0;
    workspace.sigma[source] = 1.0;
    workspace.distance[source] = 0;
    workspace.queue[tail++] = source;

    while (head < tail)
    {
        const Vertex node = workspace.queue[head++];
        workspace.stack.push_back(node);
        const std::int64_t next_distance = workspace.distance[node] + 1;
        const double path_count = workspace.sigma[node];
        for (const auto& edge : graph.neighbors_fast(node))
        {
            const Vertex neighbor = edge.get_target();
            if (workspace.distance[neighbor] < 0)
            {
                workspace.queue[tail++] = neighbor;
                workspace.distance[neighbor] = next_distance;
            }
            if (workspace.distance[neighbor] == next_distance)
            {
                workspace.sigma[neighbor] += path_count;
                workspace.predecessors[neighbor].push_back(node);
            }
        }
    }

    while (!workspace.stack.empty())
    {
        const Vertex node = workspace.stack.back();
        workspace.stack.pop_back();
        const double coefficient =
            (1.0 + workspace.delta[node]) / workspace.sigma[node];
        for (Vertex predecessor : workspace.predecessors[node])
        {
            workspace.delta[predecessor] +=
                workspace.sigma[predecessor] * coefficient;
        }
    }
}

nx::NodeScores betweenness_centrality_python(
    const Graph& graph,
    std::size_t requested_workers,
    bool combined_expensive_metrics)
{
    const std::size_t node_count = graph.num_nodes();
    nx::NodeScores scores(node_count, 0.0);
    if (node_count == 0)
    {
        return scores;
    }

    const std::size_t worker_count = resolved_worker_count(
        requested_workers,
        node_count,
        TopologicalMetricKind::Betweenness,
        combined_expensive_metrics);
    if (worker_count == 1)
    {
        BetweennessWorkspace workspace(node_count);
        for (Vertex source = 0; source < node_count; ++source)
        {
            betweenness_for_source(graph, source, workspace);
            for (Vertex node = 0; node < node_count; ++node)
            {
                if (node != source)
                {
                    scores[node] += workspace.delta[node];
                }
            }
        }
    }
    else
    {
        // Keep enough source rows to amortize executor barriers without
        // allowing the exact-order reduction buffer to grow without bound.
        // Every row is reduced later in Python source order, so worker
        // scheduling cannot alter a single floating-point addition.
        constexpr std::size_t contribution_budget_bytes =
            64U * 1024U * 1024U;
        const std::size_t rows_per_block =
            node_count > contribution_budget_bytes / sizeof(double)
            ? 1
            : std::min(
                  node_count,
                  std::max<std::size_t>(
                      1,
                      contribution_budget_bytes /
                          (node_count * sizeof(double))));

        std::vector<BetweennessWorkspace> workspaces;
        workspaces.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            workspaces.emplace_back(node_count);
        }
        std::vector<double> contributions(
            rows_per_block * node_count);

        for (std::size_t block_begin = 0;
             block_begin < node_count;
             block_begin += rows_per_block)
        {
            const std::size_t block_rows = std::min(
                rows_per_block, node_count - block_begin);
            std::atomic<std::size_t> next_row{0};

            metric_executor().run(worker_count, [&](std::size_t worker_index)
            {
                BetweennessWorkspace& workspace = workspaces[worker_index];
                for (;;)
                {
                    const std::size_t row = next_row.fetch_add(
                        1, std::memory_order_relaxed);
                    if (row >= block_rows)
                    {
                        return;
                    }
                    const std::size_t source = block_begin + row;
                    betweenness_for_source(
                        graph,
                        static_cast<Vertex>(source),
                        workspace);
                    double* output = contributions.data() + row * node_count;
                    for (std::size_t node = 0; node < node_count; ++node)
                    {
                        output[node] = node == source
                            ? 0.0
                            : workspace.delta[node];
                    }
                }
            });

            for (std::size_t row = 0; row < block_rows; ++row)
            {
                const double* input =
                    contributions.data() + row * node_count;
                for (std::size_t node = 0; node < node_count; ++node)
                {
                    scores[node] += input[node];
                }
            }
        }
    }

    if (node_count > 2)
    {
        const double denominator = static_cast<double>(
            (node_count - 1) * (node_count - 2));
        const double scale = 1.0 / denominator;
        for (double& value : scores)
        {
            value *= scale;
        }
    }
    return scores;
}

void maybe_normalize(
    MetricColumn& column,
    bool normalize)
{
    if (normalize)
    {
        normalize_float32(column);
    }
}

using CacheMap = std::unordered_map<
    std::string,
    virne::network::TopologicalMetricCalculator::CachedMetrics>;

CacheMap& metric_cache()
{
    static CacheMap cache;
    return cache;
}

std::shared_mutex& metric_cache_mutex()
{
    static std::shared_mutex mutex;
    return mutex;
}

} // namespace

namespace virne::network
{

TopologicalMetricOptions TopologicalMetricOptions::degree_only() noexcept
{
    TopologicalMetricOptions options;
    options.closeness = false;
    options.eigenvector = false;
    options.betweenness = false;
    return options;
}

TopologicalMetricCalculator::TopologicalMetricCalculator(
    const Graph& network_value,
    const TopologicalMetricOptions& options)
    : network(network_value),
      metrics(calculate(network_value, options))
{
}

TopologicalMetrics TopologicalMetricCalculator::calculate(
    const Graph& graph,
    const TopologicalMetricOptions& options)
{
    TopologicalMetrics result;
    const bool combined_expensive_metrics =
        options.closeness && options.betweenness;

    // Preserve Python's public call/error order. Expensive source-independent
    // work inside closeness and betweenness uses the deterministic executor.
    if (options.degree)
    {
        MetricColumn values = to_float32(nx::degree_centrality(graph));
        maybe_normalize(values, options.normalize);
        result.node_degree_centrality = std::move(values);
    }
    if (options.closeness)
    {
        MetricColumn values = to_float32(
            closeness_centrality_python(
                graph,
                options.worker_count,
                combined_expensive_metrics));
        maybe_normalize(values, options.normalize);
        result.node_closeness_centrality = std::move(values);
    }
    if (options.eigenvector)
    {
        MetricColumn values = to_float32(
            eigenvector_centrality_python(graph));
        maybe_normalize(values, options.normalize);
        result.node_eigenvector_centrality = std::move(values);
    }
    if (options.betweenness)
    {
        MetricColumn values = to_float32(
            betweenness_centrality_python(
                graph,
                options.worker_count,
                combined_expensive_metrics));
        maybe_normalize(values, options.normalize);
        result.node_betweenness_centrality = std::move(values);
    }

    return result;
}

void TopologicalMetricCalculator::add_to_cache(
    std::string cache_key,
    CachedMetrics metrics_value)
{
    if (!metrics_value)
    {
        throw std::invalid_argument("cached metrics must not be null");
    }
    std::unique_lock<std::shared_mutex> lock(metric_cache_mutex());
    metric_cache().insert_or_assign(
        std::move(cache_key), std::move(metrics_value));
}

TopologicalMetricCalculator::CachedMetrics
TopologicalMetricCalculator::get_from_cache(
    std::string_view cache_key)
{
    // C++17 unordered_map has no heterogeneous lookup. Materialize once at
    // this boundary, then perform one hash-table lookup.
    const std::string resolved_key(cache_key);
    std::shared_lock<std::shared_mutex> lock(metric_cache_mutex());
    const auto found = metric_cache().find(resolved_key);
    return found == metric_cache().end() ? nullptr : found->second;
}

void TopologicalMetricCalculator::clear_cache()
{
    std::unique_lock<std::shared_mutex> lock(metric_cache_mutex());
    metric_cache().clear();
}

} // namespace virne::network
