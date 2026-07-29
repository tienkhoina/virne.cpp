#include "topology_generator.h"

#include "generators/topology_generators.h"
#include "nx/connectivity.h"
#include "random_context.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace
{

std::size_t checked_node_count(
    std::int64_t value)
{
    if (value < 1)
    {
        throw std::invalid_argument(
            "num_nodes must be >= 1.");
    }

    const auto unsigned_value =
        static_cast<std::uint64_t>(value);
    if (unsigned_value >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
    {
        throw std::overflow_error(
            "num_nodes does not fit size_t");
    }

    return static_cast<std::size_t>(unsigned_value);
}

std::size_t python_range_size(
    std::int64_t value)
{
    if (value < 0)
    {
        throw std::invalid_argument(
            "Negative number of nodes not valid: " +
            std::to_string(value));
    }

    if (value == 0)
    {
        return 0;
    }

    const auto unsigned_value =
        static_cast<std::uint64_t>(value);
    if (unsigned_value >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
    {
        throw std::overflow_error(
            "grid dimension does not fit size_t");
    }

    return static_cast<std::size_t>(unsigned_value);
}

Graph python_grid_2d_graph(
    std::size_t rows,
    std::size_t columns)
{
    if (rows != 0 &&
        columns >
            std::numeric_limits<std::size_t>::max() / rows)
    {
        throw std::overflow_error(
            "grid_2d_graph node count overflows size_t");
    }

    Graph graph;
    const std::size_t node_count = rows * columns;
    for (std::size_t index = 0;
         index < node_count;
         ++index)
    {
        graph.add_node();
    }

    if (rows == 0 || columns == 0)
    {
        return graph;
    }

    const auto vertex =
        [columns](std::size_t row, std::size_t column)
        {
            return static_cast<Vertex>(
                row * columns + column);
        };

    // NetworkX 3.4 inserts every vertical grid edge before inserting any
    // horizontal edge.  Its EdgeView alone does not expose this distinction,
    // but neighbors(), BFS tie-breaking and downstream solver traversal do.
    for (std::size_t row = 0;
         row < rows - 1;
         ++row)
    {
        for (std::size_t column = 0;
             column < columns;
             ++column)
        {
            graph.add_edge(
                vertex(row, column),
                vertex(row + 1, column));
        }
    }

    for (std::size_t row = 0;
         row < rows;
         ++row)
    {
        for (std::size_t column = 0;
             column < columns - 1;
             ++column)
        {
            graph.add_edge(
                vertex(row, column),
                vertex(row, column + 1));
        }
    }

    return graph;
}

template <typename Factory>
Graph connected_like_python(
    Factory&& factory,
    const std::optional<std::size_t>& max_attempts)
{
    if (max_attempts.has_value() &&
        *max_attempts == 0)
    {
        throw std::invalid_argument(
            "connected retry requires max_attempts > 0");
    }

    std::size_t attempt = 0;
    while (!max_attempts.has_value() ||
           attempt < *max_attempts)
    {
        Graph graph = factory();
        ++attempt;
        if (nx::is_connected(graph))
        {
            return graph;
        }
    }

    throw std::runtime_error(
        "Unable to generate a connected graph within max_attempts");
}

Graph generate_resolved(
    virne::network::TopologyType type,
    std::size_t node_count,
    const virne::network::TopologyOptions& options,
    PyRandom& random)
{
    using virne::network::TopologyType;

    switch (type)
    {
    case TopologyType::Path:
        return nx::path_graph(node_count);

    case TopologyType::Star:
        return nx::star_graph(node_count - 1);

    case TopologyType::Grid2D:
        if (!options.m.has_value() ||
            !options.n.has_value())
        {
            throw std::invalid_argument(
                "'grid_2d' type requires 'm' and 'n' keyword arguments.");
        }

        return python_grid_2d_graph(
            python_range_size(*options.m),
            python_range_size(*options.n));

    case TopologyType::Waxman:
        // Preserve the Python implementation's positional-call quirk:
        // nx.waxman_graph(n, wm_alpha, wm_beta) binds wm_alpha to NetworkX's
        // beta parameter and wm_beta to its alpha parameter.
        return connected_like_python(
            [&]
            {
                return nx::waxman_graph(
                    node_count,
                    options.wm_alpha,
                    options.wm_beta,
                    random);
            },
            options.max_attempts);

    case TopologyType::Random:
        return connected_like_python(
            [&]
            {
                return nx::erdos_renyi_graph(
                    node_count,
                    options.random_prob,
                    random);
            },
            options.max_attempts);
    }

    throw std::invalid_argument(
        "TopologyType value is not implemented.");
}

std::size_t resolved_worker_count(
    std::size_t requested,
    const std::vector<virne::network::TopologyRequest>& requests)
{
    const std::size_t task_count = requests.size();
    if (task_count == 0)
    {
        return 0;
    }

    if (requested == 0)
    {
#if defined(__linux__)
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        {
            requested = 0;
            for (std::size_t cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            {
                if (CPU_ISSET(static_cast<int>(cpu), &affinity))
                {
                    ++requested;
                }
            }
        }
#endif
        if (requested == 0)
        {
            requested = std::max<std::size_t>(
                1, std::thread::hardware_concurrency());
        }

        // Final post-enum/AttrId sweeps on the eight-CPU reference cpuset
        // selected five workers for homogeneous path/grid batches and six for
        // random/star/Waxman. Mixed batches use the balanced six-worker cap.
        std::size_t tuned_limit = 6;
        const auto first_type = requests.front().type;
        const bool homogeneous = std::all_of(
            requests.begin(),
            requests.end(),
            [first_type](const virne::network::TopologyRequest& request)
            {
                return request.type == first_type;
            });
        if (homogeneous &&
            (first_type == virne::network::TopologyType::Path ||
             first_type == virne::network::TopologyType::Grid2D))
        {
            tuned_limit = 5;
        }
        requested = std::min(tuned_limit, requested);
    }

    return std::min(requested, task_count);
}

} // namespace

namespace virne::network
{

TopologyType topology_type_from_string(
    std::string_view type)
{
    if (type == "path")
    {
        return TopologyType::Path;
    }
    if (type == "star")
    {
        return TopologyType::Star;
    }
    if (type == "grid_2d")
    {
        return TopologyType::Grid2D;
    }
    if (type == "waxman")
    {
        return TopologyType::Waxman;
    }
    if (type == "random")
    {
        return TopologyType::Random;
    }

    throw std::invalid_argument(
        "Graph type '" +
        std::string(type) +
        "' is not implemented.");
}

Graph TopologyGenerator::generate(
    TopologyType type,
    std::int64_t num_nodes,
    const TopologyOptions& options)
{
    return generate(
        type,
        num_nodes,
        options,
        global_py_random());
}

Graph TopologyGenerator::generate(
    TopologyType type,
    std::int64_t num_nodes,
    const TopologyOptions& options,
    PyRandom& random)
{
    return generate_resolved(
        type,
        checked_node_count(num_nodes),
        options,
        random);
}

Graph TopologyGenerator::generate(
    std::string_view type,
    std::int64_t num_nodes,
    const TopologyOptions& options)
{
    // Python validates num_nodes before dispatching on type. Preserve that
    // error order while resolving the dynamic string exactly once.
    const std::size_t node_count =
        checked_node_count(num_nodes);
    return generate_resolved(
        topology_type_from_string(type),
        node_count,
        options,
        global_py_random());
}

Graph TopologyGenerator::generate(
    std::string_view type,
    std::int64_t num_nodes,
    const TopologyOptions& options,
    PyRandom& random)
{
    const std::size_t node_count =
        checked_node_count(num_nodes);
    return generate_resolved(
        topology_type_from_string(type),
        node_count,
        options,
        random);
}

std::vector<Graph> TopologyGenerator::generate_batch(
    const std::vector<TopologyRequest>& requests,
    std::size_t worker_count)
{
    std::vector<Graph> results(requests.size());
    std::vector<std::exception_ptr> errors(requests.size());

    const std::size_t workers =
        resolved_worker_count(
            worker_count,
            requests);
    if (workers == 0)
    {
        return results;
    }

    const auto run_index =
        [&](std::size_t index)
        {
            try
            {
                PyRandom random(requests[index].seed);
                results[index] = generate(
                    requests[index].type,
                    requests[index].num_nodes,
                    requests[index].options,
                    random);
            }
            catch (...)
            {
                errors[index] =
                    std::current_exception();
            }
        };

    if (workers == 1)
    {
        for (std::size_t index = 0;
             index < requests.size();
             ++index)
        {
            run_index(index);
        }
    }
    else
    {
        std::atomic<std::size_t> next_index{0};
        const auto run_dynamic =
            [&]
            {
                for (;;)
                {
                    const std::size_t index =
                        next_index.fetch_add(
                            1, std::memory_order_relaxed);
                    if (index >= requests.size())
                    {
                        return;
                    }
                    run_index(index);
                }
            };

        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (std::size_t worker = 0;
             worker < workers;
             ++worker)
        {
            threads.emplace_back(run_dynamic);
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
    }

    // Select the earliest failing request, not the first worker to finish.
    // This makes failure behavior deterministic across worker counts.
    for (const auto& error : errors)
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }

    return results;
}

} // namespace virne::network
