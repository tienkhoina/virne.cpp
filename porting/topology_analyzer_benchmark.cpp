#include "topology_analyzer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace controller = virne::core::controller;
namespace network = virne::network;

struct Fingerprint
{
    std::uint64_t checksum = 1469598103934665603ULL;
    std::size_t path_count = 0U;
    std::size_t vertex_count = 0U;
};

void mix_u64(std::uint64_t value, std::uint64_t& hash)
{
    for (std::size_t byte = 0U; byte < 8U; ++byte)
    {
        hash ^= value & 0xFFU;
        hash *= 1099511628211ULL;
        value >>= 8U;
    }
}

Fingerprint fingerprint(
    const std::vector<controller::PreparedTopologyAnalyzer::Paths>& results)
{
    Fingerprint result;
    mix_u64(results.size(), result.checksum);
    for (const auto& request_paths : results)
    {
        mix_u64(request_paths.size(), result.checksum);
        result.path_count += request_paths.size();
        for (const auto& path : request_paths)
        {
            mix_u64(path.size(), result.checksum);
            result.vertex_count += path.size();
            for (Vertex vertex : path)
            {
                mix_u64(vertex, result.checksum);
            }
        }
    }
    return result;
}

std::vector<EdgeEndpoints> make_edges(std::size_t node_count)
{
    std::vector<EdgeEndpoints> edges;
    edges.reserve(node_count - 1U);
    for (std::size_t node = 0U; node + 1U < node_count; ++node)
    {
        edges.emplace_back(node, node + 1U);
    }
    return edges;
}

network::BaseNetworkConstruction construction(
    std::size_t node_count,
    std::vector<EdgeEndpoints> edges)
{
    network::BaseNetworkConstruction result;
    result.incoming_graph.emplace(node_count, std::move(edges));
    return result;
}

std::vector<controller::TopologyPathRequest> make_requests(
    std::size_t count,
    std::size_t node_count)
{
    std::vector<controller::TopologyPathRequest> requests(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        const Vertex source = (index * 37U + 11U) % node_count;
        Vertex target = (index * 91U + 257U) % node_count;
        if (target == source)
        {
            target = (target + 1U) % node_count;
        }
        requests[index].virtual_link = {0U, 1U};
        requests[index].physical_pair = {source, target};
        requests[index].options.method =
            controller::ShortestPathMethod::first_shortest;
    }
    return requests;
}

std::size_t parse_size(const char* text, const char* name)
{
    const std::string value(text);
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size())
    {
        throw std::invalid_argument(std::string(name) + " is invalid");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 5)
        {
            throw std::invalid_argument(
                "usage: topology_analyzer_benchmark workers count warmups repetitions");
        }
        const std::size_t workers = parse_size(argv[1], "workers");
        const std::size_t count = parse_size(argv[2], "count");
        const std::size_t warmups = parse_size(argv[3], "warmups");
        const std::size_t repetitions = parse_size(argv[4], "repetitions");
        if (repetitions == 0U)
        {
            throw std::invalid_argument("repetitions must be positive");
        }

        constexpr std::size_t node_count = 512U;
        network::VirtualNetwork virtual_network(
            construction(2U, {{0U, 1U}}));
        network::PhysicalNetwork physical_network(
            construction(node_count, make_edges(node_count)));
        controller::TopologyAnalyzer analyzer({{}, {}});
        const auto prepared = analyzer.prepare(
            virtual_network,
            physical_network);
        const auto requests = make_requests(count, node_count);

        Fingerprint expected;
        bool has_expected = false;
        for (std::size_t index = 0U; index < warmups; ++index)
        {
            const auto results = prepared.find_shortest_paths_batch(
                requests,
                workers);
            const Fingerprint current = fingerprint(results);
            if (!has_expected)
            {
                expected = current;
                has_expected = true;
            }
            else if (current.checksum != expected.checksum ||
                     current.path_count != expected.path_count ||
                     current.vertex_count != expected.vertex_count)
            {
                throw std::runtime_error("warm-up output changed");
            }
        }

        std::vector<std::uint64_t> samples;
        samples.reserve(repetitions);
        for (std::size_t repetition = 0U;
             repetition < repetitions;
             ++repetition)
        {
            const auto begin = std::chrono::steady_clock::now();
            const auto results = prepared.find_shortest_paths_batch(
                requests,
                workers);
            const auto end = std::chrono::steady_clock::now();
            const Fingerprint current = fingerprint(results);
            if (!has_expected)
            {
                expected = current;
                has_expected = true;
            }
            else if (current.checksum != expected.checksum ||
                     current.path_count != expected.path_count ||
                     current.vertex_count != expected.vertex_count)
            {
                throw std::runtime_error("measured output changed");
            }
            samples.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin).count()));
        }
        std::sort(samples.begin(), samples.end());
        const std::uint64_t median = samples[samples.size() / 2U];
        std::cout
            << "{\"checksum\":" << expected.checksum
            << ",\"path_count\":" << expected.path_count
            << ",\"vertex_count\":" << expected.vertex_count
            << ",\"cpp_median_ns\":" << median
            << ",\"workers\":" << workers << "}\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "topology analyzer benchmark: "
                  << error.what() << '\n';
        return 1;
    }
}
