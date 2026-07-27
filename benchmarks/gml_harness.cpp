#include "graph/generators/gml_loader.h"
#include "graph/io/graph_saver.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

template <typename Function>
double median_ms(
    Function&& function,
    size_t warmups,
    size_t repetitions,
    uint64_t& checksum)
{
    for (size_t index = 0;
         index < warmups;
         ++index)
    {
        checksum += function();
    }

    std::vector<double> samples;
    samples.reserve(repetitions);
    for (size_t index = 0;
         index < repetitions;
         ++index)
    {
        const auto start = Clock::now();
        checksum += function();
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(
                stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 0)
    {
        return (samples[middle - 1] + samples[middle]) * 0.5;
    }
    return samples[middle];
}

size_t parse_size(
    const char* text,
    const char* name)
{
    const unsigned long long value = std::stoull(text);
    if (value == 0)
    {
        throw std::invalid_argument(
            std::string(name) + " must be positive");
    }
    return static_cast<size_t>(value);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc < 4)
        {
            throw std::invalid_argument(
                "usage: gml_harness --roundtrip INPUT OUTPUT | "
                "--bench INPUT OUTPUT [WARMUPS REPS]");
        }
        const std::string mode = argv[1];
        const std::string input = argv[2];
        const std::string output = argv[3];

        if (mode == "--roundtrip")
        {
            const Graph graph = GmlLoader::load(input);
            GraphSaver::save_gml(graph, output);
            std::cout << "nodes=" << graph.num_nodes() << '\n';
            std::cout << "edges=" << graph.num_edges() << '\n';
            std::cout << "graph_attrs=" << graph.graph_attrs().size() << '\n';
            return 0;
        }
        if (mode != "--bench")
        {
            throw std::invalid_argument("unknown mode: " + mode);
        }

        const size_t warmups =
            argc >= 5 ? parse_size(argv[4], "warmups") : 3;
        const size_t repetitions =
            argc >= 6 ? parse_size(argv[5], "repetitions") : 15;
        const Graph prepared = GmlLoader::load(input);
        uint64_t checksum = 0;

        const double load_ms = median_ms(
            [&]() -> uint64_t
            {
                const Graph graph = GmlLoader::load(input);
                return graph.num_nodes() + graph.num_edges() +
                       graph.graph_attrs().size();
            },
            warmups,
            repetitions,
            checksum);

        const double save_ms = median_ms(
            [&]() -> uint64_t
            {
                GraphSaver::save_gml(prepared, output);
                return std::filesystem::file_size(output);
            },
            warmups,
            repetitions,
            checksum);

        std::cout << std::setprecision(17);
        std::cout << "load_ms=" << load_ms << '\n';
        std::cout << "save_ms=" << save_ms << '\n';
        std::cout << "nodes=" << prepared.num_nodes() << '\n';
        std::cout << "edges=" << prepared.num_edges() << '\n';
        std::cout << "checksum=" << checksum << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "gml_harness: " << error.what() << '\n';
        return 1;
    }
}
