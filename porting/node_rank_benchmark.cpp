#include "../virne/solver/rank/node_rank.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace net = virne::network;
namespace attr = virne::network::attribute;
namespace rank = virne::solver::rank;

attr::AttributeFactorySpec resource_spec(std::string name)
{
    attr::AttributeFactorySpec spec{};
    spec.name = std::move(name);
    spec.owner = attr::AttributeOwner::node;
    spec.kind = attr::AttributeKind::resource;
    spec.generative = false;
    spec.restriction = attr::ConstraintRestriction::hard;
    spec.checking_level = attr::CheckingLevel::node;
    return spec;
}

double fixture_value(std::size_t resource, std::size_t node) noexcept
{
    const std::size_t numerator =
        (resource + 1U) * 4U + (node % 1024U);
    return static_cast<double>(numerator) / 4.0;
}

double expected_score(std::size_t resource_count, std::size_t node) noexcept
{
    std::size_t numerator = 0U;
    for (std::size_t resource = 0U; resource < resource_count; ++resource) {
        numerator += (resource + 1U) * 4U + (node % 1024U);
    }
    return static_cast<double>(numerator) / 4.0;
}

net::BaseNetwork make_network(
    std::size_t node_count,
    std::size_t resource_count)
{
    net::BaseNetworkConstruction construction{};
    construction.incoming_graph.emplace(
        node_count, std::vector<EdgeEndpoints>{});
    construction.config.node_attribute_specs.reserve(resource_count);
    for (std::size_t resource = 0U;
         resource < resource_count;
         ++resource) {
        construction.config.node_attribute_specs.push_back(resource_spec(
            "node_resource_" + std::to_string(resource)));
    }

    net::BaseNetwork network(std::move(construction));
    std::vector<net::NodeAttributeDataUpdate> updates;
    updates.reserve(resource_count);
    for (std::size_t resource = 0U;
         resource < resource_count;
         ++resource) {
        const auto binding = network.bind_node_attribute(
            "node_resource_" + std::to_string(resource));
        if (!binding.has_value()) {
            throw std::runtime_error("NodeRank benchmark binding failed");
        }
        net::NodeAttributeDataUpdate update{};
        update.registry_id = binding->registry_id;
        update.layout = net::AttributeDataLayout::dense;
        update.dense_values.reserve(node_count);
        for (std::size_t node = 0U; node < node_count; ++node) {
            update.dense_values.emplace_back(fixture_value(resource, node));
        }
        updates.push_back(std::move(update));
    }
    network.set_node_attrs_data(updates, 1U);
    return network;
}

void fnv_mix(std::uint64_t value, std::uint64_t& hash) noexcept
{
    constexpr std::uint64_t prime = UINT64_C(1099511628211);
    for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
        hash ^= (value >> shift) & UINT64_C(0xff);
        hash *= prime;
    }
}

std::uint64_t score_bits(double value) noexcept
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t validate_and_checksum(
    const rank::NodeRanking& ranking,
    std::size_t node_count,
    std::size_t resource_count)
{
    if (ranking.size() != node_count) {
        throw std::runtime_error("NodeRank benchmark result length mismatch");
    }
    std::uint64_t checksum = UINT64_C(1469598103934665603);
    for (std::size_t node = 0U; node < node_count; ++node) {
        const rank::NodeRankEntry& entry = ranking[node];
        const double expected = expected_score(resource_count, node);
        if (entry.node_id != static_cast<Vertex>(node) ||
            entry.kind != rank::NodeRankValueKind::scalar ||
            score_bits(entry.value) != score_bits(expected) ||
            score_bits(entry.distance) != score_bits(0.0)) {
            throw std::runtime_error("NodeRank benchmark output drift");
        }
        fnv_mix(static_cast<std::uint64_t>(entry.node_id), checksum);
        fnv_mix(score_bits(entry.value), checksum);
    }
    return checksum;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 6) {
            std::cerr << "usage: vne_node_rank_benchmark "
                         "WORKERS WARMUPS SAMPLES NODES RESOURCES\n";
            return 64;
        }
        const std::size_t workers =
            static_cast<std::size_t>(std::stoull(argv[1]));
        const std::size_t warmups =
            static_cast<std::size_t>(std::stoull(argv[2]));
        const std::size_t samples =
            static_cast<std::size_t>(std::stoull(argv[3]));
        const std::size_t node_count =
            static_cast<std::size_t>(std::stoull(argv[4]));
        const std::size_t resource_count =
            static_cast<std::size_t>(std::stoull(argv[5]));
        if (samples == 0U || resource_count == 0U) {
            throw std::invalid_argument(
                "NodeRank benchmark requires samples and resources");
        }

        net::BaseNetwork network = make_network(node_count, resource_count);
        const auto prepared = rank::NodeRanker{}.prepare(network);
        rank::NodeRankOptions options{};
        options.sort = false;
        options.workers = workers;

        std::uint64_t checksum = 0U;
        for (std::size_t warmup = 0U; warmup < warmups; ++warmup) {
            checksum = validate_and_checksum(
                prepared.rank_ffd(options), node_count, resource_count);
        }

        std::vector<std::uint64_t> samples_ns;
        samples_ns.reserve(samples);
        for (std::size_t sample = 0U; sample < samples; ++sample) {
            const auto start = std::chrono::steady_clock::now();
            rank::NodeRanking ranking = prepared.rank_ffd(options);
            const auto stop = std::chrono::steady_clock::now();
            samples_ns.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    stop - start)
                    .count()));
            const std::uint64_t current = validate_and_checksum(
                ranking, node_count, resource_count);
            if (sample != 0U && current != checksum) {
                throw std::runtime_error(
                    "NodeRank benchmark checksum changed across samples");
            }
            checksum = current;
        }
        std::vector<std::uint64_t> ordered = samples_ns;
        std::sort(ordered.begin(), ordered.end());
        const std::uint64_t median_ns = ordered[ordered.size() / 2U];

        std::cout << "{\"workers\":" << workers
                  << ",\"median_ns\":" << median_ns
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < samples_ns.size(); ++index) {
            if (index != 0U) {
                std::cout << ',';
            }
            std::cout << samples_ns[index];
        }
        std::cout << "],\"entries\":" << node_count
                  << ",\"bytes\":" << node_count * 16U
                  << ",\"checksum\":" << checksum << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "node rank benchmark error: " << error.what() << '\n';
        return 2;
    }
}
