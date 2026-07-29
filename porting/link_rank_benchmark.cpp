#include "rank/link_rank.h"

#include <algorithm>
#include <array>
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

namespace attribute = virne::network::attribute;
namespace network = virne::network;
namespace rank = virne::solver::rank;

constexpr std::size_t edge_count = 131072U;
constexpr std::size_t resource_count = 8U;
constexpr std::size_t warmup_count = 1U;
constexpr std::size_t sample_count = 3U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Digest {
    std::uint64_t checksum = fnv_offset;
    std::uint64_t bytes = 0U;
    std::size_t entries = 0U;

    void append_byte(std::uint8_t value)
    {
        checksum ^= value;
        checksum *= fnv_prime;
        ++bytes;
    }

    void append_u64(std::uint64_t value)
    {
        for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
            append_byte(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void append_double(double value)
    {
        std::uint64_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value), "double width drift");
        std::memcpy(&bits, &value, sizeof(bits));
        append_u64(bits);
    }
};

attribute::AttributeFactorySpec resource_spec(std::size_t index)
{
    attribute::AttributeFactorySpec result;
    result.name = "resource_" + std::to_string(index);
    result.owner = attribute::AttributeOwner::link;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = attribute::CheckingLevel::link;
    return result;
}

std::int64_t resource_value(std::size_t resource, std::size_t edge)
{
    const std::uint64_t multiplier =
        static_cast<std::uint64_t>(resource * 2U + 3U);
    const std::uint64_t value =
        (static_cast<std::uint64_t>(edge) * multiplier +
         static_cast<std::uint64_t>(resource * 17U)) %
        std::uint64_t{100003U};
    return static_cast<std::int64_t>(value) - std::int64_t{50000};
}

network::BaseNetwork make_network()
{
    std::vector<EdgeEndpoints> edges;
    edges.reserve(edge_count);
    for (std::size_t index = 0U; index < edge_count; ++index) {
        edges.emplace_back(
            static_cast<Vertex>(index),
            static_cast<Vertex>(index + 1U));
    }

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        edge_count + 1U,
        std::move(edges));
    construction.config.link_attribute_specs.reserve(resource_count);
    for (std::size_t resource = 0U; resource < resource_count; ++resource) {
        construction.config.link_attribute_specs.push_back(
            resource_spec(resource));
    }

    network::BaseNetwork result(std::move(construction));
    std::vector<network::LinkAttributeDataUpdate> updates;
    updates.reserve(resource_count);
    for (std::size_t resource = 0U; resource < resource_count; ++resource) {
        network::LinkAttributeDataUpdate update;
        update.registry_id = static_cast<attribute::AttributeRegistryId>(
            resource);
        update.layout = network::AttributeDataLayout::dense;
        update.dense_values.reserve(edge_count);
        for (std::size_t edge = 0U; edge < edge_count; ++edge) {
            update.dense_values.emplace_back(resource_value(resource, edge));
        }
        updates.push_back(std::move(update));
    }
    result.set_link_attrs_data(updates, 1U);
    return result;
}

Digest digest(const rank::LinkRanking& ranking)
{
    Digest result;
    result.entries = ranking.size();
    for (const rank::LinkRankEntry& entry : ranking) {
        result.append_u64(static_cast<std::uint64_t>(entry.source));
        result.append_u64(static_cast<std::uint64_t>(entry.target));
        result.append_double(entry.value);
    }
    return result;
}

void require_same(const Digest& lhs, const Digest& rhs)
{
    if (lhs.checksum != rhs.checksum || lhs.bytes != rhs.bytes ||
        lhs.entries != rhs.entries) {
        throw std::runtime_error("link rank benchmark output drift");
    }
}

double median(std::array<double, sample_count> values)
{
    std::sort(values.begin(), values.end());
    return values[sample_count / 2U];
}

void benchmark()
{
    network::BaseNetwork network = make_network();
    const rank::PreparedLinkRanker prepared = rank::LinkRanker{}.prepare(network);
    const Digest baseline = digest(prepared.rank_ffd({true, 1U}));
    if (baseline.entries != edge_count ||
        baseline.bytes != edge_count * std::uint64_t{24U}) {
        throw std::runtime_error("link rank benchmark baseline shape drift");
    }

    std::cout << std::setprecision(17);
    for (const std::size_t workers : {1U, 2U, 8U}) {
        for (std::size_t warmup = 0U; warmup < warmup_count; ++warmup) {
            require_same(
                digest(prepared.rank_ffd({true, workers})), baseline);
        }

        std::array<double, sample_count> samples{};
        for (std::size_t sample = 0U; sample < sample_count; ++sample) {
            const auto begin = std::chrono::steady_clock::now();
            const rank::LinkRanking ranking =
                prepared.rank_ffd({true, workers});
            const auto end = std::chrono::steady_clock::now();
            samples[sample] =
                std::chrono::duration<double, std::milli>(end - begin).count();
            require_same(digest(ranking), baseline);
        }

        std::cout
            << "workers=" << workers
            << "\tmedian_ms=" << median(samples)
            << "\tchecksum=" << baseline.checksum
            << "\tbytes=" << baseline.bytes
            << "\tentries=" << baseline.entries
            << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "benchmark") {
            benchmark();
            return 0;
        }
        throw std::invalid_argument(
            "usage: vne_link_rank_benchmark benchmark");
    } catch (const std::exception& error) {
        std::cerr << "link rank benchmark: " << error.what() << '\n';
        return 1;
    }
}
