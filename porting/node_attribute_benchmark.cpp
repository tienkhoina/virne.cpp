#include "attribute/node_attribute.h"

#include "numpy_random_state.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

std::uint64_t fnv_append(
    std::uint64_t hash,
    const void* data,
    std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0U; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::size_t parse_size(const char* text) {
    const std::string value(text);
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size()
        || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid size argument");
    }
    return static_cast<std::size_t>(parsed);
}

virne::network::attribute::BaseAttributeSpec node_spec(std::string name) {
    virne::network::attribute::BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = virne::network::attribute::AttributeOwner::node;
    result.kind = virne::network::attribute::AttributeKind::status;
    return result;
}

struct Result {
    std::int64_t elapsed_ns = 0;
    std::uint64_t checksum = 0U;
    std::size_t output_bytes = 0U;
    std::uint64_t next_bits = 0U;
};

Result dense_roundtrip(std::size_t count, std::size_t workers) {
    namespace attribute = virne::network::attribute;
    Graph graph(count, std::vector<EdgeEndpoints>{});
    const attribute::NodeAttribute value(node_spec("load"));
    const auto binding = value.bind(graph);
    std::vector<AttrValue> input;
    input.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        input.emplace_back(static_cast<std::int64_t>(index % 1009U) - 504);
    }

    const auto started = std::chrono::steady_clock::now();
    value.set_data_dense(graph, input, binding, workers);
    const std::vector<AttrValue> output = value.get_data(graph, binding, workers);
    const auto stopped = std::chrono::steady_clock::now();

    std::uint64_t checksum = 14695981039346656037ULL;
    for (const AttrValue& item : output) {
        const auto integer = std::get<std::int64_t>(item);
        checksum = fnv_append(checksum, &integer, sizeof(integer));
    }
    return Result{
        std::chrono::duration_cast<std::chrono::nanoseconds>(stopped - started)
            .count(),
        checksum,
        output.size() * sizeof(std::int64_t),
        0U};
}

Result position_generation(
    std::size_t count,
    std::size_t workers,
    std::uint32_t seed) {
    namespace attribute = virne::network::attribute;
    namespace utils = virne::utils;
    attribute::NodePositionSpec spec;
    spec.generative = true;
    spec.distribution.kind = utils::DistributionKind::uniform;
    spec.distribution.low = utils::DatasetScalar{-2.0};
    spec.distribution.high = utils::DatasetScalar{3.0};
    spec.dtype = utils::DatasetValueKind::floating;
    spec.minimum_radius = -0.25;
    spec.maximum_radius = 0.75;
    const attribute::NodePositionAttribute position(std::move(spec));
    NumpyRandomState rng(seed);

    const auto started = std::chrono::steady_clock::now();
    const std::vector<attribute::NodePositionValue> output =
        position.generate_positions(
            attribute::NetworkCardinality{count, 0U}, rng, workers);
    const auto stopped = std::chrono::steady_clock::now();

    std::uint64_t checksum = 14695981039346656037ULL;
    for (const auto& item : output) {
        const double x = std::get<double>(item.x);
        const double y = std::get<double>(item.y);
        checksum = fnv_append(checksum, &x, sizeof(x));
        checksum = fnv_append(checksum, &y, sizeof(y));
        checksum = fnv_append(checksum, &item.radius, sizeof(item.radius));
    }
    return Result{
        std::chrono::duration_cast<std::chrono::nanoseconds>(stopped - started)
            .count(),
        checksum,
        output.size() * 3U * sizeof(double),
        double_bits(rng.random())};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            throw std::invalid_argument(
                "usage: node_attribute_benchmark KIND COUNT WORKERS SEED");
        }
        const std::string kind(argv[1]);
        const std::size_t count = parse_size(argv[2]);
        const std::size_t workers = parse_size(argv[3]);
        const std::size_t seed_value = parse_size(argv[4]);
        if (seed_value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("seed outside uint32 range");
        }
        const Result result = kind == "dense_roundtrip"
            ? dense_roundtrip(count, workers)
            : kind == "position"
                ? position_generation(
                    count, workers, static_cast<std::uint32_t>(seed_value))
                : throw std::invalid_argument("unknown benchmark kind");
        std::cout << "protocol=1\n"
                  << "kind=" << kind << '\n'
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "next_bits=" << result.next_bits << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
