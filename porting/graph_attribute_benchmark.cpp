#include "attribute/graph_attribute.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;

struct Result {
    std::uint64_t elapsed_ns = 0U;
    std::uint64_t checksum = 0U;
    std::size_t output_bytes = 0U;
};

std::uint64_t fnv1a(const unsigned char* data, std::size_t size) noexcept {
    std::uint64_t result = UINT64_C(14695981039346656037);
    for (std::size_t index = 0U; index < size; ++index) {
        result ^= data[index];
        result *= UINT64_C(1099511628211);
    }
    return result;
}

double double_from_bits(std::uint64_t bits) noexcept {
    double result = 0.0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double input_value(std::size_t index) noexcept {
    if ((index % 4093U) == 0U) {
        return double_from_bits(
            UINT64_C(0x7ff8000000000000) |
            static_cast<std::uint64_t>((index & 0xffffU) + 1U));
    }
    if ((index % 257U) == 0U) {
        return -0.0;
    }
    const auto residue = static_cast<std::int64_t>(index % 1009U);
    return static_cast<double>(residue - std::int64_t{504}) * 0.25;
}

attribute::BaseAttributeSpec graph_spec() {
    attribute::BaseAttributeSpec result;
    result.name = "roundtrip_value";
    result.owner = attribute::AttributeOwner::graph;
    result.kind = attribute::AttributeKind::status;
    return result;
}

Result independent_roundtrip(std::size_t count, std::size_t workers) {
    std::vector<Graph> graphs;
    graphs.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        graphs.emplace_back();
        if ((index & 1U) != 0U) {
            static_cast<void>(graphs.back().attr_id("padding"));
        }
        if ((index % 3U) == 0U) {
            static_cast<void>(graphs.back().attr_id("other_padding"));
        }
    }

    const attribute::GraphAttribute item(graph_spec());
    std::vector<attribute::GraphAttributeMutableSlot> mutable_slots;
    std::vector<attribute::GraphAttributeConstSlot> const_slots;
    std::vector<AttrValue> input;
    mutable_slots.reserve(count);
    const_slots.reserve(count);
    input.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        const auto binding = item.bind(graphs[index]);
        mutable_slots.push_back({&graphs[index].graph_attrs(), binding});
        const_slots.push_back({&graphs[index].graph_attrs(), binding});
        input.emplace_back(input_value(index));
    }

    const auto begin = std::chrono::steady_clock::now();
    item.set_data_batch(mutable_slots, input, workers);
    const std::vector<AttrValue> output = item.get_data_batch(const_slots, workers);
    const auto end = std::chrono::steady_clock::now();

    std::vector<double> raw;
    raw.reserve(output.size());
    for (const AttrValue& value : output) {
        const auto* floating = std::get_if<double>(&value);
        if (floating == nullptr) {
            throw std::runtime_error("graph benchmark output lane drift");
        }
        raw.push_back(*floating);
    }
    return Result{
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count()),
        fnv1a(
            reinterpret_cast<const unsigned char*>(raw.data()),
            raw.size() * sizeof(double)),
        raw.size() * sizeof(double)};
}

std::size_t parse_size(const char* value, const char* name) {
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (value[consumed] != '\0' ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<std::size_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: vne_graph_attribute_benchmark COUNT WORKERS");
        }
        const std::size_t count = parse_size(argv[1], "count");
        const std::size_t workers = parse_size(argv[2], "workers");
        const Result result = independent_roundtrip(count, workers);
        std::cout << "protocol=1\n"
                  << "kind=independent_roundtrip\n"
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "type_tag=double\n"
                  << "bits=raw64\n"
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph_attribute_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
