#include "attribute/link_attribute.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
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

std::uint64_t fnv1a(
    const unsigned char* data,
    std::size_t size) noexcept {
    std::uint64_t result = UINT64_C(14695981039346656037);
    for (std::size_t index = 0U; index < size; ++index) {
        result ^= data[index];
        result *= UINT64_C(1099511628211);
    }
    return result;
}

std::vector<EdgeEndpoints> path_edges(std::size_t count) {
    std::vector<EdgeEndpoints> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        result.push_back({index, index + 1U});
    }
    return result;
}

attribute::BaseAttributeSpec dense_spec() {
    attribute::BaseAttributeSpec result;
    result.name = "load";
    result.owner = attribute::AttributeOwner::link;
    result.kind = attribute::AttributeKind::resource;
    return result;
}

Result dense_roundtrip(
    std::size_t count,
    std::size_t workers) {
    Graph graph(count + 1U, path_edges(count));
    const attribute::LinkAttribute item(dense_spec());
    const auto binding = item.bind(graph);
    std::vector<AttrValue> input(count);
    for (std::size_t index = 0U; index < count; ++index) {
        input[index] = static_cast<std::int64_t>(index % 1009U);
    }

    const auto begin = std::chrono::steady_clock::now();
    item.set_data_dense(graph, input, binding, workers);
    const std::vector<AttrValue> output = item.get_data(graph, binding, workers);
    const auto end = std::chrono::steady_clock::now();

    std::vector<std::int64_t> raw(output.size());
    for (std::size_t index = 0U; index < output.size(); ++index) {
        raw[index] = std::get<std::int64_t>(output[index]);
    }
    return Result{
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count()),
        fnv1a(
            reinterpret_cast<const unsigned char*>(raw.data()),
            raw.size() * sizeof(std::int64_t)),
        raw.size() * sizeof(std::int64_t)};
}

Result position_latency(
    std::size_t count,
    std::size_t workers) {
    Graph graph(count + 1U, path_edges(count));
    const AttrId position_id = graph.attr_id("pos");
    for (std::size_t index = 0U; index <= count; ++index) {
        graph.node_attrs(index).set(
            position_id,
            make_attr_list({
                AttrValue{static_cast<std::int64_t>(index % 997U)},
                AttrValue{static_cast<std::int64_t>((index * 17U) % 991U)}}));
    }
    attribute::LinkLatencySpec spec;
    spec.generative = true;
    spec.generation = attribute::LatencyGenerationKind::position;
    spec.minimum = 0.0;
    spec.maximum = 1.0;
    const attribute::LinkLatencyAttribute latency(spec);

    const auto begin = std::chrono::steady_clock::now();
    const auto binding = latency.resolve_position_binding(graph);
    const std::vector<double> output =
        latency.generate_from_position(graph, binding, workers);
    const auto end = std::chrono::steady_clock::now();

    return Result{
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count()),
        fnv1a(
            reinterpret_cast<const unsigned char*>(output.data()),
            output.size() * sizeof(double)),
        output.size() * sizeof(double)};
}

std::size_t parse_size(const char* value, const char* name) {
    std::size_t consumed = 0U;
    const auto result = static_cast<std::size_t>(std::stoull(value, &consumed));
    if (value[consumed] != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: vne_link_attribute_benchmark KIND COUNT WORKERS");
        }
        const std::string kind(argv[1]);
        const std::size_t count = parse_size(argv[2], "count");
        const std::size_t workers = parse_size(argv[3], "workers");
        Result result;
        if (kind == "dense_roundtrip") {
            result = dense_roundtrip(count, workers);
        } else if (kind == "position_latency") {
            result = position_latency(count, workers);
        } else {
            throw std::invalid_argument("unsupported benchmark kind");
        }
        std::cout << "protocol=1\n"
                  << "kind=" << kind << '\n'
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "link_attribute_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
