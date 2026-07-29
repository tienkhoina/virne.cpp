#include "attribute/base_attribute.h"

#include "numpy_random_state.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

std::uint64_t fnv1a(const unsigned char* bytes, std::size_t size) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t index = 0U; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct Digest {
    std::uint64_t checksum = 0U;
    std::size_t bytes = 0U;
};

Digest digest(const virne::utils::GeneratedData& data) {
    return std::visit(
        [](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const std::size_t bytes = values.size() * sizeof(Value);
            const auto* raw = reinterpret_cast<const unsigned char*>(values.data());
            return Digest{fnv1a(raw, bytes), bytes};
        },
        data.values);
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::size_t parse_size(const char* text) {
    std::size_t consumed = 0U;
    const unsigned long long value = std::stoull(text, &consumed);
    if (text[consumed] != '\0') {
        throw std::invalid_argument("invalid unsigned integer");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    using virne::network::attribute::AttributeKind;
    using virne::network::attribute::AttributeOwner;
    using virne::network::attribute::BaseAttribute;
    using virne::network::attribute::BaseAttributeSpec;
    using virne::network::attribute::NetworkCardinality;
    using virne::utils::DatasetScalar;
    using virne::utils::DatasetValueKind;
    using virne::utils::DistributionKind;

    try {
        if (argc != 5) {
            throw std::invalid_argument(
                "usage: base_attribute_benchmark KIND COUNT WORKERS SEED");
        }
        const std::string kind(argv[1]);
        const std::size_t count = parse_size(argv[2]);
        const std::size_t workers = parse_size(argv[3]);
        const std::size_t seed_value = parse_size(argv[4]);
        if (seed_value > UINT32_MAX) {
            throw std::invalid_argument("seed is outside uint32 range");
        }

        BaseAttributeSpec spec;
        spec.name = "benchmark";
        spec.owner = AttributeOwner::node;
        spec.kind = AttributeKind::resource;
        spec.generative = true;
        if (kind == "customized") {
            spec.distribution.kind = DistributionKind::customized;
            spec.distribution.minimum = DatasetScalar{-2.5};
            spec.distribution.maximum = DatasetScalar{7.25};
            spec.dtype = DatasetValueKind::boolean;
        } else if (kind == "exponential_int") {
            spec.distribution.kind = DistributionKind::exponential;
            spec.distribution.scale = DatasetScalar{0.5};
            spec.dtype = DatasetValueKind::integer;
        } else {
            throw std::invalid_argument("unknown benchmark kind");
        }

        const BaseAttribute attribute(std::move(spec));
        const NetworkCardinality cardinality{count, count / 2U};
        NumpyRandomState rng(static_cast<std::uint32_t>(seed_value));
        const auto started = std::chrono::steady_clock::now();
        const virne::utils::GeneratedData generated =
            attribute.generate_configured_data(cardinality, rng, workers);
        const auto stopped = std::chrono::steady_clock::now();
        const Digest output = digest(generated);
        const std::uint64_t next_bits = double_bits(rng.random());
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            stopped - started).count();

        std::cout << "protocol=1\n"
                  << "kind=" << kind << '\n'
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "elapsed_ns=" << elapsed << '\n'
                  << "checksum=" << output.checksum << '\n'
                  << "output_bytes=" << output.bytes << '\n'
                  << "next_bits=" << next_bits << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
