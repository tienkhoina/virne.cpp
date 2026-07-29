#include "attribute/attribute_factory.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

constexpr std::uint64_t fnv_offset = UINT64_C(14695981039346656037);
constexpr std::uint64_t fnv_prime = UINT64_C(1099511628211);

struct Result {
    std::uint64_t elapsed_ns = 0U;
    std::uint64_t checksum = fnv_offset;
    std::size_t output_bytes = 0U;
    std::size_t entry_count = 0U;
};

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

std::string scalar_token(const utils::DatasetScalar& value) {
    return std::visit(
        [](const auto& item) -> std::string {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, std::monostate>) {
                return "null";
            } else if constexpr (std::is_same_v<Item, std::int64_t>) {
                return "i:" + std::to_string(item);
            } else if constexpr (std::is_same_v<Item, double>) {
                return double_token(item);
            } else if constexpr (std::is_same_v<Item, bool>) {
                return item ? "b:1" : "b:0";
            } else {
                return "s:" + hex_text(item);
            }
        },
        value);
}

std::string optional_scalar_token(
    const std::optional<utils::DatasetScalar>& value) {
    return value ? scalar_token(*value) : "none";
}

std::string number_token(const attribute::AttributeNumber& value) {
    return std::visit(
        [](const auto item) -> std::string {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, bool>) {
                return item ? "b:1" : "b:0";
            } else if constexpr (std::is_same_v<Item, std::int64_t>) {
                return "i:" + std::to_string(item);
            } else {
                return double_token(item);
            }
        },
        value);
}

std::string_view distribution_name(const utils::DistributionKind kind) {
    switch (kind) {
        case utils::DistributionKind::none:
            return "none";
        case utils::DistributionKind::uniform:
            return "uniform";
        case utils::DistributionKind::normal:
            return "normal";
        case utils::DistributionKind::exponential:
            return "exponential";
        case utils::DistributionKind::poisson:
            return "poisson";
        case utils::DistributionKind::customized:
            return "customized";
    }
    return "invalid";
}

std::string_view dtype_name(
    const std::optional<utils::DatasetValueKind> dtype) {
    if (!dtype) {
        return "none";
    }
    switch (*dtype) {
        case utils::DatasetValueKind::integer:
            return "integer";
        case utils::DatasetValueKind::floating:
            return "floating";
        case utils::DatasetValueKind::boolean:
            return "boolean";
    }
    return "invalid";
}

std::string_view restriction_name(
    const attribute::ConstraintRestriction value) {
    return value == attribute::ConstraintRestriction::hard ? "hard" : "soft";
}

std::string_view checking_name(const attribute::CheckingLevel value) {
    switch (value) {
        case attribute::CheckingLevel::node:
            return "node";
        case attribute::CheckingLevel::link:
            return "link";
        case attribute::CheckingLevel::path:
            return "path";
        case attribute::CheckingLevel::graph:
            return "graph";
    }
    return "invalid";
}

std::string concrete_class(const attribute::BaseAttribute& value) {
    if (dynamic_cast<const attribute::NodeStatusAttribute*>(&value)) {
        return "NodeStatusAttribute";
    }
    if (dynamic_cast<const attribute::NodeExtremaAttribute*>(&value)) {
        return "NodeExtremaAttribute";
    }
    if (dynamic_cast<const attribute::NodeResourceAttribute*>(&value)) {
        return "NodeResourceAttribute";
    }
    if (dynamic_cast<const attribute::NodePositionAttribute*>(&value)) {
        return "NodePositionAttribute";
    }
    if (dynamic_cast<const attribute::LinkStatusAttribute*>(&value)) {
        return "LinkStatusAttribute";
    }
    if (dynamic_cast<const attribute::LinkExtremaAttribute*>(&value)) {
        return "LinkExtremaAttribute";
    }
    if (dynamic_cast<const attribute::LinkResourceAttribute*>(&value)) {
        return "LinkResourceAttribute";
    }
    if (dynamic_cast<const attribute::LinkLatencyAttribute*>(&value)) {
        return "LinkLatencyAttribute";
    }
    return "UnknownAttribute";
}

std::string canonical_attribute(const attribute::BaseAttribute& value) {
    const auto& spec = value.spec();
    std::string distribution(distribution_name(spec.distribution.kind));
    std::string restriction = "none";
    std::string checking = "none";
    std::string minimum_radius = "none";
    std::string maximum_radius = "none";
    std::string latency_generation = "none";
    std::string minimum = "none";
    std::string maximum = "none";

    if (const auto* resource =
            dynamic_cast<const attribute::NodeResourceAttribute*>(&value)) {
        restriction = restriction_name(resource->restriction());
        checking = checking_name(resource->checking_level());
    } else if (const auto* resource =
                   dynamic_cast<const attribute::LinkResourceAttribute*>(&value)) {
        restriction = restriction_name(resource->restriction());
        checking = checking_name(resource->checking_level());
    } else if (const auto* position =
                   dynamic_cast<const attribute::NodePositionAttribute*>(&value)) {
        restriction = restriction_name(position->restriction());
        minimum_radius = double_token(position->minimum_radius());
        maximum_radius = double_token(position->maximum_radius());
    } else if (const auto* latency =
                   dynamic_cast<const attribute::LinkLatencyAttribute*>(&value)) {
        restriction = restriction_name(latency->restriction());
        checking = checking_name(latency->checking_level());
        latency_generation = latency->generation_kind() ==
                                     attribute::LatencyGenerationKind::position
                                 ? "position"
                                 : "configured";
        if (latency->generation_kind() ==
            attribute::LatencyGenerationKind::position) {
            distribution = "position";
        }
        minimum = number_token(latency->minimum());
        maximum = number_token(latency->maximum());
    }

    std::ostringstream stream;
    stream << "name=" << hex_text(spec.name)
           << ";owner=" << attribute::attribute_owner_name(spec.owner)
           << ";kind=" << attribute::attribute_kind_name(spec.kind)
           << ";class=" << concrete_class(value)
           << ";generative=" << (spec.generative ? '1' : '0')
           << ";distribution=" << distribution
           << ";dtype=" << dtype_name(spec.dtype)
           << ";low=" << optional_scalar_token(spec.distribution.low)
           << ";high=" << optional_scalar_token(spec.distribution.high)
           << ";loc=" << optional_scalar_token(spec.distribution.loc)
           << ";scale=" << optional_scalar_token(spec.distribution.scale)
           << ";lam=" << optional_scalar_token(spec.distribution.lambda)
           << ";originator="
           << (spec.originator ? hex_text(*spec.originator) : "none")
           << ";constraint="
           << (spec.is_constraint ? (*spec.is_constraint ? '1' : '0') : '-')
           << ";restriction=" << restriction
           << ";checking=" << checking
           << ";min_r=" << minimum_radius
           << ";max_r=" << maximum_radius
           << ";latency_generation=" << latency_generation
           << ";minimum=" << minimum
           << ";maximum=" << maximum;
    return stream.str();
}

std::vector<attribute::AttributeFactorySpec> make_fixture(
    const std::size_t count) {
    std::vector<attribute::AttributeFactorySpec> specs;
    specs.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        attribute::AttributeFactorySpec spec;
        spec.name = "metric_" + std::to_string(index);
        switch (index % 8U) {
            case 0U:
                spec.owner = attribute::AttributeOwner::node;
                spec.kind = attribute::AttributeKind::status;
                break;
            case 1U:
                spec.owner = attribute::AttributeOwner::link;
                spec.kind = attribute::AttributeKind::status;
                break;
            case 2U:
                spec.owner = attribute::AttributeOwner::node;
                spec.kind = attribute::AttributeKind::extrema;
                spec.originator_name = "origin_" + std::to_string(index);
                break;
            case 3U:
                spec.owner = attribute::AttributeOwner::link;
                spec.kind = attribute::AttributeKind::extrema;
                spec.originator_name = "origin_" + std::to_string(index);
                break;
            case 4U:
                spec.owner = attribute::AttributeOwner::node;
                spec.kind = attribute::AttributeKind::resource;
                break;
            case 5U:
                spec.owner = attribute::AttributeOwner::link;
                spec.kind = attribute::AttributeKind::resource;
                break;
            case 6U:
                spec.owner = attribute::AttributeOwner::node;
                spec.kind = attribute::AttributeKind::position;
                break;
            case 7U:
                spec.owner = attribute::AttributeOwner::link;
                spec.kind = attribute::AttributeKind::latency;
                break;
            default:
                throw std::logic_error("unreachable factory fixture branch");
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

void fnv_bytes(std::uint64_t& checksum, const std::string_view bytes) noexcept {
    for (const unsigned char byte : bytes) {
        checksum ^= byte;
        checksum *= fnv_prime;
    }
}

Result run_factory(const std::size_t count, const std::size_t workers) {
    const auto specs = make_fixture(count);
    const auto begin = std::chrono::steady_clock::now();
    const auto registry =
        attribute::create_attributes_from_specs(specs, workers);
    const auto end = std::chrono::steady_clock::now();

    Result result;
    result.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count());
    result.entry_count = registry.size();
    bool first = true;
    for (const auto& entry : registry.entries()) {
        if (!first) {
            fnv_bytes(result.checksum, "\n");
            ++result.output_bytes;
        }
        first = false;
        const std::string canonical = canonical_attribute(*entry.attribute);
        fnv_bytes(result.checksum, canonical);
        result.output_bytes += canonical.size();
    }
    return result;
}

std::size_t parse_size(const char* value, const char* label) {
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (value[consumed] != '\0' ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return static_cast<std::size_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: vne_attribute_factory_benchmark COUNT WORKERS");
        }
        const std::size_t count = parse_size(argv[1], "count");
        const std::size_t workers = parse_size(argv[2], "workers");
        if (count == 0U) {
            throw std::invalid_argument("count must be positive");
        }
        const Result result = run_factory(count, workers);
        std::cout << "protocol=1\n"
                  << "kind=typed_attribute_factory\n"
                  << "count=" << count << '\n'
                  << "workers=" << workers << '\n'
                  << "type_tag=ordered_factory_v1\n"
                  << "elapsed_ns=" << result.elapsed_ns << '\n'
                  << "checksum=" << result.checksum << '\n'
                  << "output_bytes=" << result.output_bytes << '\n'
                  << "entry_count=" << result.entry_count << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attribute_factory_benchmark: FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
