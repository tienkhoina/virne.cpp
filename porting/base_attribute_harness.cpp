#include "attribute/base_attribute.h"

#include "numpy_random_state.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::BaseAttribute;
using attribute::BaseAttributeErrorCode;
using attribute::BaseAttributeException;
using attribute::BaseAttributeSpec;
using attribute::NetworkCardinality;
using utils::DatasetErrorCode;
using utils::DatasetException;
using utils::DatasetScalar;
using utils::DatasetValueKind;
using utils::DistributionKind;
using utils::DistributionSpec;
using utils::GeneratedData;

struct GenerationCase {
    std::string name;
    BaseAttributeSpec spec;
    NetworkCardinality network;
    std::uint32_t seed = 0U;
    std::size_t workers = 1U;
};

std::string hex_encode(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2U, '0');
    for (std::size_t index = 0U; index < value.size(); ++index) {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2U] = digits[byte >> 4U];
        result[index * 2U + 1U] = digits[byte & 0x0fU];
    }
    return result;
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string hex_bits(double value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16)
           << double_bits(value);
    return output.str();
}

std::string base_error_name(BaseAttributeErrorCode code) noexcept {
    switch (code) {
        case BaseAttributeErrorCode::invalid_owner:
            return "invalid_owner";
        case BaseAttributeErrorCode::invalid_kind:
            return "invalid_kind";
        case BaseAttributeErrorCode::not_implemented:
            return "not_implemented";
        case BaseAttributeErrorCode::not_generative:
            return "not_generative";
        case BaseAttributeErrorCode::unsupported_distribution:
            return "unsupported_distribution";
        case BaseAttributeErrorCode::invalid_custom_range:
            return "invalid_custom_range";
    }
    return "invalid_base_error";
}

std::string dataset_error_name(DatasetErrorCode code) noexcept {
    switch (code) {
        case DatasetErrorCode::invalid_distribution:
            return "invalid_distribution";
        case DatasetErrorCode::invalid_value_kind:
            return "invalid_value_kind";
        case DatasetErrorCode::invalid_topology:
            return "invalid_topology";
        case DatasetErrorCode::missing_parameter:
            return "missing_parameter";
        case DatasetErrorCode::invalid_parameter:
            return "invalid_parameter";
        case DatasetErrorCode::uniform_boolean_uninitialized:
            return "uniform_boolean_uninitialized";
        case DatasetErrorCode::unsupported_parameter_distribution:
            return "unsupported_parameter_distribution";
        case DatasetErrorCode::rng_backend_failure:
            return "rng_backend_failure";
        case DatasetErrorCode::xml_parse_failure:
            return "xml_parse_failure";
        case DatasetErrorCode::xml_schema_failure:
            return "xml_schema_failure";
        case DatasetErrorCode::unknown_endpoint:
            return "unknown_endpoint";
        case DatasetErrorCode::graph_materialization_failure:
            return "graph_materialization_failure";
        case DatasetErrorCode::gml_write_failure:
            return "gml_write_failure";
    }
    return "invalid_dataset_error";
}

std::string serialized_values(const GeneratedData& generated) {
    std::ostringstream output;
    if (const auto* values = std::get_if<std::vector<std::int64_t>>(
            &generated.values)) {
        output << "int|";
        for (std::size_t index = 0U; index < values->size(); ++index) {
            if (index != 0U) {
                output << ',';
            }
            output << (*values)[index];
        }
        return output.str();
    }
    if (const auto* values = std::get_if<std::vector<double>>(&generated.values)) {
        output << "float|";
        for (std::size_t index = 0U; index < values->size(); ++index) {
            if (index != 0U) {
                output << ',';
            }
            output << hex_bits((*values)[index]);
        }
        return output.str();
    }
    const auto& values = std::get<std::vector<std::uint8_t>>(generated.values);
    output << "bool|";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << static_cast<unsigned int>(values[index]);
    }
    return output.str();
}

DistributionSpec uniform(DatasetScalar low, DatasetScalar high) {
    DistributionSpec result;
    result.kind = DistributionKind::uniform;
    result.low = std::move(low);
    result.high = std::move(high);
    return result;
}

DistributionSpec normal(
    std::optional<DatasetScalar> location,
    std::optional<DatasetScalar> scale) {
    DistributionSpec result;
    result.kind = DistributionKind::normal;
    result.loc = std::move(location);
    result.scale = std::move(scale);
    return result;
}

DistributionSpec exponential(DatasetScalar scale) {
    DistributionSpec result;
    result.kind = DistributionKind::exponential;
    result.scale = std::move(scale);
    return result;
}

DistributionSpec poisson(DatasetScalar lambda, bool reciprocal = false) {
    DistributionSpec result;
    result.kind = DistributionKind::poisson;
    result.lambda = std::move(lambda);
    result.reciprocal = reciprocal;
    return result;
}

DistributionSpec customized(DatasetScalar minimum, DatasetScalar maximum) {
    DistributionSpec result;
    result.kind = DistributionKind::customized;
    result.minimum = std::move(minimum);
    result.maximum = std::move(maximum);
    return result;
}

BaseAttributeSpec spec(
    std::string name,
    AttributeOwner owner,
    bool generative,
    DistributionSpec distribution,
    std::optional<DatasetValueKind> dtype = std::nullopt) {
    BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.generative = generative;
    result.distribution = std::move(distribution);
    result.dtype = dtype;
    return result;
}

std::vector<GenerationCase> cases() {
    using Scalar = DatasetScalar;
    std::vector<GenerationCase> result;
    const NetworkCardinality ordinary{17U, 13U};
    result.push_back({"not_generative", spec("x", AttributeOwner::node, false,
        uniform(Scalar{0.0}, Scalar{1.0})), ordinary, 1U, 8U});
    result.push_back({"unsupported_none", spec("x", AttributeOwner::node, true,
        DistributionSpec{}), ordinary, 2U, 1U});
    DistributionSpec invalid_distribution;
    invalid_distribution.kind = static_cast<DistributionKind>(255U);
    result.push_back({"unsupported_unknown", spec("x", AttributeOwner::node, true,
        invalid_distribution), ordinary, 3U, 2U});

    result.push_back({"uniform_float_node", spec("x", AttributeOwner::node, true,
        uniform(Scalar{-2.25}, Scalar{8.5}), DatasetValueKind::floating),
        ordinary, 11U, 1U});
    result.push_back({"uniform_int_link", spec("x", AttributeOwner::link, true,
        uniform(Scalar{std::int64_t{-7}}, Scalar{std::int64_t{13}}),
        DatasetValueKind::integer), ordinary, 12U, 8U});
    result.push_back({"normal_float_default_dtype", spec("x", AttributeOwner::graph,
        true, normal(Scalar{-1.25}, Scalar{2.5})), ordinary, 13U, 0U});
    result.push_back({"normal_int", spec("x", AttributeOwner::node, true,
        normal(Scalar{-1.25}, Scalar{2.5}), DatasetValueKind::integer),
        ordinary, 14U, 2U});
    result.push_back({"normal_bool", spec("x", AttributeOwner::link, true,
        normal(Scalar{-1.25}, Scalar{2.5}), DatasetValueKind::boolean),
        ordinary, 15U, 8U});
    result.push_back({"exponential_float", spec("x", AttributeOwner::node, true,
        exponential(Scalar{0.5}), DatasetValueKind::floating), ordinary, 16U, 1U});
    result.push_back({"exponential_int", spec("x", AttributeOwner::link, true,
        exponential(Scalar{2.75}), DatasetValueKind::integer), ordinary, 17U, 2U});
    result.push_back({"exponential_bool", spec("x", AttributeOwner::graph, true,
        exponential(Scalar{2.75}), DatasetValueKind::boolean), ordinary, 18U, 8U});
    result.push_back({"poisson_int", spec("x", AttributeOwner::node, true,
        poisson(Scalar{4.25}), DatasetValueKind::integer), ordinary, 19U, 1U});
    result.push_back({"poisson_float", spec("x", AttributeOwner::link, true,
        poisson(Scalar{4.25}), DatasetValueKind::floating), ordinary, 20U, 2U});
    result.push_back({"poisson_bool", spec("x", AttributeOwner::graph, true,
        poisson(Scalar{4.25}), DatasetValueKind::boolean), ordinary, 21U, 8U});
    result.push_back({"poisson_reciprocal_ignored", spec("x", AttributeOwner::node,
        true, poisson(Scalar{0.0}, true), DatasetValueKind::integer),
        ordinary, 22U, 8U});

    result.push_back({"custom_bool_bounds", spec("x", AttributeOwner::node, true,
        customized(Scalar{false}, Scalar{true}), DatasetValueKind::boolean),
        ordinary, 31U, 0U});
    result.push_back({"custom_mixed_graph", spec("x", AttributeOwner::graph, true,
        customized(Scalar{std::int64_t{-7}}, Scalar{4.5}),
        DatasetValueKind::integer), ordinary, 32U, 2U});
    result.push_back({"custom_integral_large_span", spec("x", AttributeOwner::node,
        true, customized(Scalar{std::int64_t{4'611'686'018'427'387'904LL}},
            Scalar{std::int64_t{4'611'686'018'427'388'929LL}})),
        NetworkCardinality{31U, 0U}, 7U, 8U});
    result.push_back({"custom_zero", spec("x", AttributeOwner::node, true,
        customized(Scalar{-1.0}, Scalar{1.0})), NetworkCardinality{}, 33U, 8U});
    result.push_back({"custom_missing", spec("x", AttributeOwner::node, true,
        customized(Scalar{std::monostate{}}, Scalar{1.0})), ordinary, 34U, 1U});
    result.push_back({"custom_string", spec("x", AttributeOwner::node, true,
        customized(Scalar{std::string{"bad"}}, Scalar{1.0})), ordinary, 35U, 2U});
    result.push_back({"custom_equal", spec("x", AttributeOwner::node, true,
        customized(Scalar{1.0}, Scalar{1.0})), ordinary, 36U, 8U});
    result.push_back({"custom_reversed", spec("x", AttributeOwner::node, true,
        customized(Scalar{2.0}, Scalar{1.0})), ordinary, 37U, 1U});

    result.push_back({"normal_missing_location", spec("x", AttributeOwner::node,
        true, normal(std::nullopt, Scalar{1.0}), DatasetValueKind::floating),
        ordinary, 41U, 8U});
    result.push_back({"normal_missing_scale", spec("x", AttributeOwner::node,
        true, normal(Scalar{0.0}, std::nullopt), DatasetValueKind::floating),
        ordinary, 42U, 8U});
    result.push_back({"uniform_bool_bug", spec("x", AttributeOwner::node, true,
        uniform(Scalar{false}, Scalar{true}), DatasetValueKind::boolean),
        ordinary, 43U, 8U});
    result.push_back({"exponential_negative", spec("x", AttributeOwner::node, true,
        exponential(Scalar{-1.0}), DatasetValueKind::floating), ordinary, 44U, 1U});
    result.push_back({"poisson_negative", spec("x", AttributeOwner::node, true,
        poisson(Scalar{-1.0}), DatasetValueKind::integer), ordinary, 45U, 2U});
    return result;
}

void emit_static_cases() {
    BaseAttributeSpec rich;
    rich.name = "cpu";
    rich.owner = AttributeOwner::link;
    rich.kind = AttributeKind::resource;
    rich.generative = true;
    rich.distribution = uniform(
        DatasetScalar{std::int64_t{-3}}, DatasetScalar{7.5});
    rich.dtype = DatasetValueKind::integer;
    rich.originator = "capacity";
    rich.is_constraint = true;
    const BaseAttribute value(std::move(rich));
    std::string snapshot;
    for (const auto& entry : value.to_dict()) {
        if (!snapshot.empty()) {
            snapshot.push_back('\n');
        }
        snapshot.append(entry.name);
        snapshot.push_back('=');
        snapshot.append(utils::format_dataset_scalar(entry.value));
    }
    std::cout << "snapshot=" << hex_encode(snapshot) << '\n';
    std::cout << "repr=" << hex_encode(value.repr("Probe")) << '\n';

    BaseAttributeSpec plain;
    plain.name = "plain";
    BaseAttribute defaults(std::move(plain));
    try {
        static_cast<void>(defaults.generate_data());
    } catch (const BaseAttributeException& error) {
        std::cout << "default_generate=" << base_error_name(error.code()) << '|'
                  << hex_encode(error.what()) << '\n';
    }
    try {
        defaults.update_data();
    } catch (const BaseAttributeException& error) {
        std::cout << "default_update=" << base_error_name(error.code()) << '|'
                  << hex_encode(error.what()) << '\n';
    }
}

void emit_generation_cases() {
    for (const GenerationCase& item : cases()) {
        NumpyRandomState rng(item.seed);
        try {
            const BaseAttribute value(item.spec);
            const GeneratedData generated =
                value.generate_configured_data(item.network, rng, item.workers);
            std::cout << "case=" << item.name << "|ok|"
                      << serialized_values(generated) << '|'
                      << hex_bits(rng.random()) << '\n';
        } catch (const BaseAttributeException& error) {
            std::cout << "case=" << item.name << "|error|base|"
                      << base_error_name(error.code()) << '|'
                      << hex_encode(error.what()) << '|'
                      << hex_bits(rng.random()) << '\n';
        } catch (const DatasetException& error) {
            std::cout << "case=" << item.name << "|error|dataset|"
                      << dataset_error_name(error.code()) << '|'
                      << hex_encode(error.what()) << '|'
                      << hex_bits(rng.random()) << '\n';
        }
    }
}

}  // namespace

int main() {
    try {
        emit_static_cases();
        emit_generation_cases();
        std::cout << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
