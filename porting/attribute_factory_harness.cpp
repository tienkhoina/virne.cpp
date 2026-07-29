#include "attribute/attribute_factory.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
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

std::string_view factory_error_name(
    const attribute::AttributeFactoryErrorCode value) {
    using Code = attribute::AttributeFactoryErrorCode;
    switch (value) {
        case Code::missing_name:
            return "missing_name";
        case Code::invalid_name:
            return "invalid_name";
        case Code::missing_owner:
            return "missing_owner";
        case Code::invalid_owner:
            return "invalid_owner";
        case Code::missing_kind:
            return "missing_kind";
        case Code::invalid_kind:
            return "invalid_kind";
        case Code::unsupported_pair:
            return "unsupported_pair";
        case Code::invalid_setting_item:
            return "invalid_setting_item";
        case Code::invalid_setting_value:
            return "invalid_setting_value";
        case Code::invalid_distribution:
            return "invalid_distribution";
        case Code::invalid_dtype:
            return "invalid_dtype";
        case Code::invalid_restriction:
            return "invalid_restriction";
        case Code::invalid_checking_level:
            return "invalid_checking_level";
        case Code::invalid_numeric_field:
            return "invalid_numeric_field";
        case Code::family_mismatch:
            return "family_mismatch";
        case Code::registry_id_range:
            return "registry_id_range";
        case Code::invalid_registry_id:
            return "invalid_registry_id";
    }
    return "invalid";
}

std::string_view factory_operation_name(
    const attribute::AttributeFactoryOperation value) {
    using Operation = attribute::AttributeFactoryOperation;
    switch (value) {
        case Operation::decode_name:
            return "decode_name";
        case Operation::decode_owner:
            return "decode_owner";
        case Operation::decode_kind:
            return "decode_kind";
        case Operation::validate_pair:
            return "validate_pair";
        case Operation::decode_fields:
            return "decode_fields";
        case Operation::construct_attribute:
            return "construct_attribute";
        case Operation::validate_family:
            return "validate_family";
        case Operation::decode_setting_list:
            return "decode_setting_list";
        case Operation::build_registry:
            return "build_registry";
        case Operation::access_registry:
            return "access_registry";
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

    if (const auto* node_resource =
            dynamic_cast<const attribute::NodeResourceAttribute*>(&value)) {
        restriction = restriction_name(node_resource->restriction());
        checking = checking_name(node_resource->checking_level());
    } else if (const auto* link_resource =
                   dynamic_cast<const attribute::LinkResourceAttribute*>(&value)) {
        restriction = restriction_name(link_resource->restriction());
        checking = checking_name(link_resource->checking_level());
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

template <typename Registry>
std::string canonical_registry(const Registry& registry) {
    std::string result;
    for (const auto& entry : registry.entries()) {
        if (!result.empty()) {
            result.push_back('\n');
        }
        result += canonical_attribute(*entry.attribute);
    }
    return result;
}

attribute::AttributeFactorySpec make_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const attribute::AttributeKind kind) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = kind;
    return result;
}

std::shared_ptr<utils::SettingObject> setting_object(
    std::initializer_list<std::pair<std::string, utils::SettingValue>> fields) {
    auto result = std::make_shared<utils::SettingObject>();
    result->reserve(fields.size());
    for (const auto& field : fields) {
        result->set(field.first, field.second);
    }
    return result;
}

utils::SettingValue setting_value(
    std::initializer_list<std::pair<std::string, utils::SettingValue>> fields) {
    return utils::SettingValue(setting_object(fields));
}

template <typename Callable>
void emit_case(const std::string_view name, Callable&& callable) {
    try {
        const std::string value = std::forward<Callable>(callable)();
        std::cout << "case=" << name << "|ok|" << hex_text(value) << '\n';
    } catch (const attribute::AttributeFactoryException& error) {
        std::cout << "case=" << name << "|error|"
                  << factory_error_name(error.code()) << '|'
                  << factory_operation_name(error.operation()) << '|';
        if (error.input_index()) {
            std::cout << *error.input_index();
        } else {
            std::cout << '-';
        }
        std::cout << '\n';
    }
}

std::string create_canonical(attribute::AttributeFactorySpec spec) {
    return canonical_attribute(*attribute::create_attribute(std::move(spec)));
}

std::string create_raw_canonical(
    const std::shared_ptr<utils::SettingObject>& setting) {
    return create_canonical(
        attribute::attribute_factory_spec_from_setting(*setting));
}

std::vector<attribute::AttributeFactorySpec> worker_specs() {
    std::vector<attribute::AttributeFactorySpec> specs;
    specs.push_back(make_spec(
        "cpu", attribute::AttributeOwner::node,
        attribute::AttributeKind::resource));
    auto peak = make_spec(
        "peak", attribute::AttributeOwner::node,
        attribute::AttributeKind::extrema);
    peak.originator_name = "cpu";
    specs.push_back(std::move(peak));
    specs.push_back(make_spec(
        "state", attribute::AttributeOwner::node,
        attribute::AttributeKind::status));
    auto replacement = make_spec(
        "cpu", attribute::AttributeOwner::node,
        attribute::AttributeKind::resource);
    replacement.restriction = attribute::ConstraintRestriction::soft;
    specs.push_back(std::move(replacement));
    specs.push_back(make_spec(
        "pos", attribute::AttributeOwner::node,
        attribute::AttributeKind::position));
    return specs;
}

void registered_pair_cases() {
    using attribute::AttributeKind;
    using attribute::AttributeOwner;
    emit_case("node_status_default", [] {
        return create_canonical(
            make_spec("node-status", AttributeOwner::node, AttributeKind::status));
    });
    emit_case("link_status_default", [] {
        return create_canonical(
            make_spec("link-status", AttributeOwner::link, AttributeKind::status));
    });
    emit_case("node_extrema_default", [] {
        auto spec =
            make_spec("node-peak", AttributeOwner::node, AttributeKind::extrema);
        spec.originator_name = "cpu";
        return create_canonical(std::move(spec));
    });
    emit_case("link_extrema_default", [] {
        auto spec =
            make_spec("link-peak", AttributeOwner::link, AttributeKind::extrema);
        spec.originator_name = "bandwidth";
        return create_canonical(std::move(spec));
    });
    emit_case("node_resource_default", [] {
        return create_canonical(
            make_spec("cpu", AttributeOwner::node, AttributeKind::resource));
    });
    emit_case("link_resource_default", [] {
        return create_canonical(make_spec(
            "bandwidth", AttributeOwner::link, AttributeKind::resource));
    });
    emit_case("node_position_default", [] {
        return create_canonical(
            make_spec("pos", AttributeOwner::node, AttributeKind::position));
    });
    emit_case("link_latency_default", [] {
        return create_canonical(
            make_spec("latency", AttributeOwner::link, AttributeKind::latency));
    });
}

void raw_setting_cases() {
    emit_case("raw_resource_fields", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("cpu-rich")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("resource")},
            {"generative", utils::SettingValue(true)},
            {"distribution", utils::SettingValue("uniform")},
            {"dtype", utils::SettingValue("int")},
            {"low", utils::SettingValue(std::int64_t{5})},
            {"high", utils::SettingValue(12.5)},
            {"constraint_restrictions", utils::SettingValue("soft")},
        }));
    });
    emit_case("raw_latency_position", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("delay")},
            {"owner", utils::SettingValue("link")},
            {"type", utils::SettingValue("latency")},
            {"generative", utils::SettingValue(true)},
            {"distribution", utils::SettingValue("position")},
            {"min", utils::SettingValue(std::int64_t{2})},
            {"max", utils::SettingValue(7.0)},
        }));
    });
    emit_case("raw_distribution_null", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("nullable")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("status")},
            {"distribution", utils::SettingValue()},
            {"dtype", utils::SettingValue()},
        }));
    });
    emit_case("raw_restriction_precedence", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("precedence")},
            {"owner", utils::SettingValue("link")},
            {"type", utils::SettingValue("resource")},
            {"constraint_restrictions", utils::SettingValue("soft")},
            {"restriction", utils::SettingValue("hard")},
        }));
    });
}

void registry_cases() {
    emit_case("empty_general", [] {
        return canonical_registry(attribute::create_attributes_from_specs({}, 8U));
    });
    emit_case("duplicate_order_setting", [] {
        utils::SettingList settings;
        settings.push_back(setting_value({
            {"name", utils::SettingValue("a")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("status")},
        }));
        settings.push_back(setting_value({
            {"name", utils::SettingValue("b")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("resource")},
        }));
        settings.push_back(setting_value({
            {"name", utils::SettingValue("a")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("position")},
        }));
        return canonical_registry(
            attribute::create_node_attributes_from_setting(settings, 8U));
    });
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        emit_case("workers_" + std::to_string(workers), [workers] {
            const auto specs = worker_specs();
            return canonical_registry(
                attribute::create_node_attributes_from_specs(specs, workers));
        });
    }
}

void error_cases() {
    emit_case("missing_owner", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"type", utils::SettingValue("status")},
        }));
    });
    emit_case("false_owner", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"owner", utils::SettingValue("")},
            {"type", utils::SettingValue("status")},
        }));
    });
    emit_case("missing_kind", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"owner", utils::SettingValue("node")},
        }));
    });
    emit_case("unsupported_graph_pair", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"owner", utils::SettingValue("graph")},
            {"type", utils::SettingValue("status")},
        }));
    });
    emit_case("unsupported_node_latency", [] {
        return create_canonical(make_spec(
            "x", attribute::AttributeOwner::node,
            attribute::AttributeKind::latency));
    });
    emit_case("node_helper_mismatch", [] {
        const auto value = attribute::create_node_attribute(make_spec(
            "x", attribute::AttributeOwner::link,
            attribute::AttributeKind::status));
        return canonical_attribute(*value);
    });
    emit_case("link_helper_mismatch", [] {
        const auto value = attribute::create_link_attribute(make_spec(
            "x", attribute::AttributeOwner::node,
            attribute::AttributeKind::status));
        return canonical_attribute(*value);
    });
    emit_case("graph_helper_mismatch", [] {
        const auto value = attribute::create_graph_attribute(make_spec(
            "x", attribute::AttributeOwner::node,
            attribute::AttributeKind::status));
        return canonical_attribute(*value);
    });
    emit_case("graph_helper_unsupported", [] {
        const auto value = attribute::create_graph_attribute(make_spec(
            "x", attribute::AttributeOwner::graph,
            attribute::AttributeKind::status));
        return canonical_attribute(*value);
    });
    emit_case("null_restriction", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("resource")},
            {"constraint_restrictions", utils::SettingValue()},
        }));
    });
    emit_case("invalid_bool_field", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("resource")},
            {"generative", utils::SettingValue(std::int64_t{1})},
        }));
    });
    emit_case("invalid_checking_level", [] {
        return create_raw_canonical(setting_object({
            {"name", utils::SettingValue("x")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("resource")},
            {"checking_level", utils::SettingValue("planet")},
        }));
    });
    emit_case("invalid_setting_item", [] {
        utils::SettingList settings;
        settings.push_back(setting_value({
            {"name", utils::SettingValue("ok")},
            {"owner", utils::SettingValue("node")},
            {"type", utils::SettingValue("status")},
        }));
        settings.push_back(utils::SettingValue(std::int64_t{3}));
        return canonical_registry(
            attribute::create_node_attributes_from_setting(settings, 8U));
    });
    emit_case("lowest_worker_error", [] {
        std::vector<attribute::AttributeFactorySpec> specs;
        specs.push_back(make_spec(
            "good", attribute::AttributeOwner::node,
            attribute::AttributeKind::status));
        specs.push_back(make_spec(
            "family", attribute::AttributeOwner::link,
            attribute::AttributeKind::status));
        specs.push_back(make_spec(
            "pair", attribute::AttributeOwner::node,
            attribute::AttributeKind::latency));
        return canonical_registry(
            attribute::create_node_attributes_from_specs(specs, 8U));
    });
}

}  // namespace

int main() {
    try {
        registered_pair_cases();
        raw_setting_cases();
        registry_cases();
        error_cases();
        std::cout << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attribute_factory_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
