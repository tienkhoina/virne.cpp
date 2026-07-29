#include "base_network.h"

#include "../utils/network.h"

#include "generators/gml_loader.h"
#include "numpy_random_state.h"
#include "nx/sparse.h"
#include "py_random.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace virne::network {
namespace {

namespace attr = attribute;
namespace util = virne::utils;

[[noreturn]] void throw_network(
    BaseNetworkErrorCode code,
    BaseNetworkOperation operation,
    std::size_t input_index,
    std::string message) {
    throw BaseNetworkException(
        code, operation, input_index, std::move(message));
}

std::string_view distribution_name(util::DistributionKind kind) {
    switch (kind) {
        case util::DistributionKind::none:
            return "none";
        case util::DistributionKind::uniform:
            return "uniform";
        case util::DistributionKind::normal:
            return "normal";
        case util::DistributionKind::exponential:
            return "exponential";
        case util::DistributionKind::poisson:
            return "poisson";
        case util::DistributionKind::customized:
            return "customized";
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        BaseNetworkOperation::decode_config,
        invalid_base_network_input_index,
        "Invalid distribution enum");
}

std::string_view dtype_name(util::DatasetValueKind kind) {
    switch (kind) {
        case util::DatasetValueKind::integer:
            return "int";
        case util::DatasetValueKind::floating:
            return "float";
        case util::DatasetValueKind::boolean:
            return "bool";
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        BaseNetworkOperation::decode_config,
        invalid_base_network_input_index,
        "Invalid dtype enum");
}

std::string_view checking_level_name(attr::CheckingLevel level) {
    switch (level) {
        case attr::CheckingLevel::node:
            return "node";
        case attr::CheckingLevel::link:
            return "link";
        case attr::CheckingLevel::path:
            return "path";
        case attr::CheckingLevel::graph:
            return "graph";
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        BaseNetworkOperation::decode_config,
        invalid_base_network_input_index,
        "Invalid checking-level enum");
}

std::string_view restriction_name(attr::ConstraintRestriction restriction) {
    switch (restriction) {
        case attr::ConstraintRestriction::hard:
            return "hard";
        case attr::ConstraintRestriction::soft:
            return "soft";
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        BaseNetworkOperation::decode_config,
        invalid_base_network_input_index,
        "Invalid restriction enum");
}

util::SettingValue setting_value_from_dataset_scalar(
    const util::DatasetScalar& value) {
    return std::visit(
        [](const auto& item) -> util::SettingValue {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, std::monostate>) {
                return util::SettingValue(nullptr);
            } else {
                return util::SettingValue(item);
            }
        },
        value);
}

util::SettingValue setting_value_from_attribute_number(
    const attr::AttributeNumber& value) {
    return std::visit(
        [](const auto& item) -> util::SettingValue {
            return util::SettingValue(item);
        },
        value);
}

void append_distribution_fields(
    util::SettingObject& object,
    const util::DistributionSpec& distribution) {
    const auto append = [&object](
                            std::string_view name,
                            const std::optional<util::DatasetScalar>& value) {
        if (value.has_value()) {
            object.set(name, setting_value_from_dataset_scalar(*value));
        }
    };
    append("low", distribution.low);
    append("high", distribution.high);
    append("loc", distribution.loc);
    append("scale", distribution.scale);
    append("lam", distribution.lambda);
    append("min", distribution.minimum);
    append("max", distribution.maximum);
    if (distribution.reciprocal) {
        object.set("reciprocal", util::SettingValue(true));
    }
}

util::SettingValue setting_value_from_factory_spec(
    const attr::AttributeFactorySpec& spec) {
    auto object = std::make_shared<util::SettingObject>();
    object->reserve(20U);
    object->set("name", util::SettingValue(spec.name));
    object->set(
        "owner",
        util::SettingValue(attr::attribute_owner_name(spec.owner)));
    object->set(
        "type",
        util::SettingValue(attr::attribute_kind_name(spec.kind)));
    object->set("generative", util::SettingValue(spec.generative));

    if (spec.kind == attr::AttributeKind::latency &&
        spec.latency_generation == attr::LatencyGenerationKind::position) {
        object->set("distribution", util::SettingValue("position"));
    } else if (spec.distribution.kind != util::DistributionKind::none) {
        object->set(
            "distribution",
            util::SettingValue(distribution_name(spec.distribution.kind)));
    }
    append_distribution_fields(*object, spec.distribution);

    if (spec.dtype.has_value()) {
        object->set("dtype", util::SettingValue(dtype_name(*spec.dtype)));
    }
    if (spec.originator_name.has_value()) {
        object->set("originator", util::SettingValue(*spec.originator_name));
    }

    if (spec.kind == attr::AttributeKind::resource ||
        spec.kind == attr::AttributeKind::position ||
        spec.kind == attr::AttributeKind::latency) {
        object->set(
            "constraint_restrictions",
            util::SettingValue(restriction_name(spec.restriction)));
    }
    if (spec.checking_level.has_value()) {
        object->set(
            "checking_level",
            util::SettingValue(checking_level_name(*spec.checking_level)));
    }
    if (spec.kind == attr::AttributeKind::position) {
        object->set("min_r", util::SettingValue(spec.minimum_radius));
        object->set("max_r", util::SettingValue(spec.maximum_radius));
    }
    if (spec.kind == attr::AttributeKind::latency) {
        if (!spec.distribution.minimum.has_value()) {
            object->set(
                "min", setting_value_from_attribute_number(spec.minimum));
        }
        if (!spec.distribution.maximum.has_value()) {
            object->set(
                "max", setting_value_from_attribute_number(spec.maximum));
        }
    }
    return util::SettingValue(std::move(object));
}

util::SettingList setting_list_from_factory_specs(
    const std::vector<attr::AttributeFactorySpec>& specs) {
    util::SettingList result;
    result.reserve(specs.size());
    for (const auto& spec : specs) {
        result.push_back(setting_value_from_factory_spec(spec));
    }
    return result;
}

std::int64_t setting_integer_to_int64(
    const util::SettingInteger& value,
    BaseNetworkOperation operation,
    std::size_t input_index) {
    const std::string decimal = value.convert_to<std::string>();
    std::int64_t result = 0;
    const char* const begin = decimal.data();
    const char* const end = begin + decimal.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            operation,
            input_index,
            "Setting integer is outside the native int64 domain");
    }
    return result;
}

AttrValue attr_value_from_setting(
    const util::SettingValue& value,
    BaseNetworkOperation operation,
    std::size_t input_index,
    std::size_t depth = 0U) {
    if (depth > 256U) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            operation,
            input_index,
            "Setting value exceeds the supported nesting depth");
    }
    switch (value.kind()) {
        case util::SettingValueKind::null_value:
            throw_network(
                BaseNetworkErrorCode::invalid_config,
                operation,
                input_index,
                "Null graph metadata is outside the native AttrValue domain");
        case util::SettingValueKind::boolean:
            return value.as_bool();
        case util::SettingValueKind::integer:
            return setting_integer_to_int64(
                value.as_integer(), operation, input_index);
        case util::SettingValueKind::real:
            return value.as_real();
        case util::SettingValueKind::string:
            return value.as_string();
        case util::SettingValueKind::list: {
            std::vector<AttrValue> values;
            values.reserve(value.as_list().size());
            for (const auto& item : value.as_list()) {
                values.push_back(attr_value_from_setting(
                    item, operation, input_index, depth + 1U));
            }
            return make_attr_list(std::move(values));
        }
        case util::SettingValueKind::object: {
            const auto& source = value.as_object();
            std::vector<std::pair<std::string, AttrValue>> entries;
            entries.reserve(source.size());
            for (std::size_t index = 0U; index < source.size(); ++index) {
                const util::SettingKeyId id{
                    static_cast<std::uint32_t>(index)};
                entries.emplace_back(
                    source.key_name(id),
                    attr_value_from_setting(
                        source.at(id), operation, input_index, depth + 1U));
            }
            return make_attr_object(std::move(entries));
        }
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        operation,
        input_index,
        "Invalid setting value kind");
}

util::SettingValue setting_value_from_attr(
    const AttrValue& value,
    BaseNetworkOperation operation,
    std::size_t input_index,
    std::size_t depth = 0U) {
    if (depth > 256U) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            operation,
            input_index,
            "Attribute value exceeds the supported nesting depth");
    }
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return util::SettingValue(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return util::SettingValue(*item);
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return util::SettingValue(*item);
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        return util::SettingValue(*item);
    }
    if (const AttrList* list = attr_list(value)) {
        auto target = std::make_shared<util::SettingList>();
        target->reserve(list->values.size());
        for (const auto& item : list->values) {
            target->push_back(setting_value_from_attr(
                item, operation, input_index, depth + 1U));
        }
        return util::SettingValue(std::move(target));
    }
    if (const AttrObject* object = attr_object(value)) {
        auto target = std::make_shared<util::SettingObject>();
        target->reserve(object->entries.size());
        for (const auto& [name, item] : object->entries) {
            target->set(
                name,
                setting_value_from_attr(
                    item, operation, input_index, depth + 1U));
        }
        return util::SettingValue(std::move(target));
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        operation,
        input_index,
        "Null recursive attribute storage is invalid");
}

AttrValue attr_value_from_setting_list(
    const util::SettingList& settings,
    BaseNetworkOperation operation) {
    std::vector<AttrValue> values;
    values.reserve(settings.size());
    for (std::size_t index = 0U; index < settings.size(); ++index) {
        values.push_back(
            attr_value_from_setting(settings[index], operation, index));
    }
    return make_attr_list(std::move(values));
}

util::SettingList setting_list_from_attr_value(
    const AttrValue& value,
    BaseNetworkOperation operation) {
    const AttrList* list = attr_list(value);
    if (list == nullptr) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            operation,
            invalid_base_network_input_index,
            "Attribute setting snapshot must be a list");
    }
    util::SettingList result;
    result.reserve(list->values.size());
    for (std::size_t index = 0U; index < list->values.size(); ++index) {
        result.push_back(setting_value_from_attr(
            list->values[index], operation, index));
    }
    return result;
}

util::SettingList graph_setting_list(
    const Graph& graph,
    std::string_view name,
    BaseNetworkOperation operation) {
    const auto id = graph.attribute_registry().find(name);
    if (!id.has_value()) {
        return {};
    }
    const AttrValue* value = graph.graph_attrs().find(*id);
    if (value == nullptr) {
        return {};
    }
    return setting_list_from_attr_value(*value, operation);
}

std::string setting_item_name(
    const util::SettingValue& item,
    std::size_t input_index) {
    if (item.kind() != util::SettingValueKind::object) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::decode_config,
            input_index,
            "Attribute setting item must be an object");
    }
    const auto& object = item.as_object();
    const auto name_id = object.find_key_id("name");
    if (!name_id.has_value()) {
        throw_network(
            BaseNetworkErrorCode::missing_config_field,
            BaseNetworkOperation::decode_config,
            input_index,
            "Attribute setting item is missing name");
    }
    const auto& name = object.at(*name_id);
    if (name.kind() != util::SettingValueKind::string) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::decode_config,
            input_index,
            "Attribute setting name must be a string");
    }
    return name.as_string();
}

util::SettingList merge_setting_lists(
    const util::SettingList& incoming,
    const util::SettingList& configured) {
    util::SettingList result;
    result.reserve(incoming.size() + configured.size());
    std::unordered_map<std::string, std::size_t> positions;
    positions.reserve(incoming.size() + configured.size());
    std::size_t input_index = 0U;
    auto append = [&](const util::SettingList& source) {
        for (const auto& item : source) {
            std::string name = setting_item_name(item, input_index);
            const auto [position, inserted] =
                positions.emplace(name, result.size());
            if (inserted) {
                result.push_back(item);
            } else {
                result[position->second] = item;
            }
            ++input_index;
        }
    };
    append(incoming);
    append(configured);
    return result;
}

std::vector<attr::AttributeFactorySpec> decode_factory_specs(
    const util::SettingList& settings) {
    std::vector<attr::AttributeFactorySpec> result;
    result.reserve(settings.size());
    for (std::size_t index = 0U; index < settings.size(); ++index) {
        if (settings[index].kind() != util::SettingValueKind::object) {
            throw_network(
                BaseNetworkErrorCode::invalid_config,
                BaseNetworkOperation::decode_config,
                index,
                "Attribute setting item must be an object");
        }
        result.push_back(attr::attribute_factory_spec_from_setting(
            settings[index].as_object()));
    }
    return result;
}

const util::SettingValue* setting_field(
    const util::SettingObject& object,
    std::string_view name) {
    const auto id = object.find_key_id(name);
    return id.has_value() ? &object.at(*id) : nullptr;
}

const util::SettingValue* first_setting_field(
    const util::SettingObject& object,
    std::string_view first,
    std::string_view second) {
    if (const auto* value = setting_field(object, first)) {
        return value;
    }
    return setting_field(object, second);
}

util::SettingList optional_setting_list(
    const util::SettingValue* value,
    std::string_view field_name) {
    if (value == nullptr || value->is_null()) {
        return {};
    }
    if (value->kind() != util::SettingValueKind::list) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::decode_config,
            invalid_base_network_input_index,
            std::string(field_name) + " must be a list or null");
    }
    return value->as_list();
}

bool setting_value_is_falsey(const util::SettingValue& value) {
    switch (value.kind()) {
        case util::SettingValueKind::null_value:
            return true;
        case util::SettingValueKind::boolean:
            return !value.as_bool();
        case util::SettingValueKind::integer:
            return value.as_integer() == util::SettingInteger{};
        case util::SettingValueKind::real:
            return value.as_real() == 0.0;
        case util::SettingValueKind::string:
            return value.as_string().empty();
        case util::SettingValueKind::list:
            return value.as_list().empty();
        case util::SettingValueKind::object:
            return value.as_object().empty();
    }
    return false;
}

std::vector<GraphAttributeAssignment> graph_assignments_from_setting(
    const util::SettingValue* value) {
    if (value == nullptr || value->is_null()) {
        return {};
    }
    if (value->kind() != util::SettingValueKind::object) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::decode_config,
            invalid_base_network_input_index,
            "graph_attrs_setting must be an object or null");
    }
    const auto& object = value->as_object();
    std::vector<GraphAttributeAssignment> result;
    result.reserve(object.size());
    for (std::size_t index = 0U; index < object.size(); ++index) {
        const util::SettingKeyId id{static_cast<std::uint32_t>(index)};
        result.push_back(GraphAttributeAssignment{
            std::string(object.key_name(id)),
            attr_value_from_setting(
                object.at(id), BaseNetworkOperation::decode_config, index)});
    }
    return result;
}

AttrValue attribute_number_to_attr_value(const attr::AttributeNumber& value) {
    return std::visit(
        [](const auto& item) -> AttrValue { return AttrValue(item); }, value);
}

std::vector<AttrValue> generated_data_to_attr_values(
    util::GeneratedData generated) {
    return std::visit(
        [&generated](auto&& values) -> std::vector<AttrValue> {
            using Values = std::decay_t<decltype(values)>;
            std::vector<AttrValue> result;
            result.reserve(values.size());
            if constexpr (
                std::is_same_v<Values, std::vector<std::uint8_t>>) {
                for (const std::uint8_t value : values) {
                    result.emplace_back(value != 0U);
                }
            } else {
                for (auto& value : values) {
                    result.emplace_back(value);
                }
            }
            return result;
        },
        std::move(generated.values));
}

util::DynamicValue dynamic_value_from_attr(
    const AttrValue& value,
    std::size_t depth = 0U) {
    if (depth > 256U) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::prepare_gml,
            invalid_base_network_input_index,
            "Attribute value exceeds the supported nesting depth");
    }
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return util::DynamicValue(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return util::DynamicValue(*item);
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return util::DynamicValue(*item);
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        return util::DynamicValue(*item);
    }
    if (const AttrList* list = attr_list(value)) {
        util::DynamicValue::List result;
        result.reserve(list->values.size());
        for (const auto& item : list->values) {
            result.push_back(dynamic_value_from_attr(item, depth + 1U));
        }
        return util::DynamicValue(std::move(result));
    }
    if (const AttrObject* object = attr_object(value)) {
        util::DynamicValue::Dict result;
        result.reserve(object->entries.size());
        for (const auto& [name, item] : object->entries) {
            result.emplace_back(
                util::DynamicKey(name),
                dynamic_value_from_attr(item, depth + 1U));
        }
        return util::DynamicValue(std::move(result));
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        BaseNetworkOperation::prepare_gml,
        invalid_base_network_input_index,
        "Null recursive attribute storage is invalid");
}

util::DynamicValue dynamic_value_from_setting(
    const util::SettingValue& value,
    std::size_t input_index,
    std::size_t depth = 0U) {
    if (depth > 256U) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::prepare_gml,
            input_index,
            "Setting value exceeds the supported nesting depth");
    }
    switch (value.kind()) {
        case util::SettingValueKind::null_value:
            return util::DynamicValue(nullptr);
        case util::SettingValueKind::boolean:
            return util::DynamicValue(value.as_bool());
        case util::SettingValueKind::integer:
            return util::DynamicValue(setting_integer_to_int64(
                value.as_integer(), BaseNetworkOperation::prepare_gml,
                input_index));
        case util::SettingValueKind::real:
            return util::DynamicValue(value.as_real());
        case util::SettingValueKind::string:
            return util::DynamicValue(value.as_string());
        case util::SettingValueKind::list: {
            util::DynamicValue::List result;
            result.reserve(value.as_list().size());
            for (const auto& item : value.as_list()) {
                result.push_back(
                    dynamic_value_from_setting(item, input_index, depth + 1U));
            }
            return util::DynamicValue(std::move(result));
        }
        case util::SettingValueKind::object: {
            const auto& source = value.as_object();
            util::DynamicValue::Dict result;
            result.reserve(source.size());
            for (std::size_t index = 0U; index < source.size(); ++index) {
                const util::SettingKeyId id{
                    static_cast<std::uint32_t>(index)};
                result.emplace_back(
                    util::DynamicKey(std::string(source.key_name(id))),
                    dynamic_value_from_setting(
                        source.at(id), input_index, depth + 1U));
            }
            return util::DynamicValue(std::move(result));
        }
    }
    throw_network(
        BaseNetworkErrorCode::invalid_config,
        BaseNetworkOperation::prepare_gml,
        input_index,
        "Invalid setting value kind");
}

std::string dynamic_key_string(const util::DynamicKey& key) {
    return std::visit(
        [](const auto& value) -> std::string {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, std::monostate>) {
                return "None";
            } else if constexpr (std::is_same_v<Value, bool>) {
                return value ? "True" : "False";
            } else if constexpr (std::is_same_v<Value, std::int64_t>) {
                return std::to_string(value);
            } else if constexpr (std::is_same_v<Value, double>) {
                std::ostringstream stream;
                stream << std::setprecision(17) << value;
                return stream.str();
            } else {
                return value;
            }
        },
        key.data);
}

AttrValue attr_value_from_dynamic(
    const util::DynamicValue& value,
    std::size_t depth = 0U) {
    if (depth > 256U) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::prepare_gml,
            invalid_base_network_input_index,
            "Dynamic value exceeds the supported nesting depth");
    }
    return std::visit(
        [depth](const auto& item) -> AttrValue {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, std::monostate>) {
                return std::string("None");
            } else if constexpr (
                std::is_same_v<Item, bool> ||
                std::is_same_v<Item, std::int64_t> ||
                std::is_same_v<Item, double> ||
                std::is_same_v<Item, std::string>) {
                return item;
            } else if constexpr (std::is_same_v<Item, util::DynamicValue::List>) {
                std::vector<AttrValue> values;
                values.reserve(item.size());
                for (const auto& child : item) {
                    values.push_back(attr_value_from_dynamic(child, depth + 1U));
                }
                return make_attr_list(std::move(values));
            } else {
                std::vector<std::pair<std::string, AttrValue>> entries;
                entries.reserve(item.size());
                for (const auto& [key, child] : item) {
                    entries.emplace_back(
                        dynamic_key_string(key),
                        attr_value_from_dynamic(child, depth + 1U));
                }
                return make_attr_object(std::move(entries));
            }
        },
        value.data);
}

util::DynamicDictList dynamic_dict_list_from_setting_list(
    const util::SettingList& settings) {
    util::DynamicDictList result;
    result.reserve(settings.size());
    for (std::size_t index = 0U; index < settings.size(); ++index) {
        if (settings[index].kind() != util::SettingValueKind::object) {
            throw_network(
                BaseNetworkErrorCode::invalid_config,
                BaseNetworkOperation::prepare_gml,
                index,
                "GML attribute setting item must be an object");
        }
        auto dynamic = dynamic_value_from_setting(settings[index], index);
        result.push_back(
            std::move(dynamic.as<util::DynamicValue::Dict>()));
    }
    return result;
}

AttrValue attr_value_from_dynamic_dict_list(
    const util::DynamicDictList& dicts) {
    std::vector<AttrValue> values;
    values.reserve(dicts.size());
    for (const auto& dict : dicts) {
        values.push_back(attr_value_from_dynamic(util::DynamicValue(dict)));
    }
    return make_attr_list(std::move(values));
}

std::string formatted_dynamic_value(const AttrValue& value) {
    util::DynamicDictList input;
    input.push_back(util::DynamicValue::Dict{
        {util::DynamicKey("value"), dynamic_value_from_attr(value)}});
    const auto output = util::flatten_dict_list_for_gml(input, 1U);
    if (output.size() != 1U || output.front().size() != 1U) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::prepare_gml,
            invalid_base_network_input_index,
            "Unexpected GML formatter result");
    }
    const auto& formatted = output.front().front().second;
    if (!formatted.is<std::string>()) {
        throw_network(
            BaseNetworkErrorCode::invalid_config,
            BaseNetworkOperation::prepare_gml,
            invalid_base_network_input_index,
            "GML formatter did not return a string");
    }
    return formatted.as<std::string>();
}

std::string quote_string(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('\'');
    for (const char character : value) {
        if (character == '\\' || character == '\'') {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    result.push_back('\'');
    return result;
}

bool contains_kind(
    const std::vector<attr::AttributeKind>& kinds,
    attr::AttributeKind kind) {
    return std::find(kinds.begin(), kinds.end(), kind) != kinds.end();
}

float benchmark_float(
    const AttrValue& value,
    std::size_t row,
    std::size_t column) {
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return static_cast<float>(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return static_cast<float>(*item);
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item ? 1.0F : 0.0F;
    }
    throw_network(
        BaseNetworkErrorCode::non_numeric_benchmark_value,
        BaseNetworkOperation::prepare_benchmarks,
        row,
        "Benchmark value at column " + std::to_string(column) +
            " is not numeric");
}

attr::PreparedAttributeBenchmarkData prepare_benchmark_data(
    const BaseNetwork& network,
    const std::vector<attr::AttributeRegistryId>& definitions,
    const std::vector<std::vector<AttrValue>>& rows,
    bool node,
    bool extrema_requested,
    std::size_t column_repetitions) {
    if (definitions.size() != rows.size()) {
        throw_network(
            BaseNetworkErrorCode::ragged_benchmark_matrix,
            BaseNetworkOperation::prepare_benchmarks,
            invalid_base_network_input_index,
            "Benchmark descriptors and rows have different lengths");
    }
    attr::PreparedAttributeBenchmarkData result;
    result.extrema_requested = extrema_requested;
    result.column_repetitions = column_repetitions;
    result.attributes.reserve(definitions.size());
    result.matrix.rows = rows.size();
    result.matrix.columns = rows.empty() ? 0U : rows.front().size();
    result.matrix.values.reserve(result.matrix.rows * result.matrix.columns);

    for (std::size_t row = 0U; row < definitions.size(); ++row) {
        if (rows[row].size() != result.matrix.columns) {
            throw_network(
                BaseNetworkErrorCode::ragged_benchmark_matrix,
                BaseNetworkOperation::prepare_benchmarks,
                row,
                "Benchmark rows must be rectangular");
        }
        const auto id = definitions[row];
        const attr::BaseAttribute& definition = node
            ? static_cast<const attr::BaseAttribute&>(
                  network.node_attributes().at(id))
            : static_cast<const attr::BaseAttribute&>(
                  network.link_attributes().at(id));
        attr::AttributeBenchmarkDescriptor descriptor;
        descriptor.definition_id = id;
        descriptor.kind = definition.spec().kind;
        descriptor.name = definition.spec().name;
        if (definition.spec().kind == attr::AttributeKind::extrema) {
            if (node) {
                const auto* extrema = dynamic_cast<
                    const attr::NodeExtremaAttribute*>(&definition);
                if (extrema != nullptr) {
                    descriptor.originator_name =
                        std::string(extrema->originator_name());
                }
            } else {
                const auto* extrema = dynamic_cast<
                    const attr::LinkExtremaAttribute*>(&definition);
                if (extrema != nullptr) {
                    descriptor.originator_name =
                        std::string(extrema->originator_name());
                }
            }
        }
        result.attributes.push_back(std::move(descriptor));
        for (std::size_t column = 0U; column < rows[row].size(); ++column) {
            result.matrix.values.push_back(
                benchmark_float(rows[row][column], row, column));
        }
    }
    return result;
}

}  // namespace

BaseNetworkException::BaseNetworkException(
    BaseNetworkErrorCode code,
    BaseNetworkOperation operation,
    std::size_t input_index,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index) {}

BaseNetworkErrorCode BaseNetworkException::code() const noexcept {
    return code_;
}

BaseNetworkOperation BaseNetworkException::operation() const noexcept {
    return operation_;
}

std::size_t BaseNetworkException::input_index() const noexcept {
    return input_index_;
}

BaseNetworkConstruction base_network_construction_from_setting(
    std::optional<Graph> incoming_graph,
    const util::SettingDocument* config,
    std::vector<GraphAttributeAssignment> extra_graph_attributes,
    std::size_t factory_workers) {
    BaseNetworkConstruction result;
    result.incoming_graph = std::move(incoming_graph);
    result.extra_graph_attributes = std::move(extra_graph_attributes);
    result.config.factory_workers = factory_workers;

    util::SettingList incoming_node;
    util::SettingList incoming_link;
    if (result.incoming_graph.has_value()) {
        incoming_node = graph_setting_list(
            *result.incoming_graph,
            "node_attrs_setting",
            BaseNetworkOperation::decode_config);
        incoming_link = graph_setting_list(
            *result.incoming_graph,
            "link_attrs_setting",
            BaseNetworkOperation::decode_config);
    }

    util::SettingList configured_node;
    util::SettingList configured_link;
    if (config != nullptr) {
        result.config.source_config = *config;
        if (!setting_value_is_falsey(config->root)) {
            if (config->root.kind() != util::SettingValueKind::object) {
                throw_network(
                    BaseNetworkErrorCode::invalid_config,
                    BaseNetworkOperation::decode_config,
                    invalid_base_network_input_index,
                    "BaseNetwork config root must be an object");
            }
            const auto& root = config->root.as_object();
            configured_node = optional_setting_list(
                first_setting_field(
                    root, "node_attrs_setting", "node_attrs"),
                "node_attrs_setting");
            configured_link = optional_setting_list(
                first_setting_field(
                    root, "link_attrs_setting", "link_attrs"),
                "link_attrs_setting");

            if (const auto* value = setting_field(root, "topology")) {
                result.config.topology = attr_value_from_setting(
                    *value,
                    BaseNetworkOperation::decode_config,
                    invalid_base_network_input_index);
            }
            if (const auto* value = setting_field(root, "output")) {
                result.config.output = attr_value_from_setting(
                    *value,
                    BaseNetworkOperation::decode_config,
                    invalid_base_network_input_index);
            }
            result.config.graph_attributes = graph_assignments_from_setting(
                first_setting_field(
                    root, "graph_attrs_setting", "graph_attrs_dict"));
        }
    }

    const util::SettingList merged_node =
        merge_setting_lists(incoming_node, configured_node);
    const util::SettingList merged_link =
        merge_setting_lists(incoming_link, configured_link);
    result.config.node_attribute_specs = decode_factory_specs(merged_node);
    result.config.link_attribute_specs = decode_factory_specs(merged_link);

    if (result.incoming_graph.has_value()) {
        Graph& graph = *result.incoming_graph;
        const AttrId node_id = graph.attr_id("node_attrs_setting");
        const AttrId link_id = graph.attr_id("link_attrs_setting");
        graph.graph_attrs().set(
            node_id,
            attr_value_from_setting_list(
                setting_list_from_factory_specs(
                    result.config.node_attribute_specs),
                BaseNetworkOperation::decode_config));
        graph.graph_attrs().set(
            link_id,
            attr_value_from_setting_list(
                setting_list_from_factory_specs(
                    result.config.link_attribute_specs),
                BaseNetworkOperation::decode_config));
    }
    return result;
}

BaseNetwork::BaseNetwork() : BaseNetwork(BaseNetworkConstruction{}) {}

BaseNetwork::BaseNetwork(BaseNetworkConstruction construction)
    : graph_(construction.incoming_graph.has_value()
                 ? std::move(*construction.incoming_graph)
                 : Graph{}),
      config_snapshot_(std::move(construction.config.source_config)),
      factory_workers_(construction.config.factory_workers) {
    refresh_graph_field_ids();

    const util::SettingList incoming_node = graph_setting_list(
        graph_, "node_attrs_setting", BaseNetworkOperation::construct);
    const util::SettingList incoming_link = graph_setting_list(
        graph_, "link_attrs_setting", BaseNetworkOperation::construct);
    const util::SettingList configured_node =
        setting_list_from_factory_specs(
            construction.config.node_attribute_specs);
    const util::SettingList configured_link =
        setting_list_from_factory_specs(
            construction.config.link_attribute_specs);

    const util::SettingList merged_node =
        merge_setting_lists(incoming_node, configured_node);
    const util::SettingList merged_link =
        merge_setting_lists(incoming_link, configured_link);
    std::vector<attr::AttributeFactorySpec> node_specs =
        decode_factory_specs(merged_node);
    std::vector<attr::AttributeFactorySpec> link_specs =
        decode_factory_specs(merged_link);

    canonical_node_settings_ = setting_list_from_factory_specs(node_specs);
    canonical_link_settings_ = setting_list_from_factory_specs(link_specs);
    graph_.graph_attrs().set(
        node_settings_id_,
        attr_value_from_setting_list(
            canonical_node_settings_, BaseNetworkOperation::construct));
    graph_.graph_attrs().set(
        link_settings_id_,
        attr_value_from_setting_list(
            canonical_link_settings_, BaseNetworkOperation::construct));

    node_attributes_ = attr::create_node_attributes_from_specs(
        node_specs, factory_workers_);
    link_attributes_ = attr::create_link_attributes_from_specs(
        link_specs, factory_workers_);

    if (construction.config.topology.has_value()) {
        set_graph_attribute(
            topology_id_,
            clone_attr_value(*construction.config.topology));
    }
    if (construction.config.output.has_value()) {
        set_graph_attribute(
            output_id_,
            clone_attr_value(*construction.config.output));
    }
    set_graph_attrs_data(construction.config.graph_attributes);
    set_graph_attrs_data(construction.extra_graph_attributes);
    rebind_attribute_values();
}

BaseNetwork::BaseNetwork(BaseNetwork&& other)
    : graph_(std::move(other.graph_)),
      node_attributes_(std::move(other.node_attributes_)),
      link_attributes_(std::move(other.link_attributes_)),
      canonical_node_settings_(std::move(other.canonical_node_settings_)),
      canonical_link_settings_(std::move(other.canonical_link_settings_)),
      config_snapshot_(std::move(other.config_snapshot_)),
      factory_workers_(other.factory_workers_),
      cached_num_nodes_(other.cached_num_nodes_),
      cached_num_links_(other.cached_num_links_),
      cached_num_edges_(other.cached_num_edges_),
      topology_(std::move(other.topology_)),
      output_(std::move(other.output_)) {
    rebind_attribute_values();
}

BaseNetwork& BaseNetwork::operator=(BaseNetwork&& other) {
    if (this == &other) {
        return *this;
    }
    graph_ = std::move(other.graph_);
    node_attributes_ = std::move(other.node_attributes_);
    link_attributes_ = std::move(other.link_attributes_);
    canonical_node_settings_ = std::move(other.canonical_node_settings_);
    canonical_link_settings_ = std::move(other.canonical_link_settings_);
    config_snapshot_ = std::move(other.config_snapshot_);
    factory_workers_ = other.factory_workers_;
    cached_num_nodes_ = other.cached_num_nodes_;
    cached_num_links_ = other.cached_num_links_;
    cached_num_edges_ = other.cached_num_edges_;
    topology_ = std::move(other.topology_);
    output_ = std::move(other.output_);
    rebind_attribute_values();
    return *this;
}

const Graph& BaseNetwork::graph() const noexcept {
    return graph_;
}

Graph& BaseNetwork::graph() noexcept {
    return graph_;
}

const attr::NodeAttributeRegistry& BaseNetwork::node_attributes() const noexcept {
    return node_attributes_;
}

const attr::LinkAttributeRegistry& BaseNetwork::link_attributes() const noexcept {
    return link_attributes_;
}

const std::optional<util::SettingDocument>& BaseNetwork::config_snapshot()
    const noexcept {
    return config_snapshot_;
}

void BaseNetwork::refresh_graph_field_ids() {
    node_settings_id_ = graph_.attr_id("node_attrs_setting");
    link_settings_id_ = graph_.attr_id("link_attrs_setting");
    topology_id_ = graph_.attr_id("topology");
    output_id_ = graph_.attr_id("output");
    num_nodes_metadata_id_ = graph_.attr_id("num_nodes");
}

void BaseNetwork::sync_fixed_graph_fields() {
    if (const AttrValue* value = graph_.graph_attrs().find(topology_id_)) {
        topology_ = clone_attr_value(*value);
    } else {
        topology_.reset();
    }
    if (const AttrValue* value = graph_.graph_attrs().find(output_id_)) {
        output_ = clone_attr_value(*value);
    } else {
        output_.reset();
    }
}

void BaseNetwork::rebind_attribute_values() {
    refresh_graph_field_ids();
    std::vector<NodeNetworkAttributeBinding> node_bindings;
    node_bindings.reserve(node_attributes_.size());
    for (std::size_t index = 0U; index < node_attributes_.size(); ++index) {
        const auto id = static_cast<attr::AttributeRegistryId>(index);
        const auto& entry = node_attributes_.entries()[index];
        node_bindings.push_back(NodeNetworkAttributeBinding{
            id,
            graph_.attr_id(entry.name),
            &node_attributes_,
            &graph_});
    }

    std::vector<LinkNetworkAttributeBinding> link_bindings;
    link_bindings.reserve(link_attributes_.size());
    for (std::size_t index = 0U; index < link_attributes_.size(); ++index) {
        const auto id = static_cast<attr::AttributeRegistryId>(index);
        const auto& entry = link_attributes_.entries()[index];
        link_bindings.push_back(LinkNetworkAttributeBinding{
            id,
            graph_.attr_id(entry.name),
            &link_attributes_,
            &graph_});
    }
    node_bindings_ = std::move(node_bindings);
    link_bindings_ = std::move(link_bindings);
    sync_fixed_graph_fields();
}

const NodeNetworkAttributeBinding& BaseNetwork::checked_node_binding(
    attr::AttributeRegistryId id,
    BaseNetworkOperation operation,
    std::size_t input_index) const {
    if (static_cast<std::size_t>(id) >= node_bindings_.size()) {
        throw_network(
            BaseNetworkErrorCode::attribute_registry_mismatch,
            operation,
            input_index,
            "Node attribute registry ID is outside this network registry");
    }
    const auto& binding = node_bindings_[id];
    if (binding.registry_identity != &node_attributes_) {
        throw_network(
            BaseNetworkErrorCode::attribute_registry_mismatch,
            operation,
            input_index,
            "Node attribute binding belongs to another registry");
    }
    if (binding.graph_identity != &graph_) {
        throw_network(
            BaseNetworkErrorCode::graph_binding_mismatch,
            operation,
            input_index,
            "Node attribute binding belongs to another graph");
    }
    return binding;
}

const LinkNetworkAttributeBinding& BaseNetwork::checked_link_binding(
    attr::AttributeRegistryId id,
    BaseNetworkOperation operation,
    std::size_t input_index) const {
    if (static_cast<std::size_t>(id) >= link_bindings_.size()) {
        throw_network(
            BaseNetworkErrorCode::attribute_registry_mismatch,
            operation,
            input_index,
            "Link attribute registry ID is outside this network registry");
    }
    const auto& binding = link_bindings_[id];
    if (binding.registry_identity != &link_attributes_) {
        throw_network(
            BaseNetworkErrorCode::attribute_registry_mismatch,
            operation,
            input_index,
            "Link attribute binding belongs to another registry");
    }
    if (binding.graph_identity != &graph_) {
        throw_network(
            BaseNetworkErrorCode::graph_binding_mismatch,
            operation,
            input_index,
            "Link attribute binding belongs to another graph");
    }
    return binding;
}

std::optional<NodeNetworkAttributeBinding> BaseNetwork::bind_node_attribute(
    std::string_view name) const {
    const auto id = node_attributes_.bind(name);
    if (!id.has_value()) {
        return std::nullopt;
    }
    return checked_node_binding(
        *id,
        BaseNetworkOperation::bind_attribute,
        invalid_base_network_input_index);
}

std::optional<LinkNetworkAttributeBinding> BaseNetwork::bind_link_attribute(
    std::string_view name) const {
    const auto id = link_attributes_.bind(name);
    if (!id.has_value()) {
        return std::nullopt;
    }
    return checked_link_binding(
        *id,
        BaseNetworkOperation::bind_attribute,
        invalid_base_network_input_index);
}

void BaseNetwork::create_attrs_from_setting() {
    const AttrValue* node_snapshot =
        graph_.graph_attrs().find(node_settings_id_);
    if (node_snapshot == nullptr) {
        throw_network(
            BaseNetworkErrorCode::missing_config_field,
            BaseNetworkOperation::decode_config,
            invalid_base_network_input_index,
            "Missing node_attrs_setting metadata");
    }
    util::SettingList node_settings = setting_list_from_attr_value(
        *node_snapshot, BaseNetworkOperation::decode_config);
    attr::NodeAttributeRegistry new_nodes =
        attr::create_node_attributes_from_setting(
            node_settings, factory_workers_);
    node_attributes_ = std::move(new_nodes);
    canonical_node_settings_ = std::move(node_settings);
    rebind_attribute_values();

    const AttrValue* link_snapshot =
        graph_.graph_attrs().find(link_settings_id_);
    if (link_snapshot == nullptr) {
        throw_network(
            BaseNetworkErrorCode::missing_config_field,
            BaseNetworkOperation::decode_config,
            invalid_base_network_input_index,
            "Missing link_attrs_setting metadata");
    }
    util::SettingList link_settings = setting_list_from_attr_value(
        *link_snapshot, BaseNetworkOperation::decode_config);
    attr::LinkAttributeRegistry new_links =
        attr::create_link_attributes_from_setting(
            link_settings, factory_workers_);
    link_attributes_ = std::move(new_links);
    canonical_link_settings_ = std::move(link_settings);
    rebind_attribute_values();
}

std::size_t BaseNetwork::num_nodes() const {
    if (!cached_num_nodes_.has_value()) {
        cached_num_nodes_ = graph_.number_of_nodes();
    }
    return *cached_num_nodes_;
}

std::size_t BaseNetwork::num_links() const {
    if (!cached_num_links_.has_value()) {
        cached_num_links_ = graph_.number_of_edges();
    }
    return *cached_num_links_;
}

std::size_t BaseNetwork::num_edges() const {
    if (!cached_num_edges_.has_value()) {
        cached_num_edges_ = graph_.number_of_edges();
    }
    return *cached_num_edges_;
}

std::size_t BaseNetwork::live_num_nodes() const noexcept {
    return graph_.number_of_nodes();
}

std::size_t BaseNetwork::live_num_links() const noexcept {
    return graph_.number_of_edges();
}

void BaseNetwork::invalidate_cached_cardinalities() noexcept {
    cached_num_nodes_.reset();
    cached_num_links_.reset();
    cached_num_edges_.reset();
}

std::size_t BaseNetwork::num_node_features() const noexcept {
    return node_attributes_.size();
}

std::size_t BaseNetwork::num_link_features() const noexcept {
    return link_attributes_.size();
}

std::size_t BaseNetwork::num_node_resource_features() const noexcept {
    std::size_t result = 0U;
    for (const auto& entry : node_attributes_.entries()) {
        result += static_cast<std::size_t>(
            entry.attribute->spec().kind == attr::AttributeKind::resource);
    }
    return result;
}

std::size_t BaseNetwork::num_link_resource_features() const noexcept {
    std::size_t result = 0U;
    for (const auto& entry : link_attributes_.entries()) {
        result += static_cast<std::size_t>(
            entry.attribute->spec().kind == attr::AttributeKind::resource);
    }
    return result;
}

std::vector<attr::AttributeKind> BaseNetwork::get_node_attr_types() const {
    std::vector<attr::AttributeKind> result;
    result.reserve(node_attributes_.size());
    for (const auto& entry : node_attributes_.entries()) {
        result.push_back(entry.attribute->spec().kind);
    }
    return result;
}

std::vector<attr::AttributeKind> BaseNetwork::get_link_attr_types() const {
    std::vector<attr::AttributeKind> result;
    result.reserve(link_attributes_.size());
    for (const auto& entry : link_attributes_.entries()) {
        result.push_back(entry.attribute->spec().kind);
    }
    return result;
}

const AttrMap& BaseNetwork::graph_attributes() const noexcept {
    return graph_.graph_attrs();
}

AttrMap& BaseNetwork::graph_attributes() noexcept {
    return graph_.graph_attrs();
}

AttrId BaseNetwork::bind_graph_attribute(std::string_view name) {
    return graph_.attr_id(name);
}

const AttrValue& BaseNetwork::graph_attribute(AttrId id) const {
    return graph_.graph_attrs().at(id);
}

void BaseNetwork::set_graph_attribute(AttrId id, AttrValue value) {
    graph_.graph_attrs().set(id, std::move(value));
    if (id == num_nodes_metadata_id_) {
        return;
    }
    if (id == topology_id_) {
        topology_ = clone_attr_value(graph_.graph_attrs().at(id));
    } else if (id == output_id_) {
        output_ = clone_attr_value(graph_.graph_attrs().at(id));
    }
}

void BaseNetwork::set_graph_attrs_data(
    const std::vector<GraphAttributeAssignment>& values) {
    for (const auto& assignment : values) {
        const AttrId id = graph_.attr_id(assignment.name);
        set_graph_attribute(id, clone_attr_value(assignment.value));
    }
}

void BaseNetwork::init_graph_attrs() {
    refresh_graph_field_ids();
    if (graph_.graph_attrs().find(node_settings_id_) == nullptr) {
        graph_.graph_attrs().set(
            node_settings_id_,
            attr_value_from_setting_list(
                canonical_node_settings_, BaseNetworkOperation::construct));
    }
    if (graph_.graph_attrs().find(link_settings_id_) == nullptr) {
        graph_.graph_attrs().set(
            link_settings_id_,
            attr_value_from_setting_list(
                canonical_link_settings_, BaseNetworkOperation::construct));
    }
    sync_fixed_graph_fields();
}

std::vector<attr::AttributeRegistryId> BaseNetwork::select_node_attributes(
    const AttributeSelection& selection) const {
    std::vector<attr::AttributeRegistryId> result;
    result.reserve(node_attributes_.size());
    for (std::size_t index = 0U; index < node_attributes_.size(); ++index) {
        const auto id = static_cast<attr::AttributeRegistryId>(index);
        bool selected = true;
        if (selection.kinds.has_value()) {
            selected = contains_kind(
                *selection.kinds,
                node_attributes_.entries()[index].attribute->spec().kind);
        } else if (selection.ids.has_value()) {
            selected = std::find(
                           selection.ids->begin(),
                           selection.ids->end(),
                           id) != selection.ids->end();
        }
        if (selected) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<attr::AttributeRegistryId> BaseNetwork::select_link_attributes(
    const AttributeSelection& selection) const {
    std::vector<attr::AttributeRegistryId> result;
    result.reserve(link_attributes_.size());
    for (std::size_t index = 0U; index < link_attributes_.size(); ++index) {
        const auto id = static_cast<attr::AttributeRegistryId>(index);
        bool selected = true;
        if (selection.kinds.has_value()) {
            selected = contains_kind(
                *selection.kinds,
                link_attributes_.entries()[index].attribute->spec().kind);
        } else if (selection.ids.has_value()) {
            selected = std::find(
                           selection.ids->begin(),
                           selection.ids->end(),
                           id) != selection.ids->end();
        }
        if (selected) {
            result.push_back(id);
        }
    }
    return result;
}

void BaseNetwork::check_attrs_existence() const {
    if (graph_.number_of_nodes() == 0U) {
        throw_network(
            BaseNetworkErrorCode::no_nodes,
            BaseNetworkOperation::check_attributes,
            0U,
            "Cannot check attributes on a network without nodes");
    }
    if (graph_.number_of_edges() == 0U) {
        throw_network(
            BaseNetworkErrorCode::no_links,
            BaseNetworkOperation::check_attributes,
            0U,
            "Cannot check attributes on a network without links");
    }

    const AttrMap& first_node = graph_.node_attrs(0U);
    for (std::size_t index = 0U; index < node_attributes_.size(); ++index) {
        const auto& binding = checked_node_binding(
            static_cast<attr::AttributeRegistryId>(index),
            BaseNetworkOperation::check_attributes,
            index);
        if (first_node.find(binding.value_id) == nullptr) {
            throw_network(
                BaseNetworkErrorCode::missing_node_attribute,
                BaseNetworkOperation::check_attributes,
                index,
                "First node is missing attribute " +
                    node_attributes_.entries()[index].name);
        }
    }

    const auto [begin, end] = graph_.edges();
    if (begin == end) {
        throw_network(
            BaseNetworkErrorCode::no_links,
            BaseNetworkOperation::check_attributes,
            0U,
            "Cannot check attributes on a network without links");
    }
    const AttrMap& first_link = graph_.edge_attrs(*begin);
    for (std::size_t index = 0U; index < link_attributes_.size(); ++index) {
        const auto& binding = checked_link_binding(
            static_cast<attr::AttributeRegistryId>(index),
            BaseNetworkOperation::check_attributes,
            index);
        if (first_link.find(binding.value_id) == nullptr) {
            throw_network(
                BaseNetworkErrorCode::missing_link_attribute,
                BaseNetworkOperation::check_attributes,
                index,
                "First link is missing attribute " +
                    link_attributes_.entries()[index].name);
        }
    }
}

void BaseNetwork::generate_topology(const TopologyRequest& request) {
    PyRandom rng(request.seed);
    generate_topology(request, rng);
}

void BaseNetwork::generate_topology(
    const TopologyRequest& request,
    PyRandom& rng) {
    Graph generated = TopologyGenerator::generate(
        request.type, request.num_nodes, request.options, rng);
    AttrMap metadata = graph_.graph_attrs();
    graph_ = std::move(generated);
    graph_.graph_attrs().update(metadata);
    rebind_attribute_values();
}

void BaseNetwork::generate_attrs_data(
    NumpyRandomState& rng,
    bool node,
    bool link,
    std::size_t workers) {
    if (node) {
        for (std::size_t index = 0U; index < node_attributes_.size(); ++index) {
            const auto id = static_cast<attr::AttributeRegistryId>(index);
            const attr::NodeAttribute& definition = node_attributes_.at(id);
            const auto& spec = definition.spec();
            if (!spec.generative &&
                spec.kind != attr::AttributeKind::extrema) {
                continue;
            }
            const auto& binding = checked_node_binding(
                id, BaseNetworkOperation::generate_node_attributes, index);
            std::vector<AttrValue> values;
            if (spec.kind == attr::AttributeKind::extrema) {
                const auto* extrema =
                    dynamic_cast<const attr::NodeExtremaAttribute*>(&definition);
                if (extrema == nullptr) {
                    throw_network(
                        BaseNetworkErrorCode::attribute_registry_mismatch,
                        BaseNetworkOperation::generate_node_attributes,
                        index,
                        "Node extrema definition has an invalid concrete type");
                }
                const attr::AttributeRegistryId originator_id =
                    extrema->originator_id();
                const auto& originator_binding = checked_node_binding(
                    originator_id,
                    BaseNetworkOperation::generate_node_attributes,
                    index);
                values = extrema->generate_from_resolved_originator(
                    graph_,
                    node_attributes_.at(originator_id),
                    attr::NodeAttributeBinding{originator_binding.value_id},
                    workers);
            } else if (spec.kind == attr::AttributeKind::position) {
                const auto* position =
                    dynamic_cast<const attr::NodePositionAttribute*>(&definition);
                if (position == nullptr) {
                    throw_network(
                        BaseNetworkErrorCode::attribute_registry_mismatch,
                        BaseNetworkOperation::generate_node_attributes,
                        index,
                        "Node position definition has an invalid concrete type");
                }
                auto positions = position->generate_positions(
                    attr::NetworkCardinality{num_nodes(), num_links()},
                    rng,
                    workers);
                values.reserve(positions.size());
                for (auto& position_value : positions) {
                    std::vector<AttrValue> tuple;
                    tuple.reserve(3U);
                    tuple.push_back(
                        attribute_number_to_attr_value(position_value.x));
                    tuple.push_back(
                        attribute_number_to_attr_value(position_value.y));
                    tuple.emplace_back(position_value.radius);
                    values.push_back(make_attr_list(std::move(tuple)));
                }
            } else {
                values = generated_data_to_attr_values(
                    definition.generate_configured_data(
                        attr::NetworkCardinality{num_nodes(), num_links()},
                        rng,
                        workers));
            }
            const std::size_t expected = num_nodes();
            if (values.size() != expected) {
                throw_network(
                    BaseNetworkErrorCode::generated_length_mismatch,
                    BaseNetworkOperation::generate_node_attributes,
                    index,
                    "Generated node attribute length does not match cached "
                    "node count");
            }
            definition.set_data_dense(
                graph_,
                values,
                attr::NodeAttributeBinding{binding.value_id},
                workers);
        }
    }

    if (link) {
        for (std::size_t index = 0U; index < link_attributes_.size(); ++index) {
            const auto id = static_cast<attr::AttributeRegistryId>(index);
            const attr::LinkAttribute& definition = link_attributes_.at(id);
            const auto& spec = definition.spec();
            if (!spec.generative &&
                spec.kind != attr::AttributeKind::extrema) {
                continue;
            }
            const auto& binding = checked_link_binding(
                id, BaseNetworkOperation::generate_link_attributes, index);
            std::vector<AttrValue> values;
            if (spec.kind == attr::AttributeKind::extrema) {
                const auto* extrema =
                    dynamic_cast<const attr::LinkExtremaAttribute*>(&definition);
                if (extrema == nullptr) {
                    throw_network(
                        BaseNetworkErrorCode::attribute_registry_mismatch,
                        BaseNetworkOperation::generate_link_attributes,
                        index,
                        "Link extrema definition has an invalid concrete type");
                }
                const attr::AttributeRegistryId originator_id =
                    extrema->originator_id();
                const auto& originator_binding = checked_link_binding(
                    originator_id,
                    BaseNetworkOperation::generate_link_attributes,
                    index);
                values = extrema->generate_from_resolved_originator(
                    graph_,
                    link_attributes_.at(originator_id),
                    attr::LinkAttributeBinding{originator_binding.value_id},
                    workers);
            } else if (spec.kind == attr::AttributeKind::latency) {
                const auto* latency =
                    dynamic_cast<const attr::LinkLatencyAttribute*>(&definition);
                if (latency == nullptr) {
                    throw_network(
                        BaseNetworkErrorCode::attribute_registry_mismatch,
                        BaseNetworkOperation::generate_link_attributes,
                        index,
                        "Link latency definition has an invalid concrete type");
                }
                if (latency->generation_kind() ==
                    attr::LatencyGenerationKind::position) {
                    const attr::NodeAttributeBinding position_binding =
                        latency->resolve_position_binding(graph_);
                    auto generated = latency->generate_from_position(
                        graph_, position_binding, workers);
                    values.reserve(generated.size());
                    for (const double value : generated) {
                        values.emplace_back(value);
                    }
                } else {
                    values = generated_data_to_attr_values(
                        definition.generate_configured_data(
                            attr::NetworkCardinality{
                                num_nodes(), num_links()},
                            rng,
                            workers));
                }
            } else {
                values = generated_data_to_attr_values(
                    definition.generate_configured_data(
                        attr::NetworkCardinality{num_nodes(), num_links()},
                        rng,
                        workers));
            }
            const std::size_t expected = num_links();
            if (values.size() != expected) {
                throw_network(
                    BaseNetworkErrorCode::generated_length_mismatch,
                    BaseNetworkOperation::generate_link_attributes,
                    index,
                    "Generated link attribute length does not match cached "
                    "link count");
            }
            definition.set_data_dense(
                graph_,
                values,
                attr::LinkAttributeBinding{binding.value_id},
                workers);
        }
    }
}

void BaseNetwork::set_node_attrs_data(
    const std::vector<NodeAttributeDataUpdate>& updates,
    std::size_t workers) {
    for (std::size_t index = 0U; index < updates.size(); ++index) {
        const auto& update = updates[index];
        const auto& binding = checked_node_binding(
            update.registry_id,
            BaseNetworkOperation::set_attribute_data,
            index);
        const attr::NodeAttribute& definition =
            node_attributes_.at(update.registry_id);
        const attr::NodeAttributeBinding value_binding{binding.value_id};
        switch (update.layout) {
            case AttributeDataLayout::sparse:
                definition.set_data(
                    graph_, update.sparse_values, value_binding);
                break;
            case AttributeDataLayout::dense:
                definition.set_data_dense(
                    graph_, update.dense_values, value_binding, workers);
                break;
            default:
                throw_network(
                    BaseNetworkErrorCode::invalid_config,
                    BaseNetworkOperation::set_attribute_data,
                    index,
                    "Invalid node attribute data layout");
        }
    }
}

void BaseNetwork::set_link_attrs_data(
    const std::vector<LinkAttributeDataUpdate>& updates,
    std::size_t workers) {
    for (std::size_t index = 0U; index < updates.size(); ++index) {
        const auto& update = updates[index];
        const auto& binding = checked_link_binding(
            update.registry_id,
            BaseNetworkOperation::set_attribute_data,
            index);
        const attr::LinkAttribute& definition =
            link_attributes_.at(update.registry_id);
        const attr::LinkAttributeBinding value_binding{binding.value_id};
        switch (update.layout) {
            case AttributeDataLayout::sparse:
                definition.set_data(
                    graph_, update.sparse_values, value_binding);
                break;
            case AttributeDataLayout::dense:
                definition.set_data_dense(
                    graph_, update.dense_values, value_binding, workers);
                break;
            default:
                throw_network(
                    BaseNetworkErrorCode::invalid_config,
                    BaseNetworkOperation::set_attribute_data,
                    index,
                    "Invalid link attribute data layout");
        }
    }
}

std::vector<std::vector<AttrValue>> get_node_attrs_data(
    const BaseNetwork& network,
    const std::vector<attr::AttributeRegistryId>& definitions,
    std::size_t workers) {
    if (definitions.empty()) {
        throw_network(
            BaseNetworkErrorCode::empty_attribute_selection,
            BaseNetworkOperation::get_attribute_data,
            0U,
            "Node attribute selection is empty");
    }
    std::vector<std::vector<AttrValue>> result;
    result.reserve(definitions.size());
    for (std::size_t index = 0U; index < definitions.size(); ++index) {
        const auto id = definitions[index];
        const auto& binding = network.checked_node_binding(
            id, BaseNetworkOperation::get_attribute_data, index);
        result.push_back(network.node_attributes_.at(id).get_data(
            network.graph_,
            attr::NodeAttributeBinding{binding.value_id},
            workers));
    }
    return result;
}

std::vector<std::vector<AttrValue>> get_link_attrs_data(
    const BaseNetwork& network,
    const std::vector<attr::AttributeRegistryId>& definitions,
    std::size_t workers) {
    if (definitions.empty()) {
        throw_network(
            BaseNetworkErrorCode::empty_attribute_selection,
            BaseNetworkOperation::get_attribute_data,
            0U,
            "Link attribute selection is empty");
    }
    std::vector<std::vector<AttrValue>> result;
    result.reserve(definitions.size());
    for (std::size_t index = 0U; index < definitions.size(); ++index) {
        const auto id = definitions[index];
        const auto& binding = network.checked_link_binding(
            id, BaseNetworkOperation::get_attribute_data, index);
        result.push_back(network.link_attributes_.at(id).get_data(
            network.graph_,
            attr::LinkAttributeBinding{binding.value_id},
            workers));
    }
    return result;
}

std::vector<DistanceMatrix> get_adjacency_attrs_data(
    const BaseNetwork& network,
    const std::vector<attr::AttributeRegistryId>& definitions,
    bool normalized,
    std::size_t workers) {
    std::vector<DistanceMatrix> result;
    result.reserve(definitions.size());
    for (std::size_t index = 0U; index < definitions.size(); ++index) {
        const auto id = definitions[index];
        const auto& binding = network.checked_link_binding(
            id, BaseNetworkOperation::get_attribute_data, index);
        result.push_back(network.link_attributes_.at(id).get_adjacency_data(
            network.graph_,
            attr::LinkAttributeBinding{binding.value_id},
            normalized,
            workers));
    }
    return result;
}

std::vector<std::vector<double>> get_aggregation_attrs_data(
    const BaseNetwork& network,
    const std::vector<attr::AttributeRegistryId>& definitions,
    attr::LinkAggregation aggregation,
    bool normalized,
    std::size_t workers) {
    std::vector<std::vector<double>> result;
    result.reserve(definitions.size());
    for (std::size_t index = 0U; index < definitions.size(); ++index) {
        const auto id = definitions[index];
        const auto& binding = network.checked_link_binding(
            id, BaseNetworkOperation::get_attribute_data, index);
        if (aggregation == attr::LinkAggregation::sum && !normalized) {
            const SparseMatrix sparse = nx::attr_sparse_matrix(
                network.graph_, binding.value_id, false);
            std::vector<double> aggregate(network.graph_.num_nodes(), 0.0);
            // Frozen attr_sparse_matrix emits SciPy-compatible row-major COO.
            // Accumulating each column in that order is bit-identical to the
            // completed dense column reduction while avoiding an O(n^2)
            // matrix for sparse networks.
            for (std::size_t item = 0U; item < sparse.nnz(); ++item) {
                aggregate[sparse.col[item]] += sparse.value[item];
            }
            result.push_back(std::move(aggregate));
            continue;
        }
        result.push_back(network.link_attributes_.at(id).get_aggregation_data(
            network.graph_,
            attr::LinkAttributeBinding{binding.value_id},
            aggregation,
            normalized,
            workers));
    }
    return result;
}

attr::AttributeBenchmarkRequest prepare_attribute_benchmark_request(
    const BaseNetwork& network,
    const BaseNetworkBenchmarkSelection& selection) {
    attr::AttributeBenchmarkRequest request;
    request.workers = selection.workers;

    const AttributeSelection node_selection{selection.node_kinds, std::nullopt};
    const AttributeSelection link_selection{selection.link_kinds, std::nullopt};
    const auto node_definitions =
        network.select_node_attributes(node_selection);
    const auto link_definitions =
        network.select_link_attributes(link_selection);

    const auto extrema_requested = [](
                                       const auto& registry,
                                       const auto& definitions,
                                       const auto& requested_kinds) {
        if (requested_kinds.has_value()) {
            return contains_kind(
                *requested_kinds, attr::AttributeKind::extrema);
        }
        for (const auto id : definitions) {
            if (registry.at(id).spec().kind ==
                attr::AttributeKind::extrema) {
                return true;
            }
        }
        return false;
    };

    if (selection.node) {
        auto rows = get_node_attrs_data(
            network, node_definitions, selection.workers);
        request.node = prepare_benchmark_data(
            network,
            node_definitions,
            rows,
            true,
            extrema_requested(
                network.node_attributes(),
                node_definitions,
                selection.node_kinds),
            1U);
    }
    if (selection.link) {
        auto rows = get_link_attrs_data(
            network, link_definitions, selection.workers);
        request.link = prepare_benchmark_data(
            network,
            link_definitions,
            rows,
            false,
            extrema_requested(
                network.link_attributes(),
                link_definitions,
                selection.link_kinds),
            2U);
    }
    if (selection.link_sum) {
        const auto aggregate_rows = get_aggregation_attrs_data(
            network,
            link_definitions,
            attr::LinkAggregation::sum,
            false,
            selection.workers);
        std::vector<std::vector<AttrValue>> rows;
        rows.reserve(aggregate_rows.size());
        for (const auto& aggregate_row : aggregate_rows) {
            std::vector<AttrValue> row;
            row.reserve(aggregate_row.size());
            for (const double value : aggregate_row) {
                row.emplace_back(value);
            }
            rows.push_back(std::move(row));
        }
        request.link_sum = prepare_benchmark_data(
            network,
            link_definitions,
            rows,
            false,
            extrema_requested(
                network.link_attributes(),
                link_definitions,
                selection.link_kinds),
            1U);
    }
    return request;
}

attr::AttributeBenchmarks get_attribute_benchmarks(
    const BaseNetwork& network,
    const BaseNetworkBenchmarkSelection& selection) {
    return attr::AttributeBenchmarkManager::get_benchmarks(
        prepare_attribute_benchmark_request(network, selection));
}

CSRMatrix BaseNetwork::adjacency_matrix() const {
    return nx::to_scipy_sparse_matrix(graph_, "weight", "csr");
}

BaseNetworkView BaseNetwork::subgraph(
    const std::vector<Vertex>& nodes) const {
    return BaseNetworkView(*this, nx::subgraph(graph_, nodes));
}

BaseNetworkView BaseNetwork::subnetwork(
    const std::vector<Vertex>& nodes) const {
    return subgraph(nodes);
}

BaseNetworkView BaseNetwork::get_subgraph_view(
    nx::NodeFilter filter_node,
    nx::EdgeFilter filter_edge) const {
    return BaseNetworkView(
        *this,
        nx::subgraph_view(
            graph_, std::move(filter_node), std::move(filter_edge)));
}

BaseNetworkView BaseNetwork::get_subnetwork_view(
    nx::NodeFilter filter_node,
    nx::EdgeFilter filter_edge) const {
    return get_subgraph_view(
        std::move(filter_node), std::move(filter_edge));
}

std::string BaseNetwork::repr(std::string_view class_name) const {
    std::ostringstream stream;
    stream << class_name << "(num_nodes=" << num_nodes()
           << ", num_links=" << num_links() << ", node_attrs=[";
    for (std::size_t index = 0U; index < node_attributes_.size(); ++index) {
        if (index != 0U) {
            stream << ", ";
        }
        stream << quote_string(node_attributes_.entries()[index].name);
    }
    stream << "], link_attrs=[";
    for (std::size_t index = 0U; index < link_attributes_.size(); ++index) {
        if (index != 0U) {
            stream << ", ";
        }
        stream << quote_string(link_attributes_.entries()[index].name);
    }
    stream << "], graph_attrs={";
    bool first = true;
    for (const AttrId id : graph_.graph_attrs().attribute_ids()) {
        if (id == node_settings_id_ || id == link_settings_id_) {
            continue;
        }
        if (!first) {
            stream << ", ";
        }
        first = false;
        stream << quote_string(graph_.attr_name(id)) << ": "
               << formatted_dynamic_value(graph_.graph_attrs().at(id));
    }
    stream << "})";
    return stream.str();
}

BaseNetwork BaseNetwork::clone() const {
    BaseNetwork result;
    result.graph_ = graph_;
    result.node_attributes_ = attr::create_node_attributes_from_setting(
        canonical_node_settings_, factory_workers_);
    result.link_attributes_ = attr::create_link_attributes_from_setting(
        canonical_link_settings_, factory_workers_);
    result.canonical_node_settings_ = canonical_node_settings_;
    result.canonical_link_settings_ = canonical_link_settings_;
    result.config_snapshot_ = config_snapshot_;
    result.factory_workers_ = factory_workers_;
    result.cached_num_nodes_ = cached_num_nodes_;
    result.cached_num_links_ = cached_num_links_;
    result.cached_num_edges_ = cached_num_edges_;
    result.topology_ = topology_.has_value()
        ? std::optional<AttrValue>{clone_attr_value(*topology_)}
        : std::nullopt;
    result.output_ = output_.has_value()
        ? std::optional<AttrValue>{clone_attr_value(*output_)}
        : std::nullopt;
    result.rebind_attribute_values();
    return result;
}

Graph BaseNetwork::prepare_gml_graph() const {
    Graph result = graph_;
    result.graph_attrs().clear();

    const auto write_flattened_settings = [&](AttrId source_id) {
        const AttrValue* source = graph_.graph_attrs().find(source_id);
        util::SettingList settings;
        if (source != nullptr) {
            settings = setting_list_from_attr_value(
                *source, BaseNetworkOperation::prepare_gml);
        }
        auto dicts = dynamic_dict_list_from_setting_list(settings);
        dicts = util::flatten_dict_list_for_gml(dicts, 1U);
        const AttrId target_id = result.attr_id(graph_.attr_name(source_id));
        result.graph_attrs().set(
            target_id, attr_value_from_dynamic_dict_list(dicts));
    };

    write_flattened_settings(node_settings_id_);
    write_flattened_settings(link_settings_id_);

    for (const AttrId id : graph_.graph_attrs().attribute_ids()) {
        if (id == node_settings_id_ || id == link_settings_id_) {
            continue;
        }
        const std::string name(graph_.attr_name(id));
        const AttrValue& value = graph_.graph_attrs().at(id);
        if (const AttrObject* object = attr_object(value)) {
            util::DynamicValue::Dict source;
            source.reserve(object->entries.size());
            for (const auto& [subname, subvalue] : object->entries) {
                source.emplace_back(
                    util::DynamicKey(subname),
                    dynamic_value_from_attr(subvalue));
            }
            util::DynamicDictList flattened =
                util::flatten_dict_list_for_gml(
                    util::DynamicDictList{std::move(source)}, 1U);
            if (flattened.size() != 1U) {
                throw_network(
                    BaseNetworkErrorCode::invalid_config,
                    BaseNetworkOperation::prepare_gml,
                    invalid_base_network_input_index,
                    "Unexpected flattened graph metadata result");
            }
            for (const auto& [subkey, subvalue] : flattened.front()) {
                const std::string flattened_name =
                    name + "___" + dynamic_key_string(subkey);
                const AttrId target_id = result.attr_id(flattened_name);
                result.graph_attrs().set(
                    target_id, attr_value_from_dynamic(subvalue));
            }
        } else {
            const AttrId target_id = result.attr_id(name);
            result.graph_attrs().set(target_id, clone_attr_value(value));
        }
    }
    return result;
}

BaseNetwork BaseNetwork::from_gml(
    const std::string& path,
    std::string_view label) {
    Graph loaded = nx::read_gml(path, std::string(label));
    BaseNetworkConstruction construction;
    construction.incoming_graph = std::move(loaded);
    BaseNetwork result(std::move(construction));
    result.check_attrs_existence();

    const std::vector<AttrId> ids =
        result.graph_.graph_attrs().attribute_ids();
    constexpr std::string_view delimiter = "___";
    for (std::size_t index = 0U; index < ids.size(); ++index) {
        const AttrId id = ids[index];
        if (id == result.node_settings_id_ || id == result.link_settings_id_) {
            continue;
        }
        const std::string name(result.graph_.attr_name(id));
        const std::size_t first = name.find(delimiter);
        if (first == std::string::npos) {
            continue;
        }
        if (name.find(delimiter, first + delimiter.size()) !=
            std::string::npos) {
            throw_network(
                BaseNetworkErrorCode::invalid_gml_flattened_key,
                BaseNetworkOperation::restore_gml,
                index,
                "Flattened GML key must contain exactly one delimiter");
        }

        const std::string main_name = name.substr(0U, first);
        const std::string sub_name = name.substr(first + delimiter.size());
        const AttrValue restored_value =
            clone_attr_value(result.graph_.graph_attrs().at(id));
        const AttrId main_id = result.graph_.attr_id(main_name);
        AttrValue* main_value = result.graph_.graph_attrs().find(main_id);
        if (main_value == nullptr || attr_object(*main_value) == nullptr) {
            result.graph_.graph_attrs().set(
                main_id, make_attr_object());
            main_value = result.graph_.graph_attrs().find(main_id);
        }
        AttrObject* object = attr_object(*main_value);
        if (object == nullptr) {
            throw_network(
                BaseNetworkErrorCode::invalid_config,
                BaseNetworkOperation::restore_gml,
                index,
                "Unable to reconstruct flattened GML object");
        }
        object->set(sub_name, restored_value);
        result.graph_.graph_attrs().erase(id);
    }
    result.sync_fixed_graph_fields();
    return result;
}

void BaseNetwork::save_attrs_dict(const std::string& path) const {
    auto graph_attributes_object = std::make_shared<util::SettingObject>();
    graph_attributes_object->reserve(graph_.graph_attrs().size());
    std::size_t index = 0U;
    for (const AttrId id : graph_.graph_attrs().attribute_ids()) {
        graph_attributes_object->set(
            graph_.attr_name(id),
            setting_value_from_attr(
                graph_.graph_attrs().at(id),
                BaseNetworkOperation::save_attributes,
                index));
        ++index;
    }

    auto node_settings =
        std::make_shared<util::SettingList>(canonical_node_settings_);
    auto link_settings =
        std::make_shared<util::SettingList>(canonical_link_settings_);
    auto root = std::make_shared<util::SettingObject>();
    root->reserve(3U);
    root->set(
        "graph_attrs_dict",
        util::SettingValue(std::move(graph_attributes_object)));
    root->set("node_attrs", util::SettingValue(std::move(node_settings)));
    root->set("link_attrs", util::SettingValue(std::move(link_settings)));
    const util::SettingDocument document{util::SettingValue(std::move(root))};
    static_cast<void>(util::write_setting(document, path));
}

BaseNetworkView::BaseNetworkView(
    const BaseNetwork& parent,
    nx::GraphView graph_view)
    : parent_(&parent), graph_view_(std::move(graph_view)) {}

const nx::GraphView& BaseNetworkView::graph_view() const noexcept {
    return graph_view_;
}

const attr::NodeAttributeRegistry& BaseNetworkView::node_attributes()
    const noexcept {
    return parent_->node_attributes();
}

const attr::LinkAttributeRegistry& BaseNetworkView::link_attributes()
    const noexcept {
    return parent_->link_attributes();
}

const AttrMap& BaseNetworkView::graph_attributes() const noexcept {
    return parent_->graph_attributes();
}

std::size_t BaseNetworkView::num_nodes() const {
    const SearchMask& mask = graph_view_.mask();
    std::size_t count = 0U;
    for (Vertex node = 0U; node < parent_->graph_.num_nodes(); ++node) {
        count += static_cast<std::size_t>(mask.allows_node(node));
    }
    return count;
}

std::size_t BaseNetworkView::num_links() const {
    const SearchMask& mask = graph_view_.mask();
    std::size_t count = 0U;
    const auto [begin, end] = parent_->graph_.edges();
    for (auto edge = begin; edge != end; ++edge) {
        const Vertex source = parent_->graph_.source(*edge);
        const Vertex target = parent_->graph_.target(*edge);
        count += static_cast<std::size_t>(mask.allows(
            source,
            target,
            parent_->graph_.edge_id(*edge)));
    }
    return count;
}

const BaseNetwork& BaseNetworkView::parent() const noexcept {
    return *parent_;
}

}  // namespace virne::network
