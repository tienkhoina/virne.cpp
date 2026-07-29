#include "attribute_factory.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

namespace virne::network::attribute {
namespace {

using virne::utils::DatasetScalar;
using virne::utils::SettingInteger;
using virne::utils::SettingKeyId;
using virne::utils::SettingObject;
using virne::utils::SettingValue;
using virne::utils::SettingValueKind;

[[noreturn]] void throw_factory(
    const AttributeFactoryErrorCode code,
    const AttributeFactoryOperation operation,
    std::string message) {
    throw AttributeFactoryException(code, operation, std::move(message));
}

bool supported_pair(
    const AttributeOwner owner,
    const AttributeKind kind) noexcept {
    switch (owner) {
    case AttributeOwner::node:
        return kind == AttributeKind::status ||
               kind == AttributeKind::extrema ||
               kind == AttributeKind::resource ||
               kind == AttributeKind::position;
    case AttributeOwner::link:
        return kind == AttributeKind::status ||
               kind == AttributeKind::extrema ||
               kind == AttributeKind::resource ||
               kind == AttributeKind::latency;
    case AttributeOwner::graph:
        return false;
    }
    return false;
}

void validate_supported_pair(
    const AttributeOwner owner,
    const AttributeKind kind) {
    if (!supported_pair(owner, kind)) {
        throw_factory(
            AttributeFactoryErrorCode::unsupported_pair,
            AttributeFactoryOperation::validate_pair,
            "The attribute owner/type pair is not registered.");
    }
}

const SettingValue& required_value(
    const SettingObject& setting,
    const std::string_view key,
    const AttributeFactoryErrorCode missing_code,
    const AttributeFactoryOperation operation) {
    const std::optional<SettingKeyId> id = setting.find_key_id(key);
    if (!id.has_value()) {
        throw_factory(
            missing_code,
            operation,
            "Required attribute factory field is missing: " +
                std::string(key));
    }
    return setting.at(*id);
}

std::string required_string(
    const SettingValue& value,
    const AttributeFactoryErrorCode invalid_code,
    const AttributeFactoryOperation operation,
    const std::string_view field) {
    if (value.kind() != SettingValueKind::string) {
        throw_factory(
            invalid_code,
            operation,
            "Attribute factory field must be a string: " +
                std::string(field));
    }
    return value.as_string();
}

DatasetScalar scalar_from_setting(
    const SettingValue& value,
    const std::string_view field) {
    switch (value.kind()) {
    case SettingValueKind::null_value:
        return std::monostate{};
    case SettingValueKind::boolean:
        return value.as_bool();
    case SettingValueKind::integer: {
        const SettingInteger::BigInteger integer =
            value.as_integer().convert_to<SettingInteger::BigInteger>();
        const SettingInteger::BigInteger minimum =
            std::numeric_limits<std::int64_t>::min();
        const SettingInteger::BigInteger maximum =
            std::numeric_limits<std::int64_t>::max();
        if (integer < minimum || integer > maximum) {
            throw_factory(
                AttributeFactoryErrorCode::invalid_setting_value,
                AttributeFactoryOperation::decode_fields,
                "Integer attribute factory field is outside int64: " +
                    std::string(field));
        }
        return integer.convert_to<std::int64_t>();
    }
    case SettingValueKind::real:
        return value.as_real();
    case SettingValueKind::string:
        return value.as_string();
    case SettingValueKind::list:
    case SettingValueKind::object:
        break;
    }
    throw_factory(
        AttributeFactoryErrorCode::invalid_setting_value,
        AttributeFactoryOperation::decode_fields,
        "Unsupported attribute factory scalar field: " +
            std::string(field));
}

AttributeNumber number_from_setting(
    const SettingValue& value,
    const std::string_view field) {
    const DatasetScalar scalar = scalar_from_setting(value, field);
    if (const auto* const boolean = std::get_if<bool>(&scalar)) {
        return *boolean;
    }
    if (const auto* const integer = std::get_if<std::int64_t>(&scalar)) {
        return *integer;
    }
    if (const auto* const real = std::get_if<double>(&scalar)) {
        return *real;
    }
    throw_factory(
        AttributeFactoryErrorCode::invalid_numeric_field,
        AttributeFactoryOperation::decode_fields,
        "Attribute factory field must be numeric: " + std::string(field));
}

double double_from_setting(
    const SettingValue& value,
    const std::string_view field) {
    const AttributeNumber number = number_from_setting(value, field);
    return std::visit(
        [](const auto lane) { return static_cast<double>(lane); }, number);
}

bool boolean_from_setting(
    const SettingValue& value,
    const std::string_view field) {
    if (value.kind() != SettingValueKind::boolean) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_setting_value,
            AttributeFactoryOperation::decode_fields,
            "Attribute factory field must be boolean: " +
                std::string(field));
    }
    return value.as_bool();
}

std::optional<std::string> optional_string_from_setting(
    const SettingValue& value,
    const std::string_view field) {
    if (value.kind() == SettingValueKind::null_value) {
        return std::nullopt;
    }
    if (value.kind() != SettingValueKind::string) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_setting_value,
            AttributeFactoryOperation::decode_fields,
            "Attribute factory field must be a string or null: " +
                std::string(field));
    }
    return value.as_string();
}

std::optional<SettingKeyId> resolve_key(
    const SettingObject& setting,
    const std::string_view name) {
    return setting.find_key_id(name);
}

const SettingValue* value_at(
    const SettingObject& setting,
    const std::optional<SettingKeyId>& id) noexcept {
    return id.has_value() ? &setting.at(*id) : nullptr;
}

CheckingLevel checking_level_from_setting(const SettingValue& value) {
    if (value.kind() == SettingValueKind::null_value) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_checking_level,
            AttributeFactoryOperation::decode_fields,
            "Attribute checking level cannot be null.");
    }
    if (value.kind() != SettingValueKind::string) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_checking_level,
            AttributeFactoryOperation::decode_fields,
            "Attribute checking level must be a string.");
    }
    const std::string_view name = value.as_string();
    if (name == "node") {
        return CheckingLevel::node;
    }
    if (name == "link") {
        return CheckingLevel::link;
    }
    if (name == "path") {
        return CheckingLevel::path;
    }
    if (name == "graph") {
        return CheckingLevel::graph;
    }
    throw_factory(
        AttributeFactoryErrorCode::invalid_checking_level,
        AttributeFactoryOperation::decode_fields,
        "Unknown attribute checking level.");
}

ConstraintRestriction restriction_from_setting(const SettingValue& value) {
    if (value.kind() != SettingValueKind::string) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_restriction,
            AttributeFactoryOperation::decode_fields,
            "Attribute restriction must be a string.");
    }
    try {
        return constraint_restriction_from_string(value.as_string());
    } catch (const AttributeMethodException&) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_restriction,
            AttributeFactoryOperation::decode_fields,
            "Unknown attribute restriction.");
    }
}

virne::utils::DatasetValueKind dtype_from_setting(const SettingValue& value) {
    if (value.kind() != SettingValueKind::string) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_dtype,
            AttributeFactoryOperation::decode_fields,
            "Attribute dtype must be a string.");
    }
    try {
        return virne::utils::dataset_value_kind_from_string(value.as_string());
    } catch (const virne::utils::DatasetException&) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_dtype,
            AttributeFactoryOperation::decode_fields,
            "Unknown attribute dtype.");
    }
}

void decode_distribution(
    AttributeFactorySpec& spec,
    const SettingObject& setting,
    const std::optional<SettingKeyId>& distribution_id,
    const std::optional<SettingKeyId>& low_id,
    const std::optional<SettingKeyId>& high_id,
    const std::optional<SettingKeyId>& loc_id,
    const std::optional<SettingKeyId>& scale_id,
    const std::optional<SettingKeyId>& lam_id,
    const std::optional<SettingKeyId>& minimum_id,
    const std::optional<SettingKeyId>& maximum_id,
    const std::optional<SettingKeyId>& reciprocal_id) {
    if (const SettingValue* const distribution =
            value_at(setting, distribution_id)) {
        if (distribution->is_null()) {
            spec.distribution.kind = virne::utils::DistributionKind::none;
        } else if (distribution->kind() != SettingValueKind::string) {
            throw_factory(
                AttributeFactoryErrorCode::invalid_distribution,
                AttributeFactoryOperation::decode_fields,
                "Attribute distribution must be a string.");
        } else {
            const std::string_view name = distribution->as_string();
            if (name == "position" && spec.owner == AttributeOwner::link &&
                spec.kind == AttributeKind::latency) {
                spec.latency_generation = LatencyGenerationKind::position;
                spec.distribution.kind = virne::utils::DistributionKind::none;
            } else {
                try {
                    spec.distribution.kind =
                        virne::utils::distribution_kind_from_string(name);
                } catch (const virne::utils::DatasetException&) {
                    throw_factory(
                        AttributeFactoryErrorCode::invalid_distribution,
                        AttributeFactoryOperation::decode_fields,
                        "Unknown attribute distribution.");
                }
            }
        }
    }

    const auto decode_scalar = [&setting](
                                   const std::optional<SettingKeyId>& id,
                                   const std::string_view field)
        -> std::optional<DatasetScalar> {
        const SettingValue* const value = value_at(setting, id);
        return value == nullptr
                   ? std::nullopt
                   : std::optional<DatasetScalar>{
                         scalar_from_setting(*value, field)};
    };

    spec.distribution.low = decode_scalar(low_id, "low");
    spec.distribution.high = decode_scalar(high_id, "high");
    spec.distribution.loc = decode_scalar(loc_id, "loc");
    spec.distribution.scale = decode_scalar(scale_id, "scale");
    spec.distribution.lambda = decode_scalar(lam_id, "lam");
    spec.distribution.minimum = decode_scalar(minimum_id, "min");
    spec.distribution.maximum = decode_scalar(maximum_id, "max");
    if (const SettingValue* const reciprocal =
            value_at(setting, reciprocal_id)) {
        spec.distribution.reciprocal =
            boolean_from_setting(*reciprocal, "reciprocal");
    }
}

std::pair<std::size_t, std::size_t> block_bounds(
    const std::size_t count,
    const std::size_t blocks,
    const std::size_t block) noexcept {
    const std::size_t base = count / blocks;
    const std::size_t remainder = count % blocks;
    const std::size_t begin = block * base + std::min(block, remainder);
    return {begin, begin + base + (block < remainder ? 1U : 0U)};
}

template <typename Attribute, typename Create>
std::vector<std::unique_ptr<Attribute>> construct_batch(
    const std::vector<AttributeFactorySpec>& specs,
    const std::vector<AttributeRegistryId>& originator_ids,
    const std::size_t configured_workers,
    Create create) {
    std::vector<std::unique_ptr<Attribute>> results(specs.size());
    std::vector<std::exception_ptr> errors(specs.size());

    const auto construct_range = [&](const std::size_t begin,
                                     const std::size_t end) noexcept {
        for (std::size_t index = begin; index < end; ++index) {
            try {
                AttributeFactorySpec prepared = specs[index];
                if (prepared.kind == AttributeKind::extrema &&
                    prepared.originator_name.has_value()) {
                    prepared.originator_id = originator_ids[index];
                }
                results[index] = create(std::move(prepared));
            } catch (...) {
                errors[index] = std::current_exception();
            }
        }
    };

    const std::size_t worker_count =
        configured_workers <= 1U || specs.empty()
            ? 1U
            : std::min(configured_workers, specs.size());
    if (worker_count == 1U) {
        construct_range(0U, specs.size());
    } else {
        std::vector<std::thread> threads;
        try {
            threads.reserve(worker_count - 1U);
        } catch (...) {
            construct_range(0U, specs.size());
            for (const std::exception_ptr& error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            return results;
        }

        std::size_t next_block = 1U;
        try {
            for (; next_block < worker_count; ++next_block) {
                const auto bounds =
                    block_bounds(specs.size(), worker_count, next_block);
                threads.emplace_back(
                    [&, bounds]() noexcept {
                        construct_range(bounds.first, bounds.second);
                    });
            }
        } catch (...) {
            for (std::size_t block = next_block; block < worker_count; ++block) {
                const auto bounds =
                    block_bounds(specs.size(), worker_count, block);
                construct_range(bounds.first, bounds.second);
            }
        }
        const auto first = block_bounds(specs.size(), worker_count, 0U);
        construct_range(first.first, first.second);
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return results;
}

std::vector<AttributeRegistryId> resolved_originator_ids(
    const std::vector<AttributeFactorySpec>& specs) {
    if (specs.size() >= static_cast<std::size_t>(
                            invalid_attribute_registry_id)) {
        throw_factory(
            AttributeFactoryErrorCode::registry_id_range,
            AttributeFactoryOperation::build_registry,
            "Attribute definition count exceeds the compact ID range.");
    }

    std::unordered_map<std::string_view, AttributeRegistryId> ids;
    ids.reserve(specs.size());
    AttributeRegistryId next_id = 0U;
    for (const AttributeFactorySpec& spec : specs) {
        const auto insertion = ids.try_emplace(spec.name, next_id);
        if (insertion.second) {
            ++next_id;
        }
    }

    std::vector<AttributeRegistryId> result(
        specs.size(), invalid_attribute_registry_id);
    for (std::size_t index = 0U; index < specs.size(); ++index) {
        const std::optional<std::string>& name = specs[index].originator_name;
        if (!name.has_value()) {
            if (specs[index].originator_id.has_value()) {
                result[index] = *specs[index].originator_id;
            }
            continue;
        }
        const auto iterator = ids.find(*name);
        if (iterator != ids.end()) {
            result[index] = iterator->second;
        }
    }
    return result;
}

std::vector<AttributeFactorySpec> specs_from_settings(
    const virne::utils::SettingList& settings) {
    std::vector<AttributeFactorySpec> specs;
    specs.reserve(settings.size());
    for (std::size_t index = 0U; index < settings.size(); ++index) {
        if (settings[index].kind() != SettingValueKind::object) {
            throw AttributeFactoryException(
                AttributeFactoryErrorCode::invalid_setting_item,
                AttributeFactoryOperation::decode_setting_list,
                "Attribute setting-list item must be an object.",
                index);
        }
        try {
            specs.emplace_back(
                attribute_factory_spec_from_setting(
                    settings[index].as_object()));
        } catch (const AttributeFactoryException& error) {
            throw AttributeFactoryException(
                error.code(),
                error.operation(),
                error.what(),
                index);
        }
    }
    return specs;
}

template <typename Entry, typename Attribute>
void insert_registry_entry(
    std::vector<Entry>& entries,
    std::unordered_map<std::string_view, AttributeRegistryId>& ids,
    const std::string_view name,
    std::unique_ptr<Attribute> attribute) {
    if (entries.size() >= static_cast<std::size_t>(
                              invalid_attribute_registry_id)) {
        throw_factory(
            AttributeFactoryErrorCode::registry_id_range,
            AttributeFactoryOperation::build_registry,
            "Attribute registry exceeds the compact ID range.");
    }
    const AttributeRegistryId id =
        static_cast<AttributeRegistryId>(entries.size());
    entries.push_back(Entry{std::string(name), std::move(attribute)});
    const auto insertion = ids.try_emplace(entries.back().name, id);
    if (!insertion.second) {
        std::unique_ptr<Attribute> replacement =
            std::move(entries.back().attribute);
        entries.pop_back();
        entries[static_cast<std::size_t>(insertion.first->second)].attribute =
            std::move(replacement);
    }
}

template <typename Entry, typename Attribute>
const Attribute& registry_at(
    const std::vector<Entry>& entries,
    const AttributeRegistryId id) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (id == invalid_attribute_registry_id || index >= entries.size()) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_registry_id,
            AttributeFactoryOperation::access_registry,
            "Attribute registry ID is outside the ordered registry.");
    }
    return *entries[index].attribute;
}

template <typename Entry, typename Attribute>
const Attribute* registry_find(
    const std::vector<Entry>& entries,
    const std::unordered_map<std::string_view, AttributeRegistryId>& ids,
    const std::string_view name) noexcept {
    const auto iterator = ids.find(name);
    return iterator == ids.end()
               ? nullptr
               : entries[static_cast<std::size_t>(iterator->second)]
                     .attribute.get();
}

}  // namespace

AttributeFactoryException::AttributeFactoryException(
    const AttributeFactoryErrorCode code,
    const AttributeFactoryOperation operation,
    std::string message,
    const std::optional<std::size_t> input_index)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index) {}

AttributeFactoryErrorCode AttributeFactoryException::code() const noexcept {
    return code_;
}

AttributeFactoryOperation AttributeFactoryException::operation() const noexcept {
    return operation_;
}

const std::optional<std::size_t>&
AttributeFactoryException::input_index() const noexcept {
    return input_index_;
}

AttributeFactorySpec attribute_factory_spec_from_setting(
    const SettingObject& setting) {
    AttributeFactorySpec spec;

    const SettingValue& name_value = required_value(
        setting,
        "name",
        AttributeFactoryErrorCode::missing_name,
        AttributeFactoryOperation::decode_name);
    spec.name = required_string(
        name_value,
        AttributeFactoryErrorCode::invalid_name,
        AttributeFactoryOperation::decode_name,
        "name");

    const SettingValue& owner_value = required_value(
        setting,
        "owner",
        AttributeFactoryErrorCode::missing_owner,
        AttributeFactoryOperation::decode_owner);
    if (owner_value.kind() == SettingValueKind::null_value ||
        (owner_value.kind() == SettingValueKind::boolean &&
         !owner_value.as_bool()) ||
        (owner_value.kind() == SettingValueKind::string &&
         owner_value.as_string().empty())) {
        throw_factory(
            AttributeFactoryErrorCode::missing_owner,
            AttributeFactoryOperation::decode_owner,
            "Attribute owner is false or empty.");
    }
    const std::string owner_name = required_string(
        owner_value,
        AttributeFactoryErrorCode::invalid_owner,
        AttributeFactoryOperation::decode_owner,
        "owner");
    try {
        spec.owner = attribute_owner_from_string(owner_name);
    } catch (const BaseAttributeException&) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_owner,
            AttributeFactoryOperation::decode_owner,
            "Unknown attribute owner.");
    }

    const SettingValue& kind_value = required_value(
        setting,
        "type",
        AttributeFactoryErrorCode::missing_kind,
        AttributeFactoryOperation::decode_kind);
    if (kind_value.kind() == SettingValueKind::null_value ||
        (kind_value.kind() == SettingValueKind::boolean &&
         !kind_value.as_bool()) ||
        (kind_value.kind() == SettingValueKind::string &&
         kind_value.as_string().empty())) {
        throw_factory(
            AttributeFactoryErrorCode::missing_kind,
            AttributeFactoryOperation::decode_kind,
            "Attribute type is false or empty.");
    }
    const std::string kind_name = required_string(
        kind_value,
        AttributeFactoryErrorCode::invalid_kind,
        AttributeFactoryOperation::decode_kind,
        "type");
    try {
        spec.kind = attribute_kind_from_string(kind_name);
    } catch (const BaseAttributeException&) {
        throw_factory(
            AttributeFactoryErrorCode::invalid_kind,
            AttributeFactoryOperation::decode_kind,
            "Unknown attribute type.");
    }

    validate_supported_pair(spec.owner, spec.kind);

    // Every fixed key below is resolved exactly once for this setting object.
    const auto generative_id = resolve_key(setting, "generative");
    const auto distribution_id = resolve_key(setting, "distribution");
    const auto dtype_id = resolve_key(setting, "dtype");
    const auto originator_id = resolve_key(setting, "originator");
    const auto constraint_restrictions_id =
        resolve_key(setting, "constraint_restrictions");
    const auto restriction_id = resolve_key(setting, "restriction");
    const auto checking_level_id = resolve_key(setting, "checking_level");
    const auto minimum_radius_id = resolve_key(setting, "min_r");
    const auto maximum_radius_id = resolve_key(setting, "max_r");
    const auto minimum_id = resolve_key(setting, "min");
    const auto maximum_id = resolve_key(setting, "max");
    const auto low_id = resolve_key(setting, "low");
    const auto high_id = resolve_key(setting, "high");
    const auto loc_id = resolve_key(setting, "loc");
    const auto scale_id = resolve_key(setting, "scale");
    const auto lam_id = resolve_key(setting, "lam");
    const auto reciprocal_id = resolve_key(setting, "reciprocal");

    if (const SettingValue* const generative =
            value_at(setting, generative_id)) {
        spec.generative = boolean_from_setting(*generative, "generative");
    }
    if (const SettingValue* const dtype = value_at(setting, dtype_id)) {
        if (dtype->kind() != SettingValueKind::null_value) {
            spec.dtype = dtype_from_setting(*dtype);
        }
    }
    if (const SettingValue* const originator =
            value_at(setting, originator_id)) {
        spec.originator_name =
            optional_string_from_setting(*originator, "originator");
    }

    const SettingValue* restriction =
        value_at(setting, constraint_restrictions_id);
    if (restriction == nullptr) {
        restriction = value_at(setting, restriction_id);
    }
    if (restriction != nullptr) {
        spec.restriction = restriction_from_setting(*restriction);
    }

    if (const SettingValue* const checking =
            value_at(setting, checking_level_id)) {
        if (checking->kind() != SettingValueKind::null_value) {
            spec.checking_level = checking_level_from_setting(*checking);
        }
    }
    if (const SettingValue* const minimum_radius =
            value_at(setting, minimum_radius_id)) {
        spec.minimum_radius =
            double_from_setting(*minimum_radius, "min_r");
    }
    if (const SettingValue* const maximum_radius =
            value_at(setting, maximum_radius_id)) {
        spec.maximum_radius =
            double_from_setting(*maximum_radius, "max_r");
    }
    if (const SettingValue* const minimum = value_at(setting, minimum_id)) {
        spec.minimum = number_from_setting(*minimum, "min");
    }
    if (const SettingValue* const maximum = value_at(setting, maximum_id)) {
        spec.maximum = number_from_setting(*maximum, "max");
    }

    decode_distribution(
        spec,
        setting,
        distribution_id,
        low_id,
        high_id,
        loc_id,
        scale_id,
        lam_id,
        minimum_id,
        maximum_id,
        reciprocal_id);
    return spec;
}

std::unique_ptr<BaseAttribute> create_attribute(AttributeFactorySpec spec) {
    validate_supported_pair(spec.owner, spec.kind);
    const AttributeRegistryId originator_id =
        spec.originator_id.value_or(invalid_attribute_registry_id);

    if (spec.owner == AttributeOwner::node) {
        switch (spec.kind) {
        case AttributeKind::status:
            return std::make_unique<NodeStatusAttribute>(NodeStatusSpec{
                std::move(spec.name),
                spec.generative,
                std::move(spec.distribution),
                spec.dtype});
        case AttributeKind::extrema:
            return std::make_unique<NodeExtremaAttribute>(NodeExtremaSpec{
                std::move(spec.name),
                std::move(spec.originator_name),
                originator_id});
        case AttributeKind::resource:
            return std::make_unique<NodeResourceAttribute>(NodeResourceSpec{
                std::move(spec.name),
                spec.generative,
                std::move(spec.distribution),
                spec.dtype,
                spec.restriction,
                spec.checking_level.value_or(CheckingLevel::node)});
        case AttributeKind::position:
            return std::make_unique<NodePositionAttribute>(NodePositionSpec{
                std::move(spec.name),
                spec.generative,
                std::move(spec.distribution),
                spec.dtype,
                spec.minimum_radius,
                spec.maximum_radius,
                spec.restriction});
        case AttributeKind::latency:
            break;
        }
    } else if (spec.owner == AttributeOwner::link) {
        switch (spec.kind) {
        case AttributeKind::status:
            return std::make_unique<LinkStatusAttribute>(LinkStatusSpec{
                std::move(spec.name),
                spec.generative,
                std::move(spec.distribution),
                spec.dtype});
        case AttributeKind::extrema:
            return std::make_unique<LinkExtremaAttribute>(LinkExtremaSpec{
                std::move(spec.name),
                std::move(spec.originator_name),
                originator_id});
        case AttributeKind::resource:
            return std::make_unique<LinkResourceAttribute>(LinkResourceSpec{
                std::move(spec.name),
                spec.generative,
                std::move(spec.distribution),
                spec.dtype,
                spec.restriction,
                spec.checking_level.value_or(CheckingLevel::link)});
        case AttributeKind::latency:
            return std::make_unique<LinkLatencyAttribute>(LinkLatencySpec{
                std::move(spec.name),
                spec.generative,
                spec.latency_generation,
                std::move(spec.distribution),
                spec.dtype,
                std::move(spec.minimum),
                std::move(spec.maximum),
                spec.restriction,
                spec.checking_level.value_or(CheckingLevel::path)});
        case AttributeKind::position:
            break;
        }
    }

    throw_factory(
        AttributeFactoryErrorCode::unsupported_pair,
        AttributeFactoryOperation::construct_attribute,
        "The attribute owner/type pair has no concrete constructor.");
}

std::unique_ptr<NodeAttribute> create_node_attribute(AttributeFactorySpec spec) {
    std::unique_ptr<BaseAttribute> attribute = create_attribute(std::move(spec));
    NodeAttribute* const node = dynamic_cast<NodeAttribute*>(attribute.get());
    if (node == nullptr) {
        throw_factory(
            AttributeFactoryErrorCode::family_mismatch,
            AttributeFactoryOperation::validate_family,
            "Constructed attribute is not a node attribute.");
    }
    attribute.release();
    return std::unique_ptr<NodeAttribute>(node);
}

std::unique_ptr<LinkAttribute> create_link_attribute(AttributeFactorySpec spec) {
    std::unique_ptr<BaseAttribute> attribute = create_attribute(std::move(spec));
    LinkAttribute* const link = dynamic_cast<LinkAttribute*>(attribute.get());
    if (link == nullptr) {
        throw_factory(
            AttributeFactoryErrorCode::family_mismatch,
            AttributeFactoryOperation::validate_family,
            "Constructed attribute is not a link attribute.");
    }
    attribute.release();
    return std::unique_ptr<LinkAttribute>(link);
}

std::unique_ptr<GraphAttribute> create_graph_attribute(AttributeFactorySpec spec) {
    std::unique_ptr<BaseAttribute> attribute = create_attribute(std::move(spec));
    GraphAttribute* const graph = dynamic_cast<GraphAttribute*>(attribute.get());
    if (graph == nullptr) {
        throw_factory(
            AttributeFactoryErrorCode::family_mismatch,
            AttributeFactoryOperation::validate_family,
            "Constructed attribute is not a graph attribute.");
    }
    attribute.release();
    return std::unique_ptr<GraphAttribute>(graph);
}

std::optional<AttributeRegistryId> AttributeRegistry::bind(
    const std::string_view name) const {
    const auto iterator = ids_.find(name);
    return iterator == ids_.end()
               ? std::nullopt
               : std::optional<AttributeRegistryId>{iterator->second};
}

const BaseAttribute& AttributeRegistry::at(const AttributeRegistryId id) const {
    return registry_at<AttributeRegistryEntry, BaseAttribute>(entries_, id);
}

const BaseAttribute* AttributeRegistry::find(const std::string_view name) const {
    return registry_find<AttributeRegistryEntry, BaseAttribute>(
        entries_, ids_, name);
}

const std::vector<AttributeRegistryEntry>&
AttributeRegistry::entries() const noexcept {
    return entries_;
}

std::size_t AttributeRegistry::size() const noexcept {
    return entries_.size();
}

void AttributeRegistry::reserve(const std::size_t size) {
    if (size >= static_cast<std::size_t>(invalid_attribute_registry_id)) {
        throw_factory(
            AttributeFactoryErrorCode::registry_id_range,
            AttributeFactoryOperation::build_registry,
            "Attribute registry reserve exceeds the compact ID range.");
    }
    entries_.reserve(size);
    ids_.reserve(size);
}

void AttributeRegistry::insert_or_assign(
    const std::string_view name,
    std::unique_ptr<BaseAttribute> attribute) {
    insert_registry_entry(entries_, ids_, name, std::move(attribute));
}

std::optional<AttributeRegistryId> NodeAttributeRegistry::bind(
    const std::string_view name) const {
    const auto iterator = ids_.find(name);
    return iterator == ids_.end()
               ? std::nullopt
               : std::optional<AttributeRegistryId>{iterator->second};
}

const NodeAttribute& NodeAttributeRegistry::at(
    const AttributeRegistryId id) const {
    return registry_at<NodeAttributeRegistryEntry, NodeAttribute>(entries_, id);
}

const NodeAttribute* NodeAttributeRegistry::find(
    const std::string_view name) const {
    return registry_find<NodeAttributeRegistryEntry, NodeAttribute>(
        entries_, ids_, name);
}

const std::vector<NodeAttributeRegistryEntry>&
NodeAttributeRegistry::entries() const noexcept {
    return entries_;
}

std::size_t NodeAttributeRegistry::size() const noexcept {
    return entries_.size();
}

void NodeAttributeRegistry::reserve(const std::size_t size) {
    if (size >= static_cast<std::size_t>(invalid_attribute_registry_id)) {
        throw_factory(
            AttributeFactoryErrorCode::registry_id_range,
            AttributeFactoryOperation::build_registry,
            "Node attribute registry reserve exceeds the compact ID range.");
    }
    entries_.reserve(size);
    ids_.reserve(size);
}

void NodeAttributeRegistry::insert_or_assign(
    const std::string_view name,
    std::unique_ptr<NodeAttribute> attribute) {
    insert_registry_entry(entries_, ids_, name, std::move(attribute));
}

std::optional<AttributeRegistryId> LinkAttributeRegistry::bind(
    const std::string_view name) const {
    const auto iterator = ids_.find(name);
    return iterator == ids_.end()
               ? std::nullopt
               : std::optional<AttributeRegistryId>{iterator->second};
}

const LinkAttribute& LinkAttributeRegistry::at(
    const AttributeRegistryId id) const {
    return registry_at<LinkAttributeRegistryEntry, LinkAttribute>(entries_, id);
}

const LinkAttribute* LinkAttributeRegistry::find(
    const std::string_view name) const {
    return registry_find<LinkAttributeRegistryEntry, LinkAttribute>(
        entries_, ids_, name);
}

const std::vector<LinkAttributeRegistryEntry>&
LinkAttributeRegistry::entries() const noexcept {
    return entries_;
}

std::size_t LinkAttributeRegistry::size() const noexcept {
    return entries_.size();
}

void LinkAttributeRegistry::reserve(const std::size_t size) {
    if (size >= static_cast<std::size_t>(invalid_attribute_registry_id)) {
        throw_factory(
            AttributeFactoryErrorCode::registry_id_range,
            AttributeFactoryOperation::build_registry,
            "Link attribute registry reserve exceeds the compact ID range.");
    }
    entries_.reserve(size);
    ids_.reserve(size);
}

void LinkAttributeRegistry::insert_or_assign(
    const std::string_view name,
    std::unique_ptr<LinkAttribute> attribute) {
    insert_registry_entry(entries_, ids_, name, std::move(attribute));
}

AttributeRegistry create_attributes_from_specs(
    const std::vector<AttributeFactorySpec>& specs,
    const std::size_t workers) {
    const std::vector<AttributeRegistryId> originator_ids =
        resolved_originator_ids(specs);
    std::vector<std::unique_ptr<BaseAttribute>> attributes =
        construct_batch<BaseAttribute>(
        specs, originator_ids, workers,
        [](AttributeFactorySpec spec) {
            return create_attribute(std::move(spec));
        });

    AttributeRegistry registry;
    registry.reserve(specs.size());
    for (std::size_t index = 0U; index < specs.size(); ++index) {
        registry.insert_or_assign(
            specs[index].name, std::move(attributes[index]));
    }
    return registry;
}

NodeAttributeRegistry create_node_attributes_from_specs(
    const std::vector<AttributeFactorySpec>& specs,
    const std::size_t workers) {
    const std::vector<AttributeRegistryId> originator_ids =
        resolved_originator_ids(specs);
    std::vector<std::unique_ptr<NodeAttribute>> attributes =
        construct_batch<NodeAttribute>(
        specs, originator_ids, workers,
        [](AttributeFactorySpec spec) {
            return create_node_attribute(std::move(spec));
        });

    NodeAttributeRegistry registry;
    registry.reserve(specs.size());
    for (std::size_t index = 0U; index < specs.size(); ++index) {
        registry.insert_or_assign(
            specs[index].name, std::move(attributes[index]));
    }
    return registry;
}

LinkAttributeRegistry create_link_attributes_from_specs(
    const std::vector<AttributeFactorySpec>& specs,
    const std::size_t workers) {
    const std::vector<AttributeRegistryId> originator_ids =
        resolved_originator_ids(specs);
    std::vector<std::unique_ptr<LinkAttribute>> attributes =
        construct_batch<LinkAttribute>(
        specs, originator_ids, workers,
        [](AttributeFactorySpec spec) {
            return create_link_attribute(std::move(spec));
        });

    LinkAttributeRegistry registry;
    registry.reserve(specs.size());
    for (std::size_t index = 0U; index < specs.size(); ++index) {
        registry.insert_or_assign(
            specs[index].name, std::move(attributes[index]));
    }
    return registry;
}

AttributeRegistry create_attributes_from_setting(
    const virne::utils::SettingList& settings,
    const std::size_t workers) {
    return create_attributes_from_specs(specs_from_settings(settings), workers);
}

NodeAttributeRegistry create_node_attributes_from_setting(
    const virne::utils::SettingList& settings,
    const std::size_t workers) {
    return create_node_attributes_from_specs(
        specs_from_settings(settings), workers);
}

LinkAttributeRegistry create_link_attributes_from_setting(
    const virne::utils::SettingList& settings,
    const std::size_t workers) {
    return create_link_attributes_from_specs(
        specs_from_settings(settings), workers);
}

}  // namespace virne::network::attribute
