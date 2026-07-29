#include "attribute/attribute_factory.h"

#include <array>
#include <cstddef>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

attribute::AttributeFactorySpec spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const attribute::AttributeKind kind) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = kind;
    return result;
}

template <typename Callable>
void expect_factory_error(
    Callable&& callable,
    const attribute::AttributeFactoryErrorCode code,
    const attribute::AttributeFactoryOperation operation,
    const std::optional<std::size_t> input_index,
    const std::string& context) {
    try {
        std::forward<Callable>(callable)();
    } catch (const attribute::AttributeFactoryException& error) {
        expect(error.code() == code, context + ": error-code drift");
        expect(error.operation() == operation, context + ": operation drift");
        expect(error.input_index() == input_index,
               context + ": input-index drift");
        expect(std::string_view(error.what()).empty() == false,
               context + ": missing diagnostic");
        return;
    }
    fail(context + ": expected AttributeFactoryException");
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

void test_all_registered_pairs_and_defaults() {
    using attribute::AttributeKind;
    using attribute::AttributeOwner;

    {
        auto value = attribute::create_attribute(
            spec("node-status", AttributeOwner::node, AttributeKind::status));
        expect(dynamic_cast<attribute::NodeStatusAttribute*>(value.get()) !=
                   nullptr,
               "node/status dispatch drift");
        expect(value->spec().owner == AttributeOwner::node &&
                   value->spec().kind == AttributeKind::status &&
                   !value->spec().generative,
               "node/status fixed fields drift");
    }
    {
        auto input = spec(
            "node-extrema", AttributeOwner::node, AttributeKind::extrema);
        input.originator_name = "cpu";
        input.originator_id = 7U;
        auto value = attribute::create_attribute(std::move(input));
        const auto* extrema =
            dynamic_cast<attribute::NodeExtremaAttribute*>(value.get());
        expect(extrema != nullptr && extrema->originator_name() == "cpu" &&
                   extrema->originator_id() == 7U,
               "node/extrema fields drift");
    }
    {
        auto input = spec(
            "cpu", AttributeOwner::node, AttributeKind::resource);
        input.generative = true;
        input.distribution.kind = utils::DistributionKind::uniform;
        input.distribution.low = std::int64_t{3};
        input.distribution.high = std::int64_t{9};
        input.dtype = utils::DatasetValueKind::integer;
        input.restriction = attribute::ConstraintRestriction::soft;
        input.checking_level = attribute::CheckingLevel::link;
        auto value = attribute::create_attribute(std::move(input));
        const auto* resource =
            dynamic_cast<attribute::NodeResourceAttribute*>(value.get());
        expect(resource != nullptr &&
                   resource->restriction() ==
                       attribute::ConstraintRestriction::soft &&
                   resource->checking_level() ==
                       attribute::CheckingLevel::link &&
                   resource->spec().generative &&
                   resource->spec().distribution.kind ==
                       utils::DistributionKind::uniform &&
                   resource->spec().dtype ==
                       std::optional<utils::DatasetValueKind>{
                           utils::DatasetValueKind::integer},
               "node/resource fields drift");
    }
    {
        auto input = spec(
            "pos", AttributeOwner::node, AttributeKind::position);
        input.generative = true;
        input.minimum_radius = 2.5;
        input.maximum_radius = 8.5;
        input.restriction = attribute::ConstraintRestriction::soft;
        auto value = attribute::create_attribute(std::move(input));
        const auto* position =
            dynamic_cast<attribute::NodePositionAttribute*>(value.get());
        expect(position != nullptr && position->minimum_radius() == 2.5 &&
                   position->maximum_radius() == 8.5 &&
                   position->restriction() ==
                       attribute::ConstraintRestriction::soft,
               "node/position fields drift");
    }
    {
        auto value = attribute::create_attribute(
            spec("link-status", AttributeOwner::link, AttributeKind::status));
        expect(dynamic_cast<attribute::LinkStatusAttribute*>(value.get()) !=
                   nullptr,
               "link/status dispatch drift");
    }
    {
        auto input = spec(
            "link-extrema", AttributeOwner::link, AttributeKind::extrema);
        input.originator_name = "bw";
        input.originator_id = 3U;
        auto value = attribute::create_attribute(std::move(input));
        const auto* extrema =
            dynamic_cast<attribute::LinkExtremaAttribute*>(value.get());
        expect(extrema != nullptr && extrema->originator_name() == "bw" &&
                   extrema->originator_id() == 3U,
               "link/extrema fields drift");
    }
    {
        auto input = spec(
            "bw", AttributeOwner::link, AttributeKind::resource);
        auto value = attribute::create_attribute(std::move(input));
        const auto* resource =
            dynamic_cast<attribute::LinkResourceAttribute*>(value.get());
        expect(resource != nullptr &&
                   resource->restriction() ==
                       attribute::ConstraintRestriction::hard &&
                   resource->checking_level() ==
                       attribute::CheckingLevel::link,
               "link/resource defaults drift");
    }
    {
        auto input = spec(
            "latency", AttributeOwner::link, AttributeKind::latency);
        input.generative = true;
        input.latency_generation =
            attribute::LatencyGenerationKind::position;
        input.minimum = std::int64_t{2};
        input.maximum = 7.5;
        auto value = attribute::create_attribute(std::move(input));
        const auto* latency =
            dynamic_cast<attribute::LinkLatencyAttribute*>(value.get());
        expect(latency != nullptr &&
                   latency->generation_kind() ==
                       attribute::LatencyGenerationKind::position &&
                   std::get<std::int64_t>(latency->minimum()) == 2 &&
                   std::get<double>(latency->maximum()) == 7.5 &&
                   latency->checking_level() == attribute::CheckingLevel::path,
               "link/latency fields drift");
    }
}

void test_unsupported_pairs_and_specific_helpers() {
    using attribute::AttributeKind;
    using attribute::AttributeOwner;

    expect_factory_error(
        [] {
            static_cast<void>(attribute::create_attribute(
                spec("graph", AttributeOwner::graph, AttributeKind::status)));
        },
        attribute::AttributeFactoryErrorCode::unsupported_pair,
        attribute::AttributeFactoryOperation::validate_pair,
        std::nullopt,
        "graph omission");
    expect_factory_error(
        [] {
            static_cast<void>(attribute::create_attribute(
                spec("bad", AttributeOwner::node, AttributeKind::latency)));
        },
        attribute::AttributeFactoryErrorCode::unsupported_pair,
        attribute::AttributeFactoryOperation::validate_pair,
        std::nullopt,
        "node/latency omission");
    expect_factory_error(
        [] {
            static_cast<void>(attribute::create_node_attribute(
                spec("link", AttributeOwner::link, AttributeKind::status)));
        },
        attribute::AttributeFactoryErrorCode::family_mismatch,
        attribute::AttributeFactoryOperation::validate_family,
        std::nullopt,
        "specific node mismatch");
    expect_factory_error(
        [] {
            static_cast<void>(attribute::create_link_attribute(
                spec("node", AttributeOwner::node, AttributeKind::status)));
        },
        attribute::AttributeFactoryErrorCode::family_mismatch,
        attribute::AttributeFactoryOperation::validate_family,
        std::nullopt,
        "specific link mismatch");
    expect_factory_error(
        [] {
            static_cast<void>(attribute::create_graph_attribute(
                spec("node", AttributeOwner::node, AttributeKind::status)));
        },
        attribute::AttributeFactoryErrorCode::family_mismatch,
        attribute::AttributeFactoryOperation::validate_family,
        std::nullopt,
        "specific graph mismatch after construction");
}

void test_setting_decode() {
    const auto raw = setting_object({
        {"name", utils::SettingValue("cpu")},
        {"owner", utils::SettingValue("node")},
        {"type", utils::SettingValue("resource")},
        {"generative", utils::SettingValue(true)},
        {"distribution", utils::SettingValue("uniform")},
        {"dtype", utils::SettingValue("int")},
        {"low", utils::SettingValue(std::int64_t{5})},
        {"high", utils::SettingValue(12.5)},
        {"constraint_restrictions", utils::SettingValue("soft")},
        {"restriction", utils::SettingValue("hard")},
        {"checking_level", utils::SettingValue("link")},
    });
    const attribute::AttributeFactorySpec decoded =
        attribute::attribute_factory_spec_from_setting(*raw);
    expect(decoded.name == "cpu" &&
               decoded.owner == attribute::AttributeOwner::node &&
               decoded.kind == attribute::AttributeKind::resource &&
               decoded.generative &&
               decoded.distribution.kind == utils::DistributionKind::uniform &&
               decoded.dtype == std::optional<utils::DatasetValueKind>{
                                      utils::DatasetValueKind::integer} &&
               std::get<std::int64_t>(*decoded.distribution.low) == 5 &&
               std::get<double>(*decoded.distribution.high) == 12.5 &&
               decoded.restriction ==
                   attribute::ConstraintRestriction::soft &&
               decoded.checking_level ==
                   std::optional<attribute::CheckingLevel>{
                       attribute::CheckingLevel::link},
           "raw setting decode drift");

    const auto latency_raw = setting_object({
        {"name", utils::SettingValue("delay")},
        {"owner", utils::SettingValue("link")},
        {"type", utils::SettingValue("latency")},
        {"generative", utils::SettingValue(true)},
        {"distribution", utils::SettingValue("position")},
        {"min", utils::SettingValue(std::int64_t{2})},
        {"max", utils::SettingValue(7.0)},
    });
    const attribute::AttributeFactorySpec latency =
        attribute::attribute_factory_spec_from_setting(*latency_raw);
    expect(latency.latency_generation ==
                   attribute::LatencyGenerationKind::position &&
               latency.distribution.kind == utils::DistributionKind::none &&
               std::get<std::int64_t>(latency.minimum) == 2 &&
               std::get<double>(latency.maximum) == 7.0,
           "position latency setting decode drift");

    expect_factory_error(
        [] {
            const auto missing = setting_object({
                {"name", utils::SettingValue("x")},
                {"type", utils::SettingValue("status")},
            });
            static_cast<void>(
                attribute::attribute_factory_spec_from_setting(*missing));
        },
        attribute::AttributeFactoryErrorCode::missing_owner,
        attribute::AttributeFactoryOperation::decode_owner,
        std::nullopt,
        "missing owner");
    expect_factory_error(
        [] {
            const auto unsupported = setting_object({
                {"name", utils::SettingValue("x")},
                {"owner", utils::SettingValue("graph")},
                {"type", utils::SettingValue("status")},
            });
            static_cast<void>(
                attribute::attribute_factory_spec_from_setting(*unsupported));
        },
        attribute::AttributeFactoryErrorCode::unsupported_pair,
        attribute::AttributeFactoryOperation::validate_pair,
        std::nullopt,
        "raw graph omission");
    expect_factory_error(
        [] {
            const auto invalid = setting_object({
                {"name", utils::SettingValue("x")},
                {"owner", utils::SettingValue("node")},
                {"type", utils::SettingValue("resource")},
                {"generative", utils::SettingValue(std::int64_t{1})},
            });
            static_cast<void>(
                attribute::attribute_factory_spec_from_setting(*invalid));
        },
        attribute::AttributeFactoryErrorCode::invalid_setting_value,
        attribute::AttributeFactoryOperation::decode_fields,
        std::nullopt,
        "invalid bool field");
}

std::vector<attribute::AttributeFactorySpec> node_specs() {
    std::vector<attribute::AttributeFactorySpec> result;
    result.push_back(spec(
        "cpu", attribute::AttributeOwner::node,
        attribute::AttributeKind::resource));
    auto extrema = spec(
        "peak", attribute::AttributeOwner::node,
        attribute::AttributeKind::extrema);
    extrema.originator_name = "cpu";
    result.push_back(std::move(extrema));
    result.push_back(spec(
        "middle", attribute::AttributeOwner::node,
        attribute::AttributeKind::status));
    auto replacement = spec(
        "cpu", attribute::AttributeOwner::node,
        attribute::AttributeKind::resource);
    replacement.restriction = attribute::ConstraintRestriction::soft;
    result.push_back(std::move(replacement));
    auto missing = spec(
        "missing-extrema", attribute::AttributeOwner::node,
        attribute::AttributeKind::extrema);
    missing.originator_name = "absent";
    result.push_back(std::move(missing));
    return result;
}

void verify_node_registry(
    const attribute::NodeAttributeRegistry& registry,
    const std::string& context) {
    expect(registry.size() == 4U, context + ": dedup size drift");
    expect(registry.entries()[0U].name == "cpu" &&
               registry.entries()[1U].name == "peak" &&
               registry.entries()[2U].name == "middle" &&
               registry.entries()[3U].name == "missing-extrema",
           context + ": insertion order drift");
    expect(registry.bind("cpu") ==
               std::optional<attribute::AttributeRegistryId>{0U} &&
               registry.bind("peak") ==
                   std::optional<attribute::AttributeRegistryId>{1U} &&
               !registry.bind("absent") && registry.find("absent") == nullptr,
           context + ": compact index drift");
    const auto* cpu = dynamic_cast<const attribute::NodeResourceAttribute*>(
        &registry.at(0U));
    expect(cpu != nullptr &&
               cpu->restriction() == attribute::ConstraintRestriction::soft,
           context + ": duplicate last-value drift");
    const auto* peak = dynamic_cast<const attribute::NodeExtremaAttribute*>(
        &registry.at(1U));
    const auto* missing = dynamic_cast<const attribute::NodeExtremaAttribute*>(
        &registry.at(3U));
    expect(peak != nullptr && peak->originator_id() == 0U,
           context + ": resolved originator ID drift");
    expect(missing != nullptr &&
               missing->originator_id() ==
                   attribute::invalid_attribute_registry_id,
           context + ": missing originator sentinel drift");
}

void test_registries_workers_and_move() {
    const auto inputs = node_specs();
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        auto registry =
            attribute::create_node_attributes_from_specs(inputs, workers);
        verify_node_registry(registry, "workers " + std::to_string(workers));
        attribute::NodeAttributeRegistry moved(std::move(registry));
        verify_node_registry(moved,
                             "moved workers " + std::to_string(workers));
        attribute::NodeAttributeRegistry assigned;
        assigned = std::move(moved);
        verify_node_registry(
            assigned, "move-assigned workers " + std::to_string(workers));
        expect_factory_error(
            [&assigned] { static_cast<void>(assigned.at(99U)); },
            attribute::AttributeFactoryErrorCode::invalid_registry_id,
            attribute::AttributeFactoryOperation::access_registry,
            std::nullopt,
            "invalid registry ID");
    }

    std::array<std::future<void>, 8U> calls;
    for (std::size_t index = 0U; index < calls.size(); ++index) {
        calls[index] = std::async(std::launch::async, [inputs, index] {
            const std::size_t workers = index % 4U == 0U ? 8U : 2U;
            const auto registry =
                attribute::create_node_attributes_from_specs(inputs, workers);
            verify_node_registry(
                registry, "concurrent " + std::to_string(index));
        });
    }
    for (auto& call : calls) {
        call.get();
    }

    const std::vector<attribute::AttributeFactorySpec> bad{
        spec("good", attribute::AttributeOwner::node,
             attribute::AttributeKind::status),
        spec("bad-family", attribute::AttributeOwner::link,
             attribute::AttributeKind::status),
        spec("bad-pair", attribute::AttributeOwner::node,
             attribute::AttributeKind::latency),
    };
    expect_factory_error(
        [&bad] {
            static_cast<void>(
                attribute::create_node_attributes_from_specs(bad, 8U));
        },
        attribute::AttributeFactoryErrorCode::family_mismatch,
        attribute::AttributeFactoryOperation::validate_family,
        std::nullopt,
        "lowest-index construction failure");
}

void test_setting_lists_and_input_index() {
    utils::SettingList values;
    values.push_back(setting_value({
        {"name", utils::SettingValue("a")},
        {"owner", utils::SettingValue("node")},
        {"type", utils::SettingValue("status")},
    }));
    values.push_back(setting_value({
        {"name", utils::SettingValue("b")},
        {"owner", utils::SettingValue("node")},
        {"type", utils::SettingValue("resource")},
    }));
    values.push_back(setting_value({
        {"name", utils::SettingValue("a")},
        {"owner", utils::SettingValue("node")},
        {"type", utils::SettingValue("position")},
    }));
    const auto registry =
        attribute::create_node_attributes_from_setting(values, 8U);
    expect(registry.size() == 2U && registry.entries()[0U].name == "a" &&
               registry.entries()[1U].name == "b" &&
               dynamic_cast<const attribute::NodePositionAttribute*>(
                   &registry.at(0U)) != nullptr,
           "setting-list duplicate/order drift");

    values.push_back(utils::SettingValue(std::int64_t{3}));
    expect_factory_error(
        [&values] {
            static_cast<void>(
                attribute::create_node_attributes_from_setting(values, 8U));
        },
        attribute::AttributeFactoryErrorCode::invalid_setting_item,
        attribute::AttributeFactoryOperation::decode_setting_list,
        std::optional<std::size_t>{3U},
        "setting-list item index");
}

}  // namespace

int main() {
    try {
        test_all_registered_pairs_and_defaults();
        test_unsupported_pairs_and_specific_helpers();
        test_setting_decode();
        test_registries_workers_and_move();
        test_setting_lists_and_input_index();
        std::cout << "attribute_factory_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attribute_factory_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
