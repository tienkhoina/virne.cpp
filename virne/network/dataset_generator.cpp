#include "dataset_generator.h"

#include "attribute/attribute_factory.h"
#include "../../config/config.h"
#include "../../random/random_context.h"
#include "../utils/dataset.h"
#include "../utils/setting.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace virne::network {
namespace {

using virne::utils::DatasetAttributeSpec;
using virne::utils::DatasetScalar;
using virne::utils::DatasetTopologyKind;
using virne::utils::PhysicalDatasetSetting;
using virne::utils::SettingDocument;
using virne::utils::SettingFormat;
using virne::utils::SettingKeyId;
using virne::utils::SettingList;
using virne::utils::SettingObject;
using virne::utils::SettingValue;
using virne::utils::SettingValueKind;
using virne::utils::VirtualDatasetSetting;

constexpr std::string_view physical_setting_key = "p_net_setting";
constexpr std::string_view virtual_setting_key = "v_sim_setting";

[[noreturn]] void invalid_generator_config(const std::string& message) {
    throw std::invalid_argument("dataset Generator: " + message);
}

const SettingValue* find_value(
    const SettingObject& object,
    std::string_view key) {
    const std::optional<SettingKeyId> id = object.find_key_id(key);
    return id ? &object.at(*id) : nullptr;
}

const SettingValue& require_value(
    const SettingObject& object,
    std::string_view key) {
    const SettingValue* value = find_value(object, key);
    if (value == nullptr) {
        invalid_generator_config(
            "missing fixed setting field '" + std::string(key) + "'");
    }
    return *value;
}

const SettingObject& require_object(
    const SettingObject& object,
    std::string_view key) {
    const SettingValue& value = require_value(object, key);
    if (value.kind() != SettingValueKind::object) {
        invalid_generator_config(
            "setting field '" + std::string(key) + "' must be an object");
    }
    return value.as_object();
}

const SettingList& require_list(
    const SettingObject& object,
    std::string_view key) {
    const SettingValue& value = require_value(object, key);
    if (value.kind() != SettingValueKind::list) {
        invalid_generator_config(
            "setting field '" + std::string(key) + "' must be a list");
    }
    return value.as_list();
}

const std::string& require_string(
    const SettingObject& object,
    std::string_view key) {
    const SettingValue& value = require_value(object, key);
    if (value.kind() != SettingValueKind::string) {
        invalid_generator_config(
            "setting field '" + std::string(key) + "' must be a string");
    }
    return value.as_string();
}

std::int64_t checked_int64(const virne::utils::SettingInteger& value) {
    const auto wide =
        value.convert_to<virne::utils::SettingInteger::BigInteger>();
    const virne::utils::SettingInteger::BigInteger minimum =
        std::numeric_limits<std::int64_t>::min();
    const virne::utils::SettingInteger::BigInteger maximum =
        std::numeric_limits<std::int64_t>::max();
    if (wide < minimum || wide > maximum) {
        invalid_generator_config("integer is outside the signed 64-bit range");
    }
    return wide.convert_to<std::int64_t>();
}

std::int64_t require_integer(
    const SettingObject& object,
    std::string_view key) {
    const SettingValue& value = require_value(object, key);
    if (value.kind() != SettingValueKind::integer) {
        invalid_generator_config(
            "setting field '" + std::string(key) + "' must be an integer");
    }
    return checked_int64(value.as_integer());
}

DatasetScalar scalar_from_setting(const SettingValue& value) {
    switch (value.kind()) {
    case SettingValueKind::null_value:
        return std::monostate{};
    case SettingValueKind::boolean:
        return value.as_bool();
    case SettingValueKind::integer:
        return checked_int64(value.as_integer());
    case SettingValueKind::real:
        return value.as_real();
    case SettingValueKind::string:
        return value.as_string();
    case SettingValueKind::list:
    case SettingValueKind::object:
        invalid_generator_config("dataset path scalar cannot be a collection");
    }
    invalid_generator_config("unknown setting scalar kind");
}

std::vector<DatasetAttributeSpec> path_attribute_specs(
    const std::vector<attribute::AttributeFactorySpec>& source) {
    std::vector<DatasetAttributeSpec> result;
    result.reserve(source.size());
    for (std::size_t index = 0U; index < source.size(); ++index) {
        if (index > std::numeric_limits<virne::utils::DatasetAttrId>::max()) {
            invalid_generator_config("too many dataset attributes");
        }
        result.push_back(DatasetAttributeSpec{
            static_cast<virne::utils::DatasetAttrId>(index),
            source[index].name,
            source[index].distribution});
    }
    return result;
}

std::vector<DatasetAttributeSpec> path_attribute_specs(
    const SettingObject& root,
    std::string_view key) {
    const SettingList& settings = require_list(root, key);
    std::vector<attribute::AttributeFactorySpec> decoded;
    decoded.reserve(settings.size());
    for (std::size_t index = 0U; index < settings.size(); ++index) {
        if (settings[index].kind() != SettingValueKind::object) {
            invalid_generator_config(
                "attribute setting at index " + std::to_string(index) +
                " must be an object");
        }
        decoded.push_back(attribute::attribute_factory_spec_from_setting(
            settings[index].as_object()));
    }
    return path_attribute_specs(decoded);
}

const SettingObject& root_object(const SettingDocument& document) {
    if (document.root.kind() != SettingValueKind::object) {
        invalid_generator_config("setting subtree must have an object root");
    }
    return document.root.as_object();
}

YAML::Node config_subtree_node(
    const Config& config,
    std::string_view key) {
    // This is the only Config string lookup for the subtree. Preserve the
    // Python order: missing membership fails before seeding, while an invalid
    // present value is decoded only after seeding.
    YAML::Node subtree = config.get_raw(std::string(key));
    if (!subtree.IsDefined()) {
        invalid_generator_config(
            "missing config subtree '" + std::string(key) + "'");
    }
    return subtree;
}

std::string config_subtree_yaml(
    const YAML::Node& subtree,
    std::string_view key) {
    // The resolved snapshot crosses yaml-cpp -> SettingDocument only once.
    // All generation after this cold adapter receives direct typed fields.
    if (subtree.IsNull()) {
        invalid_generator_config(
            "config subtree '" + std::string(key) + "' must be an object");
    }
    if (!subtree.IsMap()) {
        invalid_generator_config(
            "config subtree '" + std::string(key) + "' must be an object");
    }
    return YAML::Dump(subtree);
}

SettingDocument parse_config_subtree(const std::string& yaml) {
    return virne::utils::parse_setting(yaml, SettingFormat::yaml);
}

std::optional<std::uint32_t> generator_seed(
    const Config& config,
    GeneratorSeedMode mode) {
    const char* path = nullptr;
    switch (mode) {
    case GeneratorSeedMode::compatibility_root_seed:
        path = "seed";
        break;
    case GeneratorSeedMode::composed_experiment_seed:
        path = "experiment.seed";
        break;
    default:
        invalid_generator_config("unknown seed mode");
    }

    const YAML::Node value = config.get_raw(path);
    if (!value.IsDefined() || value.IsNull()) {
        return std::nullopt;
    }
    if (!value.IsScalar()) {
        invalid_generator_config(
            std::string("seed at '") + path + "' must be a scalar");
    }
    try {
        return value.as<std::uint32_t>();
    } catch (const YAML::Exception& error) {
        invalid_generator_config(
            std::string("seed at '") + path + "' is invalid: " + error.what());
    }
}

bool should_save(GeneratorPersistence persistence) {
    switch (persistence) {
    case GeneratorPersistence::memory_only:
        return false;
    case GeneratorPersistence::save:
        return true;
    }
    invalid_generator_config("unknown persistence mode");
}

PhysicalDatasetSetting physical_path_setting(
    const SettingDocument& document) {
    const SettingObject& root = root_object(document);
    const SettingObject& output = require_object(root, "output");
    const SettingObject& topology = require_object(root, "topology");

    PhysicalDatasetSetting result;
    result.save_dir = require_string(output, "save_dir");

    if (const SettingValue* file_path = find_value(topology, "file_path")) {
        if (!file_path->is_null()) {
            if (file_path->kind() != SettingValueKind::string) {
                invalid_generator_config("topology.file_path must be a string");
            }
            result.topology.file_path = file_path->as_string();
        }
    }

    bool loaded_topology_path = false;
    if (result.topology.file_path) {
        const std::string native = result.topology.file_path->string();
        if (!native.empty() && native != "None") {
            std::error_code error;
            loaded_topology_path =
                std::filesystem::exists(*result.topology.file_path, error) &&
                !error;
        }
    }

    // Python does not access generation-only topology fields on the existing
    // file branch. Otherwise they are required in this exact order.
    if (!loaded_topology_path) {
        result.topology.num_nodes = require_integer(topology, "num_nodes");
        result.topology.topology_type =
            virne::utils::dataset_topology_kind_from_string(
                require_string(topology, "type"));
        result.topology.wm_alpha =
            scalar_from_setting(require_value(topology, "wm_alpha"));
        result.topology.wm_beta =
            scalar_from_setting(require_value(topology, "wm_beta"));
    }

    result.node_attributes = path_attribute_specs(root, "node_attrs_setting");
    result.link_attributes = path_attribute_specs(root, "link_attrs_setting");
    return result;
}

VirtualDatasetSetting virtual_path_setting(
    const VirtualNetworkSimulationConfig& config) {
    VirtualDatasetSetting result;
    if (!config.output.save_dir) {
        invalid_generator_config("missing fixed setting field 'output.save_dir'");
    }
    result.save_dir = *config.output.save_dir;
    result.node_attributes = path_attribute_specs(config.node_attribute_specs);
    result.link_attributes = path_attribute_specs(config.link_attribute_specs);
    if (config.num_virtual_networks >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        invalid_generator_config("num_v_nets exceeds the signed 64-bit range");
    }
    result.num_virtual_networks =
        static_cast<std::int64_t>(config.num_virtual_networks);

    const auto require_bound = [](const std::optional<DatasetScalar>& value,
                                  const char* name) -> std::int64_t {
        if (!value || !std::holds_alternative<std::int64_t>(*value)) {
            invalid_generator_config(
                std::string("fixed integer distribution field '") + name +
                "' is missing or invalid");
        }
        return std::get<std::int64_t>(*value);
    };
    result.size_low = require_bound(
        config.virtual_network_size.distribution.low, "v_net_size.low");
    result.size_high = require_bound(
        config.virtual_network_size.distribution.high, "v_net_size.high");

    switch (config.topology_type) {
    case TopologyType::Path:
        result.topology_type = DatasetTopologyKind::path;
        break;
    case TopologyType::Star:
        result.topology_type = DatasetTopologyKind::star;
        break;
    case TopologyType::Grid2D:
        result.topology_type = DatasetTopologyKind::grid_2d;
        break;
    case TopologyType::Waxman:
        result.topology_type = DatasetTopologyKind::waxman;
        break;
    case TopologyType::Random:
        result.topology_type = DatasetTopologyKind::random;
        break;
    }

    result.lifetime = config.lifetime.distribution;
    if (!config.arrival_rate.distribution.lambda) {
        invalid_generator_config("missing fixed setting field 'arrival_rate.lam'");
    }
    result.arrival_lambda = *config.arrival_rate.distribution.lambda;
    return result;
}

double numeric_high(const DatasetScalar& value) {
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean ? 1.0 : 0.0;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return static_cast<double>(*integer);
    }
    if (const auto* real = std::get_if<double>(&value)) {
        return *real;
    }
    invalid_generator_config("scaled 'high' must be numeric");
}

std::int64_t python_scaled_integer(
    const DatasetScalar& value,
    const double factor) {
    const double scaled = numeric_high(value) * factor;
    if (!std::isfinite(scaled)) {
        invalid_generator_config("scaled 'high' is not finite");
    }
    const long double truncated = std::trunc(static_cast<long double>(scaled));
    if (truncated <
            static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        truncated >
            static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        invalid_generator_config("scaled 'high' exceeds the signed 64-bit range");
    }
    return static_cast<std::int64_t>(truncated);
}

void scale_attribute_highs(
    std::vector<attribute::AttributeFactorySpec>& attributes,
    const double factor) {
    for (attribute::AttributeFactorySpec& attribute : attributes) {
        if (attribute.distribution.high) {
            attribute.distribution.high = DatasetScalar{
                python_scaled_integer(*attribute.distribution.high, factor)};
        }
    }
}

void scale_virtual_size_high(
    VirtualNetworkSimulationConfig& config,
    const double factor) {
    if (config.virtual_network_size.distribution.high) {
        config.virtual_network_size.distribution.high = DatasetScalar{
            python_scaled_integer(
                *config.virtual_network_size.distribution.high,
                factor)};
    }
}

VirtualNetworkSimulationConfig changeable_stage_config(
    const VirtualNetworkSimulationConfig& base,
    const std::size_t stage) {
    // Recursive snapshots remain shared read-only. Every value changed by a
    // stage is a direct scalar field, so no YAML tree is searched or mutated.
    VirtualNetworkSimulationConfig result = base;
    switch (stage) {
    case 0U:
        scale_attribute_highs(result.node_attribute_specs, 1.5);
        scale_attribute_highs(result.link_attribute_specs, 1.5);
        break;
    case 1U:
        scale_attribute_highs(result.node_attribute_specs, 2.0);
        scale_attribute_highs(result.link_attribute_specs, 2.0);
        break;
    case 2U:
        scale_virtual_size_high(result, 1.5);
        break;
    case 3U:
        scale_virtual_size_high(result, 2.0);
        break;
    default:
        invalid_generator_config("changeable stage index is invalid");
    }
    return result;
}

}  // namespace

GeneratedDataset Generator::generate_dataset(
    const Config& config,
    RandomContext& random,
    const GeneratorSelection& selection,
    const GeneratorWorkers& workers,
    GeneratorSeedMode seed_mode) {
    GeneratedDataset result;

    // Preserve Python's physical-before-virtual ordering. Each selected leaf
    // resolves and applies its seed independently. With no seed the second
    // leaf continues the caller-owned streams; with a seed it starts afresh.
    if (selection.physical_network) {
        result.physical_network.emplace(
            generate_p_net_dataset_from_config(
                config,
                random,
                selection.persistence,
                workers,
                seed_mode));
    }
    if (selection.virtual_networks) {
        result.virtual_networks.emplace(
            generate_v_nets_dataset_from_config(
                config,
                random,
                selection.persistence,
                workers,
                seed_mode));
    }
    return result;
}

PhysicalNetwork Generator::generate_p_net_dataset_from_config(
    const Config& config,
    RandomContext& random,
    GeneratorPersistence persistence,
    const GeneratorWorkers& workers,
    GeneratorSeedMode seed_mode) {
    const YAML::Node subtree =
        config_subtree_node(config, physical_setting_key);
    random.set_seed(generator_seed(config, seed_mode));
    const std::string yaml =
        config_subtree_yaml(subtree, physical_setting_key);
    const SettingDocument setting = parse_config_subtree(yaml);

    PhysicalNetworkBuildOptions build_options;
    build_options.factory_workers = workers.physical_factory_workers;
    build_options.attribute_workers = workers.physical_attribute_workers;
    PhysicalNetwork result = PhysicalNetwork::from_setting(
        setting,
        random,
        build_options);

    if (should_save(persistence)) {
        const PhysicalDatasetSetting path_setting =
            physical_path_setting(setting);
        const std::filesystem::path directory =
            virne::utils::get_p_net_dataset_dir_from_setting(path_setting);
        result.save_dataset(directory.string());
    }
    return result;
}

VirtualNetworkRequestSimulator
Generator::generate_v_nets_dataset_from_config(
    const Config& config,
    RandomContext& random,
    GeneratorPersistence persistence,
    const GeneratorWorkers& workers,
    GeneratorSeedMode seed_mode) {
    const YAML::Node subtree =
        config_subtree_node(config, virtual_setting_key);
    random.set_seed(generator_seed(config, seed_mode));
    const std::string yaml =
        config_subtree_yaml(subtree, virtual_setting_key);
    const SettingDocument setting = parse_config_subtree(yaml);
    VirtualNetworkSimulationConfig simulation_config =
        virtual_network_simulation_config_from_setting(setting);
    VirtualNetworkRequestSimulator result =
        VirtualNetworkRequestSimulator::from_setting(
            std::move(simulation_config));
    result.renew(
        random,
        true,
        true,
        std::nullopt,
        workers.virtual_simulation);

    if (should_save(persistence)) {
        const VirtualDatasetSetting path_setting =
            virtual_path_setting(result.config());
        const std::filesystem::path directory =
            virne::utils::get_v_nets_dataset_dir_from_setting(path_setting);
        result.save_dataset(
            directory,
            workers.virtual_simulation.io_workers);
    }
    return result;
}

VirtualNetworkRequestSimulator
Generator::generate_changeable_v_nets_dataset_from_config(
    const Config& config,
    RandomContext& random,
    GeneratorPersistence persistence,
    const GeneratorWorkers& workers,
    GeneratorSeedMode seed_mode) {
    const YAML::Node subtree =
        config_subtree_node(config, virtual_setting_key);
    random.set_seed(generator_seed(config, seed_mode));
    const std::string base_yaml =
        config_subtree_yaml(subtree, virtual_setting_key);
    const SettingDocument base_setting = parse_config_subtree(base_yaml);
    VirtualNetworkSimulationConfig base_config =
        virtual_network_simulation_config_from_setting(base_setting);
    VirtualNetworkRequestSimulator base_simulator =
        VirtualNetworkRequestSimulator::from_setting(std::move(base_config));

    constexpr std::size_t stage_count = 4U;
    const std::size_t request_count =
        base_simulator.config().num_virtual_networks;
    if (request_count == 0U || request_count % stage_count != 0U) {
        invalid_generator_config(
            "num_v_nets must be positive and divisible by four");
    }

    const std::size_t quarter = request_count / stage_count;
    std::vector<VirtualNetwork> selected;
    selected.reserve(request_count);

    // The shared compatibility streams make the outer stages and requests
    // intentionally sequential. Completed dependency workers may transform
    // deterministic inner lanes, but they never draw from either stream.
    for (std::size_t stage = 0U; stage < stage_count; ++stage) {
        VirtualNetworkSimulationConfig stage_config =
            changeable_stage_config(base_simulator.config(), stage);
        VirtualNetworkRequestSimulator stage_simulator =
            VirtualNetworkRequestSimulator::from_setting(
                std::move(stage_config));
        stage_simulator.renew(
            random,
            true,
            false,
            std::nullopt,
            workers.virtual_simulation);

        std::vector<VirtualNetwork> generated =
            std::move(stage_simulator).release_v_nets();
        if (generated.size() != request_count) {
            throw std::runtime_error(
                "dataset Generator: stage generated an unexpected request count");
        }
        const std::size_t begin = stage * quarter;
        const std::size_t end = begin + quarter;
        for (std::size_t index = begin; index < end; ++index) {
            selected.push_back(std::move(generated[index]));
        }
    }

    VirtualNetworkSimulationConfig merged_config = base_simulator.config();
    VirtualNetworkRequestSimulator result =
        VirtualNetworkRequestSimulator::from_state(
            std::move(merged_config),
            std::move(selected),
            {});
    result.renew_events(workers.virtual_simulation.event_workers);

    if (should_save(persistence)) {
        VirtualNetworkSimulationConfig final_path_config =
            changeable_stage_config(base_simulator.config(), 3U);
        const VirtualDatasetSetting path_setting =
            virtual_path_setting(final_path_config);
        const std::filesystem::path directory =
            virne::utils::get_v_nets_dataset_dir_from_setting(path_setting);
        result.save_dataset(
            directory,
            workers.virtual_simulation.io_workers);
    }
    return result;
}

}  // namespace virne::network
