#include "physical_network.h"

#include "generators/gml_loader.h"
#include "io/graph_saver.h"
#include "random_context.h"

#include <charconv>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace virne::network {
namespace {

using virne::utils::SettingDocument;
using virne::utils::SettingObject;
using virne::utils::SettingValue;
using virne::utils::SettingValueKind;

constexpr std::string_view node_settings_name = "node_attrs_setting";
constexpr std::string_view link_settings_name = "link_attrs_setting";

const SettingObject& config_root(const SettingDocument& config)
{
    if (config.root.kind() != SettingValueKind::object) {
        throw std::invalid_argument(
            "PhysicalNetwork config root must be an object");
    }
    return config.root.as_object();
}

const SettingValue* find_setting_value(
    const SettingObject& object,
    std::string_view name)
{
    const auto id = object.find_key_id(name);
    return id.has_value() ? &object.at(*id) : nullptr;
}

const SettingObject* topology_object(const SettingDocument& config)
{
    const SettingObject& root = config_root(config);
    const SettingValue* topology = find_setting_value(root, "topology");
    if (topology == nullptr) {
        return nullptr;
    }
    if (topology->kind() != SettingValueKind::object) {
        throw std::invalid_argument(
            "PhysicalNetwork topology setting must be an object");
    }
    return &topology->as_object();
}

std::optional<std::string> configured_file_path(
    const SettingObject* topology)
{
    if (topology == nullptr) {
        return std::nullopt;
    }

    const SettingValue* file_path =
        find_setting_value(*topology, "file_path");
    if (file_path == nullptr || file_path->is_null()) {
        return std::nullopt;
    }
    if (file_path->kind() != SettingValueKind::string) {
        throw std::invalid_argument(
            "PhysicalNetwork topology.file_path must be a string or null");
    }
    if (file_path->as_string().empty()) {
        return std::nullopt;
    }
    return file_path->as_string();
}

std::int64_t setting_int64(
    const SettingValue& value,
    std::string_view field)
{
    if (value.kind() == SettingValueKind::boolean) {
        return value.as_bool() ? 1 : 0;
    }
    if (value.kind() != SettingValueKind::integer) {
        throw std::invalid_argument(
            "PhysicalNetwork topology." + std::string(field) +
            " must be an integer");
    }

    const std::string decimal = value.as_integer().convert_to<std::string>();
    std::int64_t result = 0;
    const char* const begin = decimal.data();
    const char* const end = begin + decimal.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::out_of_range(
            "PhysicalNetwork topology." + std::string(field) +
            " is outside signed 64-bit range");
    }
    return result;
}

double setting_double(
    const SettingValue& value,
    std::string_view field)
{
    switch (value.kind()) {
    case SettingValueKind::boolean:
        return value.as_bool() ? 1.0 : 0.0;
    case SettingValueKind::integer:
        return value.as_integer().convert_to<double>();
    case SettingValueKind::real:
        return value.as_real();
    default:
        throw std::invalid_argument(
            "PhysicalNetwork topology." + std::string(field) +
            " must be numeric");
    }
}

void decode_optional_int(
    const SettingObject& topology,
    std::string_view field,
    std::optional<std::int64_t>& output)
{
    const SettingValue* value = find_setting_value(topology, field);
    if (value == nullptr) {
        return;
    }
    if (value->is_null()) {
        throw std::invalid_argument(
            "PhysicalNetwork topology." + std::string(field) +
            " must not be null");
    }
    output = setting_int64(*value, field);
}

void decode_optional_double(
    const SettingObject& topology,
    std::string_view field,
    double& output)
{
    const SettingValue* value = find_setting_value(topology, field);
    if (value == nullptr) {
        return;
    }
    if (value->is_null()) {
        throw std::invalid_argument(
            "PhysicalNetwork topology." + std::string(field) +
            " must not be null");
    }
    output = setting_double(*value, field);
}

TopologyRequest generation_request(const SettingObject* topology)
{
    const SettingValue* num_nodes = topology == nullptr
        ? nullptr
        : find_setting_value(*topology, "num_nodes");
    if (num_nodes == nullptr || num_nodes->is_null()) {
        throw std::invalid_argument(
            "'num_nodes' must be specified in topology config for topology generation");
    }

    TopologyRequest request;
    request.type = TopologyType::Waxman;
    request.num_nodes = setting_int64(*num_nodes, "num_nodes");

    const SettingValue* type = find_setting_value(*topology, "type");
    if (type != nullptr) {
        if (type->kind() != SettingValueKind::string) {
            throw std::invalid_argument(
                "PhysicalNetwork topology.type must be a string");
        }
        request.type = topology_type_from_string(type->as_string());
    }

    switch (request.type) {
    case TopologyType::Grid2D:
        decode_optional_int(*topology, "m", request.options.m);
        decode_optional_int(*topology, "n", request.options.n);
        break;
    case TopologyType::Waxman:
        decode_optional_double(
            *topology, "wm_alpha", request.options.wm_alpha);
        decode_optional_double(
            *topology, "wm_beta", request.options.wm_beta);
        break;
    case TopologyType::Random:
        decode_optional_double(
            *topology, "random_prob", request.options.random_prob);
        break;
    case TopologyType::Path:
    case TopologyType::Star:
        break;
    }

    const SettingValue* max_attempts =
        find_setting_value(*topology, "max_attempts");
    if (max_attempts != nullptr && !max_attempts->is_null()) {
        const std::int64_t value =
            setting_int64(*max_attempts, "max_attempts");
        if (value < 0) {
            throw std::invalid_argument(
                "PhysicalNetwork topology.max_attempts must be non-negative");
        }
        request.options.max_attempts = static_cast<std::size_t>(value);
    }
    return request;
}

std::vector<GraphAttributeAssignment> loaded_graph_metadata(
    const Graph& graph)
{
    const auto node_settings_id =
        graph.attribute_registry().find(node_settings_name);
    const auto link_settings_id =
        graph.attribute_registry().find(link_settings_name);

    std::vector<GraphAttributeAssignment> result;
    result.reserve(graph.graph_attrs().size());
    for (const AttrId id : graph.graph_attrs().attribute_ids()) {
        if ((node_settings_id.has_value() && id == *node_settings_id) ||
            (link_settings_id.has_value() && id == *link_settings_id)) {
            continue;
        }
        result.push_back(GraphAttributeAssignment{
            std::string(graph.attr_name(id)),
            clone_attr_value(graph.graph_attrs().at(id))});
    }
    return result;
}

void strip_structural_gml_node_ids(Graph& graph)
{
    const auto id = graph.attribute_registry().find("id");
    if (!id.has_value()) {
        return;
    }
    for (Vertex node = 0U; node < graph.num_nodes(); ++node) {
        graph.node_attrs(node).erase(*id);
    }
}

attribute::AttributeFactorySpec status_spec(
    std::string_view name,
    attribute::AttributeOwner owner)
{
    attribute::AttributeFactorySpec spec;
    spec.name = std::string(name);
    spec.owner = owner;
    spec.kind = attribute::AttributeKind::status;
    spec.generative = false;
    return spec;
}

void append_unknown_sample_status_specs(
    const Graph& loaded,
    const PhysicalNetwork& configured,
    BaseNetworkConstruction& construction)
{
    if (loaded.num_nodes() != 0U) {
        const AttrMap& sample = loaded.node_attrs(0U);
        for (const AttrId id : sample.attribute_ids()) {
            const std::string_view name = loaded.attr_name(id);
            if (!configured.node_attributes().bind(name).has_value()) {
                construction.config.node_attribute_specs.push_back(
                    status_spec(name, attribute::AttributeOwner::node));
            }
        }
    }

    const auto edge_range = loaded.edges();
    if (edge_range.first != edge_range.second) {
        const AttrMap& sample = loaded.edge_attrs(*edge_range.first);
        for (const AttrId id : sample.attribute_ids()) {
            const std::string_view name = loaded.attr_name(id);
            if (!configured.link_attributes().bind(name).has_value()) {
                construction.config.link_attribute_specs.push_back(
                    status_spec(name, attribute::AttributeOwner::link));
            }
        }
    }
}

PhysicalNetwork load_configured_gml(
    const BaseNetworkConfig& typed_config,
    const PhysicalNetwork& configured,
    const std::string& file_path)
{
    Graph loaded = GmlLoader::load(file_path, "id");
    // The frozen dense GML loader retains its structural `id` as a node
    // attribute. NetworkX consumes that field when label="id", so this leaf
    // removes it through one resolved AttrId before first-sample discovery.
    strip_structural_gml_node_ids(loaded);
    std::vector<GraphAttributeAssignment> metadata =
        loaded_graph_metadata(loaded);
    // Python starts with configured graph metadata and then updates it from
    // the loaded graph. Clear only the incoming graph-level value map here;
    // replaying `metadata` as extra assignments below retains that precedence
    // and insertion order while node/edge maps keep their loaded values.
    loaded.graph_attrs().clear();

    BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(std::move(loaded));
    construction.config = typed_config;
    construction.extra_graph_attributes = std::move(metadata);

    append_unknown_sample_status_specs(
        *construction.incoming_graph,
        configured,
        construction);

    PhysicalNetwork result(std::move(construction));
    result.graph_attributes().erase(node_settings_name);
    result.graph_attributes().erase(link_settings_name);
    return result;
}

}  // namespace

PhysicalNetwork::PhysicalNetwork() = default;

PhysicalNetwork::PhysicalNetwork(BaseNetworkConstruction construction)
    : BaseNetwork(std::move(construction))
{
}

PhysicalNetwork::PhysicalNetwork(BaseNetwork&& network)
    : BaseNetwork(std::move(network))
{
}

PhysicalNetwork::PhysicalNetwork(PhysicalNetwork&& other) noexcept
    : BaseNetwork(std::move(other)),
      build_report_(std::move(other.build_report_))
{
}

PhysicalNetwork& PhysicalNetwork::operator=(PhysicalNetwork&& other) noexcept
{
    if (this != &other) {
        BaseNetwork::operator=(std::move(other));
        build_report_ = std::move(other.build_report_);
    }
    return *this;
}

PhysicalNetwork PhysicalNetwork::from_setting(
    const SettingDocument& config,
    const PhysicalNetworkBuildOptions& options)
{
    return from_setting(config, global_random_context(), options);
}

PhysicalNetwork PhysicalNetwork::from_setting(
    const SettingDocument& config,
    RandomContext& random,
    const PhysicalNetworkBuildOptions& options)
{
    const SettingObject* topology = topology_object(config);
    const std::optional<std::string> file_path =
        configured_file_path(topology);

    const BaseNetworkConstruction typed_construction =
        base_network_construction_from_setting(
            std::nullopt,
            &config,
            {},
            options.factory_workers);
    PhysicalNetwork network(typed_construction);

    random.set_seed(options.seed);
    network.build_report_.requested_file = file_path;

    if (file_path.has_value()) {
        std::error_code exists_error;
        const bool exists = std::filesystem::exists(*file_path, exists_error);
        if (!exists_error && exists) {
            bool loaded_successfully = false;
            try {
                network = load_configured_gml(
                    typed_construction.config, network, *file_path);
                loaded_successfully = true;
            } catch (const std::exception& error) {
                network.build_report_.origin =
                    PhysicalTopologyOrigin::generated_after_gml_error;
                network.build_report_.gml_error = error.what();
            }
            if (loaded_successfully) {
                network.build_report_.origin =
                    PhysicalTopologyOrigin::loaded_gml;
                network.build_report_.requested_file = file_path;
                network.generate_attrs_data(
                    random.numpy(), true, true, options.attribute_workers);
                return network;
            }
        }
    }

    const TopologyRequest request = generation_request(topology);
    network.generate_topology(request, random.python());
    network.generate_attrs_data(
        random.numpy(), true, true, options.attribute_workers);
    return network;
}

const PhysicalNetworkBuildReport& PhysicalNetwork::build_report() const noexcept
{
    return build_report_;
}

void PhysicalNetwork::to_gml(const std::string& path) const
{
    GraphSaver::save_gml(prepare_gml_graph(), path);
}

void PhysicalNetwork::save_dataset(
    const std::string& directory,
    const std::string& file_name) const
{
    const std::filesystem::path dataset_dir(directory);
    std::filesystem::create_directories(dataset_dir);
    to_gml((dataset_dir / file_name).string());
}

PhysicalNetwork PhysicalNetwork::load_dataset(
    const std::string& directory,
    const std::string& file_name)
{
    const std::filesystem::path path =
        std::filesystem::path(directory) / file_name;
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "Dataset file '" + file_name +
            "' not found in directory: " + directory + ".\n"
            "Please ensure the dataset is generated or the path is correct.");
    }

    PhysicalNetwork result(BaseNetwork::from_gml(path.string(), "id"));
    strip_structural_gml_node_ids(result.graph());
    result.build_report_.origin = PhysicalTopologyOrigin::loaded_gml;
    result.build_report_.requested_file = path.string();
    return result;
}

PhysicalNetwork PhysicalNetwork::clone() const
{
    PhysicalNetwork result(BaseNetwork::clone());
    result.build_report_ = build_report_;
    return result;
}

}  // namespace virne::network
