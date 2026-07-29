#pragma once

#include "attribute/attribute_benchmark_manager.h"
#include "attribute/attribute_factory.h"
#include "topology/topology_generator.h"
#include "../utils/setting.h"

#include "graph.h"
#include "csr_matrix.h"
#include "nx/subgraph.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

class NumpyRandomState;
class PyRandom;

namespace virne::network {

struct NodeNetworkAttributeBinding {
    attribute::AttributeRegistryId registry_id = 0U;
    AttrId value_id = 0U;
    const attribute::NodeAttributeRegistry* registry_identity = nullptr;
    const Graph* graph_identity = nullptr;
};

struct LinkNetworkAttributeBinding {
    attribute::AttributeRegistryId registry_id = 0U;
    AttrId value_id = 0U;
    const attribute::LinkAttributeRegistry* registry_identity = nullptr;
    const Graph* graph_identity = nullptr;
};

struct GraphAttributeAssignment {
    std::string name;
    AttrValue value = std::int64_t{0};
};

struct BaseNetworkConfig {
    std::vector<attribute::AttributeFactorySpec> node_attribute_specs;
    std::vector<attribute::AttributeFactorySpec> link_attribute_specs;
    std::optional<AttrValue> topology;
    std::optional<AttrValue> output;
    std::vector<GraphAttributeAssignment> graph_attributes;
    std::optional<virne::utils::SettingDocument> source_config;
    std::size_t factory_workers = 1U;
};

struct BaseNetworkConstruction {
    std::optional<Graph> incoming_graph;
    BaseNetworkConfig config;
    std::vector<GraphAttributeAssignment> extra_graph_attributes;
};

struct AttributeSelection {
    std::optional<std::vector<attribute::AttributeKind>> kinds;
    std::optional<std::vector<attribute::AttributeRegistryId>> ids;
};

enum class AttributeDataLayout : std::uint8_t {
    sparse,
    dense,
};

struct NodeAttributeDataUpdate {
    attribute::AttributeRegistryId registry_id = 0U;
    AttributeDataLayout layout = AttributeDataLayout::dense;
    std::vector<attribute::NodeAttributeAssignment> sparse_values;
    std::vector<AttrValue> dense_values;
};

struct LinkAttributeDataUpdate {
    attribute::AttributeRegistryId registry_id = 0U;
    AttributeDataLayout layout = AttributeDataLayout::dense;
    std::vector<attribute::LinkAttributeAssignment> sparse_values;
    std::vector<AttrValue> dense_values;
};

BaseNetworkConstruction base_network_construction_from_setting(
    std::optional<Graph> incoming_graph,
    const virne::utils::SettingDocument* config,
    std::vector<GraphAttributeAssignment> extra_graph_attributes = {},
    std::size_t factory_workers = 1U);

enum class BaseNetworkErrorCode : std::uint8_t {
    invalid_config,
    missing_config_field,
    attribute_registry_mismatch,
    graph_binding_mismatch,
    no_nodes,
    no_links,
    missing_node_attribute,
    missing_link_attribute,
    generated_length_mismatch,
    empty_attribute_selection,
    non_numeric_benchmark_value,
    ragged_benchmark_matrix,
    invalid_gml_flattened_key,
};

enum class BaseNetworkOperation : std::uint8_t {
    decode_config,
    construct,
    bind_attribute,
    check_attributes,
    generate_topology,
    generate_node_attributes,
    generate_link_attributes,
    get_attribute_data,
    set_attribute_data,
    prepare_benchmarks,
    prepare_gml,
    restore_gml,
    save_attributes,
};

inline constexpr std::size_t invalid_base_network_input_index =
    std::numeric_limits<std::size_t>::max();

class BaseNetworkException : public std::runtime_error {
public:
    BaseNetworkException(
        BaseNetworkErrorCode code,
        BaseNetworkOperation operation,
        std::size_t input_index,
        std::string message);

    BaseNetworkErrorCode code() const noexcept;
    BaseNetworkOperation operation() const noexcept;
    std::size_t input_index() const noexcept;

private:
    BaseNetworkErrorCode code_;
    BaseNetworkOperation operation_;
    std::size_t input_index_;
};

class BaseNetwork;
class BaseNetworkView;

std::vector<std::vector<AttrValue>> get_node_attrs_data(
    const BaseNetwork& network,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    std::size_t workers = 1U);

std::vector<std::vector<AttrValue>> get_link_attrs_data(
    const BaseNetwork& network,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    std::size_t workers = 1U);

std::vector<DistanceMatrix> get_adjacency_attrs_data(
    const BaseNetwork& network,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    bool normalized = false,
    std::size_t workers = 1U);

std::vector<std::vector<double>> get_aggregation_attrs_data(
    const BaseNetwork& network,
    const std::vector<attribute::AttributeRegistryId>& definitions,
    attribute::LinkAggregation aggregation = attribute::LinkAggregation::sum,
    bool normalized = false,
    std::size_t workers = 1U);

struct BaseNetworkBenchmarkSelection {
    bool node = true;
    bool link = true;
    bool link_sum = true;
    std::optional<std::vector<attribute::AttributeKind>> node_kinds =
        std::vector<attribute::AttributeKind>{
            attribute::AttributeKind::resource,
            attribute::AttributeKind::extrema};
    std::optional<std::vector<attribute::AttributeKind>> link_kinds =
        std::vector<attribute::AttributeKind>{
            attribute::AttributeKind::resource,
            attribute::AttributeKind::extrema};
    std::size_t workers = 1U;
};

attribute::AttributeBenchmarkRequest prepare_attribute_benchmark_request(
    const BaseNetwork& network,
    const BaseNetworkBenchmarkSelection& selection);

attribute::AttributeBenchmarks get_attribute_benchmarks(
    const BaseNetwork& network,
    const BaseNetworkBenchmarkSelection& selection);

class BaseNetwork {
public:
    BaseNetwork();
    explicit BaseNetwork(BaseNetworkConstruction construction);
    BaseNetwork(const BaseNetwork&) = delete;
    BaseNetwork& operator=(const BaseNetwork&) = delete;
    BaseNetwork(BaseNetwork&& other);
    BaseNetwork& operator=(BaseNetwork&& other);

    const Graph& graph() const noexcept;
    Graph& graph() noexcept;
    const attribute::NodeAttributeRegistry& node_attributes() const noexcept;
    const attribute::LinkAttributeRegistry& link_attributes() const noexcept;
    const std::optional<virne::utils::SettingDocument>& config_snapshot()
        const noexcept;
    void create_attrs_from_setting();

    std::optional<NodeNetworkAttributeBinding> bind_node_attribute(
        std::string_view name) const;
    std::optional<LinkNetworkAttributeBinding> bind_link_attribute(
        std::string_view name) const;
    void rebind_attribute_values();

    std::size_t num_nodes() const;
    std::size_t num_links() const;
    std::size_t num_edges() const;
    std::size_t live_num_nodes() const noexcept;
    std::size_t live_num_links() const noexcept;
    void invalidate_cached_cardinalities() noexcept;

    std::size_t num_node_features() const noexcept;
    std::size_t num_link_features() const noexcept;
    std::size_t num_node_resource_features() const noexcept;
    std::size_t num_link_resource_features() const noexcept;
    std::vector<attribute::AttributeKind> get_node_attr_types() const;
    std::vector<attribute::AttributeKind> get_link_attr_types() const;

    const AttrMap& graph_attributes() const noexcept;
    AttrMap& graph_attributes() noexcept;
    AttrId bind_graph_attribute(std::string_view name);
    const AttrValue& graph_attribute(AttrId id) const;
    void set_graph_attribute(AttrId id, AttrValue value);
    void set_graph_attrs_data(
        const std::vector<GraphAttributeAssignment>& values);
    void init_graph_attrs();

    std::vector<attribute::AttributeRegistryId> select_node_attributes(
        const AttributeSelection& selection) const;
    std::vector<attribute::AttributeRegistryId> select_link_attributes(
        const AttributeSelection& selection) const;

    void check_attrs_existence() const;
    void generate_topology(const TopologyRequest& request);
    void generate_topology(const TopologyRequest& request, PyRandom& rng);
    void generate_attrs_data(
        NumpyRandomState& rng,
        bool node = true,
        bool link = true,
        std::size_t workers = 1U);

    void set_node_attrs_data(
        const std::vector<NodeAttributeDataUpdate>& updates,
        std::size_t workers = 1U);
    void set_link_attrs_data(
        const std::vector<LinkAttributeDataUpdate>& updates,
        std::size_t workers = 1U);

    CSRMatrix adjacency_matrix() const;
    BaseNetworkView subgraph(const std::vector<Vertex>& nodes) const;
    BaseNetworkView subnetwork(const std::vector<Vertex>& nodes) const;
    BaseNetworkView get_subgraph_view(
        nx::NodeFilter filter_node = {},
        nx::EdgeFilter filter_edge = {}) const;
    BaseNetworkView get_subnetwork_view(
        nx::NodeFilter filter_node = {},
        nx::EdgeFilter filter_edge = {}) const;

    std::string repr(std::string_view class_name = "BaseNetwork") const;
    BaseNetwork clone() const;
    Graph prepare_gml_graph() const;
    static BaseNetwork from_gml(
        const std::string& path,
        std::string_view label = "id");
    void save_attrs_dict(const std::string& path) const;

private:
    const NodeNetworkAttributeBinding& checked_node_binding(
        attribute::AttributeRegistryId id,
        BaseNetworkOperation operation,
        std::size_t input_index) const;
    const LinkNetworkAttributeBinding& checked_link_binding(
        attribute::AttributeRegistryId id,
        BaseNetworkOperation operation,
        std::size_t input_index) const;
    void refresh_graph_field_ids();
    void sync_fixed_graph_fields();

    Graph graph_;
    attribute::NodeAttributeRegistry node_attributes_;
    attribute::LinkAttributeRegistry link_attributes_;
    std::vector<NodeNetworkAttributeBinding> node_bindings_;
    std::vector<LinkNetworkAttributeBinding> link_bindings_;
    virne::utils::SettingList canonical_node_settings_;
    virne::utils::SettingList canonical_link_settings_;
    std::optional<virne::utils::SettingDocument> config_snapshot_;
    std::size_t factory_workers_ = 1U;

    mutable std::optional<std::size_t> cached_num_nodes_;
    mutable std::optional<std::size_t> cached_num_links_;
    mutable std::optional<std::size_t> cached_num_edges_;

    AttrId node_settings_id_ = 0U;
    AttrId link_settings_id_ = 0U;
    AttrId topology_id_ = 0U;
    AttrId output_id_ = 0U;
    AttrId num_nodes_metadata_id_ = 0U;
    std::optional<AttrValue> topology_;
    std::optional<AttrValue> output_;

    friend class BaseNetworkView;
    friend std::vector<std::vector<AttrValue>> get_node_attrs_data(
        const BaseNetwork&,
        const std::vector<attribute::AttributeRegistryId>&,
        std::size_t);
    friend std::vector<std::vector<AttrValue>> get_link_attrs_data(
        const BaseNetwork&,
        const std::vector<attribute::AttributeRegistryId>&,
        std::size_t);
    friend std::vector<DistanceMatrix> get_adjacency_attrs_data(
        const BaseNetwork&,
        const std::vector<attribute::AttributeRegistryId>&,
        bool,
        std::size_t);
    friend std::vector<std::vector<double>> get_aggregation_attrs_data(
        const BaseNetwork&,
        const std::vector<attribute::AttributeRegistryId>&,
        attribute::LinkAggregation,
        bool,
        std::size_t);
};

class BaseNetworkView {
public:
    const nx::GraphView& graph_view() const noexcept;
    const attribute::NodeAttributeRegistry& node_attributes() const noexcept;
    const attribute::LinkAttributeRegistry& link_attributes() const noexcept;
    const AttrMap& graph_attributes() const noexcept;
    std::size_t num_nodes() const;
    std::size_t num_links() const;
    const BaseNetwork& parent() const noexcept;

private:
    BaseNetworkView(const BaseNetwork& parent, nx::GraphView graph_view);

    const BaseNetwork* parent_;
    nx::GraphView graph_view_;

    friend class BaseNetwork;
};

}  // namespace virne::network
