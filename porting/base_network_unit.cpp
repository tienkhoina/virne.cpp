#include "base_network.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;
namespace utils = virne::utils;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

template <typename Callable>
void expect_network_error(
    Callable&& callable,
    const network::BaseNetworkErrorCode code,
    const network::BaseNetworkOperation operation,
    const std::optional<std::size_t> input_index,
    const std::string& context) {
    try {
        std::forward<Callable>(callable)();
    } catch (const network::BaseNetworkException& error) {
        expect(error.code() == code, context + ": error-code drift");
        expect(error.operation() == operation, context + ": operation drift");
        if (input_index) {
            expect(error.input_index() == *input_index,
                   context + ": input-index drift");
        }
        expect(std::string_view(error.what()).empty() == false,
               context + ": missing diagnostic");
        return;
    }
    fail(context + ": expected BaseNetworkException");
}

template <typename Callable>
void expect_any_error(Callable&& callable, const std::string& context) {
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception&) {
        return;
    }
    fail(context + ": expected exception");
}

attribute::AttributeFactorySpec factory_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const attribute::AttributeKind kind) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = kind;
    return result;
}

attribute::AttributeFactorySpec extrema_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    std::string originator) {
    auto result =
        factory_spec(std::move(name), owner, attribute::AttributeKind::extrema);
    result.originator_name = std::move(originator);
    return result;
}

network::NodeAttributeDataUpdate dense_node_update(
    const attribute::AttributeRegistryId id,
    std::vector<AttrValue> values) {
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate dense_link_update(
    const attribute::AttributeRegistryId id,
    std::vector<AttrValue> values) {
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::NodeAttributeDataUpdate sparse_node_update(
    const attribute::AttributeRegistryId id,
    std::vector<attribute::NodeAttributeAssignment> values) {
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    const attribute::AttributeRegistryId id,
    std::vector<attribute::LinkAttributeAssignment> values) {
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

void expect_values(
    const std::vector<AttrValue>& actual,
    const std::vector<AttrValue>& expected,
    const std::string& context) {
    expect(actual.size() == expected.size(), context + ": size drift");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        expect(attr_value_equal(actual[index], expected[index]),
               context + ": value drift at " + std::to_string(index));
    }
}

void expect_rows(
    const std::vector<std::vector<AttrValue>>& actual,
    const std::vector<std::vector<AttrValue>>& expected,
    const std::string& context) {
    expect(actual.size() == expected.size(), context + ": row-count drift");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        expect_values(actual[index], expected[index],
                      context + " row " + std::to_string(index));
    }
}

std::vector<std::string> registry_names(
    const attribute::NodeAttributeRegistry& registry) {
    std::vector<std::string> result;
    result.reserve(registry.size());
    for (const auto& entry : registry.entries()) {
        result.push_back(entry.name);
    }
    return result;
}

std::vector<std::string> registry_names(
    const attribute::LinkAttributeRegistry& registry) {
    std::vector<std::string> result;
    result.reserve(registry.size());
    for (const auto& entry : registry.entries()) {
        result.push_back(entry.name);
    }
    return result;
}

network::BaseNetwork make_populated_network(
    const std::size_t factory_workers = 1U,
    const bool include_status = true) {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}, {2U, 3U}});
    if (include_status) {
        construction.config.node_attribute_specs.push_back(factory_spec(
            "node_status",
            attribute::AttributeOwner::node,
            attribute::AttributeKind::status));
    }
    construction.config.node_attribute_specs.push_back(factory_spec(
        "cpu",
        attribute::AttributeOwner::node,
        attribute::AttributeKind::resource));
    construction.config.node_attribute_specs.push_back(extrema_spec(
        "cpu_extrema", attribute::AttributeOwner::node, "cpu"));
    if (include_status) {
        construction.config.link_attribute_specs.push_back(factory_spec(
            "link_status",
            attribute::AttributeOwner::link,
            attribute::AttributeKind::status));
    }
    construction.config.link_attribute_specs.push_back(factory_spec(
        "bandwidth",
        attribute::AttributeOwner::link,
        attribute::AttributeKind::resource));
    construction.config.link_attribute_specs.push_back(extrema_spec(
        "bandwidth_extrema", attribute::AttributeOwner::link, "bandwidth"));
    construction.config.factory_workers = factory_workers;
    return network::BaseNetwork(std::move(construction));
}

void fill_numeric_data(network::BaseNetwork& value, const std::size_t workers) {
    const auto cpu = value.bind_node_attribute("cpu");
    const auto cpu_extrema = value.bind_node_attribute("cpu_extrema");
    const auto bandwidth = value.bind_link_attribute("bandwidth");
    const auto bandwidth_extrema =
        value.bind_link_attribute("bandwidth_extrema");
    expect(cpu && cpu_extrema && bandwidth && bandwidth_extrema,
           "numeric fixture bindings");

    value.set_node_attrs_data(
        {dense_node_update(
             cpu->registry_id,
             {std::int64_t{1}, std::int64_t{2}, std::int64_t{3},
              std::int64_t{4}}),
         dense_node_update(
             cpu_extrema->registry_id,
             {10.0, 20.0, 30.0, 40.0})},
        workers);
    value.set_link_attrs_data(
        {dense_link_update(
             bandwidth->registry_id,
             {std::int64_t{5}, std::int64_t{6}, std::int64_t{7}}),
         dense_link_update(
             bandwidth_extrema->registry_id,
             {50.0, 60.0, 70.0})},
        workers);
}

AttrValue raw_attribute_setting(
    std::string name,
    AttrValue owner,
    AttrValue kind) {
    return make_attr_object(
        {{"name", std::move(name)},
         {"owner", std::move(owner)},
         {"type", std::move(kind)}});
}

void test_raw_construction_merge_and_explicit_rebuild() {
    Graph incoming(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const AttrId incoming_node_settings =
        incoming.attr_id("node_attrs_setting");
    const AttrId incoming_link_settings =
        incoming.attr_id("link_attrs_setting");
    const AttrId incoming_overlap = incoming.attr_id("overlap");
    incoming.graph_attrs().set(
        incoming_node_settings,
        make_attr_list(
            {raw_attribute_setting(
                 "cpu", true, std::string{"unsupported-before-override"}),
             raw_attribute_setting(
                 "incoming_status",
                 std::string{"node"},
                 std::string{"status"})}));
    incoming.graph_attrs().set(
        incoming_link_settings,
        make_attr_list(
            {raw_attribute_setting(
                 "bandwidth", std::int64_t{9}, std::string{"invalid"}),
             raw_attribute_setting(
                 "incoming_link_status",
                 std::string{"link"},
                 std::string{"status"})}));
    incoming.graph_attrs().set(incoming_overlap, std::int64_t{1});

    const utils::SettingDocument config = utils::parse_setting(
        R"json({
            "node_attrs_setting": [
                {"name":"cpu","owner":"node","type":"resource"},
                {"name":"position","owner":"node","type":"position"}
            ],
            "link_attrs_setting": [
                {"name":"bandwidth","owner":"link","type":"resource"},
                {"name":"latency","owner":"link","type":"latency"}
            ],
            "topology":"path",
            "output":"raw-output",
            "graph_attrs_setting":{"overlap":2,"config_only":3}
        })json",
        utils::SettingFormat::json);
    auto prepared = network::base_network_construction_from_setting(
        std::optional<Graph>{std::move(incoming)},
        &config,
        {{"overlap", std::int64_t{4}},
         {"extra_only", std::int64_t{5}}},
        8U);
    network::BaseNetwork value(std::move(prepared));

    expect(registry_names(value.node_attributes()) ==
               std::vector<std::string>(
                   {"cpu", "incoming_status", "position"}),
           "raw node merge/dedup order");
    expect(registry_names(value.link_attributes()) ==
               std::vector<std::string>(
                   {"bandwidth", "incoming_link_status", "latency"}),
           "raw link merge/dedup order");
    expect(value.node_attributes().at(0U).spec().kind ==
               attribute::AttributeKind::resource &&
               value.link_attributes().at(0U).spec().kind ==
                   attribute::AttributeKind::resource,
           "invalid overridden items are discarded before decode");
    expect(value.config_snapshot().has_value(), "raw config snapshot retained");
    const auto overlap = value.graph().attribute_registry().find("overlap");
    const auto config_only =
        value.graph().attribute_registry().find("config_only");
    const auto extra_only =
        value.graph().attribute_registry().find("extra_only");
    expect(overlap && config_only && extra_only,
           "raw graph metadata names resolved");
    expect(std::get<std::int64_t>(value.graph_attribute(*overlap)) == 4 &&
               std::get<std::int64_t>(value.graph_attribute(*config_only)) ==
                   3 &&
               std::get<std::int64_t>(value.graph_attribute(*extra_only)) == 5,
           "raw graph metadata precedence");

    const auto node_settings =
        value.graph().attribute_registry().find("node_attrs_setting");
    const auto link_settings =
        value.graph().attribute_registry().find("link_attrs_setting");
    expect(node_settings && link_settings, "public rebuild snapshots");
    value.graph_attributes().set(
        *node_settings,
        make_attr_list({raw_attribute_setting(
            "rebuilt_node",
            std::string{"node"},
            std::string{"status"})}));
    value.graph_attributes().set(
        *link_settings,
        make_attr_list({raw_attribute_setting(
            "invalid_link_family",
            std::string{"node"},
            std::string{"status"})}));
    expect_any_error(
        [&] { value.create_attrs_from_setting(); },
        "link rebuild failure after successful node rebuild");
    expect(registry_names(value.node_attributes()) ==
               std::vector<std::string>{"rebuilt_node"} &&
               registry_names(value.link_attributes()) ==
                   std::vector<std::string>(
                       {"bandwidth", "incoming_link_status", "latency"}),
           "node-success/link-failure partial registry replacement");
    const auto rebuilt_node = value.bind_node_attribute("rebuilt_node");
    expect(rebuilt_node &&
               rebuilt_node->registry_identity == &value.node_attributes() &&
               rebuilt_node->graph_identity == &value.graph(),
           "successful partial rebuild refreshes node binding");

    value.graph_attributes().set(
        *link_settings,
        make_attr_list({raw_attribute_setting(
            "rebuilt_link",
            std::string{"link"},
            std::string{"status"})}));
    value.create_attrs_from_setting();
    expect(registry_names(value.node_attributes()) ==
               std::vector<std::string>{"rebuilt_node"} &&
               registry_names(value.link_attributes()) ==
                   std::vector<std::string>{"rebuilt_link"},
           "successful explicit registry rebuild");
}

void test_construction_merge_registries_and_ids() {
    static_assert(!std::is_copy_constructible_v<network::BaseNetwork>);
    static_assert(!std::is_copy_assignable_v<network::BaseNetwork>);
    static_assert(std::is_move_constructible_v<network::BaseNetwork>);
    static_assert(std::is_move_assignable_v<network::BaseNetwork>);

    network::BaseNetwork empty;
    expect(empty.live_num_nodes() == 0U && empty.live_num_links() == 0U,
           "default graph cardinality");
    expect(empty.node_attributes().size() == 0U &&
               empty.link_attributes().size() == 0U,
           "default registries");

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const AttrId incoming_id =
        construction.incoming_graph->attr_id("existing");
    construction.incoming_graph->graph_attrs().set(
        incoming_id, std::int64_t{1});

    construction.config.node_attribute_specs = {
        factory_spec("dup", attribute::AttributeOwner::node,
                     attribute::AttributeKind::status),
        factory_spec("cpu", attribute::AttributeOwner::node,
                     attribute::AttributeKind::resource),
        factory_spec("dup", attribute::AttributeOwner::node,
                     attribute::AttributeKind::position)};
    construction.config.link_attribute_specs = {
        factory_spec("dup_link", attribute::AttributeOwner::link,
                     attribute::AttributeKind::status),
        factory_spec("bandwidth", attribute::AttributeOwner::link,
                     attribute::AttributeKind::resource),
        factory_spec("dup_link", attribute::AttributeOwner::link,
                     attribute::AttributeKind::latency)};
    construction.config.topology = std::string{"path"};
    construction.config.output = std::string{"result"};
    construction.config.graph_attributes = {
        {"existing", std::int64_t{2}}, {"config_only", std::int64_t{3}}};
    construction.extra_graph_attributes = {
        {"existing", std::int64_t{4}}, {"extra_only", std::int64_t{5}}};
    construction.config.factory_workers = 2U;

    network::BaseNetwork value(std::move(construction));
    expect(registry_names(value.node_attributes()) ==
               std::vector<std::string>({"dup", "cpu"}),
           "node duplicate replacement order");
    expect(registry_names(value.link_attributes()) ==
               std::vector<std::string>({"dup_link", "bandwidth"}),
           "link duplicate replacement order");
    expect(value.node_attributes().at(0U).spec().kind ==
               attribute::AttributeKind::position,
           "node duplicate replacement value");
    expect(value.link_attributes().at(0U).spec().kind ==
               attribute::AttributeKind::latency,
           "link duplicate replacement value");
    expect(value.num_node_features() == 2U &&
               value.num_link_features() == 2U &&
               value.num_node_resource_features() == 1U &&
               value.num_link_resource_features() == 1U,
           "feature counts");

    const auto existing = value.graph().attribute_registry().find("existing");
    const auto config_only =
        value.graph().attribute_registry().find("config_only");
    const auto extra_only =
        value.graph().attribute_registry().find("extra_only");
    expect(existing && config_only && extra_only, "graph metadata bindings");
    expect(std::get<std::int64_t>(value.graph_attribute(*existing)) == 4 &&
               std::get<std::int64_t>(value.graph_attribute(*config_only)) ==
                   3 &&
               std::get<std::int64_t>(value.graph_attribute(*extra_only)) == 5,
           "graph metadata overwrite precedence");

    const auto node_binding = value.bind_node_attribute("cpu");
    const auto link_binding = value.bind_link_attribute("bandwidth");
    expect(node_binding && link_binding, "definition binding");
    expect(node_binding->registry_identity == &value.node_attributes() &&
               node_binding->graph_identity == &value.graph() &&
               link_binding->registry_identity == &value.link_attributes() &&
               link_binding->graph_identity == &value.graph(),
           "binding identities");
    expect(!value.bind_node_attribute("missing") &&
               !value.bind_link_attribute("missing"),
           "missing definition binding");

    auto unrelated = make_populated_network();
    const auto unrelated_cpu = unrelated.bind_node_attribute("cpu");
    expect(unrelated_cpu && unrelated_cpu->registry_id ==
                                  node_binding->registry_id,
           "unrelated fixture equal compact id");
    expect(unrelated_cpu->registry_identity != node_binding->registry_identity &&
               unrelated_cpu->graph_identity != node_binding->graph_identity,
           "unrelated equal ids retain identity");

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        auto worker_value = make_populated_network(workers);
        expect(registry_names(worker_value.node_attributes()) ==
                   std::vector<std::string>(
                       {"node_status", "cpu", "cpu_extrema"}) &&
                   registry_names(worker_value.link_attributes()) ==
                       std::vector<std::string>(
                           {"link_status", "bandwidth", "bandwidth_extrema"}),
               "factory worker construction " + std::to_string(workers));
    }
}

void test_cardinality_caches() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    network::BaseNetwork value(std::move(construction));

    expect(value.num_nodes() == 2U, "initial cached node count");
    expect(value.num_links() == 1U, "initial cached link count");
    value.graph().add_edge(1U, 2U);
    expect(value.num_nodes() == 2U && value.num_links() == 1U,
           "cardinality caches remain stale");
    expect(value.live_num_nodes() == 3U && value.live_num_links() == 2U,
           "live cardinality extensions");
    expect(value.num_edges() == 2U,
           "link and edge caches are independently populated");
    value.graph().add_edge(2U, 3U);
    expect(value.num_edges() == 2U && value.num_links() == 1U,
           "independent stale edge/link caches");
    value.invalidate_cached_cardinalities();
    expect(value.num_nodes() == 4U && value.num_links() == 3U &&
               value.num_edges() == 3U,
           "explicit cache invalidation");
}

void test_selection_and_types() {
    auto value = make_populated_network();
    const auto node_status = value.bind_node_attribute("node_status");
    const auto cpu = value.bind_node_attribute("cpu");
    const auto cpu_extrema = value.bind_node_attribute("cpu_extrema");
    const auto link_status = value.bind_link_attribute("link_status");
    const auto bandwidth = value.bind_link_attribute("bandwidth");
    expect(node_status && cpu && cpu_extrema && link_status && bandwidth,
           "selection fixture bindings");

    expect(value.get_node_attr_types() ==
               std::vector<attribute::AttributeKind>(
                   {attribute::AttributeKind::status,
                    attribute::AttributeKind::resource,
                    attribute::AttributeKind::extrema}),
           "node type order");
    expect(value.get_link_attr_types() ==
               std::vector<attribute::AttributeKind>(
                   {attribute::AttributeKind::status,
                    attribute::AttributeKind::resource,
                    attribute::AttributeKind::extrema}),
           "link type order");

    expect(value.select_node_attributes({}) ==
               std::vector<attribute::AttributeRegistryId>(
                   {node_status->registry_id, cpu->registry_id,
                    cpu_extrema->registry_id}),
           "select all node definitions");

    network::AttributeSelection ids_only;
    ids_only.ids = std::vector<attribute::AttributeRegistryId>{
        cpu_extrema->registry_id, node_status->registry_id};
    expect(value.select_node_attributes(ids_only) ==
               std::vector<attribute::AttributeRegistryId>(
                   {node_status->registry_id, cpu_extrema->registry_id}),
           "ID selection preserves registry order");

    network::AttributeSelection both;
    both.kinds = std::vector<attribute::AttributeKind>{
        attribute::AttributeKind::resource};
    both.ids = std::vector<attribute::AttributeRegistryId>{
        node_status->registry_id};
    expect(value.select_node_attributes(both) ==
               std::vector<attribute::AttributeRegistryId>{cpu->registry_id},
           "kind selection precedence");
    both.kinds = std::vector<attribute::AttributeKind>{};
    expect(value.select_node_attributes(both).empty(),
           "present empty kind selection precedence");

    network::AttributeSelection link_ids;
    link_ids.ids = std::vector<attribute::AttributeRegistryId>{
        bandwidth->registry_id, link_status->registry_id};
    expect(value.select_link_attributes(link_ids) ==
               std::vector<attribute::AttributeRegistryId>(
                   {link_status->registry_id, bandwidth->registry_id}),
           "link selection registry order");
}

void test_data_adapters_workers_and_partial_mutation() {
    auto value = make_populated_network();
    const auto node_status = value.bind_node_attribute("node_status");
    const auto cpu = value.bind_node_attribute("cpu");
    const auto link_status = value.bind_link_attribute("link_status");
    const auto bandwidth = value.bind_link_attribute("bandwidth");
    expect(node_status && cpu && link_status && bandwidth,
           "data fixture bindings");

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        value.set_node_attrs_data(
            {dense_node_update(
                 node_status->registry_id,
                 {std::int64_t{10}, std::int64_t{11}, std::int64_t{12},
                  std::int64_t{13}}),
             dense_node_update(
                 cpu->registry_id,
                 {1.0, 2.0, 3.0, 4.0})},
            workers);
        value.set_link_attrs_data(
            {dense_link_update(
                 link_status->registry_id,
                 {true, false, true}),
             dense_link_update(
                 bandwidth->registry_id,
                 {2.0, 4.0, 8.0})},
            workers);

        expect_rows(
            network::get_node_attrs_data(
                value,
                {node_status->registry_id, cpu->registry_id},
                workers),
            {{std::int64_t{10}, std::int64_t{11}, std::int64_t{12},
              std::int64_t{13}},
             {1.0, 2.0, 3.0, 4.0}},
            "node rows worker " + std::to_string(workers));
        expect_rows(
            network::get_link_attrs_data(
                value,
                {link_status->registry_id, bandwidth->registry_id},
                workers),
            {{true, false, true}, {2.0, 4.0, 8.0}},
            "link rows worker " + std::to_string(workers));

        const auto adjacency = network::get_adjacency_attrs_data(
            value, {bandwidth->registry_id}, false, workers);
        expect(adjacency.size() == 1U && adjacency[0].rows() == 4U &&
                   adjacency[0].cols() == 4U && adjacency[0](0U, 1U) == 2.0 &&
                   adjacency[0](1U, 2U) == 4.0 &&
                   adjacency[0](2U, 3U) == 8.0,
               "adjacency adapter worker " + std::to_string(workers));
        const auto aggregation = network::get_aggregation_attrs_data(
            value,
            {bandwidth->registry_id},
            attribute::LinkAggregation::sum,
            false,
            workers);
        expect(aggregation ==
                   std::vector<std::vector<double>>{{2.0, 6.0, 12.0, 8.0}},
               "aggregation adapter worker " + std::to_string(workers));
    }

    expect_network_error(
        [&] {
            static_cast<void>(network::get_node_attrs_data(value, {}, 2U));
        },
        network::BaseNetworkErrorCode::empty_attribute_selection,
        network::BaseNetworkOperation::get_attribute_data,
        std::nullopt,
        "empty node row selection");
    expect_network_error(
        [&] {
            static_cast<void>(network::get_link_attrs_data(value, {}, 2U));
        },
        network::BaseNetworkErrorCode::empty_attribute_selection,
        network::BaseNetworkOperation::get_attribute_data,
        std::nullopt,
        "empty link row selection");
    expect(network::get_adjacency_attrs_data(value, {}, false, 8U).empty() &&
               network::get_aggregation_attrs_data(value, {},
                                                   attribute::LinkAggregation::sum,
                                                   false, 8U)
                   .empty(),
           "matrix adapters accept empty selection");

    const auto before_cpu = network::get_node_attrs_data(
        value, {cpu->registry_id}, 1U);
    const auto invalid_id = attribute::invalid_attribute_registry_id;
    expect_network_error(
        [&] {
            value.set_node_attrs_data(
                {dense_node_update(
                     node_status->registry_id,
                     {std::int64_t{90}, std::int64_t{91}, std::int64_t{92},
                      std::int64_t{93}}),
                 dense_node_update(invalid_id, std::vector<AttrValue>(4U, 0.0))},
                8U);
        },
        network::BaseNetworkErrorCode::attribute_registry_mismatch,
        network::BaseNetworkOperation::set_attribute_data,
        1U,
        "ordered node setter partial mutation");
    expect_rows(
        network::get_node_attrs_data(value, {node_status->registry_id}, 1U),
        {{std::int64_t{90}, std::int64_t{91}, std::int64_t{92},
          std::int64_t{93}}},
        "earlier node update remains");
    expect_rows(network::get_node_attrs_data(value, {cpu->registry_id}, 1U),
                before_cpu,
                "later invalid node update leaves unrelated value");

    expect_any_error(
        [&] {
            value.set_link_attrs_data(
                {dense_link_update(
                    bandwidth->registry_id, {std::int64_t{1}})},
                2U);
        },
        "short dense link update");
    expect_rows(
        network::get_link_attrs_data(value, {bandwidth->registry_id}, 1U),
        {{2.0, 4.0, 8.0}},
        "short dense link update is atomic");

    auto ragged = make_populated_network();
    const auto ragged_cpu = ragged.bind_node_attribute("cpu");
    expect(ragged_cpu.has_value(), "ragged binding");
    ragged.set_node_attrs_data(
        {sparse_node_update(
            ragged_cpu->registry_id,
            {{1U, std::int64_t{7}}, {3U, std::int64_t{9}}})});
    expect_rows(
        network::get_node_attrs_data(ragged, {ragged_cpu->registry_id}, 2U),
        {{std::int64_t{7}, std::int64_t{9}}},
        "missing node values are omitted in node order");
}

void test_existence_first_sample_and_order() {
    expect_network_error(
        [] {
            const network::BaseNetwork empty;
            empty.check_attrs_existence();
        },
        network::BaseNetworkErrorCode::no_nodes,
        network::BaseNetworkOperation::check_attributes,
        std::nullopt,
        "existence empty node precedence");

    network::BaseNetworkConstruction no_link_construction;
    no_link_construction.incoming_graph.emplace(1U, std::vector<EdgeEndpoints>{});
    network::BaseNetwork no_link(std::move(no_link_construction));
    expect_network_error(
        [&] { no_link.check_attrs_existence(); },
        network::BaseNetworkErrorCode::no_links,
        network::BaseNetworkOperation::check_attributes,
        std::nullopt,
        "existence empty link precedence");

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs.push_back(factory_spec(
        "first_node",
        attribute::AttributeOwner::node,
        attribute::AttributeKind::status));
    construction.config.node_attribute_specs.push_back(factory_spec(
        "second_node",
        attribute::AttributeOwner::node,
        attribute::AttributeKind::status));
    construction.config.link_attribute_specs.push_back(factory_spec(
        "first_link",
        attribute::AttributeOwner::link,
        attribute::AttributeKind::status));
    construction.config.link_attribute_specs.push_back(factory_spec(
        "second_link",
        attribute::AttributeOwner::link,
        attribute::AttributeKind::status));
    network::BaseNetwork value(std::move(construction));
    const auto node = value.bind_node_attribute("first_node");
    const auto second_node = value.bind_node_attribute("second_node");
    const auto link = value.bind_link_attribute("first_link");
    const auto second_link = value.bind_link_attribute("second_link");
    expect(node && second_node && link && second_link,
           "existence fixture bindings");

    expect_network_error(
        [&] { value.check_attrs_existence(); },
        network::BaseNetworkErrorCode::missing_node_attribute,
        network::BaseNetworkOperation::check_attributes,
        0U,
        "first missing node attribute");
    value.set_node_attrs_data(
        {sparse_node_update(node->registry_id, {{0U, std::int64_t{1}}})});
    expect_network_error(
        [&] { value.check_attrs_existence(); },
        network::BaseNetworkErrorCode::missing_node_attribute,
        network::BaseNetworkOperation::check_attributes,
        1U,
        "node definition-order failure");
    value.set_node_attrs_data(
        {sparse_node_update(
            second_node->registry_id, {{0U, std::int64_t{11}}})});
    expect_network_error(
        [&] { value.check_attrs_existence(); },
        network::BaseNetworkErrorCode::missing_link_attribute,
        network::BaseNetworkOperation::check_attributes,
        0U,
        "first missing link attribute");
    value.set_link_attrs_data(
        {sparse_link_update(
            link->registry_id, {{0U, 1U, std::int64_t{2}}})});
    expect_network_error(
        [&] { value.check_attrs_existence(); },
        network::BaseNetworkErrorCode::missing_link_attribute,
        network::BaseNetworkOperation::check_attributes,
        1U,
        "link definition-order failure");
    value.set_link_attrs_data(
        {sparse_link_update(
            second_link->registry_id, {{0U, 1U, std::int64_t{22}}})});
    value.check_attrs_existence();
    expect(network::get_node_attrs_data(value, {node->registry_id}).size() ==
               1U &&
               network::get_node_attrs_data(value, {node->registry_id})[0]
                       .size() ==
                   1U &&
               network::get_link_attrs_data(value, {link->registry_id})[0]
                       .size() ==
                   1U,
           "later node/link omissions deliberately ignored by sample check");
}

void test_graph_data_clone_move_and_views() {
    auto value = make_populated_network();
    fill_numeric_data(value, 2U);
    const AttrId metadata = value.bind_graph_attribute("metadata");
    value.set_graph_attribute(
        metadata,
        make_attr_object({{"nested", std::int64_t{7}}}));
    value.set_graph_attrs_data(
        {{"ordered", std::int64_t{1}},
         {"metadata", make_attr_object({{"nested", std::int64_t{8}}})},
         {"ordered_tail", std::int64_t{2}}});
    expect(std::get<std::int64_t>(
               *attr_object(value.graph_attribute(metadata))->find("nested")) ==
               8,
           "graph metadata ordered overwrite");

    const auto induced = value.subgraph({0U, 1U, 3U});
    const auto induced_alias = value.subnetwork({0U, 1U, 3U});
    expect(induced.num_nodes() == 3U && induced.num_links() == 1U &&
               induced_alias.num_nodes() == induced.num_nodes() &&
               induced_alias.num_links() == induced.num_links(),
           "induced subnetwork alias");
    expect(&induced.parent() == &value &&
               &induced.node_attributes() == &value.node_attributes() &&
               &induced.link_attributes() == &value.link_attributes() &&
               &induced.graph_attributes() == &value.graph_attributes(),
           "view shares parent registries and metadata");

    auto predicate_view = value.get_subgraph_view();
    auto predicate_alias = value.get_subnetwork_view();
    expect(predicate_view.num_nodes() == value.live_num_nodes() &&
               predicate_view.num_links() == value.live_num_links() &&
               predicate_alias.num_nodes() == predicate_view.num_nodes() &&
               predicate_alias.num_links() == predicate_view.num_links(),
           "default predicate view and alias");

    const std::size_t cached_nodes = value.num_nodes();
    value.graph().add_edge(3U, 4U);
    network::BaseNetwork cloned = value.clone();
    expect(cloned.num_nodes() == cached_nodes &&
               cloned.live_num_nodes() == value.live_num_nodes(),
           "clone preserves cached snapshot and live topology");
    expect(&cloned.node_attributes() != &value.node_attributes() &&
               &cloned.link_attributes() != &value.link_attributes() &&
               &cloned.graph() != &value.graph(),
           "clone owns independent registry and graph storage");
    const auto cloned_cpu = cloned.bind_node_attribute("cpu");
    const auto source_cpu = value.bind_node_attribute("cpu");
    expect(cloned_cpu && source_cpu &&
               cloned_cpu->registry_id == source_cpu->registry_id &&
               cloned_cpu->registry_identity != source_cpu->registry_identity &&
               cloned_cpu->graph_identity != source_cpu->graph_identity,
           "clone rebinds compact IDs to fresh identities");
    AttrObject* cloned_metadata =
        attr_object(cloned.graph_attributes().at(metadata));
    expect(cloned_metadata != nullptr, "clone recursive metadata");
    cloned_metadata->set("nested", std::int64_t{99});
    expect(std::get<std::int64_t>(
               *attr_object(value.graph_attribute(metadata))->find("nested")) ==
               8,
           "clone recursively isolates graph metadata");

    network::BaseNetwork moved(std::move(cloned));
    const auto moved_cpu = moved.bind_node_attribute("cpu");
    expect(moved_cpu && moved_cpu->registry_identity == &moved.node_attributes() &&
               moved_cpu->graph_identity == &moved.graph(),
           "move construction rebinds identities");
    network::BaseNetwork assigned;
    assigned = std::move(moved);
    const auto assigned_cpu = assigned.bind_node_attribute("cpu");
    expect(assigned_cpu &&
               assigned_cpu->registry_identity == &assigned.node_attributes() &&
               assigned_cpu->graph_identity == &assigned.graph(),
           "move assignment rebinds identities");
}

void expect_benchmark_maps_equal(
    const std::optional<attribute::AttributeBenchmarkMap>& actual,
    const std::optional<attribute::AttributeBenchmarkMap>& expected,
    const std::string& context) {
    expect(actual.has_value() == expected.has_value(),
           context + ": optional presence drift");
    if (!actual) {
        return;
    }
    expect(actual->entries().size() == expected->entries().size(),
           context + ": entry count drift");
    for (std::size_t index = 0U; index < actual->entries().size(); ++index) {
        expect(actual->entries()[index].name == expected->entries()[index].name &&
                   actual->entries()[index].value ==
                       expected->entries()[index].value,
               context + ": entry drift at " + std::to_string(index));
    }
}

void test_benchmark_request_delegation_and_errors() {
    auto value = make_populated_network();
    fill_numeric_data(value, 8U);

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        network::BaseNetworkBenchmarkSelection selection;
        selection.workers = workers;
        const auto request =
            network::prepare_attribute_benchmark_request(value, selection);
        expect(request.workers == workers && request.node && request.link &&
                   request.link_sum,
               "benchmark request group shape worker " +
                   std::to_string(workers));
        const std::string worker_context =
            " worker " + std::to_string(workers);
        expect(request.node->attributes.size() == 2U,
               "node benchmark attribute count" + worker_context);
        expect(request.node->matrix.rows == 2U,
               "node benchmark rows=" +
                   std::to_string(request.node->matrix.rows) + worker_context);
        expect(request.node->matrix.columns == 4U,
               "node benchmark columns=" +
                   std::to_string(request.node->matrix.columns) +
                   worker_context);
        expect(request.node->column_repetitions == 1U &&
                   request.node->extrema_requested,
               "node benchmark flags" + worker_context);
        expect(request.link->attributes.size() == 2U,
               "link benchmark attribute count" + worker_context);
        expect(request.link->matrix.rows == 2U,
               "link benchmark rows=" +
                   std::to_string(request.link->matrix.rows) + worker_context);
        expect(request.link->matrix.columns == 3U,
               "link benchmark columns=" +
                   std::to_string(request.link->matrix.columns) +
                   worker_context);
        expect(request.link->column_repetitions == 2U &&
                   request.link->extrema_requested,
               "direct-link benchmark flags" + worker_context);
        expect(request.link_sum->attributes.size() == 2U,
               "link-sum benchmark attribute count" + worker_context);
        expect(request.link_sum->matrix.rows == 2U,
               "link-sum benchmark rows=" +
                   std::to_string(request.link_sum->matrix.rows) +
                   worker_context);
        expect(request.link_sum->matrix.columns == 4U,
               "link-sum benchmark columns=" +
                   std::to_string(request.link_sum->matrix.columns) +
                   worker_context);
        expect(request.link_sum->column_repetitions == 1U &&
                   request.link_sum->extrema_requested,
               "link-sum benchmark flags" + worker_context);

        const auto expected =
            attribute::AttributeBenchmarkManager::get_benchmarks(request);
        const auto actual = network::get_attribute_benchmarks(value, selection);
        expect_benchmark_maps_equal(actual.node_attr_benchmarks,
                                    expected.node_attr_benchmarks,
                                    "node manager delegation");
        expect_benchmark_maps_equal(actual.link_attr_benchmarks,
                                    expected.link_attr_benchmarks,
                                    "link manager delegation");
        expect_benchmark_maps_equal(actual.link_sum_attr_benchmarks,
                                    expected.link_sum_attr_benchmarks,
                                    "link-sum manager delegation");
    }

    network::BaseNetworkBenchmarkSelection disabled;
    disabled.node = false;
    disabled.link = false;
    disabled.link_sum = false;
    const auto disabled_request =
        network::prepare_attribute_benchmark_request(value, disabled);
    expect(!disabled_request.node && !disabled_request.link &&
               !disabled_request.link_sum,
           "disabled benchmark groups remain absent");

    auto nonnumeric = make_populated_network();
    fill_numeric_data(nonnumeric, 1U);
    const auto cpu = nonnumeric.bind_node_attribute("cpu");
    expect(cpu.has_value(), "nonnumeric fixture binding");
    nonnumeric.set_node_attrs_data(
        {sparse_node_update(cpu->registry_id, {{0U, std::string{"bad"}}})});
    network::BaseNetworkBenchmarkSelection node_only;
    node_only.link = false;
    node_only.link_sum = false;
    expect_network_error(
        [&] {
            static_cast<void>(network::prepare_attribute_benchmark_request(
                nonnumeric, node_only));
        },
        network::BaseNetworkErrorCode::non_numeric_benchmark_value,
        network::BaseNetworkOperation::prepare_benchmarks,
        std::nullopt,
        "nonnumeric benchmark value");

    auto ragged = make_populated_network();
    const auto ragged_cpu = ragged.bind_node_attribute("cpu");
    const auto ragged_extrema = ragged.bind_node_attribute("cpu_extrema");
    expect(ragged_cpu && ragged_extrema, "ragged benchmark bindings");
    ragged.set_node_attrs_data(
        {dense_node_update(
             ragged_cpu->registry_id,
             {1.0, 2.0, 3.0, 4.0}),
         sparse_node_update(
             ragged_extrema->registry_id,
             {{0U, 10.0}, {1U, 20.0}, {2U, 30.0}})});
    expect_network_error(
        [&] {
            static_cast<void>(network::prepare_attribute_benchmark_request(
                ragged, node_only));
        },
        network::BaseNetworkErrorCode::ragged_benchmark_matrix,
        network::BaseNetworkOperation::prepare_benchmarks,
        std::nullopt,
        "ragged benchmark matrix");
}

void test_concurrent_independent_instances() {
    std::array<std::future<std::int64_t>, 4U> futures;
    const std::array<std::size_t, 4U> worker_counts{0U, 1U, 2U, 8U};
    for (std::size_t index = 0U; index < futures.size(); ++index) {
        futures[index] = std::async(
            std::launch::async,
            [workers = worker_counts[index]] {
                auto value = make_populated_network(workers);
                fill_numeric_data(value, workers);
                const auto cpu = value.bind_node_attribute("cpu");
                const auto bandwidth = value.bind_link_attribute("bandwidth");
                if (!cpu || !bandwidth) {
                    throw std::runtime_error("concurrent binding failure");
                }
                const auto node_rows = network::get_node_attrs_data(
                    value, {cpu->registry_id}, workers);
                const auto link_rows = network::get_link_attrs_data(
                    value, {bandwidth->registry_id}, workers);
                return std::get<std::int64_t>(node_rows[0][0]) +
                       std::get<std::int64_t>(link_rows[0][0]);
            });
    }
    for (std::size_t index = 0U; index < futures.size(); ++index) {
        expect(futures[index].get() == 6,
               "concurrent independent instance " + std::to_string(index));
    }
}

}  // namespace

int main() {
    try {
        test_raw_construction_merge_and_explicit_rebuild();
        test_construction_merge_registries_and_ids();
        test_cardinality_caches();
        test_selection_and_types();
        test_data_adapters_workers_and_partial_mutation();
        test_existence_first_sample_and_order();
        test_graph_data_clone_move_and_views();
        test_benchmark_request_delegation_and_errors();
        test_concurrent_independent_instances();
        std::cout << "base_network_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base_network_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
