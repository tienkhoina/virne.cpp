#include "resource_updator.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeNumber;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using attribute::ResourceUpdateOperation;
using controller::ConstraintLink;
using controller::LinkResourceUpdateRequest;
using controller::NodeResourceUpdateRequest;
using controller::PreparedResourceUpdator;
using controller::ResourceAmount;
using controller::ResourceId;
using controller::ResourceUpdator;
using controller::ResourceUpdatorErrorCode;
using controller::ResourceUpdatorException;
using controller::ResourceUpdatorOperation;
using controller::ResourceUpdatorSelection;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
void expect_updator_error(
    Callable&& callable,
    ResourceUpdatorErrorCode code,
    ResourceUpdatorOperation operation,
    std::optional<std::size_t> request_index = std::nullopt,
    std::optional<std::size_t> item_index = std::nullopt,
    std::optional<ResourceId> resource_id = std::nullopt)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const ResourceUpdatorException& error)
    {
        expect(error.code() == code, "resource error-code mismatch");
        expect(error.operation() == operation, "resource operation mismatch");
        expect(
            error.request_index() == request_index,
            "resource request-index mismatch");
        expect(error.item_index() == item_index, "resource item-index mismatch");
        expect(error.resource_id() == resource_id, "resource ID mismatch");
        expect(!std::string_view(error.what()).empty(), "missing diagnostic");
        return;
    }
    throw std::runtime_error("expected ResourceUpdatorException");
}

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner,
    CheckingLevel level)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = level;
    return result;
}

AttributeFactorySpec status_spec(std::string name, AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::status;
    return result;
}

AttributeFactorySpec latency_spec()
{
    AttributeFactorySpec result;
    result.name = "latency";
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::latency;
    result.checking_level = CheckingLevel::path;
    return result;
}

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec("node_int", AttributeOwner::node, CheckingLevel::node),
        resource_spec("node_float", AttributeOwner::node, CheckingLevel::node),
        resource_spec("node_bool", AttributeOwner::node, CheckingLevel::node),
        status_spec("node_status", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("link_int", AttributeOwner::link, CheckingLevel::link),
        resource_spec("link_float", AttributeOwner::link, CheckingLevel::link),
        latency_spec(),
        status_spec("link_status", AttributeOwner::link)};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        5U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 2U}, {2U, 3U}, {3U, 4U}, {0U, 4U}});
    construction.config.node_attribute_specs = {
        status_spec("node_status", AttributeOwner::node),
        resource_spec("node_bool", AttributeOwner::node, CheckingLevel::node),
        resource_spec("node_float", AttributeOwner::node, CheckingLevel::node),
        resource_spec("node_int", AttributeOwner::node, CheckingLevel::node)};
    construction.config.link_attribute_specs = {
        status_spec("link_status", AttributeOwner::link),
        resource_spec("link_float", AttributeOwner::link, CheckingLevel::link),
        resource_spec("link_int", AttributeOwner::link, CheckingLevel::link),
        latency_spec()};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    expect(binding.has_value(), "missing node fixture binding");
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing link fixture binding");
    return *binding;
}

network::NodeAttributeDataUpdate dense_node_update(
    ResourceId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate dense_link_update(
    ResourceId id,
    std::vector<AttrValue> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();

    network::NodeNetworkAttributeBinding v_node_int;
    network::NodeNetworkAttributeBinding v_node_float;
    network::NodeNetworkAttributeBinding v_node_bool;
    network::NodeNetworkAttributeBinding v_node_status;
    network::LinkNetworkAttributeBinding v_link_int;
    network::LinkNetworkAttributeBinding v_link_float;
    network::LinkNetworkAttributeBinding v_latency;
    network::LinkNetworkAttributeBinding v_link_status;

    network::NodeNetworkAttributeBinding p_node_int;
    network::NodeNetworkAttributeBinding p_node_float;
    network::NodeNetworkAttributeBinding p_node_bool;
    network::LinkNetworkAttributeBinding p_link_int;
    network::LinkNetworkAttributeBinding p_link_float;

    Fixture()
        : v_node_int(require_node_binding(virtual_network, "node_int")),
          v_node_float(require_node_binding(virtual_network, "node_float")),
          v_node_bool(require_node_binding(virtual_network, "node_bool")),
          v_node_status(require_node_binding(virtual_network, "node_status")),
          v_link_int(require_link_binding(virtual_network, "link_int")),
          v_link_float(require_link_binding(virtual_network, "link_float")),
          v_latency(require_link_binding(virtual_network, "latency")),
          v_link_status(require_link_binding(virtual_network, "link_status")),
          p_node_int(require_node_binding(physical_network, "node_int")),
          p_node_float(require_node_binding(physical_network, "node_float")),
          p_node_bool(require_node_binding(physical_network, "node_bool")),
          p_link_int(require_link_binding(physical_network, "link_int")),
          p_link_float(require_link_binding(physical_network, "link_float"))
    {
        expect(
            v_node_int.registry_id != p_node_int.registry_id &&
                v_link_int.registry_id != p_link_int.registry_id,
            "fixture must use independent registry orders");

        virtual_network.set_node_attrs_data({
            dense_node_update(
                v_node_int.registry_id,
                {std::int64_t{2}, std::int64_t{3}, std::int64_t{4}}),
            dense_node_update(v_node_float.registry_id, {1.5, 2.5, 3.5}),
            dense_node_update(v_node_bool.registry_id, {true, false, true})});
        physical_network.set_node_attrs_data({
            dense_node_update(
                p_node_int.registry_id,
                {std::int64_t{10}, std::int64_t{20}, std::int64_t{30},
                 std::int64_t{40}, std::int64_t{50}}),
            dense_node_update(
                p_node_float.registry_id,
                {10.0, 20.0, 30.0, 40.0, 50.0}),
            dense_node_update(
                p_node_bool.registry_id,
                {true, false, true, false, true})});

        virtual_network.set_link_attrs_data({
            dense_link_update(
                v_link_int.registry_id,
                {std::int64_t{3}, std::int64_t{4}}),
            dense_link_update(v_link_float.registry_id, {1.5, 2.5}),
            dense_link_update(v_latency.registry_id, {5.0, 6.0})});
        physical_network.set_link_attrs_data({
            dense_link_update(
                p_link_int.registry_id,
                {std::int64_t{10}, std::int64_t{12}, std::int64_t{14},
                 std::int64_t{16}, std::int64_t{18}}),
            dense_link_update(
                p_link_float.registry_id,
                {8.0, 9.0, 10.0, 11.0, 12.0})});
    }

    ResourceUpdatorSelection selection(bool duplicate_link = true) const
    {
        ResourceUpdatorSelection result;
        result.node_resources = {
            v_node_int.registry_id,
            v_node_float.registry_id,
            v_node_bool.registry_id};
        result.link_resources = {
            v_link_int.registry_id,
            v_link_float.registry_id};
        if (duplicate_link)
        {
            result.link_resources.push_back(v_link_int.registry_id);
        }
        return result;
    }

    PreparedResourceUpdator prepare(bool duplicate_link = true)
    {
        return ResourceUpdator(selection(duplicate_link)).prepare(
            virtual_network, physical_network);
    }
};

const AttrValue& node_value(
    const Fixture& fixture,
    Vertex node,
    const network::NodeNetworkAttributeBinding& binding)
{
    return fixture.physical_network.graph().node_attrs(node).at(binding.value_id);
}

const AttrValue& link_value(
    const Fixture& fixture,
    ConstraintLink link,
    const network::LinkNetworkAttributeBinding& binding)
{
    const Graph& graph = fixture.physical_network.graph();
    return graph.edge_attrs(graph.edge(link.source, link.target)).at(
        binding.value_id);
}

void expect_integer(const AttrValue& value, std::int64_t expected, std::string_view message)
{
    const auto* actual = std::get_if<std::int64_t>(&value);
    expect(actual != nullptr && *actual == expected, message);
}

void expect_double(const AttrValue& value, double expected, std::string_view message)
{
    const auto* actual = std::get_if<double>(&value);
    expect(actual != nullptr && *actual == expected, message);
}

void test_selection_and_empty_behavior()
{
    Fixture fixture;
    const ResourceUpdator checker(fixture.selection());
    expect(
        checker.selection().node_resources == fixture.selection().node_resources,
        "selection node fields drift");

    ResourceUpdatorSelection invalid_node;
    invalid_node.node_resources = {fixture.v_node_status.registry_id};
    expect_updator_error(
        [&]
        {
            static_cast<void>(ResourceUpdator(invalid_node).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ResourceUpdatorErrorCode::invalid_node_selection,
        ResourceUpdatorOperation::prepare,
        std::nullopt,
        0U,
        fixture.v_node_status.registry_id);

    ResourceUpdatorSelection invalid_link;
    invalid_link.link_resources = {fixture.v_latency.registry_id};
    expect_updator_error(
        [&]
        {
            static_cast<void>(ResourceUpdator(invalid_link).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ResourceUpdatorErrorCode::invalid_link_selection,
        ResourceUpdatorOperation::prepare,
        std::nullopt,
        0U,
        fixture.v_latency.registry_id);

    ResourceUpdatorSelection invalid_id;
    invalid_id.node_resources = {controller::invalid_resource_id};
    expect_updator_error(
        [&]
        {
            static_cast<void>(ResourceUpdator(invalid_id).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ResourceUpdatorErrorCode::invalid_node_selection,
        ResourceUpdatorOperation::prepare,
        std::nullopt,
        0U,
        controller::invalid_resource_id);

    PreparedResourceUpdator empty =
        ResourceUpdator(ResourceUpdatorSelection{}).prepare(
            fixture.virtual_network, fixture.physical_network);
    empty.update_node_resources(
        99U, {}, ResourceUpdateOperation::subtract, true);
    empty.update_link_resources(
        {99U, 100U}, {}, ResourceUpdateOperation::subtract, true);
    empty.update_path_resources(
        {99U, 100U}, {0U}, ResourceUpdateOperation::subtract, true);
}

void test_scalar_node_updates()
{
    Fixture fixture;
    PreparedResourceUpdator prepared = fixture.prepare(false);
    prepared.update_node_resource(
        0U,
        {fixture.v_node_int.registry_id, std::int64_t{3}},
        ResourceUpdateOperation::subtract,
        true);
    expect_integer(
        node_value(fixture, 0U, fixture.p_node_int), 7,
        "node integer subtract");

    prepared.update_node_resource(
        0U,
        {fixture.v_node_float.registry_id, 2.5},
        ResourceUpdateOperation::add,
        true);
    expect_double(
        node_value(fixture, 0U, fixture.p_node_float), 12.5,
        "node floating add");

    prepared.update_node_resource(
        0U,
        {fixture.v_node_bool.registry_id, true},
        ResourceUpdateOperation::add,
        false);
    expect_integer(
        node_value(fixture, 0U, fixture.p_node_bool), 2,
        "boolean update must promote to integer");

    expect_updator_error(
        [&]
        {
            prepared.update_node_resource(
                0U,
                {fixture.v_node_int.registry_id, std::int64_t{99}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::insufficient_resource,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        std::nullopt,
        fixture.v_node_int.registry_id);
    expect_integer(
        node_value(fixture, 0U, fixture.p_node_int), 7,
        "safe failure mutated node");

    prepared.update_node_resource(
        0U,
        {fixture.v_node_int.registry_id, std::int64_t{9}},
        ResourceUpdateOperation::subtract,
        false);
    expect_integer(
        node_value(fixture, 0U, fixture.p_node_int), -2,
        "unsafe subtract rejected negative result");

    expect_updator_error(
        [&]
        {
            prepared.update_node_resource(
                99U,
                {controller::invalid_resource_id, std::int64_t{1}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::physical_node_out_of_range,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        std::nullopt,
        controller::invalid_resource_id);

    expect_updator_error(
        [&]
        {
            prepared.update_node_resource(
                0U,
                {controller::invalid_resource_id, std::int64_t{1}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::unprepared_resource_id,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        std::nullopt,
        controller::invalid_resource_id);
}

void test_scalar_partial_missing_and_numeric_errors()
{
    Fixture partial;
    PreparedResourceUpdator prepared = partial.prepare(false);
    expect_updator_error(
        [&]
        {
            prepared.update_node_resources(
                1U,
                {{partial.v_node_int.registry_id, std::int64_t{5}},
                 {partial.v_node_float.registry_id, 999.0}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::insufficient_resource,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        1U,
        partial.v_node_float.registry_id);
    expect_integer(
        node_value(partial, 1U, partial.p_node_int), 15,
        "node list lost earlier partial update");
    expect_double(
        node_value(partial, 1U, partial.p_node_float), 20.0,
        "failing node resource mutated");

    Fixture missing;
    PreparedResourceUpdator missing_prepared = missing.prepare(false);
    missing.physical_network.graph().node_attrs(0U).erase(
        missing.p_node_int.value_id);
    expect_updator_error(
        [&]
        {
            missing_prepared.update_node_resource(
                0U,
                {missing.v_node_int.registry_id, std::int64_t{1}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::missing_resource_value,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        std::nullopt,
        missing.v_node_int.registry_id);

    Fixture nonnumeric;
    PreparedResourceUpdator nonnumeric_prepared = nonnumeric.prepare(false);
    nonnumeric.physical_network.graph().node_attrs(0U).set(
        nonnumeric.p_node_int.value_id, std::string{"bad"});
    expect_updator_error(
        [&]
        {
            nonnumeric_prepared.update_node_resource(
                0U,
                {nonnumeric.v_node_int.registry_id, std::int64_t{1}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::non_numeric_resource,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        std::nullopt,
        nonnumeric.v_node_int.registry_id);

    Fixture overflow;
    PreparedResourceUpdator overflow_prepared = overflow.prepare(false);
    overflow.physical_network.graph().node_attrs(0U).set(
        overflow.p_node_int.value_id,
        std::numeric_limits<std::int64_t>::max());
    expect_updator_error(
        [&]
        {
            overflow_prepared.update_node_resource(
                0U,
                {overflow.v_node_int.registry_id, std::int64_t{1}},
                ResourceUpdateOperation::add,
                true);
        },
        ResourceUpdatorErrorCode::numeric_update_failure,
        ResourceUpdatorOperation::update_node,
        std::nullopt,
        std::nullopt,
        overflow.v_node_int.registry_id);
    expect_integer(
        node_value(overflow, 0U, overflow.p_node_int),
        std::numeric_limits<std::int64_t>::max(),
        "overflow mutated resource");
}

void test_scalar_link_and_path_updates()
{
    Fixture scalar;
    PreparedResourceUpdator scalar_prepared = scalar.prepare(false);
    scalar_prepared.update_link_resources(
        {1U, 0U},
        {{scalar.v_link_int.registry_id, std::int64_t{2}},
         {scalar.v_link_float.registry_id, 0.5}},
        ResourceUpdateOperation::subtract,
        true);
    expect_integer(
        link_value(scalar, {0U, 1U}, scalar.p_link_int), 8,
        "reversed link integer update");
    expect_double(
        link_value(scalar, {0U, 1U}, scalar.p_link_float), 7.5,
        "reversed link floating update");

    expect_updator_error(
        [&]
        {
            scalar_prepared.update_link_resource(
                {0U, 3U},
                {scalar.v_link_int.registry_id, std::int64_t{1}},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::physical_link_not_found,
        ResourceUpdatorOperation::update_link);

    Fixture path;
    PreparedResourceUpdator path_prepared = path.prepare(true);
    const std::int64_t first_integer_before = std::get<std::int64_t>(
        link_value(path, {0U, 1U}, path.p_link_int));
    const std::int64_t second_integer_before = std::get<std::int64_t>(
        link_value(path, {1U, 2U}, path.p_link_int));
    const double first_float_before = std::get<double>(
        link_value(path, {0U, 1U}, path.p_link_float));
    const double second_float_before = std::get<double>(
        link_value(path, {1U, 2U}, path.p_link_float));
    path_prepared.update_path_resources(
        {0U, 1U},
        {0U, 1U, 2U},
        ResourceUpdateOperation::subtract,
        true);
    expect_integer(
        link_value(path, {0U, 1U}, path.p_link_int),
        first_integer_before - 6,
        "duplicate first path resource did not repeat");
    expect_integer(
        link_value(path, {1U, 2U}, path.p_link_int),
        second_integer_before - 6,
        "duplicate second path resource did not repeat");
    expect_double(
        link_value(path, {0U, 1U}, path.p_link_float),
        first_float_before - 1.5,
        "first path floating update drift");
    expect_double(
        link_value(path, {1U, 2U}, path.p_link_float),
        second_float_before - 1.5,
        "second path floating update drift");

    Fixture short_path;
    PreparedResourceUpdator short_prepared = short_path.prepare(false);
    expect_updator_error(
        [&]
        {
            short_prepared.update_path_resources(
                {0U, 1U},
                {0U},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::invalid_path,
        ResourceUpdatorOperation::update_path);
    expect_updator_error(
        [&]
        {
            short_prepared.update_path_resources(
                {0U, 2U},
                {0U},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::virtual_link_not_found,
        ResourceUpdatorOperation::update_path);
    expect_updator_error(
        [&]
        {
            short_prepared.update_path_resources(
                {0U, 1U},
                {0U, 2U},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::physical_link_not_found,
        ResourceUpdatorOperation::update_path,
        std::nullopt,
        0U);
}

void test_path_partial_failure()
{
    Fixture fixture;
    PreparedResourceUpdator prepared = fixture.prepare(false);
    fixture.physical_network.graph()
        .edge_attrs(fixture.physical_network.graph().edge(1U, 2U))
        .set(fixture.p_link_int.value_id, std::int64_t{1});
    expect_updator_error(
        [&]
        {
            prepared.update_path_resources(
                {0U, 1U},
                {0U, 1U, 2U},
                ResourceUpdateOperation::subtract,
                true);
        },
        ResourceUpdatorErrorCode::insufficient_resource,
        ResourceUpdatorOperation::update_path,
        std::nullopt,
        1U,
        fixture.v_link_int.registry_id);
    expect_integer(
        link_value(fixture, {0U, 1U}, fixture.p_link_int), 7,
        "path lost earlier physical-link update");
    expect_integer(
        link_value(fixture, {1U, 2U}, fixture.p_link_int), 1,
        "path failing link mutated");
    expect_double(
        link_value(fixture, {0U, 1U}, fixture.p_link_float), 8.0,
        "path continued to next attribute after failure");
}

std::vector<AttributeNumber> node_snapshot(const Fixture& fixture)
{
    std::vector<AttributeNumber> result;
    for (Vertex node = 0U; node < 5U; ++node)
    {
        result.push_back(std::get<std::int64_t>(
            node_value(fixture, node, fixture.p_node_int)));
        result.push_back(std::get<double>(
            node_value(fixture, node, fixture.p_node_float)));
    }
    return result;
}

std::vector<AttributeNumber> link_snapshot(const Fixture& fixture)
{
    std::vector<AttributeNumber> result;
    for (const ConstraintLink link :
         {ConstraintLink{0U, 1U}, ConstraintLink{1U, 2U},
          ConstraintLink{2U, 3U}, ConstraintLink{3U, 4U},
          ConstraintLink{0U, 4U}})
    {
        result.push_back(std::get<std::int64_t>(
            link_value(fixture, link, fixture.p_link_int)));
        result.push_back(std::get<double>(
            link_value(fixture, link, fixture.p_link_float)));
    }
    return result;
}

void test_batches()
{
    const std::vector<NodeResourceUpdateRequest> node_requests = {
        {0U, {{0U, std::int64_t{1}}, {1U, 0.5}}},
        {1U, {{0U, std::int64_t{2}}, {1U, 1.5}}},
        {2U, {{0U, std::int64_t{3}}, {1U, 2.5}}},
        {3U, {{0U, std::int64_t{4}}, {1U, 3.5}}},
        {4U, {{0U, std::int64_t{5}}, {1U, 4.5}}}};
    std::vector<AttributeNumber> node_baseline;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        Fixture fixture;
        PreparedResourceUpdator prepared = fixture.prepare(false);
        prepared.update_node_resources_batch(
            node_requests, ResourceUpdateOperation::subtract, true, workers);
        const auto snapshot = node_snapshot(fixture);
        if (node_baseline.empty())
        {
            node_baseline = snapshot;
        }
        else
        {
            expect(snapshot == node_baseline, "node batch worker drift");
        }
    }

    const std::vector<LinkResourceUpdateRequest> link_requests = {
        {{0U, 1U}, {{0U, std::int64_t{1}}, {1U, 0.5}}},
        {{1U, 2U}, {{0U, std::int64_t{2}}, {1U, 1.5}}},
        {{2U, 3U}, {{0U, std::int64_t{3}}, {1U, 2.5}}},
        {{3U, 4U}, {{0U, std::int64_t{4}}, {1U, 3.5}}},
        {{0U, 4U}, {{0U, std::int64_t{5}}, {1U, 4.5}}}};
    std::vector<AttributeNumber> link_baseline;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        Fixture fixture;
        PreparedResourceUpdator prepared = fixture.prepare(false);
        prepared.update_link_resources_batch(
            link_requests, ResourceUpdateOperation::subtract, true, workers);
        const auto snapshot = link_snapshot(fixture);
        if (link_baseline.empty())
        {
            link_baseline = snapshot;
        }
        else
        {
            expect(snapshot == link_baseline, "link batch worker drift");
        }
    }

    Fixture duplicate;
    PreparedResourceUpdator duplicate_prepared = duplicate.prepare(false);
    duplicate_prepared.update_link_resources_batch(
        {{{0U, 1U}, {{duplicate.v_link_int.registry_id, std::int64_t{2}}}},
         {{1U, 0U}, {{duplicate.v_link_int.registry_id, std::int64_t{3}}}}},
        ResourceUpdateOperation::subtract,
        true,
        8U);
    expect_integer(
        link_value(duplicate, {0U, 1U}, duplicate.p_link_int), 5,
        "duplicate-target batch did not fall back sequentially");

    Fixture atomic;
    PreparedResourceUpdator atomic_prepared = atomic.prepare(false);
    const auto before = node_snapshot(atomic);
    expect_updator_error(
        [&]
        {
            atomic_prepared.update_node_resources_batch(
                {{0U, {{atomic.v_node_int.registry_id, std::int64_t{1}}}},
                 {1U, {{atomic.v_node_int.registry_id, std::int64_t{999}}}},
                 {99U, {{atomic.v_node_int.registry_id, std::int64_t{1}}}}},
                ResourceUpdateOperation::subtract,
                true,
                8U);
        },
        ResourceUpdatorErrorCode::insufficient_resource,
        ResourceUpdatorOperation::update_node_batch,
        1U,
        0U,
        atomic.v_node_int.registry_id);
    expect(
        node_snapshot(atomic) == before,
        "disjoint batch preflight failure mutated network");
}

void test_concurrent_independent_networks()
{
    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.push_back(std::async(
            std::launch::async,
            []
            {
                Fixture fixture;
                PreparedResourceUpdator prepared = fixture.prepare(false);
                for (std::size_t iteration = 0U; iteration < 32U; ++iteration)
                {
                    prepared.update_node_resource(
                        0U,
                        {fixture.v_node_float.registry_id, 0.25},
                        ResourceUpdateOperation::add,
                        false);
                    prepared.update_link_resource(
                        {0U, 1U},
                        {fixture.v_link_float.registry_id, 0.5},
                        ResourceUpdateOperation::add,
                        false);
                }
                return
                    std::get<double>(node_value(
                        fixture, 0U, fixture.p_node_float)) == 18.0 &&
                    std::get<double>(link_value(
                        fixture, {0U, 1U}, fixture.p_link_float)) == 24.0;
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(), "concurrent independent updater drift");
    }
}

} // namespace

int main()
{
    try
    {
        test_selection_and_empty_behavior();
        test_scalar_node_updates();
        test_scalar_partial_missing_and_numeric_errors();
        test_scalar_link_and_path_updates();
        test_path_partial_failure();
        test_batches();
        test_concurrent_independent_networks();
        std::cout << "resource updator unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "resource updator unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
