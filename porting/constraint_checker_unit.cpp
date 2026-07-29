#include "constraint_checker.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
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
using attribute::GraphResourceAttribute;
using attribute::GraphResourceSpec;
using controller::ConstraintCheckResult;
using controller::ConstraintChecker;
using controller::ConstraintCheckerErrorCode;
using controller::ConstraintCheckerException;
using controller::ConstraintCheckerOperation;
using controller::ConstraintCheckerSelection;
using controller::ConstraintId;
using controller::ConstraintLink;
using controller::GraphConstraintSelection;
using controller::LinkConstraintRequest;
using controller::NodeConstraintRequest;
using controller::PathConstraintCheckResult;
using controller::PathConstraintRequest;
using controller::PreparedConstraintChecker;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
ConstraintCheckerException expect_checker_error(
    Callable&& callable,
    ConstraintCheckerErrorCode code,
    ConstraintCheckerOperation operation,
    std::optional<std::size_t> request_index = std::nullopt,
    std::optional<std::size_t> item_index = std::nullopt,
    std::optional<ConstraintId> constraint_id = std::nullopt)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const ConstraintCheckerException& error)
    {
        expect(error.code() == code, "constraint checker error code mismatch");
        expect(
            error.operation() == operation,
            "constraint checker operation mismatch");
        expect(
            error.request_index() == request_index,
            "constraint checker request index mismatch");
        expect(
            error.item_index() == item_index,
            "constraint checker item index mismatch");
        expect(
            error.constraint_id() == constraint_id,
            "constraint checker ID mismatch");
        return error;
    }
    throw std::runtime_error("expected ConstraintCheckerException");
}

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner,
    ConstraintRestriction restriction,
    CheckingLevel level)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = restriction;
    result.checking_level = level;
    return result;
}

AttributeFactorySpec latency_spec(
    std::string name,
    ConstraintRestriction restriction)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::latency;
    result.restriction = restriction;
    result.checking_level = CheckingLevel::path;
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

GraphResourceSpec graph_resource_spec(
    std::string name,
    ConstraintRestriction restriction,
    CheckingLevel level = CheckingLevel::graph)
{
    GraphResourceSpec result;
    result.name = std::move(name);
    result.restriction = restriction;
    result.checking_level = level;
    return result;
}

network::NodeAttributeDataUpdate dense_node_update(
    ConstraintId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    ConstraintId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

network::VirtualNetwork make_virtual_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec(
            "node_hard", AttributeOwner::node,
            ConstraintRestriction::hard, CheckingLevel::node),
        resource_spec(
            "node_soft", AttributeOwner::node,
            ConstraintRestriction::soft, CheckingLevel::node),
        status_spec("node_status", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec(
            "link_hard", AttributeOwner::link,
            ConstraintRestriction::hard, CheckingLevel::link),
        resource_spec(
            "link_soft", AttributeOwner::link,
            ConstraintRestriction::soft, CheckingLevel::link),
        latency_spec("latency_hard", ConstraintRestriction::hard),
        latency_spec("latency_soft", ConstraintRestriction::soft),
        status_spec("link_status", AttributeOwner::link)};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 2U}, {1U, 3U}, {0U, 3U}});

    // Deliberately use different registry orders. Preparation must resolve a
    // selected virtual definition name once and bind the physical slots; it
    // must not copy a registry ID across independent registries.
    construction.config.node_attribute_specs = {
        status_spec("node_status", AttributeOwner::node),
        resource_spec(
            "node_soft", AttributeOwner::node,
            ConstraintRestriction::soft, CheckingLevel::node),
        resource_spec(
            "node_hard", AttributeOwner::node,
            ConstraintRestriction::hard, CheckingLevel::node)};
    construction.config.link_attribute_specs = {
        latency_spec("latency_soft", ConstraintRestriction::soft),
        status_spec("link_status", AttributeOwner::link),
        resource_spec(
            "link_hard", AttributeOwner::link,
            ConstraintRestriction::hard, CheckingLevel::link),
        latency_spec("latency_hard", ConstraintRestriction::hard),
        resource_spec(
            "link_soft", AttributeOwner::link,
            ConstraintRestriction::soft, CheckingLevel::link)};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
ConstraintId require_node_id(Network& value, std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    expect(binding.has_value(), "missing node fixture binding");
    return binding->registry_id;
}

template <typename Network>
ConstraintId require_link_id(Network& value, std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing link fixture binding");
    return binding->registry_id;
}

template <typename Network>
void set_graph_value(
    Network& value,
    const GraphResourceAttribute& definition,
    AttrValue graph_value)
{
    const auto binding = definition.bind(value.graph());
    definition.set_data(value.graph(), graph_value, binding);
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();

    GraphResourceAttribute graph_hard{
        graph_resource_spec(
            "graph_hard", ConstraintRestriction::hard)};
    GraphResourceAttribute graph_soft{
        graph_resource_spec(
            "graph_soft", ConstraintRestriction::soft)};
    GraphResourceAttribute graph_boolean{
        graph_resource_spec(
            "graph_boolean", ConstraintRestriction::hard)};
    GraphResourceAttribute graph_wrong_level{
        graph_resource_spec(
            "graph_wrong_level",
            ConstraintRestriction::hard,
            CheckingLevel::node)};

    ConstraintId node_hard = 0U;
    ConstraintId node_soft = 0U;
    ConstraintId node_status = 0U;
    ConstraintId link_hard = 0U;
    ConstraintId link_soft = 0U;
    ConstraintId latency_hard = 0U;
    ConstraintId latency_soft = 0U;
    ConstraintId link_status = 0U;

    Fixture()
    {
        node_hard = require_node_id(virtual_network, "node_hard");
        node_soft = require_node_id(virtual_network, "node_soft");
        node_status = require_node_id(virtual_network, "node_status");
        link_hard = require_link_id(virtual_network, "link_hard");
        link_soft = require_link_id(virtual_network, "link_soft");
        latency_hard = require_link_id(virtual_network, "latency_hard");
        latency_soft = require_link_id(virtual_network, "latency_soft");
        link_status = require_link_id(virtual_network, "link_status");

        const ConstraintId physical_node_hard =
            require_node_id(physical_network, "node_hard");
        const ConstraintId physical_node_soft =
            require_node_id(physical_network, "node_soft");
        const ConstraintId physical_link_hard =
            require_link_id(physical_network, "link_hard");
        const ConstraintId physical_link_soft =
            require_link_id(physical_network, "link_soft");
        const ConstraintId physical_latency_hard =
            require_link_id(physical_network, "latency_hard");
        const ConstraintId physical_latency_soft =
            require_link_id(physical_network, "latency_soft");

        expect(
            node_hard != physical_node_hard &&
                link_hard != physical_link_hard,
            "fixture registries must exercise independent IDs");

        virtual_network.set_node_attrs_data({
            dense_node_update(
                node_hard,
                {std::int64_t{5}, std::int64_t{7}, std::int64_t{1}}),
            dense_node_update(node_soft, {9.5, 2.0, 1.0})});
        physical_network.set_node_attrs_data({
            dense_node_update(
                physical_node_hard,
                {std::int64_t{8}, std::int64_t{3},
                 std::int64_t{9}, std::int64_t{6}}),
            dense_node_update(
                physical_node_soft,
                {4.0, 10.0, 3.0, 1.5})});

        virtual_network.set_link_attrs_data({
            sparse_link_update(
                link_hard,
                {{0U, 1U, std::int64_t{4}},
                 {1U, 2U, std::int64_t{2}}}),
            sparse_link_update(
                link_soft,
                {{0U, 1U, 9.5}, {1U, 2U, 1.0}}),
            sparse_link_update(
                latency_hard,
                {{0U, 1U, 10.0}, {1U, 2U, 4.0}}),
            sparse_link_update(
                latency_soft,
                {{0U, 1U, 5.0}, {1U, 2U, 3.0}})});
        physical_network.set_link_attrs_data({
            sparse_link_update(
                physical_link_hard,
                {{0U, 1U, std::int64_t{6}},
                 {1U, 2U, std::int64_t{2}},
                 {1U, 3U, std::int64_t{7}},
                 {0U, 3U, std::int64_t{7}}}),
            sparse_link_update(
                physical_link_soft,
                {{0U, 1U, 3.0}, {1U, 2U, 10.0},
                 {1U, 3U, 4.0}, {0U, 3U, 4.0}}),
            sparse_link_update(
                physical_latency_hard,
                {{0U, 1U, 3.0}, {1U, 2U, 4.0},
                 {1U, 3U, 9.0}, {0U, 3U, 6.0}}),
            sparse_link_update(
                physical_latency_soft,
                {{0U, 1U, 4.0}, {1U, 2U, 5.0},
                 {1U, 3U, 8.0}, {0U, 3U, 7.0}})});

        // Force independent graph AttrIds before binding fixed definitions.
        const AttrId physical_prefix =
            physical_network.graph().attr_id("physical_only_prefix");
        physical_network.graph().graph_attrs().set(
            physical_prefix, std::int64_t{99});

        set_graph_value(
            virtual_network, graph_hard, std::int64_t{7});
        set_graph_value(
            physical_network, graph_hard, std::int64_t{10});
        set_graph_value(virtual_network, graph_soft, 20.5);
        set_graph_value(physical_network, graph_soft, 5.0);
        set_graph_value(virtual_network, graph_boolean, false);
        set_graph_value(physical_network, graph_boolean, true);
        set_graph_value(
            virtual_network, graph_wrong_level, std::int64_t{1});
        set_graph_value(
            physical_network, graph_wrong_level, std::int64_t{2});
    }

    ConstraintCheckerSelection selection() const
    {
        ConstraintCheckerSelection result;
        result.node_at_node = {node_hard, node_soft, node_hard};
        result.link_at_link = {link_hard, link_soft, link_hard};
        result.link_at_path = {
            latency_hard, latency_soft, latency_hard};
        result.graph = {
            GraphConstraintSelection{20U, &graph_hard},
            GraphConstraintSelection{21U, &graph_soft},
            GraphConstraintSelection{22U, &graph_soft},
            GraphConstraintSelection{22U, &graph_boolean}};
        return result;
    }

    PreparedConstraintChecker prepare() const
    {
        return ConstraintChecker(selection()).prepare(
            virtual_network, physical_network);
    }
};

void expect_integer_offset(
    const virne::core::SolutionAttributeValues& offsets,
    ConstraintId id,
    std::int64_t expected,
    std::string_view message)
{
    const AttributeNumber* value = offsets.find(id);
    expect(value != nullptr, message);
    const auto* integer = std::get_if<std::int64_t>(value);
    expect(integer != nullptr && *integer == expected, message);
}

void expect_double_offset(
    const virne::core::SolutionAttributeValues& offsets,
    ConstraintId id,
    double expected,
    std::string_view message)
{
    const AttributeNumber* value = offsets.find(id);
    expect(value != nullptr, message);
    const auto* floating = std::get_if<double>(value);
    expect(floating != nullptr && *floating == expected, message);
}

bool check_result_equal(
    const ConstraintCheckResult& left,
    const ConstraintCheckResult& right)
{
    return left.feasible == right.feasible && left.offsets == right.offsets;
}

bool path_result_equal(
    const PathConstraintCheckResult& left,
    const PathConstraintCheckResult& right)
{
    if (left.feasible != right.feasible ||
        !(left.path_level == right.path_level) ||
        left.link_level.size() != right.link_level.size())
    {
        return false;
    }
    for (std::size_t index = 0U; index < left.link_level.size(); ++index)
    {
        if (!(left.link_level[index].physical_link ==
              right.link_level[index].physical_link) ||
            !(left.link_level[index].offsets ==
              right.link_level[index].offsets))
        {
            return false;
        }
    }
    return true;
}

void test_empty_and_selection_contract(Fixture& fixture)
{
    ConstraintChecker empty_checker(ConstraintCheckerSelection{});
    const PreparedConstraintChecker empty = empty_checker.prepare(
        fixture.virtual_network, fixture.physical_network);
    expect(
        empty.check_graph_constraints().feasible &&
            empty.check_graph_constraints().offsets.empty(),
        "empty graph selection must be feasible");
    expect(
        empty.check_node_level_constraints(0U, 0U).feasible &&
            empty.check_node_level_constraints(0U, 0U).offsets.empty(),
        "empty node selection must be feasible");
    expect(
        empty.check_link_level_constraints(
            {0U, 1U}, {0U, 1U}).feasible,
        "empty link selection must be feasible");

    const ConstraintCheckerSelection selection = fixture.selection();
    const ConstraintChecker checker(selection);
    expect(
        checker.selection().node_at_node == selection.node_at_node &&
            checker.selection().link_at_link == selection.link_at_link &&
            checker.selection().link_at_path == selection.link_at_path &&
            checker.selection().graph.size() == selection.graph.size(),
        "checker did not retain direct selection fields");
}

void test_scalar_graph_node_and_link(Fixture& fixture)
{
    const PreparedConstraintChecker prepared = fixture.prepare();

    const ConstraintCheckResult graph = prepared.check_graph_constraints();
    expect(graph.feasible, "hard-pass and soft-fail graph constraints");
    expect_integer_offset(graph.offsets, 20U, -3, "graph hard offset");
    expect_double_offset(graph.offsets, 21U, 15.5, "graph soft offset");
    expect_integer_offset(
        graph.offsets, 22U, -1,
        "duplicate graph output ID must retain the last offset");

    const ConstraintCheckResult node_success =
        prepared.check_node_level_constraints(0U, 0U);
    expect(node_success.feasible, "soft node failure must remain feasible");
    expect_integer_offset(
        node_success.offsets, fixture.node_hard, -3,
        "node hard success offset");
    expect_double_offset(
        node_success.offsets, fixture.node_soft, 5.5,
        "node soft violation offset");

    const ConstraintCheckResult node_failure =
        prepared.check_node_level_constraints(0U, 1U);
    expect(!node_failure.feasible, "hard node failure was ignored");
    expect_integer_offset(
        node_failure.offsets, fixture.node_hard, 2,
        "node hard failure offset");
    expect_double_offset(
        node_failure.offsets, fixture.node_soft, -0.5,
        "node soft success offset");

    const ConstraintCheckResult link_success =
        prepared.check_link_level_constraints({0U, 1U}, {0U, 1U});
    expect(link_success.feasible, "soft link failure must remain feasible");
    expect_integer_offset(
        link_success.offsets, fixture.link_hard, -2,
        "link hard success offset");
    expect_double_offset(
        link_success.offsets, fixture.link_soft, 6.5,
        "link soft violation offset");

    const ConstraintCheckResult link_failure =
        prepared.check_link_level_constraints({0U, 1U}, {1U, 2U});
    expect(!link_failure.feasible, "hard link failure was ignored");
    expect_integer_offset(
        link_failure.offsets, fixture.link_hard, 2,
        "link hard failure offset");
    expect_double_offset(
        link_failure.offsets, fixture.link_soft, -0.5,
        "link soft success offset");

    const ConstraintCheckResult reversed =
        prepared.check_link_level_constraints({1U, 0U}, {1U, 0U});
    expect(
        check_result_equal(reversed, link_success),
        "undirected reversed endpoints changed link results");
}

void expect_link_offsets(
    const controller::PhysicalLinkConstraintResult& result,
    ConstraintLink expected_link,
    ConstraintId hard_id,
    std::int64_t hard_offset,
    ConstraintId soft_id,
    double soft_offset)
{
    expect(result.physical_link == expected_link, "path link order mismatch");
    expect_integer_offset(
        result.offsets, hard_id, hard_offset,
        "path link hard offset mismatch");
    expect_double_offset(
        result.offsets, soft_id, soft_offset,
        "path link soft offset mismatch");
}

void test_scalar_paths(Fixture& fixture)
{
    const PreparedConstraintChecker prepared = fixture.prepare();

    const PathConstraintCheckResult success =
        prepared.check_path_level_constraints({0U, 1U}, {0U, 3U});
    expect(success.feasible, "soft path failure must remain feasible");
    expect(success.link_level.size() == 1U, "direct path link count");
    expect_link_offsets(
        success.link_level[0], {0U, 3U},
        fixture.link_hard, -3,
        fixture.link_soft, 5.5);
    expect_double_offset(
        success.path_level, fixture.latency_hard, -4.0,
        "path hard success offset");
    expect_double_offset(
        success.path_level, fixture.latency_soft, 2.0,
        "path soft violation offset");

    const PathConstraintCheckResult link_failure =
        prepared.check_path_level_constraints(
            {0U, 1U}, {0U, 1U, 2U});
    expect(!link_failure.feasible, "path ignored link-level failure");
    expect(link_failure.link_level.size() == 2U, "two-link path count");
    expect_link_offsets(
        link_failure.link_level[0], {0U, 1U},
        fixture.link_hard, -2,
        fixture.link_soft, 6.5);
    expect_link_offsets(
        link_failure.link_level[1], {1U, 2U},
        fixture.link_hard, 2,
        fixture.link_soft, -0.5);
    expect_double_offset(
        link_failure.path_level, fixture.latency_hard, -3.0,
        "combined path hard offset");
    expect_double_offset(
        link_failure.path_level, fixture.latency_soft, 4.0,
        "combined path soft offset");

    const PathConstraintCheckResult latency_failure =
        prepared.check_path_level_constraints(
            {0U, 1U}, {0U, 1U, 3U});
    expect(!latency_failure.feasible, "path ignored hard latency failure");
    expect(latency_failure.link_level.size() == 2U, "latency path count");
    expect_double_offset(
        latency_failure.path_level, fixture.latency_hard, 2.0,
        "hard latency failure offset");
    expect_double_offset(
        latency_failure.path_level, fixture.latency_soft, 7.0,
        "soft latency failure offset");
}

void test_prepare_errors(Fixture& fixture)
{
    ConstraintCheckerSelection invalid_node;
    invalid_node.node_at_node = {fixture.node_status};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(invalid_node).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::invalid_node_selection,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        fixture.node_status);

    ConstraintCheckerSelection invalid_node_id;
    invalid_node_id.node_at_node = {controller::invalid_constraint_id};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(invalid_node_id).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::invalid_node_selection,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        controller::invalid_constraint_id);

    ConstraintCheckerSelection invalid_link;
    invalid_link.link_at_link = {fixture.latency_hard};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(invalid_link).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::invalid_link_selection,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        fixture.latency_hard);

    ConstraintCheckerSelection invalid_path;
    invalid_path.link_at_path = {fixture.link_hard};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(invalid_path).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::invalid_path_selection,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        fixture.link_hard);

    ConstraintCheckerSelection null_graph;
    null_graph.graph = {GraphConstraintSelection{31U, nullptr}};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(null_graph).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::null_graph_attribute,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        31U);

    ConstraintCheckerSelection invalid_graph;
    invalid_graph.graph = {
        GraphConstraintSelection{32U, &fixture.graph_wrong_level}};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(invalid_graph).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::invalid_graph_selection,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        32U);

    ConstraintCheckerSelection invalid_graph_id;
    invalid_graph_id.graph = {GraphConstraintSelection{
        controller::invalid_constraint_id, &fixture.graph_hard}};
    expect_checker_error(
        [&]
        {
            static_cast<void>(ConstraintChecker(invalid_graph_id).prepare(
                fixture.virtual_network, fixture.physical_network));
        },
        ConstraintCheckerErrorCode::invalid_graph_selection,
        ConstraintCheckerOperation::prepare,
        std::nullopt,
        0U,
        controller::invalid_constraint_id);
}

void test_runtime_errors(Fixture& fixture)
{
    const PreparedConstraintChecker prepared = fixture.prepare();
    expect_checker_error(
        [&]
        {
            static_cast<void>(
                prepared.check_node_level_constraints(99U, 99U));
        },
        ConstraintCheckerErrorCode::physical_node_out_of_range,
        ConstraintCheckerOperation::check_node);
    expect_checker_error(
        [&]
        {
            static_cast<void>(
                prepared.check_node_level_constraints(99U, 0U));
        },
        ConstraintCheckerErrorCode::virtual_node_out_of_range,
        ConstraintCheckerOperation::check_node);
    expect_checker_error(
        [&]
        {
            static_cast<void>(prepared.check_link_level_constraints(
                {0U, 2U}, {2U, 3U}));
        },
        ConstraintCheckerErrorCode::virtual_link_not_found,
        ConstraintCheckerOperation::check_link);
    expect_checker_error(
        [&]
        {
            static_cast<void>(prepared.check_link_level_constraints(
                {0U, 1U}, {2U, 3U}));
        },
        ConstraintCheckerErrorCode::physical_link_not_found,
        ConstraintCheckerOperation::check_link);
    expect_checker_error(
        [&]
        {
            static_cast<void>(prepared.check_path_level_constraints(
                {0U, 1U}, {0U}));
        },
        ConstraintCheckerErrorCode::invalid_path,
        ConstraintCheckerOperation::check_path);
    expect_checker_error(
        [&]
        {
            static_cast<void>(prepared.check_path_level_constraints(
                {0U, 1U}, {0U, 2U}));
        },
        ConstraintCheckerErrorCode::physical_link_not_found,
        ConstraintCheckerOperation::check_path,
        std::nullopt,
        0U);
}

void test_batch_workers_and_lowest_error(Fixture& fixture)
{
    const PreparedConstraintChecker prepared = fixture.prepare();
    const std::vector<NodeConstraintRequest> node_requests = {
        {0U, 0U}, {0U, 1U}, {1U, 2U}, {2U, 3U}};
    const std::vector<LinkConstraintRequest> link_requests = {
        {{0U, 1U}, {0U, 1U}},
        {{0U, 1U}, {1U, 2U}},
        {{1U, 0U}, {1U, 0U}},
        {{1U, 2U}, {1U, 3U}}};
    const std::vector<PathConstraintRequest> path_requests = {
        {{0U, 1U}, {0U, 3U}},
        {{0U, 1U}, {0U, 1U, 2U}},
        {{0U, 1U}, {0U, 1U, 3U}},
        {{1U, 0U}, {3U, 0U}}};

    const auto node_baseline =
        prepared.check_node_level_constraints_batch(node_requests, 1U);
    const auto link_baseline =
        prepared.check_link_level_constraints_batch(link_requests, 1U);
    const auto path_baseline =
        prepared.check_path_level_constraints_batch(path_requests, 1U);

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        const auto nodes =
            prepared.check_node_level_constraints_batch(node_requests, workers);
        const auto links =
            prepared.check_link_level_constraints_batch(link_requests, workers);
        const auto paths =
            prepared.check_path_level_constraints_batch(path_requests, workers);
        expect(nodes.size() == node_baseline.size(), "node batch size");
        expect(links.size() == link_baseline.size(), "link batch size");
        expect(paths.size() == path_baseline.size(), "path batch size");
        for (std::size_t index = 0U; index < nodes.size(); ++index)
        {
            expect(
                check_result_equal(nodes[index], node_baseline[index]),
                "node worker result/order mismatch");
        }
        for (std::size_t index = 0U; index < links.size(); ++index)
        {
            expect(
                check_result_equal(links[index], link_baseline[index]),
                "link worker result/order mismatch");
        }
        for (std::size_t index = 0U; index < paths.size(); ++index)
        {
            expect(
                path_result_equal(paths[index], path_baseline[index]),
                "path worker result/order mismatch");
        }
    }

    const std::vector<NodeConstraintRequest> failures = {
        {0U, 0U}, {0U, 99U}, {1U, 2U}, {99U, 0U}};
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect_checker_error(
            [&]
            {
                static_cast<void>(
                    prepared.check_node_level_constraints_batch(
                        failures, workers));
            },
            ConstraintCheckerErrorCode::physical_node_out_of_range,
            ConstraintCheckerOperation::check_node,
            1U);
    }
}

void test_concurrent_callers(Fixture& fixture)
{
    const PreparedConstraintChecker prepared = fixture.prepare();
    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.push_back(std::async(
            std::launch::async,
            [&prepared]
            {
                for (std::size_t iteration = 0U;
                     iteration < 32U;
                     ++iteration)
                {
                    const auto graph = prepared.check_graph_constraints();
                    const auto node =
                        prepared.check_node_level_constraints(0U, 0U);
                    const auto link = prepared.check_link_level_constraints(
                        {0U, 1U}, {0U, 1U});
                    const auto path = prepared.check_path_level_constraints(
                        {0U, 1U}, {0U, 3U});
                    if (!graph.feasible || !node.feasible ||
                        !link.feasible || !path.feasible)
                    {
                        return false;
                    }
                }
                return true;
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(), "concurrent prepared checker drift");
    }
}

} // namespace

int main()
{
    try
    {
        Fixture fixture;
        const auto run = [&](std::string_view name, auto&& test)
        {
            try
            {
                test(fixture);
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(
                    std::string(name) + ": " + error.what());
            }
        };
        run("empty/selection", test_empty_and_selection_contract);
        run("scalar graph/node/link", test_scalar_graph_node_and_link);
        run("scalar path", test_scalar_paths);
        run("prepare errors", test_prepare_errors);
        run("runtime errors", test_runtime_errors);
        run("batch workers/errors", test_batch_workers_and_lowest_error);
        run("concurrent callers", test_concurrent_callers);
        std::cout << "constraint checker unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "constraint checker unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
