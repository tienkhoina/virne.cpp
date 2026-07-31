#include "link_mapper.h"

#include <algorithm>
#include <array>
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
namespace core = virne::core;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using controller::ConstraintId;
using controller::ConstraintLink;
using controller::LinkMapper;
using controller::LinkMapperErrorCode;
using controller::LinkMapperException;
using controller::LinkMapperOperation;
using controller::LinkMapperSelection;
using controller::LinkMappingOptions;
using controller::LinkPathRanker;
using controller::LinkRouteCheckInfo;
using controller::LinkRouteOptions;
using controller::LinkRouteResult;
using controller::PhysicalLinkConstraintResult;
using controller::PhysicalPaths;
using controller::PreparedLinkMapper;
using controller::ResourceId;
using controller::ResourceUpdatorErrorCode;
using controller::ResourceUpdatorException;
using controller::ResourceUpdatorOperation;
using controller::ShortestPathMethod;

constexpr ConstraintLink first_virtual_link{0U, 1U};
constexpr ConstraintLink second_virtual_link{1U, 2U};

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
LinkMapperException expect_mapper_error(
    Callable&& callable,
    LinkMapperErrorCode code,
    LinkMapperOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const LinkMapperException& error)
    {
        expect(error.code() == code, "link mapper error code mismatch");
        expect(error.operation() == operation,
               "link mapper operation mismatch");
        expect(!std::string_view(error.what()).empty(),
               "link mapper diagnostic is empty");
        return error;
    }
    throw std::runtime_error("expected LinkMapperException");
}

template <typename Callable>
void expect_resource_error(
    Callable&& callable,
    ResourceUpdatorErrorCode code,
    ResourceUpdatorOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const ResourceUpdatorException& error)
    {
        expect(error.code() == code, "resource error code mismatch");
        expect(error.operation() == operation,
               "resource operation mismatch");
        return;
    }
    throw std::runtime_error("expected ResourceUpdatorException");
}

AttributeFactorySpec resource_spec(
    std::string name,
    ConstraintRestriction restriction)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::resource;
    result.restriction = restriction;
    result.checking_level = CheckingLevel::link;
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

AttributeFactorySpec status_spec(std::string name)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::status;
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
    construction.config.link_attribute_specs = {
        resource_spec("bw", ConstraintRestriction::hard),
        resource_spec("soft_bw", ConstraintRestriction::soft),
        latency_spec("latency_hard", ConstraintRestriction::hard),
        latency_spec("latency_soft", ConstraintRestriction::soft),
        resource_spec("aux", ConstraintRestriction::hard),
        status_spec("link_status")};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        8U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {1U, 5U},
            {0U, 2U}, {2U, 5U},
            {0U, 3U}, {3U, 4U}, {4U, 5U},
            {5U, 6U}});

    // Every selected definition intentionally receives a registry ID that
    // differs from its virtual-network ID.
    construction.config.link_attribute_specs = {
        latency_spec("latency_soft", ConstraintRestriction::soft),
        status_spec("link_status"),
        resource_spec("aux", ConstraintRestriction::hard),
        resource_spec("bw", ConstraintRestriction::hard),
        latency_spec("latency_hard", ConstraintRestriction::hard),
        resource_spec("soft_bw", ConstraintRestriction::soft)};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing fixture link binding");
    return *binding;
}

const std::array<ConstraintLink, 8U>& physical_links()
{
    static const std::array<ConstraintLink, 8U> result{{
        {0U, 1U}, {1U, 5U},
        {0U, 2U}, {2U, 5U},
        {0U, 3U}, {3U, 4U}, {4U, 5U},
        {5U, 6U}}};
    return result;
}

const std::vector<core::SolutionLink>& first_path()
{
    static const std::vector<core::SolutionLink> result{
        {0, 1}, {1, 5}};
    return result;
}

const std::vector<core::SolutionLink>& second_path()
{
    static const std::vector<core::SolutionLink> result{
        {0, 2}, {2, 5}};
    return result;
}

const std::vector<core::SolutionLink>& third_path()
{
    static const std::vector<core::SolutionLink> result{
        {0, 3}, {3, 4}, {4, 5}};
    return result;
}

core::Solution make_solution()
{
    core::SolutionMetadata metadata;
    metadata.v_net_id = 29;
    metadata.v_net_lifetime = 10.0;
    metadata.v_net_arrival_time = 2.0;
    metadata.v_net_num_nodes = 3U;
    metadata.v_net_num_edges = 2U;
    return core::Solution(metadata);
}

core::Solution make_solution_with_slots()
{
    core::Solution result = make_solution();
    result.node_slots.insert_or_assign(0, 0);
    result.node_slots.insert_or_assign(1, 5);
    result.node_slots.insert_or_assign(2, 6);
    return result;
}

template <typename OrderedTable, typename Key>
const typename OrderedTable::Entry& require_entry(
    const OrderedTable& table,
    const Key& key,
    std::string_view message)
{
    const auto id = table.find_id(key);
    expect(id.has_value(), message);
    return table.entries().at(id->value);
}

const std::vector<core::SolutionLink>& require_route(
    const core::Solution& solution,
    ConstraintLink virtual_link)
{
    return require_entry(
        solution.link_paths,
        core::SolutionLink{
            static_cast<core::SolutionNodeId>(virtual_link.source),
            static_cast<core::SolutionNodeId>(virtual_link.target)},
        "missing routed link").value;
}

const core::SolutionAttributeValues& require_info(
    const core::Solution& solution,
    ConstraintLink virtual_link,
    ConstraintLink physical_link)
{
    const core::LinkPathInfoKey key{
        {static_cast<core::SolutionNodeId>(virtual_link.source),
         static_cast<core::SolutionNodeId>(virtual_link.target)},
        {static_cast<core::SolutionNodeId>(physical_link.source),
         static_cast<core::SolutionNodeId>(physical_link.target)}};
    return require_entry(
        solution.link_paths_info, key, "missing route info").value;
}

template <typename Table>
const core::SolutionAttributeValues& require_link_values(
    const Table& table,
    ConstraintLink virtual_link,
    std::string_view message)
{
    return require_entry(
        table,
        core::SolutionLink{
            static_cast<core::SolutionNodeId>(virtual_link.source),
            static_cast<core::SolutionNodeId>(virtual_link.target)},
        message).value;
}

void expect_integer(
    const core::SolutionAttributeValues& values,
    ConstraintId id,
    std::int64_t expected,
    std::string_view message)
{
    const auto* value = values.find(id);
    expect(value != nullptr, message);
    const auto* integer = std::get_if<std::int64_t>(value);
    expect(integer != nullptr && *integer == expected, message);
}

void expect_double(
    const core::SolutionAttributeValues& values,
    ConstraintId id,
    double expected,
    std::string_view message)
{
    const auto* value = values.find(id);
    expect(value != nullptr, message);
    const auto* floating = std::get_if<double>(value);
    expect(floating != nullptr && *floating == expected, message);
}

std::size_t populated_values(const core::SolutionAttributeValues& values)
{
    return static_cast<std::size_t>(std::count_if(
        values.slots().begin(),
        values.slots().end(),
        [](const auto& value) { return value.has_value(); }));
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();

    network::LinkNetworkAttributeBinding v_bw;
    network::LinkNetworkAttributeBinding v_soft;
    network::LinkNetworkAttributeBinding v_latency_hard;
    network::LinkNetworkAttributeBinding v_latency_soft;
    network::LinkNetworkAttributeBinding v_aux;
    network::LinkNetworkAttributeBinding v_status;
    network::LinkNetworkAttributeBinding p_bw;
    network::LinkNetworkAttributeBinding p_soft;
    network::LinkNetworkAttributeBinding p_latency_hard;
    network::LinkNetworkAttributeBinding p_latency_soft;
    network::LinkNetworkAttributeBinding p_aux;

    Fixture()
        : v_bw(require_link_binding(virtual_network, "bw")),
          v_soft(require_link_binding(virtual_network, "soft_bw")),
          v_latency_hard(
              require_link_binding(virtual_network, "latency_hard")),
          v_latency_soft(
              require_link_binding(virtual_network, "latency_soft")),
          v_aux(require_link_binding(virtual_network, "aux")),
          v_status(require_link_binding(virtual_network, "link_status")),
          p_bw(require_link_binding(physical_network, "bw")),
          p_soft(require_link_binding(physical_network, "soft_bw")),
          p_latency_hard(
              require_link_binding(physical_network, "latency_hard")),
          p_latency_soft(
              require_link_binding(physical_network, "latency_soft")),
          p_aux(require_link_binding(physical_network, "aux"))
    {
        expect(
            v_bw.registry_id != p_bw.registry_id &&
                v_soft.registry_id != p_soft.registry_id &&
                v_latency_hard.registry_id !=
                    p_latency_hard.registry_id &&
                v_latency_soft.registry_id !=
                    p_latency_soft.registry_id &&
                v_aux.registry_id != p_aux.registry_id,
            "fixture must exercise independent registries");

        virtual_network.set_link_attrs_data({
            sparse_link_update(
                v_bw.registry_id,
                {{0U, 1U, std::int64_t{5}},
                 {1U, 2U, std::int64_t{2}}}),
            sparse_link_update(
                v_soft.registry_id,
                {{0U, 1U, 8.0}, {1U, 2U, 1.0}}),
            sparse_link_update(
                v_latency_hard.registry_id,
                {{0U, 1U, 8.0}, {1U, 2U, 4.0}}),
            sparse_link_update(
                v_latency_soft.registry_id,
                {{0U, 1U, 5.0}, {1U, 2U, 3.0}}),
            sparse_link_update(
                v_aux.registry_id,
                {{0U, 1U, std::int64_t{2}},
                 {1U, 2U, std::int64_t{1}}})});

        physical_network.set_link_attrs_data({
            sparse_link_update(
                p_bw.registry_id,
                {{0U, 1U, std::int64_t{4}},
                 {1U, 5U, std::int64_t{12}},
                 {0U, 2U, std::int64_t{8}},
                 {2U, 5U, std::int64_t{12}},
                 {0U, 3U, std::int64_t{12}},
                 {3U, 4U, std::int64_t{12}},
                 {4U, 5U, std::int64_t{12}},
                 {5U, 6U, std::int64_t{12}}}),
            sparse_link_update(
                p_soft.registry_id,
                {{0U, 1U, 1.0}, {1U, 5U, 1.0},
                 {0U, 2U, 1.0}, {2U, 5U, 1.0},
                 {0U, 3U, 1.0}, {3U, 4U, 1.0},
                 {4U, 5U, 1.0}, {5U, 6U, 1.0}}),
            sparse_link_update(
                p_latency_hard.registry_id,
                {{0U, 1U, 2.0}, {1U, 5U, 2.0},
                 {0U, 2U, 3.0}, {2U, 5U, 3.0},
                 {0U, 3U, 1.0}, {3U, 4U, 1.0},
                 {4U, 5U, 1.0}, {5U, 6U, 1.0}}),
            sparse_link_update(
                p_latency_soft.registry_id,
                {{0U, 1U, 2.0}, {1U, 5U, 2.0},
                 {0U, 2U, 3.0}, {2U, 5U, 3.0},
                 {0U, 3U, 1.0}, {3U, 4U, 1.0},
                 {4U, 5U, 1.0}, {5U, 6U, 1.0}}),
            sparse_link_update(
                p_aux.registry_id,
                {{0U, 1U, std::int64_t{10}},
                 {1U, 5U, std::int64_t{10}},
                 {0U, 2U, std::int64_t{10}},
                 {2U, 5U, std::int64_t{10}},
                 {0U, 3U, std::int64_t{10}},
                 {3U, 4U, std::int64_t{10}},
                 {4U, 5U, std::int64_t{10}},
                 {5U, 6U, std::int64_t{10}}})});
    }

    LinkMapperSelection default_selection() const
    {
        LinkMapperSelection selection;
        selection.link_constraints = {
            v_bw.registry_id, v_soft.registry_id};
        selection.path_constraints = {
            v_latency_hard.registry_id, v_latency_soft.registry_id};
        selection.link_resources = {
            v_bw.registry_id, v_aux.registry_id, v_bw.registry_id};
        selection.hard_constraints = {
            v_bw.registry_id, v_latency_hard.registry_id};
        return selection;
    }

    PreparedLinkMapper prepare()
    {
        return LinkMapper(default_selection()).prepare(
            virtual_network, physical_network);
    }

    PreparedLinkMapper prepare(LinkMapperSelection selection)
    {
        return LinkMapper(std::move(selection)).prepare(
            virtual_network, physical_network);
    }

    AttrMap& virtual_values(ConstraintLink link)
    {
        return virtual_network.graph().edge_attrs(
            virtual_network.graph().edge(link.source, link.target));
    }

    AttrMap& physical_values(ConstraintLink link)
    {
        return physical_network.graph().edge_attrs(
            physical_network.graph().edge(link.source, link.target));
    }

    void set_virtual(
        ConstraintLink link,
        network::LinkNetworkAttributeBinding binding,
        AttrValue value)
    {
        virtual_values(link).set(binding.value_id, std::move(value));
    }

    void set_physical(
        ConstraintLink link,
        network::LinkNetworkAttributeBinding binding,
        AttrValue value)
    {
        physical_values(link).set(binding.value_id, std::move(value));
    }

    std::int64_t physical_integer(
        ConstraintLink link,
        network::LinkNetworkAttributeBinding binding) const
    {
        const auto edge = physical_network.graph().edge(
            link.source, link.target);
        return std::get<std::int64_t>(
            physical_network.graph().edge_attrs(edge).at(binding.value_id));
    }

    std::int64_t bw(ConstraintLink link) const
    {
        return physical_integer(link, p_bw);
    }

    std::int64_t aux(ConstraintLink link) const
    {
        return physical_integer(link, p_aux);
    }
};

LinkRouteOptions safe_k_options()
{
    LinkRouteOptions result;
    result.shortest_method = ShortestPathMethod::k_shortest;
    result.k = 10;
    result.record_constraint_violation = false;
    return result;
}

void test_selection_registry_duplicate_resources_and_boundary()
{
    Fixture fixture;
    const LinkMapperSelection selection = fixture.default_selection();
    const LinkMapper mapper(selection);
    expect(
        mapper.selection().link_constraints == selection.link_constraints &&
            mapper.selection().path_constraints ==
                selection.path_constraints &&
            mapper.selection().link_resources == selection.link_resources &&
            mapper.selection().hard_constraints ==
                selection.hard_constraints &&
            mapper.selection().reusable == selection.reusable,
        "mapper did not retain typed selection fields");

    PreparedLinkMapper prepared = fixture.prepare();
    core::Solution solution = make_solution();
    const LinkRouteResult result = prepared.route(
        first_virtual_link, {0U, 5U}, solution, safe_k_options());
    expect(result.routed && result.check.constraints.feasible,
           "safe route failed");
    expect(require_route(solution, first_virtual_link) == second_path(),
           "first feasible path/order mismatch");
    expect(fixture.bw({0U, 2U}) == 3 && fixture.bw({2U, 5U}) == 7,
           "duplicate bandwidth resource was applied more than once");
    expect(fixture.aux({0U, 2U}) == 8 && fixture.aux({2U, 5U}) == 8,
           "auxiliary resource subtraction mismatch");
    const auto& info = require_info(
        solution, first_virtual_link, {0U, 2U});
    expect_integer(info, fixture.v_bw.registry_id, 5,
                   "recorded bandwidth demand mismatch");
    expect_integer(info, fixture.v_aux.registry_id, 2,
                   "recorded auxiliary demand mismatch");
    expect(populated_values(info) == 2U,
           "duplicate resource leaked into route info");

    Fixture invalid_fixture;
    LinkMapperSelection invalid = invalid_fixture.default_selection();
    invalid.link_resources = {invalid_fixture.v_status.registry_id};
    expect_mapper_error(
        [&]
        {
            static_cast<void>(invalid_fixture.prepare(invalid));
        },
        LinkMapperErrorCode::invalid_link_resource_selection,
        LinkMapperOperation::prepare);
}

void test_same_node_no_path_all_infeasible_and_ranker()
{
    Fixture same_fixture;
    PreparedLinkMapper same_mapper = same_fixture.prepare();
    core::Solution same_solution = make_solution();
    const LinkMapperException same_error = expect_mapper_error(
        [&]
        {
            static_cast<void>(same_mapper.route(
                first_virtual_link, {2U, 2U}, same_solution));
        },
        LinkMapperErrorCode::same_physical_node,
        LinkMapperOperation::route);
    expect(same_error.virtual_link() ==
               std::optional<ConstraintLink>{first_virtual_link} &&
               same_error.physical_link() ==
                   std::optional<ConstraintLink>{{2U, 2U}},
           "same-node error context mismatch");
    expect(same_solution.link_paths.empty(),
           "same-node rejection mutated paths");

    Fixture reusable_fixture;
    LinkMapperSelection reusable_selection =
        reusable_fixture.default_selection();
    reusable_selection.reusable = true;
    PreparedLinkMapper reusable_mapper =
        reusable_fixture.prepare(std::move(reusable_selection));
    core::Solution reusable_solution = make_solution();
    const LinkRouteResult reused = reusable_mapper.route(
        first_virtual_link, {2U, 2U}, reusable_solution);
    expect(reused.routed && reused.check.placeholder,
           "reusable same-node route failed");
    expect(reusable_solution.link_paths.empty() &&
               reusable_solution.link_paths_info.empty() &&
               reusable_solution.v_net_constraint_offsets.link_level.empty(),
           "reusable same-node route wrote solution state");

    Fixture no_path_fixture;
    PreparedLinkMapper no_path_mapper = no_path_fixture.prepare();
    core::Solution no_path_solution = make_solution();
    LinkRouteOptions no_path_options = safe_k_options();
    no_path_options.record_constraint_violation = true;
    const LinkRouteResult no_path = no_path_mapper.route(
        first_virtual_link,
        {0U, 7U},
        no_path_solution,
        no_path_options);
    expect(!no_path.routed && no_path.check.placeholder,
           "missing path did not return placeholder");
    expect(require_route(no_path_solution, first_virtual_link).empty(),
           "missing path did not retain canonical empty route");
    const auto& sentinel_link_offsets = require_link_values(
        no_path_solution.v_net_constraint_offsets.link_level,
        first_virtual_link,
        "missing placeholder link offsets");
    const auto& sentinel_link_violations = require_link_values(
        no_path_solution.v_net_constraint_violations.link_level,
        first_virtual_link,
        "missing placeholder link violations");
    const auto& sentinel_path_offsets = require_link_values(
        no_path_solution.v_net_constraint_offsets.path_level,
        first_virtual_link,
        "missing placeholder path offsets");
    const auto& sentinel_path_violations = require_link_values(
        no_path_solution.v_net_constraint_violations.path_level,
        first_virtual_link,
        "missing placeholder path violations");
    expect_double(sentinel_link_offsets, no_path_fixture.v_bw.registry_id,
                  100.0, "placeholder link offset type/value mismatch");
    expect_double(sentinel_link_violations,
                  no_path_fixture.v_bw.registry_id,
                  100.0,
                  "placeholder link violation type/value mismatch");
    expect_double(sentinel_path_offsets,
                  no_path_fixture.v_latency_hard.registry_id,
                  0.0,
                  "placeholder path offset type/value mismatch");
    expect_double(sentinel_path_violations,
                  no_path_fixture.v_latency_hard.registry_id,
                  0.0,
                  "placeholder path violation type/value mismatch");
    expect(no_path_solution.v_net_total_hard_constraint_violation == 100.0,
           "placeholder hard violation mismatch");

    Fixture infeasible_fixture;
    infeasible_fixture.set_virtual(
        first_virtual_link,
        infeasible_fixture.v_latency_hard,
        0.5);
    PreparedLinkMapper infeasible_mapper = infeasible_fixture.prepare();
    core::Solution infeasible_solution = make_solution();
    const LinkRouteResult infeasible = infeasible_mapper.route(
        first_virtual_link,
        {0U, 5U},
        infeasible_solution,
        safe_k_options());
    expect(!infeasible.routed && !infeasible.check.placeholder &&
               !infeasible.check.constraints.feasible,
           "all-infeasible candidate set returned wrong result");
    expect(require_route(infeasible_solution, first_virtual_link).empty(),
           "all-infeasible route committed resources");
    expect(infeasible_fixture.bw({0U, 2U}) == 8,
           "all-infeasible route mutated capacity");

    Fixture rank_fixture;
    PreparedLinkMapper rank_mapper = rank_fixture.prepare();
    core::Solution rank_solution = make_solution();
    std::size_t rank_calls = 0U;
    LinkPathRanker ranker = [&rank_calls](PhysicalPaths& paths)
    {
        ++rank_calls;
        std::stable_sort(
            paths.begin(),
            paths.end(),
            [](const auto& left, const auto& right)
            {
                return left.size() > right.size();
            });
    };
    LinkRouteOptions rank_options = safe_k_options();
    rank_options.ranker = &ranker;
    const LinkRouteResult ranked = rank_mapper.route(
        first_virtual_link,
        {0U, 5U},
        rank_solution,
        rank_options);
    expect(ranked.routed && rank_calls == 1U,
           "safe ranker call count mismatch");
    expect(require_route(rank_solution, first_virtual_link) == third_path(),
           "safe ranker did not control first feasible path");
}

void test_reroute_leak_and_partial_commit()
{
    Fixture reroute_fixture;
    PreparedLinkMapper reroute_mapper = reroute_fixture.prepare();
    core::Solution reroute_solution = make_solution();
    const LinkRouteOptions options = safe_k_options();
    expect(reroute_mapper.route(
               first_virtual_link,
               {0U, 5U},
               reroute_solution,
               options).routed,
           "initial reroute fixture failed");
    expect(require_route(reroute_solution, first_virtual_link) ==
               second_path(),
           "initial reroute path mismatch");

    expect(reroute_mapper.route(
               first_virtual_link,
               {0U, 5U},
               reroute_solution,
               options).routed,
           "second reroute failed");
    expect(require_route(reroute_solution, first_virtual_link) ==
               third_path(),
           "reroute did not select next feasible path");
    expect(reroute_fixture.bw({0U, 2U}) == 3 &&
               reroute_fixture.bw({2U, 5U}) == 7,
           "reroute unexpectedly refunded old resources");
    const core::LinkPathInfoKey old_info{
        {0, 1}, {0, 2}};
    expect(!reroute_solution.link_paths_info.contains(old_info),
           "reroute retained old path info");
    expect(reroute_solution.link_paths_info.size() == 3U,
           "reroute new path info cardinality mismatch");

    Fixture partial_fixture;
    PreparedLinkMapper partial_mapper = partial_fixture.prepare();
    core::Solution partial_solution = make_solution();
    LinkPathRanker sabotage = [&partial_fixture](PhysicalPaths&)
    {
        partial_fixture.set_physical(
            {0U, 2U}, partial_fixture.p_aux, std::int64_t{1});
    };
    LinkRouteOptions partial_options = safe_k_options();
    partial_options.ranker = &sabotage;
    expect_resource_error(
        [&]
        {
            static_cast<void>(partial_mapper.route(
                first_virtual_link,
                {0U, 5U},
                partial_solution,
                partial_options));
        },
        ResourceUpdatorErrorCode::insufficient_resource,
        ResourceUpdatorOperation::update_link);
    expect(require_route(partial_solution, first_virtual_link) ==
               second_path(),
           "partial commit lost the full prewritten path");
    expect(partial_fixture.bw({0U, 2U}) == 3,
           "partial commit lost earlier resource mutation");
    expect(partial_fixture.aux({0U, 2U}) == 1,
           "failing resource was mutated");
    expect(partial_fixture.bw({2U, 5U}) == 12 &&
               partial_fixture.aux({2U, 5U}) == 10,
           "partial commit mutated a later physical link");
    expect(partial_solution.link_paths_info.empty(),
           "partial commit wrote info before edge completion");
}

void test_unsafe_routing_modes()
{
    Fixture feasible_fixture;
    PreparedLinkMapper feasible_mapper = feasible_fixture.prepare();
    core::Solution feasible_solution = make_solution();
    std::size_t ignored_rank_calls = 0U;
    LinkPathRanker ignored_ranker = [&ignored_rank_calls](PhysicalPaths&)
    {
        ++ignored_rank_calls;
    };
    LinkRouteOptions unsafe;
    unsafe.shortest_method = ShortestPathMethod::k_shortest;
    unsafe.k = 10;
    unsafe.allow_constraint_violation = true;
    unsafe.record_constraint_violation = false;
    unsafe.ranker = &ignored_ranker;
    const LinkRouteResult feasible = feasible_mapper.route(
        first_virtual_link, {0U, 5U}, feasible_solution, unsafe);
    expect(feasible.routed && feasible.check.constraints.feasible,
           "unsafe mode missed a later feasible path");
    expect(require_route(feasible_solution, first_virtual_link) ==
               second_path(),
           "unsafe first-feasible order mismatch");
    expect(ignored_rank_calls == 0U,
           "unsafe mode unexpectedly called the ranker");

    Fixture least_fixture;
    least_fixture.set_virtual(
        first_virtual_link, least_fixture.v_bw, std::int64_t{15});
    least_fixture.set_virtual(
        first_virtual_link, least_fixture.v_soft, 0.0);
    least_fixture.set_virtual(
        first_virtual_link, least_fixture.v_latency_hard, 1.0);
    least_fixture.set_virtual(
        first_virtual_link, least_fixture.v_latency_soft, 100.0);
    PreparedLinkMapper least_mapper = least_fixture.prepare();
    core::Solution least_solution = make_solution();
    const LinkRouteResult least = least_mapper.route(
        first_virtual_link, {0U, 5U}, least_solution, unsafe);
    expect(least.routed && !least.check.constraints.feasible,
           "unsafe least-violation route result mismatch");
    expect(require_route(least_solution, first_virtual_link) == third_path(),
           "unsafe least-violation path mismatch");
    expect(least_fixture.bw({0U, 3U}) == -3 &&
               least_fixture.bw({3U, 4U}) == -3 &&
               least_fixture.bw({4U, 5U}) == -3,
           "unsafe commit did not permit negative capacity");

    Fixture tie_fixture;
    tie_fixture.set_virtual(
        first_virtual_link, tie_fixture.v_soft, 0.0);
    tie_fixture.set_virtual(
        first_virtual_link, tie_fixture.v_latency_hard, 100.0);
    tie_fixture.set_virtual(
        first_virtual_link, tie_fixture.v_latency_soft, 100.0);
    for (const ConstraintLink link :
         std::array<ConstraintLink, 4U>{
             {{0U, 1U}, {1U, 5U}, {0U, 2U}, {2U, 5U}}})
    {
        tie_fixture.set_physical(link, tie_fixture.p_bw, std::int64_t{4});
    }
    PreparedLinkMapper tie_mapper = tie_fixture.prepare();
    core::Solution tie_solution = make_solution();
    LinkRouteOptions tie_options = unsafe;
    tie_options.shortest_method = ShortestPathMethod::all_shortest;
    const LinkRouteResult tie = tie_mapper.route(
        first_virtual_link, {0U, 5U}, tie_solution, tie_options);
    expect(tie.routed && !tie.check.constraints.feasible,
           "unsafe tie fixture did not remain infeasible");
    expect(require_route(tie_solution, first_virtual_link) == first_path(),
           "unsafe minimum tie did not preserve first path");

    Fixture unsupported_fixture;
    PreparedLinkMapper unsupported_mapper = unsupported_fixture.prepare();
    core::Solution unsupported_solution = make_solution();
    unsupported_solution.link_paths.insert_or_assign(
        {99, 100}, {{1, 2}});
    LinkRouteOptions unsupported_options;
    unsupported_options.allow_constraint_violation = true;
    unsupported_options.record_constraint_violation = false;
    expect_mapper_error(
        [&]
        {
            static_cast<void>(unsupported_mapper.route(
                first_virtual_link,
                {0U, 5U},
                unsupported_solution,
                unsupported_options));
        },
        LinkMapperErrorCode::unsupported_unsafe_shortest_method,
        LinkMapperOperation::route);
    expect(unsupported_solution.link_paths.contains({99, 100}),
           "unsupported unsafe method mutated paths");
}

LinkRouteCheckInfo pooling_check(const Fixture& fixture)
{
    LinkRouteCheckInfo check;
    check.constraints.feasible = false;

    PhysicalLinkConstraintResult first;
    first.physical_link = {0U, 2U};
    first.offsets.set(fixture.v_bw.registry_id, std::int64_t{-2});
    first.offsets.set(fixture.v_soft.registry_id, 2.5);
    check.constraints.link_level.push_back(std::move(first));

    PhysicalLinkConstraintResult second;
    second.physical_link = {2U, 5U};
    second.offsets.set(fixture.v_bw.registry_id, std::int64_t{3});
    second.offsets.set(fixture.v_soft.registry_id, -1.5);
    check.constraints.link_level.push_back(std::move(second));

    check.constraints.path_level.set(
        fixture.v_latency_hard.registry_id, 4.5);
    check.constraints.path_level.set(
        fixture.v_latency_soft.registry_id, -2.0);
    return check;
}

void test_record_pooling_repeated_and_empty_hard()
{
    Fixture fixture;
    PreparedLinkMapper mapper = fixture.prepare();
    core::Solution solution = make_solution();
    const LinkRouteCheckInfo check = pooling_check(fixture);
    mapper.record_route_constraint_violation(
        first_virtual_link, check, solution);

    const auto& link_offsets = require_link_values(
        solution.v_net_constraint_offsets.link_level,
        first_virtual_link,
        "missing pooled link offsets");
    const auto& link_violations = require_link_values(
        solution.v_net_constraint_violations.link_level,
        first_virtual_link,
        "missing pooled link violations");
    const auto& path_offsets = require_link_values(
        solution.v_net_constraint_offsets.path_level,
        first_virtual_link,
        "missing pooled path offsets");
    const auto& path_violations = require_link_values(
        solution.v_net_constraint_violations.path_level,
        first_virtual_link,
        "missing pooled path violations");
    expect_integer(link_offsets, fixture.v_bw.registry_id, 3,
                   "integer max-pooled link offset mismatch");
    expect_integer(link_violations, fixture.v_bw.registry_id, 3,
                   "integer sum-positive link violation mismatch");
    expect_double(link_offsets, fixture.v_soft.registry_id, 2.5,
                  "double max-pooled link offset mismatch");
    expect_double(link_violations, fixture.v_soft.registry_id, 2.5,
                  "double sum-positive link violation mismatch");
    expect_double(path_offsets, fixture.v_latency_hard.registry_id, 4.5,
                  "path offset mismatch");
    expect_double(path_violations,
                  fixture.v_latency_hard.registry_id,
                  4.5,
                  "positive path violation mismatch");
    expect_double(path_offsets, fixture.v_latency_soft.registry_id, -2.0,
                  "negative path offset mismatch");
    expect_integer(path_violations,
                   fixture.v_latency_soft.registry_id,
                   0,
                   "negative path violation type mismatch");
    expect(solution.v_net_total_hard_constraint_violation == 4.5,
           "maximum hard violation mismatch");

    mapper.record_route_constraint_violation(
        first_virtual_link, check, solution);
    expect(solution.v_net_total_hard_constraint_violation == 9.0,
           "repeated recording did not double-count the hard total");
    expect(solution.v_net_constraint_offsets.link_level.size() == 1U &&
               solution.v_net_constraint_offsets.path_level.size() == 1U &&
               solution.v_net_constraint_violations.link_level.size() == 1U &&
               solution.v_net_constraint_violations.path_level.size() == 1U,
           "repeated recording appended instead of replacing tables");

    Fixture empty_fixture;
    LinkMapperSelection empty_selection = empty_fixture.default_selection();
    empty_selection.hard_constraints.clear();
    PreparedLinkMapper empty_mapper =
        empty_fixture.prepare(std::move(empty_selection));
    core::Solution empty_solution = make_solution();
    const LinkRouteCheckInfo empty_check = pooling_check(empty_fixture);
    expect_mapper_error(
        [&]
        {
            empty_mapper.record_route_constraint_violation(
                first_virtual_link, empty_check, empty_solution);
        },
        LinkMapperErrorCode::empty_hard_constraint_violations,
        LinkMapperOperation::record_violation);
    expect(empty_solution.v_net_constraint_offsets.link_level.size() == 1U &&
               empty_solution.v_net_constraint_offsets.path_level.size() == 1U &&
               empty_solution.v_net_constraint_violations.link_level.size() == 1U &&
               empty_solution.v_net_constraint_violations.path_level.size() == 1U,
           "empty-hard error did not preserve four prior table writes");
    expect(empty_solution.v_net_total_hard_constraint_violation == 0.0,
           "empty-hard error changed accumulated total");
}

void test_undo_success_missing_and_partial()
{
    Fixture success_fixture;
    PreparedLinkMapper success_mapper = success_fixture.prepare();
    core::Solution success_solution = make_solution();
    expect(success_mapper.route(
               first_virtual_link,
               {0U, 5U},
               success_solution,
               safe_k_options()).routed,
           "undo fixture route failed");
    expect(success_mapper.undo_route(first_virtual_link, success_solution),
           "undo returned false");
    expect(success_fixture.bw({0U, 2U}) == 8 &&
               success_fixture.bw({2U, 5U}) == 12 &&
               success_fixture.aux({0U, 2U}) == 10 &&
               success_fixture.aux({2U, 5U}) == 10,
           "undo did not restore resources exactly once");
    expect(success_solution.link_paths.empty() &&
               success_solution.link_paths_info.empty(),
           "undo did not erase path and info");
    expect_mapper_error(
        [&]
        {
            static_cast<void>(success_mapper.undo_route(
                first_virtual_link, success_solution));
        },
        LinkMapperErrorCode::route_not_found,
        LinkMapperOperation::undo_route);

    Fixture partial_fixture;
    PreparedLinkMapper partial_mapper = partial_fixture.prepare();
    core::Solution partial_solution = make_solution();
    expect(partial_mapper.route(
               first_virtual_link,
               {0U, 5U},
               partial_solution,
               safe_k_options()).routed,
           "partial undo fixture route failed");
    const core::LinkPathInfoKey missing_key{
        {0, 1}, {2, 5}};
    partial_solution.link_paths_info.erase(missing_key);
    const LinkMapperException partial_error = expect_mapper_error(
        [&]
        {
            static_cast<void>(partial_mapper.undo_route(
                first_virtual_link, partial_solution));
        },
        LinkMapperErrorCode::route_info_not_found,
        LinkMapperOperation::undo_route);
    expect(partial_error.physical_link() ==
               std::optional<ConstraintLink>{{2U, 5U}},
           "partial undo error context mismatch");
    expect(partial_fixture.bw({0U, 2U}) == 8 &&
               partial_fixture.aux({0U, 2U}) == 10,
           "partial undo lost earlier restoration");
    expect(partial_fixture.bw({2U, 5U}) == 7 &&
               partial_fixture.aux({2U, 5U}) == 8,
           "partial undo restored the failing edge");
    expect(require_route(partial_solution, first_virtual_link) ==
               second_path(),
           "partial undo erased the retained path");
    expect(!partial_solution.link_paths_info.contains(
               core::LinkPathInfoKey{{0, 1}, {0, 2}}),
           "partial undo retained earlier restored info");
}

void test_mapping_order_clone_failure_and_cardinality()
{
    Fixture order_fixture;
    PreparedLinkMapper order_mapper = order_fixture.prepare();
    core::Solution order_solution = make_solution_with_slots();
    order_solution.result = false;
    order_solution.route_result = false;
    expect(order_mapper.link_mapping(
               {second_virtual_link, first_virtual_link}, order_solution),
           "ordered whole-link mapping failed");
    expect(order_solution.link_paths.entries().size() == 2U &&
               order_solution.link_paths.entries()[0U].key ==
                   core::SolutionLink{1, 2} &&
               order_solution.link_paths.entries()[1U].key ==
                   core::SolutionLink{0, 1},
           "whole-link mapping did not retain supplied order");
    expect(!order_solution.result && !order_solution.route_result,
           "successful mapping changed pre-existing success flags");

    Fixture clone_fixture;
    PreparedLinkMapper clone_mapper = clone_fixture.prepare();
    core::Solution clone_solution = make_solution_with_slots();
    LinkMappingOptions clone_options;
    clone_options.inplace = false;
    expect(clone_mapper.link_mapping(clone_solution, clone_options),
           "inplace=false whole-link mapping failed");
    expect(clone_solution.link_paths.size() == 2U,
           "clone mapping did not populate solution paths");
    expect(clone_fixture.bw({0U, 2U}) == 8 &&
               clone_fixture.bw({2U, 5U}) == 12 &&
               clone_fixture.bw({5U, 6U}) == 12,
           "inplace=false mapping mutated original physical network");

    Fixture failure_fixture;
    failure_fixture.set_physical(
        {5U, 6U}, failure_fixture.p_bw, std::int64_t{1});
    PreparedLinkMapper failure_mapper = failure_fixture.prepare();
    core::Solution failure_solution = make_solution_with_slots();
    failure_solution.result = true;
    failure_solution.route_result = true;
    expect(!failure_mapper.link_mapping(failure_solution),
           "whole-link failure unexpectedly succeeded");
    expect(!failure_solution.result && !failure_solution.route_result,
           "whole-link failure flags mismatch");
    expect(require_route(failure_solution, first_virtual_link) ==
               second_path() &&
               require_route(failure_solution, second_virtual_link).empty(),
           "whole-link failure did not retain ordered partial paths");
    expect(failure_solution.link_paths_info.size() == 2U,
           "whole-link failure lost earlier route info");

    Fixture cardinality_fixture;
    PreparedLinkMapper cardinality_mapper = cardinality_fixture.prepare();
    core::Solution cardinality_solution = make_solution_with_slots();
    expect_mapper_error(
        [&]
        {
            static_cast<void>(cardinality_mapper.link_mapping(
                {first_virtual_link}, cardinality_solution));
        },
        LinkMapperErrorCode::mapping_cardinality_mismatch,
        LinkMapperOperation::link_mapping);
    expect(require_route(cardinality_solution, first_virtual_link) ==
               second_path(),
           "cardinality error discarded earlier route");

    Fixture missing_slot_fixture;
    PreparedLinkMapper missing_slot_mapper = missing_slot_fixture.prepare();
    core::Solution missing_slot_solution = make_solution();
    missing_slot_solution.node_slots.insert_or_assign(0, 0);
    expect_mapper_error(
        [&]
        {
            static_cast<void>(missing_slot_mapper.link_mapping(
                {first_virtual_link}, missing_slot_solution));
        },
        LinkMapperErrorCode::missing_node_slot,
        LinkMapperOperation::link_mapping);
    expect(missing_slot_solution.link_paths.empty(),
           "missing slot error occurred before canonical clear");

    Fixture unsafe_fixture;
    PreparedLinkMapper unsafe_mapper = unsafe_fixture.prepare();
    core::Solution unsafe_solution = make_solution_with_slots();
    unsafe_solution.link_paths.insert_or_assign({99, 100}, {{1, 2}});
    LinkMappingOptions unsafe_options;
    unsafe_options.allow_constraint_violation = true;
    expect_mapper_error(
        [&]
        {
            static_cast<void>(unsafe_mapper.link_mapping(
                unsafe_solution, unsafe_options));
        },
        LinkMapperErrorCode::unsupported_unsafe_link_mapping,
        LinkMapperOperation::link_mapping);
    expect(unsafe_solution.link_paths.contains({99, 100}),
           "unsupported whole-link mode cleared solution state");
}

struct RouteSnapshot
{
    core::LinkPaths paths;
    core::LinkPathsInfo info;
    core::LinkConstraintTable link_offsets;
    core::PathConstraintTable path_offsets;
    core::LinkViolationTable link_violations;
    core::PathViolationTable path_violations;
    std::array<std::int64_t, 8U> bandwidth{};
    std::array<std::int64_t, 8U> auxiliary{};
    double total_hard = 0.0;

    friend bool operator==(
        const RouteSnapshot& left,
        const RouteSnapshot& right)
    {
        return left.paths == right.paths &&
            left.info == right.info &&
            left.link_offsets == right.link_offsets &&
            left.path_offsets == right.path_offsets &&
            left.link_violations == right.link_violations &&
            left.path_violations == right.path_violations &&
            left.bandwidth == right.bandwidth &&
            left.auxiliary == right.auxiliary &&
            left.total_hard == right.total_hard;
    }
};

RouteSnapshot route_snapshot(
    ShortestPathMethod method,
    std::size_t candidate_workers,
    std::size_t topology_workers)
{
    Fixture fixture;
    PreparedLinkMapper mapper = fixture.prepare();
    core::Solution solution = make_solution();
    LinkRouteOptions options;
    options.shortest_method = method;
    options.k = 10;
    options.candidate_workers = candidate_workers;
    options.topology_constraint_workers = topology_workers;
    expect(mapper.route(
               first_virtual_link,
               {0U, 5U},
               solution,
               options).routed,
           "worker snapshot route failed");

    RouteSnapshot result;
    result.paths = solution.link_paths;
    result.info = solution.link_paths_info;
    result.link_offsets = solution.v_net_constraint_offsets.link_level;
    result.path_offsets = solution.v_net_constraint_offsets.path_level;
    result.link_violations =
        solution.v_net_constraint_violations.link_level;
    result.path_violations =
        solution.v_net_constraint_violations.path_level;
    result.total_hard = solution.v_net_total_hard_constraint_violation;
    for (std::size_t index = 0U; index < physical_links().size(); ++index)
    {
        result.bandwidth[index] = fixture.bw(physical_links()[index]);
        result.auxiliary[index] = fixture.aux(physical_links()[index]);
    }
    return result;
}

void test_workers_and_later_error_suppression()
{
    const RouteSnapshot candidate_reference = route_snapshot(
        ShortestPathMethod::all_shortest, 1U, 1U);
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(route_snapshot(
                   ShortestPathMethod::all_shortest,
                   workers,
                   1U) == candidate_reference,
               "candidate worker output/order drift");
    }

    const RouteSnapshot topology_reference = route_snapshot(
        ShortestPathMethod::available_shortest, 1U, 1U);
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(route_snapshot(
                   ShortestPathMethod::available_shortest,
                   1U,
                   workers) == topology_reference,
               "topology worker output/order drift");
    }

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        Fixture fixture;
        fixture.set_physical(
            {0U, 1U}, fixture.p_bw, std::int64_t{12});
        fixture.set_physical(
            {1U, 5U}, fixture.p_bw, std::int64_t{12});
        fixture.set_physical(
            {0U, 2U}, fixture.p_soft, std::string{"later-error"});
        PreparedLinkMapper mapper = fixture.prepare();
        core::Solution solution = make_solution();
        LinkRouteOptions options = safe_k_options();
        options.shortest_method = ShortestPathMethod::all_shortest;
        options.candidate_workers = workers;
        expect(mapper.route(
                   first_virtual_link,
                   {0U, 5U},
                   solution,
                   options).routed,
               "later candidate error was not suppressed");
        expect(require_route(solution, first_virtual_link) == first_path(),
               "later-error suppression changed first-feasible order");
    }

    Fixture window_fixture;
    window_fixture.set_physical(
        {0U, 3U}, window_fixture.p_soft, std::string{"later-error"});
    PreparedLinkMapper window_mapper = window_fixture.prepare();
    core::Solution window_solution = make_solution();
    LinkPathRanker expand_paths = [](PhysicalPaths& paths)
    {
        paths.assign(64U, std::vector<Vertex>{0U, 1U, 5U});
        paths[16U] = {0U, 2U, 5U};
        paths[17U] = {0U, 3U, 4U, 5U};
    };
    LinkRouteOptions window_options = safe_k_options();
    window_options.candidate_workers = 8U;
    window_options.ranker = &expand_paths;
    expect(
        window_mapper.route(
            first_virtual_link,
            {0U, 5U},
            window_solution,
            window_options).routed,
        "ordered path window routing failed");
    expect(
        require_route(window_solution, first_virtual_link) == second_path(),
        "ordered path window changed first-feasible/error order");
}

void test_concurrent_independent_callers()
{
    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.emplace_back(std::async(
            std::launch::async,
            [caller]
            {
                const std::array<std::size_t, 4U> widths{
                    0U, 1U, 2U, 8U};
                for (std::size_t iteration = 0U;
                     iteration < widths.size();
                     ++iteration)
                {
                    Fixture fixture;
                    PreparedLinkMapper mapper = fixture.prepare();
                    core::Solution solution = make_solution();
                    LinkRouteOptions options;
                    options.shortest_method =
                        ShortestPathMethod::all_shortest;
                    options.k = 10;
                    options.record_constraint_violation = false;
                    options.candidate_workers =
                        widths[(caller + iteration) % widths.size()];
                    options.topology_constraint_workers =
                        widths[(caller + iteration + 1U) % widths.size()];
                    if (!mapper.route(
                            first_virtual_link,
                            {0U, 5U},
                            solution,
                            options).routed ||
                        require_route(solution, first_virtual_link) !=
                            second_path() ||
                        fixture.bw({0U, 2U}) != 3 ||
                        fixture.aux({0U, 2U}) != 8)
                    {
                        return false;
                    }
                }
                return true;
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(), "concurrent independent caller drift");
    }
}

} // namespace

int main()
{
    try
    {
        const auto run = [](std::string_view name, auto&& test)
        {
            try
            {
                test();
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(
                    std::string(name) + ": " + error.what());
            }
        };

        run("selection/registry/resources/boundary",
            test_selection_registry_duplicate_resources_and_boundary);
        run("same/no/infeasible/ranker",
            test_same_node_no_path_all_infeasible_and_ranker);
        run("reroute/partial", test_reroute_leak_and_partial_commit);
        run("unsafe", test_unsafe_routing_modes);
        run("record/pooling", test_record_pooling_repeated_and_empty_hard);
        run("undo", test_undo_success_missing_and_partial);
        run("mapping", test_mapping_order_clone_failure_and_cardinality);
        run("workers/error suppression",
            test_workers_and_later_error_suppression);
        run("concurrent callers", test_concurrent_independent_callers);
        std::cout << "link mapper unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "link mapper unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
