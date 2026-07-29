#include "topology_analyzer.h"

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
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using controller::ConstraintId;
using controller::ConstraintLink;
using controller::PreparedTopologyAnalyzer;
using controller::ShortestPathMethod;
using controller::TopologyAnalyzer;
using controller::TopologyAnalyzerErrorCode;
using controller::TopologyAnalyzerException;
using controller::TopologyAnalyzerOperation;
using controller::TopologyAnalyzerSelection;
using controller::TopologyPathRequest;

using Path = std::vector<Vertex>;
using Paths = PreparedTopologyAnalyzer::Paths;

constexpr ConstraintLink virtual_link{0U, 1U};

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
TopologyAnalyzerException expect_analyzer_error(
    Callable&& callable,
    TopologyAnalyzerErrorCode code,
    TopologyAnalyzerOperation operation,
    std::optional<std::size_t> request_index = std::nullopt,
    std::optional<std::size_t> item_index = std::nullopt,
    std::optional<ConstraintId> resource_id = std::nullopt)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const TopologyAnalyzerException& error)
    {
        expect(error.code() == code, "topology analyzer error code mismatch");
        expect(
            error.operation() == operation,
            "topology analyzer operation mismatch");
        expect(
            error.request_index() == request_index,
            "topology analyzer request index mismatch");
        expect(
            error.item_index() == item_index,
            "topology analyzer item index mismatch");
        expect(
            error.resource_id() == resource_id,
            "topology analyzer resource ID mismatch");
        return error;
    }
    throw std::runtime_error("expected TopologyAnalyzerException");
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
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    construction.config.link_attribute_specs = {
        resource_spec("capacity_hard", ConstraintRestriction::hard),
        resource_spec("capacity_soft", ConstraintRestriction::soft),
        status_spec("link_status")};
    return network::VirtualNetwork(std::move(construction));
}

network::PhysicalNetwork make_physical_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        7U,
        std::vector<EdgeEndpoints>{
            {0U, 1U},
            {1U, 5U},
            {0U, 2U},
            {2U, 5U},
            {0U, 3U},
            {3U, 4U},
            {4U, 5U}});

    // Both resource IDs deliberately differ from the virtual registry.
    construction.config.link_attribute_specs = {
        resource_spec("capacity_soft", ConstraintRestriction::soft),
        status_spec("link_status"),
        resource_spec("capacity_hard", ConstraintRestriction::hard)};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
ConstraintId require_link_id(Network& value, std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing fixture link binding");
    return binding->registry_id;
}

const std::vector<ConstraintLink>& physical_links()
{
    static const std::vector<ConstraintLink> links{
        {0U, 1U},
        {1U, 5U},
        {0U, 2U},
        {2U, 5U},
        {0U, 3U},
        {3U, 4U},
        {4U, 5U}};
    return links;
}

struct Fixture
{
    network::VirtualNetwork virtual_network = make_virtual_network();
    network::PhysicalNetwork physical_network = make_physical_network();

    ConstraintId virtual_hard = 0U;
    ConstraintId virtual_soft = 0U;
    ConstraintId physical_hard = 0U;
    ConstraintId physical_soft = 0U;

    Fixture()
    {
        virtual_hard = require_link_id(virtual_network, "capacity_hard");
        virtual_soft = require_link_id(virtual_network, "capacity_soft");
        physical_hard = require_link_id(physical_network, "capacity_hard");
        physical_soft = require_link_id(physical_network, "capacity_soft");

        expect(
            virtual_hard != physical_hard &&
                virtual_soft != physical_soft,
            "fixture must exercise independent resource registry IDs");

        virtual_network.set_link_attrs_data({
            sparse_link_update(
                virtual_hard,
                {{0U, 1U, std::int64_t{5}}}),
            sparse_link_update(
                virtual_soft,
                {{0U, 1U, 100.0}})});

        physical_network.set_link_attrs_data({
            sparse_link_update(
                physical_hard,
                {{0U, 1U, std::int64_t{4}},
                 {1U, 5U, std::int64_t{12}},
                 {0U, 2U, std::int64_t{8}},
                 {2U, 5U, std::int64_t{12}},
                 {0U, 3U, std::int64_t{12}},
                 {3U, 4U, std::int64_t{12}},
                 {4U, 5U, std::int64_t{12}}}),
            sparse_link_update(
                physical_soft,
                {{0U, 1U, 1.0},
                 {1U, 5U, 1.0},
                 {0U, 2U, 1.0},
                 {2U, 5U, 1.0},
                 {0U, 3U, 1.0},
                 {3U, 4U, 1.0},
                 {4U, 5U, 1.0}})});
    }

    PreparedTopologyAnalyzer prepare(
        std::vector<ConstraintId> constraints = {},
        std::vector<ConstraintId> resources = {}) const
    {
        TopologyAnalyzerSelection selection;
        selection.constraints.link_at_link = std::move(constraints);
        selection.link_resources = std::move(resources);
        return TopologyAnalyzer(std::move(selection)).prepare(
            virtual_network, physical_network);
    }

    void set_virtual_hard(std::int64_t value)
    {
        virtual_network.set_link_attrs_data({
            sparse_link_update(
                virtual_hard, {{0U, 1U, value}})});
    }

    void set_physical_hard(
        ConstraintLink link,
        std::int64_t value)
    {
        physical_network.set_link_attrs_data({
            sparse_link_update(
                physical_hard,
                {{link.source, link.target, value}})});
    }
};

TopologyPathRequest make_request(
    ShortestPathMethod method,
    Vertex source = 0U,
    Vertex target = 5U,
    std::int64_t k = 10,
    double max_path_nodes = 1.0e6,
    std::size_t constraint_workers = 1U)
{
    TopologyPathRequest result;
    result.virtual_link = virtual_link;
    result.physical_pair = {source, target};
    result.options.method = method;
    result.options.k = k;
    result.options.max_path_nodes = max_path_nodes;
    result.options.constraint_workers = constraint_workers;
    return result;
}

const Path& first_path()
{
    static const Path value{0U, 1U, 5U};
    return value;
}

const Path& second_path()
{
    static const Path value{0U, 2U, 5U};
    return value;
}

const Path& third_path()
{
    static const Path value{0U, 3U, 4U, 5U};
    return value;
}

Paths all_simple_paths()
{
    return {first_path(), second_path(), third_path()};
}

Paths tied_shortest_paths()
{
    return {first_path(), second_path()};
}

bool mask_equal(const SearchMask& left, const SearchMask& right)
{
    return left.node_flags() == right.node_flags() &&
        left.edge_flags() == right.edge_flags();
}

bool mask_allows(
    const SearchMask& mask,
    const Graph& graph,
    ConstraintLink link)
{
    const auto edge = graph.edge(link.source, link.target);
    return mask.allows_edge(graph.edge_id(edge));
}

void expect_all_edges(
    const SearchMask& mask,
    const Graph& graph,
    bool expected,
    std::string_view message)
{
    for (const ConstraintLink link : physical_links())
    {
        expect(mask_allows(mask, graph, link) == expected, message);
    }
}

void test_selection_and_six_modes()
{
    Fixture fixture;
    TopologyAnalyzerSelection selection;
    selection.constraints.link_at_link = {
        fixture.virtual_hard, fixture.virtual_soft};
    selection.link_resources = {fixture.virtual_hard};
    const TopologyAnalyzer analyzer(selection);
    expect(
        analyzer.selection().constraints.link_at_link ==
                selection.constraints.link_at_link &&
            analyzer.selection().link_resources == selection.link_resources,
        "analyzer did not retain typed selection fields");

    const PreparedTopologyAnalyzer prepared = fixture.prepare();

    struct ModeCase
    {
        ShortestPathMethod method;
        std::int64_t k;
        Paths expected;
    };
    const std::vector<ModeCase> cases{
        {ShortestPathMethod::first_shortest, 10, {first_path()}},
        {ShortestPathMethod::k_shortest, 3, all_simple_paths()},
        {ShortestPathMethod::k_shortest_length, 3,
         tied_shortest_paths()},
        {ShortestPathMethod::all_shortest, 10,
         tied_shortest_paths()},
        {ShortestPathMethod::bfs_shortest, 10, {first_path()}},
        {ShortestPathMethod::available_shortest, 10,
         {first_path()}}};

    for (const ModeCase& item : cases)
    {
        const Paths actual = prepared.find_shortest_paths(
            make_request(item.method, 0U, 5U, item.k));
        expect(actual == item.expected, "shortest-path mode/tie order drift");
    }
}

void test_k_and_max_path_semantics()
{
    Fixture fixture;
    const PreparedTopologyAnalyzer prepared = fixture.prepare();

    for (const ShortestPathMethod method : {
             ShortestPathMethod::k_shortest,
             ShortestPathMethod::k_shortest_length})
    {
        expect(
            prepared.find_shortest_paths(
                make_request(method, 0U, 5U, 0)).empty(),
            "k=0 must return no paths");
        expect(
            prepared.find_shortest_paths(
                make_request(method, 0U, 5U, -1)).empty(),
            "negative k must return no paths");
    }

    expect(
        prepared.find_shortest_paths(
            make_request(
                ShortestPathMethod::k_shortest_length,
                0U, 5U, 3)) == tied_shortest_paths(),
        "k_shortest_length must count path nodes");
    expect(
        prepared.find_shortest_paths(
            make_request(
                ShortestPathMethod::k_shortest_length,
                0U, 5U, 4)) == all_simple_paths(),
        "k_shortest_length inclusive cutoff mismatch");

    expect(
        prepared.find_shortest_paths(
            make_request(
                ShortestPathMethod::k_shortest,
                0U, 5U, 3, 3.0)) == all_simple_paths(),
        "max_path_nodes must inspect only the first path");
    expect(
        prepared.find_shortest_paths(
            make_request(
                ShortestPathMethod::k_shortest,
                0U, 5U, 3, 2.5)).empty(),
        "max_path_nodes must compare path node count");
}

void test_endpoint_no_path_and_source_target()
{
    Fixture fixture;
    const PreparedTopologyAnalyzer prepared = fixture.prepare();
    const std::array<ShortestPathMethod, 6U> methods{
        ShortestPathMethod::first_shortest,
        ShortestPathMethod::k_shortest,
        ShortestPathMethod::k_shortest_length,
        ShortestPathMethod::all_shortest,
        ShortestPathMethod::bfs_shortest,
        ShortestPathMethod::available_shortest};

    for (const ShortestPathMethod method : methods)
    {
        expect(
            prepared.find_shortest_paths(
                make_request(method, 0U, 99U, 10)).empty(),
            "invalid endpoint must be swallowed as no path");
        expect(
            prepared.find_shortest_paths(
                make_request(method, 0U, 6U, 10)).empty(),
            "disconnected endpoint must return no path");
        expect(
            prepared.find_shortest_paths(
                make_request(method, 0U, 0U, 10)) == Paths{{0U}},
            "source-equals-target path mismatch");
    }

    expect(
        !prepared.find_bfs_shortest_path(
             virtual_link, 0U, 6U).has_value(),
        "direct BFS no-path must return nullopt");
    const auto identity = prepared.find_bfs_shortest_path(
        virtual_link, 0U, 0U);
    expect(
        identity.has_value() && *identity == Path{0U},
        "direct BFS source-equals-target mismatch");
}

void test_hard_soft_available_and_bfs()
{
    Fixture fixture;
    const PreparedTopologyAnalyzer hard_soft = fixture.prepare(
        {fixture.virtual_hard, fixture.virtual_soft});
    const PreparedTopologyAnalyzer soft_only = fixture.prepare(
        {fixture.virtual_soft});

    expect(
        hard_soft.find_shortest_paths(make_request(
            ShortestPathMethod::available_shortest)) ==
                Paths{second_path()},
        "hard available constraint did not filter first path");
    expect(
        hard_soft.find_shortest_paths(make_request(
            ShortestPathMethod::bfs_shortest)) ==
                Paths{second_path()},
        "hard BFS constraint or neighbor order drift");
    const auto hard_bfs = hard_soft.find_bfs_shortest_path(
        virtual_link, 0U, 5U);
    expect(
        hard_bfs.has_value() && *hard_bfs == second_path(),
        "direct hard-constrained BFS path mismatch");

    expect(
        soft_only.find_shortest_paths(make_request(
            ShortestPathMethod::available_shortest)) ==
                Paths{first_path()},
        "soft violation incorrectly filtered available edge");
    expect(
        soft_only.find_shortest_paths(make_request(
            ShortestPathMethod::bfs_shortest)) ==
                Paths{first_path()},
        "soft violation incorrectly filtered BFS edge");
}

void test_pruned_resource_semantics()
{
    Fixture fixture;
    const Graph& graph = fixture.physical_network.graph();

    const PreparedTopologyAnalyzer single = fixture.prepare(
        {}, {fixture.virtual_hard});
    const SearchMask single_mask = single.create_pruned_mask(
        virtual_link, 2.0, 3.0, 1U);
    expect(
        !mask_allows(single_mask, graph, {0U, 1U}),
        "prune ratio/div accepted capacity below seven");
    expect(
        mask_allows(single_mask, graph, {0U, 2U}),
        "prune must apply ratio then div exactly once");

    const PreparedTopologyAnalyzer duplicate = fixture.prepare(
        {}, {fixture.virtual_hard, fixture.virtual_hard});
    const SearchMask duplicate_mask = duplicate.create_pruned_mask(
        virtual_link, 2.0, 3.0, 1U);
    expect(
        !mask_allows(duplicate_mask, graph, {0U, 2U}) &&
            mask_allows(duplicate_mask, graph, {0U, 3U}),
        "duplicate resource ID must repeat virtual adjustment");

    const SearchMask div_mask = single.create_pruned_mask(
        virtual_link, 1.0, 2.0, 1U);
    expect(
        mask_allows(div_mask, graph, {0U, 1U}),
        "prune div must be applied after multiplication");

    const PreparedTopologyAnalyzer soft = fixture.prepare(
        {}, {fixture.virtual_soft});
    const SearchMask soft_mask = soft.create_pruned_mask(
        virtual_link, 100.0, 0.0, 1U);
    expect_all_edges(
        soft_mask, graph, true,
        "soft pruned resource must not remove an edge");

    const PreparedTopologyAnalyzer empty = fixture.prepare();
    const SearchMask empty_mask = empty.create_pruned_mask(
        virtual_link, 99.0, 77.0, 1U);
    expect_all_edges(
        empty_mask, graph, true,
        "empty pruned resource selection must retain all edges");
}

void test_live_and_snapshot_views()
{
    Fixture fixture;
    const Graph& graph = fixture.physical_network.graph();
    const PreparedTopologyAnalyzer available = fixture.prepare(
        {fixture.virtual_hard, fixture.virtual_soft});
    const auto available_view = available.create_available_network(
        virtual_link);

    expect(
        !mask_allows(available_view.mask(), graph, {0U, 1U}),
        "available view initial hard filter mismatch");
    fixture.set_virtual_hard(3);
    expect(
        mask_allows(available_view.mask(), graph, {0U, 1U}),
        "available view did not observe live virtual value");
    fixture.set_physical_hard({0U, 1U}, 2);
    expect(
        !mask_allows(available_view.mask(), graph, {0U, 1U}),
        "available view did not observe live physical value");

    Fixture pruned_fixture;
    const Graph& pruned_graph = pruned_fixture.physical_network.graph();
    const PreparedTopologyAnalyzer pruned = pruned_fixture.prepare(
        {}, {pruned_fixture.virtual_hard});
    const auto snapshot_view = pruned.create_pruned_network(
        virtual_link, 1.0, 0.0);

    pruned_fixture.set_virtual_hard(20);
    expect(
        mask_allows(snapshot_view.mask(), pruned_graph, {0U, 2U}),
        "pruned view did not snapshot virtual requirements");
    const auto fresh_view = pruned.create_pruned_network(
        virtual_link, 1.0, 0.0);
    expect(
        !mask_allows(fresh_view.mask(), pruned_graph, {0U, 2U}),
        "fresh pruned view ignored current virtual requirements");
    pruned_fixture.set_physical_hard({0U, 2U}, 4);
    expect(
        !mask_allows(snapshot_view.mask(), pruned_graph, {0U, 2U}),
        "pruned view did not observe live physical values");
}

void test_mask_workers()
{
    Fixture fixture;
    const Graph& graph = fixture.physical_network.graph();
    const PreparedTopologyAnalyzer prepared = fixture.prepare(
        {fixture.virtual_hard, fixture.virtual_soft},
        {fixture.virtual_hard, fixture.virtual_hard});
    const SearchMask available_reference = prepared.create_available_mask(
        virtual_link, 1U);
    const SearchMask pruned_reference = prepared.create_pruned_mask(
        virtual_link, 2.0, 3.0, 1U);

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(
            mask_equal(
                prepared.create_available_mask(virtual_link, workers),
                available_reference),
            "available mask worker output drift");
        expect(
            mask_equal(
                prepared.create_pruned_mask(
                    virtual_link, 2.0, 3.0, workers),
                pruned_reference),
            "pruned mask worker output drift");
    }

    expect(
        !mask_allows(available_reference, graph, {0U, 1U}) &&
            mask_allows(available_reference, graph, {0U, 2U}),
        "available mask expected flags mismatch");
    expect(
        !mask_allows(pruned_reference, graph, {0U, 2U}) &&
            mask_allows(pruned_reference, graph, {0U, 3U}),
        "pruned mask expected flags mismatch");
}

void test_batch_workers_order_and_invalid_enum()
{
    Fixture fixture;
    const PreparedTopologyAnalyzer prepared = fixture.prepare(
        {fixture.virtual_hard, fixture.virtual_soft});

    std::vector<TopologyPathRequest> requests{
        make_request(ShortestPathMethod::first_shortest),
        make_request(
            ShortestPathMethod::available_shortest,
            0U, 5U, 10, 1.0e6, 8U),
        make_request(ShortestPathMethod::bfs_shortest),
        make_request(ShortestPathMethod::k_shortest, 0U, 5U, 3),
        make_request(ShortestPathMethod::first_shortest, 0U, 6U)};

    std::vector<Paths> expected;
    expected.reserve(requests.size());
    for (const TopologyPathRequest& request : requests)
    {
        expected.push_back(prepared.find_shortest_paths(request));
    }
    expect(
        expected[0] == Paths{first_path()} &&
            expected[1] == Paths{second_path()} &&
            expected[2] == Paths{second_path()} &&
            expected[3] == all_simple_paths() && expected[4].empty(),
        "batch scalar fixture expectations drifted");

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(
            prepared.find_shortest_paths_batch(requests, workers) == expected,
            "batch result/order changed with workers");
    }

    std::vector<TopologyPathRequest> invalid{
        requests[0], requests[0], requests[1], requests[0]};
    invalid[1].options.method = static_cast<ShortestPathMethod>(255U);
    invalid[3].options.method = static_cast<ShortestPathMethod>(254U);

    expect_analyzer_error(
        [&]
        {
            static_cast<void>(prepared.find_shortest_paths(invalid[1]));
        },
        TopologyAnalyzerErrorCode::invalid_method,
        TopologyAnalyzerOperation::find_paths);

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect_analyzer_error(
            [&]
            {
                static_cast<void>(
                    prepared.find_shortest_paths_batch(invalid, workers));
            },
            TopologyAnalyzerErrorCode::invalid_method,
            TopologyAnalyzerOperation::find_paths_batch,
            1U);
    }
}

void test_concurrent_callers()
{
    Fixture fixture;
    const PreparedTopologyAnalyzer prepared = fixture.prepare(
        {fixture.virtual_hard, fixture.virtual_soft},
        {fixture.virtual_hard});
    const SearchMask expected_mask = prepared.create_available_mask(
        virtual_link, 1U);
    const TopologyPathRequest available = make_request(
        ShortestPathMethod::available_shortest);
    const TopologyPathRequest simple = make_request(
        ShortestPathMethod::k_shortest, 0U, 5U, 3);

    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.emplace_back(std::async(
            std::launch::async,
            [&, caller]
            {
                const std::array<std::size_t, 4U> widths{0U, 1U, 2U, 8U};
                const std::size_t workers = widths[caller % widths.size()];
                for (std::size_t iteration = 0U; iteration < 32U; ++iteration)
                {
                    if (prepared.find_shortest_paths(available) !=
                            Paths{second_path()} ||
                        prepared.find_shortest_paths(simple) !=
                            all_simple_paths() ||
                        !mask_equal(
                            prepared.create_available_mask(
                                virtual_link, workers),
                            expected_mask))
                    {
                        return false;
                    }
                }
                return true;
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(), "concurrent prepared analyzer output drift");
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

        run("selection/six modes", test_selection_and_six_modes);
        run("k/max semantics", test_k_and_max_path_semantics);
        run("endpoint/no-path/source-target",
            test_endpoint_no_path_and_source_target);
        run("hard-soft available/BFS", test_hard_soft_available_and_bfs);
        run("pruned resources", test_pruned_resource_semantics);
        run("live/snapshot views", test_live_and_snapshot_views);
        run("mask workers", test_mask_workers);
        run("batch/error order", test_batch_workers_order_and_invalid_enum);
        run("concurrent callers", test_concurrent_callers);

        std::cout << "topology analyzer unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "topology analyzer unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
