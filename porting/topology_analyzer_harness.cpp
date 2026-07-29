#include "topology_analyzer.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
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
using controller::TopologyAnalyzerSelection;
using controller::TopologyPathRequest;

using Path = std::vector<Vertex>;
using Paths = PreparedTopologyAnalyzer::Paths;

constexpr ConstraintLink virtual_link{0U, 1U};

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
        resource_spec("capacity_soft", ConstraintRestriction::soft)};
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

    // Reverse the two fixed registry fields to exercise one-time binding.
    construction.config.link_attribute_specs = {
        resource_spec("capacity_soft", ConstraintRestriction::soft),
        resource_spec("capacity_hard", ConstraintRestriction::hard)};
    return network::PhysicalNetwork(std::move(construction));
}

template <typename Network>
ConstraintId require_link_id(Network& value, std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    if (!binding)
    {
        throw std::runtime_error("missing topology harness link binding");
    }
    return binding->registry_id;
}

const std::vector<ConstraintLink>& physical_links()
{
    static const std::vector<ConstraintLink> value{
        {0U, 1U},
        {1U, 5U},
        {0U, 2U},
        {2U, 5U},
        {0U, 3U},
        {3U, 4U},
        {4U, 5U}};
    return value;
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
        virtual_hard = require_link_id(
            virtual_network, "capacity_hard");
        virtual_soft = require_link_id(
            virtual_network, "capacity_soft");
        physical_hard = require_link_id(
            physical_network, "capacity_hard");
        physical_soft = require_link_id(
            physical_network, "capacity_soft");

        if (virtual_hard == physical_hard ||
            virtual_soft == physical_soft)
        {
            throw std::runtime_error(
                "topology fixture registry order is not independent");
        }

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
};

TopologyPathRequest request(
    ShortestPathMethod method,
    std::int64_t k = 10,
    double max_path_nodes = 1.0e6,
    Vertex source = 0U,
    Vertex target = 5U)
{
    TopologyPathRequest result;
    result.virtual_link = virtual_link;
    result.physical_pair = {source, target};
    result.options.method = method;
    result.options.k = k;
    result.options.max_path_nodes = max_path_nodes;
    return result;
}

std::string paths_payload(const Paths& paths)
{
    std::string result = "[";
    for (std::size_t path_index = 0U;
         path_index < paths.size();
         ++path_index)
    {
        if (path_index != 0U)
        {
            result.push_back(',');
        }
        result.push_back('[');
        const Path& path = paths[path_index];
        for (std::size_t vertex_index = 0U;
             vertex_index < path.size();
             ++vertex_index)
        {
            if (vertex_index != 0U)
            {
                result.push_back(',');
            }
            result += std::to_string(path[vertex_index]);
        }
        result.push_back(']');
    }
    result.push_back(']');
    return result;
}

std::string mask_payload(
    const SearchMask& mask,
    const Graph& graph)
{
    std::string result;
    result.reserve(physical_links().size());
    for (const ConstraintLink link : physical_links())
    {
        const auto edge = graph.edge(link.source, link.target);
        result.push_back(
            mask.allows_edge(graph.edge_id(edge)) ? '1' : '0');
    }
    return result;
}

void emit(std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

void differential()
{
    Fixture fixture;
    const PreparedTopologyAnalyzer unconstrained = fixture.prepare();

    emit(
        "mode_first_shortest",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::first_shortest))));
    emit(
        "mode_k_shortest",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest, 3))));
    emit(
        "mode_k_shortest_length",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest_length, 3))));
    emit(
        "mode_all_shortest",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::all_shortest))));
    emit(
        "mode_bfs_shortest",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::bfs_shortest))));
    emit(
        "mode_available_shortest",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::available_shortest))));

    network::BaseNetworkConstruction tie_virtual_construction;
    tie_virtual_construction.incoming_graph.emplace(
        2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    network::VirtualNetwork tie_virtual(
        std::move(tie_virtual_construction));
    network::BaseNetworkConstruction tie_physical_construction;
    tie_physical_construction.incoming_graph.emplace(
        4U,
        std::vector<EdgeEndpoints>{
            {0U, 1U}, {2U, 3U}, {0U, 2U}, {0U, 3U}, {1U, 2U}});
    network::PhysicalNetwork tie_physical(
        std::move(tie_physical_construction));
    const PreparedTopologyAnalyzer tie_analyzer =
        TopologyAnalyzer({{}, {}}).prepare(tie_virtual, tie_physical);
    emit(
        "first_dijkstra_fifo_tie",
        paths_payload(tie_analyzer.find_shortest_paths(
            request(
                ShortestPathMethod::first_shortest,
                10,
                1.0e6,
                1U,
                3U))));
    emit(
        "available_dijkstra_fifo_tie",
        paths_payload(tie_analyzer.find_shortest_paths(
            request(
                ShortestPathMethod::available_shortest,
                10,
                1.0e6,
                1U,
                3U))));

    emit(
        "k_shortest_zero",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest, 0))));
    emit(
        "k_shortest_negative",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest, -1))));
    emit(
        "length_zero",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest_length, 0))));
    emit(
        "length_cutoff_four",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest_length, 4))));
    emit(
        "max_first_only",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest, 3, 3.0))));
    emit(
        "max_reject",
        paths_payload(unconstrained.find_shortest_paths(
            request(ShortestPathMethod::k_shortest, 3, 2.5))));
    emit(
        "no_path",
        paths_payload(unconstrained.find_shortest_paths(
            request(
                ShortestPathMethod::first_shortest,
                10,
                1.0e6,
                0U,
                6U))));

    const PreparedTopologyAnalyzer hard_soft = fixture.prepare(
        {fixture.virtual_hard, fixture.virtual_soft});
    emit(
        "available_hard_soft",
        paths_payload(hard_soft.find_shortest_paths(
            request(ShortestPathMethod::available_shortest))));
    emit(
        "bfs_hard_soft",
        paths_payload(hard_soft.find_shortest_paths(
            request(ShortestPathMethod::bfs_shortest))));

    const PreparedTopologyAnalyzer soft_only = fixture.prepare(
        {fixture.virtual_soft});
    emit(
        "available_soft_only",
        paths_payload(soft_only.find_shortest_paths(
            request(ShortestPathMethod::available_shortest))));
    emit(
        "bfs_soft_only",
        paths_payload(soft_only.find_shortest_paths(
            request(ShortestPathMethod::bfs_shortest))));

    const Graph& graph = fixture.physical_network.graph();
    const PreparedTopologyAnalyzer prune_single = fixture.prepare(
        {}, {fixture.virtual_hard});
    emit(
        "prune_ratio_then_div",
        mask_payload(
            prune_single.create_pruned_mask(
                virtual_link, 2.0, 3.0, 1U),
            graph));
    emit(
        "prune_equal_boundary",
        mask_payload(
            prune_single.create_pruned_mask(
                virtual_link, 1.0, 1.0, 1U),
            graph));

    const PreparedTopologyAnalyzer prune_duplicate = fixture.prepare(
        {}, {fixture.virtual_hard, fixture.virtual_hard});
    emit(
        "prune_duplicate_resource",
        mask_payload(
            prune_duplicate.create_pruned_mask(
                virtual_link, 2.0, 3.0, 1U),
            graph));

    const PreparedTopologyAnalyzer prune_soft = fixture.prepare(
        {}, {fixture.virtual_soft});
    emit(
        "prune_soft_resource",
        mask_payload(
            prune_soft.create_pruned_mask(
                virtual_link, 100.0, 0.0, 1U),
            graph));

    emit(
        "prune_empty_resources",
        mask_payload(
            unconstrained.create_pruned_mask(
                virtual_link, 99.0, 77.0, 1U),
            graph));
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1 ||
            (argc == 2 &&
             std::string_view(argv[1]) == "differential"))
        {
            differential();
            return 0;
        }
        throw std::invalid_argument(
            "usage: topology_analyzer_harness [differential]");
    }
    catch (const std::exception& error)
    {
        std::cerr << "topology analyzer harness: "
                  << error.what() << '\n';
        return 1;
    }
}
