#pragma once

#include "constraint_checker.h"

#include "attribute/link_attribute.h"
#include "nx/subgraph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace virne::core::controller
{

enum class ShortestPathMethod : std::uint8_t
{
    first_shortest,
    k_shortest,
    k_shortest_length,
    all_shortest,
    bfs_shortest,
    available_shortest,
};

enum class TopologyAnalyzerErrorCode : std::uint8_t
{
    invalid_method,
    invalid_link_resource_selection,
    missing_physical_link_resource,
    virtual_link_not_found,
    missing_resource_value,
    non_numeric_resource,
};

enum class TopologyAnalyzerOperation : std::uint8_t
{
    prepare,
    find_paths,
    find_paths_batch,
    create_available_mask,
    create_pruned_mask,
};

class TopologyAnalyzerException : public std::runtime_error
{
public:
    TopologyAnalyzerException(
        TopologyAnalyzerErrorCode code,
        TopologyAnalyzerOperation operation,
        std::string message,
        std::optional<std::size_t> request_index = std::nullopt,
        std::optional<std::size_t> item_index = std::nullopt,
        std::optional<ConstraintId> resource_id = std::nullopt);

    TopologyAnalyzerErrorCode code() const noexcept;
    TopologyAnalyzerOperation operation() const noexcept;
    const std::optional<std::size_t>& request_index() const noexcept;
    const std::optional<std::size_t>& item_index() const noexcept;
    const std::optional<ConstraintId>& resource_id() const noexcept;

private:
    TopologyAnalyzerErrorCode code_;
    TopologyAnalyzerOperation operation_;
    std::optional<std::size_t> request_index_;
    std::optional<std::size_t> item_index_;
    std::optional<ConstraintId> resource_id_;
};

struct ShortestPathOptions
{
    ShortestPathMethod method = ShortestPathMethod::k_shortest;
    std::int64_t k = 10;
    double max_path_nodes = 1.0e6;
    std::size_t constraint_workers = 1U;
};

struct TopologyPathRequest
{
    ConstraintLink virtual_link;
    ConstraintLink physical_pair;
    ShortestPathOptions options;
};

struct TopologyAnalyzerSelection
{
    ConstraintCheckerSelection constraints;
    std::vector<ConstraintId> link_resources;
};

class PreparedTopologyAnalyzer;

class TopologyAnalyzer
{
public:
    explicit TopologyAnalyzer(TopologyAnalyzerSelection selection);

    const TopologyAnalyzerSelection& selection() const noexcept;

    PreparedTopologyAnalyzer prepare(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network) const;

private:
    TopologyAnalyzerSelection selection_;
    ConstraintChecker checker_;
};

class PreparedTopologyAnalyzer
{
public:
    using Paths = std::vector<std::vector<Vertex>>;

    Paths find_shortest_paths(const TopologyPathRequest& request) const;

    std::optional<std::vector<Vertex>> find_bfs_shortest_path(
        ConstraintLink virtual_link,
        Vertex source,
        Vertex target) const;

    SearchMask create_available_mask(
        ConstraintLink virtual_link,
        std::size_t workers = 1U) const;

    ::nx::GraphView create_available_network(
        ConstraintLink virtual_link) const;

    SearchMask create_pruned_mask(
        ConstraintLink virtual_link,
        double ratio = 1.0,
        double div = 0.0,
        std::size_t workers = 1U) const;

    ::nx::GraphView create_pruned_network(
        ConstraintLink virtual_link,
        double ratio = 1.0,
        double div = 0.0) const;

    std::vector<Paths> find_shortest_paths_batch(
        const std::vector<TopologyPathRequest>& requests,
        std::size_t workers = 1U) const;

private:
    struct PreparedLinkResource
    {
        ConstraintId resource_id = 0U;
        const network::attribute::LinkResourceAttribute* attribute = nullptr;
        AttrId virtual_value_id = 0U;
        AttrId physical_value_id = 0U;
    };

    PreparedTopologyAnalyzer(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        PreparedConstraintChecker checker,
        std::vector<PreparedLinkResource> link_resources);

    AttrMap adjusted_virtual_link(
        ConstraintLink virtual_link,
        double ratio,
        double div) const;

    static bool pruned_edge_is_available(
        const std::vector<PreparedLinkResource>& resources,
        const AttrMap& adjusted_virtual_link,
        const AttrMap& physical_link);

    const network::VirtualNetwork* virtual_network_ = nullptr;
    const network::PhysicalNetwork* physical_network_ = nullptr;
    PreparedConstraintChecker checker_;
    std::vector<PreparedLinkResource> link_resources_;

    friend class TopologyAnalyzer;
};

} // namespace virne::core::controller
