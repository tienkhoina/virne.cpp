#pragma once

#include "../../network/base_network.h"
#include "../../../random/numpy_random_state.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace virne::solver::rank {

using NodeRankResourceId = network::attribute::AttributeRegistryId;

inline constexpr std::size_t invalid_node_rank_input_index =
    std::numeric_limits<std::size_t>::max();

enum class NodeRankMethod : std::uint8_t {
    order,
    random,
    ffd,
    nrm,
    nea,
    grc,
    rw,
    nps,
};

enum class NodeRankValueKind : std::uint8_t {
    scalar,
    proximity,
};

struct NodeRankEntry {
    Vertex node_id = 0U;
    NodeRankValueKind kind = NodeRankValueKind::scalar;
    double value = 0.0;
    double distance = 0.0;
};

using NodeRanking = std::vector<NodeRankEntry>;

struct NodeRankSelection {
    // nullopt selects every resource of that owner in registry order. A
    // present vector is already resolved and retains repeats/caller order.
    std::optional<std::vector<NodeRankResourceId>> node_resources;
    std::optional<std::vector<NodeRankResourceId>> link_resources;
};

struct GRCNodeRankParameters {
    double sigma = 1e-5;
    double damping = 0.85;
};

struct RWNodeRankParameters {
    double sigma = 1e-4;
    double jump_probability = 0.15;
    double forwarding_probability = 0.85;
};

struct NodeRankParameters {
    GRCNodeRankParameters grc;
    RWNodeRankParameters rw;
};

struct NodeRankOptions {
    bool sort = true;
    std::size_t workers = 1U;
    // Python has no cap. nullopt preserves that behavior; callers may opt in
    // to a deterministic error for hostile/non-convergent parameters.
    std::optional<std::size_t> max_iterations;
};

enum class NodeRankErrorCode : std::uint8_t {
    unsupported_method,
    random_stream_required,
    invalid_node_resource_selection,
    invalid_link_resource_selection,
    empty_node_resource_selection,
    empty_link_resource_selection,
    ragged_node_resource_matrix,
    non_numeric_node_resource_value,
    ranking_length_mismatch,
    stale_cardinality,
    invalid_matrix_shape,
    sparse_assignment_mismatch,
    iteration_limit_reached,
    invalid_prepared_state,
};

enum class NodeRankOperation : std::uint8_t {
    resolve_method,
    prepare,
    validate_prepared,
    gather_nodes,
    gather_links,
    build_matrix,
    randomize,
    reduce,
    iterate,
    traverse,
    sort,
};

class NodeRankException : public std::runtime_error {
public:
    NodeRankException(
        NodeRankErrorCode code,
        NodeRankOperation operation,
        std::string message,
        std::size_t input_index = invalid_node_rank_input_index,
        std::optional<NodeRankResourceId> resource_id = std::nullopt,
        std::optional<Vertex> node_id = std::nullopt);

    NodeRankErrorCode code() const noexcept;
    NodeRankOperation operation() const noexcept;
    std::size_t input_index() const noexcept;
    const std::optional<NodeRankResourceId>& resource_id() const noexcept;
    const std::optional<Vertex>& node_id() const noexcept;

private:
    NodeRankErrorCode code_;
    NodeRankOperation operation_;
    std::size_t input_index_;
    std::optional<NodeRankResourceId> resource_id_;
    std::optional<Vertex> node_id_;
};

NodeRankMethod node_rank_method_from_string(std::string_view value);
std::string_view node_rank_method_name(NodeRankMethod value) noexcept;

class PreparedNodeRanker;

class NodeRanker {
public:
    explicit NodeRanker(
        NodeRankSelection selection = {},
        NodeRankParameters parameters = {});

    PreparedNodeRanker prepare(const network::BaseNetwork& network) const;

private:
    NodeRankSelection selection_;
    NodeRankParameters parameters_;
};

class PreparedNodeRanker {
public:
    NodeRanking rank(
        NodeRankMethod method,
        NodeRankOptions options = {}) const;
    NodeRanking rank_order(NodeRankOptions options = {}) const;
    NodeRanking rank_random(
        NumpyRandomState& random,
        NodeRankOptions options = {}) const;
    NodeRanking rank_ffd(NodeRankOptions options = {}) const;
    NodeRanking rank_nrm(NodeRankOptions options = {}) const;
    NodeRanking rank_nea(NodeRankOptions options = {}) const;
    NodeRanking rank_grc(NodeRankOptions options = {}) const;
    NodeRanking rank_rw(NodeRankOptions options = {}) const;
    NodeRanking rank_nps(NodeRankOptions options = {}) const;

    const std::vector<NodeRankResourceId>& node_resource_ids() const noexcept;
    const std::vector<NodeRankResourceId>& link_resource_ids() const noexcept;

private:
    struct NodeResourceBinding {
        NodeRankResourceId registry_id = 0U;
        AttrId value_id = 0U;
        const network::attribute::NodeAttribute* definition_identity = nullptr;
    };

    struct LinkResourceBinding {
        NodeRankResourceId registry_id = 0U;
        AttrId value_id = 0U;
        const network::attribute::LinkAttribute* definition_identity = nullptr;
    };

    PreparedNodeRanker(
        const network::BaseNetwork& network,
        std::vector<NodeResourceBinding> node_resources,
        std::vector<LinkResourceBinding> link_resources,
        NodeRankParameters parameters);

    const network::BaseNetwork& checked_network() const;

    const network::BaseNetwork* network_ = nullptr;
    const Graph* graph_identity_ = nullptr;
    const ::AttributeRegistry* graph_registry_identity_ = nullptr;
    const network::attribute::NodeAttributeRegistry*
        node_registry_identity_ = nullptr;
    const network::attribute::LinkAttributeRegistry*
        link_registry_identity_ = nullptr;
    std::vector<NodeResourceBinding> node_resources_;
    std::vector<LinkResourceBinding> link_resources_;
    std::vector<NodeRankResourceId> node_resource_ids_;
    std::vector<NodeRankResourceId> link_resource_ids_;
    NodeRankParameters parameters_;
    // Graph matrix/edge routines may lazily normalize frozen graph order.
    std::shared_ptr<std::mutex> graph_read_mutex_;

    friend class NodeRanker;
};

}  // namespace virne::solver::rank
