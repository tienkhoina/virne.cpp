#pragma once

#include "../../network/base_network.h"

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

using LinkRankResourceId = network::attribute::AttributeRegistryId;
using LinkRankEdgeId = std::uint32_t;

inline constexpr std::size_t invalid_link_rank_input_index =
    std::numeric_limits<std::size_t>::max();

enum class LinkRankMethod : std::uint8_t {
    order,
    ffd,
};

struct LinkRankSelection {
    // nullopt selects every link resource in registry order. A present vector
    // is already resolved and is retained in caller order, including repeats.
    std::optional<std::vector<LinkRankResourceId>> resources;
};

struct LinkRankOptions {
    bool sort = true;
    std::size_t workers = 1U;
};

struct LinkRankEntry {
    LinkRankEdgeId edge_id = 0U;
    Vertex source = 0U;
    Vertex target = 0U;
    double value = 0.0;
};

using LinkRanking = std::vector<LinkRankEntry>;

enum class LinkRankErrorCode : std::uint8_t {
    unsupported_method,
    invalid_resource_selection,
    empty_resource_selection,
    ragged_resource_matrix,
    non_numeric_resource_value,
    ranking_length_mismatch,
    invalid_prepared_state,
};

enum class LinkRankOperation : std::uint8_t {
    resolve_method,
    prepare,
    validate_prepared,
    gather,
    reduce,
    sort,
};

class LinkRankException : public std::runtime_error {
public:
    LinkRankException(
        LinkRankErrorCode code,
        LinkRankOperation operation,
        std::string message,
        std::size_t input_index = invalid_link_rank_input_index,
        std::optional<LinkRankResourceId> resource_id = std::nullopt,
        std::optional<LinkRankEdgeId> edge_id = std::nullopt);

    LinkRankErrorCode code() const noexcept;
    LinkRankOperation operation() const noexcept;
    std::size_t input_index() const noexcept;
    const std::optional<LinkRankResourceId>& resource_id() const noexcept;
    const std::optional<LinkRankEdgeId>& edge_id() const noexcept;

private:
    LinkRankErrorCode code_;
    LinkRankOperation operation_;
    std::size_t input_index_;
    std::optional<LinkRankResourceId> resource_id_;
    std::optional<LinkRankEdgeId> edge_id_;
};

LinkRankMethod link_rank_method_from_string(std::string_view value);
std::string_view link_rank_method_name(LinkRankMethod value) noexcept;

class PreparedLinkRanker;

class LinkRanker {
public:
    explicit LinkRanker(LinkRankSelection selection = {});

    PreparedLinkRanker prepare(const network::BaseNetwork& network) const;

private:
    LinkRankSelection selection_;
};

class PreparedLinkRanker {
public:
    LinkRanking rank(
        LinkRankMethod method,
        LinkRankOptions options = {}) const;
    LinkRanking rank_order(LinkRankOptions options = {}) const;
    LinkRanking rank_ffd(LinkRankOptions options = {}) const;

    const std::vector<LinkRankResourceId>& resource_ids() const noexcept;

private:
    struct ResourceBinding {
        LinkRankResourceId registry_id = 0U;
        AttrId value_id = 0U;
        const network::attribute::LinkAttribute* definition_identity = nullptr;
    };

    PreparedLinkRanker(
        const network::BaseNetwork& network,
        std::vector<ResourceBinding> resources);

    const network::BaseNetwork& checked_network() const;

    const network::BaseNetwork* network_ = nullptr;
    const Graph* graph_identity_ = nullptr;
    const ::AttributeRegistry* graph_registry_identity_ = nullptr;
    const network::attribute::LinkAttributeRegistry*
        link_registry_identity_ = nullptr;
    std::vector<ResourceBinding> resources_;
    std::vector<LinkRankResourceId> resource_ids_;
    // Graph::edges() may perform a logically-const lazy order normalization.
    // Sharing this narrow lock across PreparedLinkRanker copies keeps
    // concurrent read-only calls safe without serializing numeric reduction.
    std::shared_ptr<std::mutex> edge_order_mutex_;

    friend class LinkRanker;
};

}  // namespace virne::solver::rank
