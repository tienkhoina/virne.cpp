#include "link_rank.h"
#include "python310_timsort.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <sstream>
#include <thread>
#include <utility>

namespace virne::solver::rank {
namespace {

struct OrderedLink {
    Edge descriptor;
    LinkRankEdgeId edge_id = 0U;
    Vertex source = 0U;
    Vertex target = 0U;
};

std::vector<OrderedLink> collect_ordered_links(const Graph& graph)
{
    std::vector<OrderedLink> links;
    links.reserve(graph.num_edges());

    const auto range = graph.edges();
    for (auto iterator = range.first; iterator != range.second; ++iterator) {
        const Edge edge = *iterator;
        links.push_back(OrderedLink{
            edge,
            graph.edge_id(edge),
            graph.source(edge),
            graph.target(edge)});
    }
    return links;
}

LinkRanking make_ranking(
    const std::vector<OrderedLink>& links,
    const std::vector<double>& values)
{
    if (links.size() != values.size()) {
        throw LinkRankException(
            LinkRankErrorCode::ranking_length_mismatch,
            LinkRankOperation::reduce,
            "Link ranking length does not match the live edge order");
    }

    LinkRanking ranking;
    ranking.reserve(links.size());
    for (std::size_t index = 0U; index < links.size(); ++index) {
        const OrderedLink& link = links[index];
        ranking.push_back(LinkRankEntry{
            link.edge_id,
            link.source,
            link.target,
            values[index]});
    }
    return ranking;
}

// Python's finite descending order is a stable numeric order. NaN makes
// Python's rich comparison non-transitive, so feeding it to a Standard
// Library sorting comparator would violate the strict-weak-order contract.
// This cold path uses a stable insertion algorithm with exactly Python's
// reverse-key comparison rule; the finite hot path remains O(E log E).
void python_safe_descending_sort(LinkRanking& ranking)
{
    const bool contains_nan = std::any_of(
        ranking.begin(),
        ranking.end(),
        [](const LinkRankEntry& entry) { return std::isnan(entry.value); });

    if (!contains_nan) {
        std::stable_sort(
            ranking.begin(),
            ranking.end(),
            [](const LinkRankEntry& lhs, const LinkRankEntry& rhs) {
                return lhs.value > rhs.value;
            });
        return;
    }

    detail::python310_timsort_reverse(ranking);
}

std::int64_t uint64_bits_as_int64(std::uint64_t value) noexcept
{
    std::int64_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "int64 width mismatch");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

}  // namespace

LinkRankException::LinkRankException(
    LinkRankErrorCode code,
    LinkRankOperation operation,
    std::string message,
    std::size_t input_index,
    std::optional<LinkRankResourceId> resource_id,
    std::optional<LinkRankEdgeId> edge_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index),
      resource_id_(resource_id),
      edge_id_(edge_id)
{
}

LinkRankErrorCode LinkRankException::code() const noexcept
{
    return code_;
}

LinkRankOperation LinkRankException::operation() const noexcept
{
    return operation_;
}

std::size_t LinkRankException::input_index() const noexcept
{
    return input_index_;
}

const std::optional<LinkRankResourceId>&
LinkRankException::resource_id() const noexcept
{
    return resource_id_;
}

const std::optional<LinkRankEdgeId>& LinkRankException::edge_id() const noexcept
{
    return edge_id_;
}

LinkRankMethod link_rank_method_from_string(std::string_view value)
{
    if (value == "order") {
        return LinkRankMethod::order;
    }
    if (value == "ffd") {
        return LinkRankMethod::ffd;
    }
    throw LinkRankException(
        LinkRankErrorCode::unsupported_method,
        LinkRankOperation::resolve_method,
        "Unsupported link rank method: " + std::string(value));
}

std::string_view link_rank_method_name(LinkRankMethod value) noexcept
{
    switch (value) {
    case LinkRankMethod::order:
        return "order";
    case LinkRankMethod::ffd:
        return "ffd";
    }
    return "unknown";
}

LinkRanker::LinkRanker(LinkRankSelection selection)
    : selection_(std::move(selection))
{
}

PreparedLinkRanker LinkRanker::prepare(
    const network::BaseNetwork& network) const
{
    std::vector<LinkRankResourceId> selected;
    if (selection_.resources.has_value()) {
        selected = *selection_.resources;
    } else {
        network::AttributeSelection selection;
        selection.kinds = std::vector<network::attribute::AttributeKind>{
            network::attribute::AttributeKind::resource};
        selected = network.select_link_attributes(selection);
    }

    std::vector<PreparedLinkRanker::ResourceBinding> resources;
    resources.reserve(selected.size());
    for (std::size_t index = 0U; index < selected.size(); ++index) {
        const LinkRankResourceId registry_id = selected[index];
        try {
            const network::attribute::LinkAttribute& definition =
                network.link_attributes().at(registry_id);
            if (definition.spec().kind !=
                network::attribute::AttributeKind::resource) {
                throw LinkRankException(
                    LinkRankErrorCode::invalid_resource_selection,
                    LinkRankOperation::prepare,
                    "Selected link attribute is not a resource",
                    index,
                    registry_id);
            }
            const network::attribute::LinkAttributeBinding binding =
                definition.bind(network.graph());
            resources.push_back(PreparedLinkRanker::ResourceBinding{
                registry_id,
                binding.value_id,
                &definition});
        } catch (const LinkRankException&) {
            throw;
        } catch (const std::exception& error) {
            throw LinkRankException(
                LinkRankErrorCode::invalid_resource_selection,
                LinkRankOperation::prepare,
                std::string("Cannot bind selected link resource: ") +
                    error.what(),
                index,
                registry_id);
        }
    }

    return PreparedLinkRanker(network, std::move(resources));
}

PreparedLinkRanker::PreparedLinkRanker(
    const network::BaseNetwork& network,
    std::vector<ResourceBinding> resources)
    : network_(&network),
      graph_identity_(&network.graph()),
      graph_registry_identity_(&network.graph().attribute_registry()),
      link_registry_identity_(&network.link_attributes()),
      resources_(std::move(resources)),
      edge_order_mutex_(std::make_shared<std::mutex>())
{
    resource_ids_.reserve(resources_.size());
    for (const ResourceBinding& resource : resources_) {
        resource_ids_.push_back(resource.registry_id);
    }
}

const network::BaseNetwork& PreparedLinkRanker::checked_network() const
{
    if (network_ == nullptr || graph_identity_ == nullptr ||
        graph_registry_identity_ == nullptr ||
        link_registry_identity_ == nullptr || edge_order_mutex_ == nullptr) {
        throw LinkRankException(
            LinkRankErrorCode::invalid_prepared_state,
            LinkRankOperation::validate_prepared,
            "Link ranker is not bound to a network");
    }

    if (&network_->graph() != graph_identity_ ||
        &network_->graph().attribute_registry() != graph_registry_identity_ ||
        &network_->link_attributes() != link_registry_identity_) {
        throw LinkRankException(
            LinkRankErrorCode::invalid_prepared_state,
            LinkRankOperation::validate_prepared,
            "Prepared link ranker network identity changed");
    }

    for (std::size_t index = 0U; index < resources_.size(); ++index) {
        const ResourceBinding& resource = resources_[index];
        try {
            if (&network_->link_attributes().at(resource.registry_id) !=
                resource.definition_identity) {
                throw LinkRankException(
                    LinkRankErrorCode::invalid_prepared_state,
                    LinkRankOperation::validate_prepared,
                    "Prepared link resource definition changed",
                    index,
                    resource.registry_id);
            }
        } catch (const LinkRankException&) {
            throw;
        } catch (const std::exception& error) {
            throw LinkRankException(
                LinkRankErrorCode::invalid_prepared_state,
                LinkRankOperation::validate_prepared,
                std::string("Prepared link resource is no longer valid: ") +
                    error.what(),
                index,
                resource.registry_id);
        }
    }
    return *network_;
}

LinkRanking PreparedLinkRanker::rank(
    LinkRankMethod method,
    LinkRankOptions options) const
{
    switch (method) {
    case LinkRankMethod::order:
        return rank_order(options);
    case LinkRankMethod::ffd:
        return rank_ffd(options);
    }
    throw LinkRankException(
        LinkRankErrorCode::unsupported_method,
        LinkRankOperation::resolve_method,
        "Unsupported typed link rank method");
}

LinkRanking PreparedLinkRanker::rank_order(LinkRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    std::vector<OrderedLink> links;
    {
        const std::lock_guard<std::mutex> lock(*edge_order_mutex_);
        links = collect_ordered_links(network.graph());
    }

    LinkRanking ranking;
    ranking.reserve(links.size());
    if (!options.sort) {
        for (std::size_t index = 0U; index < links.size(); ++index) {
            const OrderedLink& link = links[index];
            ranking.push_back(LinkRankEntry{
                link.edge_id,
                link.source,
                link.target,
                static_cast<double>(index)});
        }
        return ranking;
    }

    for (std::size_t offset = 0U; offset < links.size(); ++offset) {
        const std::size_t index = links.size() - 1U - offset;
        const OrderedLink& link = links[index];
        ranking.push_back(LinkRankEntry{
            link.edge_id,
            link.source,
            link.target,
            static_cast<double>(index)});
    }
    return ranking;
}

LinkRanking PreparedLinkRanker::rank_ffd(LinkRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    if (resources_.empty()) {
        throw LinkRankException(
            LinkRankErrorCode::empty_resource_selection,
            LinkRankOperation::gather,
            "FFD link ranking requires at least one link resource");
    }

    const Graph& graph = network.graph();
    std::vector<OrderedLink> links;
    {
        const std::lock_guard<std::mutex> lock(*edge_order_mutex_);
        links = collect_ordered_links(graph);
    }
    if (links.empty()) {
        return {};
    }

    // Resource-major compact storage mirrors LinkAttribute::get_data: a
    // missing edge value is omitted from that row, never padded.
    std::vector<const AttrValue*> matrix;
    matrix.reserve(resources_.size() * links.size());
    std::vector<std::size_t> row_lengths;
    row_lengths.reserve(resources_.size());

    for (const ResourceBinding& resource : resources_) {
        const std::size_t row_begin = matrix.size();
        for (const OrderedLink& link : links) {
            const AttrValue* value =
                graph.edge_attrs(link.descriptor).find(resource.value_id);
            if (value != nullptr) {
                matrix.push_back(value);
            }
        }
        row_lengths.push_back(matrix.size() - row_begin);
    }

    const std::size_t column_count = row_lengths.front();
    for (std::size_t row = 1U; row < row_lengths.size(); ++row) {
        if (row_lengths[row] != column_count) {
            throw LinkRankException(
                LinkRankErrorCode::ragged_resource_matrix,
                LinkRankOperation::gather,
                "Link resource rows have unequal compact lengths",
                row,
                resources_[row].registry_id);
        }
    }

    bool double_lane = false;
    for (std::size_t row = 0U; row < resources_.size(); ++row) {
        for (std::size_t column = 0U; column < column_count; ++column) {
            const AttrValue& value =
                *matrix[row * column_count + column];
            if (std::holds_alternative<double>(value)) {
                double_lane = true;
            } else if (!std::holds_alternative<std::int64_t>(value) &&
                       !std::holds_alternative<bool>(value)) {
                const std::optional<LinkRankEdgeId> edge_id =
                    column < links.size()
                        ? std::optional<LinkRankEdgeId>{links[column].edge_id}
                        : std::nullopt;
                throw LinkRankException(
                    LinkRankErrorCode::non_numeric_resource_value,
                    LinkRankOperation::reduce,
                    "Link resource value is not bool, int64, or double",
                    row,
                    resources_[row].registry_id,
                    edge_id);
            }
        }
    }

    std::vector<double> values(column_count, 0.0);
    const auto reduce_range = [&](std::size_t begin, std::size_t end) {
        if (double_lane) {
            for (std::size_t column = begin; column < end; ++column) {
                double sum = 0.0;
                for (std::size_t row = 0U; row < resources_.size(); ++row) {
                    const AttrValue& value =
                        *matrix[row * column_count + column];
                    if (const auto* floating = std::get_if<double>(&value)) {
                        sum += *floating;
                    } else if (const auto* integer =
                                   std::get_if<std::int64_t>(&value)) {
                        sum += static_cast<double>(*integer);
                    } else {
                        sum += std::get<bool>(value) ? 1.0 : 0.0;
                    }
                }
                values[column] = sum;
            }
            return;
        }

        for (std::size_t column = begin; column < end; ++column) {
            std::uint64_t sum = 0U;
            for (std::size_t row = 0U; row < resources_.size(); ++row) {
                const AttrValue& value =
                    *matrix[row * column_count + column];
                if (const auto* number = std::get_if<std::int64_t>(&value)) {
                    sum += static_cast<std::uint64_t>(*number);
                } else {
                    sum += std::get<bool>(value) ? std::uint64_t{1U}
                                                : std::uint64_t{0U};
                }
            }
            values[column] = static_cast<double>(uint64_bits_as_int64(sum));
        }
    };

    const std::size_t requested_workers = options.workers;
    const std::size_t worker_count =
        requested_workers <= 1U
            ? 1U
            : std::min(requested_workers, column_count);
    if (worker_count <= 1U) {
        reduce_range(0U, column_count);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(worker_count - 1U);
        try {
            for (std::size_t worker = 1U; worker < worker_count; ++worker) {
                const std::size_t begin =
                    (column_count * worker) / worker_count;
                const std::size_t end =
                    (column_count * (worker + 1U)) / worker_count;
                threads.emplace_back(reduce_range, begin, end);
            }
            reduce_range(0U, column_count / worker_count);
        } catch (...) {
            for (std::thread& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            throw;
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    if (column_count != links.size()) {
        throw LinkRankException(
            LinkRankErrorCode::ranking_length_mismatch,
            LinkRankOperation::reduce,
            "Compact link-resource columns do not cover every live edge");
    }

    LinkRanking ranking = make_ranking(links, values);
    if (options.sort) {
        python_safe_descending_sort(ranking);
    }
    return ranking;
}

const std::vector<LinkRankResourceId>&
PreparedLinkRanker::resource_ids() const noexcept
{
    return resource_ids_;
}

}  // namespace virne::solver::rank
