#include "node_rank.h"

#include "python310_generic_timsort.h"
#include "../../../graph/nx/shortest_paths.h"
#include "../../utils/deterministic_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

namespace virne::solver::rank {
namespace {

struct NodeResourceSums {
    bool double_lane = false;
    std::vector<std::int64_t> integers;
    std::vector<double> doubles;

    std::size_t size() const noexcept
    {
        return double_lane ? doubles.size() : integers.size();
    }

    double as_double(std::size_t index) const
    {
        return double_lane
            ? doubles[index]
            : static_cast<double>(integers[index]);
    }
};

std::int64_t uint64_bits_as_int64(std::uint64_t value) noexcept
{
    std::int64_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "int64 width mismatch");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template <typename Function>
void parallel_blocks(
    std::size_t count,
    std::size_t requested_workers,
    Function&& function,
    std::size_t minimum_items_per_worker = 4096U)
{
    virne::utils::deterministic_parallel_blocks(
        count,
        requested_workers,
        minimum_items_per_worker,
        std::forward<Function>(function));
}

std::size_t quadratic_block_grain(std::size_t extent) noexcept
{
    // Roughly 32K ordered scalar operations per worker keeps persistent-pool
    // synchronization out of small solver graphs while still exposing the
    // caller's configured width on genuinely quadratic rank work.
    constexpr std::size_t minimum_operations_per_worker = 32768U;
    return std::max<std::size_t>(
        1U,
        minimum_operations_per_worker /
            std::max<std::size_t>(1U, extent));
}

template <typename ResourceBindings>
NodeResourceSums gather_node_resource_sums(
    const Graph& graph,
    const ResourceBindings& resources,
    std::size_t workers)
{
    if (resources.empty()) {
        throw NodeRankException(
            NodeRankErrorCode::empty_node_resource_selection,
            NodeRankOperation::gather_nodes,
            "Node ranking requires at least one node resource");
    }

    const std::size_t live_nodes = graph.num_nodes();
    std::vector<const AttrValue*> matrix;
    matrix.reserve(resources.size() * live_nodes);
    std::vector<std::size_t> row_lengths;
    row_lengths.reserve(resources.size());

    for (const auto& resource : resources) {
        const std::size_t row_begin = matrix.size();
        for (std::size_t index = 0U; index < live_nodes; ++index) {
            const Vertex node = static_cast<Vertex>(index);
            const AttrValue* value =
                graph.node_attrs(node).find(resource.value_id);
            if (value != nullptr) {
                matrix.push_back(value);
            }
        }
        row_lengths.push_back(matrix.size() - row_begin);
    }

    const std::size_t column_count = row_lengths.front();
    for (std::size_t row = 1U; row < row_lengths.size(); ++row) {
        if (row_lengths[row] != column_count) {
            throw NodeRankException(
                NodeRankErrorCode::ragged_node_resource_matrix,
                NodeRankOperation::gather_nodes,
                "Node resource rows have unequal compact lengths",
                row,
                resources[row].registry_id);
        }
    }

    bool double_lane = false;
    for (std::size_t row = 0U; row < resources.size(); ++row) {
        for (std::size_t column = 0U; column < column_count; ++column) {
            const AttrValue& value =
                *matrix[row * column_count + column];
            if (std::holds_alternative<double>(value)) {
                double_lane = true;
            } else if (!std::holds_alternative<std::int64_t>(value) &&
                       !std::holds_alternative<bool>(value)) {
                const std::optional<Vertex> node_id =
                    column < live_nodes
                        ? std::optional<Vertex>{static_cast<Vertex>(column)}
                        : std::nullopt;
                throw NodeRankException(
                    NodeRankErrorCode::non_numeric_node_resource_value,
                    NodeRankOperation::reduce,
                    "Node resource value is not bool, int64, or double",
                    row,
                    resources[row].registry_id,
                    node_id);
            }
        }
    }

    NodeResourceSums result;
    result.double_lane = double_lane;
    if (double_lane) {
        result.doubles.assign(column_count, 0.0);
        const auto reduce = [&](std::size_t begin, std::size_t end) {
            for (std::size_t column = begin; column < end; ++column) {
                double sum = 0.0;
                for (std::size_t row = 0U; row < resources.size(); ++row) {
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
                result.doubles[column] = sum;
            }
        };
        parallel_blocks(column_count, workers, reduce);
        return result;
    }

    result.integers.assign(column_count, 0);
    const auto reduce = [&](std::size_t begin, std::size_t end) {
        for (std::size_t column = begin; column < end; ++column) {
            std::uint64_t sum = 0U;
            for (std::size_t row = 0U; row < resources.size(); ++row) {
                const AttrValue& value =
                    *matrix[row * column_count + column];
                if (const auto* integer = std::get_if<std::int64_t>(&value)) {
                    sum += static_cast<std::uint64_t>(*integer);
                } else {
                    sum += std::get<bool>(value) ? std::uint64_t{1U}
                                                : std::uint64_t{0U};
                }
            }
            result.integers[column] = uint64_bits_as_int64(sum);
        }
    };
    parallel_blocks(column_count, workers, reduce);
    return result;
}

void validate_scalar_ranking_length(
    const network::BaseNetwork& network,
    std::size_t score_count)
{
    const std::size_t cached_nodes = network.num_nodes();
    if (cached_nodes != score_count) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::reduce,
            "Node ranking length does not match cached network cardinality");
    }
    if (network.live_num_nodes() != cached_nodes) {
        throw NodeRankException(
            NodeRankErrorCode::stale_cardinality,
            NodeRankOperation::validate_prepared,
            "Cached and live network node cardinalities differ");
    }
}

NodeRanking make_scalar_ranking(
    const network::BaseNetwork& network,
    const std::vector<double>& values)
{
    validate_scalar_ranking_length(network, values.size());
    NodeRanking result;
    result.reserve(values.size());
    for (std::size_t index = 0U; index < values.size(); ++index) {
        result.push_back(NodeRankEntry{
            static_cast<Vertex>(index),
            NodeRankValueKind::scalar,
            values[index],
            0.0});
    }
    return result;
}

NodeRanking make_scalar_ranking(
    const network::BaseNetwork& network,
    const NodeResourceSums& sums)
{
    validate_scalar_ranking_length(network, sums.size());
    NodeRanking result;
    result.reserve(sums.size());
    for (std::size_t index = 0U; index < sums.size(); ++index) {
        result.push_back(NodeRankEntry{
            static_cast<Vertex>(index),
            NodeRankValueKind::scalar,
            sums.as_double(index),
            0.0});
    }
    return result;
}

void finalize_scalar_ranking(
    const network::BaseNetwork& network,
    NodeRanking& result)
{
    validate_scalar_ranking_length(network, result.size());
    for (std::size_t index = 0U; index < result.size(); ++index) {
        result[index].node_id = static_cast<Vertex>(index);
        result[index].kind = NodeRankValueKind::scalar;
        result[index].distance = 0.0;
    }
}

void python_scalar_descending_sort(NodeRanking& ranking)
{
    const bool contains_nan = std::any_of(
        ranking.begin(),
        ranking.end(),
        [](const NodeRankEntry& entry) { return std::isnan(entry.value); });
    if (!contains_nan) {
        std::stable_sort(
            ranking.begin(),
            ranking.end(),
            [](const NodeRankEntry& lhs, const NodeRankEntry& rhs) {
                return lhs.value > rhs.value;
            });
        return;
    }

    detail::python310_timsort_reverse(
        ranking,
        [](const NodeRankEntry& lhs, const NodeRankEntry& rhs) {
            return lhs.value < rhs.value;
        });
}

double sequential_sum(const std::vector<double>& values) noexcept
{
    double result = 0.0;
    for (double value : values) {
        result += value;
    }
    return result;
}

double node_sum_total(const NodeResourceSums& sums) noexcept
{
    if (sums.double_lane) {
        return sequential_sum(sums.doubles);
    }

    std::uint64_t total = 0U;
    for (std::int64_t value : sums.integers) {
        total += static_cast<std::uint64_t>(value);
    }
    return static_cast<double>(uint64_bits_as_int64(total));
}

double l2_delta(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) noexcept
{
    double sum = 0.0;
    for (std::size_t index = 0U; index < lhs.size(); ++index) {
        const double difference = lhs[index] - rhs[index];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

bool nps_key_less(const NodeRankEntry& lhs, const NodeRankEntry& rhs) noexcept
{
    // Python tuple comparison checks equality before ordering each element.
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    const double lhs_secondary = -lhs.value;
    const double rhs_secondary = -rhs.value;
    if (lhs_secondary != rhs_secondary) {
        return lhs_secondary < rhs_secondary;
    }
    return false;
}

void python_nps_ascending_sort(NodeRanking& ranking)
{
    const bool contains_nan = std::any_of(
        ranking.begin(),
        ranking.end(),
        [](const NodeRankEntry& entry) {
            return std::isnan(entry.distance) || std::isnan(entry.value);
        });
    if (!contains_nan) {
        std::stable_sort(ranking.begin(), ranking.end(), nps_key_less);
        return;
    }

    detail::python310_timsort(ranking, nps_key_less);
}

std::vector<double> reduce_link_aggregation_rows(
    const std::vector<std::vector<double>>& rows,
    std::size_t node_count,
    std::size_t workers)
{
    if (rows.empty()) {
        return std::vector<double>(node_count, 0.0);
    }
    for (std::size_t row = 0U; row < rows.size(); ++row) {
        if (rows[row].size() != node_count) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_matrix_shape,
                NodeRankOperation::gather_links,
                "Link aggregation row does not match live node count",
                row);
        }
    }

    std::vector<double> result(node_count, 0.0);
    const auto reduce = [&](std::size_t begin, std::size_t end) {
        for (std::size_t node = begin; node < end; ++node) {
            double sum = 0.0;
            for (const std::vector<double>& row : rows) {
                sum += row[node];
            }
            result[node] = sum;
        }
    };
    parallel_blocks(node_count, workers, reduce);
    return result;
}

std::vector<double> average_matrices(
    const std::vector<DistanceMatrix>& matrices,
    std::size_t node_count,
    std::size_t workers)
{
    if (matrices.empty()) {
        throw NodeRankException(
            NodeRankErrorCode::empty_link_resource_selection,
            NodeRankOperation::build_matrix,
            "Iterative node ranking requires at least one link resource");
    }
    if (node_count != 0U &&
        node_count > std::numeric_limits<std::size_t>::max() / node_count) {
        throw NodeRankException(
            NodeRankErrorCode::invalid_matrix_shape,
            NodeRankOperation::build_matrix,
            "Node ranking matrix extent overflows size_t");
    }
    const std::size_t cell_count = node_count * node_count;
    for (std::size_t index = 0U; index < matrices.size(); ++index) {
        if (matrices[index].n != node_count ||
            matrices[index].data.size() != cell_count) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_matrix_shape,
                NodeRankOperation::build_matrix,
                "Link adjacency matrix does not match live node count",
                index);
        }
    }

    std::vector<double> result(cell_count, 0.0);
    const double divisor = static_cast<double>(matrices.size());
    const auto average = [&](std::size_t begin, std::size_t end) {
        for (std::size_t cell = begin; cell < end; ++cell) {
            double sum = 0.0;
            for (const DistanceMatrix& matrix : matrices) {
                sum += matrix.data[cell];
            }
            result[cell] = sum / divisor;
        }
    };
    parallel_blocks(cell_count, workers, average);
    return result;
}

}  // namespace

NodeRankException::NodeRankException(
    NodeRankErrorCode code,
    NodeRankOperation operation,
    std::string message,
    std::size_t input_index,
    std::optional<NodeRankResourceId> resource_id,
    std::optional<Vertex> node_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index),
      resource_id_(resource_id),
      node_id_(node_id)
{
}

NodeRankErrorCode NodeRankException::code() const noexcept
{
    return code_;
}

NodeRankOperation NodeRankException::operation() const noexcept
{
    return operation_;
}

std::size_t NodeRankException::input_index() const noexcept
{
    return input_index_;
}

const std::optional<NodeRankResourceId>&
NodeRankException::resource_id() const noexcept
{
    return resource_id_;
}

const std::optional<Vertex>& NodeRankException::node_id() const noexcept
{
    return node_id_;
}

NodeRankMethod node_rank_method_from_string(std::string_view value)
{
    if (value == "order") {
        return NodeRankMethod::order;
    }
    if (value == "random") {
        return NodeRankMethod::random;
    }
    if (value == "ffd") {
        return NodeRankMethod::ffd;
    }
    if (value == "nrm") {
        return NodeRankMethod::nrm;
    }
    if (value == "nea") {
        return NodeRankMethod::nea;
    }
    if (value == "grc") {
        return NodeRankMethod::grc;
    }
    if (value == "rw") {
        return NodeRankMethod::rw;
    }
    if (value == "nps") {
        return NodeRankMethod::nps;
    }
    throw NodeRankException(
        NodeRankErrorCode::unsupported_method,
        NodeRankOperation::resolve_method,
        "Unsupported node rank method: " + std::string(value));
}

std::string_view node_rank_method_name(NodeRankMethod value) noexcept
{
    switch (value) {
    case NodeRankMethod::order:
        return "order";
    case NodeRankMethod::random:
        return "random";
    case NodeRankMethod::ffd:
        return "ffd";
    case NodeRankMethod::nrm:
        return "nrm";
    case NodeRankMethod::nea:
        return "nea";
    case NodeRankMethod::grc:
        return "grc";
    case NodeRankMethod::rw:
        return "rw";
    case NodeRankMethod::nps:
        return "nps";
    }
    return "unknown";
}

NodeRanker::NodeRanker(
    NodeRankSelection selection,
    NodeRankParameters parameters)
    : selection_(std::move(selection)), parameters_(parameters)
{
}

PreparedNodeRanker NodeRanker::prepare(
    const network::BaseNetwork& network) const
{
    std::vector<NodeRankResourceId> selected_nodes;
    if (selection_.node_resources.has_value()) {
        selected_nodes = *selection_.node_resources;
    } else {
        network::AttributeSelection selection;
        selection.kinds = std::vector<network::attribute::AttributeKind>{
            network::attribute::AttributeKind::resource};
        selected_nodes = network.select_node_attributes(selection);
    }

    std::vector<PreparedNodeRanker::NodeResourceBinding> node_resources;
    node_resources.reserve(selected_nodes.size());
    for (std::size_t index = 0U; index < selected_nodes.size(); ++index) {
        const NodeRankResourceId registry_id = selected_nodes[index];
        try {
            const network::attribute::NodeAttribute& definition =
                network.node_attributes().at(registry_id);
            if (definition.spec().kind !=
                network::attribute::AttributeKind::resource) {
                throw NodeRankException(
                    NodeRankErrorCode::invalid_node_resource_selection,
                    NodeRankOperation::prepare,
                    "Selected node attribute is not a resource",
                    index,
                    registry_id);
            }
            const network::attribute::NodeAttributeBinding binding =
                definition.bind(network.graph());
            node_resources.push_back(
                PreparedNodeRanker::NodeResourceBinding{
                    registry_id,
                    binding.value_id,
                    &definition});
        } catch (const NodeRankException&) {
            throw;
        } catch (const std::exception& error) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_node_resource_selection,
                NodeRankOperation::prepare,
                std::string("Cannot bind selected node resource: ") +
                    error.what(),
                index,
                registry_id);
        }
    }

    std::vector<NodeRankResourceId> selected_links;
    if (selection_.link_resources.has_value()) {
        selected_links = *selection_.link_resources;
    } else {
        network::AttributeSelection selection;
        selection.kinds = std::vector<network::attribute::AttributeKind>{
            network::attribute::AttributeKind::resource};
        selected_links = network.select_link_attributes(selection);
    }

    std::vector<PreparedNodeRanker::LinkResourceBinding> link_resources;
    link_resources.reserve(selected_links.size());
    for (std::size_t index = 0U; index < selected_links.size(); ++index) {
        const NodeRankResourceId registry_id = selected_links[index];
        try {
            const network::attribute::LinkAttribute& definition =
                network.link_attributes().at(registry_id);
            if (definition.spec().kind !=
                network::attribute::AttributeKind::resource) {
                throw NodeRankException(
                    NodeRankErrorCode::invalid_link_resource_selection,
                    NodeRankOperation::prepare,
                    "Selected link attribute is not a resource",
                    index,
                    registry_id);
            }
            const network::attribute::LinkAttributeBinding binding =
                definition.bind(network.graph());
            link_resources.push_back(
                PreparedNodeRanker::LinkResourceBinding{
                    registry_id,
                    binding.value_id,
                    &definition});
        } catch (const NodeRankException&) {
            throw;
        } catch (const std::exception& error) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_link_resource_selection,
                NodeRankOperation::prepare,
                std::string("Cannot bind selected link resource: ") +
                    error.what(),
                index,
                registry_id);
        }
    }

    return PreparedNodeRanker(
        network,
        std::move(node_resources),
        std::move(link_resources),
        parameters_);
}

PreparedNodeRanker::PreparedNodeRanker(
    const network::BaseNetwork& network,
    std::vector<NodeResourceBinding> node_resources,
    std::vector<LinkResourceBinding> link_resources,
    NodeRankParameters parameters)
    : network_(&network),
      graph_identity_(&network.graph()),
      graph_registry_identity_(&network.graph().attribute_registry()),
      node_registry_identity_(&network.node_attributes()),
      link_registry_identity_(&network.link_attributes()),
      node_resources_(std::move(node_resources)),
      link_resources_(std::move(link_resources)),
      parameters_(parameters),
      graph_read_mutex_(std::make_shared<std::mutex>())
{
    node_resource_ids_.reserve(node_resources_.size());
    for (const NodeResourceBinding& resource : node_resources_) {
        node_resource_ids_.push_back(resource.registry_id);
    }
    link_resource_ids_.reserve(link_resources_.size());
    for (const LinkResourceBinding& resource : link_resources_) {
        link_resource_ids_.push_back(resource.registry_id);
    }
}

const network::BaseNetwork& PreparedNodeRanker::checked_network() const
{
    if (network_ == nullptr || graph_identity_ == nullptr ||
        graph_registry_identity_ == nullptr ||
        node_registry_identity_ == nullptr ||
        link_registry_identity_ == nullptr || graph_read_mutex_ == nullptr) {
        throw NodeRankException(
            NodeRankErrorCode::invalid_prepared_state,
            NodeRankOperation::validate_prepared,
            "Node ranker is not bound to a network");
    }
    if (&network_->graph() != graph_identity_ ||
        &network_->graph().attribute_registry() != graph_registry_identity_ ||
        &network_->node_attributes() != node_registry_identity_ ||
        &network_->link_attributes() != link_registry_identity_) {
        throw NodeRankException(
            NodeRankErrorCode::invalid_prepared_state,
            NodeRankOperation::validate_prepared,
            "Prepared node ranker network identity changed");
    }

    for (std::size_t index = 0U; index < node_resources_.size(); ++index) {
        const NodeResourceBinding& resource = node_resources_[index];
        try {
            if (&network_->node_attributes().at(resource.registry_id) !=
                resource.definition_identity) {
                throw NodeRankException(
                    NodeRankErrorCode::invalid_prepared_state,
                    NodeRankOperation::validate_prepared,
                    "Prepared node resource definition changed",
                    index,
                    resource.registry_id);
            }
        } catch (const NodeRankException&) {
            throw;
        } catch (const std::exception& error) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_prepared_state,
                NodeRankOperation::validate_prepared,
                std::string("Prepared node resource is no longer valid: ") +
                    error.what(),
                index,
                resource.registry_id);
        }
    }
    for (std::size_t index = 0U; index < link_resources_.size(); ++index) {
        const LinkResourceBinding& resource = link_resources_[index];
        try {
            if (&network_->link_attributes().at(resource.registry_id) !=
                resource.definition_identity) {
                throw NodeRankException(
                    NodeRankErrorCode::invalid_prepared_state,
                    NodeRankOperation::validate_prepared,
                    "Prepared link resource definition changed",
                    index,
                    resource.registry_id);
            }
        } catch (const NodeRankException&) {
            throw;
        } catch (const std::exception& error) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_prepared_state,
                NodeRankOperation::validate_prepared,
                std::string("Prepared link resource is no longer valid: ") +
                    error.what(),
                index,
                resource.registry_id);
        }
    }
    return *network_;
}

NodeRanking PreparedNodeRanker::rank(
    NodeRankMethod method,
    NodeRankOptions options) const
{
    switch (method) {
    case NodeRankMethod::order:
        return rank_order(options);
    case NodeRankMethod::random:
        throw NodeRankException(
            NodeRankErrorCode::random_stream_required,
            NodeRankOperation::randomize,
            "Random node ranking requires an explicit NumpyRandomState");
    case NodeRankMethod::ffd:
        return rank_ffd(options);
    case NodeRankMethod::nrm:
        return rank_nrm(options);
    case NodeRankMethod::nea:
        return rank_nea(options);
    case NodeRankMethod::grc:
        return rank_grc(options);
    case NodeRankMethod::rw:
        return rank_rw(options);
    case NodeRankMethod::nps:
        return rank_nps(options);
    }
    throw NodeRankException(
        NodeRankErrorCode::unsupported_method,
        NodeRankOperation::resolve_method,
        "Unsupported typed node rank method");
}

NodeRanking PreparedNodeRanker::rank_order(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    static_cast<void>(options);
    const std::size_t node_count = network.live_num_nodes();
    if (node_count == 0U) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::reduce,
            "Order node ranking is undefined for an empty graph");
    }
    const double value = 1.0 / static_cast<double>(node_count);
    NodeRanking result;
    result.reserve(node_count);
    for (std::size_t index = 0U; index < node_count; ++index) {
        result.push_back(NodeRankEntry{
            static_cast<Vertex>(index),
            NodeRankValueKind::scalar,
            value,
            0.0});
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_random(
    NumpyRandomState& random,
    NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    const std::size_t node_count = network.live_num_nodes();
    std::vector<Vertex> shuffled(node_count, 0U);
    std::iota(shuffled.begin(), shuffled.end(), Vertex{0U});
    random.shuffle(shuffled);

    validate_scalar_ranking_length(network, shuffled.size());
    NodeRanking result;
    result.reserve(node_count);
    for (std::size_t index = 0U; index < node_count; ++index) {
        result.push_back(NodeRankEntry{
            static_cast<Vertex>(index),
            NodeRankValueKind::scalar,
            static_cast<double>(shuffled[index]),
            0.0});
    }
    if (options.sort) {
        python_scalar_descending_sort(result);
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_ffd(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    const NodeResourceSums sums = gather_node_resource_sums(
        network.graph(), node_resources_, options.workers);
    NodeRanking result = make_scalar_ranking(network, sums);
    if (options.sort) {
        python_scalar_descending_sort(result);
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_nrm(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    const NodeResourceSums node_sums = gather_node_resource_sums(
        network.graph(), node_resources_, options.workers);
    const std::size_t node_count = network.live_num_nodes();

    std::vector<std::vector<double>> link_rows;
    if (!link_resource_ids_.empty()) {
        const std::lock_guard<std::mutex> lock(*graph_read_mutex_);
        link_rows = network::get_aggregation_attrs_data(
            network,
            link_resource_ids_,
            network::attribute::LinkAggregation::sum,
            false,
            options.workers);
    }
    const std::vector<double> link_sums = reduce_link_aggregation_rows(
        link_rows, node_count, options.workers);
    if (node_sums.size() != node_count) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::reduce,
            "Node resource columns do not cover every live node");
    }

    NodeRanking result(node_count);
    const auto multiply = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            result[index].value =
                node_sums.as_double(index) * link_sums[index];
        }
    };
    parallel_blocks(node_count, options.workers, multiply);
    finalize_scalar_ranking(network, result);
    if (options.sort) {
        python_scalar_descending_sort(result);
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_nea(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    const Graph& graph = network.graph();
    const NodeResourceSums sums = gather_node_resource_sums(
        graph, node_resources_, options.workers);
    const std::size_t node_count = graph.num_nodes();
    if (sums.size() != node_count) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::reduce,
            "Node resource columns do not cover every live node");
    }

    NodeRanking result(node_count);
    const auto multiply = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const std::uint64_t degree = static_cast<std::uint64_t>(
                graph.degree(static_cast<Vertex>(index)));
            if (sums.double_lane) {
                result[index].value = sums.doubles[index] *
                    static_cast<double>(degree);
            } else {
                const std::uint64_t product =
                    static_cast<std::uint64_t>(sums.integers[index]) * degree;
                result[index].value = static_cast<double>(
                    uint64_bits_as_int64(product));
            }
        }
    };
    parallel_blocks(node_count, options.workers, multiply);
    finalize_scalar_ranking(network, result);
    if (options.sort) {
        python_scalar_descending_sort(result);
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_grc(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    if (link_resource_ids_.empty()) {
        throw NodeRankException(
            NodeRankErrorCode::empty_link_resource_selection,
            NodeRankOperation::build_matrix,
            "GRC node ranking requires at least one link resource");
    }
    const NodeResourceSums node_sums = gather_node_resource_sums(
        network.graph(), node_resources_, options.workers);
    const std::size_t node_count = network.live_num_nodes();
    if (node_sums.size() != node_count) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::reduce,
            "Node resource columns do not cover every live node");
    }

    const double total = node_sum_total(node_sums);
    std::vector<double> capacity(node_count, 0.0);
    for (std::size_t index = 0U; index < node_count; ++index) {
        capacity[index] = node_sums.as_double(index) / total;
    }

    std::vector<DistanceMatrix> matrices;
    {
        const std::lock_guard<std::mutex> lock(*graph_read_mutex_);
        matrices = network::get_adjacency_attrs_data(
            network,
            link_resource_ids_,
            true,
            options.workers);
    }
    const std::vector<double> matrix = average_matrices(
        matrices, node_count, options.workers);

    const double sigma = parameters_.grc.sigma;
    const double damping = parameters_.grc.damping;
    std::vector<double> teleport(node_count, 0.0);
    for (std::size_t index = 0U; index < node_count; ++index) {
        teleport[index] = (1.0 - damping) * capacity[index];
    }
    std::vector<double> current = capacity;
    std::vector<double> next(node_count, 0.0);
    double delta = std::numeric_limits<double>::infinity();
    std::size_t iterations = 0U;
    while (delta >= sigma) {
        if (options.max_iterations.has_value() &&
            iterations >= *options.max_iterations) {
            throw NodeRankException(
                NodeRankErrorCode::iteration_limit_reached,
                NodeRankOperation::iterate,
                "GRC node ranking reached the caller iteration limit");
        }
        const auto multiply = [&](std::size_t begin, std::size_t end) {
            for (std::size_t destination = begin;
                 destination < end;
                 ++destination) {
                double dot = 0.0;
                for (std::size_t source = 0U;
                     source < node_count;
                     ++source) {
                    dot = std::fma(
                        current[source],
                        matrix[source * node_count + destination],
                        dot);
                }
                const double propagated = damping * dot;
                next[destination] = teleport[destination] + propagated;
            }
        };
        parallel_blocks(
            node_count,
            options.workers,
            multiply,
            quadratic_block_grain(node_count));
        delta = l2_delta(next, current);
        current.swap(next);
        ++iterations;
    }

    NodeRanking result = make_scalar_ranking(network, current);
    if (options.sort) {
        python_scalar_descending_sort(result);
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_rw(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    if (link_resource_ids_.empty()) {
        throw NodeRankException(
            NodeRankErrorCode::empty_link_resource_selection,
            NodeRankOperation::build_matrix,
            "RW node ranking requires at least one link resource");
    }
    const NodeResourceSums node_sums = gather_node_resource_sums(
        network.graph(), node_resources_, options.workers);
    const std::size_t node_count = network.live_num_nodes();
    if (node_sums.size() != node_count) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::reduce,
            "Node resource columns do not cover every live node");
    }

    std::vector<DistanceMatrix> matrices;
    {
        const std::lock_guard<std::mutex> lock(*graph_read_mutex_);
        matrices = network::get_adjacency_attrs_data(
            network,
            link_resource_ids_,
            false,
            options.workers);
    }
    const std::vector<double> matrix = average_matrices(
        matrices, node_count, options.workers);

    std::vector<double> bandwidth(node_count, 0.0);
    const auto reduce_columns = [&](std::size_t begin, std::size_t end) {
        for (std::size_t column = begin; column < end; ++column) {
            double sum = 0.0;
            for (std::size_t row = 0U; row < node_count; ++row) {
                sum += matrix[row * node_count + column];
            }
            bandwidth[column] = sum;
        }
    };
    parallel_blocks(
        node_count,
        options.workers,
        reduce_columns,
        quadratic_block_grain(node_count));

    std::vector<double> capacity(node_count, 0.0);
    const auto build_capacity = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            capacity[index] =
                node_sums.as_double(index) * bandwidth[index];
        }
    };
    parallel_blocks(node_count, options.workers, build_capacity);
    const double denominator = sequential_sum(capacity) + 1e-9;
    std::vector<double> base(node_count, 0.0);
    for (std::size_t index = 0U; index < node_count; ++index) {
        base[index] = capacity[index] / denominator;
    }

    if (node_count == 0U) {
        throw NodeRankException(
            NodeRankErrorCode::invalid_matrix_shape,
            NodeRankOperation::build_matrix,
            "RW topology adjacency is undefined for an empty graph");
    }

    CSRMatrix topology;
    {
        const std::lock_guard<std::mutex> lock(*graph_read_mutex_);
        topology = network.adjacency_matrix();
    }
    if (topology.rows != node_count || topology.cols != node_count ||
        topology.row_ptr.size() != node_count + 1U ||
        topology.col_idx.size() != topology.values.size()) {
        throw NodeRankException(
            NodeRankErrorCode::invalid_matrix_shape,
            NodeRankOperation::build_matrix,
            "RW topology CSR shape is invalid");
    }
    const std::size_t structural_nonzero = static_cast<std::size_t>(
        std::count_if(
            topology.values.begin(),
            topology.values.end(),
            [](double value) { return value != 0.0; }));
    if (structural_nonzero != topology.values.size()) {
        throw NodeRankException(
            NodeRankErrorCode::sparse_assignment_mismatch,
            NodeRankOperation::build_matrix,
            "RW topology contains explicit zero sparse entries");
    }

    if (node_count > std::numeric_limits<std::size_t>::max() / node_count) {
        throw NodeRankException(
            NodeRankErrorCode::invalid_matrix_shape,
            NodeRankOperation::build_matrix,
            "RW matrix extent overflows size_t");
    }
    std::vector<double> flow(node_count * node_count, 0.0);
    for (std::size_t row = 0U; row < node_count; ++row) {
        if (topology.row_ptr[row] > topology.row_ptr[row + 1U] ||
            topology.row_ptr[row + 1U] > topology.values.size()) {
            throw NodeRankException(
                NodeRankErrorCode::invalid_matrix_shape,
                NodeRankOperation::build_matrix,
                "RW topology CSR row offsets are invalid",
                row,
                std::nullopt,
                static_cast<Vertex>(row));
        }
        double absolute_sum = 0.0;
        for (std::size_t offset = topology.row_ptr[row];
             offset < topology.row_ptr[row + 1U];
             ++offset) {
            const std::size_t column = topology.col_idx[offset];
            if (column >= node_count) {
                throw NodeRankException(
                    NodeRankErrorCode::invalid_matrix_shape,
                    NodeRankOperation::build_matrix,
                    "RW topology CSR column is outside the graph",
                    offset,
                    std::nullopt,
                    static_cast<Vertex>(row));
            }
            const double value = capacity[column];
            flow[row * node_count + column] = value;
            absolute_sum += std::abs(value);
        }
        if (absolute_sum != 0.0) {
            for (std::size_t offset = topology.row_ptr[row];
                 offset < topology.row_ptr[row + 1U];
                 ++offset) {
                const std::size_t column = topology.col_idx[offset];
                flow[row * node_count + column] /= absolute_sum;
            }
        }
    }

    const double jump = parameters_.rw.jump_probability;
    const double forwarding = parameters_.rw.forwarding_probability;
    std::vector<double> transition(node_count * node_count, 0.0);
    const auto build_transition = [&](std::size_t begin, std::size_t end) {
        for (std::size_t destination = begin;
             destination < end;
             ++destination) {
            const double jump_value = base[destination] * jump;
            for (std::size_t source = 0U; source < node_count; ++source) {
                const double follow_value =
                    flow[source * node_count + destination] * forwarding;
                transition[destination * node_count + source] =
                    jump_value + follow_value;
            }
        }
    };
    parallel_blocks(
        node_count,
        options.workers,
        build_transition,
        quadratic_block_grain(node_count));

    std::vector<double> current = base;
    std::vector<double> next(node_count, 0.0);
    double delta = std::numeric_limits<double>::infinity();
    std::size_t iterations = 0U;
    while (delta >= parameters_.rw.sigma) {
        if (options.max_iterations.has_value() &&
            iterations >= *options.max_iterations) {
            throw NodeRankException(
                NodeRankErrorCode::iteration_limit_reached,
                NodeRankOperation::iterate,
                "RW node ranking reached the caller iteration limit");
        }
        const auto multiply = [&](std::size_t begin, std::size_t end) {
            for (std::size_t destination = begin;
                 destination < end;
                 ++destination) {
                double dot = 0.0;
                for (std::size_t source = 0U;
                     source < node_count;
                     ++source) {
                    dot = std::fma(
                        transition[destination * node_count + source],
                        current[source],
                        dot);
                }
                next[destination] = dot;
            }
        };
        parallel_blocks(
            node_count,
            options.workers,
            multiply,
            quadratic_block_grain(node_count));
        delta = l2_delta(next, current);
        current.swap(next);
        ++iterations;
    }

    NodeRanking result = make_scalar_ranking(network, current);
    if (options.sort) {
        python_scalar_descending_sort(result);
    }
    return result;
}

NodeRanking PreparedNodeRanker::rank_nps(NodeRankOptions options) const
{
    const network::BaseNetwork& network = checked_network();
    NodeRankOptions nrm_options = options;
    nrm_options.sort = false;
    NodeRanking nrm = rank_nrm(nrm_options);
    const Graph& graph = network.graph();
    const std::size_t node_count = graph.num_nodes();
    if (node_count == 0U) {
        throw NodeRankException(
            NodeRankErrorCode::ranking_length_mismatch,
            NodeRankOperation::traverse,
            "NPS root selection is undefined for an empty graph");
    }

    Vertex root = 0U;
    std::size_t maximum_neighbors = graph.neighbors_fast(root).size();
    for (std::size_t index = 1U; index < node_count; ++index) {
        const Vertex node = static_cast<Vertex>(index);
        const std::size_t neighbor_count = graph.neighbors_fast(node).size();
        if (neighbor_count > maximum_neighbors) {
            root = node;
            maximum_neighbors = neighbor_count;
        }
    }

    const nx::SingleSourceDijkstraPathLengths distances =
        nx::single_source_dijkstra_path_length(graph, root);
    NodeRanking result;
    result.reserve(distances.size());
    for (const auto& item : distances.items()) {
        const Vertex node = item.first;
        if (node >= nrm.size()) {
            throw NodeRankException(
                NodeRankErrorCode::ranking_length_mismatch,
                NodeRankOperation::traverse,
                "NPS distance result references a node without NRM score",
                invalid_node_rank_input_index,
                std::nullopt,
                node);
        }
        result.push_back(NodeRankEntry{
            node,
            NodeRankValueKind::proximity,
            nrm[node].value,
            item.second});
    }
    if (options.sort) {
        python_nps_ascending_sort(result);
    }
    return result;
}

const std::vector<NodeRankResourceId>&
PreparedNodeRanker::node_resource_ids() const noexcept
{
    return node_resource_ids_;
}

const std::vector<NodeRankResourceId>&
PreparedNodeRanker::link_resource_ids() const noexcept
{
    return link_resource_ids_;
}

}  // namespace virne::solver::rank
