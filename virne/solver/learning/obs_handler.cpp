#include "obs_handler.h"

#include "obs_handler_detail.h"
#include "nx/shortest_paths.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace virne::solver::learning::rl_core {
namespace detail {


using AttributeBenchmarkMap =
    network::attribute::AttributeBenchmarkMap;
using AttributeKind = network::attribute::AttributeKind;
using AttributeRegistryId = network::attribute::AttributeRegistryId;
using LinkAggregation = network::attribute::LinkAggregation;

[[noreturn]] void throw_feature(
    FeatureConstructorErrorCode code,
    std::string message) {
    throw FeatureConstructorException(code, std::move(message));
}

void require_workers(std::size_t workers) {
    if (workers == 0U) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_worker_count,
            "Feature construction requires at least one worker.");
    }
}

std::vector<AttributeRegistryId> select_node_ids(
    const network::BaseNetwork& network,
    const ObservationHandler::AttributeKinds& kinds) {
    const auto ids = network.select_node_attributes(
        network::AttributeSelection{kinds, std::nullopt});
    if (ids.empty()) {
        throw_feature(
            FeatureConstructorErrorCode::missing_attribute,
            "The selected node attribute set is empty.");
    }
    return ids;
}

std::vector<AttributeRegistryId> select_link_ids(
    const network::BaseNetwork& network,
    const ObservationHandler::AttributeKinds& kinds) {
    const auto ids = network.select_link_attributes(
        network::AttributeSelection{kinds, std::nullopt});
    if (ids.empty()) {
        throw_feature(
            FeatureConstructorErrorCode::missing_attribute,
            "The selected link attribute set is empty.");
    }
    return ids;
}

template <typename Registry>
std::vector<double> resolve_benchmarks(
    const Registry& registry,
    const std::vector<AttributeRegistryId>& ids,
    const AttributeBenchmarkMap& benchmarks) {
    std::vector<double> result;
    result.reserve(ids.size());
    for (const AttributeRegistryId id : ids) {
        const auto& spec = registry.at(id).spec();
        const std::string_view key =
            spec.kind == AttributeKind::extrema
            ? (spec.originator.has_value()
                   ? std::string_view(*spec.originator)
                   : std::string_view{})
            : std::string_view(spec.name);
        if (key.empty()) {
            throw_feature(
                FeatureConstructorErrorCode::missing_benchmark,
                "An extrema attribute has no originator benchmark name.");
        }
        const double* benchmark = benchmarks.find(key);
        if (benchmark == nullptr) {
            throw_feature(
                FeatureConstructorErrorCode::missing_benchmark,
                "A selected attribute benchmark is missing.");
        }
        result.push_back(*benchmark);
    }
    return result;
}

float numeric_f32(const AttrValue& value) {
    try {
        return static_cast<float>(attr_to_double(value));
    } catch (const std::exception&) {
        throw_feature(
            FeatureConstructorErrorCode::non_numeric_attribute,
            "A selected graph attribute is not numeric.");
    }
}

double numeric_f64(const AttrValue& value) {
    try {
        return attr_to_double(value);
    } catch (const std::exception&) {
        throw_feature(
            FeatureConstructorErrorCode::non_numeric_attribute,
            "A selected graph attribute is not numeric.");
    }
}

float normalize_f32(float value, double benchmark) {
    // NumPy keeps these array operations in float32. Attribute benchmarks are
    // themselves reductions of float32 matrices, so narrowing once here is the
    // exact scalar equivalent of its broadcasting rule.
    return value / static_cast<float>(benchmark);
}

DenseMatrixF32 node_attrs_obs_by_id(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<double>* benchmarks,
    std::size_t workers) {
    require_workers(workers);
    const auto rows = network::get_node_attrs_data(network, ids, workers);
    const std::size_t nodes = network.num_nodes();
    DenseMatrixF32 result{nodes, ids.size(), {}};
    result.values.resize(nodes * ids.size());
    if (benchmarks != nullptr && benchmarks->size() != ids.size()) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "Node benchmark count does not match the selected attributes.");
    }
    for (std::size_t attribute = 0U; attribute < rows.size(); ++attribute) {
        if (rows[attribute].size() != nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "Node attribute data has an invalid shape.");
        }
        for (std::size_t node = 0U; node < nodes; ++node) {
            float value = numeric_f32(rows[attribute][node]);
            if (benchmarks != nullptr) {
                value = normalize_f32(value, (*benchmarks)[attribute]);
            }
            result.values[node * ids.size() + attribute] = value;
        }
    }
    return result;
}

DenseMatrixF32 link_attrs_obs_by_id(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<double>* benchmarks,
    std::size_t workers) {
    require_workers(workers);
    const auto rows = network::get_link_attrs_data(network, ids, workers);
    const std::size_t links = network.num_links();
    DenseMatrixF32 result{links * 2U, ids.size(), {}};
    result.values.resize(result.rows * result.columns);
    if (benchmarks != nullptr && benchmarks->size() != ids.size()) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "Link benchmark count does not match the selected attributes.");
    }
    for (std::size_t attribute = 0U; attribute < rows.size(); ++attribute) {
        if (rows[attribute].size() != links) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "Link attribute data has an invalid shape.");
        }
        for (std::size_t link = 0U; link < links; ++link) {
            float value = numeric_f32(rows[attribute][link]);
            if (benchmarks != nullptr) {
                value = normalize_f32(value, (*benchmarks)[attribute]);
            }
            result.values[link * ids.size() + attribute] = value;
            result.values[(link + links) * ids.size() + attribute] = value;
        }
    }
    return result;
}

DenseMatrixF32 link_aggr_obs_by_id(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    LinkAggregation aggregation,
    const std::vector<double>* benchmarks,
    std::size_t workers) {
    require_workers(workers);
    const auto rows = network::get_aggregation_attrs_data(
        network, ids, aggregation, false, workers);
    const std::size_t nodes = network.num_nodes();
    DenseMatrixF32 result{nodes, ids.size(), {}};
    result.values.resize(nodes * ids.size());
    if (benchmarks != nullptr && benchmarks->size() != ids.size()) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "Aggregated-link benchmark count does not match the attributes.");
    }
    for (std::size_t attribute = 0U; attribute < rows.size(); ++attribute) {
        if (rows[attribute].size() != nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "Aggregated link data has an invalid shape.");
        }
        for (std::size_t node = 0U; node < nodes; ++node) {
            float value = static_cast<float>(rows[attribute][node]);
            if (benchmarks != nullptr) {
                value = normalize_f32(value, (*benchmarks)[attribute]);
            }
            result.values[node * ids.size() + attribute] = value;
        }
    }
    return result;
}

DenseMatrixF32 concatenate_columns(
    std::size_t rows,
    const std::vector<const DenseMatrixF32*>& matrices) {
    std::size_t columns = 0U;
    for (const DenseMatrixF32* matrix : matrices) {
        if (matrix == nullptr || matrix->rows != rows ||
            matrix->values.size() != matrix->rows * matrix->columns) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "Cannot concatenate feature matrices with incompatible shapes.");
        }
        columns += matrix->columns;
    }
    DenseMatrixF32 result{rows, columns, {}};
    result.values.resize(rows * columns);
    for (std::size_t row = 0U; row < rows; ++row) {
        std::size_t output = row * columns;
        for (const DenseMatrixF32* matrix : matrices) {
            const auto begin = matrix->values.begin() +
                static_cast<std::ptrdiff_t>(row * matrix->columns);
            std::copy_n(begin, matrix->columns, result.values.begin() +
                static_cast<std::ptrdiff_t>(output));
            output += matrix->columns;
        }
    }
    return result;
}

DenseVectorF32 concatenate_vectors(
    const std::vector<const DenseVectorF32*>& vectors) {
    std::size_t size = 0U;
    for (const DenseVectorF32* vector : vectors) {
        if (vector == nullptr) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "Cannot concatenate a null feature vector.");
        }
        size += vector->values.size();
    }
    DenseVectorF32 result;
    result.values.reserve(size);
    for (const DenseVectorF32* vector : vectors) {
        result.values.insert(
            result.values.end(), vector->values.begin(), vector->values.end());
    }
    return result;
}

DenseMatrixF32 empty_matrix(std::size_t rows) {
    return DenseMatrixF32{rows, 0U, {}};
}

DenseMatrixF32 constant_column(std::size_t rows, float value) {
    return DenseMatrixF32{rows, 1U, std::vector<float>(rows, value)};
}

const network::MetricColumn& require_metric(
    const std::optional<network::MetricColumn>& metric,
    std::size_t rows) {
    if (!metric.has_value()) {
        throw_feature(
            FeatureConstructorErrorCode::missing_metric,
            "A requested topological metric was not prepared.");
    }
    if (metric->values.size() != rows) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "A topological metric has an invalid row count.");
    }
    return *metric;
}

DenseVectorF32 v_node_demand_by_id(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<AttrId>& value_ids,
    const std::vector<double>* benchmarks) {
    if (v_node_id >= v_net.num_nodes()) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_current_virtual_node,
            "The current virtual-node ID is out of range.");
    }
    if (ids.size() != value_ids.size() ||
        (benchmarks != nullptr && benchmarks->size() != ids.size())) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "Virtual-node attribute bindings have an invalid shape.");
    }
    DenseVectorF32 result;
    result.values.resize(ids.size());
    const auto& attributes = v_net.graph().node_attrs(
        static_cast<Vertex>(v_node_id));
    for (std::size_t index = 0U; index < ids.size(); ++index) {
        const AttrValue* raw = attributes.find(value_ids[index]);
        if (raw == nullptr) {
            throw_feature(
                FeatureConstructorErrorCode::missing_attribute,
                "A virtual-node attribute value is missing.");
        }
        // Python divides the scalar before constructing its float32 array.
        const double value = numeric_f64(*raw);
        result.values[index] = benchmarks == nullptr
            ? static_cast<float>(value)
            : static_cast<float>(value / (*benchmarks)[index]);
    }
    return result;
}

DenseVectorF32 v_node_aggr_by_id(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    LinkAggregation aggregation,
    const std::vector<AttributeRegistryId>& ids,
    const std::vector<AttrId>& value_ids,
    const std::vector<double>* benchmarks) {
    if (v_node_id >= v_net.num_nodes()) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_current_virtual_node,
            "The current virtual-node ID is out of range.");
    }
    const Graph& graph = v_net.graph();
    const Vertex node = static_cast<Vertex>(v_node_id);
    const auto neighbors = graph.neighbors(node);
    std::vector<Vertex> adjacent(neighbors.first, neighbors.second);
    if (adjacent.empty()) {
        throw_feature(
            FeatureConstructorErrorCode::empty_incident_links,
            "Cannot aggregate link demands for an isolated virtual node.");
    }
    if (ids.size() != value_ids.size() ||
        (benchmarks != nullptr && benchmarks->size() != ids.size())) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "Virtual-link attribute bindings have an invalid shape.");
    }

    DenseVectorF32 result;
    result.values.resize(ids.size());
    for (std::size_t attribute = 0U; attribute < ids.size(); ++attribute) {
        double value = 0.0;
        bool first = true;
        for (const Vertex neighbor : adjacent) {
            const Edge edge = graph.edge(neighbor, node);
            const AttrValue* raw = graph.edge_attrs(edge).find(value_ids[attribute]);
            if (raw == nullptr) {
                throw_feature(
                    FeatureConstructorErrorCode::missing_attribute,
                    "A virtual-link attribute value is missing.");
            }
            const double candidate = numeric_f64(*raw);
            if (first) {
                value = candidate;
                first = false;
            } else if (aggregation == LinkAggregation::maximum) {
                value = std::max(value, candidate);
            } else if (aggregation == LinkAggregation::minimum) {
                value = std::min(value, candidate);
            } else {
                value += candidate;
            }
        }
        if (aggregation == LinkAggregation::mean) {
            value /= static_cast<double>(adjacent.size());
        }
        float narrowed = static_cast<float>(value);
        if (benchmarks != nullptr) {
            narrowed = normalize_f32(narrowed, (*benchmarks)[attribute]);
        }
        result.values[attribute] = narrowed;
    }
    return result;
}

std::vector<AttrId> resolve_node_value_ids(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids) {
    std::vector<AttrId> result;
    result.reserve(ids.size());
    for (const AttributeRegistryId id : ids) {
        const auto& name = network.node_attributes().entries()[id].name;
        const auto binding = network.bind_node_attribute(name);
        if (!binding.has_value()) {
            throw_feature(
                FeatureConstructorErrorCode::missing_attribute,
                "A selected node attribute could not be bound.");
        }
        if (binding->graph_identity != &network.graph()) {
            throw_feature(
                FeatureConstructorErrorCode::graph_identity_mismatch,
                "A node attribute binding belongs to another graph.");
        }
        result.push_back(binding->value_id);
    }
    return result;
}

std::vector<AttrId> resolve_link_value_ids(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids) {
    std::vector<AttrId> result;
    result.reserve(ids.size());
    for (const AttributeRegistryId id : ids) {
        const auto& name = network.link_attributes().entries()[id].name;
        const auto binding = network.bind_link_attribute(name);
        if (!binding.has_value()) {
            throw_feature(
                FeatureConstructorErrorCode::missing_attribute,
                "A selected link attribute could not be bound.");
        }
        if (binding->graph_identity != &network.graph()) {
            throw_feature(
                FeatureConstructorErrorCode::graph_identity_mismatch,
                "A link attribute binding belongs to another graph.");
        }
        result.push_back(binding->value_id);
    }
    return result;
}


std::vector<double> resolve_node_benchmarks(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const AttributeBenchmarkMap& benchmarks) {
    return resolve_benchmarks(network.node_attributes(), ids, benchmarks);
}

std::vector<double> resolve_link_benchmarks(
    const network::BaseNetwork& network,
    const std::vector<AttributeRegistryId>& ids,
    const AttributeBenchmarkMap& benchmarks) {
    return resolve_benchmarks(network.link_attributes(), ids, benchmarks);
}

}  // namespace detail

using namespace detail;

FeatureConstructorException::FeatureConstructorException(
    FeatureConstructorErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

DenseVectorI32 ObservationHandler::get_node_order_obs(
    const network::BaseNetwork& network) {
    DenseVectorI32 result;
    result.values.resize(network.num_nodes());
    for (std::size_t node = 0U; node < result.values.size(); ++node) {
        result.values[node] = static_cast<std::int32_t>(node);
    }
    return result;
}

DenseMatrixF32 ObservationHandler::get_node_degree_obs(
    const network::BaseNetwork& network,
    std::optional<double> degree_benchmark,
    std::size_t workers) {
    require_workers(workers);
    const std::size_t nodes = network.num_nodes();
    DenseMatrixF32 result{nodes, 1U, {}};
    result.values.resize(nodes);
    const float benchmark = degree_benchmark.has_value()
        ? static_cast<float>(*degree_benchmark)
        : 1.0F;
    for (std::size_t node = 0U; node < nodes; ++node) {
        float value = static_cast<float>(
            network.graph().degree(static_cast<Vertex>(node)));
        if (degree_benchmark.has_value()) {
            value /= benchmark;
        }
        result.values[node] = value;
    }
    return result;
}

DenseMatrixF32 ObservationHandler::get_node_topological_metrics(
    const network::BaseNetwork& network,
    const network::TopologicalMetrics& metrics,
    TopologicalMetricSelection selection) {
    const std::size_t rows = network.num_nodes();
    std::vector<const network::MetricColumn*> columns;
    if (selection.degree) {
        columns.push_back(&require_metric(metrics.node_degree_centrality, rows));
    }
    if (selection.closeness) {
        columns.push_back(&require_metric(metrics.node_closeness_centrality, rows));
    }
    if (selection.eigenvector) {
        columns.push_back(&require_metric(metrics.node_eigenvector_centrality, rows));
    }
    if (selection.betweenness) {
        columns.push_back(&require_metric(metrics.node_betweenness_centrality, rows));
    }
    DenseMatrixF32 result{rows, columns.size(), {}};
    result.values.resize(rows * columns.size());
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t column = 0U; column < columns.size(); ++column) {
            result.values[row * columns.size() + column] =
                columns[column]->values[row];
        }
    }
    return result;
}

DenseMatrixF32 ObservationHandler::get_node_attrs_obs(
    const network::BaseNetwork& network,
    const AttributeKinds& kinds,
    const AttributeBenchmarkMap* benchmarks,
    std::size_t workers) {
    const auto ids = select_node_ids(network, kinds);
    const auto resolved = benchmarks == nullptr
        ? std::vector<double>{}
        : resolve_node_benchmarks(network, ids, *benchmarks);
    return node_attrs_obs_by_id(
        network, ids, benchmarks == nullptr ? nullptr : &resolved, workers);
}

DenseMatrixF32 ObservationHandler::get_link_attrs_obs(
    const network::BaseNetwork& network,
    const AttributeKinds& kinds,
    const AttributeBenchmarkMap* benchmarks,
    std::size_t workers) {
    const auto ids = select_link_ids(network, kinds);
    const auto resolved = benchmarks == nullptr
        ? std::vector<double>{}
        : resolve_link_benchmarks(network, ids, *benchmarks);
    return link_attrs_obs_by_id(
        network, ids, benchmarks == nullptr ? nullptr : &resolved, workers);
}

DenseMatrixF32 ObservationHandler::get_link_aggr_attrs_obs(
    const network::BaseNetwork& network,
    const AttributeKinds& kinds,
    LinkAggregation aggregation,
    const AttributeBenchmarkMap* benchmarks,
    std::size_t workers) {
    const auto ids = select_link_ids(network, kinds);
    const auto resolved = benchmarks == nullptr
        ? std::vector<double>{}
        : resolve_link_benchmarks(network, ids, *benchmarks);
    return link_aggr_obs_by_id(
        network,
        ids,
        aggregation,
        benchmarks == nullptr ? nullptr : &resolved,
        workers);
}

DenseVectorBool ObservationHandler::get_link_filters(
    const network::BaseNetwork& network,
    const std::vector<LinkResourceThreshold>& thresholds,
    std::size_t workers) {
    require_workers(workers);
    if (thresholds.empty()) {
        throw_feature(
            FeatureConstructorErrorCode::missing_attribute,
            "At least one link threshold is required.");
    }
    std::vector<AttributeRegistryId> ids;
    ids.reserve(thresholds.size());
    for (const auto& threshold : thresholds) {
        const auto binding = network.bind_link_attribute(threshold.name);
        if (!binding.has_value()) {
            throw_feature(
                FeatureConstructorErrorCode::missing_attribute,
                "A threshold link attribute is missing.");
        }
        ids.push_back(binding->registry_id);
    }
    const auto rows = network::get_link_attrs_data(network, ids, workers);
    const std::size_t links = network.num_links();
    DenseVectorBool result;
    result.values.assign(links * 2U, std::uint8_t{1});
    for (std::size_t attribute = 0U; attribute < rows.size(); ++attribute) {
        if (rows[attribute].size() != links) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "Threshold link attribute data has an invalid shape.");
        }
        for (std::size_t link = 0U; link < links; ++link) {
            if (numeric_f64(rows[attribute][link]) <
                thresholds[attribute].minimum) {
                result.values[link] = std::uint8_t{0};
                result.values[link + links] = std::uint8_t{0};
            }
        }
    }
    return result;
}

DenseMatrixI64 ObservationHandler::get_link_pair_obs(
    const network::BaseNetwork& network) {
    const Graph& graph = network.graph();
    const std::size_t links = network.num_links();
    DenseMatrixI64 result{links * 2U, 2U, {}};
    result.values.resize(result.rows * result.columns);
    std::size_t index = 0U;
    const auto edge_range = graph.edges();
    for (auto edge = edge_range.first; edge != edge_range.second; ++edge) {
        const auto endpoints = graph.edge_endpoints(graph.edge_id(*edge));
        result.values[index * 2U] = static_cast<std::int64_t>(endpoints.first);
        result.values[index * 2U + 1U] =
            static_cast<std::int64_t>(endpoints.second);
        result.values[(index + links) * 2U] =
            static_cast<std::int64_t>(endpoints.second);
        result.values[(index + links) * 2U + 1U] =
            static_cast<std::int64_t>(endpoints.first);
        ++index;
    }
    if (index != links) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_shape,
            "Graph link count changed while constructing edge pairs.");
    }
    return result;
}

DenseMatrixI64 ObservationHandler::get_link_index_obs(
    const network::BaseNetwork& network) {
    const DenseMatrixI64 pairs = get_link_pair_obs(network);
    DenseMatrixI64 result{2U, pairs.rows, {}};
    result.values.resize(2U * pairs.rows);
    for (std::size_t row = 0U; row < pairs.rows; ++row) {
        result.values[row] = pairs.values[row * 2U];
        result.values[pairs.rows + row] = pairs.values[row * 2U + 1U];
    }
    return result;
}

DenseMatrixF32 ObservationHandler::get_average_distance(
    const network::BaseNetwork& network,
    const core::NodeSlots& node_slots,
    bool normalization,
    std::size_t workers) {
    require_workers(workers);
    const std::size_t nodes = network.num_nodes();
    std::vector<double> distances(nodes, 0.0);
    if (!node_slots.empty()) {
        const auto all_pairs = nx::shortest_path_length(network.graph());
        for (std::size_t source_node = 0U; source_node < nodes; ++source_node) {
            double sum = 0.0;
            const auto& from_source =
                all_pairs.at(static_cast<Vertex>(source_node));
            for (const auto& slot : node_slots.entries()) {
                if (slot.value < 0 ||
                    static_cast<std::size_t>(slot.value) >= nodes) {
                    throw_feature(
                        FeatureConstructorErrorCode::invalid_shape,
                        "A physical-node slot is out of range.");
                }
                const Vertex target = static_cast<Vertex>(slot.value);
                const auto found = from_source.find(target);
                if (found == from_source.end()) {
                    throw_feature(
                        FeatureConstructorErrorCode::disconnected_network,
                        "A selected physical node is unreachable.");
                }
                sum += static_cast<double>(found->second);
            }
            distances[source_node] =
                sum / static_cast<double>(node_slots.size() + 1U);
        }
    }
    if (normalization && !distances.empty()) {
        const auto bounds =
            std::minmax_element(distances.begin(), distances.end());
        // Keep scalar bounds before mutating `distances`: minmax_element
        // returns iterators into this vector, and updating the minimum slot
        // would otherwise change the normalization of every later row.
        const double minimum = *bounds.first;
        const double maximum = *bounds.second;
        const double span = maximum - minimum;
        if (span != 0.0) {
            for (double& value : distances) {
                value = (value - minimum) / span;
            }
        } else {
            // Constant distances carry no relative position information.
            std::fill(distances.begin(), distances.end(), 0.0);
        }
    }
    DenseMatrixF32 result{nodes, 1U, {}};
    result.values.resize(nodes);
    std::transform(
        distances.begin(), distances.end(), result.values.begin(),
        [](double value) { return static_cast<float>(value); });
    return result;
}

DenseMatrixF32 ObservationHandler::get_p_net_nodes_status(
    const network::PhysicalNetwork& p_net,
    const network::VirtualNetwork& v_net,
    const core::NodeSlots& node_slots,
    std::optional<std::size_t> v_node_id) {
    const std::size_t p_nodes = p_net.num_nodes();
    const std::size_t columns = v_node_id.has_value() ? 2U : 1U;
    DenseMatrixF32 result{p_nodes, columns, {}};
    result.values.assign(p_nodes * columns, 0.0F);
    for (const auto& slot : node_slots.entries()) {
        if (slot.value < 0 ||
            static_cast<std::size_t>(slot.value) >= p_nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "A physical-node slot is out of range.");
        }
        result.values[static_cast<std::size_t>(slot.value) * columns] = 1.0F;
    }
    if (!v_node_id.has_value()) {
        return result;
    }
    if (*v_node_id >= v_net.num_nodes()) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_current_virtual_node,
            "The current virtual-node ID is out of range.");
    }
    const auto adjacent =
        v_net.graph().neighbors(static_cast<Vertex>(*v_node_id));
    for (auto neighbor = adjacent.first;
         neighbor != adjacent.second;
         ++neighbor) {
        const auto slot_id = node_slots.find_id(
            static_cast<core::SolutionNodeId>(*neighbor));
        if (!slot_id.has_value()) {
            continue;
        }
        const auto slot = node_slots.at(*slot_id);
        if (slot < 0 || static_cast<std::size_t>(slot) >= p_nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "A physical-node slot is out of range.");
        }
        result.values[static_cast<std::size_t>(slot) * columns + 1U] = 1.0F;
    }
    return result;
}

DenseVectorF32 ObservationHandler::get_v_node_status(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    std::size_t p_net_num_nodes) {
    if (v_node_id >= v_net.num_nodes() || p_net_num_nodes == 0U) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_current_virtual_node,
            "Cannot construct virtual-node status for the requested ID.");
    }
    const double virtual_nodes = static_cast<double>(v_net.num_nodes());
    const double physical_nodes = static_cast<double>(p_net_num_nodes);
    return DenseVectorF32{{
        static_cast<float>(
            (virtual_nodes - static_cast<double>(v_node_id + 1U)) /
            virtual_nodes),
        static_cast<float>(virtual_nodes / physical_nodes),
        // Preserve Python precedence: v_node_id + (1 / p_net_num_nodes).
        static_cast<float>(
            static_cast<double>(v_node_id) + 1.0 / physical_nodes),
    }};
}

DenseVectorF32 ObservationHandler::get_v_node_demand(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    const AttributeKinds& kinds,
    const AttributeBenchmarkMap* benchmarks) {
    const auto ids = select_node_ids(v_net, kinds);
    const auto value_ids = resolve_node_value_ids(v_net, ids);
    const auto resolved = benchmarks == nullptr
        ? std::vector<double>{}
        : resolve_benchmarks(v_net.node_attributes(), ids, *benchmarks);
    return v_node_demand_by_id(
        v_net,
        v_node_id,
        ids,
        value_ids,
        benchmarks == nullptr ? nullptr : &resolved);
}

DenseVectorF32 ObservationHandler::get_v_node_aggr_link_demands(
    const network::VirtualNetwork& v_net,
    std::size_t v_node_id,
    LinkAggregation aggregation,
    const AttributeKinds& kinds,
    const AttributeBenchmarkMap* benchmarks) {
    const auto ids = select_link_ids(v_net, kinds);
    const auto value_ids = resolve_link_value_ids(v_net, ids);
    const auto resolved = benchmarks == nullptr
        ? std::vector<double>{}
        : resolve_benchmarks(v_net.link_attributes(), ids, *benchmarks);
    return v_node_aggr_by_id(
        v_net,
        v_node_id,
        aggregation,
        ids,
        value_ids,
        benchmarks == nullptr ? nullptr : &resolved);
}

DenseMatrixF32 ObservationHandler::get_v_net_nodes_status(
    const network::VirtualNetwork& v_net,
    const core::NodeSlots& node_slots,
    std::optional<std::size_t> v_node_id,
    bool consist_decision,
    bool neighbor_flags) {
    const std::size_t nodes = v_net.num_nodes();
    std::size_t columns = v_node_id.has_value() ? 2U : 1U;
    if (neighbor_flags) {
        if (!v_node_id.has_value() || *v_node_id >= nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_current_virtual_node,
                "Neighbor status requires a valid current virtual-node ID.");
        }
        ++columns;
    }
    DenseMatrixF32 result{nodes, columns, {}};
    result.values.assign(nodes * columns, 0.0F);
    for (const auto& slot : node_slots.entries()) {
        if (slot.key < 0 || static_cast<std::size_t>(slot.key) >= nodes) {
            throw_feature(
                FeatureConstructorErrorCode::invalid_shape,
                "A placed virtual-node ID is out of range.");
        }
        result.values[static_cast<std::size_t>(slot.key) * columns] = 1.0F;
    }
    if (!v_node_id.has_value()) {
        return result;
    }
    if (*v_node_id < nodes) {
        result.values[*v_node_id * columns + 1U] =
            consist_decision ? 1.0F : -1.0F;
    }
    if (neighbor_flags) {
        const auto adjacent = v_net.graph().neighbors(static_cast<Vertex>(*v_node_id));
        for (auto neighbor = adjacent.first; neighbor != adjacent.second; ++neighbor) {
            if (node_slots.contains(static_cast<core::SolutionNodeId>(*neighbor))) {
                result.values[static_cast<std::size_t>(*neighbor) * columns + 2U] =
                    1.0F;
            }
        }
    }
    return result;
}

DenseMatrixF32 ObservationHandler::get_v_node_neighbor_flags(
    const network::VirtualNetwork& v_net,
    const core::NodeSlots& node_slots,
    std::size_t v_node_id) {
    const std::size_t nodes = v_net.num_nodes();
    if (v_node_id >= nodes) {
        throw_feature(
            FeatureConstructorErrorCode::invalid_current_virtual_node,
            "The current virtual-node ID is out of range.");
    }
    DenseMatrixF32 result{nodes, nodes, {}};
    result.values.assign(nodes * nodes, 0.0F);
    const auto adjacent = v_net.graph().neighbors(static_cast<Vertex>(v_node_id));
    for (auto neighbor = adjacent.first; neighbor != adjacent.second; ++neighbor) {
        if (node_slots.contains(static_cast<core::SolutionNodeId>(*neighbor))) {
            const std::size_t node = static_cast<std::size_t>(*neighbor);
            result.values[node * nodes + node] = 1.0F;
        }
    }
    return result;
}

}  // namespace virne::solver::learning::rl_core
