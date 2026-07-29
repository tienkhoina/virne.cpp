#include "link_attribute.h"

#include "nx/sparse.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <numeric>
#include <thread>
#include <type_traits>
#include <utility>

namespace virne::network::attribute {
namespace {

std::size_t configured_worker_count(
    std::size_t count,
    std::size_t configured_workers) noexcept {
    if (count == 0U || configured_workers <= 1U) {
        return 1U;
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    const std::size_t available = hardware == 0U
        ? configured_workers
        : static_cast<std::size_t>(hardware);
    return std::max(
        std::size_t{1U},
        std::min({configured_workers, count, available}));
}

template <typename Function>
void parallel_for(
    std::size_t count,
    std::size_t configured_workers,
    Function&& function) {
    const std::size_t width =
        configured_worker_count(count, configured_workers);
    const auto bounds = [count, width](std::size_t worker) noexcept {
        const std::size_t base = count / width;
        const std::size_t remainder = count % width;
        const std::size_t begin = worker * base + std::min(worker, remainder);
        return std::pair<std::size_t, std::size_t>{
            begin,
            begin + base + (worker < remainder ? 1U : 0U)};
    };
    if (width == 1U) {
        function(std::size_t{0U}, count);
        return;
    }

    std::vector<std::exception_ptr> errors(width);
    const auto run_block = [&](std::size_t worker) noexcept {
        try {
            const auto [begin, end] = bounds(worker);
            function(begin, end);
        } catch (...) {
            errors[worker] = std::current_exception();
        }
    };

    std::vector<std::thread> threads;
    try {
        threads.reserve(width - 1U);
        for (std::size_t worker = 1U; worker < width; ++worker) {
            threads.emplace_back(run_block, worker);
        }
    } catch (...) {
        for (std::thread& thread : threads) {
            thread.join();
        }
        for (std::size_t worker = threads.size() + 1U; worker < width; ++worker) {
            run_block(worker);
        }
        run_block(0U);
        for (const std::exception_ptr& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        return;
    }

    run_block(0U);
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

template <typename GraphType>
auto ordered_edges(const GraphType& graph) {
    const auto range = graph.edges();
    using Iterator = decltype(range.first);
    using EdgeType = typename std::iterator_traits<Iterator>::value_type;
    std::vector<EdgeType> result;
    result.reserve(graph.num_edges());
    for (auto iterator = range.first; iterator != range.second; ++iterator) {
        result.push_back(*iterator);
    }
    return result;
}

template <typename GraphType>
const AttrValue& get_value(
    const GraphType& graph,
    Vertex source,
    Vertex target,
    LinkAttributeBinding binding) {
    if (source >= graph.num_nodes() || target >= graph.num_nodes()
        || !graph.has_edge(source, target)) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::edge_not_found,
            LinkAttributeOperation::get,
            "Link does not exist in the graph.");
    }
    const auto edge = graph.edge(source, target);
    const AttrValue* value = graph.edge_attrs(edge).find(binding.value_id);
    if (value == nullptr) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::missing_attribute,
            LinkAttributeOperation::get,
            "Link attribute value not found.");
    }
    return *value;
}

template <typename GraphType>
void set_sparse_values(
    GraphType& graph,
    const std::vector<LinkAttributeAssignment>& assignments,
    LinkAttributeBinding binding) {
    const std::size_t count = graph.num_nodes();
    for (const LinkAttributeAssignment& assignment : assignments) {
        if (assignment.source >= count || assignment.target >= count
            || !graph.has_edge(assignment.source, assignment.target)) {
            continue;
        }
        graph.edge_attrs(graph.edge(assignment.source, assignment.target)).set(
            binding.value_id,
            clone_attr_value(assignment.value));
    }
}

template <typename GraphType>
void set_dense_values(
    GraphType& graph,
    const std::vector<AttrValue>& values,
    LinkAttributeBinding binding,
    std::size_t workers) {
    const auto edges = ordered_edges(graph);
    if (values.size() < edges.size()) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::dense_data_too_short,
            LinkAttributeOperation::set_data,
            "Link attribute data is shorter than the edge count.");
    }
    std::vector<AttrValue> prepared(edges.size());
    parallel_for(edges.size(), workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            prepared[index] = clone_attr_value(values[index]);
        }
    });
    parallel_for(edges.size(), workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            graph.edge_attrs(edges[index]).set(
                binding.value_id,
                std::move(prepared[index]));
        }
    });
}

template <typename GraphType>
std::vector<AttrValue> gather_values(
    const GraphType& graph,
    LinkAttributeBinding binding,
    std::size_t workers) {
    const auto edges = ordered_edges(graph);
    std::vector<std::optional<AttrValue>> gathered(edges.size());
    parallel_for(edges.size(), workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const AttrValue* value =
                graph.edge_attrs(edges[index]).find(binding.value_id);
            if (value != nullptr) {
                gathered[index] = clone_attr_value(*value);
            }
        }
    });
    std::vector<AttrValue> result;
    result.reserve(edges.size());
    for (std::optional<AttrValue>& value : gathered) {
        if (value.has_value()) {
            result.push_back(std::move(*value));
        }
    }
    return result;
}

std::vector<Vertex> node_order(std::size_t count) {
    std::vector<Vertex> result(count);
    std::iota(result.begin(), result.end(), Vertex{0U});
    return result;
}

template <typename GraphType>
DistanceMatrix adjacency_data(
    const GraphType& graph,
    LinkAttributeBinding binding,
    bool normalized,
    std::size_t workers) {
    const SparseMatrix sparse = nx::attr_sparse_matrix(
        graph,
        binding.value_id,
        normalized,
        node_order(graph.num_nodes()));
    DistanceMatrix result(graph.num_nodes());
    parallel_for(sparse.nnz(), workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            result(sparse.row[index], sparse.col[index]) = sparse.value[index];
        }
    });
    return result;
}

std::vector<double> aggregate_columns(
    const DistanceMatrix& matrix,
    LinkAggregation aggregation,
    std::size_t workers) {
    const std::size_t count = matrix.n;
    if (count == 0U) {
        if (aggregation == LinkAggregation::maximum
            || aggregation == LinkAggregation::minimum) {
            throw LinkAttributeException(
                LinkAttributeErrorCode::empty_aggregation,
                LinkAttributeOperation::aggregation,
                "Cannot reduce maximum or minimum over an empty matrix.");
        }
        return {};
    }
    std::vector<double> result(count, 0.0);
    parallel_for(count, workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t column = begin; column < end; ++column) {
            if (aggregation == LinkAggregation::maximum
                || aggregation == LinkAggregation::minimum) {
                double value = matrix(0U, column);
                for (std::size_t row = 1U; row < count; ++row) {
                    const double candidate = matrix(row, column);
                    if (std::isnan(value) || std::isnan(candidate)) {
                        value = std::numeric_limits<double>::quiet_NaN();
                    } else if (aggregation == LinkAggregation::maximum) {
                        value = value > candidate ? value : candidate;
                    } else {
                        value = value < candidate ? value : candidate;
                    }
                }
                result[column] = value;
                continue;
            }
            double value = 0.0;
            for (std::size_t row = 0U; row < count; ++row) {
                value += matrix(row, column);
            }
            result[column] = aggregation == LinkAggregation::mean
                ? value / static_cast<double>(count)
                : value;
        }
    });
    return result;
}

BaseAttributeSpec status_base_spec(const LinkStatusSpec& spec) {
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::status;
    result.generative = spec.generative;
    result.distribution = spec.distribution;
    result.dtype = spec.dtype;
    result.is_constraint = false;
    return result;
}

BaseAttributeSpec extrema_base_spec(const LinkExtremaSpec& spec) {
    if (!spec.originator_name.has_value()) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::missing_originator,
            LinkAttributeOperation::construct,
            "LinkExtremaAttribute requires 'originator' in config or kwargs.");
    }
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::extrema;
    result.originator = *spec.originator_name;
    result.is_constraint = false;
    return result;
}

BaseAttributeSpec resource_base_spec(const LinkResourceSpec& spec) {
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::resource;
    result.generative = spec.generative;
    result.distribution = spec.distribution;
    result.dtype = spec.dtype;
    result.is_constraint = true;
    return result;
}

BaseAttributeSpec latency_base_spec(const LinkLatencySpec& spec) {
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::link;
    result.kind = AttributeKind::latency;
    result.generative = spec.generative;
    result.distribution = spec.distribution;
    result.dtype = spec.dtype;
    result.is_constraint = true;
    return result;
}

AttributeNumber attribute_number(
    const AttrValue& value,
    LinkAttributeOperation operation) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean;
    }
    throw LinkAttributeException(
        LinkAttributeErrorCode::non_numeric_resource,
        operation,
        "Link resource attribute must be bool, int64, or double.");
}

AttrValue graph_value(const AttributeNumber& value) {
    return std::visit(
        [](const auto item) -> AttrValue { return AttrValue{item}; },
        value);
}

const AttrValue& required_resource(
    const AttrMap& values,
    AttrId id,
    std::string_view name,
    LinkAttributeOperation operation) {
    const AttrValue* value = values.find(id);
    if (value == nullptr) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::missing_resource_value,
            operation,
            "Missing attribute '" + std::string(name)
                + "' in link attribute dict.");
    }
    return *value;
}

AttrValue& required_resource(
    AttrMap& values,
    AttrId id,
    std::string_view name,
    LinkAttributeOperation operation) {
    AttrValue* value = values.find(id);
    if (value == nullptr) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::missing_resource_value,
            operation,
            "Missing attribute '" + std::string(name)
                + "' in link attribute dict.");
    }
    return *value;
}

bool update_resource(
    const AttrMap& virtual_link,
    AttrId virtual_id,
    AttrMap& physical_link,
    AttrId physical_id,
    std::string_view name,
    ResourceUpdateOperation operation,
    bool safe) {
    const AttrValue* virtual_graph_value = nullptr;
    AttrValue* physical_graph_value = nullptr;
    const bool virtual_first =
        operation == ResourceUpdateOperation::subtract && safe;
    if (virtual_first) {
        virtual_graph_value = &required_resource(
            virtual_link, virtual_id, name, LinkAttributeOperation::update_resource);
        physical_graph_value = &required_resource(
            physical_link, physical_id, name, LinkAttributeOperation::update_resource);
    } else {
        physical_graph_value = &required_resource(
            physical_link, physical_id, name, LinkAttributeOperation::update_resource);
        virtual_graph_value = &required_resource(
            virtual_link, virtual_id, name, LinkAttributeOperation::update_resource);
    }

    AttributeNumber virtual_value = std::int64_t{0};
    AttributeNumber physical_value = std::int64_t{0};
    if (virtual_first) {
        virtual_value = attribute_number(
            *virtual_graph_value, LinkAttributeOperation::update_resource);
        physical_value = attribute_number(
            *physical_graph_value, LinkAttributeOperation::update_resource);
    } else {
        physical_value = attribute_number(
            *physical_graph_value, LinkAttributeOperation::update_resource);
        virtual_value = attribute_number(
            *virtual_graph_value, LinkAttributeOperation::update_resource);
    }
    const bool result = update_resource_value(
        virtual_value, physical_value, operation, safe, name);
    *physical_graph_value = graph_value(physical_value);
    return result;
}

template <typename GraphType>
bool update_path_values(
    const LinkResourceAttribute& attribute,
    const AttrMap& virtual_link,
    AttrId virtual_id,
    GraphType& physical_graph,
    const std::vector<Vertex>& path,
    LinkAttributeBinding physical_binding,
    ResourceUpdateOperation operation,
    bool safe) {
    if (path.size() <= 1U) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::path_too_short,
            LinkAttributeOperation::update_path,
            "Path must have at least two nodes.");
    }
    for (std::size_t index = 1U; index < path.size(); ++index) {
        const auto edge = physical_graph.edge(path[index - 1U], path[index]);
        update_resource(
            virtual_link,
            virtual_id,
            physical_graph.edge_attrs(edge),
            physical_binding.value_id,
            attribute.spec().name,
            operation,
            safe);
    }
    return true;
}

double numeric_value(const AttrValue& value) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return static_cast<double>(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean ? 1.0 : 0.0;
    }
    throw LinkAttributeException(
        LinkAttributeErrorCode::invalid_position_data,
        LinkAttributeOperation::generate_latency,
        "Node position values must be numeric lists.");
}

double attribute_number_value(const AttributeNumber& value) noexcept {
    return std::visit(
        [](const auto item) noexcept -> double {
            using Value = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Value, bool>) {
                return item ? 1.0 : 0.0;
            } else {
                return static_cast<double>(item);
            }
        },
        value);
}

bool integral_number(const AttributeNumber& value, std::int64_t& result) noexcept {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        result = *integer;
        return true;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        result = *boolean ? 1 : 0;
        return true;
    }
    return false;
}

double latency_span(
    const AttributeNumber& minimum,
    const AttributeNumber& maximum) noexcept {
    std::int64_t minimum_integer = 0;
    std::int64_t maximum_integer = 0;
    if (integral_number(minimum, minimum_integer)
        && integral_number(maximum, maximum_integer)) {
        if (maximum_integer >= minimum_integer) {
            const std::uint64_t difference =
                static_cast<std::uint64_t>(maximum_integer)
                - static_cast<std::uint64_t>(minimum_integer);
            return static_cast<double>(difference);
        }
        const std::uint64_t difference =
            static_cast<std::uint64_t>(minimum_integer)
            - static_cast<std::uint64_t>(maximum_integer);
        return -static_cast<double>(difference);
    }
    return attribute_number_value(maximum) - attribute_number_value(minimum);
}

template <typename GraphType>
NodeAttributeBinding resolve_position(
    const GraphType& graph) {
    if (graph.num_nodes() == 0U) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::empty_position_network,
            LinkAttributeOperation::resolve_position,
            "The generation of latency from position requires a non-empty graph.");
    }
    const AttrMap& first_node = graph.node_attrs(Vertex{0U});
    for (const AttrId id : first_node.attribute_ids()) {
        if (graph.attr_name(id).find("pos") != std::string_view::npos) {
            return NodeAttributeBinding{id};
        }
    }
    throw LinkAttributeException(
        LinkAttributeErrorCode::missing_position_data,
        LinkAttributeOperation::resolve_position,
        "The generation of this attribute requires node position.");
}

template <typename GraphType>
const std::vector<AttrValue>& position_values(
    const GraphType& graph,
    Vertex node,
    NodeAttributeBinding binding) {
    const AttrValue* value = graph.node_attrs(node).find(binding.value_id);
    if (value == nullptr) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::missing_position_data,
            LinkAttributeOperation::generate_latency,
            "The generation of this attribute requires node position.");
    }
    const auto* list = std::get_if<AttrListPtr>(value);
    if (list == nullptr || !*list) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::invalid_position_data,
            LinkAttributeOperation::generate_latency,
            "Node position values must be numeric lists.");
    }
    return (*list)->values;
}

template <typename GraphType>
std::vector<double> position_latency(
    const LinkLatencyAttribute& attribute,
    const GraphType& graph,
    NodeAttributeBinding binding,
    std::size_t workers) {
    if (!attribute.spec().generative) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::non_generative_latency,
            LinkAttributeOperation::generate_latency,
            "Non-generative latency attribute must implement generate_data.");
    }
    if (attribute.generation_kind() != LatencyGenerationKind::position) {
        throw LinkAttributeException(
            LinkAttributeErrorCode::invalid_latency_generation,
            LinkAttributeOperation::generate_latency,
            "Latency generation kind is not position.");
    }
    const auto edges = ordered_edges(graph);
    std::vector<double> result(edges.size());
    const double minimum = attribute_number_value(attribute.minimum());
    const double span = latency_span(attribute.minimum(), attribute.maximum());
    parallel_for(edges.size(), workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const auto& left = position_values(
                graph, graph.source(edges[index]), binding);
            const auto& right = position_values(
                graph, graph.target(edges[index]), binding);
            if (left.size() != right.size()) {
                throw LinkAttributeException(
                    LinkAttributeErrorCode::invalid_position_data,
                    LinkAttributeOperation::generate_latency,
                    "Endpoint position dimensions do not match.");
            }
            double squared = 0.0;
            for (std::size_t lane = 0U; lane < left.size(); ++lane) {
                const double difference =
                    numeric_value(left[lane]) - numeric_value(right[lane]);
                squared += difference * difference;
            }
            const double distance = std::sqrt(squared);
            volatile double product = distance * span;
            result[index] = product + minimum;
        }
    });
    return result;
}

}  // namespace

LinkAttributeException::LinkAttributeException(
    LinkAttributeErrorCode code,
    LinkAttributeOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

LinkAttributeErrorCode LinkAttributeException::code() const noexcept {
    return code_;
}

LinkAttributeOperation LinkAttributeException::operation() const noexcept {
    return operation_;
}

LinkAggregation link_aggregation_from_string(std::string_view value) {
    if (value == "sum") {
        return LinkAggregation::sum;
    }
    if (value == "mean") {
        return LinkAggregation::mean;
    }
    if (value == "max") {
        return LinkAggregation::maximum;
    }
    if (value == "min") {
        return LinkAggregation::minimum;
    }
    throw LinkAttributeException(
        LinkAttributeErrorCode::unsupported_aggregation,
        LinkAttributeOperation::resolve_aggregation,
        "Aggregation '" + std::string(value) + "' is not supported.");
}

std::string_view link_aggregation_name(LinkAggregation value) noexcept {
    switch (value) {
        case LinkAggregation::sum: return "sum";
        case LinkAggregation::mean: return "mean";
        case LinkAggregation::maximum: return "max";
        case LinkAggregation::minimum: return "min";
    }
    return "unknown";
}

LinkAttribute::LinkAttribute(BaseAttributeSpec spec)
    : BaseAttribute([&spec] {
          if (spec.owner != AttributeOwner::link) {
              throw LinkAttributeException(
                  LinkAttributeErrorCode::invalid_link_spec,
                  LinkAttributeOperation::construct,
                  "LinkAttribute requires owner 'link'.");
          }
          return std::move(spec);
      }()) {}

LinkAttributeBinding LinkAttribute::bind(const Graph& graph) const {
    return LinkAttributeBinding{graph.attr_id(spec().name)};
}

LinkAttributeBinding LinkAttribute::bind(const DiGraph& graph) const {
    return LinkAttributeBinding{graph.attr_id(spec().name)};
}

const AttrValue& LinkAttribute::get(
    const Graph& graph,
    Vertex source,
    Vertex target,
    LinkAttributeBinding binding) const {
    return get_value(graph, source, target, binding);
}

const AttrValue& LinkAttribute::get(
    const DiGraph& graph,
    Vertex source,
    Vertex target,
    LinkAttributeBinding binding) const {
    return get_value(graph, source, target, binding);
}

void LinkAttribute::set_data(
    Graph& graph,
    const std::vector<LinkAttributeAssignment>& assignments,
    LinkAttributeBinding binding) const {
    set_sparse_values(graph, assignments, binding);
}

void LinkAttribute::set_data(
    DiGraph& graph,
    const std::vector<LinkAttributeAssignment>& assignments,
    LinkAttributeBinding binding) const {
    set_sparse_values(graph, assignments, binding);
}

void LinkAttribute::set_data_dense(
    Graph& graph,
    const std::vector<AttrValue>& values,
    LinkAttributeBinding binding,
    std::size_t workers) const {
    set_dense_values(graph, values, binding, workers);
}

void LinkAttribute::set_data_dense(
    DiGraph& graph,
    const std::vector<AttrValue>& values,
    LinkAttributeBinding binding,
    std::size_t workers) const {
    set_dense_values(graph, values, binding, workers);
}

std::vector<AttrValue> LinkAttribute::get_data(
    const Graph& graph,
    LinkAttributeBinding binding,
    std::size_t workers) const {
    return gather_values(graph, binding, workers);
}

std::vector<AttrValue> LinkAttribute::get_data(
    const DiGraph& graph,
    LinkAttributeBinding binding,
    std::size_t workers) const {
    return gather_values(graph, binding, workers);
}

DistanceMatrix LinkAttribute::get_adjacency_data(
    const Graph& graph,
    LinkAttributeBinding binding,
    bool normalized,
    std::size_t workers) const {
    return adjacency_data(graph, binding, normalized, workers);
}

DistanceMatrix LinkAttribute::get_adjacency_data(
    const DiGraph& graph,
    LinkAttributeBinding binding,
    bool normalized,
    std::size_t workers) const {
    return adjacency_data(graph, binding, normalized, workers);
}

std::vector<double> LinkAttribute::get_aggregation_data(
    const Graph& graph,
    LinkAttributeBinding binding,
    LinkAggregation aggregation,
    bool normalized,
    std::size_t workers) const {
    return aggregate_columns(
        get_adjacency_data(graph, binding, normalized, workers),
        aggregation,
        workers);
}

std::vector<double> LinkAttribute::get_aggregation_data(
    const DiGraph& graph,
    LinkAttributeBinding binding,
    LinkAggregation aggregation,
    bool normalized,
    std::size_t workers) const {
    return aggregate_columns(
        get_adjacency_data(graph, binding, normalized, workers),
        aggregation,
        workers);
}

bool LinkAttribute::update_path() const {
    throw LinkAttributeException(
        LinkAttributeErrorCode::update_path_not_implemented,
        LinkAttributeOperation::update_path,
        "update_path method is not implemented in abstract LinkAttribute.");
}

LinkStatusAttribute::LinkStatusAttribute(LinkStatusSpec spec)
    : LinkAttribute(status_base_spec(spec)) {}

LinkExtremaAttribute::LinkExtremaAttribute(LinkExtremaSpec spec)
    : LinkAttribute(extrema_base_spec(spec)),
      originator_name_(*spec.originator_name),
      originator_id_(spec.originator_id) {}

std::string_view LinkExtremaAttribute::originator_name() const noexcept {
    return originator_name_;
}

AttributeDefinitionId LinkExtremaAttribute::originator_id() const noexcept {
    return originator_id_;
}

std::vector<AttrValue> LinkExtremaAttribute::generate_from_resolved_originator(
    const Graph& graph,
    const LinkAttribute& originator,
    LinkAttributeBinding originator_binding,
    std::size_t workers) const {
    return originator.get_data(graph, originator_binding, workers);
}

std::vector<AttrValue> LinkExtremaAttribute::generate_from_resolved_originator(
    const DiGraph& graph,
    const LinkAttribute& originator,
    LinkAttributeBinding originator_binding,
    std::size_t workers) const {
    return originator.get_data(graph, originator_binding, workers);
}

LinkResourceAttribute::LinkResourceAttribute(LinkResourceSpec spec)
    : LinkAttribute(resource_base_spec(spec)),
      restriction_(spec.restriction),
      checking_level_(spec.checking_level) {}

ConstraintRestriction LinkResourceAttribute::restriction() const noexcept {
    return restriction_;
}

CheckingLevel LinkResourceAttribute::checking_level() const noexcept {
    return checking_level_;
}

SatisfiabilityResult LinkResourceAttribute::check_constraint_satisfiability(
    const AttrMap& virtual_link,
    AttrId virtual_id,
    const AttrMap& physical_link,
    AttrId physical_id,
    ComparisonOperation method) const {
    const AttrValue& virtual_graph_value = required_resource(
        virtual_link, virtual_id, spec().name, LinkAttributeOperation::check_resource);
    const AttrValue& physical_graph_value = required_resource(
        physical_link, physical_id, spec().name, LinkAttributeOperation::check_resource);
    return calculate_satisfiability_values(
        attribute_number(virtual_graph_value, LinkAttributeOperation::check_resource),
        attribute_number(physical_graph_value, LinkAttributeOperation::check_resource),
        method,
        restriction_);
}

bool LinkResourceAttribute::update_path(
    const AttrMap& virtual_link,
    AttrId virtual_id,
    Graph& physical_graph,
    const std::vector<Vertex>& path,
    LinkAttributeBinding physical_binding,
    ResourceUpdateOperation operation,
    bool safe) const {
    return update_path_values(
        *this, virtual_link, virtual_id, physical_graph, path,
        physical_binding, operation, safe);
}

bool LinkResourceAttribute::update_path(
    const AttrMap& virtual_link,
    AttrId virtual_id,
    DiGraph& physical_graph,
    const std::vector<Vertex>& path,
    LinkAttributeBinding physical_binding,
    ResourceUpdateOperation operation,
    bool safe) const {
    return update_path_values(
        *this, virtual_link, virtual_id, physical_graph, path,
        physical_binding, operation, safe);
}

LinkLatencyAttribute::LinkLatencyAttribute(LinkLatencySpec spec)
    : LinkAttribute(latency_base_spec(spec)),
      generation_(spec.generation),
      minimum_(spec.minimum),
      maximum_(spec.maximum),
      restriction_(spec.restriction),
      checking_level_(spec.checking_level) {}

LatencyGenerationKind LinkLatencyAttribute::generation_kind() const noexcept {
    return generation_;
}

const AttributeNumber& LinkLatencyAttribute::minimum() const noexcept {
    return minimum_;
}

const AttributeNumber& LinkLatencyAttribute::maximum() const noexcept {
    return maximum_;
}

ConstraintRestriction LinkLatencyAttribute::restriction() const noexcept {
    return restriction_;
}

CheckingLevel LinkLatencyAttribute::checking_level() const noexcept {
    return checking_level_;
}

SatisfiabilityResult LinkLatencyAttribute::check_constraint_satisfiability(
    const AttrMap& virtual_link,
    AttrId virtual_id,
    const std::vector<const AttrMap*>& physical_path,
    AttrId physical_id,
    ComparisonOperation method) const {
    AttributeNumber cumulative = std::int64_t{0};
    for (const AttrMap* physical_link : physical_path) {
        if (physical_link == nullptr) {
            throw LinkAttributeException(
                LinkAttributeErrorCode::missing_resource_value,
                LinkAttributeOperation::check_latency,
                "Physical path contains a null link attribute map.");
        }
        const AttributeNumber value = attribute_number(
            required_resource(
                *physical_link,
                physical_id,
                spec().name,
                LinkAttributeOperation::check_latency),
            LinkAttributeOperation::check_latency);
        update_resource_value(
            value,
            cumulative,
            ResourceUpdateOperation::add,
            false,
            spec().name);
    }
    const AttributeNumber virtual_value = attribute_number(
        required_resource(
            virtual_link,
            virtual_id,
            spec().name,
            LinkAttributeOperation::check_latency),
        LinkAttributeOperation::check_latency);
    return calculate_satisfiability_values(
        virtual_value,
        cumulative,
        method,
        restriction_);
}

NodeAttributeBinding LinkLatencyAttribute::resolve_position_binding(
    const Graph& graph) const {
    return resolve_position(graph);
}

NodeAttributeBinding LinkLatencyAttribute::resolve_position_binding(
    const DiGraph& graph) const {
    return resolve_position(graph);
}

std::vector<double> LinkLatencyAttribute::generate_from_position(
    const Graph& graph,
    NodeAttributeBinding position_binding,
    std::size_t workers) const {
    return position_latency(*this, graph, position_binding, workers);
}

std::vector<double> LinkLatencyAttribute::generate_from_position(
    const DiGraph& graph,
    NodeAttributeBinding position_binding,
    std::size_t workers) const {
    return position_latency(*this, graph, position_binding, workers);
}

}  // namespace virne::network::attribute
