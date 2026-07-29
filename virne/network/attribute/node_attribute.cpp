#include "node_attribute.h"

#include <algorithm>
#include <cmath>
#include <exception>
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
    } catch (...) {
        for (std::size_t worker = 0U; worker < width; ++worker) {
            run_block(worker);
        }
        for (const std::exception_ptr& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        return;
    }

    std::size_t launched = 1U;
    try {
        for (; launched < width; ++launched) {
            threads.emplace_back(run_block, launched);
        }
    } catch (...) {
        run_block(0U);
        for (std::size_t worker = launched; worker < width; ++worker) {
            run_block(worker);
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
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
const AttrValue& get_value(
    const GraphType& graph,
    Vertex node,
    NodeAttributeBinding binding) {
    if (node >= graph.num_nodes()) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::node_out_of_range,
            NodeAttributeOperation::get,
            "Node index is out of range.");
    }
    const AttrValue* value = graph.node_attrs(node).find(binding.value_id);
    if (value == nullptr) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::missing_attribute,
            NodeAttributeOperation::get,
            "Node attribute value not found.");
    }
    return *value;
}

template <typename GraphType>
void set_sparse_values(
    GraphType& graph,
    const std::vector<NodeAttributeAssignment>& assignments,
    NodeAttributeBinding binding) {
    const std::size_t count = graph.num_nodes();
    for (const NodeAttributeAssignment& assignment : assignments) {
        if (assignment.node >= count) {
            continue;
        }
        graph.node_attrs(assignment.node).set(
            binding.value_id,
            clone_attr_value(assignment.value));
    }
}

template <typename GraphType>
void set_dense_values(
    GraphType& graph,
    const std::vector<AttrValue>& values,
    NodeAttributeBinding binding,
    std::size_t workers) {
    const std::size_t count = graph.num_nodes();
    if (values.size() < count) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::dense_data_too_short,
            NodeAttributeOperation::set_data,
            "Node attribute data is shorter than the node count.");
    }

    std::vector<AttrValue> prepared(count);
    parallel_for(count, workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            prepared[index] = clone_attr_value(values[index]);
        }
    });
    parallel_for(count, workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            graph.node_attrs(index).set(
                binding.value_id,
                std::move(prepared[index]));
        }
    });
}

template <typename GraphType>
std::vector<AttrValue> gather_values(
    const GraphType& graph,
    NodeAttributeBinding binding,
    std::size_t workers) {
    const std::size_t count = graph.num_nodes();
    std::vector<std::optional<AttrValue>> gathered(count);
    parallel_for(count, workers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const AttrValue* value =
                graph.node_attrs(index).find(binding.value_id);
            if (value != nullptr) {
                gathered[index] = clone_attr_value(*value);
            }
        }
    });
    std::vector<AttrValue> result;
    result.reserve(count);
    for (std::optional<AttrValue>& value : gathered) {
        if (value.has_value()) {
            result.push_back(std::move(*value));
        }
    }
    return result;
}

BaseAttributeSpec status_base_spec(const NodeStatusSpec& spec) {
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::node;
    result.kind = AttributeKind::status;
    result.generative = spec.generative;
    result.distribution = spec.distribution;
    result.dtype = spec.dtype;
    result.is_constraint = false;
    return result;
}

BaseAttributeSpec extrema_base_spec(const NodeExtremaSpec& spec) {
    if (!spec.originator_name.has_value()) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::missing_originator,
            NodeAttributeOperation::construct,
            "NodeExtremaAttribute requires 'originator' in config or kwargs.");
    }
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::node;
    result.kind = AttributeKind::extrema;
    result.originator = *spec.originator_name;
    result.is_constraint = false;
    return result;
}

BaseAttributeSpec resource_base_spec(const NodeResourceSpec& spec) {
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::node;
    result.kind = AttributeKind::resource;
    result.generative = spec.generative;
    result.distribution = spec.distribution;
    result.dtype = spec.dtype;
    result.is_constraint = true;
    return result;
}

BaseAttributeSpec position_base_spec(const NodePositionSpec& spec) {
    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::node;
    result.kind = AttributeKind::position;
    result.generative = spec.generative;
    result.distribution = spec.distribution;
    result.dtype = spec.dtype;
    result.is_constraint = true;
    return result;
}

AttributeNumber attribute_number(
    const AttrValue& value,
    NodeAttributeOperation operation) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean;
    }
    throw NodeAttributeException(
        NodeAttributeErrorCode::non_numeric_resource,
        operation,
        "Node resource attribute must be bool, int64, or double.");
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
    NodeAttributeOperation operation) {
    const AttrValue* value = values.find(id);
    if (value == nullptr) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::missing_resource_value,
            operation,
            "Missing attribute '" + std::string(name)
                + "' in node attribute dict.");
    }
    return *value;
}

AttrValue& required_resource(
    AttrMap& values,
    AttrId id,
    std::string_view name,
    NodeAttributeOperation operation) {
    AttrValue* value = values.find(id);
    if (value == nullptr) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::missing_resource_value,
            operation,
            "Missing attribute '" + std::string(name)
                + "' in node attribute dict.");
    }
    return *value;
}

template <typename Value>
AttributeNumber generated_number(Value value) {
    if constexpr (std::is_same_v<Value, std::uint8_t>) {
        return value != 0U;
    } else {
        return value;
    }
}

template <typename Value>
double generated_double(Value value) noexcept {
    if constexpr (std::is_same_v<Value, std::uint8_t>) {
        return value != 0U ? 1.0 : 0.0;
    } else {
        return static_cast<double>(value);
    }
}

double numpy_clip(double value, double minimum, double maximum) noexcept {
    if (std::isnan(value)) {
        return value;
    }
    if (std::isnan(minimum)) {
        return minimum;
    }
    if (std::isnan(maximum)) {
        return maximum;
    }
    if (minimum > maximum) {
        return maximum;
    }
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

template <typename GraphType>
std::vector<AttrValue> existing_position_values(
    const NodePositionAttribute& attribute,
    const GraphType& graph,
    NodeAttributeBinding binding,
    std::size_t workers) {
    if (graph.num_nodes() == 0U) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::empty_position_network,
            NodeAttributeOperation::get_existing_position,
            "Node position fallback requires at least one node.");
    }
    if (graph.node_attrs(Vertex{0U}).find(binding.value_id) == nullptr) {
        throw NodeAttributeException(
            NodeAttributeErrorCode::missing_position_data,
            NodeAttributeOperation::get_existing_position,
            "Please specify how to generate node position data (set generative=True "
            "or provide \"pos\" attribute in network nodes).");
    }
    return attribute.get_data(graph, binding, workers);
}

}  // namespace

NodeAttributeException::NodeAttributeException(
    NodeAttributeErrorCode code,
    NodeAttributeOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

NodeAttributeErrorCode NodeAttributeException::code() const noexcept {
    return code_;
}

NodeAttributeOperation NodeAttributeException::operation() const noexcept {
    return operation_;
}

NodeAttribute::NodeAttribute(BaseAttributeSpec spec)
    : BaseAttribute([&spec] {
          if (spec.owner != AttributeOwner::node) {
              throw NodeAttributeException(
                  NodeAttributeErrorCode::invalid_node_spec,
                  NodeAttributeOperation::construct,
                  "NodeAttribute requires owner 'node'.");
          }
          return std::move(spec);
      }()) {}

NodeAttributeBinding NodeAttribute::bind(const Graph& graph) const {
    return NodeAttributeBinding{graph.attr_id(spec().name)};
}

NodeAttributeBinding NodeAttribute::bind(const DiGraph& graph) const {
    return NodeAttributeBinding{graph.attr_id(spec().name)};
}

const AttrValue& NodeAttribute::get(
    const Graph& graph,
    Vertex node,
    NodeAttributeBinding binding) const {
    return get_value(graph, node, binding);
}

const AttrValue& NodeAttribute::get(
    const DiGraph& graph,
    Vertex node,
    NodeAttributeBinding binding) const {
    return get_value(graph, node, binding);
}

void NodeAttribute::set_data(
    Graph& graph,
    const std::vector<NodeAttributeAssignment>& assignments,
    NodeAttributeBinding binding) const {
    set_sparse_values(graph, assignments, binding);
}

void NodeAttribute::set_data(
    DiGraph& graph,
    const std::vector<NodeAttributeAssignment>& assignments,
    NodeAttributeBinding binding) const {
    set_sparse_values(graph, assignments, binding);
}

void NodeAttribute::set_data_dense(
    Graph& graph,
    const std::vector<AttrValue>& values,
    NodeAttributeBinding binding,
    std::size_t workers) const {
    set_dense_values(graph, values, binding, workers);
}

void NodeAttribute::set_data_dense(
    DiGraph& graph,
    const std::vector<AttrValue>& values,
    NodeAttributeBinding binding,
    std::size_t workers) const {
    set_dense_values(graph, values, binding, workers);
}

std::vector<AttrValue> NodeAttribute::get_data(
    const Graph& graph,
    NodeAttributeBinding binding,
    std::size_t workers) const {
    return gather_values(graph, binding, workers);
}

std::vector<AttrValue> NodeAttribute::get_data(
    const DiGraph& graph,
    NodeAttributeBinding binding,
    std::size_t workers) const {
    return gather_values(graph, binding, workers);
}

NodeStatusAttribute::NodeStatusAttribute(NodeStatusSpec spec)
    : NodeAttribute(status_base_spec(spec)) {}

NodeExtremaAttribute::NodeExtremaAttribute(NodeExtremaSpec spec)
    : NodeAttribute(extrema_base_spec(spec)),
      originator_name_(*spec.originator_name),
      originator_id_(spec.originator_id) {}

std::string_view NodeExtremaAttribute::originator_name() const noexcept {
    return originator_name_;
}

AttributeDefinitionId NodeExtremaAttribute::originator_id() const noexcept {
    return originator_id_;
}

std::vector<AttrValue> NodeExtremaAttribute::generate_from_resolved_originator(
    const Graph& graph,
    const NodeAttribute& originator,
    NodeAttributeBinding originator_binding,
    std::size_t workers) const {
    return originator.get_data(graph, originator_binding, workers);
}

std::vector<AttrValue> NodeExtremaAttribute::generate_from_resolved_originator(
    const DiGraph& graph,
    const NodeAttribute& originator,
    NodeAttributeBinding originator_binding,
    std::size_t workers) const {
    return originator.get_data(graph, originator_binding, workers);
}

NodeResourceAttribute::NodeResourceAttribute(NodeResourceSpec spec)
    : NodeAttribute(resource_base_spec(spec)),
      restriction_(spec.restriction),
      checking_level_(spec.checking_level) {}

ConstraintRestriction NodeResourceAttribute::restriction() const noexcept {
    return restriction_;
}

CheckingLevel NodeResourceAttribute::checking_level() const noexcept {
    return checking_level_;
}

bool NodeResourceAttribute::update(
    const AttrMap& virtual_node,
    AttrId virtual_id,
    AttrMap& physical_node,
    AttrId physical_id,
    ResourceUpdateOperation operation,
    bool safe) const {
    const AttrValue* virtual_graph_value = nullptr;
    AttrValue* physical_graph_value = nullptr;
    const bool virtual_first =
        operation == ResourceUpdateOperation::subtract && safe;
    if (virtual_first) {
        virtual_graph_value = &required_resource(
            virtual_node,
            virtual_id,
            spec().name,
            NodeAttributeOperation::update_resource);
        physical_graph_value = &required_resource(
            physical_node,
            physical_id,
            spec().name,
            NodeAttributeOperation::update_resource);
    } else {
        physical_graph_value = &required_resource(
            physical_node,
            physical_id,
            spec().name,
            NodeAttributeOperation::update_resource);
        virtual_graph_value = &required_resource(
            virtual_node,
            virtual_id,
            spec().name,
            NodeAttributeOperation::update_resource);
    }

    AttributeNumber virtual_value = std::int64_t{0};
    AttributeNumber physical_value = std::int64_t{0};
    if (virtual_first) {
        virtual_value = attribute_number(
            *virtual_graph_value,
            NodeAttributeOperation::update_resource);
        physical_value = attribute_number(
            *physical_graph_value,
            NodeAttributeOperation::update_resource);
    } else {
        physical_value = attribute_number(
            *physical_graph_value,
            NodeAttributeOperation::update_resource);
        virtual_value = attribute_number(
            *virtual_graph_value,
            NodeAttributeOperation::update_resource);
    }
    const bool result = update_resource_value(
        virtual_value,
        physical_value,
        operation,
        safe,
        spec().name);
    *physical_graph_value = graph_value(physical_value);
    return result;
}

SatisfiabilityResult NodeResourceAttribute::check_constraint_satisfiability(
    const AttrMap& virtual_node,
    AttrId virtual_id,
    const AttrMap& physical_node,
    AttrId physical_id,
    ComparisonOperation method) const {
    const AttrValue& virtual_graph_value = required_resource(
        virtual_node,
        virtual_id,
        spec().name,
        NodeAttributeOperation::check_resource);
    const AttrValue& physical_graph_value = required_resource(
        physical_node,
        physical_id,
        spec().name,
        NodeAttributeOperation::check_resource);
    const AttributeNumber virtual_value = attribute_number(
        virtual_graph_value,
        NodeAttributeOperation::check_resource);
    const AttributeNumber physical_value = attribute_number(
        physical_graph_value,
        NodeAttributeOperation::check_resource);
    return calculate_satisfiability_values(
        virtual_value,
        physical_value,
        method,
        restriction_);
}

NodePositionAttribute::NodePositionAttribute(NodePositionSpec spec)
    : NodeAttribute(position_base_spec(spec)),
      minimum_radius_(spec.minimum_radius),
      maximum_radius_(spec.maximum_radius),
      restriction_(spec.restriction) {}

double NodePositionAttribute::minimum_radius() const noexcept {
    return minimum_radius_;
}

double NodePositionAttribute::maximum_radius() const noexcept {
    return maximum_radius_;
}

ConstraintRestriction NodePositionAttribute::restriction() const noexcept {
    return restriction_;
}

std::vector<NodePositionValue> NodePositionAttribute::generate_positions(
    const NetworkCardinality& network,
    NumpyRandomState& rng,
    std::size_t workers) const {
    const virne::utils::GeneratedData x =
        generate_configured_data(network, rng, workers);
    const virne::utils::GeneratedData y =
        generate_configured_data(network, rng, workers);
    const virne::utils::GeneratedData radius =
        generate_configured_data(network, rng, workers);

    std::vector<NodePositionValue> result(network.num_nodes);
    std::visit(
        [&](const auto& x_values) {
            std::visit(
                [&](const auto& y_values) {
                    std::visit(
                        [&](const auto& radius_values) {
                            if (x_values.size() != result.size()
                                || y_values.size() != result.size()
                                || radius_values.size() != result.size()) {
                                throw NodeAttributeException(
                                    NodeAttributeErrorCode::dense_data_too_short,
                                    NodeAttributeOperation::generate_position,
                                    "Generated position lane size mismatch.");
                            }
                            parallel_for(
                                result.size(),
                                workers,
                                [&](std::size_t begin, std::size_t end) {
                                    for (std::size_t index = begin;
                                         index < end;
                                         ++index) {
                                        result[index].x =
                                            generated_number(x_values[index]);
                                        result[index].y =
                                            generated_number(y_values[index]);
                                        result[index].radius = numpy_clip(
                                            generated_double(radius_values[index]),
                                            minimum_radius_,
                                            maximum_radius_);
                                    }
                                });
                        },
                        radius.values);
                },
                y.values);
        },
        x.values);
    return result;
}

NodeAttributeBinding NodePositionAttribute::bind_existing_pos(
    const Graph& graph) const {
    return NodeAttributeBinding{graph.attr_id("pos")};
}

NodeAttributeBinding NodePositionAttribute::bind_existing_pos(
    const DiGraph& graph) const {
    return NodeAttributeBinding{graph.attr_id("pos")};
}

std::vector<AttrValue> NodePositionAttribute::get_existing_pos_data(
    const Graph& graph,
    NodeAttributeBinding literal_pos_binding,
    std::size_t workers) const {
    return existing_position_values(
        *this, graph, literal_pos_binding, workers);
}

std::vector<AttrValue> NodePositionAttribute::get_existing_pos_data(
    const DiGraph& graph,
    NodeAttributeBinding literal_pos_binding,
    std::size_t workers) const {
    return existing_position_values(
        *this, graph, literal_pos_binding, workers);
}

}  // namespace virne::network::attribute
