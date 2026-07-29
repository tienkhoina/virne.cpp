#include "graph_attribute.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <thread>
#include <unordered_set>
#include <utility>

namespace virne::network::attribute {
namespace {

BaseAttributeSpec checked_graph_spec(BaseAttributeSpec spec) {
    if (spec.owner != AttributeOwner::graph) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::invalid_graph_spec,
            GraphAttributeOperation::construct,
            "GraphAttribute owner must be graph");
    }
    return spec;
}

BaseAttributeSpec make_graph_status_spec(GraphStatusSpec spec) {
    BaseAttributeSpec result;
    result.name = std::move(spec.name);
    result.owner = AttributeOwner::graph;
    result.kind = AttributeKind::status;
    result.generative = spec.generative;
    result.distribution = std::move(spec.distribution);
    result.dtype = spec.dtype;
    result.is_constraint = InformationMethodSpec::is_constraint;
    return result;
}

BaseAttributeSpec make_graph_extrema_spec(const GraphExtremaSpec& spec) {
    if (!spec.originator_name.has_value()) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::missing_originator,
            GraphAttributeOperation::construct,
            "GraphExtremaAttribute requires an originator");
    }

    BaseAttributeSpec result;
    result.name = spec.name;
    result.owner = AttributeOwner::graph;
    result.kind = AttributeKind::extrema;
    result.originator = spec.originator_name;
    result.is_constraint = InformationMethodSpec::is_constraint;
    return result;
}

BaseAttributeSpec make_graph_resource_spec(GraphResourceSpec spec) {
    BaseAttributeSpec result;
    result.name = std::move(spec.name);
    result.owner = AttributeOwner::graph;
    result.kind = AttributeKind::resource;
    result.generative = spec.generative;
    result.distribution = std::move(spec.distribution);
    result.dtype = spec.dtype;
    result.is_constraint = ConstraintMethodSpec::is_constraint;
    return result;
}

template <typename GraphType>
GraphAttributeBinding bind_impl(
    const GraphType& graph,
    const BaseAttributeSpec& spec,
    const GraphAttribute* definition) {
    return GraphAttributeBinding{
        graph.attr_id(spec.name),
        &graph.graph_attrs(),
        definition};
}

void validate_binding(
    const AttrMap& attributes,
    GraphAttributeBinding binding,
    GraphAttributeOperation operation,
    const GraphAttribute* expected_definition) {
    if (binding.graph_identity != &attributes ||
        binding.definition_identity != expected_definition) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::invalid_binding,
            operation,
            "Graph attribute binding is outside its attribute registry");
    }
}

const AttrValue& get_map_value_prevalidated(
    const AttrMap& attributes,
    GraphAttributeBinding binding,
    GraphAttributeOperation operation) {
    try {
        return attributes.at(binding.value_id);
    } catch (const std::out_of_range&) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::missing_attribute,
            operation,
            "Graph attribute value is missing");
    }
}

void validate_mutable_binding(
    const GraphAttributeMutableSlot& slot,
    const GraphAttribute* expected_definition) {
    if (slot.graph_attributes == nullptr) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::null_batch_slot,
            GraphAttributeOperation::set_data_batch,
            "Graph batch contains a null mutable attribute map");
    }

    validate_binding(
        *slot.graph_attributes,
        slot.binding,
        GraphAttributeOperation::set_data_batch,
        expected_definition);
}

std::size_t batch_worker_count(
    std::size_t count,
    std::size_t configured_workers) noexcept {
    if (count == 0U) {
        return 0U;
    }
    if (configured_workers <= 1U) {
        return 1U;
    }

    const unsigned int detected = std::thread::hardware_concurrency();
    const std::size_t allowance = detected == 0U
        ? configured_workers
        : static_cast<std::size_t>(detected);
    return std::min(
        count,
        std::min(configured_workers, std::max(std::size_t{1U}, allowance)));
}

template <typename Function>
void for_contiguous_blocks(
    std::size_t count,
    std::size_t configured_workers,
    Function&& function) {
    const std::size_t worker_count =
        batch_worker_count(count, configured_workers);
    if (worker_count == 0U) {
        return;
    }
    if (worker_count == 1U) {
        function(0U, count);
        return;
    }

    const std::size_t block_size = count / worker_count;
    const std::size_t remainder = count % worker_count;
    auto bounds = [block_size, remainder](std::size_t worker) {
        const std::size_t begin =
            worker * block_size + std::min(worker, remainder);
        const std::size_t length =
            block_size + (worker < remainder ? 1U : 0U);
        return std::pair<std::size_t, std::size_t>{begin, begin + length};
    };

    std::vector<std::exception_ptr> errors(worker_count);
    std::vector<std::thread> threads;
    try {
        threads.reserve(worker_count - 1U);
    } catch (...) {
        function(0U, count);
        return;
    }

    auto run_block = [&](std::size_t worker) noexcept {
        const auto [begin, end] = bounds(worker);
        try {
            function(begin, end);
        } catch (...) {
            errors[worker] = std::current_exception();
        }
    };

    std::size_t next_worker = 1U;
    for (; next_worker < worker_count; ++next_worker) {
        try {
            threads.emplace_back([&, worker = next_worker] {
                run_block(worker);
            });
        } catch (...) {
            break;
        }
    }

    run_block(0U);
    for (std::size_t worker = next_worker;
         worker < worker_count;
         ++worker) {
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
}

AttributeNumber numeric_resource_value(
    const AttrValue& value,
    GraphAttributeOperation operation) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return *floating;
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean;
    }
    throw GraphAttributeException(
        GraphAttributeErrorCode::non_numeric_resource,
        operation,
        "Graph resource value is not numeric");
}

AttrValue graph_attr_value(AttributeNumber value) {
    return std::visit(
        [](auto item) -> AttrValue { return item; },
        std::move(value));
}

void validate_update_operation(ResourceUpdateOperation operation) {
    switch (operation) {
    case ResourceUpdateOperation::add:
    case ResourceUpdateOperation::subtract:
        return;
    }
    throw AttributeMethodException(
        AttributeMethodErrorCode::unsupported_update_operation,
        AttributeMethodOperation::resolve_update,
        "Unsupported graph resource update operation");
}

}  // namespace

GraphAttributeException::GraphAttributeException(
    GraphAttributeErrorCode code,
    GraphAttributeOperation operation,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation) {}

GraphAttributeErrorCode GraphAttributeException::code() const noexcept {
    return code_;
}

GraphAttributeOperation GraphAttributeException::operation() const noexcept {
    return operation_;
}

GraphAttribute::GraphAttribute(BaseAttributeSpec spec)
    : BaseAttribute(checked_graph_spec(std::move(spec))) {}

GraphAttributeBinding GraphAttribute::bind(const Graph& graph) const {
    return bind_impl(graph, spec(), this);
}

GraphAttributeBinding GraphAttribute::bind(const DiGraph& graph) const {
    return bind_impl(graph, spec(), this);
}

const AttrValue& GraphAttribute::get(
    const Graph& graph,
    GraphAttributeBinding binding) const {
    validate_binding(
        graph.graph_attrs(), binding, GraphAttributeOperation::get, this);
    return get_map_value_prevalidated(
        graph.graph_attrs(), binding, GraphAttributeOperation::get);
}

const AttrValue& GraphAttribute::get(
    const DiGraph& graph,
    GraphAttributeBinding binding) const {
    validate_binding(
        graph.graph_attrs(), binding, GraphAttributeOperation::get, this);
    return get_map_value_prevalidated(
        graph.graph_attrs(), binding, GraphAttributeOperation::get);
}

const AttrValue& GraphAttribute::get_data(
    const Graph& graph,
    GraphAttributeBinding binding) const {
    validate_binding(
        graph.graph_attrs(),
        binding,
        GraphAttributeOperation::get_data,
        this);
    return get_map_value_prevalidated(
        graph.graph_attrs(), binding, GraphAttributeOperation::get_data);
}

const AttrValue& GraphAttribute::get_data(
    const DiGraph& graph,
    GraphAttributeBinding binding) const {
    validate_binding(
        graph.graph_attrs(),
        binding,
        GraphAttributeOperation::get_data,
        this);
    return get_map_value_prevalidated(
        graph.graph_attrs(), binding, GraphAttributeOperation::get_data);
}

void GraphAttribute::set_data(
    Graph& graph,
    const AttrValue& value,
    GraphAttributeBinding binding) const {
    validate_binding(
        graph.graph_attrs(),
        binding,
        GraphAttributeOperation::set_data,
        this);
    graph.graph_attrs().set(binding.value_id, value);
}

void GraphAttribute::set_data(
    DiGraph& graph,
    const AttrValue& value,
    GraphAttributeBinding binding) const {
    validate_binding(
        graph.graph_attrs(),
        binding,
        GraphAttributeOperation::set_data,
        this);
    graph.graph_attrs().set(binding.value_id, value);
}

std::vector<AttrValue> GraphAttribute::get_data_batch(
    const std::vector<GraphAttributeConstSlot>& slots,
    std::size_t workers) const {
    for (const GraphAttributeConstSlot& slot : slots) {
        if (slot.graph_attributes == nullptr) {
            throw GraphAttributeException(
                GraphAttributeErrorCode::null_batch_slot,
                GraphAttributeOperation::get_data_batch,
                "Graph batch contains a null const attribute map");
        }
        validate_binding(
            *slot.graph_attributes,
            slot.binding,
            GraphAttributeOperation::get_data_batch,
            this);
    }

    std::vector<AttrValue> result;
    if (workers <= 1U || slots.size() <= 1U) {
        result.reserve(slots.size());
        for (const GraphAttributeConstSlot& slot : slots) {
            result.emplace_back(get_map_value_prevalidated(
                *slot.graph_attributes,
                slot.binding,
                GraphAttributeOperation::get_data_batch));
        }
        return result;
    }

    result.resize(slots.size());
    for_contiguous_blocks(
        slots.size(), workers,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                result[index] = get_map_value_prevalidated(
                    *slots[index].graph_attributes,
                    slots[index].binding,
                    GraphAttributeOperation::get_data_batch);
            }
        });
    return result;
}

void GraphAttribute::set_data_batch(
    const std::vector<GraphAttributeMutableSlot>& slots,
    const std::vector<AttrValue>& values,
    std::size_t workers) const {
    if (slots.size() != values.size()) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::invalid_batch_shape,
            GraphAttributeOperation::set_data_batch,
            "Graph batch slots and values have different lengths");
    }

    auto assign = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            slots[index].graph_attributes->set(
                slots[index].binding.value_id,
                values[index]);
        }
    };

    if (workers <= 1U || slots.size() <= 1U) {
        for (const GraphAttributeMutableSlot& slot : slots) {
            validate_mutable_binding(slot, this);
        }
        assign(0U, slots.size());
        return;
    }

    std::unordered_set<AttrMap*> maps;
    maps.reserve(slots.size());
    bool duplicate_map = false;
    for (const GraphAttributeMutableSlot& slot : slots) {
        validate_mutable_binding(slot, this);
        if (!maps.insert(slot.graph_attributes).second) {
            duplicate_map = true;
        }
    }

    if (duplicate_map) {
        assign(0U, slots.size());
        return;
    }
    for_contiguous_blocks(slots.size(), workers, assign);
}

GraphStatusAttribute::GraphStatusAttribute(GraphStatusSpec spec)
    : GraphAttribute(make_graph_status_spec(std::move(spec))) {}

GraphExtremaAttribute::GraphExtremaAttribute(GraphExtremaSpec spec)
    : GraphAttribute(make_graph_extrema_spec(spec)),
      originator_name_(std::move(*spec.originator_name)),
      originator_id_(spec.originator_id) {}

std::string_view GraphExtremaAttribute::originator_name() const noexcept {
    return originator_name_;
}

AttributeDefinitionId GraphExtremaAttribute::originator_id() const noexcept {
    return originator_id_;
}

std::vector<AttrValue>
GraphExtremaAttribute::generate_from_resolved_originator(
    const Graph& graph,
    const LinkAttribute& originator,
    LinkAttributeBinding originator_binding,
    std::size_t workers) const {
    return originator.get_data(graph, originator_binding, workers);
}

std::vector<AttrValue>
GraphExtremaAttribute::generate_from_resolved_originator(
    const DiGraph& graph,
    const LinkAttribute& originator,
    LinkAttributeBinding originator_binding,
    std::size_t workers) const {
    return originator.get_data(graph, originator_binding, workers);
}

GraphResourceAttribute::GraphResourceAttribute(GraphResourceSpec spec)
    : GraphAttribute(make_graph_resource_spec(spec)),
      restriction_(spec.restriction),
      checking_level_(spec.checking_level) {}

ConstraintRestriction GraphResourceAttribute::restriction() const noexcept {
    return restriction_;
}

CheckingLevel GraphResourceAttribute::checking_level() const noexcept {
    return checking_level_;
}

bool GraphResourceAttribute::update(
    AttrMap& target,
    AttrId target_id,
    const AttrMap& operand,
    AttrId operand_id,
    ResourceUpdateOperation operation,
    bool safe) const {
    (void)safe;
    validate_update_operation(operation);

    AttrValue* target_value = target.find(target_id);
    if (target_value == nullptr) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::missing_resource_value,
            GraphAttributeOperation::update_resource,
            "Target graph resource value is missing");
    }
    const AttrValue* operand_value = operand.find(operand_id);
    if (operand_value == nullptr) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::missing_resource_value,
            GraphAttributeOperation::update_resource,
            "Operand graph resource value is missing");
    }

    AttributeNumber target_number = numeric_resource_value(
        *target_value, GraphAttributeOperation::update_resource);
    const AttributeNumber operand_number = numeric_resource_value(
        *operand_value, GraphAttributeOperation::update_resource);
    update_resource_value(
        operand_number, target_number, operation, false, spec().name);
    *target_value = graph_attr_value(std::move(target_number));
    return true;
}

SatisfiabilityResult
GraphResourceAttribute::check_constraint_satisfiability(
    const AttrMap& virtual_graph,
    AttrId virtual_id,
    const AttrMap& physical_graph,
    AttrId physical_id,
    ComparisonOperation method) const {
    const AttrValue* virtual_value = virtual_graph.find(virtual_id);
    const AttrValue* physical_value = physical_graph.find(physical_id);
    if (virtual_value == nullptr || physical_value == nullptr) {
        throw GraphAttributeException(
            GraphAttributeErrorCode::missing_resource_value,
            GraphAttributeOperation::check_resource,
            "Graph resource value is missing");
    }

    const AttributeNumber virtual_number = numeric_resource_value(
        *virtual_value, GraphAttributeOperation::check_resource);
    const AttributeNumber physical_number = numeric_resource_value(
        *physical_value, GraphAttributeOperation::check_resource);
    return calculate_satisfiability_values(
        virtual_number, physical_number, method, restriction_);
}

}  // namespace virne::network::attribute
