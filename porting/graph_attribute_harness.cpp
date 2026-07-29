#include "attribute/graph_attribute.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

double double_from_bits(std::uint64_t bits) noexcept {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::string hex_bits(double value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16)
           << double_bits(value);
    return output.str();
}

std::string hex_encode(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(value.size() * 2U, '0');
    for (std::size_t index = 0U; index < value.size(); ++index) {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2U] = digits[byte >> 4U];
        result[index * 2U + 1U] = digits[byte & 15U];
    }
    return result;
}

std::string attr_value(const AttrValue& value);

std::string attr_list_value(const AttrListPtr& list) {
    if (!list) {
        return "l:null";
    }
    std::string result = "l:[";
    for (std::size_t index = 0U; index < list->values.size(); ++index) {
        if (index != 0U) {
            result.push_back(';');
        }
        result.append(attr_value(list->values[index]));
    }
    result.push_back(']');
    return result;
}

std::string attr_value(const AttrValue& value) {
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return "i:" + std::to_string(*integer);
    }
    if (const auto* floating = std::get_if<double>(&value)) {
        return "d:" + hex_bits(*floating);
    }
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return std::string("b:") + (*boolean ? "1" : "0");
    }
    if (const auto* text = std::get_if<std::string>(&value)) {
        return "s:" + hex_encode(*text);
    }
    if (const auto* list = std::get_if<AttrListPtr>(&value)) {
        return attr_list_value(*list);
    }
    return "object";
}

std::string attribute_number(const attribute::AttributeNumber& value) {
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return std::string("b:") + (*boolean ? "1" : "0");
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return "i:" + std::to_string(*integer);
    }
    return "d:" + hex_bits(std::get<double>(value));
}

std::string attr_values(const std::vector<AttrValue>& values) {
    std::string result;
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result.append(attr_value(values[index]));
    }
    return result;
}

attribute::BaseAttributeSpec graph_spec(
    std::string name,
    attribute::AttributeKind kind = attribute::AttributeKind::status) {
    attribute::BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = attribute::AttributeOwner::graph;
    result.kind = kind;
    return result;
}

std::string graph_error_name(attribute::GraphAttributeErrorCode code) noexcept {
    using Code = attribute::GraphAttributeErrorCode;
    switch (code) {
        case Code::invalid_graph_spec: return "invalid_graph_spec";
        case Code::missing_attribute: return "missing_attribute";
        case Code::invalid_binding: return "invalid_binding";
        case Code::null_batch_slot: return "null_batch_slot";
        case Code::invalid_batch_shape: return "invalid_batch_shape";
        case Code::missing_originator: return "missing_originator";
        case Code::missing_resource_value: return "missing_resource_value";
        case Code::non_numeric_resource: return "non_numeric_resource";
    }
    return "invalid_graph_error";
}

std::string method_error_name(attribute::AttributeMethodErrorCode code) noexcept {
    using Code = attribute::AttributeMethodErrorCode;
    switch (code) {
        case Code::unsupported_update_operation:
            return "unsupported_update_operation";
        case Code::numeric_range:
            return "numeric_range";
        case Code::insufficient_resource:
            return "insufficient_resource";
        default:
            return "attribute_method_error";
    }
}

template <typename Callable>
void emit_case(std::string_view name, Callable&& callable) {
    try {
        const std::string payload = callable();
        std::cout << "case=" << name << "|ok|" << payload << '\n';
    } catch (const attribute::GraphAttributeException& error) {
        std::cout << "case=" << name << "|error|"
                  << graph_error_name(error.code()) << '|'
                  << hex_encode(error.what()) << '\n';
    } catch (const attribute::AttributeMethodException& error) {
        std::cout << "case=" << name << "|error|"
                  << method_error_name(error.code()) << '|'
                  << hex_encode(error.what()) << '\n';
    } catch (const std::exception& error) {
        std::cout << "case=" << name << "|error|dependency_error|"
                  << hex_encode(error.what()) << '\n';
    }
}

void emit_construction() {
    attribute::GraphStatusSpec status_spec;
    status_spec.name = "alive";
    const attribute::GraphStatusAttribute status(status_spec);
    std::cout << "case=status_fields|ok|alive,graph,status,"
              << (status.spec().generative ? '1' : '0') << ','
              << (status.spec().is_constraint.value_or(true) ? '1' : '0')
              << '\n';

    emit_case("extrema_missing_originator", [] {
        attribute::GraphExtremaSpec spec;
        spec.name = "capacity_max";
        static_cast<void>(attribute::GraphExtremaAttribute(std::move(spec)));
        return std::string{};
    });

    attribute::GraphExtremaSpec extrema_spec;
    extrema_spec.name = "capacity_max";
    extrema_spec.originator_name = "capacity";
    extrema_spec.originator_id = 17U;
    const attribute::GraphExtremaAttribute extrema(extrema_spec);
    std::cout << "case=extrema_fields|ok|capacity_max,graph,extrema,"
              << extrema.originator_name() << ',' << extrema.originator_id()
              << ",0,link\n";

    attribute::GraphResourceSpec hard_spec;
    hard_spec.name = "capacity";
    const attribute::GraphResourceAttribute hard(hard_spec);
    std::cout << "case=resource_fields_hard|ok|capacity,graph,resource,hard,graph,"
              << (hard.spec().is_constraint.value_or(false) ? '1' : '0')
              << '\n';

    attribute::GraphResourceSpec precedence_spec;
    precedence_spec.name = "capacity";
    precedence_spec.restriction = attribute::ConstraintRestriction::soft;
    const attribute::GraphResourceAttribute precedence(precedence_spec);
    std::cout << "case=resource_fields_precedence|ok|capacity,graph,resource,"
                 "soft,graph,"
              << (precedence.spec().is_constraint.value_or(false) ? '1' : '0')
              << '\n';
}

std::vector<AttrValue> mixed_lanes() {
    return {
        AttrValue{std::numeric_limits<std::int64_t>::min()},
        AttrValue{-0.0},
        AttrValue{double_from_bits(UINT64_C(0x7ff8000000001234))},
        AttrValue{true},
        AttrValue{std::string{"a|b"}},
        make_attr_list({
            AttrValue{std::int64_t{7}}, AttrValue{2.5}, AttrValue{false},
            AttrValue{std::string{"z"}}}),
    };
}

template <typename GraphType>
void emit_adapter(std::string_view prefix) {
    GraphType graph;
    const attribute::GraphAttribute item(graph_spec("mixed"));
    const auto binding = item.bind(graph);
    std::vector<AttrValue> observed;
    for (const AttrValue& value : mixed_lanes()) {
        item.set_data(graph, value, binding);
        observed.push_back(item.get_data(graph, binding));
    }
    std::cout << "case=" << prefix << "_lanes|ok|"
              << attr_values(observed) << '\n';

    AttrValue identity_value = make_attr_list({
        AttrValue{std::int64_t{1}}, AttrValue{-0.0},
        AttrValue{std::string{"live"}}});
    item.set_data(graph, identity_value, binding);
    const AttrValue& direct = item.get(graph, binding);
    const AttrValue& data = item.get_data(graph, binding);
    const auto& original_pointer = std::get<AttrListPtr>(identity_value);
    const auto& stored_pointer = std::get<AttrListPtr>(direct);
    std::cout << "case=" << prefix << "_identity|ok|"
              << (&direct == &data ? '1' : '0') << ','
              << (original_pointer == stored_pointer ? '1' : '0') << ','
              << attr_value(direct) << '\n';

    item.set_data(graph, AttrValue{std::int64_t{11}}, binding);
    item.set_data(graph, AttrValue{std::string{"overwritten"}}, binding);
    std::cout << "case=" << prefix << "_overwrite|ok|"
              << attr_value(item.get_data(graph, binding)) << '\n';

    const attribute::GraphAttribute empty(graph_spec(""));
    const auto empty_binding = empty.bind(graph);
    empty.set_data(graph, AttrValue{false}, empty_binding);
    std::cout << "case=" << prefix << "_empty_name|ok|"
              << attr_value(empty.get(graph, empty_binding)) << '\n';

    GraphType missing;
    const auto missing_binding = item.bind(missing);
    emit_case(std::string(prefix) + "_missing", [&] {
        return attr_value(item.get_data(missing, missing_binding));
    });
}

void emit_different_registries() {
    Graph first;
    Graph second;
    static_cast<void>(first.attr_id("padding"));
    static_cast<void>(second.attr_id("other_padding"));
    static_cast<void>(second.attr_id("second_padding"));
    const attribute::GraphAttribute item(graph_spec("mixed"));
    const auto first_binding = item.bind(first);
    const auto second_binding = item.bind(second);
    item.set_data(first, AttrValue{std::int64_t{31}}, first_binding);
    item.set_data(second, AttrValue{std::int64_t{47}}, second_binding);
    std::cout << "case=different_registries|ok|"
              << attr_value(item.get(first, first_binding)) << ','
              << attr_value(item.get(second, second_binding)) << '\n';
}

std::vector<AttrValue> batch_values() {
    return {
        AttrValue{std::int64_t{1}}, AttrValue{-0.0},
        AttrValue{double_from_bits(UINT64_C(0x7ff8000000004321))},
        AttrValue{true}, AttrValue{std::string{"five"}},
        make_attr_list({AttrValue{std::int64_t{6}}, AttrValue{7.5}}),
    };
}

std::string batch_at_workers(std::size_t workers) {
    const std::vector<AttrValue> values = batch_values();
    std::vector<Graph> graphs;
    graphs.reserve(values.size());
    for (std::size_t index = 0U; index < values.size(); ++index) {
        graphs.emplace_back();
        static_cast<void>(
            graphs.back().attr_id("padding_" + std::to_string(index)));
    }
    const attribute::GraphAttribute item(graph_spec("batch_value"));
    std::vector<attribute::GraphAttributeMutableSlot> mutable_slots;
    std::vector<attribute::GraphAttributeConstSlot> const_slots;
    mutable_slots.reserve(graphs.size());
    const_slots.reserve(graphs.size());
    for (Graph& graph : graphs) {
        const auto binding = item.bind(graph);
        mutable_slots.push_back({&graph.graph_attrs(), binding});
        const_slots.push_back({&graph.graph_attrs(), binding});
    }
    item.set_data_batch(mutable_slots, values, workers);
    return attr_values(item.get_data_batch(const_slots, workers));
}

void emit_batches() {
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        std::cout << "case=batch_w" << workers << "|ok|"
                  << batch_at_workers(workers) << '\n';
    }
    const attribute::GraphAttribute item(graph_spec("batch_value"));
    item.set_data_batch({}, {}, 8U);
    std::cout << "case=batch_empty|ok|"
              << attr_values(item.get_data_batch({}, 8U)) << '\n';

    Graph duplicate;
    const auto duplicate_binding = item.bind(duplicate);
    const std::vector<attribute::GraphAttributeMutableSlot> duplicate_slots{
        {&duplicate.graph_attrs(), duplicate_binding},
        {&duplicate.graph_attrs(), duplicate_binding},
        {&duplicate.graph_attrs(), duplicate_binding},
    };
    item.set_data_batch(
        duplicate_slots,
        {AttrValue{std::int64_t{1}}, AttrValue{2.5}, AttrValue{true}},
        8U);
    std::cout << "case=batch_duplicate_map|ok|"
              << attr_value(duplicate.graph_attrs().at(duplicate_binding.value_id))
              << '\n';

    emit_case("batch_get_null_slot", [&] {
        return attr_values(item.get_data_batch({{nullptr, {}}}, 8U));
    });
    emit_case("batch_set_null_slot", [&] {
        item.set_data_batch(
            {{nullptr, {}}}, {AttrValue{std::int64_t{1}}}, 8U);
        return std::string{};
    });
    emit_case("batch_shape_mismatch", [&] {
        item.set_data_batch(
            {{&duplicate.graph_attrs(), duplicate_binding}}, {}, 8U);
        return std::string{};
    });
}

attribute::LinkAttribute capacity_originator() {
    attribute::BaseAttributeSpec spec;
    spec.name = "capacity";
    spec.owner = attribute::AttributeOwner::link;
    spec.kind = attribute::AttributeKind::resource;
    return attribute::LinkAttribute(std::move(spec));
}

template <typename GraphType>
std::string extrema_value(std::size_t workers) {
    GraphType graph(
        4U,
        std::vector<EdgeEndpoints>{{0U, 1U}, {2U, 3U}, {1U, 3U}, {3U, 3U}});
    const attribute::LinkAttribute originator = capacity_originator();
    const auto originator_binding = originator.bind(graph);
    originator.set_data_dense(
        graph,
        {AttrValue{std::int64_t{1}}, AttrValue{-0.0}, AttrValue{true},
         AttrValue{std::string{"link"}}},
        originator_binding,
        1U);
    attribute::GraphExtremaSpec spec;
    spec.name = "capacity_max";
    spec.originator_name = "capacity";
    spec.originator_id = 17U;
    const attribute::GraphExtremaAttribute extrema(spec);
    return attr_values(extrema.generate_from_resolved_originator(
        graph, originator, originator_binding, workers));
}

void emit_extrema() {
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        std::cout << "case=extrema_graph_w" << workers << "|ok|"
                  << extrema_value<Graph>(workers) << '\n';
    }
    std::cout << "case=extrema_digraph_w8|ok|"
              << extrema_value<DiGraph>(8U) << '\n';
}

attribute::GraphResourceAttribute resource(
    attribute::ConstraintRestriction restriction) {
    attribute::GraphResourceSpec spec;
    spec.name = "capacity";
    spec.restriction = restriction;
    return attribute::GraphResourceAttribute(std::move(spec));
}

void emit_resource_check(
    std::string_view name,
    const attribute::GraphResourceAttribute& policy,
    AttrValue virtual_value,
    AttrValue physical_value,
    attribute::ComparisonOperation operation) {
    Graph virtual_graph;
    Graph physical_graph;
    const auto virtual_binding = policy.bind(virtual_graph);
    const auto physical_binding = policy.bind(physical_graph);
    virtual_graph.graph_attrs().set(
        virtual_binding.value_id, std::move(virtual_value));
    physical_graph.graph_attrs().set(
        physical_binding.value_id, std::move(physical_value));
    const auto checked = policy.check_constraint_satisfiability(
        virtual_graph.graph_attrs(), virtual_binding.value_id,
        physical_graph.graph_attrs(), physical_binding.value_id,
        operation);
    std::cout << "case=" << name << "|ok|" << (checked.flag ? '1' : '0')
              << ',' << attribute_number(checked.offset) << '\n';
}

void emit_resource_checks(
    const attribute::GraphResourceAttribute& hard,
    const attribute::GraphResourceAttribute& soft) {
    emit_resource_check(
        "resource_le_hard_pass", hard, AttrValue{std::int64_t{7}},
        AttrValue{10.5}, attribute::ComparisonOperation::less_equal);
    emit_resource_check(
        "resource_le_hard_fail", hard, AttrValue{12.0}, AttrValue{10.5},
        attribute::ComparisonOperation::less_equal);
    emit_resource_check(
        "resource_le_soft_fail", soft, AttrValue{12.0}, AttrValue{10.5},
        attribute::ComparisonOperation::less_equal);
    emit_resource_check(
        "resource_ge_hard_pass", hard, AttrValue{std::int64_t{11}},
        AttrValue{std::int64_t{10}},
        attribute::ComparisonOperation::greater_equal);
    emit_resource_check(
        "resource_ge_hard_fail", hard, AttrValue{std::int64_t{9}},
        AttrValue{std::int64_t{10}},
        attribute::ComparisonOperation::greater_equal);
    emit_resource_check(
        "resource_eq_hard_pass", hard, AttrValue{-0.0}, AttrValue{0.0},
        attribute::ComparisonOperation::equal);
    emit_resource_check(
        "resource_eq_soft_fail", soft, AttrValue{4.5}, AttrValue{4.0},
        attribute::ComparisonOperation::equal);
    emit_resource_check(
        "resource_bool_promotion", hard, AttrValue{true}, AttrValue{false},
        attribute::ComparisonOperation::less_equal);

    Graph bad_virtual;
    Graph missing_physical;
    const auto bad_virtual_binding = hard.bind(bad_virtual);
    const auto missing_physical_binding = hard.bind(missing_physical);
    bad_virtual.graph_attrs().set(
        bad_virtual_binding.value_id, AttrValue{std::string{"bad"}});
    emit_case("resource_check_physical_missing_precedence", [&] {
        const auto checked = hard.check_constraint_satisfiability(
            bad_virtual.graph_attrs(), bad_virtual_binding.value_id,
            missing_physical.graph_attrs(), missing_physical_binding.value_id);
        return attribute_number(checked.offset);
    });

    Graph missing_virtual;
    Graph bad_physical;
    const auto missing_virtual_binding = hard.bind(missing_virtual);
    const auto bad_physical_binding = hard.bind(bad_physical);
    bad_physical.graph_attrs().set(
        bad_physical_binding.value_id, AttrValue{std::string{"bad"}});
    emit_case("resource_check_virtual_missing_precedence", [&] {
        const auto checked = hard.check_constraint_satisfiability(
            missing_virtual.graph_attrs(), missing_virtual_binding.value_id,
            bad_physical.graph_attrs(), bad_physical_binding.value_id);
        return attribute_number(checked.offset);
    });

    Graph numeric_physical;
    const auto numeric_physical_binding = hard.bind(numeric_physical);
    numeric_physical.graph_attrs().set(
        numeric_physical_binding.value_id, AttrValue{std::int64_t{10}});
    emit_case("resource_check_nonnumeric", [&] {
        const auto checked = hard.check_constraint_satisfiability(
            bad_virtual.graph_attrs(), bad_virtual_binding.value_id,
            numeric_physical.graph_attrs(), numeric_physical_binding.value_id);
        return attribute_number(checked.offset);
    });
}

void emit_resource_updates(const attribute::GraphResourceAttribute& hard) {
    Graph target;
    Graph operand;
    const auto target_binding = hard.bind(target);
    const auto operand_binding = hard.bind(operand);
    target.graph_attrs().set(target_binding.value_id, AttrValue{std::int64_t{10}});
    operand.graph_attrs().set(operand_binding.value_id, AttrValue{std::int64_t{3}});
    static_cast<void>(hard.update(
        target.graph_attrs(), target_binding.value_id,
        operand.graph_attrs(), operand_binding.value_id,
        attribute::ResourceUpdateOperation::subtract, true));
    static_cast<void>(hard.update(
        target.graph_attrs(), target_binding.value_id,
        operand.graph_attrs(), operand_binding.value_id,
        attribute::ResourceUpdateOperation::add, true));
    std::cout << "case=resource_update_roundtrip|ok|"
              << attr_value(target.graph_attrs().at(target_binding.value_id))
              << '\n';

    target.graph_attrs().set(target_binding.value_id, AttrValue{std::int64_t{2}});
    static_cast<void>(hard.update(
        target.graph_attrs(), target_binding.value_id,
        operand.graph_attrs(), operand_binding.value_id,
        attribute::ResourceUpdateOperation::subtract, true));
    std::cout << "case=resource_update_negative_safe_ignored|ok|"
              << attr_value(target.graph_attrs().at(target_binding.value_id))
              << '\n';

    target.graph_attrs().set(target_binding.value_id, AttrValue{false});
    operand.graph_attrs().set(operand_binding.value_id, AttrValue{true});
    static_cast<void>(hard.update(
        target.graph_attrs(), target_binding.value_id,
        operand.graph_attrs(), operand_binding.value_id,
        attribute::ResourceUpdateOperation::add, false));
    std::cout << "case=resource_update_bool_promotion|ok|"
              << attr_value(target.graph_attrs().at(target_binding.value_id))
              << '\n';

    Graph alias;
    const auto alias_binding = hard.bind(alias);
    alias.graph_attrs().set(alias_binding.value_id, AttrValue{std::int64_t{7}});
    static_cast<void>(hard.update(
        alias.graph_attrs(), alias_binding.value_id,
        alias.graph_attrs(), alias_binding.value_id,
        attribute::ResourceUpdateOperation::add, true));
    const std::string doubled =
        attr_value(alias.graph_attrs().at(alias_binding.value_id));
    static_cast<void>(hard.update(
        alias.graph_attrs(), alias_binding.value_id,
        alias.graph_attrs(), alias_binding.value_id,
        attribute::ResourceUpdateOperation::subtract, true));
    std::cout << "case=resource_update_self_alias|ok|" << doubled << ','
              << attr_value(alias.graph_attrs().at(alias_binding.value_id))
              << '\n';

    Graph missing_target;
    Graph bad_operand;
    const auto missing_target_binding = hard.bind(missing_target);
    const auto bad_operand_binding = hard.bind(bad_operand);
    bad_operand.graph_attrs().set(
        bad_operand_binding.value_id, AttrValue{std::string{"bad"}});
    emit_case("resource_update_target_missing_precedence", [&] {
        static_cast<void>(hard.update(
            missing_target.graph_attrs(), missing_target_binding.value_id,
            bad_operand.graph_attrs(), bad_operand_binding.value_id,
            attribute::ResourceUpdateOperation::add, true));
        return std::string{};
    });

    Graph bad_target;
    Graph missing_operand;
    const auto bad_target_binding = hard.bind(bad_target);
    const auto missing_operand_binding = hard.bind(missing_operand);
    bad_target.graph_attrs().set(
        bad_target_binding.value_id, AttrValue{std::string{"bad"}});
    emit_case("resource_update_operand_missing_precedence", [&] {
        static_cast<void>(hard.update(
            bad_target.graph_attrs(), bad_target_binding.value_id,
            missing_operand.graph_attrs(), missing_operand_binding.value_id,
            attribute::ResourceUpdateOperation::add, true));
        return std::string{};
    });

    emit_case("resource_update_invalid_operation", [&] {
        static_cast<void>(hard.update(
            missing_target.graph_attrs(), missing_target_binding.value_id,
            missing_operand.graph_attrs(), missing_operand_binding.value_id,
            static_cast<attribute::ResourceUpdateOperation>(255U), true));
        return std::string{};
    });

    Graph overflow_target;
    Graph overflow_operand;
    const auto overflow_target_binding = hard.bind(overflow_target);
    const auto overflow_operand_binding = hard.bind(overflow_operand);
    overflow_target.graph_attrs().set(
        overflow_target_binding.value_id,
        AttrValue{std::numeric_limits<std::int64_t>::max()});
    overflow_operand.graph_attrs().set(
        overflow_operand_binding.value_id, AttrValue{std::int64_t{1}});
    emit_case("resource_update_overflow", [&] {
        static_cast<void>(hard.update(
            overflow_target.graph_attrs(), overflow_target_binding.value_id,
            overflow_operand.graph_attrs(), overflow_operand_binding.value_id,
            attribute::ResourceUpdateOperation::add, true));
        return std::string{};
    });
}

void emit_resources() {
    const auto hard = resource(attribute::ConstraintRestriction::hard);
    const auto soft = resource(attribute::ConstraintRestriction::soft);
    emit_resource_checks(hard, soft);
    emit_resource_updates(hard);
}

}  // namespace

int main() {
    emit_construction();
    emit_adapter<Graph>("graph");
    emit_adapter<DiGraph>("digraph");
    emit_different_registries();
    emit_batches();
    emit_extrema();
    emit_resources();
    std::cout << "status=PASS\n";
    return 0;
}
