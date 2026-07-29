#include "attribute/node_attribute.h"

#include "numpy_random_state.h"

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
namespace utils = virne::utils;

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
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
    if (const auto* string = std::get_if<std::string>(&value)) {
        return "s:" + hex_encode(*string);
    }
    return "recursive";
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

std::string values(const std::vector<AttrValue>& items) {
    std::string result;
    for (std::size_t index = 0U; index < items.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result.append(attr_value(items[index]));
    }
    return result;
}

std::string positions(
    const std::vector<attribute::NodePositionValue>& items) {
    std::string result;
    for (std::size_t index = 0U; index < items.size(); ++index) {
        if (index != 0U) {
            result.push_back(';');
        }
        result.append(attribute_number(items[index].x));
        result.push_back(',');
        result.append(attribute_number(items[index].y));
        result.append(",d:");
        result.append(hex_bits(items[index].radius));
    }
    return result;
}

attribute::BaseAttributeSpec node_spec(
    std::string name,
    attribute::AttributeKind kind = attribute::AttributeKind::status) {
    attribute::BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = attribute::AttributeOwner::node;
    result.kind = kind;
    return result;
}

std::string node_error_name(attribute::NodeAttributeErrorCode code) noexcept {
    using Code = attribute::NodeAttributeErrorCode;
    switch (code) {
        case Code::invalid_node_spec: return "invalid_node_spec";
        case Code::node_out_of_range: return "node_out_of_range";
        case Code::missing_attribute: return "missing_attribute";
        case Code::dense_data_too_short: return "dense_data_too_short";
        case Code::missing_originator: return "missing_originator";
        case Code::missing_resource_value: return "missing_resource_value";
        case Code::non_numeric_resource: return "non_numeric_resource";
        case Code::empty_position_network: return "empty_position_network";
        case Code::missing_position_data: return "missing_position_data";
    }
    return "invalid_node_error";
}

template <typename Callable>
void emit_node_case(std::string_view name, Callable&& callable) {
    try {
        const std::string payload = callable();
        std::cout << "case=" << name << "|ok|" << payload << '\n';
    } catch (const attribute::NodeAttributeException& error) {
        std::cout << "case=" << name << "|error|"
                  << node_error_name(error.code()) << '|'
                  << hex_encode(error.what()) << '\n';
    }
}

void emit_construction() {
    attribute::NodeStatusSpec status_spec;
    status_spec.name = "alive";
    const attribute::NodeStatusAttribute status(status_spec);
    std::cout << "case=status_fields|ok|"
              << status.spec().name << ",node,status,"
              << (status.spec().generative ? '1' : '0') << ","
              << (status.spec().is_constraint.value_or(true) ? '1' : '0') << '\n';

    emit_node_case("extrema_missing_originator", [] {
        attribute::NodeExtremaSpec spec;
        spec.name = "cpu_max";
        static_cast<void>(attribute::NodeExtremaAttribute(std::move(spec)));
        return std::string{};
    });
    attribute::NodeExtremaSpec extrema_spec;
    extrema_spec.name = "cpu_max";
    extrema_spec.originator_name = "cpu";
    extrema_spec.originator_id = 17U;
    const attribute::NodeExtremaAttribute extrema(extrema_spec);
    std::cout << "case=extrema_fields|ok|" << extrema.spec().name
              << ",node,extrema," << extrema.originator_name() << ','
              << extrema.originator_id() << ",0\n";

    attribute::NodeResourceSpec resource_spec;
    resource_spec.name = "cpu";
    resource_spec.restriction = attribute::ConstraintRestriction::soft;
    resource_spec.checking_level = attribute::CheckingLevel::node;
    const attribute::NodeResourceAttribute resource(resource_spec);
    std::cout << "case=resource_fields|ok|cpu,node,resource,soft,node,1\n";

    attribute::NodePositionSpec position_spec;
    position_spec.name = "where";
    position_spec.generative = true;
    position_spec.minimum_radius = -0.25;
    position_spec.maximum_radius = 0.75;
    const attribute::NodePositionAttribute position(position_spec);
    std::cout << "case=position_fields|ok|where,node,position,1,"
              << hex_bits(position.minimum_radius()) << ','
              << hex_bits(position.maximum_radius()) << ",hard,1\n";
}

template <typename GraphType>
void emit_graph_adapter(std::string_view prefix) {
    const attribute::NodeAttribute item(node_spec("mixed"));
    GraphType graph(5U, std::vector<EdgeEndpoints>{});
    const auto binding = item.bind(graph);
    const std::vector<AttrValue> dense = {
        std::int64_t{1}, 2.5, true, std::string{"four"}, std::int64_t{-5},
        std::int64_t{999}};
    item.set_data_dense(graph, dense, binding, 8U);
    std::cout << "case=" << prefix << "_dense|ok|"
              << values(item.get_data(graph, binding, 2U)) << '\n';

    GraphType sparse_graph(5U, std::vector<EdgeEndpoints>{});
    const auto sparse_binding = item.bind(sparse_graph);
    item.set_data(
        sparse_graph,
        {{3U, AttrValue{std::int64_t{30}}},
         {9U, AttrValue{std::int64_t{90}}},
         {1U, AttrValue{std::string{"one"}}},
         {3U, AttrValue{std::int64_t{31}}}},
        sparse_binding);
    std::cout << "case=" << prefix << "_sparse|ok|"
              << values(item.get_data(sparse_graph, sparse_binding, 8U)) << '\n';

    emit_node_case(std::string(prefix) + "_missing_get", [&] {
        return attr_value(item.get(sparse_graph, 0U, sparse_binding));
    });
    emit_node_case(std::string(prefix) + "_short_dense", [&] {
        item.set_data_dense(
            sparse_graph,
            std::vector<AttrValue>(4U, AttrValue{0.0}),
            sparse_binding,
            8U);
        return std::string{};
    });
}

attribute::NodeResourceAttribute resource(
    attribute::ConstraintRestriction restriction) {
    attribute::NodeResourceSpec spec;
    spec.name = "cpu";
    spec.restriction = restriction;
    return attribute::NodeResourceAttribute(std::move(spec));
}

void emit_resource() {
    const auto hard = resource(attribute::ConstraintRestriction::hard);
    const auto soft = resource(attribute::ConstraintRestriction::soft);
    Graph virtual_graph(1U, std::vector<EdgeEndpoints>{});
    Graph physical_graph(1U, std::vector<EdgeEndpoints>{});
    static_cast<void>(physical_graph.attr_id("unrelated"));
    const auto virtual_binding = hard.bind(virtual_graph);
    const auto physical_binding = hard.bind(physical_graph);

    const auto check = [&](std::string_view name,
                           const AttrValue& virtual_value,
                           const AttrValue& physical_value,
                           const attribute::NodeResourceAttribute& policy) {
        virtual_graph.node_attrs(0U).set(
            virtual_binding.value_id, clone_attr_value(virtual_value));
        physical_graph.node_attrs(0U).set(
            physical_binding.value_id, clone_attr_value(physical_value));
        const auto result = policy.check_constraint_satisfiability(
            virtual_graph.node_attrs(0U), virtual_binding.value_id,
            physical_graph.node_attrs(0U), physical_binding.value_id,
            attribute::ComparisonOperation::less_equal);
        std::cout << "case=" << name << "|ok|" << (result.flag ? '1' : '0')
                  << ',' << attribute_number(result.offset) << '\n';
    };
    check("resource_hard_pass", AttrValue{std::int64_t{7}}, AttrValue{10.5}, hard);
    check("resource_hard_fail", AttrValue{12.0}, AttrValue{10.5}, hard);
    check("resource_soft_fail", AttrValue{12.0}, AttrValue{10.5}, soft);

    virtual_graph.node_attrs(0U).set(virtual_binding.value_id, std::int64_t{3});
    physical_graph.node_attrs(0U).set(physical_binding.value_id, std::int64_t{10});
    static_cast<void>(hard.update(
        virtual_graph.node_attrs(0U), virtual_binding.value_id,
        physical_graph.node_attrs(0U), physical_binding.value_id,
        attribute::ResourceUpdateOperation::subtract, true));
    static_cast<void>(hard.update(
        virtual_graph.node_attrs(0U), virtual_binding.value_id,
        physical_graph.node_attrs(0U), physical_binding.value_id,
        attribute::ResourceUpdateOperation::add, true));
    std::cout << "case=resource_update_roundtrip|ok|"
              << attr_value(physical_graph.node_attrs(0U).at(
                     physical_binding.value_id)) << '\n';

    Graph missing(1U, std::vector<EdgeEndpoints>{});
    const auto missing_binding = hard.bind(missing);
    emit_node_case("resource_missing", [&] {
        const auto result = hard.check_constraint_satisfiability(
            missing.node_attrs(0U), missing_binding.value_id,
            physical_graph.node_attrs(0U), physical_binding.value_id);
        return attribute_number(result.offset);
    });
    missing.node_attrs(0U).set(missing_binding.value_id, std::string{"bad"});
    emit_node_case("resource_nonnumeric", [&] {
        const auto result = hard.check_constraint_satisfiability(
            missing.node_attrs(0U), missing_binding.value_id,
            physical_graph.node_attrs(0U), physical_binding.value_id);
        return attribute_number(result.offset);
    });

    Graph absent_physical(1U, std::vector<EdgeEndpoints>{});
    const auto absent_physical_binding = hard.bind(absent_physical);
    emit_node_case("resource_check_missing_precedence", [&] {
        const auto result = hard.check_constraint_satisfiability(
            missing.node_attrs(0U), missing_binding.value_id,
            absent_physical.node_attrs(0U), absent_physical_binding.value_id);
        return attribute_number(result.offset);
    });

    Graph absent_virtual(1U, std::vector<EdgeEndpoints>{});
    Graph bad_physical(1U, std::vector<EdgeEndpoints>{});
    const auto absent_virtual_binding = hard.bind(absent_virtual);
    const auto bad_physical_binding = hard.bind(bad_physical);
    bad_physical.node_attrs(0U).set(
        bad_physical_binding.value_id, std::string{"bad"});
    emit_node_case("resource_add_missing_precedence", [&] {
        static_cast<void>(hard.update(
            absent_virtual.node_attrs(0U), absent_virtual_binding.value_id,
            bad_physical.node_attrs(0U), bad_physical_binding.value_id,
            attribute::ResourceUpdateOperation::add, true));
        return std::string{};
    });
    emit_node_case("resource_sub_unsafe_missing_precedence", [&] {
        static_cast<void>(hard.update(
            absent_virtual.node_attrs(0U), absent_virtual_binding.value_id,
            bad_physical.node_attrs(0U), bad_physical_binding.value_id,
            attribute::ResourceUpdateOperation::subtract, false));
        return std::string{};
    });
    emit_node_case("resource_sub_safe_missing_precedence", [&] {
        static_cast<void>(hard.update(
            missing.node_attrs(0U), missing_binding.value_id,
            absent_physical.node_attrs(0U), absent_physical_binding.value_id,
            attribute::ResourceUpdateOperation::subtract, true));
        return std::string{};
    });
}

utils::DistributionSpec uniform_float(double low, double high) {
    utils::DistributionSpec result;
    result.kind = utils::DistributionKind::uniform;
    result.low = utils::DatasetScalar{low};
    result.high = utils::DatasetScalar{high};
    return result;
}

utils::DistributionSpec uniform_integer(std::int64_t low, std::int64_t high) {
    utils::DistributionSpec result;
    result.kind = utils::DistributionKind::uniform;
    result.low = utils::DatasetScalar{low};
    result.high = utils::DatasetScalar{high};
    return result;
}

void emit_position_case(
    std::string_view name,
    attribute::NodePositionSpec spec,
    std::size_t count,
    std::uint32_t seed,
    std::size_t workers) {
    const attribute::NodePositionAttribute position(std::move(spec));
    NumpyRandomState rng(seed);
    try {
        const auto output = position.generate_positions(
            attribute::NetworkCardinality{count, 0U}, rng, workers);
        std::cout << "case=" << name << "|ok|" << positions(output) << '|'
                  << hex_bits(rng.random()) << '\n';
    } catch (const attribute::BaseAttributeException& error) {
        std::cout << "case=" << name << "|error|base|"
                  << hex_encode(error.what()) << '|'
                  << hex_bits(rng.random()) << '\n';
    } catch (const utils::DatasetException& error) {
        std::cout << "case=" << name << "|error|dataset|"
                  << hex_encode(error.what()) << '|'
                  << hex_bits(rng.random()) << '\n';
    }
}

void emit_positions() {
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        attribute::NodePositionSpec spec;
        spec.generative = true;
        spec.distribution = uniform_float(-2.0, 3.0);
        spec.dtype = utils::DatasetValueKind::floating;
        spec.minimum_radius = -0.25;
        spec.maximum_radius = 0.75;
        emit_position_case(
            "position_uniform_float_w" + std::to_string(workers),
            std::move(spec), 17U, 101U, workers);
    }

    attribute::NodePositionSpec integer;
    integer.generative = true;
    integer.distribution = uniform_integer(-3, 4);
    integer.dtype = utils::DatasetValueKind::integer;
    integer.minimum_radius = -1.5;
    integer.maximum_radius = 2.5;
    emit_position_case("position_uniform_int", std::move(integer), 13U, 102U, 8U);

    attribute::NodePositionSpec reversed;
    reversed.generative = true;
    reversed.distribution = uniform_float(-2.0, 3.0);
    reversed.dtype = utils::DatasetValueKind::floating;
    reversed.minimum_radius = 2.0;
    reversed.maximum_radius = -1.0;
    emit_position_case("position_reversed_clip", std::move(reversed), 11U, 103U, 2U);

    attribute::NodePositionSpec signed_zero;
    signed_zero.generative = true;
    signed_zero.distribution.kind = utils::DistributionKind::normal;
    signed_zero.distribution.loc = utils::DatasetScalar{-0.0};
    signed_zero.distribution.scale = utils::DatasetScalar{0.0};
    signed_zero.dtype = utils::DatasetValueKind::floating;
    signed_zero.minimum_radius = -0.0;
    signed_zero.maximum_radius = 0.0;
    emit_position_case("position_signed_zero", std::move(signed_zero), 7U, 104U, 8U);

    attribute::NodePositionSpec nan_minimum;
    nan_minimum.generative = true;
    nan_minimum.distribution.kind = utils::DistributionKind::normal;
    nan_minimum.distribution.loc = utils::DatasetScalar{0.0};
    nan_minimum.distribution.scale = utils::DatasetScalar{1.0};
    nan_minimum.dtype = utils::DatasetValueKind::floating;
    nan_minimum.minimum_radius = std::numeric_limits<double>::quiet_NaN();
    nan_minimum.maximum_radius = 1.0;
    emit_position_case("position_nan_minimum", std::move(nan_minimum), 5U, 106U, 2U);

    attribute::NodePositionSpec nan_maximum;
    nan_maximum.generative = true;
    nan_maximum.distribution.kind = utils::DistributionKind::normal;
    nan_maximum.distribution.loc = utils::DatasetScalar{0.0};
    nan_maximum.distribution.scale = utils::DatasetScalar{1.0};
    nan_maximum.dtype = utils::DatasetValueKind::floating;
    nan_maximum.minimum_radius = -1.0;
    nan_maximum.maximum_radius = std::numeric_limits<double>::quiet_NaN();
    emit_position_case("position_nan_maximum", std::move(nan_maximum), 5U, 107U, 8U);

    attribute::NodePositionSpec boolean;
    boolean.generative = true;
    boolean.distribution.kind = utils::DistributionKind::normal;
    boolean.distribution.loc = utils::DatasetScalar{-1.0};
    boolean.distribution.scale = utils::DatasetScalar{2.0};
    boolean.dtype = utils::DatasetValueKind::boolean;
    boolean.minimum_radius = 0.0;
    boolean.maximum_radius = 1.0;
    emit_position_case("position_normal_bool", std::move(boolean), 9U, 108U, 8U);

    attribute::NodePositionSpec not_generative;
    emit_position_case("position_not_generative", std::move(not_generative), 5U, 105U, 1U);
}

void emit_existing_position() {
    attribute::NodePositionSpec spec;
    spec.name = "other_name";
    const attribute::NodePositionAttribute position(spec);

    Graph empty;
    const auto empty_binding = position.bind_existing_pos(empty);
    emit_node_case("position_existing_empty", [&] {
        return values(position.get_existing_pos_data(empty, empty_binding));
    });

    Graph graph(4U, std::vector<EdgeEndpoints>{});
    const auto binding = position.bind_existing_pos(graph);
    emit_node_case("position_existing_missing_first", [&] {
        return values(position.get_existing_pos_data(graph, binding));
    });
    graph.node_attrs(0U).set(binding.value_id, std::string{"p0"});
    graph.node_attrs(2U).set(binding.value_id, std::string{"p2"});
    std::cout << "case=position_existing_partial|ok|"
              << values(position.get_existing_pos_data(graph, binding, 8U)) << '\n';
}

}  // namespace

int main() {
    try {
        emit_construction();
        emit_graph_adapter<Graph>("graph");
        emit_graph_adapter<DiGraph>("digraph");
        emit_resource();
        emit_positions();
        emit_existing_position();
        std::cout << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
