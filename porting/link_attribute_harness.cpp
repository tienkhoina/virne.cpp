#include "attribute/link_attribute.h"

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

std::string scalar(const AttrValue& value) {
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
    return "recursive";
}

std::string scalar(const attribute::AttributeNumber& value) {
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
        result.append(scalar(items[index]));
    }
    return result;
}

std::string doubles(const std::vector<double>& items) {
    std::string result;
    for (std::size_t index = 0U; index < items.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result.append(hex_bits(items[index]));
    }
    return result;
}

attribute::BaseAttributeSpec link_spec(
    std::string name,
    attribute::AttributeKind kind = attribute::AttributeKind::status) {
    attribute::BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = attribute::AttributeOwner::link;
    result.kind = kind;
    return result;
}

std::string link_error_name(attribute::LinkAttributeErrorCode code) noexcept {
    using Code = attribute::LinkAttributeErrorCode;
    switch (code) {
        case Code::invalid_link_spec: return "invalid_link_spec";
        case Code::edge_not_found: return "edge_not_found";
        case Code::missing_attribute: return "missing_attribute";
        case Code::dense_data_too_short: return "dense_data_too_short";
        case Code::unsupported_aggregation: return "unsupported_aggregation";
        case Code::update_path_not_implemented: return "update_path_not_implemented";
        case Code::missing_originator: return "missing_originator";
        case Code::missing_resource_value: return "missing_resource_value";
        case Code::non_numeric_resource: return "non_numeric_resource";
        case Code::path_too_short: return "path_too_short";
        case Code::empty_position_network: return "empty_position_network";
        case Code::missing_position_data: return "missing_position_data";
        case Code::invalid_position_data: return "invalid_position_data";
        case Code::non_generative_latency: return "non_generative_latency";
        case Code::invalid_latency_generation: return "invalid_latency_generation";
        case Code::empty_aggregation: return "empty_aggregation";
    }
    return "invalid_link_error";
}

template <typename Callable>
void emit_case(std::string_view name, Callable&& callable) {
    try {
        const std::string payload = callable();
        std::cout << "case=" << name << "|ok|" << payload << '\n';
    } catch (const attribute::LinkAttributeException& error) {
        std::cout << "case=" << name << "|error|"
                  << link_error_name(error.code()) << '|'
                  << hex_encode(error.what()) << '\n';
    } catch (const attribute::AttributeMethodException& error) {
        const std::string code = error.code()
                == attribute::AttributeMethodErrorCode::insufficient_resource
            ? "insufficient_resource"
            : "attribute_method_error";
        std::cout << "case=" << name << "|error|" << code << '|'
                  << hex_encode(error.what()) << '\n';
    } catch (const std::exception& error) {
        std::cout << "case=" << name << "|error|dependency_error|"
                  << hex_encode(error.what()) << '\n';
    }
}

void emit_construction() {
    attribute::LinkStatusSpec status_spec;
    status_spec.name = "alive";
    const attribute::LinkStatusAttribute status(status_spec);
    std::cout << "case=status_fields|ok|alive,link,status,"
              << (status.spec().generative ? '1' : '0') << ','
              << (status.spec().is_constraint.value_or(true) ? '1' : '0') << '\n';

    emit_case("extrema_missing_originator", [] {
        attribute::LinkExtremaSpec spec;
        spec.name = "bandwidth_max";
        static_cast<void>(attribute::LinkExtremaAttribute(std::move(spec)));
        return std::string{};
    });
    attribute::LinkExtremaSpec extrema_spec;
    extrema_spec.name = "bandwidth_max";
    extrema_spec.originator_name = "bandwidth";
    extrema_spec.originator_id = 17U;
    const attribute::LinkExtremaAttribute extrema(extrema_spec);
    std::cout << "case=extrema_fields|ok|bandwidth_max,link,extrema,"
              << extrema.originator_name() << ',' << extrema.originator_id()
              << ",0\n";

    attribute::LinkResourceSpec resource_spec;
    resource_spec.name = "bandwidth";
    resource_spec.restriction = attribute::ConstraintRestriction::soft;
    const attribute::LinkResourceAttribute resource(resource_spec);
    std::cout << "case=resource_fields|ok|bandwidth,link,resource,soft,link,1\n";

    attribute::LinkLatencySpec latency_spec;
    latency_spec.generative = true;
    latency_spec.generation = attribute::LatencyGenerationKind::position;
    latency_spec.minimum = -0.25;
    latency_spec.maximum = 0.75;
    const attribute::LinkLatencyAttribute latency(latency_spec);
    std::cout << "case=latency_fields|ok|latency,link,latency,1,position,"
              << scalar(latency.minimum()) << ',' << scalar(latency.maximum())
              << ",hard,path,1\n";
}

template <typename GraphType>
void emit_adapter(std::string_view prefix) {
    GraphType graph(
        5U,
        std::vector<EdgeEndpoints>{{3U, 4U}, {0U, 2U}, {1U, 4U},
                                   {0U, 1U}, {2U, 2U}});
    const attribute::LinkAttribute item(link_spec("mixed"));
    const auto binding = item.bind(graph);
    item.set_data_dense(
        graph,
        {AttrValue{std::int64_t{1}}, AttrValue{2.5}, AttrValue{true},
         AttrValue{std::string{"four"}}, AttrValue{std::int64_t{-5}},
         AttrValue{std::int64_t{999}}},
        binding,
        8U);
    std::cout << "case=" << prefix << "_dense|ok|"
              << values(item.get_data(graph, binding, 2U)) << '\n';

    GraphType sparse(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const auto sparse_binding = item.bind(sparse);
    item.set_data(
        sparse,
        {{1U, 2U, AttrValue{std::int64_t{31}}},
         {9U, 9U, AttrValue{std::int64_t{90}}},
         {1U, 2U, AttrValue{std::int64_t{32}}}},
        sparse_binding);
    std::cout << "case=" << prefix << "_sparse|ok|"
              << values(item.get_data(sparse, sparse_binding, 8U)) << '\n';
    emit_case(std::string(prefix) + "_missing_get", [&] {
        return scalar(item.get(sparse, 0U, 1U, sparse_binding));
    });
    emit_case(std::string(prefix) + "_short_dense", [&] {
        item.set_data_dense(
            sparse,
            {AttrValue{std::int64_t{7}}},
            sparse_binding,
            8U);
        return std::string{};
    });
}

void emit_matrix() {
    Graph graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const attribute::LinkAttribute item(link_spec("capacity"));
    const auto binding = item.bind(graph);
    item.set_data_dense(graph, {AttrValue{2.0}, AttrValue{4.0}}, binding, 2U);
    std::cout << "case=matrix_raw|ok|"
              << doubles(item.get_adjacency_data(graph, binding, false, 8U).data)
              << '\n';
    std::cout << "case=matrix_normalized|ok|"
              << doubles(item.get_adjacency_data(graph, binding, true, 2U).data)
              << '\n';
    for (const auto& [name, aggregation] : {
             std::pair<std::string_view, attribute::LinkAggregation>{
                 "sum", attribute::LinkAggregation::sum},
             {"mean", attribute::LinkAggregation::mean},
             {"max", attribute::LinkAggregation::maximum},
             {"min", attribute::LinkAggregation::minimum}}) {
        std::cout << "case=aggregation_" << name << "|ok|"
                  << doubles(item.get_aggregation_data(
                         graph, binding, aggregation, false, 8U))
                  << '\n';
    }

    Graph signed_zero(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    const auto signed_binding = item.bind(signed_zero);
    item.set_data_dense(signed_zero, {AttrValue{-0.0}}, signed_binding, 1U);
    std::cout << "case=aggregation_signed_zero_max|ok|"
              << doubles(item.get_aggregation_data(
                     signed_zero, signed_binding,
                     attribute::LinkAggregation::maximum))
              << '\n';
    std::cout << "case=aggregation_signed_zero_min|ok|"
              << doubles(item.get_aggregation_data(
                     signed_zero, signed_binding,
                     attribute::LinkAggregation::minimum))
              << '\n';

    Graph nan_graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    const auto nan_binding = item.bind(nan_graph);
    item.set_data_dense(
        nan_graph,
        {AttrValue{std::numeric_limits<double>::quiet_NaN()}},
        nan_binding,
        1U);
    std::cout << "case=aggregation_nan_max|ok|"
              << doubles(item.get_aggregation_data(
                     nan_graph, nan_binding,
                     attribute::LinkAggregation::maximum))
              << '\n';
    std::cout << "case=aggregation_nan_min|ok|"
              << doubles(item.get_aggregation_data(
                     nan_graph, nan_binding,
                     attribute::LinkAggregation::minimum))
              << '\n';
}

attribute::LinkResourceAttribute resource(
    attribute::ConstraintRestriction restriction) {
    attribute::LinkResourceSpec spec;
    spec.name = "bandwidth";
    spec.restriction = restriction;
    return attribute::LinkResourceAttribute(std::move(spec));
}

void emit_resource() {
    const auto hard = resource(attribute::ConstraintRestriction::hard);
    const auto soft = resource(attribute::ConstraintRestriction::soft);
    Graph virtual_graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    Graph physical_graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    static_cast<void>(physical_graph.attr_id("unrelated"));
    const auto virtual_binding = hard.bind(virtual_graph);
    const auto physical_binding = hard.bind(physical_graph);
    AttrMap& virtual_link = virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U));
    virtual_link.set(virtual_binding.value_id, std::int64_t{7});
    AttrMap& physical_first = physical_graph.edge_attrs(physical_graph.edge(0U, 1U));
    physical_first.set(physical_binding.value_id, 10.5);
    const auto hard_result = hard.check_constraint_satisfiability(
        virtual_link, virtual_binding.value_id,
        physical_first, physical_binding.value_id,
        attribute::ComparisonOperation::less_equal);
    std::cout << "case=resource_hard_pass|ok|"
              << (hard_result.flag ? '1' : '0') << ','
              << scalar(hard_result.offset) << '\n';
    virtual_link.set(virtual_binding.value_id, 12.0);
    const auto soft_result = soft.check_constraint_satisfiability(
        virtual_link, virtual_binding.value_id,
        physical_first, physical_binding.value_id,
        attribute::ComparisonOperation::less_equal);
    std::cout << "case=resource_soft_fail|ok|"
              << (soft_result.flag ? '1' : '0') << ','
              << scalar(soft_result.offset) << '\n';

    virtual_link.set(virtual_binding.value_id, std::int64_t{3});
    physical_first.set(physical_binding.value_id, std::int64_t{10});
    physical_graph.edge_attrs(physical_graph.edge(1U, 2U)).set(
        physical_binding.value_id, std::int64_t{8});
    hard.update_path(
        virtual_link, virtual_binding.value_id,
        physical_graph, {0U, 1U, 2U}, physical_binding,
        attribute::ResourceUpdateOperation::subtract, true);
    std::cout << "case=resource_path_update|ok|"
              << scalar(physical_first.at(physical_binding.value_id)) << ','
              << scalar(physical_graph.edge_attrs(physical_graph.edge(1U, 2U))
                            .at(physical_binding.value_id))
              << '\n';
    emit_case("resource_short_path", [&] {
        hard.update_path(
            virtual_link, virtual_binding.value_id,
            physical_graph, {0U}, physical_binding,
            attribute::ResourceUpdateOperation::add, true);
        return std::string{};
    });
    physical_first.set(physical_binding.value_id, std::int64_t{10});
    physical_graph.edge_attrs(physical_graph.edge(1U, 2U)).set(
        physical_binding.value_id, std::int64_t{2});
    emit_case("resource_partial_path", [&] {
        hard.update_path(
            virtual_link, virtual_binding.value_id,
            physical_graph, {0U, 1U, 2U}, physical_binding,
            attribute::ResourceUpdateOperation::subtract, true);
        return std::string{};
    });
    std::cout << "case=resource_partial_state|ok|"
              << scalar(physical_first.at(physical_binding.value_id)) << ','
              << scalar(physical_graph.edge_attrs(physical_graph.edge(1U, 2U))
                            .at(physical_binding.value_id))
              << '\n';
}

void set_position(
    Graph& graph,
    Vertex node,
    AttrId id,
    std::initializer_list<AttrValue> coordinates) {
    graph.node_attrs(node).set(
        id,
        make_attr_list(std::vector<AttrValue>(coordinates)));
}

void emit_latency() {
    attribute::LinkLatencySpec spec;
    spec.generative = true;
    spec.generation = attribute::LatencyGenerationKind::position;
    spec.minimum = std::int64_t{10};
    spec.maximum = std::int64_t{12};
    const attribute::LinkLatencyAttribute latency(spec);
    Graph graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const AttrId first_position = graph.attr_id("node_pos_first");
    const AttrId second_position = graph.attr_id("pos_second");
    set_position(graph, 0U, first_position, {std::int64_t{0}, std::int64_t{0}});
    set_position(graph, 1U, first_position, {std::int64_t{3}, std::int64_t{4}});
    set_position(graph, 2U, first_position, {std::int64_t{3}, std::int64_t{0}});
    for (Vertex node = 0U; node < 3U; ++node) {
        set_position(graph, node, second_position, {0.0, 0.0});
    }
    const auto position_binding = latency.resolve_position_binding(graph);
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        std::cout << "case=latency_position_w" << workers << "|ok|"
                  << position_binding.value_id << '|'
                  << doubles(latency.generate_from_position(
                         graph, position_binding, workers))
                  << '\n';
    }

    Graph virtual_graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    Graph physical_graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const auto virtual_binding = latency.bind(virtual_graph);
    const auto physical_binding = latency.bind(physical_graph);
    AttrMap& virtual_link = virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U));
    virtual_link.set(virtual_binding.value_id, 10.0);
    AttrMap& first = physical_graph.edge_attrs(physical_graph.edge(0U, 1U));
    AttrMap& second = physical_graph.edge_attrs(physical_graph.edge(1U, 2U));
    first.set(physical_binding.value_id, 3.0);
    second.set(physical_binding.value_id, 4.0);
    const auto result = latency.check_constraint_satisfiability(
        virtual_link, virtual_binding.value_id, {&first, &second},
        physical_binding.value_id,
        attribute::ComparisonOperation::greater_equal);
    std::cout << "case=latency_path_check|ok|"
              << (result.flag ? '1' : '0') << ',' << scalar(result.offset)
              << '\n';

    attribute::LinkLatencySpec non_generative_spec;
    non_generative_spec.generation = attribute::LatencyGenerationKind::position;
    const attribute::LinkLatencyAttribute non_generative(non_generative_spec);
    emit_case("latency_non_generative", [&] {
        return doubles(non_generative.generate_from_position(
            graph, position_binding, 1U));
    });
}

}  // namespace

int main() {
    emit_construction();
    emit_adapter<Graph>("graph");
    emit_adapter<DiGraph>("digraph");
    emit_matrix();
    emit_resource();
    emit_latency();
    std::cout << "status=PASS\n";
    return 0;
}
