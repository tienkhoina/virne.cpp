#include "base_network.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace network = virne::network;
namespace attribute = virne::network::attribute;

std::string hex_text(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string double_token(const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

std::string float_token(const float value) {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8) << bits;
    return stream.str();
}

std::string value_token(const AttrValue& value) {
    return std::visit(
        [](const auto& item) -> std::string {
            using Item = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Item, std::int64_t>) {
                return "i:" + std::to_string(item);
            } else if constexpr (std::is_same_v<Item, double>) {
                return double_token(item);
            } else if constexpr (std::is_same_v<Item, bool>) {
                return item ? "b:1" : "b:0";
            } else if constexpr (std::is_same_v<Item, std::string>) {
                return "s:" + hex_text(item);
            } else {
                throw std::runtime_error("recursive value entered scalar harness");
            }
        },
        value);
}

std::string value_token(const double value) {
    return double_token(value);
}

template <typename Row>
std::string canonical_rows(const std::vector<Row>& rows) {
    std::string result = "[";
    for (std::size_t row_index = 0U; row_index < rows.size(); ++row_index) {
        if (row_index != 0U) {
            result.push_back(',');
        }
        result.push_back('[');
        for (std::size_t column = 0U; column < rows[row_index].size(); ++column) {
            if (column != 0U) {
                result.push_back(',');
            }
            result += value_token(rows[row_index][column]);
        }
        result.push_back(']');
    }
    result.push_back(']');
    return result;
}

std::string canonical_names(const std::vector<std::string>& values) {
    std::string result = "[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += hex_text(values[index]);
    }
    result.push_back(']');
    return result;
}

std::string_view error_name(const network::BaseNetworkErrorCode value) {
    using Code = network::BaseNetworkErrorCode;
    switch (value) {
        case Code::invalid_config:
            return "invalid_config";
        case Code::missing_config_field:
            return "missing_config_field";
        case Code::attribute_registry_mismatch:
            return "attribute_registry_mismatch";
        case Code::graph_binding_mismatch:
            return "graph_binding_mismatch";
        case Code::no_nodes:
            return "no_nodes";
        case Code::no_links:
            return "no_links";
        case Code::missing_node_attribute:
            return "missing_node_attribute";
        case Code::missing_link_attribute:
            return "missing_link_attribute";
        case Code::generated_length_mismatch:
            return "generated_length_mismatch";
        case Code::empty_attribute_selection:
            return "empty_attribute_selection";
        case Code::non_numeric_benchmark_value:
            return "non_numeric_benchmark_value";
        case Code::ragged_benchmark_matrix:
            return "ragged_benchmark_matrix";
        case Code::invalid_gml_flattened_key:
            return "invalid_gml_flattened_key";
    }
    return "invalid";
}

std::string_view operation_name(const network::BaseNetworkOperation value) {
    using Operation = network::BaseNetworkOperation;
    switch (value) {
        case Operation::decode_config:
            return "decode_config";
        case Operation::construct:
            return "construct";
        case Operation::bind_attribute:
            return "bind_attribute";
        case Operation::check_attributes:
            return "check_attributes";
        case Operation::generate_topology:
            return "generate_topology";
        case Operation::generate_node_attributes:
            return "generate_node_attributes";
        case Operation::generate_link_attributes:
            return "generate_link_attributes";
        case Operation::get_attribute_data:
            return "get_attribute_data";
        case Operation::set_attribute_data:
            return "set_attribute_data";
        case Operation::prepare_benchmarks:
            return "prepare_benchmarks";
        case Operation::prepare_gml:
            return "prepare_gml";
        case Operation::restore_gml:
            return "restore_gml";
        case Operation::save_attributes:
            return "save_attributes";
    }
    return "invalid";
}

template <typename Callable>
void emit_case(const std::string_view name, Callable&& callable) {
    try {
        const std::string value = std::forward<Callable>(callable)();
        std::cout << "case=" << name << "|ok|" << hex_text(value) << '\n';
    } catch (const network::BaseNetworkException& error) {
        std::cout << "case=" << name << "|error|" << error_name(error.code())
                  << '|' << operation_name(error.operation()) << '|';
        if (error.input_index() == network::invalid_base_network_input_index) {
            std::cout << '-';
        } else {
            std::cout << error.input_index();
        }
        std::cout << '\n';
    }
}

attribute::AttributeFactorySpec make_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const attribute::AttributeKind kind) {
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = kind;
    return result;
}

attribute::AttributeRegistryId require_node_id(
    const network::BaseNetwork& value,
    const std::string_view name) {
    const auto id = value.node_attributes().bind(name);
    if (!id) {
        throw std::runtime_error("missing node definition");
    }
    return *id;
}

attribute::AttributeRegistryId require_link_id(
    const network::BaseNetwork& value,
    const std::string_view name) {
    const auto id = value.link_attributes().bind(name);
    if (!id) {
        throw std::runtime_error("missing link definition");
    }
    return *id;
}

network::BaseNetwork make_fixture(const std::size_t count = 4U) {
    Graph graph;
    for (std::size_t index = 0U; index < count; ++index) {
        graph.add_node();
    }
    for (Vertex index = 0U; index + 1U < count; ++index) {
        graph.add_edge(index, index + 1U);
    }
    for (Vertex index = 0U; index + 2U < count; ++index) {
        graph.add_edge(index, index + 2U);
    }

    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(std::move(graph));
    construction.config.node_attribute_specs.push_back(make_spec(
        "cpu", attribute::AttributeOwner::node,
        attribute::AttributeKind::resource));
    auto peak = make_spec(
        "peak", attribute::AttributeOwner::node,
        attribute::AttributeKind::extrema);
    peak.originator_name = "cpu";
    construction.config.node_attribute_specs.push_back(std::move(peak));
    construction.config.node_attribute_specs.push_back(make_spec(
        "state", attribute::AttributeOwner::node,
        attribute::AttributeKind::status));

    construction.config.link_attribute_specs.push_back(make_spec(
        "bw", attribute::AttributeOwner::link,
        attribute::AttributeKind::resource));
    auto peak_bw = make_spec(
        "peak_bw", attribute::AttributeOwner::link,
        attribute::AttributeKind::extrema);
    peak_bw.originator_name = "bw";
    construction.config.link_attribute_specs.push_back(std::move(peak_bw));
    construction.config.link_attribute_specs.push_back(make_spec(
        "delay", attribute::AttributeOwner::link,
        attribute::AttributeKind::latency));

    construction.config.topology = AttrValue(std::string("fixture"));
    construction.config.output = AttrValue(std::string("memory"));
    construction.config.graph_attributes.push_back(
        {"zeta", AttrValue(std::int64_t{7})});
    construction.config.graph_attributes.push_back(
        {"label", AttrValue(std::string("initial"))});
    construction.extra_graph_attributes.push_back(
        {"label", AttrValue(std::string("override"))});
    construction.extra_graph_attributes.push_back(
        {"flag", AttrValue(true)});

    network::BaseNetwork result(std::move(construction));
    const std::size_t edges = result.graph().num_edges();
    std::vector<AttrValue> cpu(count);
    std::vector<AttrValue> node_peak(count);
    std::vector<AttrValue> state(count);
    for (std::size_t index = 0U; index < count; ++index) {
        cpu[index] = std::int64_t{10} * static_cast<std::int64_t>(index + 1U);
        node_peak[index] = 10.0 * static_cast<double>(index + 1U) + 1.25;
        state[index] = (index % 2U) == 0U;
    }
    std::vector<AttrValue> bw(edges);
    std::vector<AttrValue> link_peak(edges);
    std::vector<AttrValue> delay(edges);
    for (std::size_t index = 0U; index < edges; ++index) {
        bw[index] = static_cast<std::int64_t>(2U * index + 5U);
        link_peak[index] = 2.0 * static_cast<double>(index) + 6.5;
        delay[index] = 0.25 * static_cast<double>(index) + 0.5;
    }
    result.set_node_attrs_data({
        {require_node_id(result, "cpu"), network::AttributeDataLayout::dense,
         {}, std::move(cpu)},
        {require_node_id(result, "peak"), network::AttributeDataLayout::dense,
         {}, std::move(node_peak)},
        {require_node_id(result, "state"), network::AttributeDataLayout::dense,
         {}, std::move(state)},
    });
    result.set_link_attrs_data({
        {require_link_id(result, "bw"), network::AttributeDataLayout::dense,
         {}, std::move(bw)},
        {require_link_id(result, "peak_bw"), network::AttributeDataLayout::dense,
         {}, std::move(link_peak)},
        {require_link_id(result, "delay"), network::AttributeDataLayout::dense,
         {}, std::move(delay)},
    });
    return result;
}

std::vector<std::string> node_names(const network::BaseNetwork& value) {
    std::vector<std::string> result;
    result.reserve(value.node_attributes().size());
    for (const auto& entry : value.node_attributes().entries()) {
        result.push_back(entry.name);
    }
    return result;
}

std::vector<std::string> link_names(const network::BaseNetwork& value) {
    std::vector<std::string> result;
    result.reserve(value.link_attributes().size());
    for (const auto& entry : value.link_attributes().entries()) {
        result.push_back(entry.name);
    }
    return result;
}

std::vector<std::string> kind_names(
    const std::vector<attribute::AttributeKind>& values) {
    std::vector<std::string> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.emplace_back(attribute::attribute_kind_name(value));
    }
    return result;
}

std::string fixture_inventory(network::BaseNetwork& value) {
    static constexpr std::string_view metadata_names[] = {
        "topology", "output", "zeta", "label", "flag"};
    std::string metadata;
    for (const std::string_view name : metadata_names) {
        const AttrValue* item = value.graph_attributes().find(name);
        if (item == nullptr) {
            continue;
        }
        if (!metadata.empty()) {
            metadata.push_back(',');
        }
        metadata += hex_text(name);
        metadata.push_back('=');
        metadata += value_token(*item);
    }
    std::ostringstream stream;
    stream << "nodes=" << value.graph().num_nodes()
           << ";links=" << value.graph().num_edges()
           << ";node_names=" << canonical_names(node_names(value))
           << ";node_types=" << canonical_names(kind_names(value.get_node_attr_types()))
           << ";link_names=" << canonical_names(link_names(value))
           << ";link_types=" << canonical_names(kind_names(value.get_link_attr_types()))
           << ";features=" << value.num_node_features() << ','
           << value.num_link_features() << ','
           << value.num_node_resource_features() << ','
           << value.num_link_resource_features()
           << ";metadata={" << metadata << '}';
    return stream.str();
}

std::vector<std::string> selected_node_names(
    const network::BaseNetwork& value,
    const std::vector<attribute::AttributeRegistryId>& ids) {
    std::vector<std::string> result;
    result.reserve(ids.size());
    for (const auto id : ids) {
        result.push_back(value.node_attributes().entries().at(id).name);
    }
    return result;
}

std::string canonical_benchmark_map(
    const std::optional<attribute::AttributeBenchmarkMap>& value) {
    if (!value) {
        return "none";
    }
    std::string result = "{";
    for (std::size_t index = 0U; index < value->entries().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& entry = value->entries()[index];
        result += hex_text(entry.name);
        result.push_back('=');
        result += double_token(entry.value);
    }
    result.push_back('}');
    return result;
}

std::string canonical_benchmarks(
    const attribute::AttributeBenchmarks& value) {
    return "node=" + canonical_benchmark_map(value.node_attr_benchmarks)
        + ";link=" + canonical_benchmark_map(value.link_attr_benchmarks)
        + ";link_sum=" + canonical_benchmark_map(value.link_sum_attr_benchmarks);
}

std::string canonical_prepared_group(
    const std::optional<attribute::PreparedAttributeBenchmarkData>& group) {
    if (!group) {
        return "none";
    }
    std::string attributes;
    for (std::size_t index = 0U; index < group->attributes.size(); ++index) {
        if (index != 0U) {
            attributes.push_back(',');
        }
        const auto& item = group->attributes[index];
        attributes += std::to_string(item.definition_id);
        attributes.push_back(':');
        attributes += attribute::attribute_kind_name(item.kind);
        attributes.push_back(':');
        attributes += hex_text(item.name);
        attributes.push_back(':');
        attributes += item.originator_name ? hex_text(*item.originator_name) : "none";
    }
    std::string bits;
    bits.reserve(group->matrix.values.size() * 8U);
    for (const float value : group->matrix.values) {
        bits += float_token(value);
    }
    std::ostringstream stream;
    stream << "attrs=[" << attributes << "];shape=" << group->matrix.rows
           << ',' << group->matrix.columns << ";bits=" << bits
           << ";extrema=" << (group->extrema_requested ? '1' : '0')
           << ";rep=" << group->column_repetitions;
    return stream.str();
}

std::string canonical_prepared_request(
    const attribute::AttributeBenchmarkRequest& value) {
    return "workers=" + std::to_string(value.workers)
        + ";node=" + canonical_prepared_group(value.node)
        + ";link=" + canonical_prepared_group(value.link)
        + ";link_sum=" + canonical_prepared_group(value.link_sum);
}

void construction_cases() {
    emit_case("empty", [] {
        network::BaseNetwork value;
        return fixture_inventory(value);
    });
    emit_case("fixture_inventory", [] {
        auto value = make_fixture();
        return fixture_inventory(value);
    });
    emit_case("duplicate_first_order", [] {
        Graph graph;
        graph.add_node();
        graph.add_node();
        graph.add_edge(0U, 1U);
        network::BaseNetworkConstruction construction;
        construction.incoming_graph.emplace(std::move(graph));
        construction.config.node_attribute_specs = {
            make_spec("a", attribute::AttributeOwner::node,
                      attribute::AttributeKind::status),
            make_spec("b", attribute::AttributeOwner::node,
                      attribute::AttributeKind::resource),
            make_spec("a", attribute::AttributeOwner::node,
                      attribute::AttributeKind::position),
        };
        construction.config.link_attribute_specs = {
            make_spec("x", attribute::AttributeOwner::link,
                      attribute::AttributeKind::status),
            make_spec("y", attribute::AttributeOwner::link,
                      attribute::AttributeKind::resource),
            make_spec("x", attribute::AttributeOwner::link,
                      attribute::AttributeKind::latency),
        };
        network::BaseNetwork value(std::move(construction));
        return fixture_inventory(value);
    });
}

void cardinality_and_selection_cases() {
    emit_case("cardinality_cache", [] {
        auto value = make_fixture();
        const auto before_nodes = value.num_nodes();
        const auto before_links = value.num_links();
        value.graph().add_node();
        value.graph().add_edge(0U, 4U);
        std::ostringstream stream;
        stream << "before=" << before_nodes << ',' << before_links
               << ";stale=" << value.num_nodes() << ',' << value.num_links()
               << ";edge=" << value.num_edges();
        return stream.str();
    });
    emit_case("selection_precedence", [] {
        const auto value = make_fixture();
        const auto state_id = require_node_id(value, "state");
        network::AttributeSelection resource;
        resource.kinds = std::vector<attribute::AttributeKind>{
            attribute::AttributeKind::resource};
        resource.ids = std::vector<attribute::AttributeRegistryId>{state_id};
        network::AttributeSelection empty;
        empty.kinds = std::vector<attribute::AttributeKind>{};
        empty.ids = std::vector<attribute::AttributeRegistryId>{state_id};
        const auto resource_ids = value.select_node_attributes(resource);
        const auto empty_ids = value.select_node_attributes(empty);
        const auto all_ids = value.select_node_attributes({});
        return "resource=" + canonical_names(selected_node_names(value, resource_ids))
            + ";empty=" + canonical_names(selected_node_names(value, empty_ids))
            + ";all=" + canonical_names(selected_node_names(value, all_ids));
    });
}

void row_cases() {
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        emit_case("rows_workers_" + std::to_string(workers), [workers] {
            const auto value = make_fixture();
            const std::vector<attribute::AttributeRegistryId> node_ids = {
                require_node_id(value, "cpu"), require_node_id(value, "peak")};
            const std::vector<attribute::AttributeRegistryId> link_ids = {
                require_link_id(value, "bw"), require_link_id(value, "peak_bw")};
            return "node=" + canonical_rows(
                       network::get_node_attrs_data(value, node_ids, workers))
                + ";link=" + canonical_rows(
                       network::get_link_attrs_data(value, link_ids, workers));
        });
    }
    emit_case("ordered_setters", [] {
        auto value = make_fixture();
        const auto cpu = require_node_id(value, "cpu");
        const auto peak = require_node_id(value, "peak");
        const auto bw = require_link_id(value, "bw");
        const auto peak_bw = require_link_id(value, "peak_bw");
        value.set_node_attrs_data({
            {cpu, network::AttributeDataLayout::dense, {},
             {std::int64_t{1}, std::int64_t{2}, std::int64_t{3}, std::int64_t{4}}},
            {peak, network::AttributeDataLayout::dense, {},
             {1.5, 2.5, 3.5, 4.5}},
        });
        const std::size_t edges = value.graph().num_edges();
        std::vector<AttrValue> bw_values(edges);
        std::vector<AttrValue> peak_values(edges);
        for (std::size_t index = 0U; index < edges; ++index) {
            bw_values[index] = static_cast<std::int64_t>(20U + index);
            peak_values[index] = 30.5 + static_cast<double>(index);
        }
        value.set_link_attrs_data({
            {bw, network::AttributeDataLayout::dense, {}, std::move(bw_values)},
            {peak_bw, network::AttributeDataLayout::dense, {},
             std::move(peak_values)},
        });
        return "node=" + canonical_rows(
                   network::get_node_attrs_data(value, {cpu, peak}))
            + ";link=" + canonical_rows(
                   network::get_link_attrs_data(value, {bw, peak_bw}));
    });
    emit_case("empty_node_rows", [] {
        const auto value = make_fixture();
        return canonical_rows(network::get_node_attrs_data(value, {}));
    });
    emit_case("empty_link_rows", [] {
        const auto value = make_fixture();
        return canonical_rows(network::get_link_attrs_data(value, {}));
    });
    emit_case("aggregation_sum", [] {
        const auto value = make_fixture();
        return canonical_rows(network::get_aggregation_attrs_data(
            value,
            {require_link_id(value, "bw"), require_link_id(value, "peak_bw")},
            attribute::LinkAggregation::sum));
    });
}

void existence_and_copy_cases() {
    emit_case("existence_sample_only", [] {
        auto value = make_fixture();
        const auto node_binding = value.bind_node_attribute("cpu");
        const auto link_binding = value.bind_link_attribute("bw");
        if (!node_binding || !link_binding) {
            throw std::runtime_error("missing fixture binding");
        }
        value.graph().node_attrs(value.graph().num_nodes() - 1U)
            .erase(node_binding->value_id);
        value.graph().edge_attrs(value.graph().edge(2U, 3U))
            .erase(link_binding->value_id);
        value.check_attrs_existence();
        return std::string("pass");
    });
    emit_case("existence_missing_first_node", [] {
        auto value = make_fixture();
        const auto binding = value.bind_node_attribute("cpu");
        if (!binding) {
            throw std::runtime_error("missing fixture binding");
        }
        value.graph().node_attrs(0U).erase(binding->value_id);
        value.check_attrs_existence();
        return std::string("unexpected");
    });
    emit_case("existence_no_nodes", [] {
        network::BaseNetwork value;
        value.check_attrs_existence();
        return std::string("unexpected");
    });
    emit_case("existence_no_links", [] {
        Graph graph;
        graph.add_node();
        network::BaseNetworkConstruction construction;
        construction.incoming_graph.emplace(std::move(graph));
        network::BaseNetwork value(std::move(construction));
        value.check_attrs_existence();
        return std::string("unexpected");
    });
    emit_case("induced_view", [] {
        auto value = make_fixture();
        const auto view = value.subgraph({0U, 1U, 2U});
        std::ostringstream stream;
        stream << "nodes=" << view.num_nodes() << ";links=" << view.num_links()
               << ";shared="
               << ((&view.node_attributes() == &value.node_attributes()
                    && &view.link_attributes() == &value.link_attributes()) ? 1 : 0);
        return stream.str();
    });
    emit_case("clone_depth", [] {
        auto source = make_fixture();
        auto clone = source.clone();
        const auto clone_cpu = require_node_id(clone, "cpu");
        clone.set_node_attrs_data({
            {clone_cpu, network::AttributeDataLayout::dense, {},
             {std::int64_t{999}, std::int64_t{20},
              std::int64_t{30}, std::int64_t{40}}},
        });
        const auto source_rows = network::get_node_attrs_data(
            source, {require_node_id(source, "cpu")});
        const auto clone_rows = network::get_node_attrs_data(clone, {clone_cpu});
        std::ostringstream stream;
        stream << "source=" << value_token(source_rows[0][0])
               << ";clone=" << value_token(clone_rows[0][0])
               << ";registries="
               << ((&clone.node_attributes() != &source.node_attributes()
                    && &clone.link_attributes() != &source.link_attributes()) ? 1 : 0);
        return stream.str();
    });
}

void manager_cases() {
    emit_case("prepared_default", [] {
        const auto value = make_fixture();
        network::BaseNetworkBenchmarkSelection selection;
        selection.workers = 1U;
        return canonical_prepared_request(
            network::prepare_attribute_benchmark_request(value, selection));
    });
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        emit_case("manager_default_workers_" + std::to_string(workers),
                  [workers] {
            const auto value = make_fixture();
            network::BaseNetworkBenchmarkSelection selection;
            selection.workers = workers;
            return canonical_benchmarks(
                network::get_attribute_benchmarks(value, selection));
        });
    }
    emit_case("manager_all_types", [] {
        const auto value = make_fixture();
        network::BaseNetworkBenchmarkSelection selection;
        selection.node_kinds.reset();
        selection.link_kinds.reset();
        return canonical_benchmarks(
            network::get_attribute_benchmarks(value, selection));
    });
    emit_case("manager_disabled", [] {
        const auto value = make_fixture();
        network::BaseNetworkBenchmarkSelection selection;
        selection.node = false;
        selection.link = false;
        selection.link_sum = false;
        return canonical_benchmarks(
            network::get_attribute_benchmarks(value, selection));
    });
    emit_case("manager_empty_link_sum", [] {
        const network::BaseNetwork value;
        network::BaseNetworkBenchmarkSelection selection;
        selection.node = false;
        selection.link = false;
        selection.link_sum = true;
        return canonical_benchmarks(
            network::get_attribute_benchmarks(value, selection));
    });
    emit_case("manager_empty_node", [] {
        const network::BaseNetwork value;
        network::BaseNetworkBenchmarkSelection selection;
        selection.node = true;
        selection.link = false;
        selection.link_sum = false;
        return canonical_benchmarks(
            network::get_attribute_benchmarks(value, selection));
    });
    emit_case("manager_nonnumeric", [] {
        auto value = make_fixture();
        const auto peak = require_node_id(value, "peak");
        value.set_node_attrs_data({
            {peak, network::AttributeDataLayout::dense, {},
             {std::string("bad"), 21.25, 31.25, 41.25}},
        });
        network::BaseNetworkBenchmarkSelection selection;
        selection.link = false;
        selection.link_sum = false;
        return canonical_benchmarks(
            network::get_attribute_benchmarks(value, selection));
    });
}

}  // namespace

int main() {
    try {
        construction_cases();
        cardinality_and_selection_cases();
        row_cases();
        existence_and_copy_cases();
        manager_cases();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base_network_harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
