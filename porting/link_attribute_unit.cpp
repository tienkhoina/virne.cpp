#include "attribute/link_attribute.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void expect_link_error(
    attribute::LinkAttributeErrorCode code,
    attribute::LinkAttributeOperation operation,
    Callable&& callable,
    const std::string& message) {
    try {
        callable();
    } catch (const attribute::LinkAttributeException& error) {
        expect(error.code() == code, message + ": wrong error code");
        expect(error.operation() == operation, message + ": wrong operation");
        return;
    }
    throw std::runtime_error(message + ": expected LinkAttributeException");
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

bool attr_equal(const AttrValue& left, const AttrValue& right) {
    return attr_value_equal(left, right);
}

void expect_values(
    const std::vector<AttrValue>& actual,
    const std::vector<AttrValue>& expected,
    const std::string& message) {
    expect(actual.size() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        expect(attr_equal(actual[index], expected[index]),
               message + ": value mismatch at " + std::to_string(index));
    }
}

void expect_doubles(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& message) {
    expect(actual.size() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        expect(actual[index] == expected[index],
               message + ": value mismatch at " + std::to_string(index));
    }
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t result = 0U;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template <typename GraphType>
void test_adapter_for_graph(const std::string& label) {
    GraphType graph(
        5U,
        std::vector<EdgeEndpoints>{{3U, 4U}, {0U, 2U}, {1U, 4U},
                                   {0U, 1U}, {2U, 2U}});
    const attribute::LinkAttribute item(link_spec("mixed"));
    const auto binding = item.bind(graph);
    const std::vector<AttrValue> dense = {
        std::int64_t{1}, 2.5, true, std::string{"four"}, std::int64_t{-5},
        std::int64_t{999}};
    const std::vector<AttrValue> expected(dense.begin(), dense.begin() + 5);
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        item.set_data_dense(graph, dense, binding, workers);
        expect_values(item.get_data(graph, binding, workers), expected,
                      label + " dense worker " + std::to_string(workers));
    }

    const auto before = item.get_data(graph, binding, 1U);
    expect_link_error(
        attribute::LinkAttributeErrorCode::dense_data_too_short,
        attribute::LinkAttributeOperation::set_data,
        [&] {
            item.set_data_dense(
                graph, std::vector<AttrValue>(4U, AttrValue{0.0}), binding, 8U);
        },
        label + " short dense");
    expect_values(item.get_data(graph, binding, 1U), before,
                  label + " short dense mutation");

    item.set_data(
        graph,
        {{3U, 4U, AttrValue{std::int64_t{30}}},
         {99U, 1U, AttrValue{std::int64_t{90}}},
         {3U, 4U, AttrValue{std::int64_t{31}}}},
        binding);
    expect(std::get<std::int64_t>(item.get(graph, 3U, 4U, binding)) == 31,
           label + " sparse duplicate");

    GraphType partial(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const auto partial_binding = item.bind(partial);
    partial.edge_attrs(partial.edge(1U, 2U)).set(
        partial_binding.value_id, std::int64_t{7});
    expect_values(item.get_data(partial, partial_binding, 2U),
                  {AttrValue{std::int64_t{7}}}, label + " partial get");
    expect_link_error(
        attribute::LinkAttributeErrorCode::missing_attribute,
        attribute::LinkAttributeOperation::get,
        [&] { static_cast<void>(item.get(partial, 0U, 1U, partial_binding)); },
        label + " missing get");
    expect_link_error(
        attribute::LinkAttributeErrorCode::edge_not_found,
        attribute::LinkAttributeOperation::get,
        [&] { static_cast<void>(item.get(partial, 0U, 2U, partial_binding)); },
        label + " absent edge");
}

void test_resolvers_and_base() {
    expect(attribute::link_aggregation_from_string("sum")
               == attribute::LinkAggregation::sum,
           "sum resolver");
    expect(attribute::link_aggregation_from_string("mean")
               == attribute::LinkAggregation::mean,
           "mean resolver");
    expect(attribute::link_aggregation_from_string("max")
               == attribute::LinkAggregation::maximum,
           "max resolver");
    expect(attribute::link_aggregation_from_string("min")
               == attribute::LinkAggregation::minimum,
           "min resolver");
    expect_link_error(
        attribute::LinkAttributeErrorCode::unsupported_aggregation,
        attribute::LinkAttributeOperation::resolve_aggregation,
        [] { static_cast<void>(attribute::link_aggregation_from_string("SUM")); },
        "unsupported aggregation");

    auto invalid = link_spec("bad");
    invalid.owner = attribute::AttributeOwner::node;
    expect_link_error(
        attribute::LinkAttributeErrorCode::invalid_link_spec,
        attribute::LinkAttributeOperation::construct,
        [&] { static_cast<void>(attribute::LinkAttribute(std::move(invalid))); },
        "invalid owner");

    const attribute::LinkAttribute base(link_spec("value"));
    expect_link_error(
        attribute::LinkAttributeErrorCode::update_path_not_implemented,
        attribute::LinkAttributeOperation::update_path,
        [&] { static_cast<void>(base.update_path()); },
        "abstract path update");
}

void test_matrix_and_aggregation() {
    Graph graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const attribute::LinkAttribute item(link_spec("weight"));
    const auto binding = item.bind(graph);
    item.set_data_dense(
        graph,
        {AttrValue{2.0}, AttrValue{4.0}},
        binding,
        2U);
    const DistanceMatrix raw = item.get_adjacency_data(graph, binding, false, 8U);
    const std::vector<double> expected_raw = {
        0.0, 2.0, 0.0,
        2.0, 0.0, 4.0,
        0.0, 4.0, 0.0};
    expect_doubles(raw.data, expected_raw, "raw undirected adjacency");
    const DistanceMatrix normalized =
        item.get_adjacency_data(graph, binding, true, 2U);
    expect_doubles(
        normalized.data,
        {0.0, 1.0, 0.0,
         1.0 / 3.0, 0.0, 2.0 / 3.0,
         0.0, 1.0, 0.0},
        "normalized undirected adjacency");
    expect_doubles(
        item.get_aggregation_data(
            graph, binding, attribute::LinkAggregation::sum, false, 8U),
        {2.0, 6.0, 4.0},
        "sum aggregation");
    expect_doubles(
        item.get_aggregation_data(
            graph, binding, attribute::LinkAggregation::mean, false, 2U),
        {2.0 / 3.0, 2.0, 4.0 / 3.0},
        "mean aggregation");
    expect_doubles(
        item.get_aggregation_data(
            graph, binding, attribute::LinkAggregation::maximum, false, 1U),
        {2.0, 4.0, 4.0},
        "max aggregation");
    expect_doubles(
        item.get_aggregation_data(
            graph, binding, attribute::LinkAggregation::minimum, false, 0U),
        {0.0, 0.0, 0.0},
        "min aggregation");

    DiGraph directed(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const auto directed_binding = item.bind(directed);
    item.set_data_dense(
        directed,
        {AttrValue{2.0}, AttrValue{4.0}},
        directed_binding,
        8U);
    expect_doubles(
        item.get_adjacency_data(directed, directed_binding, false, 2U).data,
        {0.0, 2.0, 0.0,
         0.0, 0.0, 4.0,
         0.0, 0.0, 0.0},
        "directed adjacency");

    Graph signed_zero(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    const auto signed_binding = item.bind(signed_zero);
    item.set_data_dense(signed_zero, {AttrValue{-0.0}}, signed_binding, 1U);
    const auto signed_max = item.get_aggregation_data(
        signed_zero, signed_binding, attribute::LinkAggregation::maximum);
    const auto signed_min = item.get_aggregation_data(
        signed_zero, signed_binding, attribute::LinkAggregation::minimum);
    expect(double_bits(signed_max[0]) == double_bits(0.0)
               && double_bits(signed_max[1]) == double_bits(0.0),
           "signed-zero maximum reduction");
    expect(double_bits(signed_min[0]) == double_bits(0.0)
               && double_bits(signed_min[1]) == double_bits(0.0),
           "signed-zero minimum reduction");

    Graph nan_graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    const auto nan_binding = item.bind(nan_graph);
    item.set_data_dense(
        nan_graph,
        {AttrValue{std::numeric_limits<double>::quiet_NaN()}},
        nan_binding,
        1U);
    for (const auto aggregation : {
             attribute::LinkAggregation::maximum,
             attribute::LinkAggregation::minimum}) {
        const auto values = item.get_aggregation_data(
            nan_graph, nan_binding, aggregation, false, 2U);
        expect(std::isnan(values[0]) && std::isnan(values[1]),
               "NaN reduction propagation");
    }
}

attribute::LinkResourceAttribute resource_attribute(
    attribute::ConstraintRestriction restriction =
        attribute::ConstraintRestriction::hard) {
    attribute::LinkResourceSpec spec;
    spec.name = "bandwidth";
    spec.restriction = restriction;
    return attribute::LinkResourceAttribute(std::move(spec));
}

void test_typed_classes_and_resource() {
    attribute::LinkStatusSpec status_spec;
    status_spec.name = "alive";
    const attribute::LinkStatusAttribute status(status_spec);
    expect(status.spec().owner == attribute::AttributeOwner::link,
           "status owner");
    expect(status.spec().kind == attribute::AttributeKind::status,
           "status kind");
    expect(status.spec().is_constraint == std::optional<bool>{false},
           "status constraint");

    attribute::LinkExtremaSpec missing_extrema;
    missing_extrema.name = "bandwidth_max";
    expect_link_error(
        attribute::LinkAttributeErrorCode::missing_originator,
        attribute::LinkAttributeOperation::construct,
        [&] {
            static_cast<void>(
                attribute::LinkExtremaAttribute(std::move(missing_extrema)));
        },
        "missing extrema originator");
    attribute::LinkExtremaSpec extrema_spec;
    extrema_spec.name = "bandwidth_max";
    extrema_spec.originator_name = "bandwidth";
    extrema_spec.originator_id = 12U;
    const attribute::LinkExtremaAttribute extrema(extrema_spec);
    expect(extrema.originator_name() == "bandwidth", "extrema name");
    expect(extrema.originator_id() == 12U, "extrema id");

    const auto hard = resource_attribute();
    const auto soft = resource_attribute(attribute::ConstraintRestriction::soft);
    Graph virtual_graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    Graph physical_graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    static_cast<void>(physical_graph.attr_id("unrelated"));
    const auto virtual_binding = hard.bind(virtual_graph);
    const auto physical_binding = hard.bind(physical_graph);
    virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)).set(
        virtual_binding.value_id, std::int64_t{3});
    physical_graph.edge_attrs(physical_graph.edge(0U, 1U)).set(
        physical_binding.value_id, std::int64_t{10});
    physical_graph.edge_attrs(physical_graph.edge(1U, 2U)).set(
        physical_binding.value_id, std::int64_t{8});

    const auto hard_check = hard.check_constraint_satisfiability(
        virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)),
        virtual_binding.value_id,
        physical_graph.edge_attrs(physical_graph.edge(0U, 1U)),
        physical_binding.value_id,
        attribute::ComparisonOperation::less_equal);
    expect(hard_check.flag, "hard resource check");
    const auto soft_check = soft.check_constraint_satisfiability(
        virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)),
        virtual_binding.value_id,
        physical_graph.edge_attrs(physical_graph.edge(0U, 1U)),
        physical_binding.value_id,
        attribute::ComparisonOperation::greater_equal);
    expect(soft_check.flag, "soft resource policy");

    expect(hard.update_path(
               virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)),
               virtual_binding.value_id,
               physical_graph,
               {0U, 1U, 2U},
               physical_binding,
               attribute::ResourceUpdateOperation::subtract,
               true),
           "resource path subtract result");
    expect(std::get<std::int64_t>(
               physical_graph.edge_attrs(physical_graph.edge(0U, 1U))
                   .at(physical_binding.value_id)) == 7,
           "first path update");
    expect(std::get<std::int64_t>(
               physical_graph.edge_attrs(physical_graph.edge(1U, 2U))
                   .at(physical_binding.value_id)) == 5,
           "second path update");
    expect_link_error(
        attribute::LinkAttributeErrorCode::path_too_short,
        attribute::LinkAttributeOperation::update_path,
        [&] {
            hard.update_path(
                virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)),
                virtual_binding.value_id,
                physical_graph,
                {0U},
                physical_binding,
                attribute::ResourceUpdateOperation::add,
                true);
        },
        "short path");

    physical_graph.edge_attrs(physical_graph.edge(0U, 1U)).set(
        physical_binding.value_id, std::int64_t{10});
    physical_graph.edge_attrs(physical_graph.edge(1U, 2U)).set(
        physical_binding.value_id, std::int64_t{2});
    bool insufficient = false;
    try {
        hard.update_path(
            virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)),
            virtual_binding.value_id,
            physical_graph,
            {0U, 1U, 2U},
            physical_binding,
            attribute::ResourceUpdateOperation::subtract,
            true);
    } catch (const attribute::AttributeMethodException&) {
        insufficient = true;
    }
    expect(insufficient, "partial path should reject second edge");
    expect(std::get<std::int64_t>(
               physical_graph.edge_attrs(physical_graph.edge(0U, 1U))
                   .at(physical_binding.value_id)) == 7,
           "partial path first mutation");
    expect(std::get<std::int64_t>(
               physical_graph.edge_attrs(physical_graph.edge(1U, 2U))
                   .at(physical_binding.value_id)) == 2,
           "partial path failing edge unchanged");

    const auto originator_binding = hard.bind(physical_graph);
    expect_values(
        extrema.generate_from_resolved_originator(
            physical_graph, hard, originator_binding, 8U),
        {AttrValue{std::int64_t{7}}, AttrValue{std::int64_t{2}}},
        "extrema delegation");
}

void set_position(
    Graph& graph,
    Vertex node,
    AttrId id,
    std::initializer_list<AttrValue> values) {
    graph.node_attrs(node).set(
        id,
        make_attr_list(std::vector<AttrValue>(values)));
}

void test_latency() {
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
    expect(position_binding.value_id == first_position,
           "position resolver must retain first matching field");
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        expect_doubles(
            latency.generate_from_position(graph, position_binding, workers),
            {20.0, 18.0},
            "position latency worker " + std::to_string(workers));
    }

    attribute::LinkLatencySpec non_generative_spec;
    non_generative_spec.generation = attribute::LatencyGenerationKind::position;
    const attribute::LinkLatencyAttribute non_generative(non_generative_spec);
    expect_link_error(
        attribute::LinkAttributeErrorCode::non_generative_latency,
        attribute::LinkAttributeOperation::generate_latency,
        [&] {
            static_cast<void>(non_generative.generate_from_position(
                graph, position_binding, 1U));
        },
        "non-generative latency");

    Graph virtual_graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    Graph physical_graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    const auto virtual_binding = latency.bind(virtual_graph);
    const auto physical_binding = latency.bind(physical_graph);
    virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)).set(
        virtual_binding.value_id, 10.0);
    physical_graph.edge_attrs(physical_graph.edge(0U, 1U)).set(
        physical_binding.value_id, 3.0);
    physical_graph.edge_attrs(physical_graph.edge(1U, 2U)).set(
        physical_binding.value_id, 4.0);
    const std::vector<const AttrMap*> path = {
        &physical_graph.edge_attrs(physical_graph.edge(0U, 1U)),
        &physical_graph.edge_attrs(physical_graph.edge(1U, 2U))};
    const auto result = latency.check_constraint_satisfiability(
        virtual_graph.edge_attrs(virtual_graph.edge(0U, 1U)),
        virtual_binding.value_id,
        path,
        physical_binding.value_id,
        attribute::ComparisonOperation::greater_equal);
    expect(result.flag, "latency path check");
    expect(std::get<double>(result.offset) == -3.0, "latency path offset");
}

void test_concurrent_graphs() {
    std::vector<std::exception_ptr> errors(4U);
    std::vector<std::thread> threads;
    for (std::size_t worker = 0U; worker < errors.size(); ++worker) {
        threads.emplace_back([worker, &errors] {
            try {
                Graph graph(
                    128U,
                    [&] {
                        std::vector<EdgeEndpoints> edges;
                        edges.reserve(127U);
                        for (Vertex node = 1U; node < 128U; ++node) {
                            edges.push_back({node - 1U, node});
                        }
                        return edges;
                    }());
                const attribute::LinkAttribute item(link_spec("load"));
                const auto binding = item.bind(graph);
                std::vector<AttrValue> values(graph.num_edges());
                for (std::size_t index = 0U; index < values.size(); ++index) {
                    values[index] = static_cast<std::int64_t>(index + worker);
                }
                item.set_data_dense(graph, values, binding, 8U);
                expect_values(item.get_data(graph, binding, 8U), values,
                              "concurrent graph");
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
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

}  // namespace

int main() {
    try {
        test_resolvers_and_base();
        test_adapter_for_graph<Graph>("graph");
        test_adapter_for_graph<DiGraph>("digraph");
        test_matrix_and_aggregation();
        test_typed_classes_and_resource();
        test_latency();
        test_concurrent_graphs();
        std::cout << "link_attribute_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "link_attribute_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
