#include "attribute/node_attribute.h"

#include "numpy_random_state.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Callable>
void expect_node_error(
    Callable&& callable,
    attribute::NodeAttributeErrorCode code,
    attribute::NodeAttributeOperation operation) {
    try {
        std::forward<Callable>(callable)();
    } catch (const attribute::NodeAttributeException& error) {
        expect(error.code() == code, "node error code drift");
        expect(error.operation() == operation, "node error operation drift");
        return;
    }
    fail("expected NodeAttributeException");
}

attribute::BaseAttributeSpec base_node_spec(
    std::string name,
    attribute::AttributeKind kind = attribute::AttributeKind::status) {
    attribute::BaseAttributeSpec result;
    result.name = std::move(name);
    result.owner = attribute::AttributeOwner::node;
    result.kind = kind;
    return result;
}

std::vector<AttrValue> ordinary_values(std::size_t count) {
    std::vector<AttrValue> result;
    result.reserve(count + 2U);
    for (std::size_t index = 0U; index < count + 2U; ++index) {
        switch (index % 4U) {
            case 0U:
                result.emplace_back(static_cast<std::int64_t>(index));
                break;
            case 1U:
                result.emplace_back(static_cast<double>(index) + 0.25);
                break;
            case 2U:
                result.emplace_back(index % 2U == 0U);
                break;
            default:
                result.emplace_back(std::string("value-") + std::to_string(index));
                break;
        }
    }
    return result;
}

void expect_values_equal(
    const std::vector<AttrValue>& actual,
    const std::vector<AttrValue>& expected,
    const std::string& context) {
    expect(actual.size() == expected.size(), context + ": size drift");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        if (!attr_value_equal(actual[index], expected[index])) {
            fail(context + ": value drift at " + std::to_string(index));
        }
    }
}

template <typename GraphType>
void test_basic_graph_adapter(const std::string& context) {
    constexpr std::size_t count = 31U;
    GraphType graph(count, std::vector<EdgeEndpoints>{});
    const attribute::NodeAttribute value(base_node_spec("cpu"));
    const attribute::NodeAttributeBinding binding = value.bind(graph);
    const std::vector<AttrValue> input = ordinary_values(count);

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        GraphType target(count, std::vector<EdgeEndpoints>{});
        const auto target_binding = value.bind(target);
        value.set_data_dense(target, input, target_binding, workers);
        std::vector<AttrValue> expected(input.begin(), input.begin() + count);
        expect_values_equal(
            value.get_data(target, target_binding, workers),
            expected,
            context + " dense worker");
        for (std::size_t node = 0U; node < count; ++node) {
            expect(
                attr_value_equal(value.get(target, node, target_binding), input[node]),
                context + ": direct get drift");
        }
    }

    const std::vector<attribute::NodeAttributeAssignment> sparse = {
        {2U, AttrValue{std::int64_t{10}}},
        {count + 5U, AttrValue{std::int64_t{99}}},
        {2U, AttrValue{std::int64_t{11}}},
        {7U, AttrValue{std::string{"seven"}}},
    };
    value.set_data(graph, sparse, binding);
    expect(std::get<std::int64_t>(value.get(graph, 2U, binding)) == 11,
           context + ": sparse last write drift");
    expect(std::get<std::string>(value.get(graph, 7U, binding)) == "seven",
           context + ": sparse string drift");
    expect(value.get_data(graph, binding, 8U).size() == 2U,
           context + ": sparse omission drift");

    const GraphType untouched = graph;
    const std::vector<AttrValue> short_values(count - 1U, AttrValue{0.0});
    expect_node_error(
        [&] { value.set_data_dense(graph, short_values, binding, 8U); },
        attribute::NodeAttributeErrorCode::dense_data_too_short,
        attribute::NodeAttributeOperation::set_data);
    expect_values_equal(
        value.get_data(graph, binding),
        value.get_data(untouched, value.bind(untouched)),
        context + " no partial short write");

    expect_node_error(
        [&] { static_cast<void>(value.get(graph, count + 1U, binding)); },
        attribute::NodeAttributeErrorCode::node_out_of_range,
        attribute::NodeAttributeOperation::get);
    expect_node_error(
        [&] { static_cast<void>(value.get(graph, 0U, binding)); },
        attribute::NodeAttributeErrorCode::missing_attribute,
        attribute::NodeAttributeOperation::get);
}

void test_registry_separation() {
    Graph virtual_graph(2U, std::vector<EdgeEndpoints>{});
    Graph physical_graph(2U, std::vector<EdgeEndpoints>{});
    static_cast<void>(physical_graph.attr_id("unrelated-a"));
    static_cast<void>(physical_graph.attr_id("unrelated-b"));
    const attribute::NodeAttribute value(base_node_spec("cpu"));
    const auto virtual_binding = value.bind(virtual_graph);
    const auto physical_binding = value.bind(physical_graph);
    expect(virtual_binding.value_id != physical_binding.value_id,
           "test did not create distinct registry IDs");
    value.set_data_dense(
        virtual_graph,
        std::vector<AttrValue>{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{3}}},
        virtual_binding);
    value.set_data_dense(
        physical_graph,
        std::vector<AttrValue>{AttrValue{std::int64_t{20}}, AttrValue{std::int64_t{30}}},
        physical_binding);
    expect(std::get<std::int64_t>(value.get(virtual_graph, 1U, virtual_binding)) == 3,
           "virtual registry binding drift");
    expect(std::get<std::int64_t>(value.get(physical_graph, 1U, physical_binding)) == 30,
           "physical registry binding drift");
}

void test_typed_classes_and_extrema() {
    attribute::NodeStatusSpec status_spec;
    status_spec.name = "alive";
    const attribute::NodeStatusAttribute status(status_spec);
    expect(status.spec().owner == attribute::AttributeOwner::node,
           "status owner drift");
    expect(status.spec().kind == attribute::AttributeKind::status,
           "status kind drift");
    expect(status.spec().is_constraint == std::optional<bool>{false},
           "status constraint drift");

    attribute::NodeExtremaSpec missing;
    missing.name = "cpu_max";
    expect_node_error(
        [&] { static_cast<void>(attribute::NodeExtremaAttribute(missing)); },
        attribute::NodeAttributeErrorCode::missing_originator,
        attribute::NodeAttributeOperation::construct);

    attribute::NodeExtremaSpec extrema_spec;
    extrema_spec.name = "cpu_max";
    extrema_spec.originator_name = "cpu";
    extrema_spec.originator_id = 17U;
    const attribute::NodeExtremaAttribute extrema(extrema_spec);
    expect(extrema.originator_name() == "cpu", "extrema originator name drift");
    expect(extrema.originator_id() == 17U, "extrema definition ID drift");
    expect(extrema.spec().is_constraint == std::optional<bool>{false},
           "extrema constraint drift");

    Graph graph(9U, std::vector<EdgeEndpoints>{});
    const attribute::NodeAttribute originator(base_node_spec("cpu"));
    const auto originator_binding = originator.bind(graph);
    const std::vector<AttrValue> values = ordinary_values(graph.num_nodes());
    originator.set_data_dense(graph, values, originator_binding, 2U);
    const auto value_count =
        static_cast<std::vector<AttrValue>::difference_type>(graph.num_nodes());
    std::vector<AttrValue> expected(values.begin(), values.begin() + value_count);
    expect_values_equal(
        extrema.generate_from_resolved_originator(
            graph, originator, originator_binding, 8U),
        expected,
        "extrema delegation");
}

attribute::NodeResourceAttribute resource_attribute(
    attribute::ConstraintRestriction restriction =
        attribute::ConstraintRestriction::hard) {
    attribute::NodeResourceSpec spec;
    spec.name = "cpu";
    spec.restriction = restriction;
    spec.checking_level = attribute::CheckingLevel::node;
    return attribute::NodeResourceAttribute(std::move(spec));
}

void test_resource_policy() {
    const auto hard = resource_attribute();
    const auto soft = resource_attribute(attribute::ConstraintRestriction::soft);
    expect(hard.checking_level() == attribute::CheckingLevel::node,
           "resource checking level drift");
    expect(hard.spec().kind == attribute::AttributeKind::resource,
           "resource kind drift");
    expect(hard.spec().is_constraint == std::optional<bool>{true},
           "resource constraint drift");

    Graph virtual_graph(1U, std::vector<EdgeEndpoints>{});
    Graph physical_graph(1U, std::vector<EdgeEndpoints>{});
    static_cast<void>(physical_graph.attr_id("other"));
    const auto virtual_binding = hard.bind(virtual_graph);
    const auto physical_binding = hard.bind(physical_graph);
    virtual_graph.node_attrs(0U).set(virtual_binding.value_id, std::int64_t{7});
    physical_graph.node_attrs(0U).set(physical_binding.value_id, 10.5);

    const auto result = hard.check_constraint_satisfiability(
        virtual_graph.node_attrs(0U), virtual_binding.value_id,
        physical_graph.node_attrs(0U), physical_binding.value_id,
        attribute::ComparisonOperation::less_equal);
    expect(result.flag, "hard resource flag drift");
    expect(double_bits(std::get<double>(result.offset)) == double_bits(-3.5),
           "hard resource offset drift");

    virtual_graph.node_attrs(0U).set(virtual_binding.value_id, 12.0);
    const auto soft_result = soft.check_constraint_satisfiability(
        virtual_graph.node_attrs(0U), virtual_binding.value_id,
        physical_graph.node_attrs(0U), physical_binding.value_id,
        attribute::ComparisonOperation::less_equal);
    expect(soft_result.flag, "soft resource flag drift");
    expect(double_bits(std::get<double>(soft_result.offset)) == double_bits(1.5),
           "soft resource offset drift");

    virtual_graph.node_attrs(0U).set(virtual_binding.value_id, std::int64_t{2});
    physical_graph.node_attrs(0U).set(physical_binding.value_id, std::int64_t{10});
    expect(hard.update(
               virtual_graph.node_attrs(0U), virtual_binding.value_id,
               physical_graph.node_attrs(0U), physical_binding.value_id,
               attribute::ResourceUpdateOperation::subtract, true),
           "resource update return drift");
    expect(std::get<std::int64_t>(
               physical_graph.node_attrs(0U).at(physical_binding.value_id)) == 8,
           "resource subtract drift");
    static_cast<void>(hard.update(
        virtual_graph.node_attrs(0U), virtual_binding.value_id,
        physical_graph.node_attrs(0U), physical_binding.value_id,
        attribute::ResourceUpdateOperation::add, true));
    expect(std::get<std::int64_t>(
               physical_graph.node_attrs(0U).at(physical_binding.value_id)) == 10,
           "resource add drift");

    Graph missing_graph(1U, std::vector<EdgeEndpoints>{});
    const auto missing_id = hard.bind(missing_graph).value_id;
    expect_node_error(
        [&] {
            static_cast<void>(hard.check_constraint_satisfiability(
                missing_graph.node_attrs(0U), missing_id,
                physical_graph.node_attrs(0U), physical_binding.value_id));
        },
        attribute::NodeAttributeErrorCode::missing_resource_value,
        attribute::NodeAttributeOperation::check_resource);
    missing_graph.node_attrs(0U).set(missing_id, std::string{"bad"});
    expect_node_error(
        [&] {
            static_cast<void>(hard.check_constraint_satisfiability(
                missing_graph.node_attrs(0U), missing_id,
                physical_graph.node_attrs(0U), physical_binding.value_id));
        },
        attribute::NodeAttributeErrorCode::non_numeric_resource,
        attribute::NodeAttributeOperation::check_resource);

    Graph missing_physical(1U, std::vector<EdgeEndpoints>{});
    const auto missing_physical_id = hard.bind(missing_physical).value_id;
    expect_node_error(
        [&] {
            static_cast<void>(hard.check_constraint_satisfiability(
                missing_graph.node_attrs(0U), missing_id,
                missing_physical.node_attrs(0U), missing_physical_id));
        },
        attribute::NodeAttributeErrorCode::missing_resource_value,
        attribute::NodeAttributeOperation::check_resource);

    Graph access_virtual(1U, std::vector<EdgeEndpoints>{});
    Graph access_physical(1U, std::vector<EdgeEndpoints>{});
    const auto access_virtual_id = hard.bind(access_virtual).value_id;
    const auto access_physical_id = hard.bind(access_physical).value_id;
    access_physical.node_attrs(0U).set(
        access_physical_id, std::string{"bad"});
    for (const auto operation : {
             attribute::ResourceUpdateOperation::add,
             attribute::ResourceUpdateOperation::subtract}) {
        expect_node_error(
            [&] {
                static_cast<void>(hard.update(
                    access_virtual.node_attrs(0U), access_virtual_id,
                    access_physical.node_attrs(0U), access_physical_id,
                    operation, false));
            },
            attribute::NodeAttributeErrorCode::missing_resource_value,
            attribute::NodeAttributeOperation::update_resource);
    }
    access_virtual.node_attrs(0U).set(
        access_virtual_id, std::string{"bad"});
    static_cast<void>(access_physical.node_attrs(0U).erase(access_physical_id));
    expect_node_error(
        [&] {
            static_cast<void>(hard.update(
                access_virtual.node_attrs(0U), access_virtual_id,
                access_physical.node_attrs(0U), access_physical_id,
                attribute::ResourceUpdateOperation::subtract, true));
        },
        attribute::NodeAttributeErrorCode::missing_resource_value,
        attribute::NodeAttributeOperation::update_resource);
}

utils::DistributionSpec uniform_distribution(double low, double high) {
    utils::DistributionSpec result;
    result.kind = utils::DistributionKind::uniform;
    result.low = utils::DatasetScalar{low};
    result.high = utils::DatasetScalar{high};
    return result;
}

template <typename Value>
attribute::AttributeNumber expected_number(Value value) {
    if constexpr (std::is_same_v<Value, std::uint8_t>) {
        return value != 0U;
    } else {
        return value;
    }
}

template <typename Value>
double expected_double(Value value) {
    if constexpr (std::is_same_v<Value, std::uint8_t>) {
        return value == 0U ? 0.0 : 1.0;
    } else {
        return static_cast<double>(value);
    }
}

bool number_equal(
    const attribute::AttributeNumber& left,
    const attribute::AttributeNumber& right) {
    if (left.index() != right.index()) {
        return false;
    }
    if (const auto* value = std::get_if<double>(&left)) {
        return double_bits(*value) == double_bits(std::get<double>(right));
    }
    return left == right;
}

std::vector<attribute::NodePositionValue> expected_positions(
    const utils::GeneratedData& x,
    const utils::GeneratedData& y,
    const utils::GeneratedData& radius,
    double minimum,
    double maximum) {
    std::vector<attribute::NodePositionValue> result;
    std::visit(
        [&](const auto& xs) {
            std::visit(
                [&](const auto& ys) {
                    std::visit(
                        [&](const auto& rs) {
                            result.resize(xs.size());
                            for (std::size_t index = 0U; index < xs.size(); ++index) {
                                result[index].x = expected_number(xs[index]);
                                result[index].y = expected_number(ys[index]);
                                const double value = expected_double(rs[index]);
                                result[index].radius =
                                    value < minimum ? minimum
                                    : (value > maximum ? maximum : value);
                            }
                        },
                        radius.values);
                },
                y.values);
        },
        x.values);
    return result;
}

void test_position_generation() {
    attribute::NodePositionSpec spec;
    spec.generative = true;
    spec.distribution = uniform_distribution(-2.0, 3.0);
    spec.dtype = utils::DatasetValueKind::floating;
    spec.minimum_radius = -0.25;
    spec.maximum_radius = 0.75;
    const attribute::NodePositionAttribute position(spec);
    expect(position.spec().kind == attribute::AttributeKind::position,
           "position kind drift");
    expect(position.spec().is_constraint == std::optional<bool>{true},
           "position constraint drift");

    const attribute::NetworkCardinality network{257U, 0U};
    NumpyRandomState expected_rng(701U);
    attribute::BaseAttributeSpec base_spec = position.spec();
    const attribute::BaseAttribute base(std::move(base_spec));
    const auto x = base.generate_configured_data(network, expected_rng, 1U);
    const auto y = base.generate_configured_data(network, expected_rng, 1U);
    const auto radius = base.generate_configured_data(network, expected_rng, 1U);
    const auto expected = expected_positions(
        x, y, radius, spec.minimum_radius, spec.maximum_radius);
    const std::uint64_t expected_next = double_bits(expected_rng.random());

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        NumpyRandomState rng(701U);
        const auto actual = position.generate_positions(network, rng, workers);
        expect(actual.size() == expected.size(), "position size drift");
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            expect(number_equal(actual[index].x, expected[index].x),
                   "position x drift");
            expect(number_equal(actual[index].y, expected[index].y),
                   "position y drift");
            expect(double_bits(actual[index].radius)
                       == double_bits(expected[index].radius),
                   "position radius drift");
        }
        expect(double_bits(rng.random()) == expected_next,
               "position RNG continuation drift");
    }

    attribute::NodePositionSpec reversed = spec;
    reversed.minimum_radius = 2.0;
    reversed.maximum_radius = -1.0;
    const attribute::NodePositionAttribute reversed_position(reversed);
    NumpyRandomState reversed_rng(702U);
    const auto reversed_values = reversed_position.generate_positions(
        attribute::NetworkCardinality{17U, 0U}, reversed_rng, 8U);
    for (const auto& item : reversed_values) {
        expect(double_bits(item.radius) == double_bits(-1.0),
               "reversed clip bounds drift");
    }
}

void test_existing_position() {
    attribute::NodePositionSpec spec;
    spec.name = "custom_position_name";
    const attribute::NodePositionAttribute position(spec);

    Graph empty;
    const auto empty_binding = position.bind_existing_pos(empty);
    expect_node_error(
        [&] {
            static_cast<void>(position.get_existing_pos_data(
                empty, empty_binding));
        },
        attribute::NodeAttributeErrorCode::empty_position_network,
        attribute::NodeAttributeOperation::get_existing_position);

    Graph graph(4U, std::vector<EdgeEndpoints>{});
    const auto binding = position.bind_existing_pos(graph);
    expect(graph.attr_name(binding.value_id) == "pos",
           "position fallback did not bind literal pos");
    expect_node_error(
        [&] {
            static_cast<void>(position.get_existing_pos_data(graph, binding));
        },
        attribute::NodeAttributeErrorCode::missing_position_data,
        attribute::NodeAttributeOperation::get_existing_position);

    graph.node_attrs(0U).set(binding.value_id, std::string{"p0"});
    graph.node_attrs(2U).set(binding.value_id, std::string{"p2"});
    const auto values = position.get_existing_pos_data(graph, binding, 8U);
    expect(values.size() == 2U, "position fallback omission drift");
    expect(std::get<std::string>(values[0]) == "p0", "position fallback order 0");
    expect(std::get<std::string>(values[1]) == "p2", "position fallback order 1");
}

void test_concurrent_independent_graphs() {
    constexpr std::size_t callers = 4U;
    std::array<std::exception_ptr, callers> errors{};
    std::vector<std::thread> threads;
    for (std::size_t caller = 0U; caller < callers; ++caller) {
        threads.emplace_back([caller, &errors] {
            try {
                Graph graph(4097U, std::vector<EdgeEndpoints>{});
                const attribute::NodeAttribute value(base_node_spec("load"));
                const auto binding = value.bind(graph);
                std::vector<AttrValue> input(
                    graph.num_nodes(),
                    AttrValue{static_cast<std::int64_t>(caller)});
                value.set_data_dense(graph, input, binding, caller == 0U ? 1U : 8U);
                const auto output = value.get_data(graph, binding, 8U);
                if (output.size() != graph.num_nodes()) {
                    fail("concurrent output size drift");
                }
            } catch (...) {
                errors[caller] = std::current_exception();
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

}  // namespace

int main() {
    try {
        test_basic_graph_adapter<Graph>("Graph");
        test_basic_graph_adapter<DiGraph>("DiGraph");
        test_registry_separation();
        test_typed_classes_and_extrema();
        test_resource_policy();
        test_position_generation();
        test_existing_position();
        test_concurrent_independent_graphs();
        std::cout << "node_attribute_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "node_attribute_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
