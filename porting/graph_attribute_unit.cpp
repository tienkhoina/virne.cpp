#include "../virne/network/attribute/graph_attribute.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace virne::network::attribute;

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        fail(message);
    }
}

std::uint64_t double_bits(double value) noexcept {
    std::uint64_t result = 0U;
    static_assert(sizeof(result) == sizeof(value), "unexpected double width");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double double_from_bits(std::uint64_t bits) noexcept {
    double result = 0.0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool exact_attr_value(const AttrValue& lhs, const AttrValue& rhs);

bool exact_attr_list(const AttrListPtr& lhs, const AttrListPtr& rhs) {
    if (!lhs || !rhs) {
        return lhs == rhs;
    }
    if (lhs->values.size() != rhs->values.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < lhs->values.size(); ++index) {
        if (!exact_attr_value(lhs->values[index], rhs->values[index])) {
            return false;
        }
    }
    return true;
}

bool exact_attr_object(const AttrObjectPtr& lhs, const AttrObjectPtr& rhs) {
    if (!lhs || !rhs) {
        return lhs == rhs;
    }
    if (lhs->entries.size() != rhs->entries.size()) {
        return false;
    }
    for (std::size_t index = 0U; index < lhs->entries.size(); ++index) {
        if (lhs->entries[index].first != rhs->entries[index].first ||
            !exact_attr_value(
                lhs->entries[index].second,
                rhs->entries[index].second)) {
            return false;
        }
    }
    return true;
}

bool exact_attr_value(const AttrValue& lhs, const AttrValue& rhs) {
    if (lhs.index() != rhs.index()) {
        return false;
    }
    if (const auto* value = std::get_if<std::int64_t>(&lhs)) {
        return *value == std::get<std::int64_t>(rhs);
    }
    if (const auto* value = std::get_if<double>(&lhs)) {
        return double_bits(*value) == double_bits(std::get<double>(rhs));
    }
    if (const auto* value = std::get_if<bool>(&lhs)) {
        return *value == std::get<bool>(rhs);
    }
    if (const auto* value = std::get_if<std::string>(&lhs)) {
        return *value == std::get<std::string>(rhs);
    }
    if (const auto* value = std::get_if<AttrListPtr>(&lhs)) {
        return exact_attr_list(*value, std::get<AttrListPtr>(rhs));
    }
    return exact_attr_object(
        std::get<AttrObjectPtr>(lhs),
        std::get<AttrObjectPtr>(rhs));
}

void require_exact(
    const AttrValue& actual,
    const AttrValue& expected,
    std::string_view message) {
    require(exact_attr_value(actual, expected), message);
}

void require_exact_values(
    const std::vector<AttrValue>& actual,
    const std::vector<AttrValue>& expected,
    std::string_view message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require_exact(actual[index], expected[index], message);
    }
}

void require_number(
    const AttributeNumber& actual,
    const AttributeNumber& expected,
    std::string_view message) {
    require(actual.index() == expected.index(), message);
    if (const auto* value = std::get_if<bool>(&actual)) {
        require(*value == std::get<bool>(expected), message);
        return;
    }
    if (const auto* value = std::get_if<std::int64_t>(&actual)) {
        require(*value == std::get<std::int64_t>(expected), message);
        return;
    }
    require(
        double_bits(std::get<double>(actual)) ==
            double_bits(std::get<double>(expected)),
        message);
}

template <typename Function>
std::string expect_graph_error(
    Function&& function,
    GraphAttributeErrorCode code,
    GraphAttributeOperation operation,
    std::string_view message) {
    try {
        function();
    } catch (const GraphAttributeException& error) {
        if (error.code() != code || error.operation() != operation) {
            throw std::runtime_error(
                std::string(message) + " (actual graph error code=" +
                std::to_string(static_cast<unsigned int>(error.code())) +
                ", operation=" +
                std::to_string(static_cast<unsigned int>(error.operation())) +
                ")");
        }
        return error.what();
    } catch (const std::exception&) {
        fail(message);
    }
    fail(message);
}

template <typename Function>
void expect_method_error(
    Function&& function,
    AttributeMethodErrorCode code,
    AttributeMethodOperation operation,
    std::string_view message) {
    try {
        function();
    } catch (const AttributeMethodException& error) {
        if (error.code() != code || error.operation() != operation) {
            throw std::runtime_error(
                std::string(message) + " (actual method error code=" +
                std::to_string(static_cast<unsigned int>(error.code())) +
                ", operation=" +
                std::to_string(static_cast<unsigned int>(error.operation())) +
                ")");
        }
        return;
    } catch (const std::exception&) {
        fail(message);
    }
    fail(message);
}

BaseAttributeSpec graph_spec(std::string name, AttributeKind kind) {
    BaseAttributeSpec spec;
    spec.name = std::move(name);
    spec.owner = AttributeOwner::graph;
    spec.kind = kind;
    return spec;
}

GraphResourceSpec graph_resource_spec(std::string name) {
    GraphResourceSpec spec;
    spec.name = std::move(name);
    return spec;
}

template <typename GraphType>
AttrId intern_attr(GraphType& graph, std::string_view name) {
    graph.graph_attrs()[name] = std::int64_t{0};
    const AttrId id = graph.attr_id(name);
    require(graph.graph_attrs().erase(id), "fixture attribute interning must erase its slot");
    return id;
}

std::vector<AttrValue> lane_values(std::uint64_t salt = 0U) {
    const std::uint64_t nan_bits = UINT64_C(0x7ff8000000000042) + salt;
    return {
        std::numeric_limits<std::int64_t>::min(),
        double_from_bits(UINT64_C(0x8000000000000000)),
        double_from_bits(nan_bits),
        true,
        std::string("metadata"),
        make_attr_list({
            std::int64_t{7},
            double_from_bits(UINT64_C(0x8000000000000000)),
            double_from_bits(nan_bits)}),
        make_attr_object({
            {"tag", std::string("nested")},
            {"payload", double_from_bits(nan_bits)}}),
        AttrListPtr{},
        AttrObjectPtr{},
    };
}

template <typename GraphType>
void test_scalar_surface(std::string_view type_name) {
    GraphAttribute attribute(graph_spec("meta", AttributeKind::status));

    GraphType missing_registry;
    const GraphAttributeBinding missing_binding =
        attribute.bind(missing_registry);
    require(
        missing_binding.value_id == missing_registry.attr_id("meta"),
        "bind must intern and return the graph-local AttrId");
    expect_graph_error(
        [&]() {
            static_cast<void>(attribute.get(
                missing_registry, missing_binding));
        },
        GraphAttributeErrorCode::missing_attribute,
        GraphAttributeOperation::get,
        "get must reject a bound but unset graph attribute");

    GraphType graph;
    const AttrId id = intern_attr(graph, "meta");
    const GraphAttributeBinding binding = attribute.bind(graph);
    require(binding.value_id == id, "bind must return the graph-local AttrId");

    expect_graph_error(
        [&]() { static_cast<void>(attribute.get(graph, binding)); },
        GraphAttributeErrorCode::missing_attribute,
        GraphAttributeOperation::get,
        "get must distinguish an interned but missing graph slot");
    expect_graph_error(
        [&]() { static_cast<void>(attribute.get_data(graph, binding)); },
        GraphAttributeErrorCode::missing_attribute,
        GraphAttributeOperation::get_data,
        "get_data must distinguish an interned but missing graph slot");

    const GraphAttributeBinding invalid{
        std::numeric_limits<AttrId>::max()};
    expect_graph_error(
        [&]() { static_cast<void>(attribute.get(graph, invalid)); },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::get,
        "get must reject an AttrId outside the local registry");
    expect_graph_error(
        [&]() { attribute.set_data(graph, std::int64_t{1}, invalid); },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::set_data,
        "set_data must reject an AttrId outside the local registry");

    const std::vector<AttrValue> values = lane_values();
    for (const AttrValue& value : values) {
        attribute.set_data(graph, value, binding);
        const AttrValue& from_get = attribute.get(graph, binding);
        const AttrValue& from_get_data = attribute.get_data(graph, binding);
        require_exact(from_get, value, "set/get must preserve every AttrValue lane");
        require_exact(
            from_get_data,
            value,
            "set/get_data must preserve recursive values and floating bits");
        require(
            &from_get == &graph.graph_attrs().at(id) &&
                &from_get_data == &graph.graph_attrs().at(id),
            "scalar getters must return the live graph slot by const reference");
        if (const auto* list = std::get_if<AttrListPtr>(&value)) {
            require(
                std::get<AttrListPtr>(from_get) == *list,
                "set_data must retain recursive list object identity");
        }
        if (const auto* object = std::get_if<AttrObjectPtr>(&value)) {
            require(
                std::get<AttrObjectPtr>(from_get) == *object,
                "set_data must retain recursive object identity");
        }
    }

    graph.graph_attrs().at(id) = std::string(type_name);
    require_exact(
        attribute.get(graph, binding),
        std::string(type_name),
        "a live scalar reference must observe direct slot overwrite");

    GraphAttribute empty_name(graph_spec("", AttributeKind::status));
    const AttrId empty_id = intern_attr(graph, "");
    const GraphAttributeBinding empty_binding = empty_name.bind(graph);
    require(empty_binding.value_id == empty_id, "an empty graph name is valid");
    empty_name.set_data(graph, std::int64_t{91}, empty_binding);
    require_exact(
        empty_name.get_data(graph, empty_binding),
        std::int64_t{91},
        "empty-name set/get_data must use its resolved ID");
}

void test_graph_local_bindings() {
    GraphAttribute attribute(graph_spec("capacity", AttributeKind::resource));
    Graph first;
    DiGraph second;
    const AttrId first_id = intern_attr(first, "capacity");
    static_cast<void>(intern_attr(second, "padding-0"));
    static_cast<void>(intern_attr(second, "padding-1"));
    const AttrId second_id = intern_attr(second, "capacity");
    require(first_id != second_id, "fixture must create distinct registry IDs");

    const GraphAttributePairBinding pair{
        attribute.bind(first),
        attribute.bind(second)};
    require(
        pair.virtual_graph.value_id == first_id &&
            pair.physical_graph.value_id == second_id,
        "each graph must resolve the dynamic name independently");
    attribute.set_data(first, std::int64_t{11}, pair.virtual_graph);
    attribute.set_data(second, std::int64_t{29}, pair.physical_graph);
    require_exact(
        attribute.get(first, pair.virtual_graph),
        std::int64_t{11},
        "first graph binding must remain local");
    require_exact(
        attribute.get(second, pair.physical_graph),
        std::int64_t{29},
        "second graph binding must remain local");

    expect_graph_error(
        [&]() {
            static_cast<void>(attribute.get(second, pair.virtual_graph));
        },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::get,
        "a same-range foreign ID must not read another graph field");
    expect_graph_error(
        [&]() {
            attribute.set_data(
                second, std::int64_t{99}, pair.virtual_graph);
        },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::set_data,
        "a same-range foreign ID must not overwrite another graph field");
    expect_graph_error(
        [&]() {
            static_cast<void>(attribute.get_data_batch(
                {{&second.graph_attrs(), pair.virtual_graph}}, 8U));
        },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::get_data_batch,
        "batch validation must reject a same-range foreign ID before workers");
}

void test_fixed_specs() {
    BaseAttributeSpec invalid = graph_spec("bad", AttributeKind::status);
    invalid.owner = AttributeOwner::node;
    expect_graph_error(
        [&]() { GraphAttribute rejected(std::move(invalid)); },
        GraphAttributeErrorCode::invalid_graph_spec,
        GraphAttributeOperation::construct,
        "GraphAttribute must reject a non-graph owner");

    GraphStatusSpec status_spec;
    status_spec.name = "ready";
    status_spec.generative = true;
    GraphStatusAttribute status(std::move(status_spec));
    require(status.spec().name == "ready", "status name must be retained");
    require(
        status.spec().owner == AttributeOwner::graph &&
            status.spec().kind == AttributeKind::status,
        "status owner and kind must be fixed typed fields");
    require(
        status.spec().is_constraint == std::optional<bool>{false},
        "status information policy must force is_constraint=false");
    require(status.spec().generative, "status generation field must be retained");

    expect_graph_error(
        []() {
            GraphExtremaAttribute rejected(
                GraphExtremaSpec{"peak", std::nullopt, 3U});
        },
        GraphAttributeErrorCode::missing_originator,
        GraphAttributeOperation::construct,
        "graph extrema must require a non-null originator");

    GraphExtremaAttribute empty_originator(
        GraphExtremaSpec{"peak", std::string{}, 41U});
    require(
        empty_originator.originator_name().empty() &&
            empty_originator.originator_id() == 41U,
        "an empty but present extrema originator must remain valid");
    require(
        empty_originator.spec().owner == AttributeOwner::graph &&
            empty_originator.spec().kind == AttributeKind::extrema &&
            empty_originator.spec().is_constraint == std::optional<bool>{false},
        "extrema fixed graph/information fields must be direct invariants");

    GraphResourceSpec resource_spec;
    resource_spec.name = "capacity";
    resource_spec.generative = true;
    resource_spec.restriction = ConstraintRestriction::soft;
    resource_spec.checking_level = CheckingLevel::path;
    GraphResourceAttribute resource(std::move(resource_spec));
    require(
        resource.spec().owner == AttributeOwner::graph &&
            resource.spec().kind == AttributeKind::resource &&
            resource.spec().is_constraint == std::optional<bool>{true},
        "resource fixed graph/constraint fields must be direct invariants");
    require(
        resource.restriction() == ConstraintRestriction::soft &&
            resource.checking_level() == CheckingLevel::path,
        "resource restriction and checking level must be typed direct fields");
    require(resource.spec().generative, "resource generation field must be retained");

    GraphResourceAttribute defaults(graph_resource_spec("load"));
    require(
        defaults.restriction() == ConstraintRestriction::hard &&
            defaults.checking_level() == CheckingLevel::graph,
        "resource direct defaults must be hard/graph");
}

template <typename GraphType>
void test_extrema_delegation() {
    GraphType graph;
    graph.add_nodes_from({0U, 1U, 2U, 3U});
    const AttrId edge_id = intern_attr(graph, "edge-load");
    const auto first = graph.add_edge(2U, 3U);
    const auto omitted = graph.add_edge(0U, 2U);
    const auto last = graph.add_edge(3U, 1U);
    graph.edge_attrs(first).set(edge_id, std::int64_t{17});
    graph.edge_attrs(last).set(
        edge_id,
        double_from_bits(UINT64_C(0x8000000000000000)));
    require(
        graph.edge_attrs(omitted).find(edge_id) == nullptr,
        "fixture must retain one unattributed edge");

    BaseAttributeSpec corrected = graph_spec("edge-load", AttributeKind::status);
    corrected.owner = AttributeOwner::link;
    LinkAttribute link_originator(std::move(corrected));
    const LinkAttributeBinding link_binding = link_originator.bind(graph);

    GraphExtremaAttribute extrema(
        GraphExtremaSpec{"peak-load", std::string("edge-load"), 73U});
    std::vector<AttrValue> expected;
    if constexpr (std::is_same_v<GraphType, Graph>) {
        expected = {
            double_from_bits(UINT64_C(0x8000000000000000)),
            std::int64_t{17}};
    } else {
        expected = {
            std::int64_t{17},
            double_from_bits(UINT64_C(0x8000000000000000))};
    }
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        require_exact_values(
            extrema.generate_from_resolved_originator(
                graph,
                link_originator,
                link_binding,
                workers),
            expected,
            "graph extrema must delegate to the resolved link registry");
    }
}

struct ResourceMaps {
    Graph virtual_graph;
    DiGraph physical_graph;
    AttrId virtual_id = 0U;
    AttrId physical_id = 0U;
};

ResourceMaps resource_maps() {
    ResourceMaps maps;
    static_cast<void>(intern_attr(maps.physical_graph, "padding"));
    maps.virtual_id = intern_attr(maps.virtual_graph, "capacity");
    maps.physical_id = intern_attr(maps.physical_graph, "capacity");
    return maps;
}

void set_resource_values(ResourceMaps& maps, AttrValue virtual_value, AttrValue physical_value) {
    maps.virtual_graph.graph_attrs().set(maps.virtual_id, std::move(virtual_value));
    maps.physical_graph.graph_attrs().set(maps.physical_id, std::move(physical_value));
}

void test_resource_checks() {
    GraphResourceAttribute hard(graph_resource_spec("capacity"));
    GraphResourceSpec soft_spec;
    soft_spec.name = "capacity";
    soft_spec.restriction = ConstraintRestriction::soft;
    GraphResourceAttribute soft(std::move(soft_spec));

    ResourceMaps maps = resource_maps();
    set_resource_values(maps, std::int64_t{3}, std::int64_t{5});
    SatisfiabilityResult result = hard.check_constraint_satisfiability(
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        maps.physical_graph.graph_attrs(),
        maps.physical_id,
        ComparisonOperation::less_equal);
    require(result.flag, "hard le true flag must be retained");
    require_number(result.offset, std::int64_t{-2}, "hard le offset must be raw");

    set_resource_values(maps, std::int64_t{7}, std::int64_t{5});
    result = hard.check_constraint_satisfiability(
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        maps.physical_graph.graph_attrs(),
        maps.physical_id);
    require(!result.flag, "hard le false flag must be retained");
    require_number(result.offset, std::int64_t{2}, "hard false offset must not clamp");
    result = soft.check_constraint_satisfiability(
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        maps.physical_graph.graph_attrs(),
        maps.physical_id);
    require(result.flag, "soft restriction must force the returned flag true");
    require_number(result.offset, std::int64_t{2}, "soft restriction must preserve offset");

    set_resource_values(maps, std::int64_t{3}, std::int64_t{5});
    result = hard.check_constraint_satisfiability(
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        maps.physical_graph.graph_attrs(),
        maps.physical_id,
        ComparisonOperation::greater_equal);
    require(!result.flag, "hard ge false flag must be retained");
    require_number(result.offset, std::int64_t{2}, "ge offset must be p-v");

    set_resource_values(maps, true, false);
    result = hard.check_constraint_satisfiability(
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        maps.physical_graph.graph_attrs(),
        maps.physical_id,
        ComparisonOperation::equal);
    require(!result.flag, "bool equality must compare promoted values");
    require_number(result.offset, std::int64_t{1}, "bool equality offset must be integer");

    set_resource_values(
        maps,
        std::int64_t{INT64_C(9007199254740993)},
        double_from_bits(UINT64_C(0x4340000000000000)));
    result = hard.check_constraint_satisfiability(
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        maps.physical_graph.graph_attrs(),
        maps.physical_id,
        ComparisonOperation::greater_equal);
    require(result.flag, "mixed int/double comparison must remain Python-exact above 2^53");
    require_number(result.offset, 0.0, "mixed offset must use binary64 subtraction");

    set_resource_values(
        maps,
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max());
    expect_method_error(
        [&]() {
            static_cast<void>(hard.check_constraint_satisfiability(
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                ComparisonOperation::less_equal));
        },
        AttributeMethodErrorCode::numeric_range,
        AttributeMethodOperation::calculate_satisfiability,
        "integer offset overflow must be a typed numeric_range failure");

    set_resource_values(maps, std::int64_t{1}, std::int64_t{2});
    expect_method_error(
        [&]() {
            static_cast<void>(hard.check_constraint_satisfiability(
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                static_cast<ComparisonOperation>(255U)));
        },
        AttributeMethodErrorCode::unsupported_comparison,
        AttributeMethodOperation::calculate_satisfiability,
        "unknown comparisons must remain typed enum failures");

    maps = resource_maps();
    maps.virtual_graph.graph_attrs().set(maps.virtual_id, std::string("bad"));
    expect_graph_error(
        [&]() {
            static_cast<void>(hard.check_constraint_satisfiability(
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                maps.physical_graph.graph_attrs(),
                maps.physical_id));
        },
        GraphAttributeErrorCode::missing_resource_value,
        GraphAttributeOperation::check_resource,
        "physical missing lookup must precede virtual numeric conversion");

    maps = resource_maps();
    maps.physical_graph.graph_attrs().set(maps.physical_id, std::string("bad"));
    expect_graph_error(
        [&]() {
            static_cast<void>(hard.check_constraint_satisfiability(
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                maps.physical_graph.graph_attrs(),
                maps.physical_id));
        },
        GraphAttributeErrorCode::missing_resource_value,
        GraphAttributeOperation::check_resource,
        "virtual missing lookup must precede physical numeric conversion");

    set_resource_values(maps, make_attr_list({std::int64_t{1}}), std::int64_t{3});
    expect_graph_error(
        [&]() {
            static_cast<void>(hard.check_constraint_satisfiability(
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                maps.physical_graph.graph_attrs(),
                maps.physical_id));
        },
        GraphAttributeErrorCode::non_numeric_resource,
        GraphAttributeOperation::check_resource,
        "recursive resource values must fail before numeric work");
}

void test_resource_updates() {
    GraphResourceAttribute resource(graph_resource_spec("capacity"));
    ResourceMaps maps = resource_maps();

    expect_method_error(
        [&]() {
            static_cast<void>(resource.update(
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                static_cast<ResourceUpdateOperation>(255U),
                true));
        },
        AttributeMethodErrorCode::unsupported_update_operation,
        AttributeMethodOperation::resolve_update,
        "update operation validation must precede graph-slot lookup");

    set_resource_values(maps, std::int64_t{3}, std::int64_t{10});
    require(
        resource.update(
            maps.physical_graph.graph_attrs(),
            maps.physical_id,
            maps.virtual_graph.graph_attrs(),
            maps.virtual_id,
            ResourceUpdateOperation::add),
        "graph add must return true");
    require_exact(
        maps.physical_graph.graph_attrs().at(maps.physical_id),
        std::int64_t{13},
        "graph update must mutate target += operand");

    require(
        resource.update(
            maps.physical_graph.graph_attrs(),
            maps.physical_id,
            maps.virtual_graph.graph_attrs(),
            maps.virtual_id,
            ResourceUpdateOperation::subtract,
            false),
        "graph subtract must return true");
    require_exact(
        maps.physical_graph.graph_attrs().at(maps.physical_id),
        std::int64_t{10},
        "graph update must mutate target -= operand");

    set_resource_values(maps, std::int64_t{20}, std::int64_t{5});
    require(
        resource.update(
            maps.physical_graph.graph_attrs(),
            maps.physical_id,
            maps.virtual_graph.graph_attrs(),
            maps.virtual_id,
            ResourceUpdateOperation::subtract,
            true),
        "graph subtraction must ignore safe=true");
    require_exact(
        maps.physical_graph.graph_attrs().at(maps.physical_id),
        std::int64_t{-15},
        "graph subtraction must permit a negative result");

    set_resource_values(maps, true, true);
    resource.update(
        maps.physical_graph.graph_attrs(),
        maps.physical_id,
        maps.virtual_graph.graph_attrs(),
        maps.virtual_id,
        ResourceUpdateOperation::add);
    require_exact(
        maps.physical_graph.graph_attrs().at(maps.physical_id),
        std::int64_t{2},
        "bool graph arithmetic must promote to the integer lane");

    Graph alias_graph;
    const AttrId alias_id = intern_attr(alias_graph, "capacity");
    alias_graph.graph_attrs().set(alias_id, std::int64_t{9});
    resource.update(
        alias_graph.graph_attrs(),
        alias_id,
        alias_graph.graph_attrs(),
        alias_id,
        ResourceUpdateOperation::add,
        true);
    require_exact(
        alias_graph.graph_attrs().at(alias_id),
        std::int64_t{18},
        "self-alias add must double the original value");
    resource.update(
        alias_graph.graph_attrs(),
        alias_id,
        alias_graph.graph_attrs(),
        alias_id,
        ResourceUpdateOperation::subtract,
        true);
    require_exact(
        alias_graph.graph_attrs().at(alias_id),
        std::int64_t{0},
        "self-alias subtract must become zero");

    set_resource_values(
        maps,
        std::int64_t{1},
        std::numeric_limits<std::int64_t>::max());
    expect_method_error(
        [&]() {
            static_cast<void>(resource.update(
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                ResourceUpdateOperation::add));
        },
        AttributeMethodErrorCode::numeric_range,
        AttributeMethodOperation::update_resource,
        "target addition overflow must be a typed numeric_range failure");
    require_exact(
        maps.physical_graph.graph_attrs().at(maps.physical_id),
        std::numeric_limits<std::int64_t>::max(),
        "overflow must leave the target unchanged");

    maps = resource_maps();
    maps.physical_graph.graph_attrs().set(maps.physical_id, std::string("bad"));
    expect_graph_error(
        [&]() {
            static_cast<void>(resource.update(
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                ResourceUpdateOperation::add));
        },
        GraphAttributeErrorCode::missing_resource_value,
        GraphAttributeOperation::update_resource,
        "missing operand lookup must precede target numeric conversion");

    maps = resource_maps();
    maps.virtual_graph.graph_attrs().set(maps.virtual_id, std::string("bad"));
    expect_graph_error(
        [&]() {
            static_cast<void>(resource.update(
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                ResourceUpdateOperation::add));
        },
        GraphAttributeErrorCode::missing_resource_value,
        GraphAttributeOperation::update_resource,
        "missing target lookup must precede operand numeric conversion");

    set_resource_values(maps, std::int64_t{2}, std::string("bad"));
    expect_graph_error(
        [&]() {
            static_cast<void>(resource.update(
                maps.physical_graph.graph_attrs(),
                maps.physical_id,
                maps.virtual_graph.graph_attrs(),
                maps.virtual_id,
                ResourceUpdateOperation::add));
        },
        GraphAttributeErrorCode::non_numeric_resource,
        GraphAttributeOperation::update_resource,
        "nonnumeric target must be a typed graph error");
}

struct BatchFixture {
    std::array<Graph, 6U> graphs;
    std::array<DiGraph, 6U> digraphs;
    std::vector<GraphAttributeBinding> bindings;
    std::vector<GraphAttributeMutableSlot> mutable_slots;
    std::vector<GraphAttributeConstSlot> const_slots;
};

BatchFixture make_batch_fixture(const GraphAttribute& attribute) {
    BatchFixture fixture;
    fixture.bindings.reserve(fixture.graphs.size() + fixture.digraphs.size());
    fixture.mutable_slots.reserve(fixture.graphs.size() + fixture.digraphs.size());
    fixture.const_slots.reserve(fixture.graphs.size() + fixture.digraphs.size());

    for (std::size_t index = 0U; index < fixture.graphs.size(); ++index) {
        for (std::size_t filler = 0U; filler < index; ++filler) {
            static_cast<void>(intern_attr(
                fixture.graphs[index],
                "graph-padding-" + std::to_string(filler)));
        }
        static_cast<void>(intern_attr(fixture.graphs[index], "bulk"));
        fixture.bindings.push_back(attribute.bind(fixture.graphs[index]));
        const GraphAttributeBinding binding = fixture.bindings.back();
        fixture.mutable_slots.push_back(
            {&fixture.graphs[index].graph_attrs(), binding});
        fixture.const_slots.push_back(
            {&fixture.graphs[index].graph_attrs(), binding});
    }
    for (std::size_t index = 0U; index < fixture.digraphs.size(); ++index) {
        for (std::size_t filler = 0U; filler < index + 1U; ++filler) {
            static_cast<void>(intern_attr(
                fixture.digraphs[index],
                "digraph-padding-" + std::to_string(filler)));
        }
        static_cast<void>(intern_attr(fixture.digraphs[index], "bulk"));
        fixture.bindings.push_back(attribute.bind(fixture.digraphs[index]));
        const GraphAttributeBinding binding = fixture.bindings.back();
        fixture.mutable_slots.push_back(
            {&fixture.digraphs[index].graph_attrs(), binding});
        fixture.const_slots.push_back(
            {&fixture.digraphs[index].graph_attrs(), binding});
    }
    return fixture;
}

std::vector<AttrValue> batch_values(std::size_t count, std::uint64_t salt) {
    const std::vector<AttrValue> lanes = lane_values(salt);
    std::vector<AttrValue> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        result.push_back(lanes[index % lanes.size()]);
    }
    return result;
}

void test_batch_surface() {
    GraphAttribute attribute(graph_spec("bulk", AttributeKind::status));

    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        BatchFixture fixture = make_batch_fixture(attribute);
        const std::vector<AttrValue> values = batch_values(
            fixture.mutable_slots.size(),
            static_cast<std::uint64_t>(workers));
        attribute.set_data_batch(fixture.mutable_slots, values, workers);
        require_exact_values(
            attribute.get_data_batch(fixture.const_slots, workers),
            values,
            "batch set/get must retain input order, lanes, bits, and local IDs");
    }

    require(
        attribute.get_data_batch({}, 8U).empty(),
        "empty batch get must return an empty vector");
    attribute.set_data_batch({}, {}, 8U);

    Graph graph;
    const AttrId id = intern_attr(graph, "bulk");
    const GraphAttributeBinding binding = attribute.bind(graph);
    graph.graph_attrs().set(id, std::int64_t{41});
    const std::vector<GraphAttributeMutableSlot> null_mutable{
        {&graph.graph_attrs(), binding},
        {nullptr, binding}};
    expect_graph_error(
        [&]() {
            attribute.set_data_batch(
                null_mutable,
                {std::int64_t{99}, std::int64_t{100}},
                8U);
        },
        GraphAttributeErrorCode::null_batch_slot,
        GraphAttributeOperation::set_data_batch,
        "batch set must reject every null pointer before worker launch");
    require_exact(
        graph.graph_attrs().at(id),
        std::int64_t{41},
        "null batch validation must occur before any write");

    const std::vector<GraphAttributeConstSlot> null_const{
        {&graph.graph_attrs(), binding},
        {nullptr, binding}};
    expect_graph_error(
        [&]() { static_cast<void>(attribute.get_data_batch(null_const, 8U)); },
        GraphAttributeErrorCode::null_batch_slot,
        GraphAttributeOperation::get_data_batch,
        "batch get must reject every null pointer before worker launch");

    expect_graph_error(
        [&]() {
            attribute.set_data_batch(
                {{&graph.graph_attrs(), binding}},
                {},
                8U);
        },
        GraphAttributeErrorCode::invalid_batch_shape,
        GraphAttributeOperation::set_data_batch,
        "batch set must reject shape mismatch before mutation");
    require_exact(
        graph.graph_attrs().at(id),
        std::int64_t{41},
        "shape failure must leave all graph slots unchanged");

    const GraphAttributeBinding invalid{
        std::numeric_limits<AttrId>::max()};
    expect_graph_error(
        [&]() {
            static_cast<void>(attribute.get_data_batch(
                {{&graph.graph_attrs(), invalid}},
                0U));
        },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::get_data_batch,
        "batch get must reject a binding outside the map registry");
    expect_graph_error(
        [&]() {
            attribute.set_data_batch(
                {{&graph.graph_attrs(), invalid}},
                {std::int64_t{1}},
                0U);
        },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::set_data_batch,
        "batch set must reject a binding outside the map registry");

    Graph missing;
    static_cast<void>(intern_attr(missing, "bulk"));
    const GraphAttributeBinding missing_binding = attribute.bind(missing);
    expect_graph_error(
        [&]() {
            static_cast<void>(attribute.get_data_batch(
                {{&missing.graph_attrs(), missing_binding}},
                0U));
        },
        GraphAttributeErrorCode::missing_attribute,
        GraphAttributeOperation::get_data_batch,
        "batch get must preserve missing-value errors");
}

void test_duplicate_map_fallback() {
    GraphAttribute attribute(graph_spec("bulk", AttributeKind::status));
    Graph graph;
    const AttrId id = intern_attr(graph, "bulk");
    const GraphAttributeBinding binding = attribute.bind(graph);

    constexpr std::size_t duplicate_count = 257U;
    std::vector<GraphAttributeMutableSlot> slots(
        duplicate_count,
        GraphAttributeMutableSlot{&graph.graph_attrs(), binding});
    std::vector<AttrValue> values;
    values.reserve(duplicate_count);
    for (std::size_t index = 0U; index < duplicate_count; ++index) {
        values.emplace_back(static_cast<std::int64_t>(index));
    }

    for (std::size_t round = 0U; round < 8U; ++round) {
        graph.graph_attrs().set(id, std::int64_t{-1});
        attribute.set_data_batch(slots, values, 8U);
        require_exact(
            graph.graph_attrs().at(id),
            static_cast<std::int64_t>(duplicate_count - 1U),
            "duplicate mutable maps must use canonical sequential last-write order");
    }

    const AttrId other_id = intern_attr(graph, "other");
    GraphAttribute other_attribute(
        graph_spec("other", AttributeKind::status));
    const GraphAttributeBinding other_binding =
        other_attribute.bind(graph);
    const std::vector<GraphAttributeMutableSlot> interleaved{
        {&graph.graph_attrs(), binding},
        {&graph.graph_attrs(), other_binding},
        {&graph.graph_attrs(), binding}};
    graph.graph_attrs().set(id, std::int64_t{41});
    graph.graph_attrs().set(other_id, std::int64_t{43});
    expect_graph_error(
        [&]() {
            attribute.set_data_batch(
                interleaved,
                {std::int64_t{1}, std::int64_t{7}, std::int64_t{3}},
                8U);
        },
        GraphAttributeErrorCode::invalid_binding,
        GraphAttributeOperation::set_data_batch,
        "one fixed GraphAttribute must reject another field's same-map ID");
    require_exact(
        graph.graph_attrs().at(id),
        std::int64_t{41},
        "batch binding validation must finish before the first write");
    require_exact(
        graph.graph_attrs().at(other_id),
        std::int64_t{43},
        "invalid same-map binding must leave the unrelated field unchanged");
}

void test_concurrent_independent_callers() {
    GraphAttribute attribute(graph_spec("bulk", AttributeKind::status));
    const std::array<std::size_t, 4U> worker_counts{0U, 1U, 2U, 8U};
    std::vector<std::future<void>> callers;
    callers.reserve(worker_counts.size());

    for (std::size_t caller = 0U; caller < worker_counts.size(); ++caller) {
        callers.push_back(std::async(
            std::launch::async,
            [caller, workers = worker_counts[caller], &attribute]() {
                for (std::size_t round = 0U; round < 8U; ++round) {
                    constexpr std::size_t graph_count = 32U;
                    std::array<Graph, graph_count> graphs;
                    std::vector<GraphAttributeMutableSlot> mutable_slots;
                    std::vector<GraphAttributeConstSlot> const_slots;
                    std::vector<AttrValue> values;
                    mutable_slots.reserve(graph_count);
                    const_slots.reserve(graph_count);
                    values.reserve(graph_count);
                    for (std::size_t index = 0U; index < graph_count; ++index) {
                        const std::size_t filler_count = (index + caller) % 5U;
                        for (std::size_t filler = 0U; filler < filler_count; ++filler) {
                            static_cast<void>(intern_attr(
                                graphs[index],
                                "padding-" + std::to_string(filler)));
                        }
                        const AttrId id = intern_attr(graphs[index], "bulk");
                        const GraphAttributeBinding binding = attribute.bind(graphs[index]);
                        require(
                            binding.value_id == id,
                            "concurrent caller must retain graph-local IDs");
                        mutable_slots.push_back({&graphs[index].graph_attrs(), binding});
                        const_slots.push_back({&graphs[index].graph_attrs(), binding});
                        const std::size_t encoded =
                            caller * 100000U + round * 1000U + index;
                        values.emplace_back(static_cast<std::int64_t>(encoded));
                    }
                    attribute.set_data_batch(mutable_slots, values, workers);
                    require_exact_values(
                        attribute.get_data_batch(const_slots, workers),
                        values,
                        "concurrent independent batch callers must remain exact");
                }
            }));
    }

    for (std::future<void>& caller : callers) {
        caller.get();
    }
}

}  // namespace

int main() {
    try {
        test_scalar_surface<Graph>("Graph");
        test_scalar_surface<DiGraph>("DiGraph");
        test_graph_local_bindings();
        test_fixed_specs();
        test_extrema_delegation<Graph>();
        test_extrema_delegation<DiGraph>();
        test_resource_checks();
        test_resource_updates();
        test_batch_surface();
        test_duplicate_map_fallback();
        test_concurrent_independent_callers();
        std::cout << "graph_attribute_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph_attribute_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
