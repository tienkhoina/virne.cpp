#include "counter.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace core = virne::core;
namespace network = virne::network;

using attribute::AttributeFactorySpec;
using attribute::AttributeKind;
using attribute::AttributeOwner;
using attribute::CheckingLevel;
using attribute::ConstraintRestriction;
using core::Counter;
using core::CounterErrorCode;
using core::CounterException;
using core::CounterNumber;
using core::CounterOperation;
using core::CounterOptions;
using core::CounterRecord;
using core::CounterRecords;
using core::CounterResourceId;
using core::CounterSelection;
using core::CounterSummary;
using core::PreparedCounter;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

void expect_near(
    double actual,
    double expected,
    double tolerance,
    std::string_view message)
{
    expect(std::abs(actual - expected) <= tolerance, message);
}

template <typename Callable>
CounterException expect_counter_error(
    Callable&& callable,
    CounterErrorCode code,
    CounterOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const CounterException& error)
    {
        expect(error.code() == code, "counter error code mismatch");
        expect(error.operation() == operation,
               "counter operation mismatch");
        expect(!std::string_view(error.what()).empty(),
               "counter diagnostic is empty");
        return error;
    }
    throw std::runtime_error("expected CounterException");
}

std::int64_t as_integer(
    const CounterNumber& value,
    std::string_view message)
{
    const auto* integer = std::get_if<std::int64_t>(&value);
    expect(integer != nullptr, message);
    return *integer;
}

double as_double(
    const CounterNumber& value,
    std::string_view message)
{
    const auto* floating = std::get_if<double>(&value);
    expect(floating != nullptr, message);
    return *floating;
}

AttributeFactorySpec resource_spec(
    std::string name,
    AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::resource;
    result.restriction = ConstraintRestriction::hard;
    result.checking_level = owner == AttributeOwner::node
        ? CheckingLevel::node
        : CheckingLevel::link;
    return result;
}

AttributeFactorySpec status_spec(
    std::string name,
    AttributeOwner owner)
{
    AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = AttributeKind::status;
    return result;
}

network::NodeAttributeDataUpdate dense_node_update(
    CounterResourceId id,
    std::vector<AttrValue> values)
{
    network::NodeAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::dense;
    result.dense_values = std::move(values);
    return result;
}

network::LinkAttributeDataUpdate sparse_link_update(
    CounterResourceId id,
    std::vector<attribute::LinkAttributeAssignment> values)
{
    network::LinkAttributeDataUpdate result;
    result.registry_id = id;
    result.layout = network::AttributeDataLayout::sparse;
    result.sparse_values = std::move(values);
    return result;
}

template <typename Network>
network::NodeNetworkAttributeBinding require_node_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_node_attribute(name);
    expect(binding.has_value(), "missing fixture node binding");
    return *binding;
}

template <typename Network>
network::LinkNetworkAttributeBinding require_link_binding(
    Network& value,
    std::string_view name)
{
    const auto binding = value.bind_link_attribute(name);
    expect(binding.has_value(), "missing fixture link binding");
    return *binding;
}

network::VirtualNetwork make_main_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        status_spec("node_status", AttributeOwner::node),
        resource_spec("cpu", AttributeOwner::node),
        resource_spec("memory", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("aux", AttributeOwner::link),
        status_spec("link_status", AttributeOwner::link),
        resource_spec("bw", AttributeOwner::link)};
    return network::VirtualNetwork(std::move(construction));
}

core::Solution make_solution()
{
    core::SolutionMetadata metadata;
    metadata.v_net_id = 41;
    metadata.v_net_lifetime = 4.0;
    metadata.v_net_arrival_time = 1.0;
    metadata.v_net_num_nodes = 3U;
    metadata.v_net_num_edges = 2U;
    return core::Solution(metadata);
}

struct MainFixture
{
    network::VirtualNetwork network = make_main_network();
    network::NodeNetworkAttributeBinding cpu;
    network::NodeNetworkAttributeBinding memory;
    network::NodeNetworkAttributeBinding node_status;
    network::LinkNetworkAttributeBinding aux;
    network::LinkNetworkAttributeBinding bw;
    network::LinkNetworkAttributeBinding link_status;

    explicit MainFixture(bool with_lifetime = true)
        : cpu(require_node_binding(network, "cpu")),
          memory(require_node_binding(network, "memory")),
          node_status(require_node_binding(network, "node_status")),
          aux(require_link_binding(network, "aux")),
          bw(require_link_binding(network, "bw")),
          link_status(require_link_binding(network, "link_status"))
    {
        network.set_node_attrs_data({
            dense_node_update(
                cpu.registry_id,
                {std::int64_t{10}, std::int64_t{20}, std::int64_t{30}}),
            dense_node_update(memory.registry_id, {1.5, 2.5, 3.5})});
        network.set_link_attrs_data({
            sparse_link_update(
                aux.registry_id,
                {{0U, 1U, 1.5}, {1U, 2U, 2.5}}),
            sparse_link_update(
                bw.registry_id,
                {{0U, 1U, std::int64_t{5}},
                 {1U, 2U, std::int64_t{7}}})});
        if (with_lifetime)
        {
            network.set_lifetime(4.0);
        }

        expect(memory.registry_id == bw.registry_id,
               "fixture must reuse one numeric ID across registry domains");
        expect(cpu.value_id != cpu.registry_id &&
                   bw.value_id != bw.registry_id,
               "fixture must exercise graph-local IDs");
    }

    PreparedCounter prepare() const
    {
        return Counter().prepare(network);
    }

    PreparedCounter prepare(CounterSelection selection) const
    {
        return Counter(std::move(selection)).prepare(network);
    }

    core::Solution mapped_solution(bool complete_info = true) const
    {
        core::Solution solution = make_solution();
        solution.node_slots.insert_or_assign(0, 20);
        solution.node_slots.insert_or_assign(2, 22);
        solution.link_paths.insert_or_assign(
            {0, 1}, {{0, 1}, {1, 2}});
        solution.link_paths.insert_or_assign({1, 2}, {});

        core::SolutionAttributeValues first;
        first.set(bw.registry_id, std::int64_t{5});
        first.set(aux.registry_id, 1.5);
        solution.link_paths_info.insert_or_assign(
            {{0, 1}, {0, 1}}, first);
        if (complete_info)
        {
            core::SolutionAttributeValues second;
            second.set(bw.registry_id, std::int64_t{5});
            second.set(aux.registry_id, 1.5);
            solution.link_paths_info.insert_or_assign(
                {{0, 1}, {1, 2}}, second);
        }
        return solution;
    }
};

network::VirtualNetwork make_numeric_network()
{
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}});
    construction.config.node_attribute_specs = {
        resource_spec("node_resource", AttributeOwner::node)};
    construction.config.link_attribute_specs = {
        resource_spec("link_resource", AttributeOwner::link)};
    return network::VirtualNetwork(std::move(construction));
}

struct NumericFixture
{
    network::VirtualNetwork network = make_numeric_network();
    network::NodeNetworkAttributeBinding node;
    network::LinkNetworkAttributeBinding link;

    NumericFixture(
        std::vector<AttrValue> node_values,
        std::vector<AttrValue> link_values)
        : node(require_node_binding(network, "node_resource")),
          link(require_link_binding(network, "link_resource"))
    {
        network.set_node_attrs_data({
            dense_node_update(node.registry_id, std::move(node_values))});
        network.set_link_attrs_data({
            sparse_link_update(
                link.registry_id,
                {{0U, 1U, std::move(link_values.at(0U))},
                 {1U, 2U, std::move(link_values.at(1U))}})});
    }

    PreparedCounter prepare() const
    {
        return Counter().prepare(network);
    }
};

void test_selection_registry_and_resource_sums()
{
    MainFixture fixture;
    const Counter default_counter;
    expect(!default_counter.selection().node_resources.has_value() &&
               !default_counter.selection().link_resources.has_value(),
           "default selection did not retain nullopt");
    const PreparedCounter prepared = default_counter.prepare(fixture.network);
    expect(as_double(
               prepared.calculate_sum_node_resource(),
               "default node sum lane") == 67.5,
           "default node resource sum mismatch");
    expect(as_double(
               prepared.calculate_sum_link_resource(),
               "default link sum lane") == 16.0,
           "default link resource sum mismatch");
    expect(as_double(
               prepared.calculate_sum_network_resource(),
               "default network sum lane") == 83.5,
           "default network resource sum mismatch");
    expect(as_double(
               prepared.calculate_sum_network_resource(false, true),
               "link-only network sum lane") == 16.0,
           "link-only resource flag mismatch");
    expect(as_double(
               prepared.calculate_sum_network_resource(true, false),
               "node-only network sum lane") == 67.5,
           "node-only resource flag mismatch");
    expect(as_integer(
               prepared.calculate_sum_network_resource(false, false),
               "disabled network sum lane") == 0,
           "both-disabled resource sum mismatch");

    CounterSelection duplicates;
    duplicates.node_resources = std::vector<CounterResourceId>{
        fixture.cpu.registry_id, fixture.cpu.registry_id};
    duplicates.link_resources = std::vector<CounterResourceId>{
        fixture.bw.registry_id, fixture.bw.registry_id};
    const PreparedCounter duplicate_counter = fixture.prepare(duplicates);
    expect(as_integer(
               duplicate_counter.calculate_sum_node_resource(),
               "duplicate node sum lane") == 120,
           "duplicate node selection was collapsed");
    expect(as_integer(
               duplicate_counter.calculate_sum_link_resource(),
               "duplicate link sum lane") == 24,
           "duplicate link selection was collapsed");

    CounterSelection explicit_selection;
    explicit_selection.node_resources =
        std::vector<CounterResourceId>{fixture.memory.registry_id};
    explicit_selection.link_resources =
        std::vector<CounterResourceId>{fixture.bw.registry_id};
    const PreparedCounter explicit_counter =
        fixture.prepare(explicit_selection);
    expect(as_double(
               explicit_counter.calculate_sum_node_resource(),
               "explicit node sum lane") == 7.5,
           "explicit node registry domain mismatch");
    expect(as_integer(
               explicit_counter.calculate_sum_link_resource(),
               "explicit link sum lane") == 12,
           "explicit link registry domain mismatch");

    CounterSelection empty;
    empty.node_resources = std::vector<CounterResourceId>{};
    empty.link_resources = std::vector<CounterResourceId>{};
    const PreparedCounter empty_counter = fixture.prepare(empty);
    expect_counter_error(
        [&]
        {
            static_cast<void>(empty_counter.calculate_sum_node_resource());
        },
        CounterErrorCode::empty_node_resource_selection,
        CounterOperation::sum_node_resources);
    expect_counter_error(
        [&]
        {
            static_cast<void>(empty_counter.calculate_sum_link_resource());
        },
        CounterErrorCode::empty_link_resource_selection,
        CounterOperation::sum_link_resources);

    CounterSelection wrong_node;
    wrong_node.node_resources = std::vector<CounterResourceId>{
        fixture.node_status.registry_id};
    expect_counter_error(
        [&]
        {
            static_cast<void>(fixture.prepare(wrong_node));
        },
        CounterErrorCode::invalid_node_resource_selection,
        CounterOperation::prepare);

    CounterSelection wrong_link;
    wrong_link.link_resources = std::vector<CounterResourceId>{
        fixture.link_status.registry_id};
    expect_counter_error(
        [&]
        {
            static_cast<void>(fixture.prepare(wrong_link));
        },
        CounterErrorCode::invalid_link_resource_selection,
        CounterOperation::prepare);
}

void test_numeric_lanes_wrap_errors_and_workers()
{
    NumericFixture boolean(
        {true, false, true}, {true, true});
    const PreparedCounter bool_counter = boolean.prepare();
    expect(as_integer(
               bool_counter.calculate_sum_node_resource(),
               "bool node sum lane") == 2,
           "bool node sum mismatch");
    expect(as_integer(
               bool_counter.calculate_sum_network_resource(),
               "bool network sum lane") == 4,
           "bool network promotion mismatch");

    NumericFixture wrapping(
        {std::numeric_limits<std::int64_t>::max(),
         std::int64_t{1}, std::int64_t{0}},
        {std::int64_t{0}, std::int64_t{0}});
    expect(as_integer(
               wrapping.prepare().calculate_sum_node_resource(),
               "wrapped node sum lane") ==
               std::numeric_limits<std::int64_t>::min(),
           "int64 NumPy wrap mismatch");

    const double subnormal = std::numeric_limits<double>::denorm_min();
    NumericFixture mixed(
        {std::int64_t{1}, 0.5, true}, {-0.0, subnormal});
    const PreparedCounter mixed_counter = mixed.prepare();
    expect(as_double(
               mixed_counter.calculate_sum_node_resource(),
               "mixed node sum lane") == 2.5,
           "mixed int/bool/double promotion mismatch");
    const double tiny_link = as_double(
        mixed_counter.calculate_sum_link_resource(),
        "subnormal link sum lane");
    expect(tiny_link == subnormal && !std::signbit(tiny_link),
           "signed-zero/subnormal reduction mismatch");

    NumericFixture non_finite(
        {std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(), 0.0},
        {0.0, 0.0});
    expect(std::isnan(as_double(
               non_finite.prepare().calculate_sum_node_resource(),
               "non-finite node sum lane")),
           "infinity cancellation did not produce NaN");

    MainFixture worker_fixture;
    const PreparedCounter worker_counter = worker_fixture.prepare();
    const CounterNumber node_reference =
        worker_counter.calculate_sum_node_resource();
    const CounterNumber link_reference =
        worker_counter.calculate_sum_link_resource();
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(worker_counter.calculate_sum_node_resource({workers}) ==
                   node_reference,
               "node worker result/type drift");
        expect(worker_counter.calculate_sum_link_resource({workers}) ==
                   link_reference,
               "link worker result/type drift");
    }

    NumericFixture missing(
        {std::int64_t{1}, std::int64_t{2}, std::int64_t{3}},
        {std::int64_t{1}, std::int64_t{1}});
    missing.network.graph().node_attrs(0U).erase(missing.node.value_id);
    const PreparedCounter missing_counter = missing.prepare();
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        const CounterException error = expect_counter_error(
            [&]
            {
                static_cast<void>(
                    missing_counter.calculate_sum_node_resource({workers}));
            },
            CounterErrorCode::missing_node_resource_value,
            CounterOperation::sum_node_resources);
        expect(error.resource_id() ==
                   std::optional<CounterResourceId>{
                       missing.node.registry_id},
               "missing node resource context mismatch");
    }

    NumericFixture nonnumeric(
        {std::int64_t{1}, std::int64_t{2}, std::int64_t{3}},
        {std::int64_t{1}, std::int64_t{1}});
    nonnumeric.network.graph().node_attrs(1U).set(
        nonnumeric.node.value_id, std::string{"bad"});
    const PreparedCounter nonnumeric_counter = nonnumeric.prepare();
    expect_counter_error(
        [&]
        {
            static_cast<void>(
                nonnumeric_counter.calculate_sum_node_resource({8U}));
        },
        CounterErrorCode::non_numeric_resource_value,
        CounterOperation::sum_node_resources);
}

struct PartialMetrics
{
    double node_revenue = 0.0;
    double link_revenue = 0.0;
    double revenue = 0.0;
    double link_cost = 0.0;
    double path_cost = 0.0;
    double node_cost = 0.0;
    double cost = 0.0;
    double ratio = 0.0;

    friend bool operator==(
        const PartialMetrics& left,
        const PartialMetrics& right)
    {
        return left.node_revenue == right.node_revenue &&
            left.link_revenue == right.link_revenue &&
            left.revenue == right.revenue &&
            left.link_cost == right.link_cost &&
            left.path_cost == right.path_cost &&
            left.node_cost == right.node_cost &&
            left.cost == right.cost && left.ratio == right.ratio;
    }
};

PartialMetrics partial_metrics(const core::Solution& solution)
{
    return PartialMetrics{
        solution.v_net_node_revenue,
        solution.v_net_link_revenue,
        solution.v_net_revenue,
        solution.v_net_link_cost,
        solution.v_net_path_cost,
        solution.v_net_node_cost,
        solution.v_net_cost,
        solution.v_net_r2c_ratio};
}

void set_partial_sentinels(core::Solution& solution)
{
    solution.v_net_node_revenue = -1.0;
    solution.v_net_link_revenue = -2.0;
    solution.v_net_revenue = -3.0;
    solution.v_net_link_cost = -4.0;
    solution.v_net_path_cost = -5.0;
    solution.v_net_node_cost = -6.0;
    solution.v_net_cost = -7.0;
    solution.v_net_r2c_ratio = -8.0;
}

void test_partial_count_helpers_and_atomic_errors()
{
    MainFixture fixture;
    const PreparedCounter counter = fixture.prepare();
    core::Solution solution = fixture.mapped_solution();
    counter.count_partial_solution(solution, {2U});
    expect(solution.v_net_node_revenue == 22.5,
           "partial normalized node revenue mismatch");
    expect(solution.v_net_link_revenue == 16.0,
           "partial link revenue mismatch");
    expect(solution.v_net_revenue == 61.0,
           "partial total normalization bug mismatch");
    expect(solution.v_net_link_cost == 13.0 &&
               solution.v_net_path_cost == -3.0,
           "partial routed link/path cost mismatch");
    expect(solution.v_net_node_cost == 22.5 &&
               solution.v_net_cost == 35.5,
           "partial total cost mismatch");
    expect_near(solution.v_net_r2c_ratio, 61.0 / 35.5, 1.0e-15,
                "partial r2c mismatch");

    const auto first_info_id = solution.link_paths_info.find_id(
        core::LinkPathInfoKey{{0, 1}, {0, 1}});
    expect(first_info_id.has_value(), "missing rerun info fixture");
    solution.link_paths_info.at(*first_info_id).set(
        fixture.bw.registry_id, std::int64_t{10});
    counter.count_partial_solution(solution, {8U});
    expect(solution.v_net_link_cost == 18.0 &&
               solution.v_net_path_cost == 2.0 &&
               solution.v_net_cost == 40.5,
           "partial rerun did not overwrite from current info");

    expect(as_double(
               counter.calculate_v_net_link_cost(solution),
               "manual link cost lane") == 18.0,
           "manual link cost helper mismatch");
    expect(as_double(
               counter.calculate_v_net_cost(solution),
               "virtual cost lane") == 85.5,
           "virtual cost helper mismatch");
    expect(as_double(
               counter.calculate_v_net_revenue(),
               "virtual revenue lane") == 83.5,
           "virtual revenue helper mismatch");

    core::Solution missing_info = fixture.mapped_solution(false);
    set_partial_sentinels(missing_info);
    const PartialMetrics before_missing = partial_metrics(missing_info);
    const CounterException missing_error = expect_counter_error(
        [&]
        {
            counter.count_partial_solution(missing_info);
        },
        CounterErrorCode::missing_route_info,
        CounterOperation::count_partial_solution);
    expect(missing_error.virtual_link() ==
               std::optional<core::SolutionLink>{{0, 1}} &&
               missing_error.physical_link() ==
                   std::optional<core::SolutionLink>{{1, 2}},
           "missing route info context mismatch");
    expect(partial_metrics(missing_info) == before_missing,
           "partial scan error mutated metrics");

    core::Solution invalid_node = make_solution();
    invalid_node.node_slots.insert_or_assign(-1, 0);
    set_partial_sentinels(invalid_node);
    const PartialMetrics invalid_node_before = partial_metrics(invalid_node);
    expect_counter_error(
        [&]
        {
            counter.count_partial_solution(invalid_node);
        },
        CounterErrorCode::invalid_solution_node,
        CounterOperation::count_partial_solution);
    expect(partial_metrics(invalid_node) == invalid_node_before,
           "invalid node changed partial metrics");

    core::Solution invalid_link = make_solution();
    invalid_link.link_paths.insert_or_assign({-1, 1}, {});
    set_partial_sentinels(invalid_link);
    const PartialMetrics invalid_link_before = partial_metrics(invalid_link);
    expect_counter_error(
        [&]
        {
            counter.count_partial_solution(invalid_link);
        },
        CounterErrorCode::invalid_solution_link,
        CounterOperation::count_partial_solution);
    expect(partial_metrics(invalid_link) == invalid_link_before,
           "invalid link changed partial metrics");

    CounterSelection zero_selection;
    zero_selection.node_resources = std::vector<CounterResourceId>{};
    const PreparedCounter zero_counter = fixture.prepare(zero_selection);
    core::Solution zero_solution = fixture.mapped_solution();
    set_partial_sentinels(zero_solution);
    const PartialMetrics zero_before = partial_metrics(zero_solution);
    expect_counter_error(
        [&]
        {
            zero_counter.count_partial_solution(zero_solution);
        },
        CounterErrorCode::empty_node_resource_selection,
        CounterOperation::count_partial_solution);
    expect(partial_metrics(zero_solution) == zero_before,
           "zero-node-resource division wrote partial metrics");
}

void test_complete_count_success_failure_and_partial_state()
{
    MainFixture fixture;
    const PreparedCounter counter = fixture.prepare();
    core::Solution success = fixture.mapped_solution();
    success.result = true;
    success.v_net_demand = 9.0;
    success.place_result = false;
    success.route_result = false;
    success.early_rejection = true;
    counter.count_solution(success, {2U});
    expect(success.num_placed_nodes == std::optional<std::size_t>{2U} &&
               success.num_routed_links == std::optional<std::size_t>{2U},
           "complete mapping counters mismatch");
    if (!(success.v_net_node_demand == 33.75 &&
          success.v_net_link_demand == 16.0 &&
          success.v_net_demand == 42.75))
    {
        throw std::runtime_error(
            "complete old-demand bug mismatch: node=" +
            std::to_string(success.v_net_node_demand) +
            ", link=" + std::to_string(success.v_net_link_demand) +
            ", total=" + std::to_string(success.v_net_demand));
    }
    expect(success.place_result && success.route_result &&
               !success.early_rejection,
           "complete success flag mutation mismatch");
    expect(success.v_net_node_revenue == 33.75 &&
               success.v_net_link_revenue == 16.0 &&
               success.v_net_revenue == 49.75,
           "complete success revenue mismatch");
    expect(success.v_net_node_cost == 33.75 &&
               success.v_net_link_cost == 13.0 &&
               success.v_net_path_cost == -3.0 &&
               success.v_net_cost == 46.75,
           "complete success cost mismatch");
    expect_near(success.v_net_r2c_ratio, 49.75 / 46.75, 1.0e-15,
                "complete success ratio mismatch");
    expect(success.v_net_time_revenue == 199.0 &&
               success.v_net_time_cost == 187.0,
           "complete lifetime totals mismatch");
    expect_near(success.v_net_time_rc_ratio,
                (49.75 / 46.75) * 4.0,
                1.0e-15,
                "complete lifetime ratio mismatch");

    counter.count_solution(success, {8U});
    expect(success.v_net_demand == 76.5,
           "repeated complete count did not reuse old demand");

    core::Solution failure = fixture.mapped_solution();
    failure.result = false;
    failure.v_net_demand = 4.0;
    failure.place_result = false;
    failure.route_result = false;
    failure.early_rejection = true;
    failure.v_net_node_cost = 77.0;
    failure.v_net_link_cost = 88.0;
    counter.count_solution(failure);
    expect(failure.v_net_node_demand == 33.75 &&
               failure.v_net_link_demand == 16.0 &&
               failure.v_net_demand == 37.75,
           "complete failure demand mutation mismatch");
    expect(!failure.place_result && !failure.route_result &&
               failure.early_rejection,
           "complete failure changed stale flags");
    expect(failure.v_net_node_revenue == 0.0 &&
               failure.v_net_link_revenue == 0.0 &&
               failure.v_net_revenue == 0.0 &&
               failure.v_net_path_cost == 0.0 &&
               failure.v_net_cost == 0.0 &&
               failure.v_net_r2c_ratio == 0.0,
           "complete failure zero fields mismatch");
    expect(failure.v_net_node_cost == 77.0 &&
               failure.v_net_link_cost == 88.0,
           "complete failure cleared stale costs");
    expect(failure.v_net_time_revenue == 0.0 &&
               failure.v_net_time_cost == 0.0 &&
               failure.v_net_time_rc_ratio == 0.0,
           "complete failure lifetime zeros mismatch");

    core::Solution info_error = fixture.mapped_solution(false);
    info_error.result = true;
    info_error.v_net_link_cost = -10.0;
    info_error.v_net_path_cost = -11.0;
    info_error.v_net_cost = -12.0;
    expect_counter_error(
        [&]
        {
            counter.count_solution(info_error);
        },
        CounterErrorCode::missing_route_info,
        CounterOperation::count_solution);
    expect(info_error.num_placed_nodes ==
               std::optional<std::size_t>{2U} &&
               info_error.num_routed_links ==
                   std::optional<std::size_t>{2U} &&
               info_error.place_result && info_error.route_result &&
               !info_error.early_rejection,
           "complete link-info error lost earlier direct writes");
    expect(info_error.v_net_node_revenue == 33.75 &&
               info_error.v_net_link_revenue == 16.0 &&
               info_error.v_net_node_cost == 33.75,
           "complete link-info error lost earlier revenue/cost writes");
    expect(info_error.v_net_link_cost == -10.0 &&
               info_error.v_net_path_cost == -11.0 &&
               info_error.v_net_cost == -12.0,
           "complete link-info error wrote later cost fields");

    MainFixture no_lifetime(false);
    const PreparedCounter no_lifetime_counter = no_lifetime.prepare();
    core::Solution late_error = no_lifetime.mapped_solution();
    late_error.result = true;
    late_error.v_net_time_revenue = -1.0;
    late_error.v_net_time_cost = -2.0;
    late_error.v_net_time_rc_ratio = -3.0;
    expect_counter_error(
        [&]
        {
            no_lifetime_counter.count_solution(late_error);
        },
        CounterErrorCode::missing_virtual_lifetime,
        CounterOperation::count_solution);
    expect(late_error.v_net_revenue == 49.75 &&
               late_error.v_net_cost == 46.75,
           "late lifetime error lost preceding totals");
    expect(late_error.v_net_time_revenue == -1.0 &&
               late_error.v_net_time_cost == -2.0 &&
               late_error.v_net_time_rc_ratio == -3.0,
           "late lifetime error wrote time fields");

    NumericFixture zero_fixture(
        {std::int64_t{0}, std::int64_t{0}, std::int64_t{0}},
        {std::int64_t{0}, std::int64_t{0}});
    zero_fixture.network.set_lifetime(4.0);
    const PreparedCounter zero_counter = zero_fixture.prepare();
    core::Solution zero = make_solution();
    zero.result = true;
    zero_counter.count_solution(zero);
    expect(zero.v_net_cost == 0.0 && zero.v_net_revenue == 0.0 &&
               zero.v_net_r2c_ratio == 0.0,
           "exact zero-cost ratio branch mismatch");

    const network::BaseNetwork& base_network = fixture.network;
    const PreparedCounter base_counter = Counter().prepare(base_network);
    core::Solution base_partial = fixture.mapped_solution();
    expect_counter_error(
        [&]
        {
            base_counter.count_partial_solution(base_partial);
        },
        CounterErrorCode::virtual_network_required,
        CounterOperation::count_partial_solution);
    core::Solution base_complete = fixture.mapped_solution();
    expect_counter_error(
        [&]
        {
            base_counter.count_solution(base_complete);
        },
        CounterErrorCode::virtual_network_required,
        CounterOperation::count_solution);
}

CounterRecords ordinary_records()
{
    CounterRecords result;
    result.reward_column_present = true;
    result.rows.resize(4U);

    CounterRecord& first = result.rows[0U];
    first.success_count = 0;
    first.virtual_network_count = 1;
    first.event_type = network::VirtualEventType::arrival;
    first.v_net_r2c_ratio = 2.0;
    first.total_time_revenue = 10.0;
    first.total_time_cost = 5.0;
    first.virtual_network_arrival_time = 0.0;
    first.early_rejection = true;
    first.place_result = false;
    first.route_result = true;
    first.total_cost = 40.0;
    first.total_revenue = 50.0;
    first.physical_available_resource = 80.0;
    first.physical_node_available_resource = 40.0;
    first.physical_link_available_resource = 40.0;
    first.inservice_count = 1;
    first.hard_constraint_violation = 1.0;
    first.max_single_step_hard_constraint_violation = 2.0;
    first.reward = 1.0;

    CounterRecord& second = result.rows[1U];
    second.success_count = 1;
    second.virtual_network_count = 2;
    second.event_type = network::VirtualEventType::leave;
    second.v_net_r2c_ratio = 99.0;
    second.total_time_revenue = 20.0;
    second.total_time_cost = 8.0;
    second.virtual_network_arrival_time = 2.0;
    second.total_cost = 70.0;
    second.total_revenue = 90.0;
    second.physical_available_resource = 60.0;
    second.physical_node_available_resource = 35.0;
    second.physical_link_available_resource = 25.0;
    second.inservice_count = 3;
    second.hard_constraint_violation = 0.0;
    second.max_single_step_hard_constraint_violation = 3.0;
    second.reward = 100.0;

    CounterRecord& third = result.rows[2U];
    third.success_count = 1;
    third.virtual_network_count = 3;
    third.event_type = network::VirtualEventType::arrival;
    third.v_net_r2c_ratio = 4.0;
    third.total_time_revenue = 30.0;
    third.total_time_cost = 10.0;
    third.virtual_network_arrival_time = 4.0;
    third.early_rejection = false;
    third.place_result = true;
    third.route_result = false;
    third.total_cost = 100.0;
    third.total_revenue = 150.0;
    third.physical_available_resource = 50.0;
    third.physical_node_available_resource = 30.0;
    third.physical_link_available_resource = 20.0;
    third.inservice_count = 2;
    third.hard_constraint_violation = 2.0;
    third.max_single_step_hard_constraint_violation = 4.0;
    third.reward = 3.0;

    CounterRecord& last = result.rows[3U];
    last.success_count = 2;
    last.virtual_network_count = 4;
    last.event_type = network::VirtualEventType::leave;
    last.v_net_r2c_ratio = 9.0;
    last.total_time_revenue = 50.0;
    last.total_time_cost = 20.0;
    last.virtual_network_arrival_time = 10.0;
    last.total_cost = 200.0;
    last.total_revenue = 300.0;
    last.physical_available_resource = 70.0;
    last.physical_node_available_resource = 20.0;
    last.physical_link_available_resource = 10.0;
    last.inservice_count = 5;
    last.hard_constraint_violation = 3.0;
    last.max_single_step_hard_constraint_violation = 5.0;
    last.reward = 9.0;
    return result;
}

void test_summary_ordinary_empty_nan_and_reward()
{
    const CounterSummary summary = core::summary_records(ordinary_records());
    expect(summary.acceptance_rate == 0.5 &&
               summary.average_r2c_ratio == 3.0 &&
               summary.long_term_time_r2c_ratio == 2.5 &&
               summary.long_term_average_time_revenue == 5.0,
           "ordinary summary ratios mismatch");
    expect(summary.success_count == 2 &&
               summary.early_rejection_count == 1U &&
               summary.place_failure_count == 1U &&
               summary.route_failure_count == 1U,
           "ordinary arrival filters/counts mismatch");
    expect(summary.total_cost == 200.0 &&
               summary.total_revenue == 300.0 &&
               summary.total_time_revenue == 50.0 &&
               summary.total_time_cost == 20.0 &&
               summary.long_term_r2c_ratio == 1.5 &&
               summary.total_simulation_time == 10.0 &&
               summary.long_term_average_revenue == 30.0 &&
               summary.long_term_average_cost == 20.0,
           "ordinary final-positional summary mismatch");
    expect(summary.minimum_physical_available_resource == 50.0 &&
               summary.minimum_physical_node_available_resource == 20.0 &&
               summary.minimum_physical_link_available_resource == 10.0 &&
               summary.maximum_inservice_count == 5,
           "ordinary all-row extrema mismatch");
    expect(summary.total_violation == 6.0 &&
               summary.total_max_single_step_violation == 14.0 &&
               summary.average_reward == 2.0,
           "ordinary violation/reward summary mismatch");

    expect_counter_error(
        []
        {
            static_cast<void>(core::summary_records(CounterRecords{}));
        },
        CounterErrorCode::empty_records,
        CounterOperation::summarize_records);

    CounterRecords absent_reward = ordinary_records();
    absent_reward.reward_column_present = false;
    expect(core::summary_records(absent_reward).average_reward == 0.0,
           "absent reward column did not produce literal zero");

    CounterRecords no_arrival;
    no_arrival.reward_column_present = true;
    no_arrival.rows.push_back(ordinary_records().rows.back());
    const CounterSummary no_arrival_summary =
        core::summary_records(no_arrival);
    expect(std::isnan(no_arrival_summary.average_r2c_ratio) &&
               std::isnan(no_arrival_summary.average_reward) &&
               no_arrival_summary.early_rejection_count == 0U &&
               no_arrival_summary.place_failure_count == 0U &&
               no_arrival_summary.route_failure_count == 0U,
           "no-arrival summary semantics mismatch");

    CounterRecords zero_denominator;
    zero_denominator.reward_column_present = true;
    CounterRecord zero;
    zero.event_type = network::VirtualEventType::arrival;
    zero.success_count = 0;
    zero.virtual_network_count = 0;
    zero.v_net_r2c_ratio = 1.0;
    zero.total_time_revenue = 1.0;
    zero.total_time_cost = 0.0;
    zero.virtual_network_arrival_time = 0.0;
    zero.total_revenue = 1.0;
    zero.total_cost = 0.0;
    zero.reward = 2.0;
    zero_denominator.rows.push_back(zero);
    const CounterSummary zero_summary =
        core::summary_records(zero_denominator);
    expect(std::isnan(zero_summary.acceptance_rate) &&
               std::isinf(zero_summary.long_term_time_r2c_ratio) &&
               std::isinf(zero_summary.long_term_average_time_revenue) &&
               std::isinf(zero_summary.long_term_r2c_ratio) &&
               std::isinf(zero_summary.long_term_average_revenue),
           "zero-denominator IEEE summary mismatch");

    CounterRecords nan_records;
    nan_records.reward_column_present = true;
    CounterRecord nan;
    nan.event_type = network::VirtualEventType::arrival;
    nan.success_count = 1;
    nan.virtual_network_count = 1;
    nan.v_net_r2c_ratio = std::numeric_limits<double>::quiet_NaN();
    nan.total_time_revenue = 0.0;
    nan.total_time_cost = 1.0;
    nan.virtual_network_arrival_time = 1.0;
    nan.total_cost = 1.0;
    nan.total_revenue = 1.0;
    nan.physical_available_resource =
        std::numeric_limits<double>::quiet_NaN();
    nan.physical_node_available_resource =
        std::numeric_limits<double>::quiet_NaN();
    nan.physical_link_available_resource =
        std::numeric_limits<double>::quiet_NaN();
    nan.hard_constraint_violation =
        std::numeric_limits<double>::quiet_NaN();
    nan.max_single_step_hard_constraint_violation =
        std::numeric_limits<double>::quiet_NaN();
    nan.reward = std::nullopt;
    nan_records.rows.push_back(nan);
    const CounterSummary nan_summary = core::summary_records(nan_records);
    expect(std::isnan(nan_summary.average_r2c_ratio) &&
               std::isnan(nan_summary.minimum_physical_available_resource) &&
               nan_summary.total_violation == 0.0 &&
               nan_summary.total_max_single_step_violation == 0.0 &&
               std::isnan(nan_summary.average_reward),
           "NaN skip semantics mismatch");
}

void test_summary_csv_read_before_binding()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "virne_counter_unit_valid.csv";
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        expect(stream.is_open(), "failed to create CSV fixture");
        stream << "a,b\n1,2\n";
        expect(stream.good(), "failed to write CSV fixture");
    }
    expect_counter_error(
        [&]
        {
            static_cast<void>(core::summary_csv(path.string()));
        },
        CounterErrorCode::legacy_summary_csv_binding,
        CounterOperation::summarize_csv);
    const bool removed = std::filesystem::remove(path);
    expect(removed, "failed to remove CSV fixture");

    const std::filesystem::path missing =
        std::filesystem::temp_directory_path() /
        "virne_counter_unit_missing.csv";
    std::filesystem::remove(missing);
    bool read_failed_first = false;
    try
    {
        static_cast<void>(core::summary_csv(missing.string()));
    }
    catch (const CounterException& error)
    {
        expect(error.code() != CounterErrorCode::legacy_summary_csv_binding,
               "missing CSV reached legacy binding before read failure");
        read_failed_first = true;
    }
    catch (const std::exception&)
    {
        read_failed_first = true;
    }
    expect(read_failed_first, "missing CSV unexpectedly succeeded");
}

void test_concurrent_independent_callers()
{
    MainFixture fixture;
    const PreparedCounter counter = fixture.prepare();
    std::vector<std::future<bool>> callers;
    callers.reserve(8U);
    for (std::size_t caller = 0U; caller < 8U; ++caller)
    {
        callers.emplace_back(std::async(
            std::launch::async,
            [&fixture, &counter, caller]
            {
                const std::array<std::size_t, 4U> widths{
                    0U, 1U, 2U, 8U};
                for (std::size_t iteration = 0U;
                     iteration < widths.size();
                     ++iteration)
                {
                    core::Solution solution = fixture.mapped_solution();
                    const std::size_t workers =
                        widths[(caller + iteration) % widths.size()];
                    counter.count_partial_solution(solution, {workers});
                    if (solution.v_net_revenue != 61.0 ||
                        solution.v_net_cost != 35.5 ||
                        counter.calculate_sum_network_resource(
                            true, true, {workers}) != CounterNumber{83.5})
                    {
                        return false;
                    }
                }
                return true;
            }));
    }
    for (auto& caller : callers)
    {
        expect(caller.get(), "concurrent independent counter drift");
    }
}

} // namespace

int main()
{
    try
    {
        const auto run = [](std::string_view name, auto&& test)
        {
            try
            {
                test();
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(
                    std::string(name) + ": " + error.what());
            }
        };

        run("selection/registry/sums",
            test_selection_registry_and_resource_sums);
        run("numeric/wrap/workers",
            test_numeric_lanes_wrap_errors_and_workers);
        run("partial/helpers/errors",
            test_partial_count_helpers_and_atomic_errors);
        run("complete/partial-state",
            test_complete_count_success_failure_and_partial_state);
        run("summary", test_summary_ordinary_empty_nan_and_reward);
        run("summary CSV", test_summary_csv_read_before_binding);
        run("concurrent callers", test_concurrent_independent_callers);
        std::cout << "counter unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "counter unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
