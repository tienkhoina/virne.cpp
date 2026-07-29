#include "virtual_network_request_simulator.h"

#include "random_context.h"
#include "setting.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace attribute = virne::network::attribute;
namespace network = virne::network;
namespace utils = virne::utils;

using network::SimulationDistribution;
using network::VirtualEventType;
using network::VirtualNetwork;
using network::VirtualNetworkEvent;
using network::VirtualNetworkRequestSimulator;
using network::VirtualNetworkSimulationConfig;
using network::VirtualNetworkSimulatorErrorCode;
using network::VirtualNetworkSimulatorException;
using network::VirtualNetworkSimulatorOperation;
using network::VirtualSimulationWorkers;

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

void expect(const bool condition, const std::string& message)
{
    if (!condition) {
        fail(message);
    }
}

template <typename Callable>
void expect_any_error(Callable&& callable, const std::string& context)
{
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception&) {
        return;
    }
    fail(context + ": expected exception");
}

template <typename Callable>
void expect_simulator_error(
    Callable&& callable,
    const VirtualNetworkSimulatorErrorCode code,
    const VirtualNetworkSimulatorOperation operation,
    const std::string& context)
{
    try {
        std::forward<Callable>(callable)();
    } catch (const VirtualNetworkSimulatorException& error) {
        expect(error.code() == code, context + ": error-code drift");
        expect(error.operation() == operation, context + ": operation drift");
        expect(
            std::string_view(error.what()).empty() == false,
            context + ": missing diagnostic");
        return;
    }
    fail(context + ": expected VirtualNetworkSimulatorException");
}

std::uint64_t raw64(const double value) noexcept
{
    std::uint64_t result = 0U;
    static_assert(sizeof(result) == sizeof(value), "binary64 size mismatch");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void expect_doubles(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& context)
{
    expect(actual.size() == expected.size(), context + ": size drift");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        expect(
            raw64(actual[index]) == raw64(expected[index]),
            context + ": value drift at " + std::to_string(index));
    }
}

utils::DistributionSpec uniform_spec(
    utils::DatasetScalar low,
    utils::DatasetScalar high)
{
    utils::DistributionSpec result;
    result.kind = utils::DistributionKind::uniform;
    result.low = std::move(low);
    result.high = std::move(high);
    return result;
}

utils::DistributionSpec normal_spec(const double location, const double scale)
{
    utils::DistributionSpec result;
    result.kind = utils::DistributionKind::normal;
    result.loc = location;
    result.scale = scale;
    return result;
}

SimulationDistribution distribution(
    const utils::DatasetValueKind value_kind,
    utils::DistributionSpec spec)
{
    SimulationDistribution result;
    result.value_kind = value_kind;
    result.distribution = std::move(spec);
    return result;
}

SimulationDistribution fixed_integer(const std::int64_t value)
{
    return distribution(
        utils::DatasetValueKind::integer,
        uniform_spec(value, value));
}

SimulationDistribution fixed_floating(const double value)
{
    return distribution(
        utils::DatasetValueKind::floating,
        uniform_spec(value, value));
}

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    const attribute::AttributeOwner owner,
    const std::int64_t low,
    const std::int64_t high)
{
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    result.generative = true;
    result.distribution = uniform_spec(low, high);
    result.dtype = utils::DatasetValueKind::integer;
    return result;
}

VirtualNetworkSimulationConfig base_config(const std::size_t count)
{
    VirtualNetworkSimulationConfig result;
    result.num_virtual_networks = count;
    result.virtual_network_size = fixed_integer(3);
    result.lifetime = fixed_floating(5.0);
    result.arrival_rate = fixed_integer(0);
    result.topology_type = network::TopologyType::Path;
    return result;
}

utils::DistributionRequest request_for(
    const SimulationDistribution& source,
    const std::size_t count)
{
    utils::DistributionRequest request;
    request.count = count;
    request.value_kind = source.value_kind;
    request.distribution = source.distribution;
    return request;
}

std::vector<double> generated_to_double(
    const utils::GeneratedData& generated)
{
    std::vector<double> result;
    switch (generated.value_kind) {
    case utils::DatasetValueKind::integer: {
        const auto& values = std::get<std::vector<std::int64_t>>(
            generated.values);
        result.reserve(values.size());
        for (const std::int64_t value : values) {
            result.push_back(static_cast<double>(value));
        }
        return result;
    }
    case utils::DatasetValueKind::floating:
        return std::get<std::vector<double>>(generated.values);
    case utils::DatasetValueKind::boolean: {
        const auto& values = std::get<std::vector<std::uint8_t>>(
            generated.values);
        result.reserve(values.size());
        for (const std::uint8_t value : values) {
            result.push_back(value == 0U ? 0.0 : 1.0);
        }
        return result;
    }
    }
    fail("invalid generated value kind");
}

std::int64_t signed_from_bits(const std::uint64_t value) noexcept
{
    std::int64_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::vector<double> cumulative_arrivals(
    const utils::GeneratedData& generated)
{
    std::vector<double> result;
    switch (generated.value_kind) {
    case utils::DatasetValueKind::integer: {
        const auto& values = std::get<std::vector<std::int64_t>>(
            generated.values);
        result.reserve(values.size());
        std::uint64_t total = 0U;
        for (const std::int64_t value : values) {
            total += static_cast<std::uint64_t>(value);
            result.push_back(static_cast<double>(signed_from_bits(total)));
        }
        return result;
    }
    case utils::DatasetValueKind::floating: {
        const auto& values = std::get<std::vector<double>>(generated.values);
        result.reserve(values.size());
        double total = 0.0;
        for (const double value : values) {
            total += value;
            result.push_back(total);
        }
        return result;
    }
    case utils::DatasetValueKind::boolean: {
        const auto& values = std::get<std::vector<std::uint8_t>>(
            generated.values);
        result.reserve(values.size());
        std::int64_t total = 0;
        for (const std::uint8_t value : values) {
            total += value == 0U ? 0 : 1;
            result.push_back(static_cast<double>(total));
        }
        return result;
    }
    }
    fail("invalid arrival value kind");
}

struct ExpectedArrangement {
    std::vector<std::int64_t> sizes;
    std::vector<double> lifetimes;
    std::vector<double> arrivals;
    std::optional<std::vector<double>> max_latencies;
    std::uint32_t next_word = 0U;
    double next_normal = 0.0;
};

ExpectedArrangement arrange_with_dependencies(
    const VirtualNetworkSimulationConfig& config,
    NumpyRandomState& random)
{
    ExpectedArrangement result;
    const std::size_t count = config.num_virtual_networks;
    const auto sizes = utils::generate_data_with_distribution(
        request_for(config.virtual_network_size, count), random, 1U);
    result.sizes = std::get<std::vector<std::int64_t>>(sizes.values);
    result.lifetimes = generated_to_double(
        utils::generate_data_with_distribution(
            request_for(config.lifetime, count), random, 1U));
    result.arrivals = cumulative_arrivals(
        utils::generate_data_with_distribution(
            request_for(config.arrival_rate, count), random, 1U));
    if (config.max_latency) {
        result.max_latencies = generated_to_double(
            utils::generate_data_with_distribution(
                request_for(*config.max_latency, count), random, 1U));
    }
    result.next_word = random.next_uint32();
    result.next_normal = random.normal();
    return result;
}

void expect_arrangement(
    const VirtualNetworkRequestSimulator& simulator,
    const ExpectedArrangement& expected,
    const std::string& context)
{
    expect(
        simulator.arranged_sizes() == expected.sizes,
        context + ": sizes drift");
    expect_doubles(
        simulator.arranged_lifetimes(), expected.lifetimes,
        context + ": lifetimes");
    expect_doubles(
        simulator.arranged_arrival_times(), expected.arrivals,
        context + ": arrivals");
    expect(
        simulator.arranged_max_latencies().has_value() ==
            expected.max_latencies.has_value(),
        context + ": max-latency presence drift");
    if (expected.max_latencies) {
        expect_doubles(
            *simulator.arranged_max_latencies(), *expected.max_latencies,
            context + ": max latencies");
    }
}

void test_config_and_zero()
{
    VirtualNetworkSimulationConfig invalid = base_config(1U);
    invalid.virtual_network_size.value_kind =
        utils::DatasetValueKind::floating;
    expect_simulator_error(
        [&] {
            static_cast<void>(VirtualNetworkRequestSimulator::from_setting(
                invalid));
        },
        VirtualNetworkSimulatorErrorCode::invalid_size_distribution,
        VirtualNetworkSimulatorOperation::validate_config,
        "floating size lane");

    if (std::numeric_limits<std::size_t>::max() >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        VirtualNetworkSimulationConfig overflow = base_config(0U);
        overflow.num_virtual_networks =
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()) + 1U;
        expect_simulator_error(
            [&] {
                static_cast<void>(
                    VirtualNetworkRequestSimulator::from_setting(overflow));
            },
            VirtualNetworkSimulatorErrorCode::request_count_overflow,
            VirtualNetworkSimulatorOperation::validate_config,
            "request count overflow");
    }

    VirtualNetworkSimulationConfig zero_config = base_config(0U);
    zero_config.max_latency = fixed_floating(4.0);
    VirtualNetworkRequestSimulator zero =
        VirtualNetworkRequestSimulator::from_setting(zero_config);
    RandomContext actual(17U);
    RandomContext expected(17U);
    zero.renew(actual, true, true, std::nullopt, {});
    const ExpectedArrangement expected_buffers =
        arrange_with_dependencies(zero_config, expected.numpy());
    expect_arrangement(zero, expected_buffers, "zero request arrangement");
    expect(zero.num_v_nets() == 0U, "zero request networks");
    expect(zero.num_events() == 0U, "zero request events");
    expect(
        actual.numpy().next_uint32() == expected_buffers.next_word,
        "zero request NumPy continuation");
    expect(
        raw64(actual.numpy().normal()) ==
            raw64(expected_buffers.next_normal),
        "zero request normal continuation");
    expect(
        !zero.event_id(0, VirtualEventType::arrival).has_value(),
        "zero request event index");
}

void test_arrangement_order_lanes_and_workers()
{
    std::vector<VirtualNetworkSimulationConfig> cases;
    {
        auto config = base_config(17U);
        config.virtual_network_size = distribution(
            utils::DatasetValueKind::integer,
            uniform_spec(std::int64_t{2}, std::int64_t{8}));
        config.lifetime = distribution(
            utils::DatasetValueKind::integer,
            uniform_spec(std::int64_t{-4}, std::int64_t{12}));
        config.arrival_rate = distribution(
            utils::DatasetValueKind::integer,
            uniform_spec(std::int64_t{-2}, std::int64_t{5}));
        config.max_latency = distribution(
            utils::DatasetValueKind::floating,
            normal_spec(10.0, 2.0));
        cases.push_back(std::move(config));
    }
    {
        auto config = base_config(19U);
        config.lifetime = distribution(
            utils::DatasetValueKind::floating,
            normal_spec(7.0, 1.5));
        config.arrival_rate = distribution(
            utils::DatasetValueKind::floating,
            normal_spec(2.0, 0.25));
        config.max_latency = distribution(
            utils::DatasetValueKind::boolean,
            normal_spec(0.0, 1.0));
        cases.push_back(std::move(config));
    }
    {
        auto config = base_config(23U);
        config.lifetime = distribution(
            utils::DatasetValueKind::boolean,
            normal_spec(0.0, 1.0));
        config.arrival_rate = distribution(
            utils::DatasetValueKind::boolean,
            normal_spec(0.0, 1.0));
        config.max_latency.reset();
        cases.push_back(std::move(config));
    }

    const std::array<std::size_t, 4U> workers{0U, 1U, 2U, 8U};
    for (std::size_t case_index = 0U; case_index < cases.size(); ++case_index) {
        NumpyRandomState expected_random(100U +
            static_cast<std::uint32_t>(case_index));
        const ExpectedArrangement expected = arrange_with_dependencies(
            cases[case_index], expected_random);
        for (const std::size_t width : workers) {
            NumpyRandomState actual_random(100U +
                static_cast<std::uint32_t>(case_index));
            auto simulator = VirtualNetworkRequestSimulator::from_setting(
                cases[case_index]);
            simulator.arrange_v_nets(actual_random, width);
            const std::string context =
                "arrangement case=" + std::to_string(case_index) +
                " workers=" + std::to_string(width);
            expect_arrangement(simulator, expected, context);
            expect(
                actual_random.next_uint32() == expected.next_word,
                context + ": next word drift");
            expect(
                raw64(actual_random.normal()) == raw64(expected.next_normal),
                context + ": normal continuation drift");
        }
    }
}

const AttrValue& graph_value(
    const VirtualNetwork& value,
    const std::string_view name,
    const std::string& context)
{
    const AttrValue* found = value.graph_attributes().find(name);
    expect(found != nullptr, context + ": missing graph value");
    return *found;
}

std::string attr_scalar(const AttrValue& value)
{
    if (const auto* item = std::get_if<std::int64_t>(&value)) {
        return "i" + std::to_string(*item);
    }
    if (const auto* item = std::get_if<double>(&value)) {
        return "d" + std::to_string(raw64(*item));
    }
    if (const auto* item = std::get_if<bool>(&value)) {
        return *item ? "b1" : "b0";
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        return "s" + *item;
    }
    return "recursive";
}

std::string simulator_snapshot(const VirtualNetworkRequestSimulator& value)
{
    std::ostringstream output;
    output << value.num_v_nets() << ':' << value.num_events() << '|';
    for (const VirtualNetwork& request : value.v_nets()) {
        output << request.request_id().value_or(-1) << ','
               << raw64(request.arrival_time().value_or(-1.0)) << ','
               << raw64(request.lifetime().value_or(-1.0)) << ','
               << (request.max_latency()
                       ? std::to_string(raw64(*request.max_latency()))
                       : std::string{"none"})
               << ',' << request.graph().num_nodes() << ','
               << request.graph().num_edges() << ';';
        const auto cpu = request.bind_node_attribute("cpu");
        if (cpu) {
            const auto rows = network::get_node_attrs_data(
                request, {cpu->registry_id}, 1U);
            for (const AttrValue& item : rows.front()) {
                output << attr_scalar(item) << ',';
            }
        }
        const auto bandwidth = request.bind_link_attribute("bandwidth");
        if (bandwidth) {
            const auto rows = network::get_link_attrs_data(
                request, {bandwidth->registry_id}, 1U);
            for (const AttrValue& item : rows.front()) {
                output << attr_scalar(item) << ',';
            }
        }
        output << '|';
    }
    for (const VirtualNetworkEvent& event : value.events()) {
        output << event.id() << ','
               << static_cast<unsigned>(event.type()) << ','
               << event.virtual_network_id() << ','
               << raw64(event.time()) << ';';
    }
    return output.str();
}

VirtualNetworkSimulationConfig generated_path_config()
{
    auto config = base_config(3U);
    config.max_latency = fixed_floating(7.5);
    config.node_attribute_specs.push_back(resource_spec(
        "cpu", attribute::AttributeOwner::node, 1, 9));
    config.link_attribute_specs.push_back(resource_spec(
        "bandwidth", attribute::AttributeOwner::link, 10, 19));
    config.topology_metadata = make_attr_object({
        {"family", std::string{"path"}},
        {"nested", make_attr_list({std::int64_t{3}, 4.5})},
    });
    config.output_metadata = make_attr_object({
        {"events", std::string{"enabled"}},
    });
    config.source_setting = utils::parse_setting(
        R"json({"nested":{"value":7},"name":"sim"})json",
        utils::SettingFormat::json);
    return config;
}

const AttrObjectPtr& object_value(const AttrValue& value, const std::string& context)
{
    expect(
        std::holds_alternative<AttrObjectPtr>(value),
        context + ": expected object");
    const AttrObjectPtr& result = std::get<AttrObjectPtr>(value);
    expect(result != nullptr, context + ": null object");
    return result;
}

void test_generation_metadata_events_and_workers()
{
    const VirtualNetworkSimulationConfig config = generated_path_config();
    const std::array<std::size_t, 4U> widths{0U, 1U, 2U, 8U};
    std::optional<std::string> baseline;
    std::optional<std::uint32_t> baseline_python;
    std::optional<std::uint32_t> baseline_numpy;

    for (const std::size_t width : widths) {
        VirtualSimulationWorkers workers;
        workers.factory_workers = width;
        workers.arrangement_workers = width;
        workers.attribute_workers = width;
        workers.event_workers = width;
        RandomContext random(999U);
        auto simulator = VirtualNetworkRequestSimulator::from_setting(config);
        simulator.renew(random, true, true, 43U, workers);

        expect(simulator.num_v_nets() == 3U, "generated request count");
        expect(simulator.num_events() == 6U, "generated event count");
        const AttrObjectPtr& config_topology = object_value(
            *simulator.config().topology_metadata, "config topology");
        const AttrObjectPtr& config_output = object_value(
            *simulator.config().output_metadata, "config output");
        std::vector<const AttrObject*> network_topologies;
        std::vector<const AttrObject*> network_outputs;

        for (std::size_t index = 0U; index < simulator.v_nets().size(); ++index) {
            const VirtualNetwork& request = simulator.v_nets()[index];
            expect(
                request.request_id() ==
                    std::optional<std::int64_t>{
                        static_cast<std::int64_t>(index)},
                "request id");
            expect(
                request.arrival_time() == std::optional<double>{0.0},
                "request arrival");
            expect(
                request.lifetime() == std::optional<double>{5.0},
                "request lifetime");
            expect(
                request.max_latency() == std::optional<double>{7.5},
                "request max latency");
            expect(
                request.graph().num_nodes() == 3U &&
                    request.graph().num_edges() == 2U,
                "request path topology");
            expect(
                std::get<std::int64_t>(graph_value(
                    request, "id", "request id metadata")) ==
                    static_cast<std::int64_t>(index),
                "request graph id");
            expect(
                raw64(std::get<double>(graph_value(
                    request, "arrival_time", "arrival metadata"))) ==
                    raw64(0.0),
                "request graph arrival");
            expect(
                raw64(std::get<double>(graph_value(
                    request, "lifetime", "lifetime metadata"))) == raw64(5.0),
                "request graph lifetime");
            expect(
                raw64(std::get<double>(graph_value(
                    request, "max_latency", "max latency metadata"))) ==
                    raw64(7.5),
                "request graph max latency");

            const AttrObjectPtr& request_topology = object_value(
                graph_value(request, "topology", "topology metadata"),
                "request topology");
            const AttrObjectPtr& request_output = object_value(
                graph_value(request, "output", "output metadata"),
                "request output");
            expect(
                request_topology.get() != config_topology.get() &&
                    request_output.get() != config_output.get(),
                "request recursive metadata cloned from config");
            for (const AttrObject* previous : network_topologies) {
                expect(
                    previous != request_topology.get(),
                    "request topology metadata independence");
            }
            for (const AttrObject* previous : network_outputs) {
                expect(
                    previous != request_output.get(),
                    "request output metadata independence");
            }
            network_topologies.push_back(request_topology.get());
            network_outputs.push_back(request_output.get());
        }

        for (std::size_t index = 0U; index < 3U; ++index) {
            const VirtualNetworkEvent& arrival = simulator.events()[index];
            const VirtualNetworkEvent& leave = simulator.events()[index + 3U];
            expect(
                arrival.id() == index &&
                    arrival.type() == VirtualEventType::arrival &&
                    arrival.virtual_network_id() ==
                        static_cast<std::int64_t>(index) &&
                    raw64(arrival.time()) == raw64(0.0),
                "arrival tie order");
            expect(
                leave.id() == index + 3U &&
                    leave.type() == VirtualEventType::leave &&
                    leave.virtual_network_id() ==
                        static_cast<std::int64_t>(index) &&
                    raw64(leave.time()) == raw64(5.0),
                "leave tie order");
            expect(
                simulator.event_id(
                    static_cast<std::int64_t>(index),
                    VirtualEventType::arrival) ==
                    std::optional<std::size_t>{index},
                "arrival event index");
            expect(
                simulator.event_id(
                    static_cast<std::int64_t>(index),
                    VirtualEventType::leave) ==
                    std::optional<std::size_t>{index + 3U},
                "leave event index");
        }
        expect(
            !simulator.event_id(-1, VirtualEventType::arrival).has_value(),
            "negative event query");
        expect(
            !simulator.event_id(
                0, static_cast<VirtualEventType>(255U)).has_value(),
            "invalid event enum query");

        const std::string snapshot = simulator_snapshot(simulator);
        const std::uint32_t next_python = random.python().getrandbits32();
        const std::uint32_t next_numpy = random.numpy().next_uint32();
        if (!baseline) {
            baseline = snapshot;
            baseline_python = next_python;
            baseline_numpy = next_numpy;
        } else {
            expect(snapshot == *baseline, "generated worker output drift");
            expect(next_python == *baseline_python, "generated Py RNG drift");
            expect(next_numpy == *baseline_numpy, "generated NumPy RNG drift");
        }
    }
}

void test_from_state_sparse_duplicate_and_clone_move()
{
    auto config = base_config(3U);
    config.topology_metadata = make_attr_object({{"deep", std::int64_t{1}}});
    config.source_setting = utils::parse_setting(
        R"json({"deep":{"value":2}})json", utils::SettingFormat::json);

    std::vector<VirtualNetworkEvent> events;
    events.emplace_back(90U, VirtualEventType::arrival, 1, 9.0);
    events.emplace_back(91U, VirtualEventType::arrival, 1, 8.0);
    events.emplace_back(92U, VirtualEventType::leave, 1, 10.0);
    events.emplace_back(93U, VirtualEventType::arrival, 42, 7.0);
    events.emplace_back(94U, VirtualEventType::leave, 42, 11.0);
    const std::vector<VirtualNetworkEvent> expected_events = events;

    auto simulator = VirtualNetworkRequestSimulator::from_state(
        config, {}, std::move(events));
    expect(simulator.events() == expected_events, "from_state event order/ids");
    expect(
        simulator.event_id(1, VirtualEventType::arrival) ==
            std::optional<std::size_t>{91U},
        "dense duplicate last-write-wins");
    expect(
        simulator.event_id(1, VirtualEventType::leave) ==
            std::optional<std::size_t>{92U},
        "dense leave lookup");
    expect(
        simulator.event_id(42, VirtualEventType::arrival) ==
            std::optional<std::size_t>{93U},
        "sparse arrival lookup");
    expect(
        simulator.event_id(42, VirtualEventType::leave) ==
            std::optional<std::size_t>{94U},
        "sparse leave lookup");
    expect(
        !simulator.event_id(2, VirtualEventType::arrival).has_value(),
        "missing dense event lookup");
    expect(
        !simulator.event_id(999, VirtualEventType::leave).has_value(),
        "missing sparse event lookup");

    auto cloned = simulator.clone();
    expect(cloned.events() == simulator.events(), "clone event values");
    expect(
        cloned.event_id(42, VirtualEventType::leave) ==
            simulator.event_id(42, VirtualEventType::leave),
        "clone sparse index");
    const AttrObjectPtr& source_topology = object_value(
        *simulator.config().topology_metadata, "source topology config");
    const AttrObjectPtr& clone_topology = object_value(
        *cloned.config().topology_metadata, "clone topology config");
    expect(
        source_topology.get() != clone_topology.get(),
        "clone recursive config metadata independence");
    expect(
        simulator.config().source_setting->root.object_ptr().get() !=
            cloned.config().source_setting->root.object_ptr().get(),
        "clone source setting independence");

    VirtualNetworkRequestSimulator moved(std::move(cloned));
    expect(
        moved.event_id(42, VirtualEventType::arrival) ==
            std::optional<std::size_t>{93U},
        "move construction event index");
    auto assigned = VirtualNetworkRequestSimulator::from_setting(base_config(0U));
    assigned = std::move(moved);
    expect(
        assigned.events() == expected_events &&
            assigned.event_id(1, VirtualEventType::arrival) ==
                std::optional<std::size_t>{91U},
        "move assignment state");

    RandomContext release_random(61U);
    auto releasable = VirtualNetworkRequestSimulator::from_setting(
        base_config(3U));
    releasable.renew(release_random, true, true);
    auto released = std::move(releasable).release_v_nets();
    expect(released.size() == 3U, "release request count");
    expect(
        released.front().request_id() == std::optional<std::int64_t>{0} &&
            released.back().request_id() ==
                std::optional<std::int64_t>{2},
        "release preserves request order and IDs");
}

void test_renew_flags_and_seed()
{
    auto config = base_config(4U);
    config.virtual_network_size = distribution(
        utils::DatasetValueKind::integer,
        uniform_spec(std::int64_t{2}, std::int64_t{5}));
    config.lifetime = distribution(
        utils::DatasetValueKind::floating,
        uniform_spec(1.0, 20.0));
    config.arrival_rate = distribution(
        utils::DatasetValueKind::integer,
        uniform_spec(std::int64_t{0}, std::int64_t{5}));

    auto simulator = VirtualNetworkRequestSimulator::from_setting(config);
    RandomContext random(1U);
    simulator.renew(random, true, true, 101U, {});
    const std::string original_networks = simulator_snapshot(simulator);
    const std::vector<VirtualNetworkEvent> original_events = simulator.events();

    simulator.renew(random, false, false, 202U, {});
    RandomContext expected_seed(202U);
    expect(
        random.python().getrandbits32() ==
            expected_seed.python().getrandbits32(),
        "seed occurs when both renew flags are false: Python");
    expect(
        random.numpy().next_uint32() == expected_seed.numpy().next_uint32(),
        "seed occurs when both renew flags are false: NumPy");
    expect(
        simulator_snapshot(simulator) == original_networks,
        "both false preserves state");

    simulator.renew(random, true, false, 303U, {});
    expect(
        simulator.events() == original_events,
        "network-only renewal leaves stale events");
    const std::string renewed_networks = simulator_snapshot(simulator);
    expect(
        renewed_networks != original_networks,
        "network-only renewal changes generated requests");
    simulator.renew(random, false, true, std::nullopt, {});
    expect(
        simulator.events() != original_events,
        "event-only renewal observes current requests");

    const std::string before_clone_renew = simulator_snapshot(simulator);
    auto cloned = simulator.clone();
    RandomContext clone_random(0U);
    cloned.renew(clone_random, true, true, 404U, {});
    expect(
        simulator_snapshot(simulator) == before_clone_renew,
        "clone renewal does not mutate source");
}

struct RetrySeeds {
    std::uint32_t all_success = 0U;
    std::uint32_t second_failure = 0U;
};

RetrySeeds find_retry_seeds(const network::TopologyOptions& options)
{
    RetrySeeds result;
    bool found_success = false;
    bool found_failure = false;
    for (std::uint32_t seed = 0U;
         seed < 100000U && (!found_success || !found_failure);
         ++seed) {
        RandomContext random(seed);
        bool first_success = false;
        bool second_success = false;
        try {
            static_cast<void>(network::TopologyGenerator::generate(
                network::TopologyType::Random, 5, options,
                random.python()));
            first_success = true;
            static_cast<void>(network::TopologyGenerator::generate(
                network::TopologyType::Random, 5, options,
                random.python()));
            second_success = true;
        } catch (const std::exception&) {
        }
        if (first_success && second_success && !found_success) {
            result.all_success = seed;
            found_success = true;
        }
        if (first_success && !second_success && !found_failure) {
            result.second_failure = seed;
            found_failure = true;
        }
    }
    expect(found_success, "unable to find two-success topology seed");
    expect(found_failure, "unable to find second-failure topology seed");
    return result;
}

void test_failed_network_renew_preserves_old_commit()
{
    auto config = base_config(2U);
    config.virtual_network_size = fixed_integer(5);
    config.topology_type = network::TopologyType::Random;
    config.topology_options.random_prob = 0.35;
    config.topology_options.max_attempts = 1U;
    const RetrySeeds seeds = find_retry_seeds(config.topology_options);

    auto simulator = VirtualNetworkRequestSimulator::from_setting(config);
    RandomContext random(0U);
    simulator.renew(random, true, true, seeds.all_success, {});
    const std::string before = simulator_snapshot(simulator);

    expect_any_error(
        [&] {
            simulator.renew(
                random, true, true, seeds.second_failure, {});
        },
        "second request topology failure");
    expect(
        simulator_snapshot(simulator) == before,
        "failed network renewal preserves old networks/events");

    RandomContext expected(seeds.second_failure);
    static_cast<void>(network::TopologyGenerator::generate(
        network::TopologyType::Random, 5, config.topology_options,
        expected.python()));
    expect_any_error(
        [&] {
            static_cast<void>(network::TopologyGenerator::generate(
                network::TopologyType::Random, 5,
                config.topology_options, expected.python()));
        },
        "expected second topology failure");
    expect(
        random.python().getrandbits32() ==
            expected.python().getrandbits32(),
        "failed renewal Python RNG continuation");
}

void test_independent_concurrent_contexts()
{
    const auto config = generated_path_config();
    const std::array<std::size_t, 4U> widths{0U, 1U, 2U, 8U};
    struct Result {
        std::string snapshot;
        std::uint32_t python = 0U;
        std::uint32_t numpy = 0U;
    };
    std::array<std::future<Result>, 4U> futures;
    for (std::size_t index = 0U; index < widths.size(); ++index) {
        futures[index] = std::async(
            std::launch::async,
            [config, width = widths[index]] {
                VirtualSimulationWorkers workers;
                workers.factory_workers = width;
                workers.arrangement_workers = width;
                workers.attribute_workers = width;
                workers.event_workers = width;
                RandomContext random(0U);
                auto simulator =
                    VirtualNetworkRequestSimulator::from_setting(config);
                simulator.renew(random, true, true, 55U, workers);
                return Result{
                    simulator_snapshot(simulator),
                    random.python().getrandbits32(),
                    random.numpy().next_uint32(),
                };
            });
    }
    const Result baseline = futures.front().get();
    for (std::size_t index = 1U; index < futures.size(); ++index) {
        const Result current = futures[index].get();
        expect(
            current.snapshot == baseline.snapshot,
            "concurrent simulator output drift");
        expect(
            current.python == baseline.python,
            "concurrent simulator Python RNG drift");
        expect(
            current.numpy == baseline.numpy,
            "concurrent simulator NumPy RNG drift");
    }
}

}  // namespace

int main()
{
    try {
        test_config_and_zero();
        test_arrangement_order_lanes_and_workers();
        test_generation_metadata_events_and_workers();
        test_from_state_sparse_duplicate_and_clone_move();
        test_renew_flags_and_seed();
        test_failed_network_renew_preserves_old_commit();
        test_independent_concurrent_contexts();
        std::cout << "VirtualNetworkRequestSimulator unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkRequestSimulator unit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
