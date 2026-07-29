#include "environment.h"

#include "attribute/attribute_factory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
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
namespace utils = virne::utils;

constexpr std::size_t physical_node_count = 4096U;
constexpr std::size_t physical_link_count = physical_node_count - 1U;
constexpr std::size_t virtual_node_count = 8U;
constexpr std::size_t request_count = 96U;
constexpr std::size_t event_count = request_count * 2U;
constexpr std::size_t warmup_count = 1U;
constexpr std::size_t sample_count = 3U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct PhysicalFixture
{
    network::PhysicalNetwork network;
    AttrId node_value_id = 0U;
    AttrId link_value_id = 0U;
};

struct EnvironmentFixture
{
    std::unique_ptr<core::SolutionStepEnvironment> environment;
    AttrId node_value_id = 0U;
    AttrId link_value_id = 0U;
    std::vector<core::Solution> input_templates;
};

struct ScheduledEvent
{
    double time = 0.0;
    network::VirtualEventType type = network::VirtualEventType::leave;
    network::VirtualRequestId request_id = 0;
};

struct OutputDigest
{
    std::uint64_t checksum = fnv_offset;
    std::uint64_t bytes = 0U;

    void append_byte(const std::uint8_t value)
    {
        checksum ^= value;
        checksum *= fnv_prime;
        ++bytes;
    }

    void append_u64(const std::uint64_t value)
    {
        for (unsigned int shift = 0U; shift < 64U; shift += 8U)
        {
            append_byte(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void append_i64(const std::int64_t value)
    {
        append_u64(static_cast<std::uint64_t>(value));
    }

    void append_double(const double value)
    {
        std::uint64_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value), "double width drift");
        std::memcpy(&bits, &value, sizeof(bits));
        append_u64(bits);
    }

    void append_string(const std::string& value)
    {
        append_u64(static_cast<std::uint64_t>(value.size()));
        for (const unsigned char byte : value)
        {
            append_byte(byte);
        }
    }
};

struct EpisodeOutput
{
    std::uint64_t checksum = 0U;
    std::uint64_t physical_checksum = 0U;
    std::uint64_t output_bytes = 0U;
    std::size_t records = 0U;
    std::size_t accepted = 0U;
};

attribute::AttributeFactorySpec resource_spec(
    std::string name,
    const attribute::AttributeOwner owner)
{
    attribute::AttributeFactorySpec result;
    result.name = std::move(name);
    result.owner = owner;
    result.kind = attribute::AttributeKind::resource;
    result.restriction = attribute::ConstraintRestriction::hard;
    result.checking_level = owner == attribute::AttributeOwner::node
        ? attribute::CheckingLevel::node
        : attribute::CheckingLevel::link;
    return result;
}

network::BaseNetworkConstruction construction(Graph graph)
{
    network::BaseNetworkConstruction result;
    result.incoming_graph = std::move(graph);
    result.config.node_attribute_specs.push_back(resource_spec(
        "cpu", attribute::AttributeOwner::node));
    result.config.link_attribute_specs.push_back(resource_spec(
        "bandwidth", attribute::AttributeOwner::link));
    return result;
}

double node_capacity(const std::size_t index)
{
    return 10000.0 + static_cast<double>(index % 31U) * 0.25;
}

double link_capacity(const std::size_t index)
{
    return 20000.0 + static_cast<double>(index % 29U) * 0.5;
}

double node_demand(
    const std::size_t request,
    const std::size_t index)
{
    return 0.5 + static_cast<double>((request + index) % 8U) * 0.125;
}

double link_demand(
    const std::size_t request,
    const std::size_t index)
{
    return 0.25 + static_cast<double>((request + index) % 8U) * 0.0625;
}

bool request_is_accepted(const std::size_t request)
{
    return request % 3U != 1U;
}

std::size_t expected_accepted_count()
{
    std::size_t result = 0U;
    for (std::size_t request = 0U; request < request_count; ++request)
    {
        result += request_is_accepted(request) ? 1U : 0U;
    }
    return result;
}

PhysicalFixture make_physical_network()
{
    std::vector<EdgeEndpoints> edges;
    edges.reserve(physical_link_count);
    for (std::size_t index = 0U; index < physical_link_count; ++index)
    {
        edges.emplace_back(
            static_cast<Vertex>(index),
            static_cast<Vertex>(index + 1U));
    }

    network::PhysicalNetwork result(construction(Graph(
        physical_node_count, std::move(edges))));
    const auto node = result.bind_node_attribute("cpu");
    const auto link = result.bind_link_attribute("bandwidth");
    if (!node.has_value() || !link.has_value())
    {
        throw std::runtime_error("environment benchmark physical binding failure");
    }

    std::vector<AttrValue> node_values(physical_node_count);
    std::vector<AttrValue> link_values(physical_link_count);
    for (std::size_t index = 0U; index < physical_node_count; ++index)
    {
        node_values[index] = node_capacity(index);
        if (index < physical_link_count)
        {
            link_values[index] = link_capacity(index);
        }
    }
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        node->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(node_values)}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        link->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(link_values)}});
    return PhysicalFixture{
        std::move(result), node->value_id, link->value_id};
}

network::VirtualNetwork make_virtual_network(const std::size_t request)
{
    std::vector<EdgeEndpoints> edges;
    edges.reserve(virtual_node_count - 1U);
    for (std::size_t index = 0U; index + 1U < virtual_node_count; ++index)
    {
        edges.emplace_back(
            static_cast<Vertex>(index),
            static_cast<Vertex>(index + 1U));
    }
    network::VirtualNetwork result(construction(Graph(
        virtual_node_count, std::move(edges))));
    const auto node = result.bind_node_attribute("cpu");
    const auto link = result.bind_link_attribute("bandwidth");
    if (!node.has_value() || !link.has_value())
    {
        throw std::runtime_error("environment benchmark virtual binding failure");
    }

    std::vector<AttrValue> node_values(virtual_node_count);
    std::vector<AttrValue> link_values(virtual_node_count - 1U);
    for (std::size_t index = 0U; index < virtual_node_count; ++index)
    {
        node_values[index] = node_demand(request, index);
    }
    for (std::size_t index = 0U; index + 1U < virtual_node_count; ++index)
    {
        link_values[index] = link_demand(request, index);
    }
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        node->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(node_values)}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        link->registry_id,
        network::AttributeDataLayout::dense,
        {},
        std::move(link_values)}});
    result.set_request_id(static_cast<network::VirtualRequestId>(request));
    result.set_arrival_time(static_cast<double>(request));
    result.set_lifetime(8.5);
    return result;
}

network::VirtualNetworkRequestSimulator make_simulator()
{
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = request_count;
    config.virtual_network_size.value_kind = utils::DatasetValueKind::integer;
    config.virtual_network_size.distribution.kind =
        utils::DistributionKind::uniform;
    config.virtual_network_size.distribution.low =
        static_cast<std::int64_t>(virtual_node_count);
    config.virtual_network_size.distribution.high =
        static_cast<std::int64_t>(virtual_node_count + 1U);
    config.lifetime.value_kind = utils::DatasetValueKind::floating;
    config.lifetime.distribution.kind = utils::DistributionKind::uniform;
    config.lifetime.distribution.low = 8.5;
    config.lifetime.distribution.high = 8.5;
    config.arrival_rate.value_kind = utils::DatasetValueKind::floating;
    config.arrival_rate.distribution.kind = utils::DistributionKind::uniform;
    config.arrival_rate.distribution.low = 1.0;
    config.arrival_rate.distribution.high = 1.0;
    config.topology_type = network::TopologyType::Path;
    config.node_attribute_specs.push_back(resource_spec(
        "cpu", attribute::AttributeOwner::node));
    config.link_attribute_specs.push_back(resource_spec(
        "bandwidth", attribute::AttributeOwner::link));

    std::vector<network::VirtualNetwork> virtual_networks;
    virtual_networks.reserve(request_count);
    std::vector<ScheduledEvent> scheduled;
    scheduled.reserve(event_count);
    for (std::size_t request = 0U; request < request_count; ++request)
    {
        virtual_networks.push_back(make_virtual_network(request));
        const auto request_id =
            static_cast<network::VirtualRequestId>(request);
        scheduled.push_back(ScheduledEvent{
            static_cast<double>(request),
            network::VirtualEventType::arrival,
            request_id});
        scheduled.push_back(ScheduledEvent{
            static_cast<double>(request) + 8.5,
            network::VirtualEventType::leave,
            request_id});
    }
    std::stable_sort(
        scheduled.begin(),
        scheduled.end(),
        [](const ScheduledEvent& lhs, const ScheduledEvent& rhs) {
            return lhs.time < rhs.time;
        });

    std::vector<network::VirtualNetworkEvent> events;
    events.reserve(event_count);
    for (std::size_t index = 0U; index < scheduled.size(); ++index)
    {
        const auto& event = scheduled[index];
        events.emplace_back(
            index, event.type, event.request_id, event.time);
    }
    return network::VirtualNetworkRequestSimulator::from_state(
        std::move(config),
        std::move(virtual_networks),
        std::move(events));
}

core::EnvironmentConfig make_config(
    const std::filesystem::path& root,
    const std::size_t workers)
{
    core::EnvironmentConfig config;
    config.controller.node_resources = {0U};
    config.controller.link_resources = {0U};
    config.counter.node_resources = std::vector<core::CounterResourceId>{0U};
    config.counter.link_resources = std::vector<core::CounterResourceId>{0U};
    config.recorder.save_root_dir = root;
    config.recorder.solver_name = "environment-benchmark";
    config.recorder.run_id = "w" + std::to_string(workers);
    config.recorder.record_dir_name = "records";
    config.recorder.temporary_records = false;
    config.workers = core::EnvironmentWorkers{workers, workers, workers};
    return config;
}

core::Solution make_solution(
    const network::VirtualNetwork& virtual_network,
    const std::size_t request)
{
    core::Solution solution = core::Solution::from_v_net(virtual_network);
    if (!request_is_accepted(request))
    {
        solution.result = false;
        solution.place_result = false;
        solution.route_result = false;
        return solution;
    }

    solution.result = true;
    const std::size_t offset =
        (request * 13U) % (physical_node_count - virtual_node_count);
    for (std::size_t index = 0U; index < virtual_node_count; ++index)
    {
        const auto virtual_node = static_cast<core::SolutionNodeId>(index);
        const auto physical_node = static_cast<core::SolutionNodeId>(
            offset + index);
        core::SolutionAttributeValues values;
        values.set(0U, node_demand(request, index));
        solution.node_slots.insert_or_assign(virtual_node, physical_node);
        solution.node_slots_info.insert_or_assign(
            core::NodeSlotInfoKey{virtual_node, physical_node},
            std::move(values));
    }
    for (std::size_t index = 0U; index + 1U < virtual_node_count; ++index)
    {
        const core::SolutionLink virtual_link{
            static_cast<core::SolutionNodeId>(index),
            static_cast<core::SolutionNodeId>(index + 1U)};
        const core::SolutionLink physical_link{
            static_cast<core::SolutionNodeId>(
                offset + index),
            static_cast<core::SolutionNodeId>(
                offset + index + 1U)};
        core::SolutionAttributeValues values;
        values.set(0U, link_demand(request, index));
        solution.link_paths.insert_or_assign(
            virtual_link, std::vector<core::SolutionLink>{physical_link});
        solution.link_paths_info.insert_or_assign(
            core::LinkPathInfoKey{virtual_link, physical_link},
            std::move(values));
    }
    return solution;
}

EnvironmentFixture make_environment(
    const std::filesystem::path& root,
    const std::size_t workers)
{
    PhysicalFixture physical = make_physical_network();
    auto environment = std::make_unique<core::SolutionStepEnvironment>(
        std::move(physical.network),
        make_simulator(),
        make_config(root, workers));
    environment->reset();

    std::vector<core::Solution> inputs;
    inputs.reserve(request_count);
    const auto& virtual_networks = environment->simulator().v_nets();
    for (std::size_t request = 0U; request < request_count; ++request)
    {
        inputs.push_back(make_solution(virtual_networks[request], request));
    }
    return EnvironmentFixture{
        std::move(environment),
        physical.node_value_id,
        physical.link_value_id,
        std::move(inputs)};
}

double numeric_value(const AttrValue& value)
{
    if (const auto* number = std::get_if<double>(&value))
    {
        return *number;
    }
    if (const auto* number = std::get_if<std::int64_t>(&value))
    {
        return static_cast<double>(*number);
    }
    if (const auto* number = std::get_if<bool>(&value))
    {
        return *number ? 1.0 : 0.0;
    }
    throw std::runtime_error("environment benchmark nonnumeric resource");
}

OutputDigest physical_digest(
    const core::SolutionStepEnvironment& environment,
    const AttrId node_value_id,
    const AttrId link_value_id)
{
    OutputDigest result;
    const auto& graph = environment.physical_network().graph();
    for (std::size_t index = 0U; index < physical_node_count; ++index)
    {
        result.append_double(numeric_value(
            graph.node_attrs(static_cast<Vertex>(index)).at(node_value_id)));
    }
    for (std::size_t index = 0U; index < physical_link_count; ++index)
    {
        const Edge edge = graph.edge(
            static_cast<Vertex>(index),
            static_cast<Vertex>(index + 1U));
        result.append_double(numeric_value(
            graph.edge_attrs(edge).at(link_value_id)));
    }
    return result;
}

std::uint8_t phase_value(const core::EnvironmentPhase phase)
{
    switch (phase)
    {
    case core::EnvironmentPhase::unready:
        return 0U;
    case core::EnvironmentPhase::active:
        return 1U;
    case core::EnvironmentPhase::finished:
        return 2U;
    }
    throw std::runtime_error("environment benchmark invalid phase");
}

EpisodeOutput episode_output(
    const EnvironmentFixture& fixture,
    const std::size_t accepted)
{
    const auto& environment = *fixture.environment;
    const auto& state = environment.state();
    const auto& recorder = environment.recorder();
    const auto& recorder_state = recorder.state();
    if (state.phase != core::EnvironmentPhase::finished ||
        state.num_processed_events != event_count ||
        recorder.memory().size() != event_count ||
        recorder_state.virtual_network_count !=
            static_cast<std::int64_t>(request_count) ||
        recorder_state.success_count !=
            static_cast<std::int64_t>(expected_accepted_count()) ||
        recorder_state.inservice_count != 0 ||
        accepted != expected_accepted_count())
    {
        throw std::runtime_error("environment benchmark final state gate mismatch");
    }

    const OutputDigest physical = physical_digest(
        environment, fixture.node_value_id, fixture.link_value_id);
    OutputDigest output;
    output.append_byte(phase_value(state.phase));
    output.append_u64(state.num_processed_events);
    output.append_i64(recorder_state.virtual_network_count);
    output.append_i64(recorder_state.success_count);
    output.append_i64(recorder_state.inservice_count);
    output.append_u64(recorder.memory().size());

    const auto& graph = environment.physical_network().graph();
    for (std::size_t index = 0U; index < physical_node_count; ++index)
    {
        output.append_double(numeric_value(
            graph.node_attrs(static_cast<Vertex>(index)).at(
                fixture.node_value_id)));
    }
    for (std::size_t index = 0U; index < physical_link_count; ++index)
    {
        const Edge edge = graph.edge(
            static_cast<Vertex>(index),
            static_cast<Vertex>(index + 1U));
        output.append_double(numeric_value(
            graph.edge_attrs(edge).at(fixture.link_value_id)));
    }
    for (const core::RecorderRecord& record : recorder.memory())
    {
        if (!record.state.event.has_value())
        {
            throw std::runtime_error("environment benchmark record event missing");
        }
        output.append_i64(record.state.event->event_id);
        output.append_byte(static_cast<std::uint8_t>(
            record.state.event->type));
        output.append_i64(record.solution.v_net_id);
        output.append_byte(record.solution.result ? 1U : 0U);
        output.append_string(record.solution.description);
    }
    return EpisodeOutput{
        output.checksum,
        physical.checksum,
        output.bytes,
        recorder.memory().size(),
        accepted};
}

std::size_t run_episode(
    core::SolutionStepEnvironment& environment,
    std::vector<core::Solution>& inputs)
{
    std::size_t accepted = 0U;
    for (std::size_t request = 0U; request < inputs.size(); ++request)
    {
        const core::EnvironmentStepResult result =
            environment.step(inputs[request]);
        const bool expected = request_is_accepted(request);
        if (result.accepted != expected ||
            result.done != (request + 1U == request_count))
        {
            throw std::runtime_error("environment benchmark step gate mismatch");
        }
        accepted += result.accepted ? 1U : 0U;
    }
    return accepted;
}

void require_same_output(
    const EpisodeOutput& lhs,
    const EpisodeOutput& rhs,
    const std::string_view stage)
{
    if (lhs.checksum != rhs.checksum ||
        lhs.physical_checksum != rhs.physical_checksum ||
        lhs.output_bytes != rhs.output_bytes ||
        lhs.records != rhs.records ||
        lhs.accepted != rhs.accepted)
    {
        throw std::runtime_error(
            "environment benchmark output drift at " + std::string(stage));
    }
}

double median(std::array<double, sample_count> values)
{
    std::sort(values.begin(), values.end());
    return values[sample_count / 2U];
}

void benchmark(const std::filesystem::path& root)
{
    std::cout << std::setprecision(17);
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        EnvironmentFixture fixture = make_environment(
            root / ("workers-" + std::to_string(workers)), workers);
        const OutputDigest baseline_physical = physical_digest(
            *fixture.environment,
            fixture.node_value_id,
            fixture.link_value_id);

        std::vector<core::Solution> gate_inputs = fixture.input_templates;
        const std::size_t gate_accepted = run_episode(
            *fixture.environment, gate_inputs);
        const EpisodeOutput expected = episode_output(fixture, gate_accepted);
        if (expected.physical_checksum != baseline_physical.checksum)
        {
            throw std::runtime_error(
                "environment benchmark physical restoration mismatch: " +
                std::to_string(baseline_physical.checksum) + " != " +
                std::to_string(expected.physical_checksum));
        }

        for (std::size_t warmup = 0U; warmup < warmup_count; ++warmup)
        {
            fixture.environment->reset();
            std::vector<core::Solution> inputs = fixture.input_templates;
            const std::size_t accepted = run_episode(
                *fixture.environment, inputs);
            require_same_output(
                expected,
                episode_output(fixture, accepted),
                "warmup");
        }

        std::array<double, sample_count> samples{};
        for (std::size_t sample = 0U; sample < sample_count; ++sample)
        {
            fixture.environment->reset();
            std::vector<core::Solution> inputs = fixture.input_templates;
            const auto begin = std::chrono::steady_clock::now();
            const std::size_t accepted = run_episode(
                *fixture.environment, inputs);
            const auto end = std::chrono::steady_clock::now();
            samples[sample] = std::chrono::duration<double, std::milli>(
                end - begin).count();
            require_same_output(
                expected,
                episode_output(fixture, accepted),
                "sample");
        }

        std::cout
            << "workers=" << workers
            << ";requests=" << request_count
            << ";events=" << event_count
            << ";accepted=" << expected.accepted
            << ";records=" << expected.records
            << ";checksum=" << expected.checksum
            << ";physical_checksum=" << expected.physical_checksum
            << ";output_bytes=" << expected.output_bytes
            << ";median_ms=" << median(samples)
            << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "benchmark")
        {
            benchmark(std::filesystem::path(argv[2]));
            return 0;
        }
        throw std::invalid_argument(
            "usage: environment_benchmark benchmark <temporary-root>");
    }
    catch (const std::exception& error)
    {
        std::cerr << "environment benchmark: " << error.what() << '\n';
        return 1;
    }
}
