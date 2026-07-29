#include "environment.h"

#include "attribute/attribute_factory.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

namespace attribute = virne::network::attribute;
namespace controller = virne::core::controller;
namespace core = virne::core;
namespace network = virne::network;
namespace utils = virne::utils;

using core::EnvironmentAdmissionPolicy;
using core::EnvironmentFailureReason;
using core::EnvironmentPhase;
using core::EnvironmentStepResult;
using core::EnvironmentWorkers;
using core::RecorderRecord;
using core::Solution;
using core::SolutionMetadata;
using core::SolutionStepEnvironment;

struct PhysicalFixture
{
    network::PhysicalNetwork network;
    AttrId cpu_value_id = 0U;
    AttrId bandwidth_value_id = 0U;
};

struct EnvironmentFixture
{
    std::unique_ptr<SolutionStepEnvironment> environment;
    AttrId cpu_value_id = 0U;
    AttrId bandwidth_value_id = 0U;
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

PhysicalFixture make_physical_network()
{
    Graph graph(2U, std::vector<EdgeEndpoints>{{0U, 1U}});
    network::PhysicalNetwork result(construction(std::move(graph)));
    const auto cpu = result.bind_node_attribute("cpu");
    const auto bandwidth = result.bind_link_attribute("bandwidth");
    if (!cpu.has_value() || !bandwidth.has_value())
    {
        throw std::runtime_error("environment fixture binding failure");
    }
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {100.0, 200.0}}});
    result.set_link_attrs_data({network::LinkAttributeDataUpdate{
        bandwidth->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {300.0}}});
    return PhysicalFixture{
        std::move(result), cpu->value_id, bandwidth->value_id};
}

network::VirtualNetwork make_virtual_network(
    const network::VirtualRequestId request_id,
    const double arrival_time)
{
    Graph graph(1U, std::vector<EdgeEndpoints>{});
    network::VirtualNetwork result(construction(std::move(graph)));
    const auto cpu = result.bind_node_attribute("cpu");
    if (!cpu.has_value())
    {
        throw std::runtime_error("environment virtual binding failure");
    }
    result.set_node_attrs_data({network::NodeAttributeDataUpdate{
        cpu->registry_id,
        network::AttributeDataLayout::dense,
        {},
        {10.0}}});
    result.set_request_id(request_id);
    result.set_arrival_time(arrival_time);
    result.set_lifetime(1.0);
    return result;
}

network::VirtualNetworkRequestSimulator make_simulator(
    const std::size_t request_count)
{
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = request_count;
    config.virtual_network_size.value_kind = utils::DatasetValueKind::integer;
    config.virtual_network_size.distribution.kind =
        utils::DistributionKind::uniform;
    config.virtual_network_size.distribution.low = std::int64_t{1};
    config.virtual_network_size.distribution.high = std::int64_t{2};
    config.lifetime.value_kind = utils::DatasetValueKind::floating;
    config.lifetime.distribution.kind = utils::DistributionKind::uniform;
    config.lifetime.distribution.low = 1.0;
    config.lifetime.distribution.high = 2.0;
    config.arrival_rate.value_kind = utils::DatasetValueKind::floating;
    config.arrival_rate.distribution.kind = utils::DistributionKind::uniform;
    config.arrival_rate.distribution.low = 1.0;
    config.arrival_rate.distribution.high = 2.0;
    config.topology_type = network::TopologyType::Path;
    config.node_attribute_specs.push_back(resource_spec(
        "cpu", attribute::AttributeOwner::node));
    config.link_attribute_specs.push_back(resource_spec(
        "bandwidth", attribute::AttributeOwner::link));

    std::vector<network::VirtualNetwork> virtual_networks;
    virtual_networks.reserve(request_count);
    for (std::size_t index = 0U; index < request_count; ++index)
    {
        virtual_networks.push_back(make_virtual_network(
            static_cast<network::VirtualRequestId>(index),
            static_cast<double>(index) * 2.0));
    }

    std::vector<network::VirtualNetworkEvent> events;
    events.reserve(request_count * 2U);
    for (std::size_t index = 0U; index < request_count; ++index)
    {
        const auto request_id =
            static_cast<network::VirtualRequestId>(index);
        const double time = static_cast<double>(index) * 2.0;
        events.emplace_back(
            index * 2U,
            network::VirtualEventType::arrival,
            request_id,
            time);
        events.emplace_back(
            index * 2U + 1U,
            network::VirtualEventType::leave,
            request_id,
            time + 1.0);
    }
    return network::VirtualNetworkRequestSimulator::from_state(
        std::move(config),
        std::move(virtual_networks),
        std::move(events));
}

core::EnvironmentConfig make_config(
    const std::filesystem::path& root,
    std::string case_name,
    const std::size_t workers,
    const EnvironmentAdmissionPolicy admission = {})
{
    core::EnvironmentConfig config;
    config.controller.node_resources = {0U};
    config.controller.link_resources = {0U};
    config.counter.node_resources = std::vector<core::CounterResourceId>{0U};
    config.counter.link_resources = std::vector<core::CounterResourceId>{0U};
    config.recorder.save_root_dir = root;
    config.recorder.solver_name = std::move(case_name);
    config.recorder.run_id = "run";
    config.recorder.record_dir_name = "records";
    config.recorder.temporary_records = false;
    config.workers = EnvironmentWorkers{workers, workers, workers};
    config.admission = admission;
    return config;
}

EnvironmentFixture make_environment(
    const std::filesystem::path& root,
    std::string case_name,
    const std::size_t request_count,
    const std::size_t workers,
    const EnvironmentAdmissionPolicy admission = {})
{
    PhysicalFixture physical = make_physical_network();
    auto environment = std::make_unique<SolutionStepEnvironment>(
        std::move(physical.network),
        make_simulator(request_count),
        make_config(
            root,
            std::move(case_name),
            workers,
            admission));
    environment->reset();
    return EnvironmentFixture{
        std::move(environment),
        physical.cpu_value_id,
        physical.bandwidth_value_id};
}

Solution accepted_solution(SolutionStepEnvironment& environment)
{
    Solution solution = environment.make_solution();
    solution.result = true;
    solution.node_slots.insert_or_assign(0, 0);
    core::SolutionAttributeValues values;
    values.set(0U, 10.0);
    solution.node_slots_info.insert_or_assign(
        core::NodeSlotInfoKey{0, 0},
        std::move(values));
    return solution;
}

Solution rejected_solution(
    SolutionStepEnvironment& environment,
    const EnvironmentFailureReason reason)
{
    Solution solution = environment.make_solution();
    solution.result = false;
    switch (reason)
    {
    case EnvironmentFailureReason::early_rejection:
        solution.early_rejection = true;
        solution.place_result = false;
        solution.route_result = false;
        break;
    case EnvironmentFailureReason::placement:
        solution.place_result = false;
        solution.route_result = false;
        break;
    case EnvironmentFailureReason::routing:
        solution.route_result = false;
        break;
    case EnvironmentFailureReason::unknown:
    case EnvironmentFailureReason::none:
        break;
    }
    return solution;
}

std::string hex_text(const std::string_view value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.resize(value.size() * 2U);
    for (std::size_t index = 0U; index < value.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(value[index]);
        result[index * 2U] = digits[byte >> 4U];
        result[index * 2U + 1U] = digits[byte & 0x0fU];
    }
    return result;
}

std::string double_token(const double value)
{
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "double width drift");
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
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
    throw std::runtime_error("environment nonnumeric resource");
}

std::string phase_token(const EnvironmentPhase phase)
{
    switch (phase)
    {
    case EnvironmentPhase::unready:
        return "unready";
    case EnvironmentPhase::active:
        return "active";
    case EnvironmentPhase::finished:
        return "finished";
    }
    return "invalid";
}

std::string failure_token(const EnvironmentFailureReason reason)
{
    switch (reason)
    {
    case EnvironmentFailureReason::none:
        return "none";
    case EnvironmentFailureReason::early_rejection:
        return "early";
    case EnvironmentFailureReason::placement:
        return "place";
    case EnvironmentFailureReason::routing:
        return "route";
    case EnvironmentFailureReason::unknown:
        return "unknown";
    }
    return "invalid";
}

std::string record_token(const RecorderRecord& record)
{
    if (!record.state.event.has_value())
    {
        throw std::runtime_error("environment record missing event");
    }
    const auto& event = *record.state.event;
    return
        std::to_string(event.event_id) + "," +
        std::to_string(static_cast<unsigned int>(event.type)) + "," +
        std::to_string(record.solution.v_net_id) + "," +
        (record.solution.result ? "1" : "0") + "," +
        hex_text(record.solution.description);
}

std::string step_token(const EnvironmentStepResult& step)
{
    return
        "accepted=" + std::string(step.accepted ? "1" : "0") +
        ",reason=" + failure_token(step.failure_reason) +
        ",done=" + (step.done ? "1" : "0") +
        ",auto=" + std::to_string(step.auto_released_events) +
        ",summary=" + (step.summary.has_value() ? "1" : "0") +
        ",record=" + record_token(step.record);
}

std::string snapshot_token(
    const SolutionStepEnvironment& environment,
    const AttrId cpu_value_id,
    const AttrId bandwidth_value_id)
{
    const auto& environment_state = environment.state();
    const auto& recorder = environment.recorder();
    const auto& recorder_state = recorder.state();

    std::string current = "none";
    if (environment_state.phase == EnvironmentPhase::active &&
        environment_state.current_event.has_value())
    {
        const auto& event = *environment_state.current_event;
        current =
            std::to_string(event.schedule_index) + "," +
            std::to_string(event.event_id) + "," +
            std::to_string(static_cast<unsigned int>(event.type)) + "," +
            std::to_string(event.request_id) + "," +
            std::to_string(event.request_index) + "," +
            double_token(event.time);
    }

    const auto& graph = environment.physical_network().graph();
    const double node_zero = numeric_value(
        graph.node_attrs(0U).at(cpu_value_id));
    const double node_one = numeric_value(
        graph.node_attrs(1U).at(cpu_value_id));
    const double link = numeric_value(
        graph.edge_attrs(graph.edge(0U, 1U)).at(bandwidth_value_id));

    std::string records;
    for (const auto& record : recorder.memory())
    {
        if (!records.empty())
        {
            records.push_back('|');
        }
        records += record_token(record);
    }
    return
        "phase=" + phase_token(environment_state.phase) +
        ",processed=" +
            std::to_string(environment_state.num_processed_events) +
        ",current=" + current +
        ",counts=" +
            std::to_string(recorder_state.virtual_network_count) + "," +
            std::to_string(recorder_state.success_count) + "," +
            std::to_string(recorder_state.inservice_count) +
        ",resources=" + double_token(node_zero) + "," +
            double_token(node_one) + "," + double_token(link) +
        ",memory=" + records;
}

std::string mixed_case(
    const std::filesystem::path& root,
    const std::size_t workers)
{
    EnvironmentFixture fixture = make_environment(
        root,
        "mixed-w" + std::to_string(workers),
        2U,
        workers);
    auto& environment = *fixture.environment;
    Solution first = accepted_solution(environment);
    const EnvironmentStepResult first_result = environment.step(first);
    const std::string first_payload =
        step_token(first_result) + ";" + snapshot_token(
            environment, fixture.cpu_value_id, fixture.bandwidth_value_id);

    Solution second = rejected_solution(
        environment, EnvironmentFailureReason::placement);
    const EnvironmentStepResult second_result = environment.step(second);
    return
        "first{" + first_payload + "};second{" +
        step_token(second_result) + ";" + snapshot_token(
            environment, fixture.cpu_value_id, fixture.bandwidth_value_id) +
        "}";
}

std::string rejection_case(
    const std::filesystem::path& root,
    std::string case_name,
    const EnvironmentFailureReason reason,
    const bool hard_violation,
    const bool admission_rejection)
{
    EnvironmentAdmissionPolicy admission;
    if (admission_rejection)
    {
        admission.r2c_ratio_threshold = 0.5;
        admission.virtual_network_size_threshold = 0U;
    }
    EnvironmentFixture fixture = make_environment(
        root,
        std::move(case_name),
        1U,
        1U,
        admission);
    auto& environment = *fixture.environment;
    Solution solution = admission_rejection || hard_violation
        ? environment.make_solution()
        : rejected_solution(environment, reason);
    if (admission_rejection || hard_violation)
    {
        solution.result = true;
    }
    if (hard_violation)
    {
        solution.v_net_total_hard_constraint_violation = 1.0;
    }
    const EnvironmentStepResult result = environment.step(solution);
    return step_token(result) + ";" + snapshot_token(
        environment, fixture.cpu_value_id, fixture.bandwidth_value_id);
}

void emit(const std::string_view name, const std::string& payload)
{
    std::cout << name << '\t' << payload << '\n';
}

void differential(const std::filesystem::path& root)
{
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        emit(
            "mixed_workers_" + std::to_string(workers),
            mixed_case(root, workers));
    }
    emit("reject_early", rejection_case(
        root, "reject-early", EnvironmentFailureReason::early_rejection,
        false, false));
    emit("reject_place", rejection_case(
        root, "reject-place", EnvironmentFailureReason::placement,
        false, false));
    emit("reject_route", rejection_case(
        root, "reject-route", EnvironmentFailureReason::routing,
        false, false));
    emit("reject_unknown", rejection_case(
        root, "reject-unknown", EnvironmentFailureReason::unknown,
        false, false));
    emit("reject_hard", rejection_case(
        root, "reject-hard", EnvironmentFailureReason::unknown,
        true, false));
    emit("reject_admission", rejection_case(
        root, "reject-admission", EnvironmentFailureReason::unknown,
        false, true));
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "differential")
        {
            differential(std::filesystem::path(argv[2]));
            return 0;
        }
        throw std::invalid_argument(
            "usage: environment_harness differential <temporary-root>");
    }
    catch (const std::exception& error)
    {
        std::cerr << "environment harness: " << error.what() << '\n';
        return 1;
    }
}
