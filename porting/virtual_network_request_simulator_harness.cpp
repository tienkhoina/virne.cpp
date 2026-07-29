#include "virtual_network_request_simulator.h"

#include "random_context.h"

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
#include <utility>
#include <vector>

namespace {

namespace network = virne::network;
namespace utils = virne::utils;

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

void emit(const std::string_view name, const std::string& payload) {
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

network::SimulationDistribution integer_uniform(
    const std::int64_t low,
    const std::int64_t high) {
    network::SimulationDistribution result;
    result.value_kind = utils::DatasetValueKind::integer;
    result.distribution.kind = utils::DistributionKind::uniform;
    result.distribution.low = low;
    result.distribution.high = high;
    return result;
}

network::SimulationDistribution floating_uniform(
    const double low,
    const double high) {
    network::SimulationDistribution result;
    result.value_kind = utils::DatasetValueKind::floating;
    result.distribution.kind = utils::DistributionKind::uniform;
    result.distribution.low = low;
    result.distribution.high = high;
    return result;
}

network::SimulationDistribution floating_exponential(const double scale) {
    network::SimulationDistribution result;
    result.value_kind = utils::DatasetValueKind::floating;
    result.distribution.kind = utils::DistributionKind::exponential;
    result.distribution.scale = scale;
    return result;
}

network::SimulationDistribution poisson(
    const utils::DatasetValueKind value_kind,
    const double lambda) {
    network::SimulationDistribution result;
    result.value_kind = value_kind;
    result.distribution.kind = utils::DistributionKind::poisson;
    result.distribution.lambda = lambda;
    return result;
}

network::VirtualNetworkSimulationConfig make_config(
    const utils::DatasetValueKind arrival_kind,
    const bool with_max_latency,
    const std::size_t request_count = 5U) {
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = request_count;
    config.virtual_network_size = integer_uniform(2, 5);
    config.lifetime = floating_exponential(3.25);
    config.arrival_rate = poisson(arrival_kind, 1.25);
    if (with_max_latency) {
        config.max_latency = floating_uniform(0.25, 2.0);
    }
    config.topology_type = network::TopologyType::Path;
    return config;
}

network::VirtualNetworkSimulationConfig make_tie_config() {
    auto config = make_config(utils::DatasetValueKind::boolean, false, 3U);
    config.virtual_network_size = integer_uniform(2, 2);
    config.lifetime = poisson(utils::DatasetValueKind::floating, 0.0);
    config.arrival_rate = poisson(utils::DatasetValueKind::boolean, 0.0);
    return config;
}

std::string integer_vector_token(const std::vector<std::int64_t>& values) {
    std::string result = "[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += std::to_string(values[index]);
    }
    result.push_back(']');
    return result;
}

std::string double_vector_token(const std::vector<double>& values) {
    std::string result = "[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        result += double_token(values[index]);
    }
    result.push_back(']');
    return result;
}

std::string optional_event_token(
    const std::optional<network::VirtualEventId>& value) {
    return value ? std::to_string(*value) : "none";
}

std::string arrangement_payload(
    const network::VirtualNetworkRequestSimulator& simulator) {
    std::string result =
        "sizes=" + integer_vector_token(simulator.arranged_sizes()) +
        ";lifetimes=" + double_vector_token(simulator.arranged_lifetimes()) +
        ";arrivals=" + double_vector_token(simulator.arranged_arrival_times()) +
        ";max=";
    const auto& max_latencies = simulator.arranged_max_latencies();
    result += max_latencies ? double_vector_token(*max_latencies) : "none";
    return result;
}

std::string graph_payload(const Graph& graph) {
    std::string result = "N=[";
    auto [node, node_end] = graph.nodes();
    bool first = true;
    for (; node != node_end; ++node) {
        if (!first) {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(*node);
    }
    result += "];E=[";
    auto [edge, edge_end] = graph.edges();
    first = true;
    for (; edge != edge_end; ++edge) {
        if (!first) {
            result.push_back(',');
        }
        first = false;
        result += std::to_string(graph.source(*edge)) + "-" +
            std::to_string(graph.target(*edge));
    }
    result.push_back(']');
    return result;
}

std::string networks_payload(
    const std::vector<network::VirtualNetwork>& virtual_networks) {
    std::string result = "[";
    for (std::size_t index = 0U; index < virtual_networks.size(); ++index) {
        if (index != 0U) {
            result.push_back('/');
        }
        const auto& value = virtual_networks[index];
        if (!value.request_id() || !value.arrival_time() || !value.lifetime()) {
            throw std::runtime_error("generated virtual network fields are missing");
        }
        result += "id=" + std::to_string(*value.request_id()) +
            ",arrival=" + double_token(*value.arrival_time()) +
            ",lifetime=" + double_token(*value.lifetime()) +
            ",max=" +
            (value.max_latency() ? double_token(*value.max_latency()) : "none") +
            "," + graph_payload(value.graph());
    }
    result.push_back(']');
    return result;
}

std::string events_payload(
    const std::vector<network::VirtualNetworkEvent>& events) {
    std::string result = "[";
    for (std::size_t index = 0U; index < events.size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& event = events[index];
        result += std::to_string(event.id()) + ":" +
            std::to_string(static_cast<unsigned int>(event.type())) + ":" +
            std::to_string(event.virtual_network_id()) + ":" +
            double_token(event.time());
    }
    result.push_back(']');
    return result;
}

std::string lookup_payload(
    const network::VirtualNetworkRequestSimulator& simulator) {
    std::string result = "[";
    for (std::size_t index = 0U; index < simulator.v_nets().size(); ++index) {
        if (index != 0U) {
            result.push_back(',');
        }
        const auto& request_id = simulator.v_nets()[index].request_id();
        if (!request_id) {
            throw std::runtime_error("generated request id is missing");
        }
        result += std::to_string(*request_id) + ":" +
            optional_event_token(simulator.event_id(
                *request_id, network::VirtualEventType::arrival)) + ":" +
            optional_event_token(simulator.event_id(
                *request_id, network::VirtualEventType::leave));
    }
    result.push_back(']');
    return result;
}

std::string full_payload(
    const network::VirtualNetworkRequestSimulator& simulator,
    const std::uint32_t next_python,
    const std::uint32_t next_numpy) {
    return arrangement_payload(simulator) +
        ";num_vnets=" + std::to_string(simulator.num_v_nets()) +
        ";num_events=" + std::to_string(simulator.num_events()) +
        ";networks=" + networks_payload(simulator.v_nets()) +
        ";events=" + events_payload(simulator.events()) +
        ";lookup=" + lookup_payload(simulator) +
        ";next_py=" + std::to_string(next_python) +
        ";next_np=" + std::to_string(next_numpy);
}

network::VirtualSimulationWorkers all_workers(const std::size_t workers) {
    network::VirtualSimulationWorkers result;
    result.factory_workers = workers;
    result.arrangement_workers = workers;
    result.attribute_workers = workers;
    result.event_workers = workers;
    result.io_workers = workers;
    return result;
}

void emit_arrangement_cases() {
    struct CaseSpec {
        std::string_view name;
        utils::DatasetValueKind arrival_kind;
        bool with_max_latency;
        std::uint32_t seed;
    };
    const std::vector<CaseSpec> cases = {
        {"integer", utils::DatasetValueKind::integer, false, 701U},
        {"floating", utils::DatasetValueKind::floating, true, 702U},
        {"boolean", utils::DatasetValueKind::boolean, true, 703U},
    };
    for (const auto& spec : cases) {
        for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
            NumpyRandomState random(spec.seed);
            auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
                make_config(spec.arrival_kind, spec.with_max_latency));
            simulator.arrange_v_nets(random, workers);
            emit(
                "arrange_" + std::string(spec.name) + "_w" +
                    std::to_string(workers),
                arrangement_payload(simulator) +
                    ";next_np=" + std::to_string(random.next_uint32()));
        }
    }
}

void emit_renew_cases() {
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        RandomContext random(991U);
        auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
            make_config(utils::DatasetValueKind::integer, true));
        simulator.renew(random, true, true, std::nullopt, all_workers(workers));
        const auto next_python = random.python().getrandbits32();
        const auto next_numpy = random.numpy().next_uint32();
        emit(
            "renew_path_w" + std::to_string(workers),
            full_payload(simulator, next_python, next_numpy));
    }

    RandomContext tie_random(313U);
    auto tie_simulator = network::VirtualNetworkRequestSimulator::from_setting(
        make_tie_config());
    tie_simulator.renew(
        tie_random, true, true, std::nullopt, all_workers(8U));
    const auto next_python = tie_random.python().getrandbits32();
    const auto next_numpy = tie_random.numpy().next_uint32();
    emit(
        "tie_order",
        full_payload(tie_simulator, next_python, next_numpy));
}

network::VirtualNetworkRequestSimulator make_loaded_simulator() {
    std::vector<network::VirtualNetworkEvent> events;
    events.emplace_back(
        41U, network::VirtualEventType::arrival, 1'000'000, 3.0);
    events.emplace_back(
        42U, network::VirtualEventType::leave, 1'000'000, 4.0);
    events.emplace_back(
        43U, network::VirtualEventType::arrival, 1'000'000, 5.0);
    events.emplace_back(44U, network::VirtualEventType::leave, 7, 6.0);
    return network::VirtualNetworkRequestSimulator::from_state(
        make_config(utils::DatasetValueKind::integer, false, 0U),
        {},
        std::move(events));
}

void emit_loaded_state_case() {
    auto simulator = make_loaded_simulator();
    emit(
        "loaded_index",
        "num_vnets=" + std::to_string(simulator.num_v_nets()) +
            ";num_events=" + std::to_string(simulator.num_events()) +
            ";events=" + events_payload(simulator.events()) +
            ";sparse_arrival=" + optional_event_token(simulator.event_id(
                1'000'000, network::VirtualEventType::arrival)) +
            ";sparse_leave=" + optional_event_token(simulator.event_id(
                1'000'000, network::VirtualEventType::leave)) +
            ";id7_arrival=" + optional_event_token(simulator.event_id(
                7, network::VirtualEventType::arrival)) +
            ";id7_leave=" + optional_event_token(simulator.event_id(
                7, network::VirtualEventType::leave)));
}

std::string simulator_error_code_name(
    const network::VirtualNetworkSimulatorErrorCode code) {
    switch (code) {
    case network::VirtualNetworkSimulatorErrorCode::invalid_size_distribution:
        return "invalid_size_distribution";
    case network::VirtualNetworkSimulatorErrorCode::request_count_overflow:
        return "request_count_overflow";
    case network::VirtualNetworkSimulatorErrorCode::generated_lane_mismatch:
        return "generated_lane_mismatch";
    case network::VirtualNetworkSimulatorErrorCode::invalid_setting:
        return "invalid_setting";
    case network::VirtualNetworkSimulatorErrorCode::missing_setting_field:
        return "missing_setting_field";
    case network::VirtualNetworkSimulatorErrorCode::invalid_setting_value:
        return "invalid_setting_value";
    case network::VirtualNetworkSimulatorErrorCode::missing_source_setting:
        return "missing_source_setting";
    case network::VirtualNetworkSimulatorErrorCode::invalid_dataset_layout:
        return "invalid_dataset_layout";
    case network::VirtualNetworkSimulatorErrorCode::invalid_event_setting:
        return "invalid_event_setting";
    case network::VirtualNetworkSimulatorErrorCode::event_count_mismatch:
        return "event_count_mismatch";
    case network::VirtualNetworkSimulatorErrorCode::io_failure:
        return "io_failure";
    }
    return "unknown";
}

std::string simulator_operation_name(
    const network::VirtualNetworkSimulatorOperation operation) {
    switch (operation) {
    case network::VirtualNetworkSimulatorOperation::validate_config:
        return "validate_config";
    case network::VirtualNetworkSimulatorOperation::arrange_networks:
        return "arrange_networks";
    case network::VirtualNetworkSimulatorOperation::decode_setting:
        return "decode_setting";
    case network::VirtualNetworkSimulatorOperation::save_networks:
        return "save_networks";
    case network::VirtualNetworkSimulatorOperation::save_events:
        return "save_events";
    case network::VirtualNetworkSimulatorOperation::save_setting:
        return "save_setting";
    case network::VirtualNetworkSimulatorOperation::load_layout:
        return "load_layout";
    case network::VirtualNetworkSimulatorOperation::load_setting:
        return "load_setting";
    case network::VirtualNetworkSimulatorOperation::load_events:
        return "load_events";
    case network::VirtualNetworkSimulatorOperation::load_networks:
        return "load_networks";
    case network::VirtualNetworkSimulatorOperation::validate_dataset:
        return "validate_dataset";
    case network::VirtualNetworkSimulatorOperation::cache_dataset:
        return "cache_dataset";
    }
    return "unknown";
}

void emit_native_boundary_cases() {
    auto invalid_config = make_config(
        utils::DatasetValueKind::integer, false, 1U);
    invalid_config.virtual_network_size.value_kind =
        utils::DatasetValueKind::floating;
    try {
        auto ignored = network::VirtualNetworkRequestSimulator::from_setting(
            std::move(invalid_config));
        static_cast<void>(ignored);
        emit("native_invalid_size_kind", "error=none");
    } catch (const network::VirtualNetworkSimulatorException& error) {
        emit(
            "native_invalid_size_kind",
            "code=" + simulator_error_code_name(error.code()) +
                ";operation=" + simulator_operation_name(error.operation()));
    }

    auto simulator = make_loaded_simulator();
    emit(
        "native_invalid_enum_lookup",
        "id=" + optional_event_token(simulator.event_id(
            1'000'000, static_cast<network::VirtualEventType>(2U))));
}

void emit_cases() {
    emit_arrangement_cases();
    emit_renew_cases();
    emit_loaded_state_case();
    emit_native_boundary_cases();
}

}  // namespace

int main() {
    try {
        emit_cases();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkRequestSimulator harness: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
