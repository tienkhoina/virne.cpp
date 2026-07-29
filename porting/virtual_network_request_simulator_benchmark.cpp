#include "virtual_network_request_simulator.h"

#include "numpy_random_state.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace network = virne::network;
namespace utils = virne::utils;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

enum class BenchmarkKind : std::uint8_t {
    arrange,
    schedule,
};

BenchmarkKind benchmark_kind_from_string(const std::string_view value) {
    if (value == "arrange") {
        return BenchmarkKind::arrange;
    }
    if (value == "schedule") {
        return BenchmarkKind::schedule;
    }
    throw std::invalid_argument("benchmark kind must be arrange or schedule");
}

void add_byte(std::uint64_t& checksum, const std::uint8_t value) {
    checksum = (checksum ^ value) * fnv_prime;
}

void add_u32(std::uint64_t& checksum, const std::uint32_t value) {
    for (std::size_t shift = 0U; shift < 32U; shift += 8U) {
        add_byte(checksum, static_cast<std::uint8_t>(value >> shift));
    }
}

void add_u64(std::uint64_t& checksum, const std::uint64_t value) {
    for (std::size_t shift = 0U; shift < 64U; shift += 8U) {
        add_byte(checksum, static_cast<std::uint8_t>(value >> shift));
    }
}

void add_double(std::uint64_t& checksum, const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    add_u64(checksum, bits);
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

network::SimulationDistribution floating_exponential(const double scale) {
    network::SimulationDistribution result;
    result.value_kind = utils::DatasetValueKind::floating;
    result.distribution.kind = utils::DistributionKind::exponential;
    result.distribution.scale = scale;
    return result;
}

network::SimulationDistribution floating_poisson(const double lambda) {
    network::SimulationDistribution result;
    result.value_kind = utils::DatasetValueKind::floating;
    result.distribution.kind = utils::DistributionKind::poisson;
    result.distribution.lambda = lambda;
    return result;
}

network::VirtualNetworkSimulationConfig make_config(const std::size_t count) {
    network::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = count;
    config.virtual_network_size = integer_uniform(2, 5);
    config.lifetime = floating_exponential(3.25);
    config.arrival_rate = floating_poisson(1.25);
    config.topology_type = network::TopologyType::Path;
    return config;
}

std::uint64_t arrangement_fingerprint(
    const network::VirtualNetworkRequestSimulator& simulator,
    const std::uint32_t next_numpy) {
    std::uint64_t checksum = fnv_offset;
    for (const std::int64_t value : simulator.arranged_sizes()) {
        add_u64(checksum, static_cast<std::uint64_t>(value));
    }
    for (const double value : simulator.arranged_lifetimes()) {
        add_double(checksum, value);
    }
    for (const double value : simulator.arranged_arrival_times()) {
        add_double(checksum, value);
    }
    add_u32(checksum, next_numpy);
    return checksum;
}

std::uint64_t event_fingerprint(
    const std::vector<network::VirtualNetworkEvent>& events) {
    std::uint64_t checksum = fnv_offset;
    for (const network::VirtualNetworkEvent& event : events) {
        add_u64(checksum, event.id());
        add_byte(checksum, static_cast<std::uint8_t>(event.type()));
        add_u64(
            checksum,
            static_cast<std::uint64_t>(event.virtual_network_id()));
        add_double(checksum, event.time());
    }
    return checksum;
}

std::vector<network::VirtualNetwork> make_virtual_networks(
    const std::size_t count) {
    std::vector<network::VirtualNetwork> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        network::VirtualNetwork virtual_network;
        virtual_network.set_request_id(static_cast<std::int64_t>(index));
        virtual_network.set_arrival_time(
            static_cast<double>((index * 37U) % 4096U) +
            (index % 3U == 0U ? 0.25 : 0.0));
        virtual_network.set_lifetime(
            static_cast<double>((index * 13U) % 127U) + 1.0);
        result.push_back(std::move(virtual_network));
    }
    return result;
}

void emit_result(
    const std::string_view kind,
    const std::size_t count,
    const std::size_t workers,
    const std::int64_t elapsed_ns,
    const std::uint64_t checksum,
    const std::size_t output_bytes,
    const std::size_t entry_count) {
    std::cout
        << "protocol=1\n"
        << "kind=" << kind << '\n'
        << "count=" << count << '\n'
        << "workers=" << workers << '\n'
        << "elapsed_ns=" << elapsed_ns << '\n'
        << "checksum=" << checksum << '\n'
        << "output_bytes=" << output_bytes << '\n'
        << "entry_count=" << entry_count << '\n'
        << "type_tag=simulator_le_v1\n"
        << "status=PASS\n";
}

void run_arrangement(const std::size_t count, const std::size_t workers) {
    auto simulator = network::VirtualNetworkRequestSimulator::from_setting(
        make_config(count));
    NumpyRandomState random(991U);
    const auto begin = std::chrono::steady_clock::now();
    simulator.arrange_v_nets(random, workers);
    const auto end = std::chrono::steady_clock::now();
    const std::uint32_t next_numpy = random.next_uint32();
    emit_result(
        "simulator_arrange",
        count,
        workers,
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count(),
        arrangement_fingerprint(simulator, next_numpy),
        count * 24U + 4U,
        count);
}

void run_schedule(const std::size_t count, const std::size_t workers) {
    auto simulator = network::VirtualNetworkRequestSimulator::from_state(
        make_config(count), make_virtual_networks(count), {});
    const auto begin = std::chrono::steady_clock::now();
    simulator.renew_events(workers);
    const auto end = std::chrono::steady_clock::now();
    emit_result(
        "simulator_schedule",
        count,
        workers,
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count(),
        event_fingerprint(simulator.events()),
        simulator.events().size() * 25U,
        simulator.events().size());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: virtual_network_request_simulator_benchmark "
                "<arrange|schedule> <count> <workers>");
        }
        const BenchmarkKind kind = benchmark_kind_from_string(argv[1]);
        const auto count = static_cast<std::size_t>(std::stoull(argv[2]));
        const auto workers = static_cast<std::size_t>(std::stoull(argv[3]));
        if (count == 0U) {
            throw std::invalid_argument("simulator benchmark count must be positive");
        }
        if (kind == BenchmarkKind::arrange) {
            run_arrangement(count, workers);
        } else {
            run_schedule(count, workers);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkRequestSimulator benchmark: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
