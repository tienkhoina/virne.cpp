#include "virtual_network_event.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

namespace network = virne::network;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void add_byte(std::uint64_t& checksum, const std::uint8_t value)
{
    checksum = (checksum ^ value) * fnv_prime;
}

void add_u64(std::uint64_t& checksum, const std::uint64_t value)
{
    for (std::size_t shift = 0U; shift < 64U; shift += 8U) {
        add_byte(checksum, static_cast<std::uint8_t>(value >> shift));
    }
}

std::vector<network::VirtualNetworkEventInput> make_inputs(
    const std::size_t count)
{
    std::vector<network::VirtualNetworkEventInput> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        result.push_back({
            index,
            index % 2U == 0U
                ? network::VirtualEventType::arrival
                : network::VirtualEventType::leave,
            static_cast<network::VirtualRequestId>(index / 2U),
            static_cast<double>((index * 37U) % 1009U) +
                (index % 3U == 0U ? 0.25 : 0.0),
        });
    }
    return result;
}

std::uint64_t fingerprint(
    const std::vector<network::VirtualNetworkEvent>& events)
{
    std::uint64_t checksum = fnv_offset;
    for (const auto& event : events) {
        add_u64(checksum, event.id());
        add_byte(checksum, static_cast<std::uint8_t>(event.type()));
        add_u64(
            checksum,
            static_cast<std::uint64_t>(event.virtual_network_id()));
        std::uint64_t time_bits = 0U;
        const double time = event.time();
        std::memcpy(&time_bits, &time, sizeof(time_bits));
        add_u64(checksum, time_bits);
    }
    return checksum;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: virtual_network_event_benchmark <count> <workers>");
        }
        const auto count = static_cast<std::size_t>(std::stoull(argv[1]));
        const auto workers = static_cast<std::size_t>(std::stoull(argv[2]));
        if (count == 0U) {
            throw std::invalid_argument("event benchmark count must be positive");
        }
        const auto inputs = make_inputs(count);
        const auto begin = std::chrono::steady_clock::now();
        auto events = network::make_virtual_network_events(inputs, workers);
        network::stable_sort_virtual_network_events(events);
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin).count();
        std::cout
            << "protocol=1\n"
            << "kind=virtual_network_event_construct_sort\n"
            << "count=" << count << '\n'
            << "workers=" << workers << '\n'
            << "elapsed_ns=" << elapsed << '\n'
            << "checksum=" << fingerprint(events) << '\n'
            << "output_bytes=" << count * 25U << '\n'
            << "entry_count=" << count << '\n'
            << "type_tag=event_le_v1\n"
            << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkEvent benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
