#include "virtual_network_event.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace network = virne::network;

void expect(const bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void expect_error(
    Callable&& callable,
    const std::string& fragment,
    const std::string& context)
{
    try {
        callable();
    } catch (const std::invalid_argument& error) {
        expect(
            std::string(error.what()).find(fragment) != std::string::npos,
            context + ": diagnostic drift");
        return;
    }
    throw std::runtime_error(context + ": expected invalid_argument");
}

void test_scalar_and_validation()
{
    network::VirtualNetworkEvent value(
        9U, network::VirtualEventType::arrival, 4, 2.5);
    expect(value.id() == 9U, "id");
    expect(value.type() == network::VirtualEventType::arrival, "type");
    expect(value.virtual_network_id() == 4, "virtual id");
    expect(value.time() == 2.5, "time");
    expect(
        value.repr() ==
            "VirtualNetworkEvent(v_net_id=4, time=2.5, type=1, id=9)",
        "repr");

    value.set_id(std::numeric_limits<std::size_t>::max());
    value.set_type(network::VirtualEventType::leave);
    value.set_virtual_network_id(0);
    value.set_time(-0.0);
    expect(value.id() == std::numeric_limits<std::size_t>::max(), "max id");
    expect(std::signbit(value.time()), "negative zero preserved");
    expect(
        value.repr().find("time=-0.0") != std::string::npos,
        "negative zero repr");

    network::VirtualNetworkEvent infinite(
        0U, network::VirtualEventType::arrival, 0,
        std::numeric_limits<double>::infinity());
    expect(infinite.repr().find("time=inf") != std::string::npos, "infinity");

    expect_error(
        [] {
            network::VirtualNetworkEvent value(
                0U, static_cast<network::VirtualEventType>(2U), -1, -1.0);
            static_cast<void>(value);
        },
        "Event type", "validation order type");
    expect_error(
        [] {
            network::VirtualNetworkEvent value(
                0U, network::VirtualEventType::arrival, -1, -1.0);
            static_cast<void>(value);
        },
        "Virtual network ID", "validation order id");
    expect_error(
        [] {
            network::VirtualNetworkEvent value(
                0U, network::VirtualEventType::arrival, 0, -1.0);
            static_cast<void>(value);
        },
        "Event time", "negative time");
    expect_error(
        [] {
            network::VirtualNetworkEvent value(
                0U, network::VirtualEventType::arrival, 0,
                std::numeric_limits<double>::quiet_NaN());
            static_cast<void>(value);
        },
        "Event time", "native NaN rejection");

    network::VirtualNetworkEvent unchanged(
        3U, network::VirtualEventType::arrival, 5, 7.0);
    expect_error(
        [&] { unchanged.set_virtual_network_id(-3); },
        "Virtual network ID", "setter id");
    expect(unchanged.virtual_network_id() == 5, "setter id atomicity");
    expect_error(
        [&] { unchanged.set_time(-2.0); },
        "Event time", "setter time");
    expect(unchanged.time() == 7.0, "setter time atomicity");
}

std::vector<network::VirtualNetworkEventInput> inputs(const std::size_t count)
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
            static_cast<double>((index * 37U) % 31U),
        });
    }
    return result;
}

void test_batch_and_sort()
{
    const auto source = inputs(4096U);
    const auto baseline = network::make_virtual_network_events(source, 0U);
    for (const std::size_t workers : {1U, 2U, 8U}) {
        expect(
            network::make_virtual_network_events(source, workers) == baseline,
            "batch worker invariance=" + std::to_string(workers));
    }

    auto invalid = source;
    invalid[100U].virtual_network_id = -1;
    invalid[3000U].type = static_cast<network::VirtualEventType>(7U);
    expect_error(
        [&] {
            static_cast<void>(
                network::make_virtual_network_events(invalid, 8U));
        },
        "Virtual network ID", "lowest input error");

    std::vector<network::VirtualNetworkEvent> ties;
    ties.emplace_back(0U, network::VirtualEventType::arrival, 0, 1.0);
    ties.emplace_back(1U, network::VirtualEventType::arrival, 1, 0.0);
    ties.emplace_back(2U, network::VirtualEventType::leave, 0, 1.0);
    ties.emplace_back(3U, network::VirtualEventType::leave, 1, 0.0);
    network::stable_sort_virtual_network_events(ties);
    const std::array<std::size_t, 4U> expected{1U, 3U, 0U, 2U};
    for (std::size_t index = 0U; index < ties.size(); ++index) {
        expect(ties[index].id() == expected[index], "stable tie order");
    }

    std::array<std::future<std::vector<network::VirtualNetworkEvent>>, 4U> jobs;
    const std::array<std::size_t, 4U> worker_counts{0U, 1U, 2U, 8U};
    for (std::size_t index = 0U; index < jobs.size(); ++index) {
        jobs[index] = std::async(
            std::launch::async,
            [source, workers = worker_counts[index]] {
                auto events = network::make_virtual_network_events(
                    source, workers);
                network::stable_sort_virtual_network_events(events);
                return events;
            });
    }
    auto sorted_baseline = baseline;
    network::stable_sort_virtual_network_events(sorted_baseline);
    for (auto& job : jobs) {
        expect(job.get() == sorted_baseline, "concurrent independent batch");
    }
}

}  // namespace

int main()
{
    try {
        test_scalar_and_validation();
        test_batch_and_sort();
        std::cout << "VirtualNetworkEvent unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkEvent unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
