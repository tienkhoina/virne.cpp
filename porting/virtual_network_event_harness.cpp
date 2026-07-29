#include "virtual_network_event.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace network = virne::network;

std::string hex_text(const std::string_view value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw : value) {
        const auto byte = static_cast<unsigned char>(raw);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string double_token(const double value)
{
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << "d:" << std::hex << std::setfill('0') << std::setw(16) << bits;
    return stream.str();
}

void emit(const std::string_view name, const std::string& payload)
{
    std::cout << "case=" << name << "|ok|" << hex_text(payload) << '\n';
}

template <typename Callable>
std::string error_payload(Callable&& callable)
{
    try {
        callable();
    } catch (const std::invalid_argument& error) {
        return std::string("error=") + error.what();
    }
    return "error=none";
}

std::vector<network::VirtualNetworkEventInput> batch_inputs()
{
    std::vector<network::VirtualNetworkEventInput> result;
    result.reserve(32U);
    for (std::size_t index = 0U; index < 32U; ++index) {
        result.push_back({
            index,
            index % 2U == 0U
                ? network::VirtualEventType::arrival
                : network::VirtualEventType::leave,
            static_cast<network::VirtualRequestId>(index / 2U),
            static_cast<double>((index * 37U) % 11U) +
                (index % 3U == 0U ? 0.25 : 0.0),
        });
    }
    return result;
}

std::string events_payload(
    const std::vector<network::VirtualNetworkEvent>& events)
{
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

void emit_cases()
{
    network::VirtualNetworkEvent ordinary(
        9U, network::VirtualEventType::arrival, 4, 2.5);
    emit(
        "ordinary",
        "id=" + std::to_string(ordinary.id()) +
        ";type=" + std::to_string(static_cast<unsigned int>(ordinary.type())) +
        ";vnet=" + std::to_string(ordinary.virtual_network_id()) +
        ";time=" + double_token(ordinary.time()) +
        ";repr=" + ordinary.repr());

    network::VirtualNetworkEvent negative_zero(
        0U, network::VirtualEventType::leave, 0, -0.0);
    emit(
        "negative_zero",
        "time=" + double_token(negative_zero.time()) +
        ";repr=" + negative_zero.repr());

    network::VirtualNetworkEvent infinite(
        1U, network::VirtualEventType::arrival, 2,
        std::numeric_limits<double>::infinity());
    emit(
        "infinity",
        "time=" + double_token(infinite.time()) +
        ";repr=" + infinite.repr());

    emit(
        "invalid_type",
        error_payload([] {
            network::VirtualNetworkEvent event(
                0U, static_cast<network::VirtualEventType>(2U), 0, 0.0);
            static_cast<void>(event);
        }));
    emit(
        "negative_vnet",
        error_payload([] {
            network::VirtualNetworkEvent event(
                0U, network::VirtualEventType::arrival, -1, 0.0);
            static_cast<void>(event);
        }));
    emit(
        "negative_time",
        error_payload([] {
            network::VirtualNetworkEvent event(
                0U, network::VirtualEventType::arrival, 0, -1.0);
            static_cast<void>(event);
        }));
    emit(
        "validation_order",
        error_payload([] {
            network::VirtualNetworkEvent event(
                0U, static_cast<network::VirtualEventType>(7U), -1, -1.0);
            static_cast<void>(event);
        }));

    network::VirtualNetworkEvent changed(
        5U, network::VirtualEventType::arrival, 9, 4.0);
    changed.set_id(11U);
    changed.set_type(network::VirtualEventType::leave);
    changed.set_virtual_network_id(3);
    changed.set_time(8.5);
    emit(
        "typed_setters",
        "id=" + std::to_string(changed.id()) +
        ";type=" + std::to_string(static_cast<unsigned int>(changed.type())) +
        ";vnet=" + std::to_string(changed.virtual_network_id()) +
        ";time=" + double_token(changed.time()));

    const auto inputs = batch_inputs();
    for (const std::size_t workers : {0U, 1U, 2U, 8U}) {
        auto events = network::make_virtual_network_events(inputs, workers);
        emit("batch_w" + std::to_string(workers), events_payload(events));
    }

    std::vector<network::VirtualNetworkEvent> ties;
    ties.emplace_back(0U, network::VirtualEventType::arrival, 0, 1.0);
    ties.emplace_back(1U, network::VirtualEventType::arrival, 1, 0.0);
    ties.emplace_back(2U, network::VirtualEventType::leave, 0, 1.0);
    ties.emplace_back(3U, network::VirtualEventType::leave, 1, 0.0);
    network::stable_sort_virtual_network_events(ties);
    emit("stable_ties", events_payload(ties));

    emit(
        "native_nan_rejected",
        error_payload([] {
            network::VirtualNetworkEvent event(
                0U, network::VirtualEventType::arrival, 0,
                std::numeric_limits<double>::quiet_NaN());
            static_cast<void>(event);
        }));
}

}  // namespace

int main()
{
    try {
        emit_cases();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VirtualNetworkEvent harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
