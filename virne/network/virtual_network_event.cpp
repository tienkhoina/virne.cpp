#include "virtual_network_event.h"

#include "dataset.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

namespace virne::network {
namespace {

void validate_type(const VirtualEventType value)
{
    if (value != VirtualEventType::leave &&
        value != VirtualEventType::arrival) {
        throw std::invalid_argument(
            "Event type must be 0 (leave) or 1 (arrival)");
    }
}

void validate_virtual_network_id(const VirtualRequestId value)
{
    if (value < 0) {
        throw std::invalid_argument("Virtual network ID must be non-negative");
    }
}

void validate_time(const double value)
{
    if (value < 0.0 || std::isnan(value)) {
        throw std::invalid_argument("Event time must be non-negative");
    }
}

struct IndexedError {
    std::size_t index = 0U;
    std::exception_ptr error;
};

}  // namespace

VirtualNetworkEvent::VirtualNetworkEvent(VirtualNetworkEventInput input)
    : VirtualNetworkEvent(
          input.id,
          input.type,
          input.virtual_network_id,
          input.time)
{
}

VirtualNetworkEvent::VirtualNetworkEvent(
    const VirtualEventId id,
    const VirtualEventType type,
    const VirtualRequestId virtual_network_id,
    const double time)
    : id_(id),
      type_(type),
      virtual_network_id_(virtual_network_id),
      time_(time)
{
    validate_type(type_);
    validate_virtual_network_id(virtual_network_id_);
    validate_time(time_);
}

void VirtualNetworkEvent::set_type(const VirtualEventType value)
{
    validate_type(value);
    type_ = value;
}

void VirtualNetworkEvent::set_virtual_network_id(const VirtualRequestId value)
{
    validate_virtual_network_id(value);
    virtual_network_id_ = value;
}

void VirtualNetworkEvent::set_time(const double value)
{
    validate_time(value);
    time_ = value;
}

std::string VirtualNetworkEvent::repr() const
{
    return
        "VirtualNetworkEvent(v_net_id=" +
        std::to_string(virtual_network_id_) +
        ", time=" + virne::utils::format_dataset_scalar(time_) +
        ", type=" +
        std::to_string(static_cast<unsigned int>(type_)) +
        ", id=" + std::to_string(id_) + ")";
}

std::vector<VirtualNetworkEvent> make_virtual_network_events(
    const std::vector<VirtualNetworkEventInput>& inputs,
    const std::size_t workers)
{
    if (inputs.empty()) {
        return {};
    }
    const VirtualNetworkEvent empty(
        0U, VirtualEventType::leave, 0, 0.0);
    std::vector<VirtualNetworkEvent> result(inputs.size(), empty);
    if (workers <= 1U || inputs.size() == 1U) {
        for (std::size_t index = 0U; index < inputs.size(); ++index) {
            result[index] = VirtualNetworkEvent(inputs[index]);
        }
        return result;
    }

    const std::size_t lane_count = std::min(workers, inputs.size());
    std::vector<std::thread> threads;
    threads.reserve(lane_count);
    std::vector<IndexedError> errors(lane_count);
    const std::size_t base = inputs.size() / lane_count;
    const std::size_t remainder = inputs.size() % lane_count;
    std::size_t begin = 0U;
    for (std::size_t lane = 0U; lane < lane_count; ++lane) {
        const std::size_t width = base + (lane < remainder ? 1U : 0U);
        const std::size_t end = begin + width;
        threads.emplace_back([&, lane, begin, end] {
            for (std::size_t index = begin; index < end; ++index) {
                try {
                    result[index] = VirtualNetworkEvent(inputs[index]);
                } catch (...) {
                    errors[lane].index = index;
                    errors[lane].error = std::current_exception();
                    return;
                }
            }
        });
        begin = end;
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    const IndexedError* first = nullptr;
    for (const IndexedError& error : errors) {
        if (error.error && (first == nullptr || error.index < first->index)) {
            first = &error;
        }
    }
    if (first != nullptr) {
        std::rethrow_exception(first->error);
    }
    return result;
}

void stable_sort_virtual_network_events(
    std::vector<VirtualNetworkEvent>& events)
{
    std::stable_sort(
        events.begin(), events.end(),
        [](const VirtualNetworkEvent& lhs, const VirtualNetworkEvent& rhs) {
            return lhs.time() < rhs.time();
        });
}

}  // namespace virne::network
