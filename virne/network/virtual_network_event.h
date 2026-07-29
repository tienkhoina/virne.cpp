#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace virne::network {

using VirtualRequestId = std::int64_t;
using VirtualEventId = std::size_t;

enum class VirtualEventType : std::uint8_t {
    leave = 0,
    arrival = 1,
};

struct VirtualNetworkEventInput {
    VirtualEventId id = 0U;
    VirtualEventType type = VirtualEventType::leave;
    VirtualRequestId virtual_network_id = 0;
    double time = 0.0;
};

class VirtualNetworkEvent {
public:
    explicit VirtualNetworkEvent(VirtualNetworkEventInput input);
    VirtualNetworkEvent(
        VirtualEventId id,
        VirtualEventType type,
        VirtualRequestId virtual_network_id,
        double time);

    VirtualEventId id() const noexcept { return id_; }
    VirtualEventType type() const noexcept { return type_; }
    VirtualRequestId virtual_network_id() const noexcept {
        return virtual_network_id_;
    }
    double time() const noexcept { return time_; }

    void set_id(VirtualEventId value) noexcept { id_ = value; }
    void set_type(VirtualEventType value);
    void set_virtual_network_id(VirtualRequestId value);
    void set_time(double value);

    std::string repr() const;

    friend bool operator==(
        const VirtualNetworkEvent& lhs,
        const VirtualNetworkEvent& rhs) noexcept {
        return lhs.id_ == rhs.id_ && lhs.type_ == rhs.type_ &&
            lhs.virtual_network_id_ == rhs.virtual_network_id_ &&
            lhs.time_ == rhs.time_;
    }

private:
    VirtualEventId id_ = 0U;
    VirtualEventType type_ = VirtualEventType::leave;
    VirtualRequestId virtual_network_id_ = 0;
    double time_ = 0.0;
};

std::vector<VirtualNetworkEvent> make_virtual_network_events(
    const std::vector<VirtualNetworkEventInput>& inputs,
    std::size_t workers = 1U);

void stable_sort_virtual_network_events(
    std::vector<VirtualNetworkEvent>& events);

}  // namespace virne::network
