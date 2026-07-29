#pragma once

#include "base_network.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace virne::network {

class VirtualNetwork final : public BaseNetwork {
public:
    VirtualNetwork();
    explicit VirtualNetwork(BaseNetworkConstruction construction);
    explicit VirtualNetwork(BaseNetwork&& network);

    VirtualNetwork(const VirtualNetwork&) = delete;
    VirtualNetwork& operator=(const VirtualNetwork&) = delete;
    VirtualNetwork(VirtualNetwork&& other);
    VirtualNetwork& operator=(VirtualNetwork&& other);

    const std::optional<std::int64_t>& request_id() const noexcept;
    const std::optional<double>& arrival_time() const noexcept;
    const std::optional<double>& lifetime() const noexcept;
    const std::optional<double>& max_latency() const noexcept;

    void set_request_id(std::int64_t value) noexcept;
    void set_arrival_time(double value) noexcept;
    void set_lifetime(double value) noexcept;
    void set_max_latency(double value) noexcept;

    double total_node_resource_demand(
        std::size_t workers = 1U) const noexcept;
    double total_link_resource_demand(
        std::size_t workers = 1U) const noexcept;
    double total_resource_demand(
        std::size_t workers = 1U) const noexcept;
    void invalidate_cached_total_resource_demand() noexcept;

    void to_gml(const std::string& path) const;
    VirtualNetwork clone() const;

private:
    void initialize_fixed_fields_from_metadata();
    double resource_demand(bool node, std::size_t workers) const;

    std::optional<std::int64_t> request_id_;
    std::optional<double> arrival_time_;
    std::optional<double> lifetime_;
    std::optional<double> max_latency_;
    mutable std::optional<double> cached_total_resource_demand_;
};

}  // namespace virne::network
