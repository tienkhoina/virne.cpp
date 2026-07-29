#include "virtual_network.h"

#include "io/graph_saver.h"

#include <stdexcept>
#include <utility>
#include <variant>

namespace virne::network {
namespace {

std::optional<std::int64_t> metadata_integer(
    const Graph& graph,
    const AttrMap& metadata,
    const std::string_view name) {
    const auto id = graph.attribute_registry().find(name);
    if (!id.has_value()) {
        return std::nullopt;
    }
    const AttrValue* const value = metadata.find(*id);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (const auto* integer = std::get_if<std::int64_t>(value)) {
        return *integer;
    }
    if (const auto* boolean = std::get_if<bool>(value)) {
        return *boolean ? 1 : 0;
    }
    return std::nullopt;
}

std::optional<double> metadata_double(
    const Graph& graph,
    const AttrMap& metadata,
    const std::string_view name) {
    const auto id = graph.attribute_registry().find(name);
    if (!id.has_value()) {
        return std::nullopt;
    }
    const AttrValue* const value = metadata.find(*id);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (const auto* real = std::get_if<double>(value)) {
        return *real;
    }
    if (const auto* integer = std::get_if<std::int64_t>(value)) {
        return static_cast<double>(*integer);
    }
    if (const auto* boolean = std::get_if<bool>(value)) {
        return *boolean ? 1.0 : 0.0;
    }
    return std::nullopt;
}

}  // namespace

VirtualNetwork::VirtualNetwork() : BaseNetwork() {
    initialize_fixed_fields_from_metadata();
}

VirtualNetwork::VirtualNetwork(BaseNetworkConstruction construction)
    : BaseNetwork(std::move(construction)) {
    initialize_fixed_fields_from_metadata();
}

VirtualNetwork::VirtualNetwork(BaseNetwork&& network)
    : BaseNetwork(std::move(network)) {
    initialize_fixed_fields_from_metadata();
}

VirtualNetwork::VirtualNetwork(VirtualNetwork&& other)
    : BaseNetwork(std::move(other)),
      request_id_(other.request_id_),
      arrival_time_(other.arrival_time_),
      lifetime_(other.lifetime_),
      max_latency_(other.max_latency_),
      cached_total_resource_demand_(other.cached_total_resource_demand_) {}

VirtualNetwork& VirtualNetwork::operator=(VirtualNetwork&& other) {
    if (this == &other) {
        return *this;
    }
    BaseNetwork::operator=(std::move(other));
    request_id_ = other.request_id_;
    arrival_time_ = other.arrival_time_;
    lifetime_ = other.lifetime_;
    max_latency_ = other.max_latency_;
    cached_total_resource_demand_ = other.cached_total_resource_demand_;
    return *this;
}

const std::optional<std::int64_t>& VirtualNetwork::request_id() const noexcept {
    return request_id_;
}

const std::optional<double>& VirtualNetwork::arrival_time() const noexcept {
    return arrival_time_;
}

const std::optional<double>& VirtualNetwork::lifetime() const noexcept {
    return lifetime_;
}

const std::optional<double>& VirtualNetwork::max_latency() const noexcept {
    return max_latency_;
}

void VirtualNetwork::set_request_id(const std::int64_t value) noexcept {
    request_id_ = value;
}

void VirtualNetwork::set_arrival_time(const double value) noexcept {
    arrival_time_ = value;
}

void VirtualNetwork::set_lifetime(const double value) noexcept {
    lifetime_ = value;
}

void VirtualNetwork::set_max_latency(const double value) noexcept {
    max_latency_ = value;
}

double VirtualNetwork::resource_demand(
    const bool node,
    const std::size_t workers) const {
    AttributeSelection selection;
    selection.kinds = std::vector<attribute::AttributeKind>{
        attribute::AttributeKind::resource};
    const auto definitions = node
        ? select_node_attributes(selection)
        : select_link_attributes(selection);
    const auto rows = node
        ? get_node_attrs_data(*this, definitions, workers)
        : get_link_attrs_data(*this, definitions, workers);

    const std::size_t columns = rows.front().size();
    bool floating = false;
    for (const auto& row : rows) {
        if (row.size() != columns) {
            throw std::invalid_argument("Virtual resource rows are ragged");
        }
        for (const AttrValue& value : row) {
            if (std::holds_alternative<double>(value)) {
                floating = true;
            } else if (!std::holds_alternative<std::int64_t>(value) &&
                       !std::holds_alternative<bool>(value)) {
                throw std::invalid_argument(
                    "Virtual resource value is not numeric");
            }
        }
    }

    if (floating) {
        double result = 0.0;
        for (const auto& row : rows) {
            for (const AttrValue& value : row) {
                result += attr_to_double(value);
            }
        }
        return result;
    }

    std::uint64_t result = 0U;
    for (const auto& row : rows) {
        for (const AttrValue& value : row) {
            if (const auto* integer = std::get_if<std::int64_t>(&value)) {
                result += static_cast<std::uint64_t>(*integer);
            } else {
                result += std::get<bool>(value) ? 1U : 0U;
            }
        }
    }
    return static_cast<double>(static_cast<std::int64_t>(result));
}

double VirtualNetwork::total_node_resource_demand(
    const std::size_t workers) const noexcept {
    try {
        return resource_demand(true, workers);
    } catch (...) {
        return 0.0;
    }
}

double VirtualNetwork::total_link_resource_demand(
    const std::size_t workers) const noexcept {
    try {
        return resource_demand(false, workers);
    } catch (...) {
        return 0.0;
    }
}

double VirtualNetwork::total_resource_demand(
    const std::size_t workers) const noexcept {
    if (!cached_total_resource_demand_.has_value()) {
        try {
            cached_total_resource_demand_ =
                resource_demand(true, workers) +
                resource_demand(false, workers);
        } catch (...) {
            cached_total_resource_demand_ = 0.0;
        }
    }
    return *cached_total_resource_demand_;
}

void VirtualNetwork::invalidate_cached_total_resource_demand() noexcept {
    cached_total_resource_demand_.reset();
}

void VirtualNetwork::to_gml(const std::string& path) const {
    nx::write_gml(prepare_gml_graph(), path);
}

VirtualNetwork VirtualNetwork::clone() const {
    VirtualNetwork result(BaseNetwork::clone());
    result.request_id_ = request_id_;
    result.arrival_time_ = arrival_time_;
    result.lifetime_ = lifetime_;
    result.max_latency_ = max_latency_;
    result.cached_total_resource_demand_ = cached_total_resource_demand_;
    return result;
}

void VirtualNetwork::initialize_fixed_fields_from_metadata() {
    request_id_ = metadata_integer(graph(), graph_attributes(), "id");
    arrival_time_ = metadata_double(graph(), graph_attributes(), "arrival_time");
    lifetime_ = metadata_double(graph(), graph_attributes(), "lifetime");
    max_latency_ = metadata_double(graph(), graph_attributes(), "max_latency");
}

}  // namespace virne::network
