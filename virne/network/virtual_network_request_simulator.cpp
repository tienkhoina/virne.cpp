#include "virtual_network_request_simulator.h"
#include "virtual_network_request_simulator_detail.h"

#include "random_context.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace virne::network {
namespace {

std::size_t explicit_workers(std::size_t workers) noexcept {
    return workers == 0U ? 1U : workers;
}

[[noreturn]] void throw_lane_mismatch() {
    throw VirtualNetworkSimulatorException(
        VirtualNetworkSimulatorErrorCode::generated_lane_mismatch,
        VirtualNetworkSimulatorOperation::arrange_networks,
        "generated distribution lane does not match its fixed value kind");
}

std::vector<double> generated_values_as_doubles(
    virne::utils::GeneratedData values) {
    switch (values.value_kind) {
    case virne::utils::DatasetValueKind::integer:
    {
        auto* source = std::get_if<std::vector<std::int64_t>>(&values.values);
        if (source == nullptr) {
            throw_lane_mismatch();
        }
        std::vector<double> result(source->size());
        std::transform(
            source->begin(),
            source->end(),
            result.begin(),
            [](std::int64_t value) { return static_cast<double>(value); });
        return result;
    }
    case virne::utils::DatasetValueKind::floating:
    {
        auto* source = std::get_if<std::vector<double>>(&values.values);
        if (source == nullptr) {
            throw_lane_mismatch();
        }
        return std::move(*source);
    }
    case virne::utils::DatasetValueKind::boolean:
    {
        auto* source = std::get_if<std::vector<std::uint8_t>>(&values.values);
        if (source == nullptr) {
            throw_lane_mismatch();
        }
        std::vector<double> result(source->size());
        std::transform(
            source->begin(),
            source->end(),
            result.begin(),
            [](std::uint8_t value) { return value == 0U ? 0.0 : 1.0; });
        return result;
    }
    }
    throw_lane_mismatch();
}

std::int64_t signed_from_modular_bits(std::uint64_t value) noexcept {
    constexpr std::uint64_t sign_bit = std::uint64_t{1} << 63U;
    if (value < sign_bit) {
        return static_cast<std::int64_t>(value);
    }
    return std::numeric_limits<std::int64_t>::min() +
        static_cast<std::int64_t>(value - sign_bit);
}

std::vector<double> cumulative_arrival_times(
    virne::utils::GeneratedData intervals) {
    switch (intervals.value_kind) {
    case virne::utils::DatasetValueKind::integer:
    {
        const auto* source =
            std::get_if<std::vector<std::int64_t>>(&intervals.values);
        if (source == nullptr) {
            throw_lane_mismatch();
        }
        std::vector<double> result(source->size());
        std::uint64_t total = 0U;
        for (std::size_t index = 0U; index < source->size(); ++index) {
            total += static_cast<std::uint64_t>((*source)[index]);
            result[index] =
                static_cast<double>(signed_from_modular_bits(total));
        }
        return result;
    }
    case virne::utils::DatasetValueKind::floating:
    {
        const auto* source = std::get_if<std::vector<double>>(&intervals.values);
        if (source == nullptr) {
            throw_lane_mismatch();
        }
        std::vector<double> result(source->size());
        double total = 0.0;
        for (std::size_t index = 0U; index < source->size(); ++index) {
            total += (*source)[index];
            result[index] = total;
        }
        return result;
    }
    case virne::utils::DatasetValueKind::boolean:
    {
        const auto* source =
            std::get_if<std::vector<std::uint8_t>>(&intervals.values);
        if (source == nullptr) {
            throw_lane_mismatch();
        }
        std::vector<double> result(source->size());
        std::int64_t total = 0;
        for (std::size_t index = 0U; index < source->size(); ++index) {
            total += (*source)[index] == 0U ? 0 : 1;
            result[index] = static_cast<double>(total);
        }
        return result;
    }
    }
    throw_lane_mismatch();
}

virne::utils::DistributionRequest distribution_request(
    std::size_t count,
    const SimulationDistribution& distribution) {
    return {
        count,
        distribution.value_kind,
        distribution.distribution,
    };
}

struct SettingCloneContext {
    std::unordered_map<
        const virne::utils::SettingList*,
        virne::utils::SettingListPtr> lists;
    std::unordered_map<
        const virne::utils::SettingObject*,
        virne::utils::SettingObjectPtr> objects;
};

virne::utils::SettingValue clone_setting_value(
    const virne::utils::SettingValue& value,
    SettingCloneContext& context) {
    using virne::utils::SettingKeyId;
    using virne::utils::SettingValue;
    using virne::utils::SettingValueKind;

    switch (value.kind()) {
    case SettingValueKind::null_value:
        return SettingValue{};
    case SettingValueKind::boolean:
        return SettingValue{value.as_bool()};
    case SettingValueKind::integer:
        return SettingValue{value.as_integer()};
    case SettingValueKind::real:
        return SettingValue{value.as_real()};
    case SettingValueKind::string:
        return SettingValue{value.as_string()};
    case SettingValueKind::list:
    {
        const auto& source_pointer = value.list_ptr();
        const auto found = context.lists.find(source_pointer.get());
        if (found != context.lists.end()) {
            return SettingValue{found->second};
        }
        auto destination = std::make_shared<virne::utils::SettingList>();
        context.lists.emplace(source_pointer.get(), destination);
        destination->reserve(source_pointer->size());
        for (const SettingValue& item : *source_pointer) {
            destination->push_back(clone_setting_value(item, context));
        }
        return SettingValue{std::move(destination)};
    }
    case SettingValueKind::object:
    {
        const auto& source_pointer = value.object_ptr();
        const auto found = context.objects.find(source_pointer.get());
        if (found != context.objects.end()) {
            return SettingValue{found->second};
        }
        auto destination = std::make_shared<virne::utils::SettingObject>();
        context.objects.emplace(source_pointer.get(), destination);
        destination->reserve(source_pointer->size());
        for (std::size_t index = 0U; index < source_pointer->size(); ++index) {
            const SettingKeyId id{static_cast<std::uint32_t>(index)};
            destination->set_owned(
                std::string(source_pointer->key_name(id)),
                clone_setting_value(source_pointer->at(id), context));
        }
        return SettingValue{std::move(destination)};
    }
    }
    throw std::logic_error("invalid setting value kind");
}

virne::utils::SettingDocument clone_setting_document_impl(
    const virne::utils::SettingDocument& source) {
    SettingCloneContext context;
    return {clone_setting_value(source.root, context)};
}

VirtualNetworkSimulationConfig clone_config(
    const VirtualNetworkSimulationConfig& source) {
    VirtualNetworkSimulationConfig result = source;
    if (source.topology_metadata) {
        result.topology_metadata = clone_attr_value(*source.topology_metadata);
    }
    if (source.output_metadata) {
        result.output_metadata = clone_attr_value(*source.output_metadata);
    }
    if (source.source_setting) {
        result.source_setting =
            detail::clone_setting_document(*source.source_setting);
    }
    return result;
}

void validate_config(const VirtualNetworkSimulationConfig& config) {
    if (config.virtual_network_size.value_kind !=
        virne::utils::DatasetValueKind::integer) {
        throw VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::invalid_size_distribution,
            VirtualNetworkSimulatorOperation::validate_config,
            "virtual-network size distribution must produce int64 values");
    }
    const std::size_t maximum_request_count = std::min(
        static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max()),
        std::numeric_limits<std::size_t>::max() / 2U);
    if (config.num_virtual_networks > maximum_request_count) {
        throw VirtualNetworkSimulatorException(
            VirtualNetworkSimulatorErrorCode::request_count_overflow,
            VirtualNetworkSimulatorOperation::validate_config,
            "virtual-network count cannot be represented by request/event IDs");
    }
}

std::optional<std::size_t> checked_event_type_slot(
    VirtualEventType type) noexcept {
    switch (type) {
    case VirtualEventType::leave:
        return 0U;
    case VirtualEventType::arrival:
        return 1U;
    }
    return std::nullopt;
}

}  // namespace

namespace detail {

virne::utils::SettingDocument clone_setting_document(
    const virne::utils::SettingDocument& source) {
    return clone_setting_document_impl(source);
}

}  // namespace detail

VirtualNetworkSimulatorException::VirtualNetworkSimulatorException(
    VirtualNetworkSimulatorErrorCode code,
    VirtualNetworkSimulatorOperation operation,
    std::string message,
    std::optional<std::size_t> input_index,
    std::filesystem::path path)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index),
      path_(std::move(path)) {}

VirtualNetworkSimulatorErrorCode
VirtualNetworkSimulatorException::code() const noexcept {
    return code_;
}

VirtualNetworkSimulatorOperation
VirtualNetworkSimulatorException::operation() const noexcept {
    return operation_;
}

const std::optional<std::size_t>&
VirtualNetworkSimulatorException::input_index() const noexcept {
    return input_index_;
}

const std::filesystem::path&
VirtualNetworkSimulatorException::path() const noexcept {
    return path_;
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::from_setting(
    VirtualNetworkSimulationConfig config) {
    return VirtualNetworkRequestSimulator{
        std::move(config),
        {},
        {}};
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::from_state(
    VirtualNetworkSimulationConfig config,
    std::vector<VirtualNetwork> virtual_networks,
    std::vector<VirtualNetworkEvent> events) {
    return VirtualNetworkRequestSimulator{
        std::move(config),
        std::move(virtual_networks),
        std::move(events)};
}

VirtualNetworkRequestSimulator::VirtualNetworkRequestSimulator(
    VirtualNetworkSimulationConfig config,
    std::vector<VirtualNetwork> virtual_networks,
    std::vector<VirtualNetworkEvent> events)
    : config_(std::move(config)),
      virtual_networks_(std::move(virtual_networks)),
      events_(std::move(events)) {
    validate_config(config_);
    rebuild_event_index();
}

void VirtualNetworkRequestSimulator::renew(
    RandomContext& random,
    bool renew_virtual_networks,
    bool renew_event_schedule,
    std::optional<std::uint32_t> seed,
    const VirtualSimulationWorkers& workers) {
    random.set_seed(seed);
    if (renew_virtual_networks) {
        renew_v_nets(random, workers);
    }
    if (renew_event_schedule) {
        renew_events(workers.event_workers);
    }
}

void VirtualNetworkRequestSimulator::arrange_v_nets(
    NumpyRandomState& random,
    std::size_t workers) {
    const std::size_t count = config_.num_virtual_networks;
    const std::size_t cast_workers = explicit_workers(workers);

    virne::utils::GeneratedData sizes =
        virne::utils::generate_data_with_distribution(
            distribution_request(count, config_.virtual_network_size),
            random,
            cast_workers);
    auto* size_values =
        std::get_if<std::vector<std::int64_t>>(&sizes.values);
    if (sizes.value_kind != virne::utils::DatasetValueKind::integer ||
        size_values == nullptr) {
        throw_lane_mismatch();
    }
    arranged_sizes_ = std::move(*size_values);

    arranged_lifetimes_ = generated_values_as_doubles(
        virne::utils::generate_data_with_distribution(
            distribution_request(count, config_.lifetime),
            random,
            cast_workers));

    arranged_arrival_times_ = cumulative_arrival_times(
        virne::utils::generate_data_with_distribution(
            distribution_request(count, config_.arrival_rate),
            random,
            cast_workers));

    if (config_.max_latency) {
        arranged_max_latencies_ = generated_values_as_doubles(
            virne::utils::generate_data_with_distribution(
                distribution_request(count, *config_.max_latency),
                random,
                cast_workers));
    } else {
        arranged_max_latencies_.reset();
    }
}

void VirtualNetworkRequestSimulator::renew_v_nets(
    RandomContext& random,
    const VirtualSimulationWorkers& workers) {
    arrange_v_nets(random.numpy(), workers.arrangement_workers);

    std::vector<VirtualNetwork> generated;
    generated.reserve(config_.num_virtual_networks);
    for (std::size_t index = 0U;
         index < config_.num_virtual_networks;
         ++index) {
        BaseNetworkConstruction construction;
        construction.config.node_attribute_specs =
            config_.node_attribute_specs;
        construction.config.link_attribute_specs =
            config_.link_attribute_specs;
        if (config_.topology_metadata) {
            construction.config.topology =
                clone_attr_value(*config_.topology_metadata);
        }
        if (config_.output_metadata) {
            construction.config.output =
                clone_attr_value(*config_.output_metadata);
        }
        construction.config.factory_workers = workers.factory_workers;
        construction.config.graph_attributes = {
            {"id", static_cast<std::int64_t>(index)},
            {"arrival_time", arranged_arrival_times_[index]},
            {"lifetime", arranged_lifetimes_[index]},
        };

        VirtualNetwork virtual_network{std::move(construction)};
        if (arranged_max_latencies_) {
            const double value = (*arranged_max_latencies_)[index];
            virtual_network.set_max_latency(value);
            const AttrId id =
                virtual_network.bind_graph_attribute("max_latency");
            virtual_network.set_graph_attribute(id, value);
        }

        TopologyRequest topology_request;
        topology_request.type = config_.topology_type;
        topology_request.num_nodes = arranged_sizes_[index];
        topology_request.options = config_.topology_options;
        virtual_network.generate_topology(
            topology_request,
            random.python());
        virtual_network.generate_attrs_data(
            random.numpy(),
            true,
            true,
            workers.attribute_workers);
        generated.push_back(std::move(virtual_network));
    }
    virtual_networks_ = std::move(generated);
    // `renew(..., networks=true, events=false)` intentionally preserves the
    // old event schedule. Reclassify its numeric keys against the new dense
    // request range so event_id() remains safe and retains that stale mapping.
    rebuild_event_index();
}

void VirtualNetworkRequestSimulator::renew_events(std::size_t workers) {
    const std::size_t count = virtual_networks_.size();
    std::vector<VirtualNetworkEventInput> inputs(count * 2U);
    for (std::size_t index = 0U; index < count; ++index) {
        const VirtualNetwork& virtual_network = virtual_networks_[index];
        const VirtualRequestId request_id = virtual_network.request_id()
            .value_or(static_cast<VirtualRequestId>(index));
        const double arrival = virtual_network.arrival_time().value_or(0.0);
        const double lifetime = virtual_network.lifetime().value_or(0.0);
        inputs[index] = {
            0U,
            VirtualEventType::arrival,
            request_id,
            arrival,
        };
        inputs[count + index] = {
            0U,
            VirtualEventType::leave,
            request_id,
            arrival + lifetime,
        };
    }

    std::vector<VirtualNetworkEvent> generated =
        make_virtual_network_events(inputs, workers);
    stable_sort_virtual_network_events(generated);
    for (std::size_t index = 0U; index < generated.size(); ++index) {
        generated[index].set_id(index);
    }
    events_ = std::move(generated);
    rebuild_event_index();
}

const VirtualNetworkSimulationConfig&
VirtualNetworkRequestSimulator::config() const noexcept {
    return config_;
}

std::size_t VirtualNetworkRequestSimulator::num_v_nets() const noexcept {
    return virtual_networks_.size();
}

std::size_t VirtualNetworkRequestSimulator::num_events() const noexcept {
    return events_.size();
}

const std::vector<VirtualNetwork>&
VirtualNetworkRequestSimulator::v_nets() const noexcept {
    return virtual_networks_;
}

const std::vector<VirtualNetworkEvent>&
VirtualNetworkRequestSimulator::events() const noexcept {
    return events_;
}

std::vector<VirtualNetwork>
VirtualNetworkRequestSimulator::release_v_nets() && noexcept {
    return std::move(virtual_networks_);
}

const std::vector<std::int64_t>&
VirtualNetworkRequestSimulator::arranged_sizes() const noexcept {
    return arranged_sizes_;
}

const std::vector<double>&
VirtualNetworkRequestSimulator::arranged_lifetimes() const noexcept {
    return arranged_lifetimes_;
}

const std::vector<double>&
VirtualNetworkRequestSimulator::arranged_arrival_times() const noexcept {
    return arranged_arrival_times_;
}

const std::optional<std::vector<double>>&
VirtualNetworkRequestSimulator::arranged_max_latencies() const noexcept {
    return arranged_max_latencies_;
}

std::optional<VirtualEventId> VirtualNetworkRequestSimulator::event_id(
    VirtualRequestId virtual_network_id,
    VirtualEventType type) const noexcept {
    const std::optional<std::size_t> slot = checked_event_type_slot(type);
    if (!slot) {
        return std::nullopt;
    }
    if (virtual_network_id >= 0 &&
        static_cast<std::uint64_t>(virtual_network_id) <
            static_cast<std::uint64_t>(virtual_networks_.size())) {
        const std::size_t offset =
            static_cast<std::size_t>(virtual_network_id) * 2U +
            *slot;
        return dense_event_ids_[offset];
    }
    const auto found = sparse_event_ids_.find(virtual_network_id);
    if (found == sparse_event_ids_.end()) {
        return std::nullopt;
    }
    return found->second[*slot];
}

VirtualNetworkRequestSimulator
VirtualNetworkRequestSimulator::clone() const {
    std::vector<VirtualNetwork> virtual_networks;
    virtual_networks.reserve(virtual_networks_.size());
    for (const VirtualNetwork& virtual_network : virtual_networks_) {
        virtual_networks.push_back(virtual_network.clone());
    }

    VirtualNetworkRequestSimulator result{
        clone_config(config_),
        std::move(virtual_networks),
        events_};
    result.arranged_sizes_ = arranged_sizes_;
    result.arranged_lifetimes_ = arranged_lifetimes_;
    result.arranged_arrival_times_ = arranged_arrival_times_;
    result.arranged_max_latencies_ = arranged_max_latencies_;
    return result;
}

void VirtualNetworkRequestSimulator::rebuild_event_index() {
    std::vector<std::optional<VirtualEventId>> dense(
        virtual_networks_.size() * 2U);
    std::unordered_map<VirtualRequestId, EventPair> sparse;
    sparse.reserve(events_.size());

    for (const VirtualNetworkEvent& event : events_) {
        const VirtualRequestId request_id = event.virtual_network_id();
        const std::size_t slot = *checked_event_type_slot(event.type());
        if (request_id >= 0 &&
            static_cast<std::uint64_t>(request_id) <
                static_cast<std::uint64_t>(virtual_networks_.size())) {
            dense[static_cast<std::size_t>(request_id) * 2U + slot] =
                event.id();
        } else {
            sparse[request_id][slot] = event.id();
        }
    }

    dense_event_ids_ = std::move(dense);
    sparse_event_ids_ = std::move(sparse);
}

}  // namespace virne::network
