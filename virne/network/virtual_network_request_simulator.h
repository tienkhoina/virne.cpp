#pragma once

#include "virtual_network.h"
#include "virtual_network_event.h"
#include "../utils/dataset.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class NumpyRandomState;
class RandomContext;

namespace virne::network {

struct SimulationDistribution {
    virne::utils::DatasetValueKind value_kind =
        virne::utils::DatasetValueKind::floating;
    virne::utils::DistributionSpec distribution;
};

struct VirtualSimulationOutput {
    std::optional<std::string> save_dir;
    std::string events_file_name = "events.yaml";
    std::string setting_file_name = "v_sim_setting.yaml";
};

struct VirtualNetworkSimulationConfig {
    std::size_t num_virtual_networks = 0U;
    SimulationDistribution virtual_network_size;
    SimulationDistribution lifetime;
    SimulationDistribution arrival_rate;
    std::optional<SimulationDistribution> max_latency;
    TopologyType topology_type = TopologyType::Random;
    TopologyOptions topology_options;
    std::vector<attribute::AttributeFactorySpec> node_attribute_specs;
    std::vector<attribute::AttributeFactorySpec> link_attribute_specs;
    std::optional<AttrValue> topology_metadata;
    std::optional<AttrValue> output_metadata;
    VirtualSimulationOutput output;
    std::optional<virne::utils::SettingDocument> source_setting;
};

struct VirtualSimulationWorkers {
    std::size_t factory_workers = 1U;
    std::size_t arrangement_workers = 1U;
    std::size_t attribute_workers = 1U;
    std::size_t event_workers = 1U;
    std::size_t io_workers = 1U;
};

enum class VirtualNetworkSimulatorErrorCode : std::uint8_t {
    invalid_size_distribution,
    request_count_overflow,
    generated_lane_mismatch,
    invalid_setting,
    missing_setting_field,
    invalid_setting_value,
    missing_source_setting,
    invalid_dataset_layout,
    invalid_event_setting,
    event_count_mismatch,
    io_failure,
};

enum class VirtualNetworkSimulatorOperation : std::uint8_t {
    validate_config,
    arrange_networks,
    decode_setting,
    save_networks,
    save_events,
    save_setting,
    load_layout,
    load_setting,
    load_events,
    load_networks,
    validate_dataset,
    cache_dataset,
};

class VirtualNetworkSimulatorException : public std::runtime_error {
public:
    VirtualNetworkSimulatorException(
        VirtualNetworkSimulatorErrorCode code,
        VirtualNetworkSimulatorOperation operation,
        std::string message,
        std::optional<std::size_t> input_index = std::nullopt,
        std::filesystem::path path = {});

    VirtualNetworkSimulatorErrorCode code() const noexcept;
    VirtualNetworkSimulatorOperation operation() const noexcept;
    const std::optional<std::size_t>& input_index() const noexcept;
    const std::filesystem::path& path() const noexcept;

private:
    VirtualNetworkSimulatorErrorCode code_;
    VirtualNetworkSimulatorOperation operation_;
    std::optional<std::size_t> input_index_;
    std::filesystem::path path_;
};

VirtualNetworkSimulationConfig virtual_network_simulation_config_from_setting(
    const virne::utils::SettingDocument& setting);

class VirtualNetworkDatasetCache;

class VirtualNetworkRequestSimulator {
public:
    static VirtualNetworkRequestSimulator from_setting(
        VirtualNetworkSimulationConfig config);
    static VirtualNetworkRequestSimulator from_setting(
        const virne::utils::SettingDocument& setting,
        std::optional<std::uint32_t> seed = std::nullopt);
    static VirtualNetworkRequestSimulator from_setting(
        const virne::utils::SettingDocument& setting,
        RandomContext& random,
        std::optional<std::uint32_t> seed = std::nullopt);

    // Cold compatibility/load boundary. Events retain their supplied order
    // and IDs; only the compact numeric lookup index is rebuilt.
    static VirtualNetworkRequestSimulator from_state(
        VirtualNetworkSimulationConfig config,
        std::vector<VirtualNetwork> virtual_networks,
        std::vector<VirtualNetworkEvent> events);

    VirtualNetworkRequestSimulator(
        const VirtualNetworkRequestSimulator&) = delete;
    VirtualNetworkRequestSimulator& operator=(
        const VirtualNetworkRequestSimulator&) = delete;
    VirtualNetworkRequestSimulator(
        VirtualNetworkRequestSimulator&&) = default;
    VirtualNetworkRequestSimulator& operator=(
        VirtualNetworkRequestSimulator&&) = default;

    void renew(
        RandomContext& random,
        bool renew_virtual_networks = true,
        bool renew_event_schedule = true,
        std::optional<std::uint32_t> seed = std::nullopt,
        const VirtualSimulationWorkers& workers = {});

    void arrange_v_nets(
        NumpyRandomState& random,
        std::size_t workers = 1U);
    void renew_v_nets(
        RandomContext& random,
        const VirtualSimulationWorkers& workers = {});
    void renew_events(std::size_t workers = 1U);

    const VirtualNetworkSimulationConfig& config() const noexcept;
    std::size_t num_v_nets() const noexcept;
    std::size_t num_events() const noexcept;
    const std::vector<VirtualNetwork>& v_nets() const noexcept;
    const std::vector<VirtualNetworkEvent>& events() const noexcept;

    // Cold ownership-transfer boundary for orchestration such as the
    // four-stage changeable workload. Requiring an expiring rvalue prevents
    // a live simulator from retaining event indices into released storage.
    std::vector<VirtualNetwork> release_v_nets() && noexcept;

    const std::vector<std::int64_t>& arranged_sizes() const noexcept;
    const std::vector<double>& arranged_lifetimes() const noexcept;
    const std::vector<double>& arranged_arrival_times() const noexcept;
    const std::optional<std::vector<double>>& arranged_max_latencies()
        const noexcept;

    std::optional<VirtualEventId> event_id(
        VirtualRequestId virtual_network_id,
        VirtualEventType type) const noexcept;

    void save_setting(const std::filesystem::path& path) const;
    void save_dataset(
        const std::filesystem::path& directory,
        std::size_t workers = 1U) const;
    static VirtualNetworkRequestSimulator load_dataset(
        const std::filesystem::path& directory,
        VirtualNetworkDatasetCache& cache,
        std::size_t workers = 1U);
    static VirtualNetworkRequestSimulator load_dataset(
        const std::filesystem::path& directory,
        std::size_t workers = 1U);

    VirtualNetworkRequestSimulator clone() const;

private:
    using EventPair = std::array<std::optional<VirtualEventId>, 2U>;

    VirtualNetworkRequestSimulator(
        VirtualNetworkSimulationConfig config,
        std::vector<VirtualNetwork> virtual_networks,
        std::vector<VirtualNetworkEvent> events);

    void rebuild_event_index();

    VirtualNetworkSimulationConfig config_;
    std::vector<VirtualNetwork> virtual_networks_;
    std::vector<VirtualNetworkEvent> events_;

    std::vector<std::int64_t> arranged_sizes_;
    std::vector<double> arranged_lifetimes_;
    std::vector<double> arranged_arrival_times_;
    std::optional<std::vector<double>> arranged_max_latencies_;

    // Generated dense request IDs use two direct slots. Loaded/sparse IDs
    // cross one numeric hash boundary, never a string boundary.
    std::vector<std::optional<VirtualEventId>> dense_event_ids_;
    std::unordered_map<VirtualRequestId, EventPair> sparse_event_ids_;
};

class VirtualNetworkDatasetCache {
public:
    VirtualNetworkDatasetCache() = default;
    VirtualNetworkDatasetCache(const VirtualNetworkDatasetCache&) = delete;
    VirtualNetworkDatasetCache& operator=(
        const VirtualNetworkDatasetCache&) = delete;

    std::optional<VirtualNetworkRequestSimulator> find(
        const std::string& raw_directory) const;
    void store(
        std::string raw_directory,
        const VirtualNetworkRequestSimulator& simulator);
    std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<
        std::string,
        std::shared_ptr<const VirtualNetworkRequestSimulator>> entries_;
};

VirtualNetworkDatasetCache& global_virtual_network_dataset_cache();

}  // namespace virne::network
