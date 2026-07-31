#pragma once

#include "counter.h"
#include "../network/physical_network.h"
#include "../utils/class_dict.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace virne::core {

struct RecorderConfig {
    std::filesystem::path save_root_dir;
    std::string solver_name;
    std::string run_id;
    std::string record_dir_name = "records";
    bool temporary_records = true;
};

struct RecorderOptions {
    std::size_t workers = 1U;
};

struct RecorderEvent {
    std::int64_t event_id = 0;
    network::VirtualEventType type = network::VirtualEventType::leave;
};

struct RecorderInitialPhysicalState {
    double available_resource = 0.0;
    double node_available_resource = 0.0;
    double link_available_resource = 0.0;
};

struct RecorderState {
    std::optional<RecorderEvent> event;
    std::int64_t virtual_network_count = 0;
    std::int64_t success_count = 0;
    std::int64_t inservice_count = 0;
    double total_revenue = 0.0;
    double total_cost = 0.0;
    double total_time_revenue = 0.0;
    double total_time_cost = 0.0;
    double long_term_r2c_ratio = 0.0;
    double long_term_time_r2c_ratio = 0.0;
    std::size_t running_physical_node_count = 0U;
    double physical_available_resource = 0.0;
    double physical_node_available_resource = 0.0;
    double physical_link_available_resource = 0.0;
    double physical_node_resource_utilization = 0.0;
    double physical_link_resource_utilization = 0.0;
};

struct RecorderRecord {
    RecorderRecord(
        RecorderState state_value,
        Solution solution_value,
        utils::ClassDict extra_value = {});

    RecorderState state;
    Solution solution;
    utils::ClassDict extra;
};

enum class RecorderErrorCode : std::uint8_t {
    missing_event,
    invalid_event_type,
    missing_initial_physical_state,
    zero_initial_node_resource,
    zero_initial_link_resource,
    invalid_time_ratio,
    invalid_physical_node,
    missing_virtual_network_record,
    record_index_out_of_range,
    membership_mismatch,
    temporary_saving_disabled,
    invalid_filename,
    unsupported_extra_value,
    filesystem_failure,
};

enum class RecorderOperation : std::uint8_t {
    construct,
    reset,
    count_initial_physical_network,
    count,
    update_state,
    add_record,
    lookup_record,
    temporary_save_record,
    save_records,
    summarize_records,
    append_summary,
};

class RecorderException final : public std::runtime_error {
public:
    RecorderException(
        RecorderErrorCode code,
        RecorderOperation operation,
        std::string message,
        std::optional<std::int64_t> event_id = std::nullopt,
        std::optional<SolutionNodeId> virtual_network_id = std::nullopt,
        std::optional<SolutionNodeId> physical_node_id = std::nullopt);

    RecorderErrorCode code() const noexcept;
    RecorderOperation operation() const noexcept;
    const std::optional<std::int64_t>& event_id() const noexcept;
    const std::optional<SolutionNodeId>& virtual_network_id() const noexcept;
    const std::optional<SolutionNodeId>& physical_node_id() const noexcept;

private:
    RecorderErrorCode code_;
    RecorderOperation operation_;
    std::optional<std::int64_t> event_id_;
    std::optional<SolutionNodeId> virtual_network_id_;
    std::optional<SolutionNodeId> physical_node_id_;
};

// A cold summary-only extension point. It deliberately knows nothing about
// RL, feature construction, training, logging sinks, or solver registries.
// Future optional modules may append their already-serialized typed columns
// without adding virtual dispatch or dynamic field lookup to Recorder::count.
struct RecorderSummaryColumn {
    std::string name;
    std::string value;
};

class Recorder;

class RecorderSummaryExtension {
public:
    virtual ~RecorderSummaryExtension() = default;

    virtual void append_columns(
        const Recorder& recorder,
        const CounterSummary& summary,
        std::vector<RecorderSummaryColumn>& columns) const = 0;
};

class Recorder {
public:
    Recorder(Counter counter, RecorderConfig config);

    void reset();
    void set_event(RecorderEvent event) noexcept;
    void count_initial_physical_network(
        const network::PhysicalNetwork& physical_network,
        RecorderOptions options = {});

    RecorderRecord count(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        Solution& solution,
        RecorderOptions options = {});

    RecorderRecord count_prepared(
        const PreparedCounter& virtual_counter,
        const network::PhysicalNetwork& physical_network,
        Solution& solution,
        RecorderOptions options = {});

    // Arrival fast path for an Environment that already ran the matching
    // PreparedCounter in order to apply its admission policy.  The supplied
    // Solution must therefore already contain its complete cost/revenue
    // fields.  Leave events do not use this overload.
    RecorderRecord count_precomputed_arrival(
        const network::PhysicalNetwork& physical_network,
        Solution& solution,
        RecorderOptions options = {});

    const RecorderRecord& add_record(RecorderRecord record);
    const RecorderRecord& record_by_event(std::int64_t event_id) const;
    const RecorderRecord& record_by_virtual_network(
        SolutionNodeId virtual_network_id) const;
    std::vector<SolutionNodeId> running_physical_nodes() const;

    void temporary_save_record(const RecorderRecord& record);
    std::filesystem::path save_records(
        std::string_view filename,
        RecorderOptions options = {}) const;
    CounterSummary summary_records() const;
    std::filesystem::path append_summary(
        const CounterSummary& summary,
        std::string_view filename = "summary.csv") const;

    void set_summary_extension(
        std::shared_ptr<const RecorderSummaryExtension> extension) noexcept;

    const RecorderState& state() const noexcept;
    const std::optional<RecorderInitialPhysicalState>& initial_physical_state()
        const noexcept;
    const std::vector<RecorderRecord>& memory() const noexcept;
    const std::filesystem::path& summary_dir() const noexcept;
    const std::filesystem::path& record_dir() const noexcept;
    const std::optional<std::filesystem::path>& temp_save_path() const noexcept;

private:
    RecorderRecord count_impl(
        const PreparedCounter* virtual_counter,
        const network::PhysicalNetwork& physical_network,
        Solution& solution,
        RecorderOptions options);
    void update_count_state(
        const network::PhysicalNetwork& physical_network,
        const Solution& solution,
        RecorderOptions options);
    void check_event_for_count() const;
    void add_membership(const Solution& solution);
    void remove_membership(const Solution& solution);

    Counter counter_;
    RecorderConfig config_;
    std::filesystem::path summary_dir_;
    std::filesystem::path record_dir_;
    std::optional<std::filesystem::path> temp_save_path_;

    RecorderState state_;
    std::optional<RecorderInitialPhysicalState> initial_physical_state_;
    std::optional<PreparedCounter> prepared_physical_counter_;
    const network::PhysicalNetwork* prepared_physical_network_ = nullptr;
    std::size_t physical_node_capacity_ = 0U;

    std::vector<RecorderRecord> memory_;
    std::unordered_map<SolutionNodeId, std::int64_t>
        virtual_network_event_indices_;

    // Dense slots serve every hot membership operation by numeric ID. The
    // append-only encounter order preserves Python dict insertion order for
    // running_physical_nodes(); multiplicity remains in each vector.
    std::vector<std::vector<SolutionNodeId>> physical_node_memberships_;
    std::vector<bool> physical_node_encountered_;
    std::vector<SolutionNodeId> physical_node_encounter_order_;
    std::size_t live_membership_node_count_ = 0U;

    std::shared_ptr<const RecorderSummaryExtension> summary_extension_;
};

} // namespace virne::core
