#pragma once

#include "solution.h"
#include "../network/base_network.h"
#include "../network/virtual_network.h"
#include "../network/virtual_network_event.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace virne::core {

using CounterResourceId = network::attribute::AttributeRegistryId;
using CounterNumber = std::variant<std::int64_t, double>;

struct CounterSelection {
    std::optional<std::vector<CounterResourceId>> node_resources;
    std::optional<std::vector<CounterResourceId>> link_resources;
};

struct CounterOptions {
    std::size_t workers = 1U;
};

enum class CounterErrorCode : std::uint8_t {
    invalid_node_resource_selection,
    invalid_link_resource_selection,
    attribute_registry_mismatch,
    graph_binding_mismatch,
    missing_node_resource_value,
    missing_link_resource_value,
    non_numeric_resource_value,
    empty_node_resource_selection,
    empty_link_resource_selection,
    numeric_overflow,
    invalid_solution_node,
    invalid_solution_link,
    missing_route_info,
    virtual_network_required,
    missing_virtual_lifetime,
    empty_records,
    invalid_record_value,
    missing_record_column,
    legacy_summary_csv_binding,
};

enum class CounterOperation : std::uint8_t {
    prepare,
    sum_node_resources,
    sum_link_resources,
    count_link_cost,
    count_partial_solution,
    count_solution,
    summarize_records,
    summarize_csv,
};

class CounterException final : public std::runtime_error {
public:
    CounterException(
        CounterErrorCode code,
        CounterOperation operation,
        std::string message,
        std::optional<CounterResourceId> resource_id = std::nullopt,
        std::optional<SolutionNodeId> virtual_node = std::nullopt,
        std::optional<SolutionLink> virtual_link = std::nullopt,
        std::optional<SolutionLink> physical_link = std::nullopt,
        std::optional<std::size_t> row_index = std::nullopt);

    CounterErrorCode code() const noexcept;
    CounterOperation operation() const noexcept;
    const std::optional<CounterResourceId>& resource_id() const noexcept;
    const std::optional<SolutionNodeId>& virtual_node() const noexcept;
    const std::optional<SolutionLink>& virtual_link() const noexcept;
    const std::optional<SolutionLink>& physical_link() const noexcept;
    const std::optional<std::size_t>& row_index() const noexcept;

private:
    CounterErrorCode code_;
    CounterOperation operation_;
    std::optional<CounterResourceId> resource_id_;
    std::optional<SolutionNodeId> virtual_node_;
    std::optional<SolutionLink> virtual_link_;
    std::optional<SolutionLink> physical_link_;
    std::optional<std::size_t> row_index_;
};

class PreparedCounter;

class Counter {
public:
    explicit Counter(CounterSelection selection = {});

    const CounterSelection& selection() const noexcept;

    PreparedCounter prepare(const network::BaseNetwork& network) const;
    PreparedCounter prepare(
        const network::VirtualNetwork& virtual_network) const;

private:
    PreparedCounter prepare_impl(
        const network::BaseNetwork& network,
        const network::VirtualNetwork* virtual_network) const;

    CounterSelection selection_;
};

class PreparedCounter {
public:
    CounterNumber calculate_sum_network_resource(
        bool node = true,
        bool link = true,
        CounterOptions options = {}) const;
    CounterNumber calculate_sum_node_resource(
        CounterOptions options = {}) const;
    CounterNumber calculate_sum_link_resource(
        CounterOptions options = {}) const;

    CounterNumber calculate_v_net_link_cost(
        const Solution& solution) const;
    CounterNumber calculate_v_net_cost(
        const Solution& solution,
        CounterOptions options = {}) const;
    CounterNumber calculate_v_net_revenue(
        CounterOptions options = {}) const;

    void count_partial_solution(
        Solution& solution,
        CounterOptions options = {}) const;
    void count_solution(
        Solution& solution,
        CounterOptions options = {}) const;

private:
    struct ResourceBinding {
        CounterResourceId registry_id = 0U;
        AttrId value_id = 0U;
    };

    PreparedCounter(
        const network::BaseNetwork& network,
        const network::VirtualNetwork* virtual_network,
        std::vector<ResourceBinding> node_resources,
        std::vector<ResourceBinding> link_resources,
        const network::attribute::NodeAttributeRegistry* node_registry,
        const network::attribute::LinkAttributeRegistry* link_registry,
        const Graph* graph,
        const ::AttributeRegistry* graph_attribute_registry);

    void validate_network_binding(CounterOperation operation) const;
    CounterNumber calculate_resource_sum(
        bool node,
        CounterOptions options) const;
    CounterNumber calculate_link_cost(
        const Solution& solution,
        CounterOperation operation) const;

    const network::BaseNetwork* network_ = nullptr;
    const network::VirtualNetwork* virtual_network_ = nullptr;
    std::vector<ResourceBinding> node_resources_;
    std::vector<ResourceBinding> link_resources_;
    const network::attribute::NodeAttributeRegistry* node_registry_ = nullptr;
    const network::attribute::LinkAttributeRegistry* link_registry_ = nullptr;
    const Graph* graph_ = nullptr;
    const ::AttributeRegistry* graph_attribute_registry_ = nullptr;

    friend class Counter;
};

struct CounterRecord {
    std::int64_t success_count = 0;
    std::int64_t virtual_network_count = 0;
    network::VirtualEventType event_type = network::VirtualEventType::leave;
    double v_net_r2c_ratio = 0.0;
    double total_time_revenue = 0.0;
    double total_time_cost = 0.0;
    double virtual_network_arrival_time = 0.0;
    bool early_rejection = false;
    bool place_result = true;
    bool route_result = true;
    double total_cost = 0.0;
    double total_revenue = 0.0;
    double physical_available_resource = 0.0;
    double physical_node_available_resource = 0.0;
    double physical_link_available_resource = 0.0;
    std::int64_t inservice_count = 0;
    double hard_constraint_violation = 0.0;
    double max_single_step_hard_constraint_violation = 0.0;
    std::optional<double> reward;
};

struct CounterRecords {
    std::vector<CounterRecord> rows;
    bool reward_column_present = false;
};

struct CounterSummary {
    double acceptance_rate = 0.0;
    double average_r2c_ratio = 0.0;
    double long_term_time_r2c_ratio = 0.0;
    double long_term_average_time_revenue = 0.0;
    std::int64_t success_count = 0;
    std::size_t early_rejection_count = 0U;
    std::size_t place_failure_count = 0U;
    std::size_t route_failure_count = 0U;
    double total_cost = 0.0;
    double total_revenue = 0.0;
    double total_time_revenue = 0.0;
    double total_time_cost = 0.0;
    double long_term_r2c_ratio = 0.0;
    double total_simulation_time = 0.0;
    double long_term_average_revenue = 0.0;
    double long_term_average_cost = 0.0;
    double minimum_physical_available_resource = 0.0;
    double minimum_physical_node_available_resource = 0.0;
    double minimum_physical_link_available_resource = 0.0;
    std::int64_t maximum_inservice_count = 0;
    double total_violation = 0.0;
    double total_max_single_step_violation = 0.0;
    double average_reward = 0.0;
};

CounterSummary summary_records(const CounterRecords& records);
CounterSummary summary_csv(const std::string& path);

}  // namespace virne::core
