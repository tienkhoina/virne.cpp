#include "recorder.h"

#include "../utils/utils_config.h"
#include "../../csv/csv.h"

#include <algorithm>
#include <any>
#include <charconv>
#include <cmath>
#include <exception>
#include <limits>
#include <system_error>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace virne::core {
namespace {

double counter_number_as_double(const CounterNumber& value) noexcept {
    return std::visit(
        [](const auto number) noexcept {
            return static_cast<double>(number);
        },
        value);
}

std::string format_integer(const std::int64_t value) {
    char buffer[32];
    const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("failed to serialize signed integer");
    }
    return std::string(buffer, result.ptr);
}

std::string format_unsigned(const std::uint64_t value) {
    char buffer[32];
    const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("failed to serialize unsigned integer");
    }
    return std::string(buffer, result.ptr);
}

std::string format_size(const std::size_t value) {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    return format_unsigned(static_cast<std::uint64_t>(value));
}

std::string format_double(const double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return std::signbit(value) ? "-inf" : "inf";
    }

    char buffer[128];
    const auto result = std::to_chars(
        std::begin(buffer),
        std::end(buffer),
        value,
        std::chars_format::general);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("failed to serialize floating-point value");
    }
    std::string output(buffer, result.ptr);
    if (output.find_first_of(".eE") == std::string::npos) {
        output += ".0";
    }
    return output;
}

std::string format_bool(const bool value) {
    return value ? "True" : "False";
}

std::string format_optional_size(const std::optional<std::size_t>& value) {
    return value.has_value() ? format_size(*value) : std::string{};
}

template <typename Function>
void parallel_indexed(
    const std::size_t count,
    const std::size_t requested_workers,
    Function&& function) {
    if (count == 0U) {
        return;
    }
    const std::size_t workers = requested_workers <= 1U || count <= 1U
        ? 1U
        : std::min(requested_workers, count);
    if (workers == 1U) {
        for (std::size_t index = 0U; index < count; ++index) {
            function(index);
        }
        return;
    }

    std::vector<std::exception_ptr> errors(count);
    auto run_range = [&](const std::size_t begin, const std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            try {
                function(index);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        }
    };

    const std::size_t block = count / workers;
    const std::size_t remainder = count % workers;
    std::vector<std::thread> threads;
    threads.reserve(workers - 1U);
    std::size_t begin = 0U;
    try {
        for (std::size_t worker = 0U; worker + 1U < workers; ++worker) {
            const std::size_t length = block + (worker < remainder ? 1U : 0U);
            const std::size_t end = begin + length;
            threads.emplace_back(run_range, begin, end);
            begin = end;
        }
    } catch (...) {
        for (auto& thread : threads) {
            thread.join();
        }
        run_range(begin, count);
        for (std::size_t index = 0U; index < count; ++index) {
            if (errors[index]) {
                std::rethrow_exception(errors[index]);
            }
        }
        return;
    }

    run_range(begin, count);
    for (auto& thread : threads) {
        thread.join();
    }
    for (std::size_t index = 0U; index < count; ++index) {
        if (errors[index]) {
            std::rethrow_exception(errors[index]);
        }
    }
}

std::filesystem::path checked_component(
    const std::string_view raw,
    const RecorderOperation operation) {
    if (raw.empty() || raw == "." || raw == ".." ||
        raw.find('/') != std::string_view::npos ||
        raw.find('\\') != std::string_view::npos) {
        throw RecorderException(
            RecorderErrorCode::invalid_filename,
            operation,
            "recorder path component must be one non-empty filename");
    }
    const std::filesystem::path value{std::string(raw)};
    if (value.has_root_path() || value.filename() != value) {
        throw RecorderException(
            RecorderErrorCode::invalid_filename,
            operation,
            "recorder path component escapes its configured directory");
    }
    return value;
}

std::filesystem::path checked_filename(
    const std::string_view raw,
    const RecorderOperation operation) {
    return checked_component(raw, operation);
}

std::string serialize_attribute_number(
    const network::attribute::AttributeNumber& value) {
    return std::visit(
        [](const auto number) -> std::string {
            using Number = std::decay_t<decltype(number)>;
            if constexpr (std::is_same_v<Number, bool>) {
                return format_bool(number);
            } else if constexpr (std::is_same_v<Number, std::int64_t>) {
                return format_integer(number);
            } else {
                return format_double(number);
            }
        },
        value);
}

std::string serialize_attribute_values(const SolutionAttributeValues& values) {
    std::string output = "{";
    bool first = true;
    const auto& slots = values.slots();
    for (std::size_t id = 0U; id < slots.size(); ++id) {
        if (!slots[id].has_value()) {
            continue;
        }
        if (!first) {
            output += ", ";
        }
        first = false;
        output += format_size(id);
        output += ": ";
        output += serialize_attribute_number(*slots[id]);
    }
    output += '}';
    return output;
}

std::string serialize_solution_link(const SolutionLink& link) {
    return "(" + format_integer(link.source) + ", " +
        format_integer(link.target) + ")";
}

std::string serialize_node_slots(const NodeSlots& values) {
    std::string output = "{";
    bool first = true;
    for (const auto& entry : values.entries()) {
        if (!first) {
            output += ", ";
        }
        first = false;
        output += format_integer(entry.key);
        output += ": ";
        output += format_integer(entry.value);
    }
    output += '}';
    return output;
}

std::string serialize_link_paths(const LinkPaths& values) {
    std::string output = "{";
    bool first_entry = true;
    for (const auto& entry : values.entries()) {
        if (!first_entry) {
            output += ", ";
        }
        first_entry = false;
        output += serialize_solution_link(entry.key);
        output += ": [";
        bool first_link = true;
        for (const auto& link : entry.value) {
            if (!first_link) {
                output += ", ";
            }
            first_link = false;
            output += serialize_solution_link(link);
        }
        output += ']';
    }
    output += '}';
    return output;
}

std::string serialize_node_slots_info(const NodeSlotsInfo& values) {
    std::string output = "{";
    bool first = true;
    for (const auto& entry : values.entries()) {
        if (!first) {
            output += ", ";
        }
        first = false;
        output += "(" + format_integer(entry.key.virtual_node) + ", " +
            format_integer(entry.key.physical_node) + "): ";
        output += serialize_attribute_values(entry.value);
    }
    output += '}';
    return output;
}

std::string serialize_link_paths_info(const LinkPathsInfo& values) {
    std::string output = "{";
    bool first = true;
    for (const auto& entry : values.entries()) {
        if (!first) {
            output += ", ";
        }
        first = false;
        output += "(" + serialize_solution_link(entry.key.virtual_link) +
            ", " + serialize_solution_link(entry.key.physical_link) + "): ";
        output += serialize_attribute_values(entry.value);
    }
    output += '}';
    return output;
}

template <typename Table>
std::string serialize_node_table(const Table& values) {
    std::string output = "{";
    bool first = true;
    for (const auto& entry : values.entries()) {
        if (!first) {
            output += ", ";
        }
        first = false;
        output += format_integer(entry.key);
        output += ": ";
        output += serialize_attribute_values(entry.value);
    }
    output += '}';
    return output;
}

template <typename Table>
std::string serialize_link_table(const Table& values) {
    std::string output = "{";
    bool first = true;
    for (const auto& entry : values.entries()) {
        if (!first) {
            output += ", ";
        }
        first = false;
        output += serialize_solution_link(entry.key);
        output += ": ";
        output += serialize_attribute_values(entry.value);
    }
    output += '}';
    return output;
}

std::string serialize_step_values(
    const SolutionStepConstraintValues& values) {
    return "{'node_level': " + serialize_attribute_values(values.node_level) +
        ", 'link_level': " + serialize_attribute_values(values.link_level) +
        ", 'path_level': " + serialize_attribute_values(values.path_level) +
        '}';
}

std::string serialize_offsets(const SolutionConstraintOffsets& values) {
    return "{'node_level': " + serialize_node_table(values.node_level) +
        ", 'link_level': " + serialize_link_table(values.link_level) +
        ", 'path_level': " + serialize_link_table(values.path_level) + '}';
}

std::string serialize_violations(const SolutionConstraintViolations& values) {
    return "{'node_level': " + serialize_node_table(values.node_level) +
        ", 'link_level': " + serialize_link_table(values.link_level) +
        ", 'path_level': " + serialize_link_table(values.path_level) + '}';
}

template <typename Value, typename Formatter>
std::string serialize_list(
    const std::vector<Value>& values,
    Formatter&& formatter) {
    std::string output = "[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            output += ", ";
        }
        output += formatter(values[index]);
    }
    output += ']';
    return output;
}

const std::vector<std::string>& fixed_record_columns() {
    static const std::vector<std::string> columns{
        "event_id",
        "event_type",
        "v_net_count",
        "success_count",
        "inservice_count",
        "total_revenue",
        "total_cost",
        "total_time_revenue",
        "total_time_cost",
        "long_term_r2c_ratio",
        "long_term_time_r2c_ratio",
        "num_running_p_net_nodes",
        "p_net_available_resource",
        "p_net_node_available_resource",
        "p_net_link_available_resource",
        "p_net_node_resource_utilization",
        "p_net_link_resource_utilization",
        "v_net_id",
        "v_net_lifetime",
        "v_net_arrival_time",
        "v_net_num_nodes",
        "v_net_num_edges",
        "result",
        "node_slots",
        "link_paths",
        "node_slots_info",
        "link_paths_info",
        "v_net_cost",
        "v_net_revenue",
        "v_net_demand",
        "v_net_node_demand",
        "v_net_link_demand",
        "v_net_node_revenue",
        "v_net_link_revenue",
        "v_net_node_cost",
        "v_net_link_cost",
        "v_net_path_cost",
        "v_net_r2c_ratio",
        "v_net_time_cost",
        "v_net_time_revenue",
        "v_net_time_rc_ratio",
        "description",
        "v_net_total_hard_constraint_violation",
        "v_net_single_step_constraint_offset",
        "v_net_constraint_offsets",
        "v_net_constraint_violations",
        "v_net_single_step_violation_list",
        "v_net_single_step_hard_constraint_offset",
        "v_net_max_single_step_hard_constraint_violation",
        "place_result",
        "route_result",
        "early_rejection",
        "revoke_times",
        "selected_actions",
        "num_interactions",
        "v_net_reward",
        "num_placed_nodes",
        "num_routed_links",
        "num_attempt_times",
    };
    return columns;
}

std::vector<std::string> fixed_record_values(const RecorderRecord& record) {
    const RecorderState& state = record.state;
    const Solution& solution = record.solution;
    std::vector<std::string> values;
    values.reserve(fixed_record_columns().size());

    values.push_back(state.event.has_value()
        ? format_integer(state.event->event_id)
        : std::string{});
    values.push_back(state.event.has_value()
        ? format_unsigned(static_cast<std::uint8_t>(state.event->type))
        : std::string{});
    values.push_back(format_integer(state.virtual_network_count));
    values.push_back(format_integer(state.success_count));
    values.push_back(format_integer(state.inservice_count));
    values.push_back(format_double(state.total_revenue));
    values.push_back(format_double(state.total_cost));
    values.push_back(format_double(state.total_time_revenue));
    values.push_back(format_double(state.total_time_cost));
    values.push_back(format_double(state.long_term_r2c_ratio));
    values.push_back(format_double(state.long_term_time_r2c_ratio));
    values.push_back(format_size(state.running_physical_node_count));
    values.push_back(format_double(state.physical_available_resource));
    values.push_back(format_double(state.physical_node_available_resource));
    values.push_back(format_double(state.physical_link_available_resource));
    values.push_back(format_double(state.physical_node_resource_utilization));
    values.push_back(format_double(state.physical_link_resource_utilization));

    values.push_back(format_integer(solution.v_net_id));
    values.push_back(format_double(solution.v_net_lifetime));
    values.push_back(format_double(solution.v_net_arrival_time));
    values.push_back(format_size(solution.v_net_num_nodes));
    values.push_back(format_size(solution.v_net_num_edges));
    values.push_back(format_bool(solution.result));
    values.push_back(serialize_node_slots(solution.node_slots));
    values.push_back(serialize_link_paths(solution.link_paths));
    values.push_back(serialize_node_slots_info(solution.node_slots_info));
    values.push_back(serialize_link_paths_info(solution.link_paths_info));
    values.push_back(format_double(solution.v_net_cost));
    values.push_back(format_double(solution.v_net_revenue));
    values.push_back(format_double(solution.v_net_demand));
    values.push_back(format_double(solution.v_net_node_demand));
    values.push_back(format_double(solution.v_net_link_demand));
    values.push_back(format_double(solution.v_net_node_revenue));
    values.push_back(format_double(solution.v_net_link_revenue));
    values.push_back(format_double(solution.v_net_node_cost));
    values.push_back(format_double(solution.v_net_link_cost));
    values.push_back(format_double(solution.v_net_path_cost));
    values.push_back(format_double(solution.v_net_r2c_ratio));
    values.push_back(format_double(solution.v_net_time_cost));
    values.push_back(format_double(solution.v_net_time_revenue));
    values.push_back(format_double(solution.v_net_time_rc_ratio));
    values.push_back(solution.description);
    values.push_back(format_double(
        solution.v_net_total_hard_constraint_violation));
    values.push_back(serialize_step_values(
        solution.v_net_single_step_constraint_offset));
    values.push_back(serialize_offsets(solution.v_net_constraint_offsets));
    values.push_back(serialize_violations(
        solution.v_net_constraint_violations));
    values.push_back(serialize_list(
        solution.v_net_single_step_violation_list,
        [](const network::attribute::AttributeNumber& value) {
            return serialize_attribute_number(value);
        }));
    values.push_back(format_double(
        solution.v_net_single_step_hard_constraint_offset));
    values.push_back(format_double(
        solution.v_net_max_single_step_hard_constraint_violation));
    values.push_back(format_bool(solution.place_result));
    values.push_back(format_bool(solution.route_result));
    values.push_back(format_bool(solution.early_rejection));
    values.push_back(format_integer(solution.revoke_times));
    values.push_back(serialize_list(
        solution.selected_actions,
        [](const std::int64_t value) { return format_integer(value); }));
    values.push_back(format_integer(solution.num_interactions));
    values.push_back(format_double(solution.v_net_reward));
    values.push_back(format_optional_size(solution.num_placed_nodes));
    values.push_back(format_optional_size(solution.num_routed_links));
    values.push_back(format_optional_size(solution.num_attempt_times));
    return values;
}

std::string any_to_string(
    const std::any& value,
    const RecorderOperation operation);

std::string any_list_to_string(
    const utils::ClassAnyList& values,
    const RecorderOperation operation) {
    return serialize_list(
        values,
        [operation](const std::any& item) {
            return any_to_string(item, operation);
        });
}

std::string any_mapping_to_string(
    const utils::ClassMapping& mapping,
    const RecorderOperation operation) {
    std::string output = "{";
    for (std::size_t index = 0U; index < mapping.items.size(); ++index) {
        if (index != 0U) {
            output += ", ";
        }
        output += any_to_string(mapping.items[index].key, operation);
        output += ": ";
        output += any_to_string(mapping.items[index].value, operation);
    }
    output += '}';
    return output;
}

std::string any_to_string(
    const std::any& value,
    const RecorderOperation operation) {
    if (!value.has_value()) {
        return {};
    }
    const std::type_info& type = value.type();
    if (type == typeid(std::string)) {
        return std::any_cast<const std::string&>(value);
    }
    if (type == typeid(bool)) {
        return format_bool(std::any_cast<bool>(value));
    }
    if (type == typeid(std::int64_t)) {
        return format_integer(std::any_cast<std::int64_t>(value));
    }
    if (type == typeid(std::int32_t)) {
        return format_integer(std::any_cast<std::int32_t>(value));
    }
    if (type == typeid(std::int16_t)) {
        return format_integer(std::any_cast<std::int16_t>(value));
    }
    if (type == typeid(std::int8_t)) {
        return format_integer(std::any_cast<std::int8_t>(value));
    }
    if (type == typeid(std::uint64_t)) {
        return format_unsigned(std::any_cast<std::uint64_t>(value));
    }
    if (type == typeid(std::uint32_t)) {
        return format_unsigned(std::any_cast<std::uint32_t>(value));
    }
    if (type == typeid(std::uint16_t)) {
        return format_unsigned(std::any_cast<std::uint16_t>(value));
    }
    if (type == typeid(std::uint8_t)) {
        return format_unsigned(std::any_cast<std::uint8_t>(value));
    }
    if (type == typeid(std::size_t)) {
        return format_size(std::any_cast<std::size_t>(value));
    }
    if (type == typeid(double)) {
        return format_double(std::any_cast<double>(value));
    }
    if (type == typeid(float)) {
        return format_double(static_cast<double>(std::any_cast<float>(value)));
    }
    if (type == typeid(std::filesystem::path)) {
        return std::any_cast<const std::filesystem::path&>(value).string();
    }
    if (type == typeid(utils::ClassAnyListPtr)) {
        const auto& pointer = std::any_cast<const utils::ClassAnyListPtr&>(value);
        return pointer ? any_list_to_string(*pointer, operation) : std::string{};
    }
    if (type == typeid(utils::ClassMappingPtr)) {
        const auto& pointer = std::any_cast<const utils::ClassMappingPtr&>(value);
        return pointer ? any_mapping_to_string(*pointer, operation)
                       : std::string{};
    }
    throw RecorderException(
        RecorderErrorCode::unsupported_extra_value,
        operation,
        "unsupported ClassDict value in recorder CSV boundary");
}

csvio::DataFrame record_frame(
    const std::vector<const RecorderRecord*>& records,
    const RecorderOptions options,
    const RecorderOperation operation) {
    csvio::DataFrame frame;
    frame.columns = fixed_record_columns();

    std::vector<utils::ClassDictSnapshot> extras(records.size());
    std::unordered_map<std::string, std::size_t> dynamic_columns;
    for (std::size_t row = 0U; row < records.size(); ++row) {
        extras[row] = records[row]->extra.to_dict();
        for (const auto& item : extras[row]) {
            if (dynamic_columns.find(item.key) != dynamic_columns.end()) {
                continue;
            }
            if (std::find(frame.columns.begin(), frame.columns.end(), item.key) !=
                frame.columns.end()) {
                throw RecorderException(
                    RecorderErrorCode::unsupported_extra_value,
                    operation,
                    "dynamic recorder field duplicates a fixed column");
            }
            const std::size_t column = frame.columns.size();
            frame.columns.push_back(item.key);
            dynamic_columns.emplace(item.key, column);
        }
    }

    frame.rows.resize(records.size());
    parallel_indexed(records.size(), options.workers, [&](const std::size_t row) {
        frame.rows[row] = fixed_record_values(*records[row]);
        frame.rows[row].resize(frame.columns.size());
        for (const auto& item : extras[row]) {
            const auto found = dynamic_columns.find(item.key);
            frame.rows[row][found->second] = any_to_string(item.value, operation);
        }
    });
    frame.validate();
    return frame;
}

const std::vector<std::string>& summary_columns() {
    static const std::vector<std::string> columns{
        "acceptance_rate",
        "average_r2c_ratio",
        "long_term_time_r2c_ratio",
        "long_term_average_time_revenue",
        "success_count",
        "early_rejection_count",
        "place_failure_count",
        "route_failure_count",
        "total_cost",
        "total_revenue",
        "total_time_revenue",
        "total_time_cost",
        "long_term_r2c_ratio",
        "total_simulation_time",
        "long_term_average_revenue",
        "long_term_average_cost",
        "minimum_physical_available_resource",
        "minimum_physical_node_available_resource",
        "minimum_physical_link_available_resource",
        "maximum_inservice_count",
        "total_violation",
        "total_max_single_step_violation",
        "average_reward",
    };
    return columns;
}

std::vector<std::string> summary_values(const CounterSummary& summary) {
    return {
        format_double(summary.acceptance_rate),
        format_double(summary.average_r2c_ratio),
        format_double(summary.long_term_time_r2c_ratio),
        format_double(summary.long_term_average_time_revenue),
        format_integer(summary.success_count),
        format_size(summary.early_rejection_count),
        format_size(summary.place_failure_count),
        format_size(summary.route_failure_count),
        format_double(summary.total_cost),
        format_double(summary.total_revenue),
        format_double(summary.total_time_revenue),
        format_double(summary.total_time_cost),
        format_double(summary.long_term_r2c_ratio),
        format_double(summary.total_simulation_time),
        format_double(summary.long_term_average_revenue),
        format_double(summary.long_term_average_cost),
        format_double(summary.minimum_physical_available_resource),
        format_double(summary.minimum_physical_node_available_resource),
        format_double(summary.minimum_physical_link_available_resource),
        format_integer(summary.maximum_inservice_count),
        format_double(summary.total_violation),
        format_double(summary.total_max_single_step_violation),
        format_double(summary.average_reward),
    };
}

} // namespace

RecorderRecord::RecorderRecord(
    RecorderState state_value,
    Solution solution_value,
    utils::ClassDict extra_value)
    : state(std::move(state_value)),
      solution(std::move(solution_value)),
      extra(utils::ClassDict::from_dict(extra_value.to_dict())) {}

RecorderException::RecorderException(
    const RecorderErrorCode code,
    const RecorderOperation operation,
    std::string message,
    const std::optional<std::int64_t> event_id,
    const std::optional<SolutionNodeId> virtual_network_id,
    const std::optional<SolutionNodeId> physical_node_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      event_id_(event_id),
      virtual_network_id_(virtual_network_id),
      physical_node_id_(physical_node_id) {}

RecorderErrorCode RecorderException::code() const noexcept {
    return code_;
}

RecorderOperation RecorderException::operation() const noexcept {
    return operation_;
}

const std::optional<std::int64_t>& RecorderException::event_id() const noexcept {
    return event_id_;
}

const std::optional<SolutionNodeId>&
RecorderException::virtual_network_id() const noexcept {
    return virtual_network_id_;
}

const std::optional<SolutionNodeId>&
RecorderException::physical_node_id() const noexcept {
    return physical_node_id_;
}

Recorder::Recorder(Counter counter, RecorderConfig config)
    : counter_(std::move(counter)), config_(std::move(config)) {
    checked_component(config_.solver_name, RecorderOperation::construct);
    checked_component(config_.run_id, RecorderOperation::construct);
    checked_component(config_.record_dir_name, RecorderOperation::construct);

    summary_dir_ = utils::get_run_id_dir(utils::RunDirectoryInput{
        config_.save_root_dir,
        config_.solver_name,
        config_.run_id,
    });
    record_dir_ = summary_dir_ / config_.record_dir_name;
    try {
        std::filesystem::create_directories(record_dir_);
    } catch (const std::filesystem::filesystem_error& error) {
        throw RecorderException(
            RecorderErrorCode::filesystem_failure,
            RecorderOperation::construct,
            error.what());
    }
    reset();
}

void Recorder::reset() {
    state_ = RecorderState{};
    memory_.clear();
    virtual_network_event_indices_.clear();
    physical_node_memberships_.assign(physical_node_capacity_, {});
    physical_node_encountered_.assign(physical_node_capacity_, false);
    physical_node_encounter_order_.clear();
    live_membership_node_count_ = 0U;

    if (!config_.temporary_records) {
        temp_save_path_.reset();
        return;
    }

    try {
        for (std::size_t suffix = 0U;; ++suffix) {
            const auto candidate =
                record_dir_ / ("temp-" + format_size(suffix) + ".csv");
            if (!std::filesystem::exists(candidate)) {
                temp_save_path_ = candidate;
                break;
            }
        }
    } catch (const std::filesystem::filesystem_error& error) {
        throw RecorderException(
            RecorderErrorCode::filesystem_failure,
            RecorderOperation::reset,
            error.what());
    }
}

void Recorder::set_event(const RecorderEvent event) noexcept {
    state_.event = event;
}

void Recorder::count_initial_physical_network(
    const network::PhysicalNetwork& physical_network,
    const RecorderOptions options) {
    prepared_physical_counter_.emplace(counter_.prepare(physical_network));
    prepared_physical_network_ = &physical_network;
    initial_physical_state_.emplace();

    const CounterOptions counter_options{options.workers};
    initial_physical_state_->available_resource = counter_number_as_double(
        prepared_physical_counter_->calculate_sum_network_resource(
            true, true, counter_options));
    initial_physical_state_->node_available_resource = counter_number_as_double(
        prepared_physical_counter_->calculate_sum_network_resource(
            true, false, counter_options));
    initial_physical_state_->link_available_resource = counter_number_as_double(
        prepared_physical_counter_->calculate_sum_network_resource(
            false, true, counter_options));

    physical_node_capacity_ = physical_network.num_nodes();
    physical_node_memberships_.resize(physical_node_capacity_);
    physical_node_encountered_.resize(physical_node_capacity_, false);
}

void Recorder::check_event_for_count() const {
    if (!state_.event.has_value()) {
        throw RecorderException(
            RecorderErrorCode::missing_event,
            RecorderOperation::count,
            "Recorder::count requires a current event");
    }
    if (state_.event->type != network::VirtualEventType::leave &&
        state_.event->type != network::VirtualEventType::arrival) {
        throw RecorderException(
            RecorderErrorCode::invalid_event_type,
            RecorderOperation::count,
            "unsupported virtual-network event type",
            state_.event->event_id);
    }
}

RecorderRecord Recorder::count(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    Solution& solution,
    const RecorderOptions options) {
    check_event_for_count();
    if (state_.event->type == network::VirtualEventType::arrival) {
        const PreparedCounter virtual_counter = counter_.prepare(virtual_network);
        return count_impl(
            &virtual_counter, physical_network, solution, options);
    }
    return count_impl(nullptr, physical_network, solution, options);
}

RecorderRecord Recorder::count_prepared(
    const PreparedCounter& virtual_counter,
    const network::PhysicalNetwork& physical_network,
    Solution& solution,
    const RecorderOptions options) {
    check_event_for_count();
    return count_impl(
        &virtual_counter, physical_network, solution, options);
}

RecorderRecord Recorder::count_impl(
    const PreparedCounter* const virtual_counter,
    const network::PhysicalNetwork& physical_network,
    Solution& solution,
    const RecorderOptions options) {
    if (state_.event->type == network::VirtualEventType::arrival) {
        if (virtual_counter == nullptr) {
            throw RecorderException(
                RecorderErrorCode::invalid_event_type,
                RecorderOperation::count,
                "arrival count requires a prepared virtual counter",
                state_.event->event_id,
                solution.v_net_id);
        }
        virtual_counter->count_solution(
            solution, CounterOptions{options.workers});
    }

    update_count_state(physical_network, solution, options);
    return RecorderRecord(state_, solution);
}

void Recorder::update_count_state(
    const network::PhysicalNetwork& physical_network,
    const Solution& solution,
    const RecorderOptions options) {
    std::optional<PreparedCounter> local_physical_counter;
    const PreparedCounter* physical_counter = nullptr;
    if (prepared_physical_counter_.has_value() &&
        prepared_physical_network_ == &physical_network) {
        physical_counter = &*prepared_physical_counter_;
    } else {
        local_physical_counter.emplace(counter_.prepare(physical_network));
        physical_counter = &*local_physical_counter;
    }

    const CounterOptions counter_options{options.workers};
    state_.physical_available_resource = counter_number_as_double(
        physical_counter->calculate_sum_network_resource(
            true, true, counter_options));
    state_.physical_node_available_resource = counter_number_as_double(
        physical_counter->calculate_sum_network_resource(
            true, false, counter_options));
    state_.physical_link_available_resource = counter_number_as_double(
        physical_counter->calculate_sum_network_resource(
            false, true, counter_options));

    if (!initial_physical_state_.has_value()) {
        throw RecorderException(
            RecorderErrorCode::missing_initial_physical_state,
            RecorderOperation::update_state,
            "initial physical-network resources have not been counted",
            state_.event->event_id,
            solution.v_net_id);
    }
    if (initial_physical_state_->node_available_resource == 0.0) {
        throw RecorderException(
            RecorderErrorCode::zero_initial_node_resource,
            RecorderOperation::update_state,
            "initial physical node resource is zero",
            state_.event->event_id,
            solution.v_net_id);
    }
    state_.physical_node_resource_utilization = 1.0 -
        state_.physical_node_available_resource /
            initial_physical_state_->node_available_resource;

    if (initial_physical_state_->link_available_resource == 0.0) {
        throw RecorderException(
            RecorderErrorCode::zero_initial_link_resource,
            RecorderOperation::update_state,
            "initial physical link resource is zero",
            state_.event->event_id,
            solution.v_net_id);
    }
    state_.physical_link_resource_utilization = 1.0 -
        state_.physical_link_available_resource /
            initial_physical_state_->link_available_resource;

    if (state_.event->type == network::VirtualEventType::leave) {
        const RecorderRecord& arrival =
            record_by_virtual_network(solution.v_net_id);
        if (arrival.solution.result) {
            --state_.inservice_count;
            remove_membership(solution);
        }
        return;
    }

    virtual_network_event_indices_[solution.v_net_id] =
        state_.event->event_id;
    ++state_.virtual_network_count;
    if (!solution.result) {
        return;
    }

    ++state_.success_count;
    ++state_.inservice_count;
    state_.total_revenue += solution.v_net_revenue;
    state_.total_cost += solution.v_net_cost;
    state_.total_time_revenue += solution.v_net_time_revenue;
    state_.total_time_cost += solution.v_net_time_cost;
    state_.long_term_r2c_ratio = state_.total_cost != 0.0
        ? state_.total_revenue / state_.total_cost
        : 0.0;
    state_.long_term_time_r2c_ratio = state_.total_time_cost != 0.0
        ? state_.total_time_revenue / state_.total_time_cost
        : 0.0;
    if (!(state_.long_term_time_r2c_ratio <= 1.0)) {
        throw RecorderException(
            RecorderErrorCode::invalid_time_ratio,
            RecorderOperation::update_state,
            "long-term time revenue/cost ratio exceeds one or is NaN",
            state_.event->event_id,
            solution.v_net_id);
    }
    add_membership(solution);
}

void Recorder::add_membership(const Solution& solution) {
    for (const auto& entry : solution.node_slots.entries()) {
        const SolutionNodeId physical_node = entry.value;
        if (physical_node < 0 ||
            static_cast<std::uint64_t>(physical_node) >=
                physical_node_memberships_.size()) {
            throw RecorderException(
                RecorderErrorCode::invalid_physical_node,
                RecorderOperation::update_state,
                "solution refers to a physical node outside the dense domain",
                state_.event->event_id,
                solution.v_net_id,
                physical_node);
        }
        const std::size_t slot = static_cast<std::size_t>(physical_node);
        if (!physical_node_encountered_[slot]) {
            physical_node_encountered_[slot] = true;
            physical_node_encounter_order_.push_back(physical_node);
        }
        auto& memberships = physical_node_memberships_[slot];
        if (memberships.empty()) {
            ++live_membership_node_count_;
        }
        memberships.push_back(solution.v_net_id);
    }
    state_.running_physical_node_count = live_membership_node_count_;
}

void Recorder::remove_membership(const Solution& solution) {
    for (const auto& entry : solution.node_slots.entries()) {
        const SolutionNodeId physical_node = entry.value;
        if (physical_node < 0 ||
            static_cast<std::uint64_t>(physical_node) >=
                physical_node_memberships_.size()) {
            throw RecorderException(
                RecorderErrorCode::invalid_physical_node,
                RecorderOperation::update_state,
                "solution refers to a physical node outside the dense domain",
                state_.event->event_id,
                solution.v_net_id,
                physical_node);
        }
        auto& memberships =
            physical_node_memberships_[static_cast<std::size_t>(physical_node)];
        const auto found = std::find(
            memberships.begin(), memberships.end(), solution.v_net_id);
        if (found == memberships.end()) {
            throw RecorderException(
                RecorderErrorCode::membership_mismatch,
                RecorderOperation::update_state,
                "virtual network is absent from its physical-node membership",
                state_.event->event_id,
                solution.v_net_id,
                physical_node);
        }
        memberships.erase(found);
        if (memberships.empty()) {
            --live_membership_node_count_;
        }
    }
    state_.running_physical_node_count = live_membership_node_count_;
}

const RecorderRecord& Recorder::add_record(RecorderRecord record) {
    // ClassDict intentionally exposes shared recursive container types. The
    // recorder is a value-snapshot boundary, so force its documented deep
    // snapshot representation before publishing the history entry.
    record.extra = utils::ClassDict::from_dict(record.extra.to_dict());
    memory_.push_back(std::move(record));
    if (config_.temporary_records) {
        temporary_save_record(memory_.back());
    }
    return memory_.back();
}

const RecorderRecord& Recorder::record_by_event(
    const std::int64_t event_id) const {
    std::int64_t index = event_id;
    if (index < 0) {
        index += static_cast<std::int64_t>(memory_.size());
    }
    if (index < 0 ||
        static_cast<std::uint64_t>(index) >= memory_.size()) {
        throw RecorderException(
            RecorderErrorCode::record_index_out_of_range,
            RecorderOperation::lookup_record,
            "recorder history index is out of range",
            event_id);
    }
    return memory_[static_cast<std::size_t>(index)];
}

const RecorderRecord& Recorder::record_by_virtual_network(
    const SolutionNodeId virtual_network_id) const {
    const auto found =
        virtual_network_event_indices_.find(virtual_network_id);
    if (found == virtual_network_event_indices_.end()) {
        throw RecorderException(
            RecorderErrorCode::missing_virtual_network_record,
            RecorderOperation::lookup_record,
            "virtual network has no recorded arrival event",
            std::nullopt,
            virtual_network_id);
    }
    return record_by_event(found->second);
}

std::vector<SolutionNodeId> Recorder::running_physical_nodes() const {
    std::vector<SolutionNodeId> result;
    result.reserve(live_membership_node_count_);
    for (const SolutionNodeId physical_node :
         physical_node_encounter_order_) {
        const auto& memberships = physical_node_memberships_[
            static_cast<std::size_t>(physical_node)];
        if (!memberships.empty()) {
            result.push_back(physical_node);
        }
    }
    return result;
}

void Recorder::temporary_save_record(const RecorderRecord& record) {
    if (!config_.temporary_records || !temp_save_path_.has_value()) {
        throw RecorderException(
            RecorderErrorCode::temporary_saving_disabled,
            RecorderOperation::temporary_save_record,
            "temporary recorder CSV is disabled");
    }
    const std::vector<const RecorderRecord*> records{&record};
    csvio::append_csv(
        temp_save_path_->string(),
        record_frame(
            records,
            RecorderOptions{},
            RecorderOperation::temporary_save_record));
}

std::filesystem::path Recorder::save_records(
    const std::string_view filename,
    const RecorderOptions options) const {
    const std::filesystem::path destination =
        record_dir_ / checked_filename(filename, RecorderOperation::save_records);
    std::vector<const RecorderRecord*> records;
    records.reserve(memory_.size());
    for (const auto& record : memory_) {
        records.push_back(&record);
    }
    csvio::write_csv(
        destination.string(),
        record_frame(records, options, RecorderOperation::save_records));

    if (temp_save_path_.has_value()) {
        std::error_code ignored;
        std::filesystem::remove(*temp_save_path_, ignored);
    }
    return destination;
}

CounterSummary Recorder::summary_records() const {
    CounterRecords records;
    records.rows.reserve(memory_.size());
    records.reward_column_present = true;
    for (const auto& record : memory_) {
        CounterRecord row;
        row.success_count = record.state.success_count;
        row.virtual_network_count = record.state.virtual_network_count;
        row.event_type = record.state.event.has_value()
            ? record.state.event->type
            : network::VirtualEventType::leave;
        row.v_net_r2c_ratio = record.solution.v_net_r2c_ratio;
        row.total_time_revenue = record.state.total_time_revenue;
        row.total_time_cost = record.state.total_time_cost;
        row.virtual_network_arrival_time =
            record.solution.v_net_arrival_time;
        row.early_rejection = record.solution.early_rejection;
        row.place_result = record.solution.place_result;
        row.route_result = record.solution.route_result;
        row.total_cost = record.state.total_cost;
        row.total_revenue = record.state.total_revenue;
        row.physical_available_resource =
            record.state.physical_available_resource;
        row.physical_node_available_resource =
            record.state.physical_node_available_resource;
        row.physical_link_available_resource =
            record.state.physical_link_available_resource;
        row.inservice_count = record.state.inservice_count;
        row.hard_constraint_violation =
            record.solution.v_net_total_hard_constraint_violation;
        row.max_single_step_hard_constraint_violation =
            record.solution.v_net_max_single_step_hard_constraint_violation;
        row.reward = record.solution.v_net_reward;
        records.rows.push_back(std::move(row));
    }
    return virne::core::summary_records(records);
}

std::filesystem::path Recorder::append_summary(
    const CounterSummary& summary,
    const std::string_view filename) const {
    const std::filesystem::path destination = summary_dir_ /
        checked_filename(filename, RecorderOperation::append_summary);
    csvio::DataFrame frame;
    frame.columns = summary_columns();
    frame.rows.push_back(summary_values(summary));

    if (summary_extension_) {
        std::vector<RecorderSummaryColumn> extension_columns;
        summary_extension_->append_columns(*this, summary, extension_columns);
        for (auto& column : extension_columns) {
            if (column.name.empty()) {
                throw RecorderException(
                    RecorderErrorCode::unsupported_extra_value,
                    RecorderOperation::append_summary,
                    "summary extension returned an empty column name");
            }
            frame.columns.push_back(std::move(column.name));
            frame.rows.front().push_back(std::move(column.value));
        }
    }
    frame.validate();
    csvio::append_csv(destination.string(), frame);
    return destination;
}

void Recorder::set_summary_extension(
    std::shared_ptr<const RecorderSummaryExtension> extension) noexcept {
    summary_extension_ = std::move(extension);
}

const RecorderState& Recorder::state() const noexcept {
    return state_;
}

const std::optional<RecorderInitialPhysicalState>&
Recorder::initial_physical_state() const noexcept {
    return initial_physical_state_;
}

const std::vector<RecorderRecord>& Recorder::memory() const noexcept {
    return memory_;
}

const std::filesystem::path& Recorder::summary_dir() const noexcept {
    return summary_dir_;
}

const std::filesystem::path& Recorder::record_dir() const noexcept {
    return record_dir_;
}

const std::optional<std::filesystem::path>&
Recorder::temp_save_path() const noexcept {
    return temp_save_path_;
}

} // namespace virne::core
