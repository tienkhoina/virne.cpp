#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace virne::core {

enum class LoggerLevel : std::uint8_t {
    debug,
    info,
    warning,
    error,
    critical,
};

struct LoggerBackends {
    bool console = true;
    bool file = false;
};

struct LoggerOptions {
    std::size_t workers = 1U;
};

struct LoggerConfig {
    std::filesystem::path save_root_dir;
    std::string solver_name;
    std::string run_id;
    std::string log_dir_name = "logs";
    std::string log_file_name = "run.log";
    LoggerBackends backends;
    LoggerLevel level = LoggerLevel::warning;
    std::size_t log_show_interval = 1U;
    std::string project_name;
    std::string experiment_name;
};

struct LoggerMetricId {
    std::uint32_t value = 0U;

    friend bool operator==(
        const LoggerMetricId left,
        const LoggerMetricId right) noexcept {
        return left.value == right.value;
    }
};

struct LoggerMetricRow {
    std::optional<std::int64_t> step;
    std::vector<std::optional<double>> values;
};

struct LoggerEntry {
    std::string message;
    LoggerLevel level = LoggerLevel::info;
    std::optional<LoggerMetricRow> metrics;
};

class Logger;

struct LoggerEventView {
    std::string_view message;
    LoggerLevel level = LoggerLevel::info;
    std::optional<std::int64_t> step;
    const std::vector<std::optional<double>>* metric_values = nullptr;
};

class LoggerSink {
public:
    virtual ~LoggerSink() = default;

    virtual void write(
        const Logger& logger,
        const LoggerEventView& event) = 0;
    virtual void close() noexcept = 0;
};

enum class LoggerErrorCode : std::uint8_t {
    invalid_path_component,
    missing_log_file_name,
    invalid_metric_name,
    metric_id_space_exhausted,
    metric_schema_locked,
    metric_width_mismatch,
    invalid_metric_value,
    invalid_show_interval,
    invalid_step,
    null_sink,
    logger_closed,
    filesystem_failure,
    file_open_failure,
    file_write_failure,
};

enum class LoggerOperation : std::uint8_t {
    construct,
    register_metric,
    log,
    log_batch,
    attach_sink,
    close,
};

class LoggerException final : public std::runtime_error {
public:
    LoggerException(
        LoggerErrorCode code,
        LoggerOperation operation,
        std::string message,
        std::optional<std::size_t> input_index = std::nullopt,
        std::optional<LoggerMetricId> metric_id = std::nullopt);

    LoggerErrorCode code() const noexcept;
    LoggerOperation operation() const noexcept;
    const std::optional<std::size_t>& input_index() const noexcept;
    const std::optional<LoggerMetricId>& metric_id() const noexcept;

private:
    LoggerErrorCode code_;
    LoggerOperation operation_;
    std::optional<std::size_t> input_index_;
    std::optional<LoggerMetricId> metric_id_;
};

class Logger {
public:
    explicit Logger(LoggerConfig config);
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    LoggerMetricId register_metric(std::string_view name);
    std::vector<LoggerMetricId> register_metrics(
        const std::vector<std::string>& names);

    void log(const LoggerEntry& entry);
    void log_batch(
        const std::vector<LoggerEntry>& entries,
        LoggerOptions options = {});

    void debug(std::string_view message);
    void info(std::string_view message);
    void warning(std::string_view message);
    void error(std::string_view message);
    void critical(std::string_view message);

    void attach_sink(std::shared_ptr<LoggerSink> sink);
    void close() noexcept;

    const LoggerConfig& config() const noexcept;
    const std::filesystem::path& log_dir() const noexcept;
    const std::optional<std::filesystem::path>& log_file_path() const noexcept;
    const std::vector<std::string>& metric_names() const noexcept;

private:
    struct FormattedEntry {
        std::optional<std::string> message_line;
        std::optional<std::string> interval_line;
        std::optional<std::vector<std::string>> metric_row;
        std::optional<LoggerErrorCode> deferred_error;
    };

    FormattedEntry format_entry(
        const LoggerEntry& entry,
        LoggerOperation operation,
        std::optional<std::size_t> input_index) const;
    void emit_standard_line(const std::string& line);
    void append_metric_rows(
        const std::vector<const std::vector<std::string>*>& rows);
    void emit_sink(const LoggerEntry& entry);
    void check_open(LoggerOperation operation) const;

    LoggerConfig config_;
    std::filesystem::path log_dir_;
    std::optional<std::filesystem::path> log_file_path_;
    std::filesystem::path metric_file_path_;
    std::ofstream log_file_;

    std::vector<std::string> metric_names_;
    std::vector<bool> metric_interval_visible_;
    std::unordered_map<std::string, LoggerMetricId> metric_ids_;
    std::vector<std::shared_ptr<LoggerSink>> sinks_;

    bool metric_schema_locked_ = false;
    bool closed_ = false;
    mutable std::mutex mutex_;
};

} // namespace virne::core
