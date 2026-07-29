#include "logger.h"

#include "../utils/utils_config.h"
#include "../../csv/csv.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <system_error>
#include <thread>
#include <utility>

namespace virne::core {
namespace {

std::filesystem::path checked_component(
    const std::string_view raw,
    const LoggerOperation operation) {
    if (raw.empty() || raw == "." || raw == ".." ||
        raw.find('/') != std::string_view::npos ||
        raw.find('\\') != std::string_view::npos) {
        throw LoggerException(
            LoggerErrorCode::invalid_path_component,
            operation,
            "logger path component must be one non-empty filename");
    }
    const std::filesystem::path value{std::string(raw)};
    if (value.has_root_path() || value.filename() != value) {
        throw LoggerException(
            LoggerErrorCode::invalid_path_component,
            operation,
            "logger path component escapes its configured directory");
    }
    return value;
}

std::uint8_t level_rank(const LoggerLevel level) noexcept {
    switch (level) {
    case LoggerLevel::debug:
        return 10U;
    case LoggerLevel::info:
        return 20U;
    case LoggerLevel::warning:
        return 30U;
    case LoggerLevel::error:
        return 40U;
    case LoggerLevel::critical:
        return 50U;
    }
    return 20U;
}

std::string_view level_name(const LoggerLevel level) noexcept {
    switch (level) {
    case LoggerLevel::debug:
        return "DEBUG";
    case LoggerLevel::info:
        return "INFO";
    case LoggerLevel::warning:
        return "WARNING";
    case LoggerLevel::error:
        return "ERROR";
    case LoggerLevel::critical:
        return "CRITICAL";
    }
    return "INFO";
}

std::string_view level_color(const LoggerLevel level) noexcept {
    switch (level) {
    case LoggerLevel::debug:
        return "\x1b[36m";
    case LoggerLevel::info:
        return "\x1b[32m";
    case LoggerLevel::warning:
        return "\x1b[33m";
    case LoggerLevel::error:
        return "\x1b[31m";
    case LoggerLevel::critical:
        return "\x1b[1;31m";
    }
    return "\x1b[32m";
}

bool level_enabled(
    const LoggerLevel value,
    const LoggerLevel threshold) noexcept {
    return level_rank(value) >= level_rank(threshold);
}

std::string standard_line(
    const LoggerLevel level,
    const std::string_view message) {
    const std::string_view name = level_name(level);
    std::string result;
    result.reserve(message.size() + 32U);
    result += level_color(level);
    result += name;
    result.append(8U - name.size(), ' ');
    result += "\x1b[0m ";
    result += message;
    // colorlog's default reset=True appends a final reset when the formatted
    // message itself does not already end in one.
    result += "\x1b[0m\n";
    return result;
}

std::string format_integer(const std::int64_t value) {
    char buffer[32];
    const auto result = std::to_chars(
        std::begin(buffer), std::end(buffer), value);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("failed to serialize logger integer");
    }
    return std::string(buffer, result.ptr);
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
        throw std::runtime_error("failed to serialize logger metric");
    }
    std::string output(buffer, result.ptr);
    if (output.find_first_of(".eE") == std::string::npos) {
        output += ".0";
    }
    return output;
}

std::string format_interval_value(const double value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::showpos << std::fixed << std::setprecision(4) << value;
    return stream.str();
}

std::string format_step(const std::int64_t step) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setfill('0') << std::setw(6) << step;
    return stream.str();
}

bool interval_metric_name(const std::string_view name) noexcept {
    constexpr std::array<std::string_view, 5U> fragments{
        "loss", "prob", "return", "penalty", "value"};
    return std::any_of(
        fragments.begin(), fragments.end(),
        [&](const std::string_view fragment) {
            return name.find(fragment) != std::string_view::npos;
        });
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
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        return;
    }

    run_range(begin, count);
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

} // namespace

LoggerException::LoggerException(
    const LoggerErrorCode code,
    const LoggerOperation operation,
    std::string message,
    const std::optional<std::size_t> input_index,
    const std::optional<LoggerMetricId> metric_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      input_index_(input_index),
      metric_id_(metric_id) {}

LoggerErrorCode LoggerException::code() const noexcept {
    return code_;
}

LoggerOperation LoggerException::operation() const noexcept {
    return operation_;
}

const std::optional<std::size_t>&
LoggerException::input_index() const noexcept {
    return input_index_;
}

const std::optional<LoggerMetricId>&
LoggerException::metric_id() const noexcept {
    return metric_id_;
}

Logger::Logger(LoggerConfig config)
    : config_(std::move(config)) {
    checked_component(config_.solver_name, LoggerOperation::construct);
    checked_component(config_.run_id, LoggerOperation::construct);
    checked_component(config_.log_dir_name, LoggerOperation::construct);
    if (config_.log_show_interval > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        throw LoggerException(
            LoggerErrorCode::invalid_show_interval,
            LoggerOperation::construct,
            "logger display interval must fit int64");
    }
    if (config_.backends.file && config_.log_file_name.empty()) {
        throw LoggerException(
            LoggerErrorCode::missing_log_file_name,
            LoggerOperation::construct,
            "file logger backend requires a filename");
    }
    if (!config_.log_file_name.empty()) {
        checked_component(config_.log_file_name, LoggerOperation::construct);
    }

    const auto run_directory = utils::get_run_id_dir(utils::RunDirectoryInput{
        config_.save_root_dir,
        config_.solver_name,
        config_.run_id,
    });
    log_dir_ = run_directory / config_.log_dir_name;
    metric_file_path_ = log_dir_ / "training_info.csv";
    if (!config_.log_file_name.empty()) {
        log_file_path_ = log_dir_ / config_.log_file_name;
    }

    try {
        std::filesystem::create_directories(log_dir_);
    } catch (const std::filesystem::filesystem_error& error) {
        throw LoggerException(
            LoggerErrorCode::filesystem_failure,
            LoggerOperation::construct,
            error.what());
    }

    if (config_.backends.file) {
        log_file_.open(
            *log_file_path_,
            std::ios::out | std::ios::app | std::ios::binary);
        if (!log_file_) {
            throw LoggerException(
                LoggerErrorCode::file_open_failure,
                LoggerOperation::construct,
                "failed to open logger file backend");
        }
    }
}

Logger::~Logger() {
    close();
}

LoggerMetricId Logger::register_metric(const std::string_view name) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_open(LoggerOperation::register_metric);
    const std::string key(name);
    const auto existing = metric_ids_.find(key);
    if (existing != metric_ids_.end()) {
        return existing->second;
    }
    if (name.empty() || name == "update_time") {
        throw LoggerException(
            LoggerErrorCode::invalid_metric_name,
            LoggerOperation::register_metric,
            "logger metric name is empty or reserved");
    }
    if (metric_schema_locked_) {
        throw LoggerException(
            LoggerErrorCode::metric_schema_locked,
            LoggerOperation::register_metric,
            "logger metric schema is locked after its first CSV row");
    }
    if (metric_names_.size() >= static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw LoggerException(
            LoggerErrorCode::metric_id_space_exhausted,
            LoggerOperation::register_metric,
            "logger metric ID space is exhausted");
    }
    const LoggerMetricId id{
        static_cast<std::uint32_t>(metric_names_.size())};
    metric_names_.push_back(key);
    try {
        metric_interval_visible_.push_back(interval_metric_name(name));
        metric_ids_.emplace(key, id);
    } catch (...) {
        if (metric_interval_visible_.size() > metric_names_.size() - 1U) {
            metric_interval_visible_.pop_back();
        }
        metric_names_.pop_back();
        throw;
    }
    return id;
}

std::vector<LoggerMetricId> Logger::register_metrics(
    const std::vector<std::string>& names) {
    std::vector<LoggerMetricId> result;
    result.reserve(names.size());
    for (const auto& name : names) {
        result.push_back(register_metric(name));
    }
    return result;
}

Logger::FormattedEntry Logger::format_entry(
    const LoggerEntry& entry,
    const LoggerOperation operation,
    const std::optional<std::size_t> input_index) const {
    FormattedEntry result;
    const bool standard_enabled =
        config_.backends.console || config_.backends.file;
    const bool enabled = level_enabled(entry.level, config_.level);
    if (standard_enabled && enabled && !entry.message.empty()) {
        result.message_line = standard_line(entry.level, entry.message);
    }
    if (!entry.metrics.has_value()) {
        return result;
    }
    if (entry.metrics->values.size() != metric_names_.size()) {
        throw LoggerException(
            LoggerErrorCode::metric_width_mismatch,
            operation,
            "logger metric row width differs from registered schema",
            input_index);
    }

    if (standard_enabled) {
        result.metric_row.emplace();
        result.metric_row->reserve(metric_names_.size() + 1U);
        result.metric_row->push_back(entry.metrics->step.has_value()
            ? format_integer(*entry.metrics->step)
            : std::string{});
        for (const auto& value : entry.metrics->values) {
            result.metric_row->push_back(
                value.has_value() ? format_double(*value) : std::string{});
        }
    }

    if (!standard_enabled) {
        return result;
    }
    if (!entry.metrics->step.has_value()) {
        result.deferred_error = LoggerErrorCode::invalid_step;
        return result;
    }
    if (*entry.metrics->step == 0) {
        return result;
    }
    if (config_.log_show_interval == 0U) {
        result.deferred_error = LoggerErrorCode::invalid_show_interval;
        return result;
    }
    if (*entry.metrics->step %
            static_cast<std::int64_t>(config_.log_show_interval) != 0) {
        return result;
    }

    std::string message = "Update time: ";
    message += format_step(*entry.metrics->step);
    message += " | ";
    bool first = true;
    for (std::size_t id = 0U; id < entry.metrics->values.size(); ++id) {
        if (!metric_interval_visible_[id]) {
            continue;
        }
        if (!entry.metrics->values[id].has_value()) {
            result.deferred_error = LoggerErrorCode::invalid_metric_value;
            result.interval_line.reset();
            return result;
        }
        if (!first) {
            message += " & ";
        }
        first = false;
        message += format_interval_value(*entry.metrics->values[id]);
    }
    if (enabled) {
        result.interval_line = standard_line(entry.level, message);
    }
    return result;
}

void Logger::emit_standard_line(const std::string& line) {
    if (config_.backends.console) {
        std::cerr.write(line.data(), static_cast<std::streamsize>(line.size()));
        std::cerr.flush();
    }
    if (config_.backends.file) {
        log_file_.write(line.data(), static_cast<std::streamsize>(line.size()));
        log_file_.flush();
        if (!log_file_) {
            throw LoggerException(
                LoggerErrorCode::file_write_failure,
                LoggerOperation::log,
                "failed to write logger file backend");
        }
    }
}

void Logger::append_metric_rows(
    const std::vector<const std::vector<std::string>*>& rows) {
    if (rows.empty()) {
        return;
    }
    csvio::DataFrame frame;
    frame.columns.reserve(metric_names_.size() + 1U);
    frame.columns.emplace_back("update_time");
    frame.columns.insert(
        frame.columns.end(), metric_names_.begin(), metric_names_.end());
    frame.rows.reserve(rows.size());
    for (const auto* row : rows) {
        frame.rows.push_back(*row);
    }
    csvio::append_csv(metric_file_path_.string(), frame);
}

void Logger::emit_sink(const LoggerEntry& entry) {
    const LoggerEventView view{
        entry.message,
        entry.level,
        entry.metrics.has_value() ? entry.metrics->step : std::nullopt,
        entry.metrics.has_value() ? &entry.metrics->values : nullptr,
    };
    for (const auto& sink : sinks_) {
        sink->write(*this, view);
    }
}

void Logger::check_open(const LoggerOperation operation) const {
    if (closed_) {
        throw LoggerException(
            LoggerErrorCode::logger_closed,
            operation,
            "logger is already closed");
    }
}

void Logger::log(const LoggerEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_open(LoggerOperation::log);
    const FormattedEntry formatted =
        format_entry(entry, LoggerOperation::log, std::nullopt);
    if (formatted.message_line.has_value()) {
        emit_standard_line(*formatted.message_line);
    }
    if (formatted.metric_row.has_value()) {
        metric_schema_locked_ = true;
        const std::vector<const std::vector<std::string>*> rows{
            &*formatted.metric_row};
        append_metric_rows(rows);
    }
    if (formatted.deferred_error.has_value()) {
        throw LoggerException(
            *formatted.deferred_error,
            LoggerOperation::log,
            *formatted.deferred_error == LoggerErrorCode::invalid_step
                ? "logger metric progress requires an integer step"
                : *formatted.deferred_error ==
                        LoggerErrorCode::invalid_metric_value
                    ? "logger progress metric is missing"
                    : "logger display interval is zero");
    }
    if (formatted.interval_line.has_value()) {
        emit_standard_line(*formatted.interval_line);
    }
    emit_sink(entry);
}

void Logger::log_batch(
    const std::vector<LoggerEntry>& entries,
    const LoggerOptions options) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_open(LoggerOperation::log_batch);
    std::vector<FormattedEntry> formatted(entries.size());
    parallel_indexed(entries.size(), options.workers, [&](const std::size_t index) {
        formatted[index] = format_entry(
            entries[index], LoggerOperation::log_batch, index);
    });

    // Invalid progress inputs follow the scalar Python partial-state order.
    // Keep the optimized single-append path exclusively for valid batches.
    const auto deferred = std::find_if(
        formatted.begin(), formatted.end(),
        [](const FormattedEntry& value) {
            return value.deferred_error.has_value();
        });
    if (deferred != formatted.end()) {
        const std::size_t error_index = static_cast<std::size_t>(
            std::distance(formatted.begin(), deferred));
        for (std::size_t index = 0U; index <= error_index; ++index) {
            if (formatted[index].message_line.has_value()) {
                emit_standard_line(*formatted[index].message_line);
            }
            if (formatted[index].metric_row.has_value()) {
                metric_schema_locked_ = true;
                const std::vector<const std::vector<std::string>*> rows{
                    &*formatted[index].metric_row};
                append_metric_rows(rows);
            }
            if (formatted[index].deferred_error.has_value()) {
                throw LoggerException(
                    *formatted[index].deferred_error,
                    LoggerOperation::log_batch,
                    *formatted[index].deferred_error ==
                            LoggerErrorCode::invalid_step
                        ? "logger metric progress requires an integer step"
                        : *formatted[index].deferred_error ==
                                LoggerErrorCode::invalid_metric_value
                            ? "logger progress metric is missing"
                            : "logger display interval is zero",
                    index);
            }
            if (formatted[index].interval_line.has_value()) {
                emit_standard_line(*formatted[index].interval_line);
            }
            emit_sink(entries[index]);
        }
    }

    std::vector<const std::vector<std::string>*> metric_rows;
    metric_rows.reserve(entries.size());
    for (const auto& value : formatted) {
        if (value.metric_row.has_value()) {
            metric_rows.push_back(&*value.metric_row);
        }
    }
    if (!metric_rows.empty()) {
        metric_schema_locked_ = true;
        append_metric_rows(metric_rows);
    }

    for (std::size_t index = 0U; index < entries.size(); ++index) {
        if (formatted[index].message_line.has_value()) {
            emit_standard_line(*formatted[index].message_line);
        }
        if (formatted[index].interval_line.has_value()) {
            emit_standard_line(*formatted[index].interval_line);
        }
        emit_sink(entries[index]);
    }
}

void Logger::debug(const std::string_view message) {
    log(LoggerEntry{std::string(message), LoggerLevel::debug, std::nullopt});
}

void Logger::info(const std::string_view message) {
    log(LoggerEntry{std::string(message), LoggerLevel::info, std::nullopt});
}

void Logger::warning(const std::string_view message) {
    log(LoggerEntry{std::string(message), LoggerLevel::warning, std::nullopt});
}

void Logger::error(const std::string_view message) {
    log(LoggerEntry{std::string(message), LoggerLevel::error, std::nullopt});
}

void Logger::critical(const std::string_view message) {
    log(LoggerEntry{
        std::string(message), LoggerLevel::critical, std::nullopt});
}

void Logger::attach_sink(std::shared_ptr<LoggerSink> sink) {
    if (!sink) {
        throw LoggerException(
            LoggerErrorCode::null_sink,
            LoggerOperation::attach_sink,
            "cannot attach a null logger sink");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    check_open(LoggerOperation::attach_sink);
    sinks_.push_back(std::move(sink));
}

void Logger::close() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        if (log_file_.is_open()) {
            log_file_.flush();
            log_file_.close();
        }
        for (const auto& sink : sinks_) {
            sink->close();
        }
    } catch (...) {
        // Destruction and explicit close mirror Python's best-effort teardown.
    }
}

const LoggerConfig& Logger::config() const noexcept {
    return config_;
}

const std::filesystem::path& Logger::log_dir() const noexcept {
    return log_dir_;
}

const std::optional<std::filesystem::path>&
Logger::log_file_path() const noexcept {
    return log_file_path_;
}

const std::vector<std::string>& Logger::metric_names() const noexcept {
    return metric_names_;
}

} // namespace virne::core
