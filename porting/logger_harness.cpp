#include "logger.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace core = virne::core;

using core::Logger;
using core::LoggerConfig;
using core::LoggerEntry;
using core::LoggerErrorCode;
using core::LoggerException;
using core::LoggerLevel;
using core::LoggerMetricRow;
using core::LoggerOptions;

class StderrCapture
{
public:
    StderrCapture()
        : original_(std::cerr.rdbuf(stream_.rdbuf()))
    {
    }

    ~StderrCapture()
    {
        std::cerr.rdbuf(original_);
    }

    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    std::string bytes() const
    {
        return stream_.str();
    }

private:
    std::ostringstream stream_;
    std::streambuf* original_;
};

struct Observation
{
    std::string standard_error;
    std::string log_file;
    std::string metric_file;
    std::string error = "none";
};

std::string read_binary_if_present(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input)
    {
        return {};
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

std::string hex_bytes(const std::string_view bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.resize(bytes.size() * 2U);
    for (std::size_t index = 0U; index < bytes.size(); ++index)
    {
        const auto value = static_cast<unsigned char>(bytes[index]);
        result[index * 2U] = digits[value >> 4U];
        result[index * 2U + 1U] = digits[value & 0x0fU];
    }
    return result;
}

std::string error_token(const LoggerException& error)
{
    switch (error.code())
    {
    case LoggerErrorCode::invalid_step:
        return "invalid_step";
    case LoggerErrorCode::invalid_show_interval:
        return "invalid_interval";
    default:
        return "logger_error_" +
            std::to_string(static_cast<unsigned int>(error.code()));
    }
}

LoggerConfig make_config(
    const std::filesystem::path& root,
    std::string case_name)
{
    LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = std::move(case_name);
    config.run_id = "run";
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = true;
    config.backends.file = true;
    config.level = LoggerLevel::debug;
    config.log_show_interval = 1U;
    return config;
}

Observation observe(
    const LoggerConfig& config,
    const std::function<void(Logger&)>& action)
{
    Observation result;
    {
        StderrCapture capture;
        try
        {
            Logger logger(config);
            action(logger);
            logger.close();
        }
        catch (const LoggerException& error)
        {
            result.error = error_token(error);
        }
        catch (const std::exception& error)
        {
            result.error = std::string("std_error:") + error.what();
        }
        result.standard_error = capture.bytes();
    }

    const std::filesystem::path directory =
        config.save_root_dir /
        config.solver_name /
        config.run_id /
        config.log_dir_name;
    result.log_file = read_binary_if_present(directory / config.log_file_name);
    result.metric_file =
        read_binary_if_present(directory / "training_info.csv");
    return result;
}

std::string payload(const Observation& value)
{
    return
        "stderr=" + hex_bytes(value.standard_error) +
        ";file=" + hex_bytes(value.log_file) +
        ";csv=" + hex_bytes(value.metric_file) +
        ";error=" + value.error;
}

void emit(const std::string_view name, const Observation& value)
{
    std::cout << name << '\t' << payload(value) << '\n';
}

LoggerEntry metric_entry(
    std::string message,
    const LoggerLevel level,
    const std::optional<std::int64_t> step,
    std::vector<std::optional<double>> values)
{
    LoggerEntry result;
    result.message = std::move(message);
    result.level = level;
    result.metrics = LoggerMetricRow{step, std::move(values)};
    return result;
}

std::vector<LoggerEntry> batch_entries()
{
    std::vector<LoggerEntry> result;
    result.push_back(metric_entry(
        "batch-a",
        LoggerLevel::info,
        1,
        {1.0, 0.25, -2.0}));
    result.push_back(metric_entry(
        "",
        LoggerLevel::warning,
        2,
        {1.5, 0.5, -1.0}));
    result.push_back(metric_entry(
        "filtered-debug",
        LoggerLevel::debug,
        0,
        {2.0, 0.75, 0.0}));
    result.push_back(LoggerEntry{
        "batch-d", LoggerLevel::error, std::nullopt});
    return result;
}

void differential(const std::filesystem::path& root)
{
    {
        LoggerConfig config = make_config(root, "ansi_console_file");
        emit("ansi_console_file", observe(config, [](Logger& logger) {
            logger.debug("debug");
            logger.info("info");
            logger.warning("warning");
            logger.error("error");
            logger.critical("critical");
        }));
    }
    {
        LoggerConfig config = make_config(root, "level_filter");
        config.level = LoggerLevel::warning;
        emit("level_filter", observe(config, [](Logger& logger) {
            logger.debug("drop-debug");
            logger.info("drop-info");
            logger.warning("keep-warning");
            logger.error("keep-error");
            logger.critical("keep-critical");
        }));
    }
    {
        LoggerConfig config = make_config(root, "message_csv_progress");
        config.log_show_interval = 2U;
        emit("message_csv_progress", observe(config, [](Logger& logger) {
            logger.register_metrics({
                "loss,\"quoted\"", "accuracy", "return_value"});
            logger.log(metric_entry(
                "epoch \"one\", ok",
                LoggerLevel::info,
                2,
                {1.25, std::nullopt, -0.5}));
            logger.log(metric_entry(
                "",
                LoggerLevel::info,
                3,
                {2.0, 0.75, 4.25}));
        }));
    }
    {
        LoggerConfig config = make_config(root, "partial_missing_step");
        config.log_show_interval = 2U;
        emit("partial_missing_step", observe(config, [](Logger& logger) {
            logger.register_metric("loss");
            logger.log(metric_entry(
                "before missing step",
                LoggerLevel::info,
                std::nullopt,
                {0.5}));
        }));
    }
    {
        LoggerConfig config = make_config(root, "partial_zero_interval");
        config.log_show_interval = 0U;
        emit("partial_zero_interval", observe(config, [](Logger& logger) {
            logger.register_metric("loss");
            logger.log(metric_entry(
                "before zero interval",
                LoggerLevel::warning,
                1,
                {-0.25}));
        }));
    }
    {
        LoggerConfig config =
            make_config(root, "step_zero_with_zero_interval");
        config.log_show_interval = 0U;
        emit(
            "step_zero_with_zero_interval",
            observe(config, [](Logger& logger) {
                logger.register_metric("loss");
                logger.log(metric_entry(
                    "zero step",
                    LoggerLevel::info,
                    0,
                    {3.0}));
            }));
    }

    for (const std::size_t workers : {1U, 2U, 8U})
    {
        const std::string name =
            "batch_workers_" + std::to_string(workers);
        LoggerConfig config = make_config(root, name);
        config.backends.console = false;
        config.level = LoggerLevel::info;
        config.log_show_interval = 2U;
        emit(name, observe(config, [workers](Logger& logger) {
            logger.register_metrics({
                "loss_total", "accuracy", "value_head"});
            logger.log_batch(batch_entries(), LoggerOptions{workers});
        }));
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "differential")
        {
            differential(std::filesystem::path(argv[2]));
            return 0;
        }
        throw std::invalid_argument(
            "usage: logger_harness differential <temporary-root>");
    }
    catch (const std::exception& error)
    {
        std::cerr << "logger harness: " << error.what() << '\n';
        return 1;
    }
}
