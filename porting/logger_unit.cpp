#include "logger.h"

#include "csv.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

namespace core = virne::core;
namespace fs = std::filesystem;

using core::Logger;
using core::LoggerConfig;
using core::LoggerEntry;
using core::LoggerErrorCode;
using core::LoggerEventView;
using core::LoggerException;
using core::LoggerLevel;
using core::LoggerMetricRow;
using core::LoggerOperation;
using core::LoggerOptions;
using core::LoggerSink;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
LoggerException expect_logger_error(
    Callable&& callable,
    LoggerErrorCode code,
    LoggerOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const LoggerException& error)
    {
        expect(error.code() == code, "logger error code mismatch");
        expect(error.operation() == operation, "logger operation mismatch");
        expect(!std::string_view(error.what()).empty(),
               "logger diagnostic is empty");
        return error;
    }
    throw std::runtime_error("expected LoggerException");
}

std::string read_binary(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    expect(stream.is_open(), "failed to open logger output");
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

std::size_t count_substring(
    std::string_view value,
    std::string_view needle)
{
    expect(!needle.empty(), "empty substring count needle");
    std::size_t count = 0U;
    std::size_t offset = 0U;
    while ((offset = value.find(needle, offset)) != std::string_view::npos)
    {
        ++count;
        offset += needle.size();
    }
    return count;
}

class TemporaryRoot
{
public:
    TemporaryRoot()
    {
        const auto stamp = std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
        const fs::path parent = fs::temp_directory_path();
        for (std::uint32_t suffix = 0U; suffix < 1000U; ++suffix)
        {
            path_ = parent /
                ("virne_logger_unit_" + std::to_string(stamp) + "_" +
                 std::to_string(suffix));
            std::error_code error;
            if (fs::create_directory(path_, error))
            {
                return;
            }
            if (error)
            {
                throw std::runtime_error(
                    "failed to create logger unit temporary directory: " +
                    error.message());
            }
        }
        throw std::runtime_error("failed to allocate unique logger unit path");
    }

    ~TemporaryRoot()
    {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    TemporaryRoot(const TemporaryRoot&) = delete;
    TemporaryRoot& operator=(const TemporaryRoot&) = delete;

    const fs::path& path() const noexcept
    {
        return path_;
    }

private:
    fs::path path_;
};

class StderrCapture
{
public:
    StderrCapture()
        : previous_(std::cerr.rdbuf(stream_.rdbuf()))
    {
    }

    ~StderrCapture()
    {
        std::cerr.rdbuf(previous_);
    }

    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    std::string str() const
    {
        return stream_.str();
    }

private:
    std::ostringstream stream_;
    std::streambuf* previous_;
};

LoggerConfig make_config(
    const fs::path& root,
    std::string run_id)
{
    LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "typed-logger";
    config.run_id = std::move(run_id);
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = false;
    config.backends.file = true;
    config.level = LoggerLevel::debug;
    config.log_show_interval = 1U;
    return config;
}

struct CapturedEvent
{
    std::string message;
    LoggerLevel level = LoggerLevel::info;
    std::optional<std::int64_t> step;
    std::optional<std::vector<std::optional<double>>> values;

    friend bool operator==(
        const CapturedEvent& left,
        const CapturedEvent& right)
    {
        return left.message == right.message &&
            left.level == right.level &&
            left.step == right.step &&
            left.values == right.values;
    }
};

class CapturingSink final : public LoggerSink
{
public:
    void write(
        const Logger&,
        const LoggerEventView& event) override
    {
        CapturedEvent copy;
        copy.message = event.message;
        copy.level = event.level;
        copy.step = event.step;
        if (event.metric_values != nullptr)
        {
            copy.values = *event.metric_values;
        }
        events.push_back(std::move(copy));
    }

    void close() noexcept override
    {
        ++close_count;
    }

    std::vector<CapturedEvent> events;
    std::size_t close_count = 0U;
};

void test_directory_path_and_constructor_validation(
    const fs::path& root)
{
    const LoggerConfig config = make_config(root, "paths");
    Logger logger(config);
    const fs::path expected_dir =
        root / "typed-logger" / "paths" / "logs";
    const fs::path expected_file = expected_dir / "run.log";
    expect(logger.config().solver_name == "typed-logger" &&
               logger.config().run_id == "paths",
           "logger did not retain direct config fields");
    expect(logger.log_dir() == expected_dir,
           "logger run directory join mismatch");
    expect(logger.log_file_path() ==
               std::optional<fs::path>{expected_file},
           "logger file path mismatch");
    expect(fs::is_directory(expected_dir),
           "logger did not create its directory");
    expect(fs::is_regular_file(expected_file),
           "logger file backend was not opened eagerly");
    expect(read_binary(expected_file).empty(),
           "new logger file is not empty");
    logger.close();

    LoggerConfig escaping = make_config(root, "bad-component");
    escaping.log_dir_name = "../escape";
    expect_logger_error(
        [&]
        {
            Logger invalid(std::move(escaping));
        },
        LoggerErrorCode::invalid_path_component,
        LoggerOperation::construct);

    LoggerConfig missing_name = make_config(root, "missing-name");
    missing_name.log_file_name.clear();
    expect_logger_error(
        [&]
        {
            Logger invalid(std::move(missing_name));
        },
        LoggerErrorCode::missing_log_file_name,
        LoggerOperation::construct);
}

void test_exact_ansi_levels(const fs::path& root)
{
    LoggerConfig config = make_config(root, "ansi");
    config.backends.console = true;
    StderrCapture capture;
    Logger logger(config);
    logger.debug("d");
    logger.info("i");
    logger.warning("w");
    logger.error("e");
    logger.critical("c");
    logger.close();

    const std::string expected =
        "\x1b[36mDEBUG   \x1b[0m d\x1b[0m\n"
        "\x1b[32mINFO    \x1b[0m i\x1b[0m\n"
        "\x1b[33mWARNING \x1b[0m w\x1b[0m\n"
        "\x1b[31mERROR   \x1b[0m e\x1b[0m\n"
        "\x1b[1;31mCRITICAL\x1b[0m c\x1b[0m\n";
    expect(capture.str() == expected,
           "console ANSI level bytes differ from the fixed format");
    expect(read_binary(*logger.log_file_path()) == expected,
           "file ANSI level bytes differ from console bytes");
}

void test_threshold_empty_and_multiline(const fs::path& root)
{
    LoggerConfig config = make_config(root, "filter");
    config.level = LoggerLevel::warning;
    Logger logger(config);
    logger.debug("debug-hidden");
    logger.info("info-hidden");
    logger.warning("");
    logger.error("first\nsecond");
    logger.close();

    const std::string expected =
        "\x1b[31mERROR   \x1b[0m first\nsecond\x1b[0m\n";
    expect(read_binary(*logger.log_file_path()) == expected,
           "threshold, empty, or multiline behavior drifted");
}

void test_metric_schema_and_csv_rows(const fs::path& root)
{
    LoggerConfig config = make_config(root, "metrics");
    config.level = LoggerLevel::critical;
    config.log_show_interval = 100U;
    Logger logger(config);
    const auto ids = logger.register_metrics({"loss", "score"});
    expect(ids.size() == 2U && ids[0U].value == 0U && ids[1U].value == 1U,
           "metric IDs are not dense registration-order slots");
    expect(logger.register_metric("loss") == ids[0U],
           "duplicate metric registration changed its ID");
    expect(logger.metric_names() ==
               std::vector<std::string>({"loss", "score"}),
           "metric schema order drifted");

    logger.log(LoggerEntry{
        "", LoggerLevel::info,
        LoggerMetricRow{std::int64_t{1}, {1.5, std::nullopt}}});
    logger.log(LoggerEntry{
        "", LoggerLevel::info,
        LoggerMetricRow{std::int64_t{2}, {std::nullopt, -2.0}}});

    const fs::path metric_file = logger.log_dir() / "training_info.csv";
    const std::string expected =
        "update_time,loss,score\r\n"
        "1,1.5,\r\n"
        "2,,-2.0\r\n";
    expect(read_binary(metric_file) == expected,
           "metric CSV schema/header/rows differ");

    expect(logger.register_metric("loss") == ids[0U],
           "locked schema rejected an existing metric");
    expect_logger_error(
        [&]
        {
            static_cast<void>(logger.register_metric("new_metric"));
        },
        LoggerErrorCode::metric_schema_locked,
        LoggerOperation::register_metric);
    expect(logger.metric_names().size() == 2U,
           "schema-lock failure mutated metric names");
    expect(read_binary(metric_file) == expected,
           "schema-lock failure mutated the CSV");

    expect_logger_error(
        [&]
        {
            logger.log(LoggerEntry{
                "must-not-emit", LoggerLevel::critical,
                LoggerMetricRow{std::int64_t{3}, {1.0}}});
        },
        LoggerErrorCode::metric_width_mismatch,
        LoggerOperation::log);
    logger.close();
    expect(read_binary(metric_file) == expected,
           "width failure changed the stable CSV");
    expect(read_binary(*logger.log_file_path()).empty(),
           "metric validation failure emitted a standard line");
}

void test_progress_filter_and_order(const fs::path& root)
{
    LoggerConfig config = make_config(root, "progress");
    config.log_show_interval = 3U;
    Logger logger(config);
    logger.register_metrics({
        "accuracy",
        "train_loss",
        "probability",
        "RETURN",
        "critic_value",
        "penalty_term",
        "return_mean"});

    logger.log(LoggerEntry{
        "tick",
        LoggerLevel::warning,
        LoggerMetricRow{
            std::int64_t{6},
            {100.0, 1.25, 4.0, 8.0, -2.0, 0.0, 3.5}}});
    logger.close();

    const std::string expected =
        "\x1b[33mWARNING \x1b[0m tick\x1b[0m\n"
        "\x1b[33mWARNING \x1b[0m Update time: 000006 | "
        "+1.2500 & +4.0000 & -2.0000 & +0.0000 & +3.5000\x1b[0m\n";
    expect(read_binary(*logger.log_file_path()) == expected,
           "progress metric filtering or ID order drifted");
}

void test_deferred_errors_preserve_partial_rows(const fs::path& root)
{
    const auto run_case = [&](std::string run_id,
                              std::size_t interval,
                              std::optional<std::int64_t> step,
                              LoggerErrorCode expected_code,
                              std::string_view expected_step)
    {
        LoggerConfig config = make_config(root, std::move(run_id));
        config.log_show_interval = interval;
        Logger logger(config);
        logger.register_metric("loss");
        const auto sink = std::make_shared<CapturingSink>();
        logger.attach_sink(sink);

        expect_logger_error(
            [&]
            {
                logger.log(LoggerEntry{
                    "before-error",
                    LoggerLevel::info,
                    LoggerMetricRow{step, {1.0}}});
            },
            expected_code,
            LoggerOperation::log);

        const std::string expected_log =
            "\x1b[32mINFO    \x1b[0m before-error\x1b[0m\n";
        expect(read_binary(*logger.log_file_path()) == expected_log,
               "deferred error lost or reordered the standard message");
        const std::string expected_csv =
            "update_time,loss\r\n" + std::string(expected_step) +
            ",1.0\r\n";
        expect(read_binary(logger.log_dir() / "training_info.csv") ==
                   expected_csv,
               "deferred error did not preserve its partial CSV row");
        expect(sink->events.empty(),
               "deferred error reached optional sinks");
        logger.close();
        expect(sink->close_count == 1U,
               "deferred-error sink was not closed once");
    };

    run_case(
        "missing-step", 1U, std::nullopt,
        LoggerErrorCode::invalid_step, "");
    run_case(
        "zero-interval", 0U, std::int64_t{2},
        LoggerErrorCode::invalid_show_interval, "2");
}

void test_close_and_sink_idempotence(const fs::path& root)
{
    const auto sink = std::make_shared<CapturingSink>();
    {
        LoggerConfig config = make_config(root, "close");
        config.backends.file = false;
        config.log_file_name.clear();
        Logger logger(config);
        logger.attach_sink(sink);
        logger.info("sink-only");
        expect(sink->events == std::vector<CapturedEvent>({
                   CapturedEvent{
                       "sink-only", LoggerLevel::info,
                       std::nullopt, std::nullopt}}),
               "sink did not receive the typed event exactly once");
        logger.close();
        logger.close();
        expect(sink->close_count == 1U,
               "explicit close is not idempotent");
        expect_logger_error(
            [&]
            {
                logger.info("closed");
            },
            LoggerErrorCode::logger_closed,
            LoggerOperation::log);
    }
    expect(sink->close_count == 1U,
           "destructor closed an already closed sink again");
}

struct BatchArtifacts
{
    std::string log_bytes;
    std::string csv_bytes;
    std::vector<CapturedEvent> events;
};

BatchArtifacts run_batch(
    const fs::path& root,
    std::size_t workers)
{
    LoggerConfig config = make_config(
        root, "batch-" + std::to_string(workers));
    config.level = LoggerLevel::info;
    config.log_show_interval = 3U;
    Logger logger(config);
    logger.register_metrics({"loss", "score", "policy_value"});
    const auto sink = std::make_shared<CapturingSink>();
    logger.attach_sink(sink);

    std::vector<LoggerEntry> entries;
    entries.reserve(19U);
    for (std::int64_t index = 0; index < 19; ++index)
    {
        LoggerEntry entry;
        entry.message = index % 4 == 0
            ? std::string{}
            : "entry-" + std::to_string(index) +
                (index == 5 ? "\ncontinued" : "");
        entry.level = index % 5 == 0
            ? LoggerLevel::debug
            : (index % 5 == 1
                ? LoggerLevel::info
                : (index % 5 == 2
                    ? LoggerLevel::warning
                    : (index % 5 == 3
                        ? LoggerLevel::error
                        : LoggerLevel::critical)));
        LoggerMetricRow row;
        row.step = index;
        row.values = {
            static_cast<double>(index) + 0.25,
            index % 3 == 1
                ? std::optional<double>{}
                : std::optional<double>{
                    static_cast<double>(index) * -0.5},
            static_cast<double>(index) + 10.0};
        entry.metrics = std::move(row);
        entries.push_back(std::move(entry));
    }

    logger.log_batch(entries, LoggerOptions{workers});
    logger.close();
    expect(sink->close_count == 1U,
           "batch sink close count mismatch");
    return BatchArtifacts{
        read_binary(*logger.log_file_path()),
        read_binary(logger.log_dir() / "training_info.csv"),
        sink->events};
}

void test_batch_worker_equivalence(const fs::path& root)
{
    const BatchArtifacts baseline = run_batch(root, 0U);
    expect(!baseline.log_bytes.empty() && !baseline.csv_bytes.empty(),
           "batch fixture did not exercise standard and metric output");
    expect(baseline.events.size() == 19U,
           "batch fixture did not reach every sink event");
    for (const std::size_t workers : {1U, 2U, 8U})
    {
        const BatchArtifacts actual = run_batch(root, workers);
        expect(actual.log_bytes == baseline.log_bytes,
               "log_batch workers changed standard output bytes");
        expect(actual.csv_bytes == baseline.csv_bytes,
               "log_batch workers changed CSV bytes");
        expect(actual.events == baseline.events,
               "log_batch workers changed sink order/content");
    }
}

void test_concurrent_rows_and_single_header(const fs::path& root)
{
    LoggerConfig config = make_config(root, "concurrent");
    config.level = LoggerLevel::critical;
    config.log_show_interval = 1000000U;
    Logger logger(config);
    logger.register_metric("loss");

    constexpr std::size_t caller_count = 8U;
    constexpr std::size_t rows_per_caller = 25U;
    std::vector<std::future<void>> callers;
    callers.reserve(caller_count);
    for (std::size_t caller = 0U; caller < caller_count; ++caller)
    {
        callers.emplace_back(std::async(
            std::launch::async,
            [&logger, caller]
            {
                for (std::size_t offset = 0U;
                     offset < rows_per_caller;
                     ++offset)
                {
                    const std::int64_t id = static_cast<std::int64_t>(
                        caller * rows_per_caller + offset + 1U);
                    logger.log(LoggerEntry{
                        "",
                        LoggerLevel::info,
                        LoggerMetricRow{
                            id,
                            {static_cast<double>(id) + 0.5}}});
                }
            }));
    }
    for (auto& caller : callers)
    {
        caller.get();
    }
    logger.close();

    const fs::path csv_path = logger.log_dir() / "training_info.csv";
    const std::string raw = read_binary(csv_path);
    expect(count_substring(raw, "update_time,loss\r\n") == 1U,
           "concurrent logger duplicated the CSV header");

    const csvio::DataFrame frame = csvio::read_csv(csv_path.string());
    expect(frame.columns ==
               std::vector<std::string>({"update_time", "loss"}),
           "concurrent logger changed the ordered CSV schema");
    expect(frame.nrows() == caller_count * rows_per_caller,
           "concurrent logger lost or duplicated metric rows");
    const std::size_t step_id = frame.column_index("update_time");
    const std::size_t loss_id = frame.column_index("loss");
    std::vector<bool> seen(caller_count * rows_per_caller, false);
    for (std::size_t row = 0U; row < frame.nrows(); ++row)
    {
        const std::int64_t step = frame.int64_at(row, step_id);
        expect(step >= 1 &&
                   step <= static_cast<std::int64_t>(seen.size()),
               "concurrent CSV contains an out-of-range step");
        const std::size_t dense = static_cast<std::size_t>(step - 1);
        expect(!seen[dense], "concurrent CSV contains a duplicate step");
        seen[dense] = true;
        expect(frame.double_at(row, loss_id) ==
                   static_cast<double>(step) + 0.5,
               "concurrent CSV paired a metric with the wrong step");
    }
    for (const bool value : seen)
    {
        expect(value, "concurrent CSV omitted a step");
    }
    expect(read_binary(*logger.log_file_path()).empty(),
           "thresholded concurrent metrics emitted standard log bytes");
}

} // namespace

int main()
{
    try
    {
        const TemporaryRoot root;
        const auto run = [](std::string_view name, auto&& test)
        {
            try
            {
                test();
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(
                    std::string(name) + ": " + error.what());
            }
        };

        run("directory/path", [&]
        {
            test_directory_path_and_constructor_validation(root.path());
        });
        run("ANSI levels", [&]
        {
            test_exact_ansi_levels(root.path());
        });
        run("threshold/empty/multiline", [&]
        {
            test_threshold_empty_and_multiline(root.path());
        });
        run("metric schema/CSV", [&]
        {
            test_metric_schema_and_csv_rows(root.path());
        });
        run("progress filter/order", [&]
        {
            test_progress_filter_and_order(root.path());
        });
        run("deferred partial rows", [&]
        {
            test_deferred_errors_preserve_partial_rows(root.path());
        });
        run("close/sink", [&]
        {
            test_close_and_sink_idempotence(root.path());
        });
        run("batch workers", [&]
        {
            test_batch_worker_equivalence(root.path());
        });
        run("concurrent rows/header", [&]
        {
            test_concurrent_rows_and_single_header(root.path());
        });
        std::cout << "logger unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "logger unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
