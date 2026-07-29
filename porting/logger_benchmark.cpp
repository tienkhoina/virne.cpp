#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace core = virne::core;
namespace fs = std::filesystem;

constexpr std::size_t metric_count = 8U;
constexpr std::size_t entry_count = 4096U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

std::size_t parse_size(const char* text) {
    const std::string value(text);
    std::size_t position = 0U;
    const unsigned long long parsed = std::stoull(value, &position, 10);
    if (position != value.size() ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(
            "Logger benchmark argument is not a valid size");
    }
    return static_cast<std::size_t>(parsed);
}

std::vector<std::string> make_metric_names() {
    std::vector<std::string> result;
    result.reserve(metric_count);
    for (std::size_t id = 0U; id < metric_count; ++id) {
        result.push_back("metric_" + std::to_string(id));
    }
    return result;
}

double metric_value(const std::size_t entry, const std::size_t id) noexcept {
    return static_cast<double>(
        (entry * (id + 3U) + id * 17U) % 1009U);
}

std::vector<core::LoggerEntry> make_entries() {
    std::vector<core::LoggerEntry> result(entry_count);
    for (std::size_t index = 0U; index < entry_count; ++index) {
        core::LoggerMetricRow row;
        row.step = static_cast<std::int64_t>(index);
        row.values.resize(metric_count);
        for (std::size_t id = 0U; id < metric_count; ++id) {
            row.values[id] = metric_value(index, id);
        }
        result[index].message.clear();
        result[index].level = core::LoggerLevel::info;
        result[index].metrics = std::move(row);
    }
    return result;
}

std::string read_binary(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error(
            "Logger benchmark failed to open output: " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

std::size_t validate_csv(const std::string& contents) {
    const std::string expected_header =
        "update_time,metric_0,metric_1,metric_2,metric_3,metric_4,"
        "metric_5,metric_6,metric_7\r\n";
    if (contents.size() < expected_header.size() ||
        contents.compare(0U, expected_header.size(), expected_header) != 0) {
        throw std::runtime_error("Logger benchmark CSV header drifted");
    }
    const std::size_t line_feeds = static_cast<std::size_t>(
        std::count(contents.begin(), contents.end(), '\n'));
    const std::size_t carriage_returns = static_cast<std::size_t>(
        std::count(contents.begin(), contents.end(), '\r'));
    if (line_feeds != entry_count + 1U ||
        carriage_returns != line_feeds ||
        contents.size() < 2U ||
        contents[contents.size() - 2U] != '\r' ||
        contents.back() != '\n') {
        throw std::runtime_error("Logger benchmark CSV row framing drifted");
    }
    return line_feeds;
}

std::uint64_t fingerprint(const std::string_view value) noexcept {
    std::uint64_t result = fnv_offset;
    for (const char raw_byte : value) {
        result =
            (result ^ static_cast<unsigned char>(raw_byte)) * fnv_prime;
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: logger_benchmark <output_root> <workers>");
        }
        const fs::path output_root(argv[1]);
        const std::size_t workers = parse_size(argv[2]);

        core::LoggerConfig config;
        config.save_root_dir = output_root;
        config.solver_name = "logger-benchmark";
        config.run_id = "workers-" + std::to_string(workers);
        config.log_dir_name = "logs";
        config.log_file_name = "run.log";
        config.backends.console = false;
        config.backends.file = true;
        config.level = core::LoggerLevel::critical;
        config.log_show_interval = entry_count + 1U;

        core::Logger logger(std::move(config));
        const std::vector<std::string> names = make_metric_names();
        const std::vector<core::LoggerMetricId> ids =
            logger.register_metrics(names);
        if (ids.size() != metric_count || logger.metric_names() != names) {
            throw std::runtime_error(
                "Logger benchmark metric schema registration drifted");
        }
        for (std::size_t id = 0U; id < metric_count; ++id) {
            if (ids[id].value != id) {
                throw std::runtime_error(
                    "Logger benchmark metric IDs are not dense");
            }
        }
        const std::vector<core::LoggerEntry> entries = make_entries();

        const auto started = std::chrono::steady_clock::now();
        logger.log_batch(entries, core::LoggerOptions{workers});
        const auto stopped = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                stopped - started).count();

        logger.close();
        const fs::path metric_path = logger.log_dir() / "training_info.csv";
        const std::string csv_bytes = read_binary(metric_path);
        const std::size_t csv_rows = validate_csv(csv_bytes);
        const std::string log_bytes = read_binary(*logger.log_file_path());
        if (!log_bytes.empty()) {
            throw std::runtime_error(
                "Logger benchmark empty messages changed run.log");
        }

        std::cout
            << "protocol=1\n"
            << "kind=logger_dense_metric_batch_csv\n"
            << "semantics=exact_python_ordered_csv_v1\n"
            << "backend=file\n"
            << "message=empty\n"
            << "entry_count=" << entry_count << '\n'
            << "metric_count=" << metric_count << '\n'
            << "workers=" << workers << '\n'
            << "elapsed_ns=" << elapsed << '\n'
            << "checksum=" << fingerprint(csv_bytes) << '\n'
            << "output_bytes=" << csv_bytes.size() << '\n'
            << "csv_rows=" << csv_rows << '\n'
            << "log_bytes=" << log_bytes.size() << '\n'
            << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "logger_benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
