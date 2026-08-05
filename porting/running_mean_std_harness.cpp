#include "../virne/solver/learning/rl_core/running_mean_std.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rl = virne::solver::learning::rl_core;

namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct Arguments {
    bool benchmark = false;
    int threads = 1;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 25;
};

struct Summary {
    std::size_t feature_count = 0U;
    std::vector<double> mean;
    std::vector<double> variance;
    double count = 0.0;
};

struct CaseResult {
    std::string name;
    Summary output;
};

struct BenchmarkResult {
    std::vector<std::int64_t> samples_ns;
    Summary output;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: vne_running_mean_std_harness [--benchmark] [--threads N]"
        " [--warmups N] [--repetitions N] [--iterations N]");
}

int positive_argument(const char* value, const char* option) {
    try {
        const auto parsed = std::stoll(value);
        if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            usage_error(std::string(option) + " requires a positive integer");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " requires a positive integer");
    }
}

Arguments parse_arguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto read_positive = [&](const char* option) {
            if (++index >= argc) {
                usage_error(std::string(option) + " requires a value");
            }
            return positive_argument(argv[index], option);
        };
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--threads") {
            result.threads = read_positive("--threads");
        } else if (argument == "--warmups") {
            result.warmups = read_positive("--warmups");
        } else if (argument == "--repetitions") {
            result.repetitions = read_positive("--repetitions");
        } else if (argument == "--iterations") {
            result.iterations = read_positive("--iterations");
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }
    return result;
}

Summary summarize(const rl::RunningMeanStd& stats) {
    return Summary{
        stats.feature_count(), stats.mean(), stats.variance(), stats.count()};
}

void update_byte(std::uint64_t& checksum, const std::uint8_t value) {
    checksum ^= value;
    checksum *= fnv_prime;
}

void update_u64(std::uint64_t& checksum, const std::uint64_t value) {
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        update_byte(
            checksum, static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void update_double(std::uint64_t& checksum, const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    update_u64(checksum, bits);
}

std::uint64_t summary_checksum(const Summary& summary) {
    std::uint64_t checksum = fnv_offset;
    update_u64(checksum, summary.feature_count);
    for (const double value : summary.mean) update_double(checksum, value);
    for (const double value : summary.variance) update_double(checksum, value);
    update_double(checksum, summary.count);
    return checksum;
}

std::vector<CaseResult> run_cases() {
    rl::RunningMeanStd two_batch(2U, 0.0);
    two_batch.update(
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, 4U);
    two_batch.update({9.0, 10.0, 11.0, 12.0}, 2U);

    rl::RunningMeanStd default_stats(3U);

    rl::RunningMeanStd broadcast(3U, 0.0);
    broadcast.update_from_moments({2.0}, {1.0}, 4.0);
    broadcast.update_from_moments(
        {4.0, 6.0, 8.0}, {0.5, 1.0, 2.0}, 2.0);

    const auto free_result = rl::update_mean_var_count_from_moments(
        {0.0, 0.0}, {1.0, 1.0}, 2.0, {4.0}, {0.0}, 2.0);

    return {
        {"two_batch", summarize(two_batch)},
        {"initial_default", summarize(default_stats)},
        {"moments_broadcast", summarize(broadcast)},
        {"free_function", Summary{
             free_result.mean.size(), free_result.mean,
             free_result.variance, free_result.count}},
    };
}

struct ErrorResults {
    bool invalid_feature_count = false;
    bool sample_length_mismatch = false;
    bool moment_length_mismatch = false;
};

ErrorResults run_error_cases() {
    ErrorResults result;
    try {
        static_cast<void>(rl::RunningMeanStd(0U));
    } catch (const rl::RunningMeanStdException& error) {
        result.invalid_feature_count =
            error.code() == rl::RunningMeanStdErrorCode::invalid_feature_count;
    }
    try {
        rl::RunningMeanStd stats(2U);
        stats.update({1.0, 2.0, 3.0}, 2U);
    } catch (const rl::RunningMeanStdException& error) {
        result.sample_length_mismatch =
            error.code() == rl::RunningMeanStdErrorCode::sample_length_mismatch;
    }
    try {
        rl::RunningMeanStd stats(2U);
        stats.update_from_moments({1.0, 2.0, 3.0}, {1.0}, 1.0);
    } catch (const rl::RunningMeanStdException& error) {
        result.moment_length_mismatch =
            error.code() == rl::RunningMeanStdErrorCode::moment_length_mismatch;
    }
    return result;
}

std::vector<double> benchmark_samples() {
    constexpr std::size_t rows = 4096U;
    constexpr std::size_t columns = 8U;
    std::vector<double> result(rows * columns);
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t column = 0U; column < columns; ++column) {
            const auto integer = static_cast<std::int64_t>(
                (row * 37U + column * 13U) % 257U) - 128;
            result[row * columns + column] =
                static_cast<double>(integer) * 0.03125;
        }
    }
    return result;
}

BenchmarkResult run_benchmark(const Arguments& arguments) {
    constexpr std::size_t columns = 8U;
    constexpr std::size_t rows = 4096U;
    const auto samples = benchmark_samples();
    rl::RunningMeanStd stats(columns);
    for (int warmup = 0; warmup < arguments.warmups; ++warmup) {
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            stats.update(samples, rows);
        }
    }
    BenchmarkResult result;
    result.samples_ns.reserve(static_cast<std::size_t>(arguments.repetitions));
    for (int repetition = 0; repetition < arguments.repetitions; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
            stats.update(samples, rows);
        }
        result.samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin)
                .count());
    }
    result.output = summarize(stats);
    return result;
}

void write_double_vector(
    std::ostream& output,
    const std::vector<double>& values) {
    output << '[' << std::setprecision(17);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_summary(std::ostream& output, const Summary& summary) {
    output << "{\"feature_count\":" << summary.feature_count
           << ",\"mean\":";
    write_double_vector(output, summary.mean);
    output << ",\"variance\":";
    write_double_vector(output, summary.variance);
    output << ",\"count\":" << std::setprecision(17) << summary.count
           << ",\"checksum\":\"" << summary_checksum(summary) << "\"}";
}

void write_cases(std::ostream& output, const std::vector<CaseResult>& cases) {
    output << '[';
    for (std::size_t index = 0U; index < cases.size(); ++index) {
        if (index != 0U) output << ',';
        output << "{\"name\":\"" << cases[index].name
               << "\",\"output\":";
        write_summary(output, cases[index].output);
        output << '}';
    }
    output << ']';
}

void write_errors(std::ostream& output, const ErrorResults& errors) {
    output << "{\"invalid_feature_count\":"
           << (errors.invalid_feature_count ? "true" : "false")
           << ",\"sample_length_mismatch\":"
           << (errors.sample_length_mismatch ? "true" : "false")
           << ",\"moment_length_mismatch\":"
           << (errors.moment_length_mismatch ? "true" : "false") << '}';
}

void write_benchmark(
    std::ostream& output,
    const Arguments& arguments,
    const BenchmarkResult& benchmark) {
    output << "{\"name\":\"running_mean_std_4096x8\",\"threads\":"
           << arguments.threads << ",\"warmups\":" << arguments.warmups
           << ",\"repetitions\":" << arguments.repetitions
           << ",\"iterations\":" << arguments.iterations
           << ",\"samples_ns\":[";
    for (std::size_t index = 0U; index < benchmark.samples_ns.size(); ++index) {
        if (index != 0U) output << ',';
        output << benchmark.samples_ns[index];
    }
    output << "],\"output\":";
    write_summary(output, benchmark.output);
    output << '}';
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto cases = run_cases();
        const auto errors = run_error_cases();
        std::cout << "{\"component\":\"running_mean_std\","
                     "\"fixture_version\":1,\"cases\":";
        write_cases(std::cout, cases);
        std::cout << ",\"errors\":";
        write_errors(std::cout, errors);
        if (arguments.benchmark) {
            const auto benchmark = run_benchmark(arguments);
            std::cout << ",\"benchmark\":";
            write_benchmark(std::cout, arguments, benchmark);
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "running-mean-std harness error: " << error.what() << '\n';
        return 2;
    }
}
