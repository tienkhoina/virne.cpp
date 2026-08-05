#include "../virne/solver/learning/rl_core/rollout_returns.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl = virne::solver::learning::rl_core;

namespace {

constexpr std::size_t benchmark_steps = 4096U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct BenchmarkOptions {
    bool enabled = false;
    int threads = 1;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 25;
};

struct BenchmarkFixture {
    std::vector<double> items;
    std::vector<double> values;
    std::vector<std::uint8_t> dones;
};

struct OutputSummary {
    std::size_t length = 0U;
    std::uint64_t checksum = fnv_offset;
    double last_return = 0.0;
    double last_advantage = 0.0;
};

int positive_argument(const std::string& value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return parsed;
}

BenchmarkOptions parse_arguments(const int argc, char** argv) {
    BenchmarkOptions result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--benchmark") {
            result.enabled = true;
        } else if (argument == "--threads" && index + 1 < argc) {
            result.threads = positive_argument(argv[++index], "--threads");
        } else if (argument == "--warmups" && index + 1 < argc) {
            result.warmups = positive_argument(argv[++index], "--warmups");
        } else if (argument == "--repetitions" && index + 1 < argc) {
            result.repetitions = positive_argument(argv[++index], "--repetitions");
        } else if (argument == "--iterations" && index + 1 < argc) {
            result.iterations = positive_argument(argv[++index], "--iterations");
        } else {
            throw std::runtime_error(
                "unknown or incomplete harness argument: " + argument);
        }
    }
    return result;
}

void write_double_array(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << std::setprecision(17) << values[index];
    }
    std::cout << ']';
}

void update_checksum(std::uint64_t& checksum, const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    // Hash an explicit least-significant-byte-first representation so the
    // fixture checksum is independent of host byte order.
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        checksum ^= (bits >> shift) & 0xffU;
        checksum *= fnv_prime;
    }
}

OutputSummary summarize(const rl::ReturnAdvantageResult& result) {
    if (result.returns.size() != result.advantages.size()) {
        throw std::runtime_error("benchmark output lengths differ");
    }
    OutputSummary summary;
    summary.length = result.returns.size();
    for (const double value : result.returns) {
        update_checksum(summary.checksum, value);
    }
    for (const double value : result.advantages) {
        update_checksum(summary.checksum, value);
    }
    if (!result.returns.empty()) {
        summary.last_return = result.returns.back();
        summary.last_advantage = result.advantages.back();
    }
    return summary;
}

BenchmarkFixture make_benchmark_fixture() {
    BenchmarkFixture fixture;
    fixture.items.resize(benchmark_steps);
    fixture.values.resize(benchmark_steps);
    fixture.dones.assign(benchmark_steps, 0U);
    for (std::size_t index = 0U; index < benchmark_steps; ++index) {
        const auto item_integer = static_cast<std::int64_t>((index * 37U) % 101U) - 50;
        const auto value_integer = static_cast<std::int64_t>((index * 19U) % 79U) - 39;
        fixture.items[index] = static_cast<double>(item_integer) * 0.03125;
        fixture.values[index] = static_cast<double>(value_integer) * 0.015625;
        fixture.dones[index] = ((index + 1U) % 127U == 0U) ? 1U : 0U;
    }
    fixture.dones.back() = 1U;
    return fixture;
}

void write_output_summary(const OutputSummary& summary) {
    std::cout << "{\"length\":" << summary.length
              << ",\"checksum\":\"" << summary.checksum
              << "\",\"last_return\":" << std::setprecision(17)
              << summary.last_return
              << ",\"last_advantage\":" << summary.last_advantage << '}';
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        const std::vector<double> items{1.2, -0.4, 0.7, 0.1, -0.3};
        const std::vector<double> values{0.2, 0.3, -0.1, 0.4, 0.05};
        const std::vector<std::uint8_t> dones{0U, 0U, 1U, 0U, 1U};
        const std::vector<double> costs{1.0, -2.0, 0.5, 2.0, -1.0};
        constexpr double last_value = 0.25;
        constexpr double gamma = 0.91;
        constexpr double gae_lambda = 0.83;

        const auto gae = rl::compute_returns_with_gae(
            items, values, dones, last_value, gamma, gae_lambda);
        const auto mc = rl::compute_returns_with_mc(items, dones, gamma);
        const auto td = rl::compute_returns_with_td(
            items, values, dones, last_value, gamma);
        const auto reachability = rl::compute_cost_returns(
            costs, dones, 0.8, rl::CostReturnMethod::reachability);
        const auto cumulative = rl::compute_cost_returns(
            costs, dones, 0.8, rl::CostReturnMethod::cumulative);

        std::cout << "{\"component\":\"rollout_returns\",\"fixture_version\":1,"
                     "\"outputs\":{\"gae\":{\"returns\":";
        write_double_array(gae.returns);
        std::cout << ",\"advantages\":";
        write_double_array(gae.advantages);
        std::cout << "},\"mc\":";
        write_double_array(mc);
        std::cout << ",\"td\":";
        write_double_array(td);
        std::cout << ",\"cost_reachability\":";
        write_double_array(reachability);
        std::cout << ",\"cost_cumulative\":";
        write_double_array(cumulative);
        std::cout << '}';

        if (options.enabled) {
            const auto fixture = make_benchmark_fixture();
            rl::ReturnAdvantageResult output;
            for (int warmup = 0; warmup < options.warmups; ++warmup) {
                for (int iteration = 0; iteration < options.iterations; ++iteration) {
                    output = rl::compute_returns_with_gae(
                        fixture.items, fixture.values, fixture.dones,
                        0.25, 0.99, 0.98);
                }
            }
            std::vector<std::int64_t> samples_ns;
            samples_ns.reserve(static_cast<std::size_t>(options.repetitions));
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < options.iterations; ++iteration) {
                    output = rl::compute_returns_with_gae(
                        fixture.items, fixture.values, fixture.dones,
                        0.25, 0.99, 0.98);
                }
                samples_ns.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count());
            }
            const auto summary = summarize(output);
            std::cout << ",\"benchmark\":{\"name\":\"gae_returns_4096\","
                         "\"threads\":" << options.threads
                      << ",\"warmups\":" << options.warmups
                      << ",\"repetitions\":" << options.repetitions
                      << ",\"iterations\":" << options.iterations
                      << ",\"samples_ns\":[";
            for (std::size_t index = 0U; index < samples_ns.size(); ++index) {
                if (index != 0U) std::cout << ',';
                std::cout << samples_ns[index];
            }
            std::cout << "],\"output\":";
            write_output_summary(summary);
            std::cout << '}';
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rollout-returns harness error: " << error.what() << '\n';
        return 1;
    }
}
