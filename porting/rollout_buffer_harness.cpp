#include "../virne/solver/learning/rl_core/rollout_buffer.h"

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
constexpr std::size_t benchmark_selection = 1024U;
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
    rl::RolloutBuffer buffer;
    std::vector<std::size_t> indices;
};

struct BufferSummary {
    std::size_t size = 0U;
    std::size_t curr_idx = 0U;
    std::uint64_t checksum = fnv_offset;
    rl::ObservationId first_observation = 0U;
    rl::ObservationId last_observation = 0U;
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

template <typename T>
void write_integer_array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << static_cast<std::int64_t>(values[index]);
    }
    std::cout << ']';
}

void write_observation_array(const std::vector<rl::ObservationId>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void write_double_array(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << std::setprecision(17) << values[index];
    }
    std::cout << ']';
}

void write_split_summaries(const std::vector<rl::RolloutBuffer>& buffers) {
    std::cout << '[';
    for (std::size_t index = 0U; index < buffers.size(); ++index) {
        if (index != 0U) std::cout << ',';
        const auto& buffer = buffers[index];
        std::cout << "{\"size\":" << buffer.size()
                  << ",\"curr_idx\":" << buffer.curr_idx
                  << ",\"observations\":";
        write_observation_array(buffer.observations);
        std::cout << '}';
    }
    std::cout << ']';
}

rl::RolloutBuffer make_buffer() {
    rl::RolloutBuffer result;
    const std::vector<double> rewards{1.2, -0.4, 0.7, 0.1, -0.3};
    const std::vector<double> values{0.2, 0.3, -0.1, 0.4, 0.05};
    const std::vector<std::uint8_t> dones{0U, 0U, 1U, 0U, 1U};
    for (std::size_t index = 0U; index < rewards.size(); ++index) {
        result.add(
            100U + index,
            static_cast<std::int64_t>(10U + index),
            rewards[index],
            dones[index] != 0U,
            -0.1 * static_cast<double>(index + 1U),
            values[index],
            200U + index,
            true);
    }
    result.extend(rl::RolloutExtensionId::feasibility_flags);
    result.extend("feasibility_budgets");
    result.feasibility_flags = {1.0, 1.0, 0.0, 1.0, 0.0};
    result.feasibility_budgets = {0.5, 0.5, 1.0, 1.5, 2.0};
    return result;
}

rl::RolloutBufferWithCost make_cost_buffer() {
    rl::RolloutBufferWithCost result;
    const auto source = make_buffer();
    for (std::size_t index = 0U; index < source.size(); ++index) {
        result.add(
            source.observations[index], source.actions[index],
            source.rewards[index], source.dones[index] != 0U,
            source.logprobs[index], source.values[index],
            source.next_observations[index], true);
    }
    result.costs = {1.0, -2.0, 0.5, 2.0, -1.0};
    return result;
}

rl::RolloutBufferWithAvgReturn make_avg_buffer() {
    rl::RolloutBufferWithAvgReturn result;
    const std::vector<double> rewards{1.0, -1.0, 2.0};
    const std::vector<double> values{0.1, 0.2, 0.3};
    for (std::size_t index = 0U; index < rewards.size(); ++index) {
        result.add(index, static_cast<std::int64_t>(index), rewards[index],
                   index == 2U, -0.2, values[index]);
    }
    return result;
}

void update_byte(std::uint64_t& checksum, const std::uint8_t value) {
    checksum ^= value;
    checksum *= fnv_prime;
}

void update_u64(std::uint64_t& checksum, const std::uint64_t value) {
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        update_byte(checksum, static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void update_double(std::uint64_t& checksum, const double value) {
    std::uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    update_u64(checksum, bits);
}

BufferSummary summarize(const rl::RolloutBuffer& buffer) {
    BufferSummary result;
    result.size = buffer.size();
    result.curr_idx = buffer.curr_idx;
    for (const auto value : buffer.observations) update_u64(result.checksum, value);
    for (const auto value : buffer.actions) {
        update_u64(result.checksum, static_cast<std::uint64_t>(value));
    }
    for (const double value : buffer.rewards) update_double(result.checksum, value);
    for (const auto value : buffer.dones) update_byte(result.checksum, value);
    for (const auto value : buffer.next_observations) {
        update_u64(result.checksum, value);
    }
    for (const double value : buffer.logprobs) update_double(result.checksum, value);
    for (const double value : buffer.values) update_double(result.checksum, value);
    for (const double value : buffer.feasibility_flags) {
        update_double(result.checksum, value);
    }
    for (const double value : buffer.feasibility_budgets) {
        update_double(result.checksum, value);
    }
    if (!buffer.observations.empty()) {
        result.first_observation = buffer.observations.front();
        result.last_observation = buffer.observations.back();
    }
    return result;
}

BenchmarkFixture make_benchmark_fixture() {
    BenchmarkFixture fixture;
    fixture.buffer.extend(rl::RolloutExtensionId::feasibility_flags);
    fixture.buffer.extend(rl::RolloutExtensionId::feasibility_budgets);
    fixture.buffer.feasibility_flags.reserve(benchmark_steps);
    fixture.buffer.feasibility_budgets.reserve(benchmark_steps);
    for (std::size_t index = 0U; index < benchmark_steps; ++index) {
        const auto reward_integer =
            static_cast<std::int64_t>((index * 37U) % 101U) - 50;
        const auto value_integer =
            static_cast<std::int64_t>((index * 19U) % 79U) - 39;
        fixture.buffer.add(
            100000U + index,
            static_cast<std::int64_t>(index % 97U),
            static_cast<double>(reward_integer) * 0.03125,
            ((index + 1U) % 127U == 0U) || (index + 1U == benchmark_steps),
            -static_cast<double>((index % 67U) + 1U) * 0.015625,
            static_cast<double>(value_integer) * 0.0078125,
            200000U + index,
            true);
        fixture.buffer.feasibility_flags.push_back(index % 3U == 0U ? 1.0 : 0.0);
        fixture.buffer.feasibility_budgets.push_back(
            static_cast<double>((index * 13U) % 97U) * 0.0625);
    }
    fixture.indices.reserve(benchmark_selection);
    for (std::size_t index = 0U; index < benchmark_selection; ++index) {
        fixture.indices.push_back((index * 17U) % benchmark_steps);
    }
    return fixture;
}

void write_buffer_summary(const BufferSummary& summary) {
    std::cout << "{\"size\":" << summary.size
              << ",\"curr_idx\":" << summary.curr_idx
              << ",\"checksum\":\"" << summary.checksum
              << "\",\"first_observation\":" << summary.first_observation
              << ",\"last_observation\":" << summary.last_observation << '}';
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        const auto lifecycle = make_buffer();
        const auto selected = lifecycle.get_subbuffer({4U, 1U});
        const auto ratio_splits = lifecycle.split_with_ratios({0.4, 0.6});
        const auto instance_splits = lifecycle.split_with_instance();
        auto merged = ratio_splits.front();
        const auto merge_curr_idx_before = merged.curr_idx;
        merged.merge(ratio_splits.back());

        auto gae = make_buffer();
        gae.compute_returns_and_advantages(
            0.25, 0.91, 0.83, rl::RolloutBufferMethod::gae);
        auto mc = make_buffer();
        mc.compute_returns_and_advantages(
            0.25, 0.91, 0.83, rl::RolloutBufferMethod::mc);
        auto td = make_buffer();
        td.compute_returns_and_advantages(
            0.25, 0.91, 0.83, rl::RolloutBufferMethod::td);

        auto cost = make_cost_buffer();
        cost.compute_cost_returns(0.8, rl::CostReturnMethod::reachability);
        const auto reachability = cost.cost_returns;
        cost.compute_cost_returns(0.8, rl::CostReturnMethod::cumulative);
        const auto cost_selected = cost.get_cost_subbuffer({3U, 0U});

        auto avg_gae = make_avg_buffer();
        avg_gae.compute_returns_and_advantages(
            0.4, 0.9, 0.8, rl::RolloutBufferMethod::gae);
        auto avg_td = make_avg_buffer();
        avg_td.compute_returns_and_advantages(
            0.4, 0.9, 0.8, rl::RolloutBufferMethod::td);

        std::cout << "{\"component\":\"rollout_buffer\",\"fixture_version\":1,"
                     "\"outputs\":{\"lifecycle\":{\"size\":"
                  << lifecycle.size() << ",\"curr_idx\":" << lifecycle.curr_idx
                  << ",\"observations\":";
        write_observation_array(lifecycle.observations);
        std::cout << ",\"actions\":";
        write_integer_array(lifecycle.actions);
        std::cout << ",\"rewards\":";
        write_double_array(lifecycle.rewards);
        std::cout << ",\"dones\":";
        write_integer_array(lifecycle.dones);
        std::cout << ",\"next_observations\":";
        write_observation_array(lifecycle.next_observations);
        std::cout << ",\"logprobs\":";
        write_double_array(lifecycle.logprobs);
        std::cout << ",\"values\":";
        write_double_array(lifecycle.values);
        std::cout << ",\"feasibility_flags_enabled\":"
                  << (lifecycle.feasibility_flags_enabled ? "true" : "false")
                  << ",\"feasibility_budgets_enabled\":"
                  << (lifecycle.feasibility_budgets_enabled ? "true" : "false")
                  << ",\"feasibility_flags\":";
        write_double_array(lifecycle.feasibility_flags);
        std::cout << ",\"feasibility_budgets\":";
        write_double_array(lifecycle.feasibility_budgets);

        std::cout << "},\"selected\":{\"size\":" << selected.size()
                  << ",\"curr_idx\":" << selected.curr_idx
                  << ",\"observations\":";
        write_observation_array(selected.observations);
        std::cout << ",\"actions\":";
        write_integer_array(selected.actions);
        std::cout << ",\"feasibility_flags\":";
        write_double_array(selected.feasibility_flags);
        std::cout << ",\"feasibility_budgets\":";
        write_double_array(selected.feasibility_budgets);

        std::cout << "},\"ratio_splits\":";
        write_split_summaries(ratio_splits);
        std::cout << ",\"instance_splits\":";
        write_split_summaries(instance_splits);
        std::cout << ",\"merge\":{\"curr_idx_before\":"
                  << merge_curr_idx_before << ",\"curr_idx_after\":"
                  << merged.curr_idx << ",\"size\":" << merged.size()
                  << ",\"observations\":";
        write_observation_array(merged.observations);
        std::cout << ",\"feasibility_flags\":";
        write_double_array(merged.feasibility_flags);
        std::cout << ",\"feasibility_budgets\":";
        write_double_array(merged.feasibility_budgets);

        std::cout << "},\"returns\":{\"gae\":{\"returns\":";
        write_double_array(gae.returns);
        std::cout << ",\"advantages\":";
        write_double_array(gae.advantages);
        std::cout << "},\"mc\":{\"returns\":";
        write_double_array(mc.returns);
        std::cout << ",\"advantages\":";
        write_double_array(mc.advantages);
        std::cout << "},\"td\":{\"returns\":";
        write_double_array(td.returns);
        std::cout << ",\"advantages\":";
        write_double_array(td.advantages);

        std::cout << "}},\"cost\":{\"reachability\":";
        write_double_array(reachability);
        std::cout << ",\"after_cumulative\":{\"costs\":";
        write_double_array(cost.costs);
        std::cout << ",\"cost_returns\":";
        write_double_array(cost.cost_returns);
        std::cout << ",\"selected_costs\":";
        write_double_array(cost_selected.costs);
        std::cout << ",\"selected_cost_returns\":";
        write_double_array(cost_selected.cost_returns);

        std::cout << "}},\"average_return\":{\"gae\":{\"dones\":";
        write_integer_array(avg_gae.dones);
        std::cout << ",\"rewards\":";
        write_double_array(avg_gae.rewards);
        std::cout << ",\"returns\":";
        write_double_array(avg_gae.returns);
        std::cout << ",\"advantages\":";
        write_double_array(avg_gae.advantages);
        std::cout << "},\"td\":{\"dones\":";
        write_integer_array(avg_td.dones);
        std::cout << ",\"rewards\":";
        write_double_array(avg_td.rewards);
        std::cout << ",\"returns\":";
        write_double_array(avg_td.returns);
        std::cout << ",\"advantages\":";
        write_double_array(avg_td.advantages);
        std::cout << "}}}";

        if (options.enabled) {
            const auto fixture = make_benchmark_fixture();
            rl::RolloutBuffer output;
            for (int warmup = 0; warmup < options.warmups; ++warmup) {
                for (int iteration = 0; iteration < options.iterations; ++iteration) {
                    output = fixture.buffer.get_subbuffer(fixture.indices);
                }
            }
            std::vector<std::int64_t> samples_ns;
            samples_ns.reserve(static_cast<std::size_t>(options.repetitions));
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < options.iterations; ++iteration) {
                    output = fixture.buffer.get_subbuffer(fixture.indices);
                }
                samples_ns.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count());
            }
            const auto summary = summarize(output);
            std::cout << ",\"benchmark\":{\"name\":\"subbuffer_select_4096_to_1024\","
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
            write_buffer_summary(summary);
            std::cout << '}';
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rollout-buffer harness error: " << error.what() << '\n';
        return 1;
    }
}
