#include "../virne/solver/base_solver.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using virne::solver::Solver;
using virne::solver::SolverCategory;
using virne::solver::SolverConfig;
using virne::solver::SolverConfigInput;
using virne::solver::SolverDependencies;
using virne::solver::SolverId;
using virne::solver::SolverRegistry;

struct CleanupRoot {
    std::filesystem::path path;

    ~CleanupRoot() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

virne::core::RecorderConfig recorder_config(
    const std::filesystem::path& root) {
    virne::core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "base-solver-benchmark";
    config.run_id = "recorder";
    config.temporary_records = false;
    return config;
}

virne::core::LoggerConfig logger_config(
    const std::filesystem::path& root) {
    virne::core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "base-solver-benchmark";
    config.run_id = "logger";
    config.backends.console = false;
    config.backends.file = false;
    config.level = virne::core::LoggerLevel::critical;
    return config;
}

struct DependencyFixture {
    CleanupRoot cleanup{
        std::filesystem::temp_directory_path() /
        "virne-base-solver-benchmark"};
    virne::core::controller::Controller controller{
        virne::core::controller::ControllerSelection{}};
    virne::core::Counter counter{virne::core::CounterSelection{}};
    virne::core::Recorder recorder{
        virne::core::Counter(virne::core::CounterSelection{}),
        recorder_config(cleanup.path)};
    virne::core::Logger logger{logger_config(cleanup.path)};

    SolverDependencies dependencies() {
        return SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger),
        };
    }
};

struct Options {
    std::size_t workers = 1U;
    std::size_t count = 4096U;
    std::size_t warmups = 1U;
    std::size_t samples = 3U;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::runtime_error("benchmark option requires a value");
        }
        const std::string name(argv[index]);
        const auto value = static_cast<std::size_t>(
            std::stoull(argv[index + 1]));
        if (name == "--workers") {
            options.workers = value;
        } else if (name == "--count") {
            options.count = value;
        } else if (name == "--warmups") {
            options.warmups = value;
        } else if (name == "--samples") {
            options.samples = value;
        } else {
            throw std::runtime_error("unknown benchmark option: " + name);
        }
    }
    if (options.count == 0U || options.samples == 0U) {
        throw std::runtime_error("count and samples must be positive");
    }
    return options;
}

virne::solver::SolverFactory base_factory() {
    return [](SolverDependencies dependencies, SolverConfig config) {
        return std::make_unique<Solver>(dependencies, std::move(config));
    };
}

std::vector<SolverConfigInput> make_inputs(std::size_t count) {
    std::vector<SolverConfigInput> inputs;
    inputs.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        SolverConfigInput input;
        input.seed = static_cast<std::uint32_t>(index);
        input.verbose = static_cast<int>(index % 5U);
        input.run_directory.save_root_dir = "/benchmark";
        input.run_directory.solver_name =
            "solver-" + std::to_string(index % 32U);
        input.run_directory.run_id = "run-" + std::to_string(index);
        input.reusable = index % 2U != 0U;
        input.node_ranking_method =
            virne::solver::rank::NodeRankMethod::nps;
        input.link_ranking_method =
            virne::solver::rank::LinkRankMethod::ffd;
        input.matching_method =
            virne::core::controller::NodeMatchingMethod::l2s2;
        input.shortest_method =
            virne::core::controller::ShortestPathMethod::all_shortest;
        input.k_shortest = static_cast<std::int64_t>(10U + index % 7U);
        input.allow_rejection = index % 3U == 0U;
        input.allow_revocable = index % 5U == 0U;
        inputs.push_back(std::move(input));
    }
    return inputs;
}

std::vector<SolverId> make_solver_ids(
    SolverRegistry& registry,
    std::size_t count) {
    std::vector<SolverId> registered;
    registered.reserve(32U);
    for (std::size_t index = 0U; index < 32U; ++index) {
        registered.push_back(registry.register_solver(
            "solver-" + std::to_string(index),
            SolverCategory::heuristic,
            base_factory()));
    }
    registry.freeze();

    std::vector<SolverId> ids;
    ids.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        ids.push_back(registered[index % registered.size()]);
    }
    return ids;
}

struct TimedBatch {
    std::uint64_t elapsed_ns = 0U;
    std::vector<std::unique_ptr<Solver>> solvers;
};

TimedBatch construct_batch(
    const SolverRegistry& registry,
    DependencyFixture& fixture,
    const std::vector<SolverId>& ids,
    const std::vector<SolverConfigInput>& inputs,
    std::size_t requested_workers) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<Solver>> solvers(inputs.size());
    const std::size_t workers = requested_workers <= 1U
                                    ? 1U
                                    : std::min(requested_workers, inputs.size());
    if (workers == 1U) {
        for (std::size_t index = 0U; index < inputs.size(); ++index) {
            auto config = virne::solver::make_solver_config(inputs[index]);
            solvers[index] = registry.create(
                ids[index], fixture.dependencies(), std::move(config));
        }
    } else {
        std::vector<std::exception_ptr> errors(workers);
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                try {
                    const auto begin = inputs.size() * worker / workers;
                    const auto end = inputs.size() * (worker + 1U) / workers;
                    for (std::size_t index = begin; index < end; ++index) {
                        auto config =
                            virne::solver::make_solver_config(inputs[index]);
                        solvers[index] = registry.create(
                            ids[index],
                            fixture.dependencies(),
                            std::move(config));
                    }
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }
    const auto finish = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        finish - start);
    return TimedBatch{
        static_cast<std::uint64_t>(elapsed.count()),
        std::move(solvers),
    };
}

struct HashWriter {
    std::uint64_t hash = 14695981039346656037ULL;
    std::size_t bytes = 0U;

    void byte(std::uint8_t value) noexcept {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
        ++bytes;
    }

    void u32(std::uint32_t value) noexcept {
        for (std::size_t offset = 0U; offset < 4U; ++offset) {
            byte(static_cast<std::uint8_t>(value & 0xffU));
            value >>= 8U;
        }
    }

    void u64(std::uint64_t value) noexcept {
        for (std::size_t offset = 0U; offset < 8U; ++offset) {
            byte(static_cast<std::uint8_t>(value & 0xffU));
            value >>= 8U;
        }
    }

    void i64(std::int64_t value) noexcept {
        u64(static_cast<std::uint64_t>(value));
    }

    void string(const std::string& value) {
        if (value.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("benchmark string is too large");
        }
        u32(static_cast<std::uint32_t>(value.size()));
        for (const unsigned char item : value) {
            byte(item);
        }
    }
};

HashWriter checksum_batch(
    const TimedBatch& batch,
    DependencyFixture& fixture) {
    HashWriter writer;
    for (const auto& solver : batch.solvers) {
        if (!solver) {
            throw std::runtime_error("benchmark factory returned null");
        }
        const auto& config = solver->config();
        if (!config.seed.has_value()) {
            throw std::runtime_error("benchmark seed is absent");
        }
        writer.u32(*config.seed);
        writer.i64(static_cast<std::int64_t>(config.verbose));
        writer.u64(solver->num_arrived_virtual_networks());
        writer.string(config.save_dir.generic_string());
        writer.byte(static_cast<std::uint8_t>(
            config.reusable ? 1U : 0U));
        writer.string("nps");
        writer.string("ffd");
        writer.string("l2s2");
        writer.string("all_shortest");
        writer.i64(config.k_shortest);
        writer.byte(static_cast<std::uint8_t>(
            config.allow_rejection ? 1U : 0U));
        writer.byte(static_cast<std::uint8_t>(
            config.allow_revocable ? 1U : 0U));
        const bool identities =
            &solver->controller() == &fixture.controller &&
            &solver->recorder() == &fixture.recorder &&
            &solver->counter() == &fixture.counter &&
            &solver->logger() == &fixture.logger;
        writer.byte(static_cast<std::uint8_t>(identities ? 1U : 0U));
    }
    return writer;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        DependencyFixture fixture;
        SolverRegistry registry;
        const auto inputs = make_inputs(options.count);
        const auto ids = make_solver_ids(registry, options.count);

        for (std::size_t warmup = 0U; warmup < options.warmups; ++warmup) {
            const auto batch = construct_batch(
                registry, fixture, ids, inputs, options.workers);
            static_cast<void>(checksum_batch(batch, fixture));
        }

        std::vector<std::uint64_t> samples;
        samples.reserve(options.samples);
        std::optional<std::uint64_t> checksum;
        std::size_t output_bytes = 0U;
        for (std::size_t sample = 0U; sample < options.samples; ++sample) {
            const auto batch = construct_batch(
                registry, fixture, ids, inputs, options.workers);
            const auto output = checksum_batch(batch, fixture);
            if (checksum.has_value() && *checksum != output.hash) {
                throw std::runtime_error("benchmark checksum drift");
            }
            if (sample != 0U && output_bytes != output.bytes) {
                throw std::runtime_error("benchmark output-byte drift");
            }
            checksum = output.hash;
            output_bytes = output.bytes;
            samples.push_back(batch.elapsed_ns);
        }

        auto ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        const auto median = ordered[ordered.size() / 2U];
        std::cout << "{\"workers\":" << options.workers
                  << ",\"entries\":" << options.count
                  << ",\"bytes\":" << output_bytes
                  << ",\"checksum\":" << *checksum
                  << ",\"median_ns\":" << median
                  << ",\"samples_ns\":[";
        for (std::size_t index = 0U; index < samples.size(); ++index) {
            if (index != 0U) {
                std::cout << ',';
            }
            std::cout << samples[index];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base solver benchmark: FAIL: " << error.what() << '\n';
        return 1;
    }
}
