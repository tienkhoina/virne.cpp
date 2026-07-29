#include "../virne/solver/base_solver.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
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
using virne::solver::SolverErrorCode;
using virne::solver::SolverException;
using virne::solver::SolverId;
using virne::solver::SolverOperation;
using virne::solver::SolverRegistry;

std::string json_string(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const char item : value) {
        const auto byte = static_cast<unsigned char>(item);
        switch (byte) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (byte < 0x20U) {
                constexpr char hex[] = "0123456789abcdef";
                result += "\\u00";
                result.push_back(hex[(byte >> 4U) & 0x0fU]);
                result.push_back(hex[byte & 0x0fU]);
            } else {
                result.push_back(static_cast<char>(byte));
            }
        }
    }
    result.push_back('"');
    return result;
}

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
    config.solver_name = "base-solver-differential";
    config.run_id = "recorder";
    config.temporary_records = false;
    return config;
}

virne::core::LoggerConfig logger_config(
    const std::filesystem::path& root) {
    virne::core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "base-solver-differential";
    config.run_id = "logger";
    config.backends.console = false;
    config.backends.file = false;
    config.level = virne::core::LoggerLevel::critical;
    return config;
}

struct DependencyFixture {
    CleanupRoot cleanup{
        std::filesystem::temp_directory_path() /
        "virne-base-solver-differential"};
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

SolverConfig explicit_config(
    const std::string& root,
    const std::string& solver_name,
    const std::string& run_id,
    int verbose) {
    SolverConfigInput input;
    input.seed = 17U;
    input.verbose = verbose;
    input.run_directory.save_root_dir = root;
    input.run_directory.solver_name = solver_name;
    input.run_directory.run_id = run_id;
    input.reusable = true;
    input.node_ranking_method = virne::solver::rank::NodeRankMethod::nps;
    input.link_ranking_method = virne::solver::rank::LinkRankMethod::ffd;
    input.matching_method =
        virne::core::controller::NodeMatchingMethod::l2s2;
    input.shortest_method =
        virne::core::controller::ShortestPathMethod::all_shortest;
    input.k_shortest = -7;
    input.allow_rejection = true;
    input.allow_revocable = false;
    return virne::solver::make_solver_config(input);
}

virne::solver::SolverFactory base_factory() {
    return [](SolverDependencies dependencies, SolverConfig config) {
        return std::make_unique<Solver>(dependencies, std::move(config));
    };
}

bool parallel_factory_check(
    const SolverRegistry& registry,
    SolverId id,
    DependencyFixture& fixture,
    const SolverConfig& config,
    std::size_t workers) {
    if (workers == 0U) {
        workers = 1U;
    }
    std::vector<std::exception_ptr> errors(workers);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0U; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            try {
                for (std::size_t iteration = 0U; iteration < 64U;
                     ++iteration) {
                    const auto& descriptor = registry.descriptor(id);
                    if (descriptor.id != id || descriptor.name != "alpha") {
                        throw std::runtime_error("descriptor mismatch");
                    }
                    auto solver = registry.create(
                        id, fixture.dependencies(), config);
                    if (&solver->controller() != &fixture.controller ||
                        &solver->recorder() != &fixture.recorder ||
                        &solver->counter() != &fixture.counter ||
                        &solver->logger() != &fixture.logger) {
                        throw std::runtime_error("dependency identity mismatch");
                    }
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
    return true;
}

std::size_t parse_workers(int argc, char** argv) {
    if (argc == 1) {
        return 1U;
    }
    if (argc != 3 || std::string(argv[1]) != "--workers") {
        throw std::runtime_error("usage: base_solver_harness [--workers N]");
    }
    const auto value = std::stoull(argv[2]);
    return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto workers = parse_workers(argc, argv);
        DependencyFixture fixture;
        const auto config = explicit_config("/root/save", "demo", "r1", 7);
        Solver solver(fixture.dependencies(), config);

        bool solve_error = false;
        solver.ready();
        virne::network::VirtualNetwork virtual_network;
        virne::network::PhysicalNetwork physical_network;
        try {
            static_cast<void>(solver.solve(
                virne::solver::SolverInstance{
                    virtual_network, physical_network}));
        } catch (const SolverException& error) {
            solve_error =
                error.code() == SolverErrorCode::solve_not_implemented &&
                error.operation() == SolverOperation::solve;
        }

        SolverRegistry registry;
        const auto alpha = registry.register_solver(
            "alpha", SolverCategory::heuristic, base_factory());
        const auto beta = registry.register_solver(
            "beta", SolverCategory::exact, base_factory());
        const auto pre_copy = registry.list_registered();
        auto changed_copy = pre_copy;
        changed_copy.clear();
        const bool copy_isolated =
            changed_copy.empty() && registry.list_registered().size() == 2U;

        bool duplicate_error = false;
        try {
            static_cast<void>(registry.register_solver(
                "alpha", SolverCategory::meta_heuristic, base_factory()));
        } catch (const SolverException& error) {
            duplicate_error =
                error.code() == SolverErrorCode::duplicate_solver_name &&
                registry.size() == 2U;
        }
        bool missing_error = false;
        try {
            static_cast<void>(registry.resolve("missing"));
        } catch (const SolverException& error) {
            missing_error = error.code() == SolverErrorCode::unknown_solver;
        }
        registry.freeze();
        const bool factory_identity = parallel_factory_check(
            registry, alpha, fixture, config, workers);

        const auto default_verbose = explicit_config(
            "/root/save", "demo", "r1", 1);
        const auto absolute_solver = explicit_config(
            "/root", "/absolute", "run", 1);
        const auto absolute_run = explicit_config(
            "/root", "demo", "/absolute-run", 1);
        const auto empty_path = explicit_config("/root", "", "", 1);

        std::cout
            << "{"
            << "\"constructor_explicit\":{"
            << "\"seed\":17,"
            << "\"verbose\":" << config.verbose << ','
            << "\"num_arrived\":"
            << solver.num_arrived_virtual_networks() << ','
            << "\"save_dir\":"
            << json_string(config.save_dir.generic_string()) << ','
            << "\"reusable\":true,"
            << "\"node_ranking_method\":\"nps\","
            << "\"link_ranking_method\":\"ffd\","
            << "\"matching_mathod\":\"l2s2\","
            << "\"shortest_method\":\"all_shortest\","
            << "\"k_shortest\":" << config.k_shortest << ','
            << "\"allow_rejection\":true,"
            << "\"allow_revocable\":false},"
            << "\"constructor_default_verbose\":"
            << default_verbose.verbose << ','
            << "\"path_absolute_solver\":"
            << json_string(absolute_solver.save_dir.generic_string()) << ','
            << "\"path_absolute_run\":"
            << json_string(absolute_run.save_dir.generic_string()) << ','
            << "\"path_empty\":"
            << json_string(empty_path.save_dir.generic_string()) << ','
            << "\"ready\":\"none\","
            << "\"solve\":"
            << json_string(solve_error ? "not_implemented" : "wrong") << ','
            << "\"registry_order\":["
            << json_string(pre_copy[0].name) << ','
            << json_string(pre_copy[1].name) << "],"
            << "\"registry_categories\":["
            << json_string(std::string(virne::solver::solver_category_name(
                   registry.descriptor(alpha).category)))
            << ','
            << json_string(std::string(virne::solver::solver_category_name(
                   registry.descriptor(beta).category)))
            << "],"
            << "\"registry_copy\":"
            << (copy_isolated ? "true" : "false") << ','
            << "\"registry_duplicate\":"
            << json_string(duplicate_error ? "duplicate" : "wrong") << ','
            << "\"registry_missing\":"
            << json_string(missing_error ? "not_implemented" : "wrong")
            << ','
            << "\"registry_factory_identity\":"
            << (factory_identity ? "true" : "false")
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base solver harness: FAIL: " << error.what() << '\n';
        return 1;
    }
}
