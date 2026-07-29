#include "../virne/solver/base_solver.h"

#include "../virne/core/controller/controller.h"
#include "../virne/core/counter.h"
#include "../virne/core/logger.h"
#include "../virne/core/recorder.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <atomic>
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
using virne::solver::SolverErrorCode;
using virne::solver::SolverException;
using virne::solver::SolverId;
using virne::solver::SolverOperation;
using virne::solver::SolverRegistry;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void expect_solver_error(
    Function&& function,
    SolverErrorCode code,
    SolverOperation operation,
    const std::string& label,
    std::optional<SolverId> id = std::nullopt) {
    try {
        std::forward<Function>(function)();
    } catch (const SolverException& error) {
        require(error.code() == code, label + ": wrong error code");
        require(
            error.operation() == operation,
            label + ": wrong error operation");
        require(error.solver_id() == id, label + ": wrong solver ID");
        return;
    }
    throw std::runtime_error(label + ": expected SolverException");
}

std::filesystem::path unique_temp_root() {
    static std::atomic<std::uint64_t> sequence{0U};
    const auto tick = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
    const auto suffix = sequence.fetch_add(1U, std::memory_order_relaxed);
    return std::filesystem::temp_directory_path() /
           ("virne-base-solver-unit-" + std::to_string(tick) + "-" +
            std::to_string(suffix));
}

struct CleanupRoot {
    std::filesystem::path path;

    explicit CleanupRoot(std::filesystem::path value)
        : path(std::move(value)) {}

    ~CleanupRoot() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

virne::core::RecorderConfig recorder_config(
    const std::filesystem::path& root) {
    virne::core::RecorderConfig config;
    config.save_root_dir = root;
    config.solver_name = "base-solver-unit";
    config.run_id = "recorder";
    config.record_dir_name = "records";
    config.temporary_records = false;
    return config;
}

virne::core::LoggerConfig logger_config(
    const std::filesystem::path& root) {
    virne::core::LoggerConfig config;
    config.save_root_dir = root;
    config.solver_name = "base-solver-unit";
    config.run_id = "logger";
    config.log_dir_name = "logs";
    config.log_file_name = "run.log";
    config.backends.console = false;
    config.backends.file = false;
    config.level = virne::core::LoggerLevel::critical;
    config.log_show_interval = 1U;
    return config;
}

struct DependencyFixture {
    CleanupRoot cleanup;
    std::filesystem::path root;
    virne::core::controller::Controller controller;
    virne::core::Counter counter;
    virne::core::Recorder recorder;
    virne::core::Logger logger;

    DependencyFixture()
        : cleanup(unique_temp_root()),
          root(cleanup.path),
          controller(virne::core::controller::ControllerSelection{}),
          counter(virne::core::CounterSelection{}),
          recorder(
              virne::core::Counter(virne::core::CounterSelection{}),
              recorder_config(root)),
          logger(logger_config(root)) {}

    SolverDependencies dependencies() {
        return SolverDependencies{
            std::cref(controller),
            std::ref(recorder),
            std::cref(counter),
            std::ref(logger),
        };
    }
};

SolverConfig config_for(
    const std::filesystem::path& root,
    std::string solver_name = "probe",
    std::string run_id = "run") {
    SolverConfigInput input;
    input.seed = 42U;
    input.verbose = 9;
    input.run_directory.save_root_dir = root;
    input.run_directory.solver_name = std::move(solver_name);
    input.run_directory.run_id = std::move(run_id);
    input.reusable = true;
    input.node_ranking_method = virne::solver::rank::NodeRankMethod::nps;
    input.link_ranking_method = virne::solver::rank::LinkRankMethod::ffd;
    input.matching_method =
        virne::core::controller::NodeMatchingMethod::l2s2;
    input.shortest_method =
        virne::core::controller::ShortestPathMethod::all_shortest;
    input.k_shortest = -7;
    input.allow_rejection = true;
    input.allow_revocable = true;
    return virne::solver::make_solver_config(input);
}

void test_categories() {
    const std::vector<std::pair<std::string, SolverCategory>> cases{
        {"unknown", SolverCategory::unknown},
        {"rounding", SolverCategory::rounding},
        {"exact", SolverCategory::exact},
        {"heuristic", SolverCategory::heuristic},
        {"node_ranking", SolverCategory::node_ranking},
        {"meta_heuristic", SolverCategory::meta_heuristic},
        {"r_learning", SolverCategory::reinforcement_learning},
        {"u_learning", SolverCategory::unsupervised_learning},
    };
    for (const auto& item : cases) {
        const auto parsed = virne::solver::solver_category_from_string(
            item.first);
        require(parsed == item.second, "category parser mismatch");
        require(
            virne::solver::solver_category_name(parsed) == item.first,
            "category name mismatch");
    }
    require(
        virne::solver::solver_category_name(
            static_cast<SolverCategory>(255U)) == "unknown",
        "invalid category name must be unknown");
    expect_solver_error(
        [] {
            static_cast<void>(
                virne::solver::solver_category_from_string("missing"));
        },
        SolverErrorCode::unsupported_category,
        SolverOperation::parse_category,
        "unsupported category");
}

void test_config_and_solver(DependencyFixture& fixture) {
    const auto config = config_for(fixture.root);
    require(config.seed == 42U, "seed snapshot mismatch");
    require(config.verbose == 9, "verbose snapshot mismatch");
    require(
        config.save_dir == fixture.root / "probe" / "run",
        "save directory mismatch");
    require(config.reusable, "reusable mismatch");
    require(
        config.node_ranking_method ==
            virne::solver::rank::NodeRankMethod::nps,
        "node rank method mismatch");
    require(
        config.link_ranking_method ==
            virne::solver::rank::LinkRankMethod::ffd,
        "link rank method mismatch");
    require(
        config.matching_method ==
            virne::core::controller::NodeMatchingMethod::l2s2,
        "matching method mismatch");
    require(
        config.shortest_method ==
            virne::core::controller::ShortestPathMethod::all_shortest,
        "shortest method mismatch");
    require(config.k_shortest == -7, "k-shortest snapshot mismatch");
    require(config.allow_rejection, "rejection snapshot mismatch");
    require(config.allow_revocable, "revocable snapshot mismatch");

    require(
        config_for("/root", "/absolute", "run").save_dir ==
            std::filesystem::path("/absolute/run"),
        "absolute solver component path mismatch");
    require(
        config_for("/root", "solver", "/absolute-run").save_dir ==
            std::filesystem::path("/absolute-run"),
        "absolute run component path mismatch");
    require(
        config_for("/root", "", "").save_dir.generic_string() ==
            "/root/",
        "empty path component mismatch");

    Solver solver(fixture.dependencies(), config);
    require(&solver.controller() == &fixture.controller, "controller identity");
    require(&solver.recorder() == &fixture.recorder, "recorder identity");
    require(&solver.counter() == &fixture.counter, "counter identity");
    require(&solver.logger() == &fixture.logger, "logger identity");
    require(
        solver.num_arrived_virtual_networks() == 0U,
        "initial arrival count mismatch");
    solver.ready();
    solver.increment_num_arrived_virtual_networks();
    require(
        solver.num_arrived_virtual_networks() == 1U,
        "arrival increment mismatch");
    solver.set_num_arrived_virtual_networks(99U);
    require(
        solver.num_arrived_virtual_networks() == 99U,
        "arrival setter mismatch");
    solver.set_num_arrived_virtual_networks(
        std::numeric_limits<std::uint64_t>::max());
    expect_solver_error(
        [&solver] { solver.increment_num_arrived_virtual_networks(); },
        SolverErrorCode::arrived_count_overflow,
        SolverOperation::update_arrived_count,
        "arrival overflow");

    virne::network::VirtualNetwork virtual_network;
    virne::network::PhysicalNetwork physical_network;
    const virne::solver::SolverInstance instance{
        virtual_network,
        physical_network,
    };
    expect_solver_error(
        [&solver, &instance] {
            static_cast<void>(solver.solve(instance));
        },
        SolverErrorCode::solve_not_implemented,
        SolverOperation::solve,
        "base solve");
}

virne::solver::SolverFactory base_factory() {
    return [](SolverDependencies dependencies, SolverConfig config) {
        return std::make_unique<Solver>(dependencies, std::move(config));
    };
}

void test_registry(DependencyFixture& fixture) {
    struct FactoryProbeError final : std::runtime_error {
        FactoryProbeError()
            : std::runtime_error("factory probe") {}
    };

    SolverRegistry throwing_registry;
    const auto throwing_id = throwing_registry.register_solver(
        "throwing",
        SolverCategory::heuristic,
        [](SolverDependencies, SolverConfig)
            -> std::unique_ptr<Solver> { throw FactoryProbeError(); });
    try {
        static_cast<void>(throwing_registry.create(
            throwing_id,
            fixture.dependencies(),
            config_for(fixture.root, "throwing", "factory")));
        throw std::runtime_error("factory exception did not propagate");
    } catch (const FactoryProbeError&) {
    }
    require(
        throwing_registry.size() == 1U,
        "throwing factory changed registry structure");

    SolverRegistry registry;
    require(!registry.frozen(), "new registry must be mutable");
    require(registry.size() == 0U, "new registry size mismatch");

    expect_solver_error(
        [&registry] {
            static_cast<void>(registry.register_solver(
                "", SolverCategory::heuristic, base_factory()));
        },
        SolverErrorCode::empty_solver_name,
        SolverOperation::register_solver,
        "empty registry name");
    expect_solver_error(
        [&registry] {
            static_cast<void>(registry.register_solver(
                "bad-category",
                static_cast<SolverCategory>(255U),
                base_factory()));
        },
        SolverErrorCode::unsupported_category,
        SolverOperation::register_solver,
        "invalid registry category");
    expect_solver_error(
        [&registry] {
            static_cast<void>(registry.register_solver(
                "empty-factory",
                SolverCategory::heuristic,
                virne::solver::SolverFactory{}));
        },
        SolverErrorCode::empty_factory,
        SolverOperation::register_solver,
        "empty registry factory");

    const auto alpha = registry.register_solver(
        "alpha", SolverCategory::heuristic, base_factory());
    const auto beta = registry.register_solver(
        "beta", SolverCategory::exact, base_factory());
    const auto learning_seam = registry.register_solver(
        "reserved-r",
        SolverCategory::reinforcement_learning,
        base_factory());
    const auto null_solver = registry.register_solver(
        "null",
        SolverCategory::unknown,
        [](SolverDependencies, SolverConfig) {
            return std::unique_ptr<Solver>{};
        });

    require(alpha == SolverId{0U}, "first compact ID mismatch");
    require(beta == SolverId{1U}, "second compact ID mismatch");
    require(learning_seam == SolverId{2U}, "reserved ID mismatch");
    require(null_solver == SolverId{3U}, "null-factory ID mismatch");
    require(registry.size() == 4U, "registered size mismatch");
    require(registry.resolve("alpha") == alpha, "pre-freeze resolve mismatch");

    auto listed = registry.list_registered();
    require(listed.size() == 4U, "list size mismatch");
    require(listed[0].name == "alpha", "list order alpha mismatch");
    require(listed[1].name == "beta", "list order beta mismatch");
    require(
        listed[2].category == SolverCategory::reinforcement_learning,
        "reserved category mismatch");
    listed[0].name = "mutated-copy";
    require(
        registry.list_registered()[0].name == "alpha",
        "list must be an independent snapshot");

    expect_solver_error(
        [&registry] {
            static_cast<void>(registry.register_solver(
                "alpha", SolverCategory::meta_heuristic, base_factory()));
        },
        SolverErrorCode::duplicate_solver_name,
        SolverOperation::register_solver,
        "duplicate registry name");
    require(registry.size() == 4U, "duplicate changed registry size");
    expect_solver_error(
        [&registry] { static_cast<void>(registry.resolve("missing")); },
        SolverErrorCode::unknown_solver,
        SolverOperation::resolve_solver,
        "unknown registry name");
    expect_solver_error(
        [&registry, alpha] {
            static_cast<void>(registry.descriptor(alpha));
        },
        SolverErrorCode::registry_not_frozen,
        SolverOperation::lookup_solver,
        "descriptor before freeze",
        alpha);

    auto pre_freeze = registry.create(
        beta,
        fixture.dependencies(),
        config_for(fixture.root, "pre", "freeze"));
    require(
        pre_freeze->config().save_dir == fixture.root / "pre" / "freeze",
        "pre-freeze factory config mismatch");

    expect_solver_error(
        [&registry, &fixture, null_solver] {
            static_cast<void>(registry.create(
                null_solver,
                fixture.dependencies(),
                config_for(fixture.root, "null", "solver")));
        },
        SolverErrorCode::factory_returned_null,
        SolverOperation::create_solver,
        "null factory result",
        null_solver);

    registry.freeze();
    registry.freeze();
    require(registry.frozen(), "registry freeze mismatch");
    require(
        registry.descriptor(alpha).name == "alpha",
        "frozen descriptor mismatch");
    require(
        registry.descriptor(beta).category == SolverCategory::exact,
        "frozen descriptor category mismatch");
    expect_solver_error(
        [&registry] {
            static_cast<void>(registry.register_solver(
                "late", SolverCategory::heuristic, base_factory()));
        },
        SolverErrorCode::registry_frozen,
        SolverOperation::register_solver,
        "registration after freeze");
    expect_solver_error(
        [&registry] {
            static_cast<void>(registry.descriptor(SolverId{99U}));
        },
        SolverErrorCode::invalid_solver_id,
        SolverOperation::lookup_solver,
        "invalid descriptor ID",
        SolverId{99U});
    expect_solver_error(
        [&registry, &fixture] {
            static_cast<void>(registry.create(
                SolverId{99U},
                fixture.dependencies(),
                config_for(fixture.root, "invalid", "id")));
        },
        SolverErrorCode::invalid_solver_id,
        SolverOperation::create_solver,
        "invalid factory ID",
        SolverId{99U});

    const auto run_concurrent = [&](std::size_t workers) {
        std::vector<std::exception_ptr> errors(workers);
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                try {
                    const auto resolved = registry.resolve("alpha");
                    require(resolved == alpha, "concurrent resolve mismatch");
                    const auto snapshot = registry.list_registered();
                    require(
                        snapshot.size() == 4U && snapshot[0].id == alpha,
                        "concurrent registry snapshot mismatch");
                    for (std::size_t iteration = 0U; iteration < 128U;
                         ++iteration) {
                        const auto& descriptor = registry.descriptor(alpha);
                        require(
                            descriptor.id == alpha &&
                                descriptor.category ==
                                    SolverCategory::heuristic,
                            "concurrent descriptor mismatch");
                        auto solver = registry.create(
                            alpha,
                            fixture.dependencies(),
                            config_for(fixture.root, "parallel", "run"));
                        require(
                            &solver->controller() == &fixture.controller,
                            "concurrent dependency identity mismatch");
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
    };

    run_concurrent(1U);
    run_concurrent(2U);
    run_concurrent(8U);
}

void test_register_freeze_transition() {
    for (std::size_t round = 0U; round < 16U; ++round) {
        SolverRegistry registry;
        std::atomic<bool> start{false};
        std::optional<SolverErrorCode> registration_error;
        std::thread registrar([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                static_cast<void>(registry.register_solver(
                    "race", SolverCategory::heuristic, base_factory()));
            } catch (const SolverException& error) {
                registration_error = error.code();
            }
        });
        std::thread freezer([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            registry.freeze();
        });
        start.store(true, std::memory_order_release);
        registrar.join();
        freezer.join();

        require(registry.frozen(), "freeze transition did not publish");
        if (registry.size() == 0U) {
            require(
                registration_error == SolverErrorCode::registry_frozen,
                "freeze-first transition returned the wrong error");
        } else {
            require(registry.size() == 1U, "transition registry size mismatch");
            require(
                !registration_error.has_value(),
                "register-first transition reported an error");
            require(
                registry.resolve("race").value == 0U,
                "register-first transition lost its ID");
        }
    }
}

}  // namespace

int main() {
    try {
        test_categories();
        DependencyFixture fixture;
        test_config_and_solver(fixture);
        test_registry(fixture);
        test_register_freeze_transition();
        std::cout << "base solver unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base solver unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
