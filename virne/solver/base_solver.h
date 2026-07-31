#pragma once

#include "../core/controller/node_mapper.h"
#include "../core/controller/topology_analyzer.h"
#include "../core/solution.h"
#include "../utils/utils_config.h"
#include "rank/link_rank.h"
#include "rank/node_rank.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace virne::core {

class Counter;
class Logger;
class Recorder;
namespace controller {
class Controller;
class PreparedControllerMutation;
}

}  // namespace virne::core

namespace virne::network {
class PhysicalNetwork;
class VirtualNetwork;
}  // namespace virne::network

namespace virne::solver {

struct SolverConfigInput {
    std::optional<std::uint32_t> seed;
    int verbose = 1;
    utils::RunDirectoryInput run_directory;
    bool reusable = false;
    rank::NodeRankMethod node_ranking_method = rank::NodeRankMethod::order;
    rank::LinkRankMethod link_ranking_method = rank::LinkRankMethod::order;
    core::controller::NodeMatchingMethod matching_method =
        core::controller::NodeMatchingMethod::greedy;
    core::controller::ShortestPathMethod shortest_method =
        core::controller::ShortestPathMethod::k_shortest;
    std::int64_t k_shortest = 10;
    bool allow_rejection = false;
    bool allow_revocable = false;
};

struct SolverConfig {
    std::optional<std::uint32_t> seed;
    int verbose = 1;
    std::filesystem::path save_dir;
    bool reusable = false;
    rank::NodeRankMethod node_ranking_method = rank::NodeRankMethod::order;
    rank::LinkRankMethod link_ranking_method = rank::LinkRankMethod::order;
    core::controller::NodeMatchingMethod matching_method =
        core::controller::NodeMatchingMethod::greedy;
    core::controller::ShortestPathMethod shortest_method =
        core::controller::ShortestPathMethod::k_shortest;
    std::int64_t k_shortest = 10;
    bool allow_rejection = false;
    bool allow_revocable = false;
};

SolverConfig make_solver_config(const SolverConfigInput& input);

struct SolverDependencies {
    std::reference_wrapper<const core::controller::Controller> controller;
    std::reference_wrapper<core::Recorder> recorder;
    std::reference_wrapper<const core::Counter> counter;
    std::reference_wrapper<core::Logger> logger;
};

struct SolverInstance {
    const network::VirtualNetwork& virtual_network;
    const network::PhysicalNetwork& physical_network;
};

// Explicit mutable system seam. The Environment owns both mutable objects and
// guarantees their lifetime for the synchronous call. Old SolverInstance
// remains const and unchanged for leaf/differential callers.
struct MutableSolverInstance {
    const network::VirtualNetwork& virtual_network;
    network::PhysicalNetwork& physical_network;
    core::controller::PreparedControllerMutation& mutation;
};

enum class SolverMutationState : std::uint8_t {
    detached,
    committed,
};

struct MutableSolverResult {
    core::Solution solution;
    SolverMutationState mutation_state = SolverMutationState::detached;
};

struct SolverId {
    std::uint32_t value = 0U;
};

constexpr bool operator==(SolverId lhs, SolverId rhs) noexcept {
    return lhs.value == rhs.value;
}

constexpr bool operator!=(SolverId lhs, SolverId rhs) noexcept {
    return !(lhs == rhs);
}

enum class SolverCategory : std::uint8_t {
    unknown,
    rounding,
    exact,
    heuristic,
    node_ranking,
    meta_heuristic,
    reinforcement_learning,
    unsupervised_learning,
};

enum class SolverErrorCode : std::uint8_t {
    unsupported_category,
    empty_solver_name,
    duplicate_solver_name,
    empty_factory,
    registry_frozen,
    registry_not_frozen,
    unknown_solver,
    invalid_solver_id,
    factory_returned_null,
    arrived_count_overflow,
    solve_not_implemented,
};

enum class SolverOperation : std::uint8_t {
    parse_category,
    construct_solver,
    ready,
    solve,
    update_arrived_count,
    register_solver,
    freeze_registry,
    resolve_solver,
    lookup_solver,
    list_registered,
    create_solver,
};

class SolverException : public std::runtime_error {
public:
    SolverException(
        SolverErrorCode code,
        SolverOperation operation,
        std::string message,
        std::optional<SolverId> solver_id = std::nullopt);

    SolverErrorCode code() const noexcept;
    SolverOperation operation() const noexcept;
    const std::optional<SolverId>& solver_id() const noexcept;

private:
    SolverErrorCode code_;
    SolverOperation operation_;
    std::optional<SolverId> solver_id_;
};

SolverCategory solver_category_from_string(std::string_view value);
std::string_view solver_category_name(SolverCategory value) noexcept;

class Solver {
public:
    Solver(SolverDependencies dependencies, SolverConfig config);
    virtual ~Solver() = default;

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    virtual void ready();
    virtual core::Solution solve(const SolverInstance& instance);
    virtual MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance);

    const SolverConfig& config() const noexcept;
    const core::controller::Controller& controller() const noexcept;
    core::Recorder& recorder() const noexcept;
    const core::Counter& counter() const noexcept;
    core::Logger& logger() const noexcept;

    std::uint64_t num_arrived_virtual_networks() const noexcept;
    void set_num_arrived_virtual_networks(std::uint64_t value) noexcept;
    void increment_num_arrived_virtual_networks();

private:
    const core::controller::Controller* controller_;
    core::Recorder* recorder_;
    const core::Counter* counter_;
    core::Logger* logger_;
    SolverConfig config_;
    std::uint64_t num_arrived_virtual_networks_ = 0U;
};

using SolverFactory = std::function<std::unique_ptr<Solver>(
    SolverDependencies, SolverConfig)>;

struct SolverDescriptor {
    SolverId id;
    std::string name;
    SolverCategory category = SolverCategory::unknown;
};

class SolverRegistry {
public:
    SolverRegistry();
    ~SolverRegistry();

    SolverRegistry(const SolverRegistry&) = delete;
    SolverRegistry& operator=(const SolverRegistry&) = delete;
    SolverRegistry(SolverRegistry&&) = delete;
    SolverRegistry& operator=(SolverRegistry&&) = delete;

    SolverId register_solver(
        std::string name,
        SolverCategory category,
        SolverFactory factory);
    void freeze();
    bool frozen() const noexcept;
    std::size_t size() const noexcept;

    SolverId resolve(std::string_view name) const;
    const SolverDescriptor& descriptor(SolverId id) const;
    std::vector<SolverDescriptor> list_registered() const;
    std::unique_ptr<Solver> create(
        SolverId id,
        SolverDependencies dependencies,
        SolverConfig config) const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace virne::solver
