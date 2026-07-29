#include "base_solver.h"

#include "../core/solution.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace virne::solver {
namespace {

bool valid_category(SolverCategory category) noexcept {
    const auto value = static_cast<std::uint8_t>(category);
    return value <= static_cast<std::uint8_t>(
                        SolverCategory::unsupervised_learning);
}

[[noreturn]] void throw_invalid_id(
    SolverId id,
    SolverOperation operation) {
    throw SolverException(
        SolverErrorCode::invalid_solver_id,
        operation,
        "solver ID is outside the registered descriptor range",
        id);
}

}  // namespace

SolverConfig make_solver_config(const SolverConfigInput& input) {
    SolverConfig config;
    config.seed = input.seed;
    config.verbose = input.verbose;
    config.save_dir = utils::get_run_id_dir(input.run_directory);
    config.reusable = input.reusable;
    config.node_ranking_method = input.node_ranking_method;
    config.link_ranking_method = input.link_ranking_method;
    config.matching_method = input.matching_method;
    config.shortest_method = input.shortest_method;
    config.k_shortest = input.k_shortest;
    config.allow_rejection = input.allow_rejection;
    config.allow_revocable = input.allow_revocable;
    return config;
}

SolverException::SolverException(
    SolverErrorCode code,
    SolverOperation operation,
    std::string message,
    std::optional<SolverId> solver_id)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      solver_id_(solver_id) {}

SolverErrorCode SolverException::code() const noexcept {
    return code_;
}

SolverOperation SolverException::operation() const noexcept {
    return operation_;
}

const std::optional<SolverId>& SolverException::solver_id() const noexcept {
    return solver_id_;
}

SolverCategory solver_category_from_string(std::string_view value) {
    if (value == "unknown") {
        return SolverCategory::unknown;
    }
    if (value == "rounding") {
        return SolverCategory::rounding;
    }
    if (value == "exact") {
        return SolverCategory::exact;
    }
    if (value == "heuristic") {
        return SolverCategory::heuristic;
    }
    if (value == "node_ranking") {
        return SolverCategory::node_ranking;
    }
    if (value == "meta_heuristic") {
        return SolverCategory::meta_heuristic;
    }
    if (value == "r_learning") {
        return SolverCategory::reinforcement_learning;
    }
    if (value == "u_learning") {
        return SolverCategory::unsupervised_learning;
    }
    throw SolverException(
        SolverErrorCode::unsupported_category,
        SolverOperation::parse_category,
        "unsupported solver category: " + std::string(value));
}

std::string_view solver_category_name(SolverCategory value) noexcept {
    switch (value) {
    case SolverCategory::unknown:
        return "unknown";
    case SolverCategory::rounding:
        return "rounding";
    case SolverCategory::exact:
        return "exact";
    case SolverCategory::heuristic:
        return "heuristic";
    case SolverCategory::node_ranking:
        return "node_ranking";
    case SolverCategory::meta_heuristic:
        return "meta_heuristic";
    case SolverCategory::reinforcement_learning:
        return "r_learning";
    case SolverCategory::unsupervised_learning:
        return "u_learning";
    }
    return "unknown";
}

Solver::Solver(SolverDependencies dependencies, SolverConfig config)
    : controller_(&dependencies.controller.get()),
      recorder_(&dependencies.recorder.get()),
      counter_(&dependencies.counter.get()),
      logger_(&dependencies.logger.get()),
      config_(std::move(config)) {}

void Solver::ready() {}

core::Solution Solver::solve(const SolverInstance& instance) {
    static_cast<void>(instance);
    throw SolverException(
        SolverErrorCode::solve_not_implemented,
        SolverOperation::solve,
        "solver implementation is not available");
}

const SolverConfig& Solver::config() const noexcept {
    return config_;
}

const core::controller::Controller& Solver::controller() const noexcept {
    return *controller_;
}

core::Recorder& Solver::recorder() const noexcept {
    return *recorder_;
}

const core::Counter& Solver::counter() const noexcept {
    return *counter_;
}

core::Logger& Solver::logger() const noexcept {
    return *logger_;
}

std::uint64_t Solver::num_arrived_virtual_networks() const noexcept {
    return num_arrived_virtual_networks_;
}

void Solver::set_num_arrived_virtual_networks(std::uint64_t value) noexcept {
    num_arrived_virtual_networks_ = value;
}

void Solver::increment_num_arrived_virtual_networks() {
    if (num_arrived_virtual_networks_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw SolverException(
            SolverErrorCode::arrived_count_overflow,
            SolverOperation::update_arrived_count,
            "arrived virtual-network count overflow");
    }
    ++num_arrived_virtual_networks_;
}

struct SolverRegistry::State {
    struct Slot {
        SolverDescriptor descriptor;
        SolverFactory factory;
    };

    mutable std::shared_mutex mutex;
    std::vector<Slot> slots;
    std::unordered_map<std::string, SolverId> names;
    std::atomic<bool> frozen{false};
    std::atomic<std::size_t> slot_count{0U};
};

SolverRegistry::SolverRegistry()
    : state_(std::make_unique<State>()) {}

SolverRegistry::~SolverRegistry() = default;

SolverId SolverRegistry::register_solver(
    std::string name,
    SolverCategory category,
    SolverFactory factory) {
    if (name.empty()) {
        throw SolverException(
            SolverErrorCode::empty_solver_name,
            SolverOperation::register_solver,
            "solver name may not be empty");
    }
    if (!valid_category(category)) {
        throw SolverException(
            SolverErrorCode::unsupported_category,
            SolverOperation::register_solver,
            "solver category is outside the typed enum range");
    }
    if (!factory) {
        throw SolverException(
            SolverErrorCode::empty_factory,
            SolverOperation::register_solver,
            "solver factory may not be empty");
    }

    std::unique_lock<std::shared_mutex> lock(state_->mutex);
    if (state_->frozen.load(std::memory_order_relaxed)) {
        throw SolverException(
            SolverErrorCode::registry_frozen,
            SolverOperation::register_solver,
            "solver registry is frozen");
    }
    if (state_->names.find(name) != state_->names.end()) {
        throw SolverException(
            SolverErrorCode::duplicate_solver_name,
            SolverOperation::register_solver,
            "solver '" + name + "' is already registered");
    }
    if (state_->slots.size() >=
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw SolverException(
            SolverErrorCode::invalid_solver_id,
            SolverOperation::register_solver,
            "solver registry exhausted its compact ID range");
    }

    const SolverId id{static_cast<std::uint32_t>(state_->slots.size())};
    State::Slot slot{SolverDescriptor{id, std::move(name), category},
                     std::move(factory)};

    const auto inserted = state_->names.emplace(slot.descriptor.name, id);
    try {
        state_->slots.push_back(std::move(slot));
    } catch (...) {
        state_->names.erase(inserted.first);
        throw;
    }
    state_->slot_count.store(state_->slots.size(), std::memory_order_release);
    return id;
}

void SolverRegistry::freeze() {
    std::unique_lock<std::shared_mutex> lock(state_->mutex);
    state_->frozen.store(true, std::memory_order_release);
}

bool SolverRegistry::frozen() const noexcept {
    return state_->frozen.load(std::memory_order_acquire);
}

std::size_t SolverRegistry::size() const noexcept {
    return state_->slot_count.load(std::memory_order_acquire);
}

SolverId SolverRegistry::resolve(std::string_view name) const {
    const auto find_id = [this, name]() -> SolverId {
        const auto iterator = state_->names.find(std::string(name));
        if (iterator == state_->names.end()) {
            throw SolverException(
                SolverErrorCode::unknown_solver,
                SolverOperation::resolve_solver,
                "solver '" + std::string(name) + "' is not implemented");
        }
        return iterator->second;
    };

    if (frozen()) {
        return find_id();
    }
    std::shared_lock<std::shared_mutex> lock(state_->mutex);
    return find_id();
}

const SolverDescriptor& SolverRegistry::descriptor(SolverId id) const {
    if (!frozen()) {
        throw SolverException(
            SolverErrorCode::registry_not_frozen,
            SolverOperation::lookup_solver,
            "freeze the solver registry before retaining descriptor references",
            id);
    }
    if (static_cast<std::size_t>(id.value) >= state_->slots.size()) {
        throw_invalid_id(id, SolverOperation::lookup_solver);
    }
    return state_->slots[static_cast<std::size_t>(id.value)].descriptor;
}

std::vector<SolverDescriptor> SolverRegistry::list_registered() const {
    const auto copy_descriptors = [this]() {
        std::vector<SolverDescriptor> result;
        result.reserve(state_->slots.size());
        for (const auto& slot : state_->slots) {
            result.push_back(slot.descriptor);
        }
        return result;
    };

    if (frozen()) {
        return copy_descriptors();
    }
    std::shared_lock<std::shared_mutex> lock(state_->mutex);
    return copy_descriptors();
}

std::unique_ptr<Solver> SolverRegistry::create(
    SolverId id,
    SolverDependencies dependencies,
    SolverConfig config) const {
    const auto checked_slot = [this, id]() -> const State::Slot& {
        if (static_cast<std::size_t>(id.value) >= state_->slots.size()) {
            throw_invalid_id(id, SolverOperation::create_solver);
        }
        return state_->slots[static_cast<std::size_t>(id.value)];
    };

    std::unique_ptr<Solver> solver;
    if (frozen()) {
        // Frozen slots are immutable. Calling the stored factory directly
        // avoids a std::function copy on the compact-ID construction path.
        solver = checked_slot().factory(dependencies, std::move(config));
    } else {
        SolverFactory factory;
        std::shared_lock<std::shared_mutex> lock(state_->mutex);
        factory = checked_slot().factory;
        lock.unlock();
        solver = factory(dependencies, std::move(config));
    }
    if (!solver) {
        throw SolverException(
            SolverErrorCode::factory_returned_null,
            SolverOperation::create_solver,
            "solver factory returned null",
            id);
    }
    return solver;
}

}  // namespace virne::solver
