#pragma once

#include "../base_solver.h"
#include "rl_core/learning_session.h"

#include <array>
#include <string_view>

namespace virne::solver::learning {

// Source-specific environment constructors are resolved once at registration.
enum class LearningEnvironmentProfile : std::uint8_t {
    placement_vanilla, placement_vanilla_more_metrics, joint_node,
    placement_node, placement_adaptive, joint_dual, joint_dual_gradual,
};
struct LearningSolverDescriptor {
    std::string_view name;
    rl_policy::PolicyKind policy;
    rl_core::UpdateAlgorithm algorithm;
    LearningEnvironmentProfile environment;
    bool force_successful_experience = false;
};

// Only the sequential, non-recurrent PG/A2C/PPO source registrations are listed.
const std::array<LearningSolverDescriptor, 34>& registered_learning_solvers() noexcept;
const LearningSolverDescriptor& learning_solver_descriptor(std::string_view name);

struct LearningSolverOptions {
    rl_core::LearningSessionOptions session;
    rl_core::InstanceEnvironmentOptions environment;
    rank::NodeRankParameters node_ranking;
    // Runtime decoding is selected once at the CLI boundary. Training still
    // samples internally; inference can opt into the same stochastic policy.
    bool inference_sample = false;
};

// Resolves the policy, algorithm and mandatory environment overrides of a
// canonical Python registration. Other typed hyperparameters are retained.
LearningSolverOptions resolve_learning_solver_options(
    std::string_view name, LearningSolverOptions options = {});

struct LearningSolverResult {
    core::Solution solution;
    rl_core::EpisodeLearningResult episode;
};

class LearningSolver final : public Solver {
public:
    LearningSolver(SolverDependencies dependencies, SolverConfig config,
        std::string_view canonical_name, LearningSolverOptions options = {});
    ~LearningSolver() override;

    // The core Solver seam always performs greedy evaluation and returns a
    // detached journal. The owning system environment deploys that journal.
    core::Solution solve(const SolverInstance& instance) override;
    LearningSolverResult evaluate(const SolverInstance& instance, bool sample = false);
    LearningSolverResult learn(const SolverInstance& instance);
    std::optional<double> update();

    std::string_view canonical_name() const noexcept;
    const LearningSolverOptions& learning_options() const noexcept;
    std::int64_t pending_steps() const noexcept;
    std::int64_t update_time() const noexcept;
    void set_sampling_seed(std::uint64_t seed);
    std::vector<double> parameter_values() const;

    // Session checkpoint bytes include policy, Adam, pending rollout and RNG.
    // Restore into the same solver/environment configuration used to save it.
    std::vector<std::uint8_t> checkpoint() const;
    void restore_checkpoint(const std::vector<std::uint8_t>& bytes);

private:
    const LearningSolverDescriptor* descriptor_;
    LearningSolverOptions options_;
    std::unique_ptr<rl_core::LearningSession> session_;
    rl_core::InstanceEnvironment make_environment(const SolverInstance& instance) const;
};

// Registration leaves the shared registry open so other solver families can
// be added before the caller freezes it. Factories own their captured options.
std::vector<SolverId> register_learning_solvers(
    SolverRegistry& registry, LearningSolverOptions options = {});

}  // namespace virne::solver::learning
