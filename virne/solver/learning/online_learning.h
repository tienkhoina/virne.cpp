#pragma once
#include "solver_registry.h"
#include "rl_core/online_environment.h"

namespace virne::solver::learning {
struct OnlineLearningOptions { bool train=true, sample=false, flush_rollout=true, capture=false; };
struct OnlineLearningEpoch {
    std::size_t requests=0,accepted=0;
    std::int64_t updates=0,pending_steps=0;
    double reward=0.;
    std::optional<double> final_loss;
    std::vector<rl_core::EpisodeLearningResult> episodes;
    std::vector<std::int64_t> inservice_counts;
};
// Uses the same detached Solution contract as the standard Solver registry.
// The caller selects reset/renew behavior before invoking an epoch.
OnlineLearningEpoch run_online_learning_epoch(rl_core::OnlineEnvironment& environment,
    LearningSolver& solver,OnlineLearningOptions options={});
}  // namespace virne::solver::learning
