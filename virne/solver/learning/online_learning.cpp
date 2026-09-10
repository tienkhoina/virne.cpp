#include "online_learning.h"

namespace virne::solver::learning {
OnlineLearningEpoch run_online_learning_epoch(rl_core::OnlineEnvironment& environment,
    LearningSolver& solver,const OnlineLearningOptions options) {
    OnlineLearningEpoch result;
    while(!environment.done()) {
        const auto& lifecycle=environment.lifecycle();
        const SolverInstance instance{lifecycle.current_virtual_network(),lifecycle.physical_network()};
        auto learned=options.train?solver.learn(instance):solver.evaluate(instance,options.sample);
        const auto transition=environment.step_solution(learned.solution);
        ++result.requests;
        if(transition.admission->accepted) ++result.accepted;
        result.reward+=learned.episode.solution.cumulative_reward;
        if(options.capture) {
            result.episodes.push_back(std::move(learned.episode));
            result.inservice_counts.push_back(environment.lifecycle().recorder().state().inservice_count);
        }
        solver.increment_num_arrived_virtual_networks();
    }
    if(options.train && options.flush_rollout) result.final_loss=solver.update();
    result.updates=solver.update_time();result.pending_steps=solver.pending_steps();
    return result;
}
}  // namespace virne::solver::learning
