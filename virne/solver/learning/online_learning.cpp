#include "online_learning.h"

#include "../../core/logger.h"

#include <optional>

namespace {
const std::vector<std::string>& learning_metric_names() {
    static const std::vector<std::string> names{
        "lr", "loss/loss", "loss/actor_loss", "loss/critic_loss",
        "loss/entropy_loss", "value/logprob", "value/old_action_logprob",
        "info/approx_kl", "value/value", "value/return",
        "value/advantage", "value/reward", "grad/grad_clipped"};
    return names;
}

double metric_value(const std::optional<double>& value) {
    return value.value_or(0.0);
}
} // namespace

namespace virne::solver::learning {
OnlineLearningEpoch run_online_learning_epoch(rl_core::OnlineEnvironment& environment,
    LearningSolver& solver,const OnlineLearningOptions options) {
    OnlineLearningEpoch result;
    if (options.log_updates && options.train) {
        solver.logger().register_metrics(learning_metric_names());
    }
    while(!environment.done()) {
        const auto& lifecycle=environment.lifecycle();
        const SolverInstance instance{lifecycle.current_virtual_network(),lifecycle.physical_network()};
        auto learned=options.train?solver.learn(instance):solver.evaluate(instance,options.sample);
        if (options.log_updates && options.train && learned.episode.update_loss) {
            const auto& session = solver.learning_options().session;
            const std::vector<std::optional<double>> values{
                session.learning_rates.actor.value_or(0.0),
                *learned.episode.update_loss,
                metric_value(learned.episode.actor_loss),
                metric_value(learned.episode.critic_loss),
                metric_value(learned.episode.entropy_loss),
                metric_value(learned.episode.log_probability_mean),
                metric_value(learned.episode.old_log_probability_mean),
                metric_value(learned.episode.approximate_kl),
                metric_value(learned.episode.value_mean),
                metric_value(learned.episode.return_mean),
                metric_value(learned.episode.advantage_mean),
                metric_value(learned.episode.reward_mean),
                metric_value(learned.episode.gradient_norm)};
            solver.logger().log(core::LoggerEntry{
                "Learning update",
                core::LoggerLevel::info,
                core::LoggerMetricRow{learned.episode.update_time, values}});
        }
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
