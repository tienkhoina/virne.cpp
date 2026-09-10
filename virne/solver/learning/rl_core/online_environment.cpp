#include "online_environment.h"
#include "reward_calculator.h"
#include <stdexcept>

namespace virne::solver::learning::rl_core {
OnlineEnvironment::OnlineEnvironment(network::PhysicalNetwork physical, network::VirtualNetworkRequestSimulator simulator,
    core::EnvironmentConfig lifecycle, InstanceEnvironmentOptions options)
    : lifecycle_(std::move(physical),std::move(simulator),std::move(lifecycle)),options_(std::move(options)) {}
void OnlineEnvironment::prepare_instance() {
    instance_.reset(); credited_instance_reward_=0.;
    if (done()) return;
    instance_ = std::make_unique<InstanceEnvironment>(lifecycle_.physical_network().clone(),
        lifecycle_.current_virtual_network().clone(),options_);
}
OnlineObservation OnlineEnvironment::reset() {
    instance_.reset(); cumulative_reward_ = 0.; lifecycle_.reset();
    if (lifecycle_.current_event().type == network::VirtualEventType::leave) lifecycle_.drain_leaves();
    prepare_instance(); return observation();
}
OnlineObservation OnlineEnvironment::observation() const {
    OnlineObservation result; result.done = done();
    if (lifecycle_.state().phase == core::EnvironmentPhase::unready) throw std::logic_error("online RL environment requires reset");
    if (!result.done) {
        result.event = lifecycle_.current_event();
        if (!instance_) throw std::logic_error("online RL instance is not ready");
        result.instance = instance_->observation();
        if (options_.mode == InstanceStepMode::solution) {
            result.instance->action_mask = {1,1}; result.instance->num_actions = 2;
        } else if (options_.mode == InstanceStepMode::node_slots) {
            const auto& raw = result.instance->action_mask;
            const auto virtual_size = result.instance->v_net_size;
            std::vector<std::uint8_t> current(instance_->physical_network().num_nodes());
            for (std::size_t node=0;node<current.size();++node)
                current[node]=raw[node*static_cast<std::size_t>(virtual_size)+static_cast<std::size_t>(result.instance->curr_v_node_id)];
            result.instance->action_mask=std::move(current);
            result.instance->num_actions=static_cast<std::int64_t>(result.instance->action_mask.size());
        }
    }
    return result;
}
OnlineStep OnlineEnvironment::transition(InstanceStep step) {
    OnlineStep result;
    result.episode = std::move(step.info); result.reward = step.reward; result.request_done = step.done;
    cumulative_reward_ += step.reward; credited_instance_reward_ += step.reward;
    result.cumulative_reward = cumulative_reward_;
    if (step.done) {
        auto solution = instance_->solution_snapshot();
        result.admission = lifecycle_.step(solution,core::EnvironmentSolutionState::detached);
        prepare_instance();
    }
    result.done = done(); result.observation = observation(); return result;
}
OnlineStep OnlineEnvironment::step(const std::int64_t action) { return transition(instance().step(action)); }
OnlineStep OnlineEnvironment::step_node_slots(const std::vector<NodeAssignment>& assignments) { return transition(instance().step_node_slots(assignments)); }
OnlineStep OnlineEnvironment::step_solution(const core::Solution& solution) {
    auto journal=solution;
    const auto& virtual_net=lifecycle_.current_virtual_network();
    core::Counter{}.prepare(virtual_net).count_solution(journal);
    auto calculator=RewardCalculatorRegistry::create({static_cast<RewardCalculatorKind>(options_.reward),options_.intermediate_reward});
    const double reward=calculator->compute(instance().physical_network(),virtual_net,journal);
    auto step=instance().step_solution(journal);step.reward=reward;
    return transition(std::move(step));
}
OnlineStep OnlineEnvironment::reject() { return transition(instance().reject()); }
OnlineStep OnlineEnvironment::revoke() { return transition(instance().revoke()); }
OnlineStep OnlineEnvironment::finish_instance() {
    auto& active = instance();
    if (!active.done()) throw std::logic_error("cannot commit an unfinished RL instance");
    const auto info = active.info();
    return transition({active.observation(),info.cumulative_reward-credited_instance_reward_,true,info});
}
InstanceEnvironment& OnlineEnvironment::instance() {
    if (!instance_) throw std::logic_error("online RL environment has no active instance");
    return *instance_;
}
const network::PhysicalNetwork& OnlineEnvironment::physical_network() const noexcept {
    return instance_ ? instance_->physical_network() : lifecycle_.physical_network();
}
const core::SolutionStepEnvironment& OnlineEnvironment::lifecycle() const noexcept { return lifecycle_; }
double OnlineEnvironment::cumulative_reward() const noexcept { return cumulative_reward_; }
bool OnlineEnvironment::done() const noexcept { return lifecycle_.state().phase == core::EnvironmentPhase::finished; }
}  // namespace virne::solver::learning::rl_core
