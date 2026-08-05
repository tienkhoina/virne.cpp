#include "reward_calculator.h"

#include "../../../core/solution.h"
#include "../../../network/physical_network.h"
#include "../../../network/virtual_network.h"

#include <limits>

namespace virne::solver::learning::rl_core {

namespace {

[[noreturn]] void throw_reward(
    const RewardCalculatorErrorCode code,
    std::string message) {
    throw RewardCalculatorException(code, std::move(message));
}

bool completed(const core::Solution& solution) noexcept {
    return solution.place_result && solution.route_result;
}

double r2c(const core::Solution& solution) noexcept {
    return solution.v_net_r2c_ratio;
}

}  // namespace

RewardCalculatorException::RewardCalculatorException(
    const RewardCalculatorErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

double current_place_progress(
    const network::VirtualNetwork& v_net,
    const core::Solution& solution) {
    const std::size_t total_nodes = v_net.num_nodes();
    const std::size_t denominator = total_nodes > 1U ? total_nodes - 1U : 1U;
    return static_cast<double>(solution.node_slots.size())
        / static_cast<double>(denominator);
}

BaseRewardCalculator::BaseRewardCalculator(RewardCalculatorOptions options)
    : options_(options) {
    if (options_.intermediate_reward != -1.0
        && options_.intermediate_reward < 0.0) {
        throw_reward(
            RewardCalculatorErrorCode::invalid_intermediate_reward,
            "intermediate_reward must be non-negative or -1");
    }
}

void BaseRewardCalculator::commit_reward(
    core::Solution& solution,
    const double reward) noexcept {
    solution.v_net_reward += reward;
    v_net_reward_ += reward;
}

double GradualIntermediateRewardCalculator::compute(
    const network::PhysicalNetwork&,
    const network::VirtualNetwork& v_net,
    core::Solution& solution) {
    double reward = 0.0;
    if (solution.result) {
        reward = r2c(solution);
    } else if (completed(solution)) {
        reward = 0.1 * current_place_progress(v_net, solution) * r2c(solution);
    } else {
        reward = -current_place_progress(v_net, solution);
    }
    commit_reward(solution, reward);
    return reward;
}

double AdaptiveWeightRewardCalculator::compute(
    const network::PhysicalNetwork&,
    const network::VirtualNetwork& v_net,
    core::Solution& solution) {
    const std::size_t node_count = v_net.num_nodes();
    if (node_count == 0U) {
        throw_reward(
            RewardCalculatorErrorCode::zero_virtual_nodes,
            "adaptive reward requires at least one virtual node");
    }
    const double weight = 1.0 / static_cast<double>(node_count);
    const double reward = solution.result
        ? r2c(solution)
        : (completed(solution) ? weight : -weight);
    commit_reward(solution, reward);
    return reward;
}

double FixedWeightRewardCalculator::compute(
    const network::PhysicalNetwork&,
    const network::VirtualNetwork&,
    core::Solution& solution) {
    const double weight = options_.intermediate_reward;
    const double reward = solution.result
        ? r2c(solution)
        : (completed(solution) ? weight : -weight);
    commit_reward(solution, reward);
    return reward;
}

double VanillaRewardCalculator::compute(
    const network::PhysicalNetwork&,
    const network::VirtualNetwork&,
    core::Solution& solution) {
    const double reward = solution.result ? r2c(solution) : 0.0;
    commit_reward(solution, reward);
    return reward;
}

RewardCalculatorKind RewardCalculatorRegistry::kind_from_name(
    const std::string_view name_value) {
    if (name_value == "gradual_intermediate") {
        return RewardCalculatorKind::gradual_intermediate;
    }
    if (name_value == "adaptive_intermediate") {
        return RewardCalculatorKind::adaptive_intermediate;
    }
    if (name_value == "fixed_intermediate") {
        return RewardCalculatorKind::fixed_intermediate;
    }
    if (name_value == "vanilla") {
        return RewardCalculatorKind::vanilla;
    }
    throw_reward(
        RewardCalculatorErrorCode::invalid_kind,
        "reward calculator is not registered: " + std::string(name_value));
}

std::string_view RewardCalculatorRegistry::name(
    const RewardCalculatorKind kind) noexcept {
    switch (kind) {
    case RewardCalculatorKind::gradual_intermediate:
        return "gradual_intermediate";
    case RewardCalculatorKind::adaptive_intermediate:
        return "adaptive_intermediate";
    case RewardCalculatorKind::fixed_intermediate:
        return "fixed_intermediate";
    case RewardCalculatorKind::vanilla:
        return "vanilla";
    }
    return "";
}

std::array<RewardCalculatorKind, RewardCalculatorRegistry::size()>
RewardCalculatorRegistry::list_registered() noexcept {
    return {
        RewardCalculatorKind::gradual_intermediate,
        RewardCalculatorKind::adaptive_intermediate,
        RewardCalculatorKind::fixed_intermediate,
        RewardCalculatorKind::vanilla,
    };
}

std::unique_ptr<BaseRewardCalculator> RewardCalculatorRegistry::create(
    const RewardCalculatorOptions options) {
    switch (options.kind) {
    case RewardCalculatorKind::gradual_intermediate:
        return std::make_unique<GradualIntermediateRewardCalculator>(options);
    case RewardCalculatorKind::adaptive_intermediate:
        return std::make_unique<AdaptiveWeightRewardCalculator>(options);
    case RewardCalculatorKind::fixed_intermediate:
        return std::make_unique<FixedWeightRewardCalculator>(options);
    case RewardCalculatorKind::vanilla:
        return std::make_unique<VanillaRewardCalculator>(options);
    }
    throw_reward(
        RewardCalculatorErrorCode::invalid_kind,
        "reward calculator enum is not registered");
}

}  // namespace virne::solver::learning::rl_core
