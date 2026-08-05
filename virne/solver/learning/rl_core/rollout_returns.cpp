#include "rollout_returns.h"

#include <limits>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

void validate_lengths(
    const std::size_t item_count,
    const std::vector<double>& values,
    const std::vector<std::uint8_t>& dones) {
    if (values.size() != item_count || dones.size() != item_count) {
        throw RolloutReturnException(
            RolloutReturnErrorCode::length_mismatch,
            "rollout items, values, and dones must have equal lengths");
    }
}

void validate_item_lengths(
    const std::size_t item_count,
    const std::vector<std::uint8_t>& dones) {
    if (dones.size() != item_count) {
        throw RolloutReturnException(
            RolloutReturnErrorCode::length_mismatch,
            "rollout items and dones must have equal lengths");
    }
}

// Python's built-in max keeps the first value when the second comparison is
// false.  Spelling that rule explicitly also preserves Python's NaN behavior;
// std::max has a different result for a NaN first operand.
double python_max(const double left, const double right) noexcept {
    return right > left ? right : left;
}

}  // namespace

RolloutReturnException::RolloutReturnException(
    const RolloutReturnErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

ReturnAdvantageResult compute_returns_with_gae(
    const std::vector<double>& items,
    const std::vector<double>& values,
    const std::vector<std::uint8_t>& dones,
    const double last_value,
    const double gamma,
    const double gae_lambda) {
    validate_lengths(items.size(), values, dones);
    ReturnAdvantageResult result;
    result.returns.assign(items.size(), 0.0);
    result.advantages.assign(items.size(), 0.0);
    double last_gae_lam = 0.0;
    for (std::size_t reverse = 0U; reverse < items.size(); ++reverse) {
        const std::size_t step = items.size() - 1U - reverse;
        const double next_values = step == items.size() - 1U
            ? last_value
            : values[step + 1U];
        const double next_non_terminal = 1.0 - static_cast<double>(dones[step]);
        const double delta = items[step]
            + gamma * next_values * next_non_terminal - values[step];
        last_gae_lam = delta
            + gamma * gae_lambda * next_non_terminal * last_gae_lam;
        result.advantages[step] = last_gae_lam;
        result.returns[step] = result.advantages[step] + values[step];
    }
    return result;
}

std::vector<double> compute_returns_with_mc(
    const std::vector<double>& items,
    const std::vector<std::uint8_t>& dones,
    const double gamma) {
    validate_item_lengths(items.size(), dones);
    std::vector<double> result(items.size(), 0.0);
    double discounted_item = 0.0;
    for (std::size_t reverse = 0U; reverse < items.size(); ++reverse) {
        const std::size_t step = items.size() - 1U - reverse;
        if (dones[step] != 0U) discounted_item = 0.0;
        discounted_item = items[step] + gamma * discounted_item;
        result[step] = discounted_item;
    }
    return result;
}

std::vector<double> compute_returns_with_td(
    const std::vector<double>& items,
    const std::vector<double>& values,
    const std::vector<std::uint8_t>& dones,
    const double last_value,
    const double gamma) {
    validate_lengths(items.size(), values, dones);
    std::vector<double> result(items.size(), 0.0);
    for (std::size_t reverse = 0U; reverse < items.size(); ++reverse) {
        const std::size_t step = items.size() - 1U - reverse;
        const double next_values = step == items.size() - 1U
            ? last_value
            : values[step + 1U];
        const double next_non_terminal = 1.0 - static_cast<double>(dones[step]);
        const double discounted_item = items[step]
            + gamma * next_values * next_non_terminal - values[step];
        result[step] = discounted_item;
    }
    return result;
}

CostReturnMethod cost_return_method_from_name(const std::string_view name) {
    if (name == "reachability") return CostReturnMethod::reachability;
    if (name == "cumulative") return CostReturnMethod::cumulative;
    throw RolloutReturnException(
        RolloutReturnErrorCode::invalid_method,
        "unsupported cost return method: " + std::string(name));
}

std::vector<double> compute_cost_returns(
    const std::vector<double>& costs,
    const std::vector<std::uint8_t>& dones,
    const double gamma,
    const CostReturnMethod method) {
    validate_item_lengths(costs.size(), dones);
    std::vector<double> result(costs.size(), 0.0);
    double state_cost = 0.0;
    if (method == CostReturnMethod::reachability) {
        for (std::size_t reverse = 0U; reverse < costs.size(); ++reverse) {
            const std::size_t step = costs.size() - 1U - reverse;
            if (dones[step] != 0U) {
                state_cost = -std::numeric_limits<double>::infinity();
            }
            state_cost = python_max(costs[step], gamma * state_cost);
            result[step] = state_cost;
        }
        return result;
    }
    if (method == CostReturnMethod::cumulative) {
        state_cost = 0.0;
        for (std::size_t reverse = 0U; reverse < costs.size(); ++reverse) {
            const std::size_t step = costs.size() - 1U - reverse;
            if (dones[step] != 0U) state_cost = 0.0;
            const double nonnegative_cost = costs[step] >= 0.0 ? costs[step] : 0.0;
            state_cost = nonnegative_cost + gamma * state_cost;
            result[step] = state_cost;
        }
        return result;
    }
    throw RolloutReturnException(
        RolloutReturnErrorCode::invalid_method,
        "unsupported cost return method enum");
}

}  // namespace virne::solver::learning::rl_core
