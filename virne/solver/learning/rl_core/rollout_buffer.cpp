#include "rollout_buffer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace virne::solver::learning::rl_core {

namespace {

template <typename T>
void append_selected(
    std::vector<T>& destination,
    const std::vector<T>& source,
    const std::vector<std::size_t>& indices) {
    destination.reserve(destination.size() + indices.size());
    for (const std::size_t index : indices) destination.push_back(source[index]);
}

template <typename T>
void append_all(std::vector<T>& destination, const std::vector<T>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

template <typename T>
void append_range(
    std::vector<T>& destination,
    const std::vector<T>& source,
    const std::size_t begin,
    const std::size_t count) {
    destination.insert(
        destination.end(), source.begin() + static_cast<std::ptrdiff_t>(begin),
        source.begin() + static_cast<std::ptrdiff_t>(begin + count));
}

void validate_selected_indices(
    const std::vector<std::size_t>& indices,
    const std::size_t source_size) {
    for (const std::size_t index : indices) {
        if (index >= source_size) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "rollout subbuffer index is out of range");
        }
    }
}

}  // namespace

RolloutBufferException::RolloutBufferException(
    const RolloutBufferErrorCode code,
    std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

RolloutExtensionId rollout_extension_id_from_name(const std::string_view name) {
    if (name == "hidden_states") return RolloutExtensionId::hidden_states;
    if (name == "cell_states") return RolloutExtensionId::cell_states;
    if (name == "action_mask") return RolloutExtensionId::action_mask;
    if (name == "entropies") return RolloutExtensionId::entropies;
    if (name == "feasibility_flags") return RolloutExtensionId::feasibility_flags;
    if (name == "feasibility_budgets") {
        return RolloutExtensionId::feasibility_budgets;
    }
    throw RolloutBufferException(
        RolloutBufferErrorCode::invalid_extension,
        "unsupported rollout extension: " + std::string(name));
}

RolloutBufferMethod rollout_buffer_method_from_name(const std::string_view name) {
    if (name == "gae") return RolloutBufferMethod::gae;
    if (name == "mc") return RolloutBufferMethod::mc;
    if (name == "td") return RolloutBufferMethod::td;
    throw RolloutBufferException(
        RolloutBufferErrorCode::invalid_method,
        "unsupported rollout buffer method: " + std::string(name));
}

std::string_view rollout_extension_name(const RolloutExtensionId id) noexcept {
    switch (id) {
    case RolloutExtensionId::hidden_states: return "hidden_states";
    case RolloutExtensionId::cell_states: return "cell_states";
    case RolloutExtensionId::action_mask: return "action_mask";
    case RolloutExtensionId::entropies: return "entropies";
    case RolloutExtensionId::feasibility_flags: return "feasibility_flags";
    case RolloutExtensionId::feasibility_budgets: return "feasibility_budgets";
    }
    return {};
}

void RolloutBuffer::reset() {
    curr_idx = 0U;
    observations.clear();
    actions.clear();
    rewards.clear();
    dones.clear();
    next_observations.clear();
    has_next_observation.clear();
    logprobs.clear();
    values.clear();
    advantages.clear();
    returns.clear();
    hidden_state_ids.clear();
    cell_state_ids.clear();
    action_mask_ids.clear();
    entropy_ids.clear();
    feasibility_flags.clear();
    feasibility_budgets.clear();
}

void RolloutBuffer::reserve(const std::size_t transition_count) {
    observations.reserve(transition_count);
    actions.reserve(transition_count);
    rewards.reserve(transition_count);
    dones.reserve(transition_count);
    next_observations.reserve(transition_count);
    has_next_observation.reserve(transition_count);
    logprobs.reserve(transition_count);
    values.reserve(transition_count);
    advantages.reserve(transition_count);
    returns.reserve(transition_count);
    if (hidden_states_enabled) hidden_state_ids.reserve(transition_count);
    if (cell_states_enabled) cell_state_ids.reserve(transition_count);
    if (action_mask_enabled) action_mask_ids.reserve(transition_count);
    if (entropies_enabled) entropy_ids.reserve(transition_count);
    if (feasibility_flags_enabled) feasibility_flags.reserve(transition_count);
    if (feasibility_budgets_enabled) feasibility_budgets.reserve(transition_count);
}

void RolloutBuffer::add(
    const ObservationId observation,
    const std::int64_t action,
    const double reward,
    const bool done,
    const double logprob,
    const double value,
    const ObservationId next_observation,
    const bool has_next) {
    observations.push_back(observation);
    actions.push_back(action);
    rewards.push_back(reward);
    dones.push_back(done ? 1U : 0U);
    next_observations.push_back(next_observation);
    has_next_observation.push_back(has_next ? 1U : 0U);
    logprobs.push_back(logprob);
    values.push_back(value);
    ++curr_idx;
}

void RolloutBuffer::extend(const std::string_view name) {
    extend(rollout_extension_id_from_name(name));
}

void RolloutBuffer::extend(const RolloutExtensionId id) {
    switch (id) {
    case RolloutExtensionId::hidden_states:
        hidden_states_enabled = true;
        hidden_state_ids.clear();
        break;
    case RolloutExtensionId::cell_states:
        cell_states_enabled = true;
        cell_state_ids.clear();
        break;
    case RolloutExtensionId::action_mask:
        action_mask_enabled = true;
        action_mask_ids.clear();
        break;
    case RolloutExtensionId::entropies:
        entropies_enabled = true;
        entropy_ids.clear();
        break;
    case RolloutExtensionId::feasibility_flags:
        feasibility_flags_enabled = true;
        feasibility_flags.clear();
        break;
    case RolloutExtensionId::feasibility_budgets:
        feasibility_budgets_enabled = true;
        feasibility_budgets.clear();
        break;
    default:
        throw RolloutBufferException(
            RolloutBufferErrorCode::invalid_extension,
            "unsupported rollout extension enum");
    }
}

bool RolloutBuffer::is_extension_enabled(const RolloutExtensionId id) const noexcept {
    switch (id) {
    case RolloutExtensionId::hidden_states: return hidden_states_enabled;
    case RolloutExtensionId::cell_states: return cell_states_enabled;
    case RolloutExtensionId::action_mask: return action_mask_enabled;
    case RolloutExtensionId::entropies: return entropies_enabled;
    case RolloutExtensionId::feasibility_flags: return feasibility_flags_enabled;
    case RolloutExtensionId::feasibility_budgets: return feasibility_budgets_enabled;
    }
    return false;
}

void RolloutBuffer::validate_transition_lengths() const {
    const std::size_t count = size();
    if (observations.size() != count || actions.size() != count
        || rewards.size() != count || dones.size() != count
        || next_observations.size() != count
        || has_next_observation.size() != count || values.size() != count) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "rollout basic lanes must have equal lengths");
    }
}

void RolloutBuffer::copy_indices_into(
    RolloutBuffer& destination,
    const std::vector<std::size_t>& indices) const {
    validate_transition_lengths();
    validate_selected_indices(indices, size());
    destination.curr_idx = 0U;
    destination.hidden_states_enabled = hidden_states_enabled;
    destination.cell_states_enabled = cell_states_enabled;
    destination.action_mask_enabled = action_mask_enabled;
    destination.entropies_enabled = entropies_enabled;
    destination.feasibility_flags_enabled = feasibility_flags_enabled;
    destination.feasibility_budgets_enabled = feasibility_budgets_enabled;
    append_selected(destination.observations, observations, indices);
    append_selected(destination.actions, actions, indices);
    append_selected(destination.rewards, rewards, indices);
    append_selected(destination.dones, dones, indices);
    append_selected(destination.next_observations, next_observations, indices);
    append_selected(destination.has_next_observation, has_next_observation, indices);
    append_selected(destination.logprobs, logprobs, indices);
    append_selected(destination.values, values, indices);
    if (!advantages.empty()) {
        if (advantages.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "advantages length differs from rollout size");
        }
        append_selected(destination.advantages, advantages, indices);
    }
    if (!returns.empty()) {
        if (returns.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "returns length differs from rollout size");
        }
        append_selected(destination.returns, returns, indices);
    }
    if (hidden_states_enabled && !hidden_state_ids.empty()) {
        if (hidden_state_ids.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "hidden_states extension length differs from rollout size");
        }
        append_selected(destination.hidden_state_ids, hidden_state_ids, indices);
    }
    if (cell_states_enabled && !cell_state_ids.empty()) {
        if (cell_state_ids.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "cell_states extension length differs from rollout size");
        }
        append_selected(destination.cell_state_ids, cell_state_ids, indices);
    }
    if (action_mask_enabled && !action_mask_ids.empty()) {
        if (action_mask_ids.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "action_mask extension length differs from rollout size");
        }
        append_selected(destination.action_mask_ids, action_mask_ids, indices);
    }
    if (entropies_enabled && !entropy_ids.empty()) {
        if (entropy_ids.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "entropies extension length differs from rollout size");
        }
        append_selected(destination.entropy_ids, entropy_ids, indices);
    }
    if (feasibility_flags_enabled && !feasibility_flags.empty()) {
        if (feasibility_flags.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "feasibility_flags extension length differs from rollout size");
        }
        append_selected(destination.feasibility_flags, feasibility_flags, indices);
    }
    if (feasibility_budgets_enabled && !feasibility_budgets.empty()) {
        if (feasibility_budgets.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "feasibility_budgets extension length differs from rollout size");
        }
        append_selected(destination.feasibility_budgets, feasibility_budgets, indices);
    }
}

void RolloutBuffer::copy_range_into(
    RolloutBuffer& destination,
    const std::size_t begin,
    const std::size_t count) const {
    validate_transition_lengths();
    if (begin > size() || count > size() - begin) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "rollout range is out of bounds");
    }
    destination.curr_idx = 0U;
    destination.hidden_states_enabled = hidden_states_enabled;
    destination.cell_states_enabled = cell_states_enabled;
    destination.action_mask_enabled = action_mask_enabled;
    destination.entropies_enabled = entropies_enabled;
    destination.feasibility_flags_enabled = feasibility_flags_enabled;
    destination.feasibility_budgets_enabled = feasibility_budgets_enabled;
    append_range(destination.observations, observations, begin, count);
    append_range(destination.actions, actions, begin, count);
    append_range(destination.rewards, rewards, begin, count);
    append_range(destination.dones, dones, begin, count);
    append_range(destination.next_observations, next_observations, begin, count);
    append_range(destination.has_next_observation, has_next_observation, begin, count);
    append_range(destination.logprobs, logprobs, begin, count);
    append_range(destination.values, values, begin, count);
    if (!advantages.empty()) {
        if (advantages.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "advantages length differs from rollout size");
        }
        append_range(destination.advantages, advantages, begin, count);
    }
    if (!returns.empty()) {
        if (returns.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                "returns length differs from rollout size");
        }
        append_range(destination.returns, returns, begin, count);
    }
    const auto copy_extension = [this, &destination, begin, count](
                                    const bool enabled,
                                    const auto& source,
                                    auto& target,
                                    const char* name) {
        if (!enabled || source.empty()) return;
        if (source.size() != size()) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                std::string(name) + " extension length differs from rollout size");
        }
        append_range(target, source, begin, count);
    };
    copy_extension(hidden_states_enabled, hidden_state_ids,
                   destination.hidden_state_ids, "hidden_states");
    copy_extension(cell_states_enabled, cell_state_ids,
                   destination.cell_state_ids, "cell_states");
    copy_extension(action_mask_enabled, action_mask_ids,
                   destination.action_mask_ids, "action_mask");
    copy_extension(entropies_enabled, entropy_ids,
                   destination.entropy_ids, "entropies");
    copy_extension(feasibility_flags_enabled, feasibility_flags,
                   destination.feasibility_flags, "feasibility_flags");
    copy_extension(feasibility_budgets_enabled, feasibility_budgets,
                   destination.feasibility_budgets, "feasibility_budgets");
}

RolloutBuffer RolloutBuffer::get_subbuffer(
    const std::vector<std::size_t>& indices) const {
    RolloutBuffer result;
    copy_indices_into(result, indices);
    return result;
}

std::vector<RolloutBuffer> RolloutBuffer::split_with_ratios(
    const std::vector<double>& ratios) const {
    validate_transition_lengths();
    if (ratios.empty()) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::invalid_ratio,
            "rollout split ratios cannot be empty");
    }
    double total = 0.0;
    for (const double ratio : ratios) {
        if (!std::isfinite(ratio) || ratio < 0.0) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::invalid_ratio,
                "rollout split ratios must be finite and non-negative");
        }
        total += ratio;
    }
    if (total != 1.0) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::invalid_ratio,
            "rollout split ratios must sum to one");
    }
    std::vector<RolloutBuffer> result;
    result.reserve(ratios.size());
    std::size_t cursor = 0U;
    for (std::size_t split = 0U; split < ratios.size(); ++split) {
        const std::size_t count = split + 1U == ratios.size()
            ? size() - cursor
            : static_cast<std::size_t>(static_cast<double>(size()) * ratios[split]);
        result.emplace_back();
        copy_range_into(result.back(), cursor, count);
        cursor += count;
    }
    return result;
}

std::vector<RolloutBuffer> RolloutBuffer::split_with_instance() const {
    validate_transition_lengths();
    std::vector<RolloutBuffer> result;
    std::size_t cursor = 0U;
    for (std::size_t index = 0U; index < size(); ++index) {
        if (dones[index] == 0U) continue;
        // Preserve buffer.py's observable absolute-index slicing, including
        // its historical behavior for more than two terminal transitions.
        const std::size_t remaining = size() - cursor;
        const std::size_t count = std::min(index + 1U, remaining);
        result.emplace_back();
        copy_range_into(result.back(), cursor, count);
        cursor += count;
    }
    return result;
}

void RolloutBuffer::append_extension_values(const RolloutBuffer& buffer) {
    if (hidden_states_enabled) {
        if (!buffer.hidden_states_enabled) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::disabled_extension,
                "source buffer lacks hidden_states extension");
        }
        append_all(hidden_state_ids, buffer.hidden_state_ids);
    }
    if (cell_states_enabled) {
        if (!buffer.cell_states_enabled) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::disabled_extension,
                "source buffer lacks cell_states extension");
        }
        append_all(cell_state_ids, buffer.cell_state_ids);
    }
    if (action_mask_enabled) {
        if (!buffer.action_mask_enabled) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::disabled_extension,
                "source buffer lacks action_mask extension");
        }
        append_all(action_mask_ids, buffer.action_mask_ids);
    }
    if (entropies_enabled) {
        if (!buffer.entropies_enabled) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::disabled_extension,
                "source buffer lacks entropies extension");
        }
        append_all(entropy_ids, buffer.entropy_ids);
    }
    if (feasibility_flags_enabled) {
        if (!buffer.feasibility_flags_enabled) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::disabled_extension,
                "source buffer lacks feasibility_flags extension");
        }
        append_all(feasibility_flags, buffer.feasibility_flags);
    }
    if (feasibility_budgets_enabled) {
        if (!buffer.feasibility_budgets_enabled) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::disabled_extension,
                "source buffer lacks feasibility_budgets extension");
        }
        append_all(feasibility_budgets, buffer.feasibility_budgets);
    }
}

void RolloutBuffer::merge(const RolloutBuffer& buffer) {
    if (this == &buffer) {
        const RolloutBuffer snapshot = buffer;
        merge(snapshot);
        return;
    }
    validate_transition_lengths();
    buffer.validate_transition_lengths();
    append_all(observations, buffer.observations);
    append_all(actions, buffer.actions);
    append_all(rewards, buffer.rewards);
    append_all(dones, buffer.dones);
    append_all(next_observations, buffer.next_observations);
    append_all(has_next_observation, buffer.has_next_observation);
    append_all(logprobs, buffer.logprobs);
    append_all(values, buffer.values);
    append_all(advantages, buffer.advantages);
    append_all(returns, buffer.returns);
    append_extension_values(buffer);
}

void RolloutBuffer::compute_returns_and_advantages(
    const double last_value,
    const double gamma,
    const double gae_lambda,
    const RolloutBufferMethod method) {
    validate_transition_lengths();
    switch (method) {
    case RolloutBufferMethod::gae: {
        auto result = compute_returns_with_gae(
            rewards, values, dones, last_value, gamma, gae_lambda);
        returns = std::move(result.returns);
        advantages = std::move(result.advantages);
        return;
    }
    case RolloutBufferMethod::mc:
        returns = compute_returns_with_mc(rewards, dones, gamma);
        advantages.assign(size(), 0.0);
        return;
    case RolloutBufferMethod::td:
        returns = compute_returns_with_td(rewards, values, dones, last_value, gamma);
        advantages.assign(size(), 0.0);
        return;
    }
    throw RolloutBufferException(
        RolloutBufferErrorCode::invalid_method,
        "unsupported rollout buffer method enum");
}

void RolloutBufferWithCost::reset() {
    RolloutBuffer::reset();
    costs.clear();
    cost_values.clear();
    cost_returns.clear();
    baseline_cost_returns.clear();
}

void RolloutBufferWithCost::reserve(const std::size_t transition_count) {
    RolloutBuffer::reserve(transition_count);
    costs.reserve(transition_count);
    cost_values.reserve(transition_count);
    cost_returns.reserve(transition_count);
    baseline_cost_returns.reserve(transition_count);
}

void RolloutBufferWithCost::validate_cost_lane_lengths() const {
    const std::size_t transition_count = size();
    const auto validate_aligned_lane = [transition_count](
                                           const std::size_t lane_size,
                                           const char* lane_name) {
        if (lane_size != 0U && lane_size != transition_count) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                std::string(lane_name) + " length differs from rollout size");
        }
    };
    validate_aligned_lane(costs.size(), "costs");
    validate_aligned_lane(cost_values.size(), "cost_values");
    validate_aligned_lane(
        baseline_cost_returns.size(), "baseline_cost_returns");

    // compute_cost_returns prepends a complete transition-aligned generation
    // on every call, matching Python.  Therefore k * transition_count is a
    // valid calculated-lane length, while a partial generation is malformed.
    if (!cost_returns.empty()
        && (transition_count == 0U
            || cost_returns.size() % transition_count != 0U)) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "cost_returns length is not transition aligned");
    }
}

void RolloutBufferWithCost::merge(const RolloutBufferWithCost& buffer) {
    if (this == &buffer) {
        const RolloutBufferWithCost snapshot = buffer;
        merge(snapshot);
        return;
    }
    validate_cost_lane_lengths();
    buffer.validate_cost_lane_lengths();

    // Keep every populated cost lane aligned after concatenation.  Empty
    // aggregate buffers remain a zero-copy schema boundary for the first
    // episode, while mixing populated and absent lanes in non-empty buffers is
    // rejected before the base columns are mutated.
    const std::size_t merged_transition_count = size() + buffer.size();
    const auto validate_merged_aligned_lane = [merged_transition_count](
                                                  const std::size_t lane_size,
                                                  const char* lane_name) {
        if (lane_size != 0U && lane_size != merged_transition_count) {
            throw RolloutBufferException(
                RolloutBufferErrorCode::length_mismatch,
                std::string("merged ") + lane_name
                    + " length differs from rollout size");
        }
    };
    validate_merged_aligned_lane(
        costs.size() + buffer.costs.size(), "costs");
    validate_merged_aligned_lane(
        cost_values.size() + buffer.cost_values.size(), "cost_values");
    validate_merged_aligned_lane(
        baseline_cost_returns.size() + buffer.baseline_cost_returns.size(),
        "baseline_cost_returns");
    const std::size_t merged_cost_return_count =
        cost_returns.size() + buffer.cost_returns.size();
    if (merged_cost_return_count != 0U
        && (merged_transition_count == 0U
            || merged_cost_return_count % merged_transition_count != 0U)) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "merged cost_returns length is not transition aligned");
    }
    RolloutBuffer::merge(buffer);
    append_all(costs, buffer.costs);
    append_all(cost_values, buffer.cost_values);
    append_all(cost_returns, buffer.cost_returns);
    append_all(baseline_cost_returns, buffer.baseline_cost_returns);
}

void RolloutBufferWithCost::merge(const RolloutBuffer& buffer) {
    const auto* cost_buffer = dynamic_cast<const RolloutBufferWithCost*>(&buffer);
    if (cost_buffer != nullptr) {
        merge(*cost_buffer);
        return;
    }
    validate_cost_lane_lengths();
    if (buffer.size() != 0U
        && (!costs.empty() || !cost_values.empty() || !cost_returns.empty()
            || !baseline_cost_returns.empty())) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "cannot append a base rollout to populated cost lanes");
    }
    RolloutBuffer::merge(buffer);
}

void RolloutBufferWithCost::compute_cost_returns(
    const double gamma,
    const CostReturnMethod method) {
    validate_transition_lengths();
    if (costs.size() != size()) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "rollout costs and rewards must have equal lengths");
    }
    if (method == CostReturnMethod::cumulative) {
        for (double& cost : costs) cost = cost >= 0.0 ? cost : 0.0;
    }
    auto computed =
        ::virne::solver::learning::rl_core::compute_cost_returns(
            costs, dones, gamma, method);
    // Python prepends each new result to an existing cost_returns list.  A
    // single range insertion at the front is equivalent and remains O(n).
    if (cost_returns.empty()) {
        cost_returns = std::move(computed);
        return;
    }
    cost_returns.insert(cost_returns.begin(), computed.begin(), computed.end());
}

RolloutBufferWithCost RolloutBufferWithCost::get_cost_subbuffer(
    const std::vector<std::size_t>& indices) const {
    validate_cost_lane_lengths();
    RolloutBufferWithCost result;
    copy_indices_into(result, indices);
    if (!costs.empty()) {
        validate_selected_indices(indices, costs.size());
        append_selected(result.costs, costs, indices);
    }
    if (!cost_values.empty()) {
        validate_selected_indices(indices, cost_values.size());
        append_selected(result.cost_values, cost_values, indices);
    }
    if (!cost_returns.empty()) {
        validate_selected_indices(indices, cost_returns.size());
        append_selected(result.cost_returns, cost_returns, indices);
    }
    if (!baseline_cost_returns.empty()) {
        validate_selected_indices(indices, baseline_cost_returns.size());
        append_selected(result.baseline_cost_returns, baseline_cost_returns, indices);
    }
    return result;
}

void RolloutBufferWithAvgReturn::compute_returns_and_advantages(
    const double last_value,
    const double gamma,
    const double gae_lambda,
    const RolloutBufferMethod method) {
    validate_transition_lengths();
    if (method != RolloutBufferMethod::gae && method != RolloutBufferMethod::td) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::invalid_method,
            "average return buffer supports only gae and td");
    }
    if (size() == 0U) {
        throw RolloutBufferException(
            RolloutBufferErrorCode::length_mismatch,
            "average return buffer cannot be empty");
    }
    dones.back() = 0U;
    const double mean_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0)
        / static_cast<double>(size());
    returns.assign(size(), 0.0);
    advantages.assign(size(), 0.0);
    if (method == RolloutBufferMethod::gae) {
        for (double& reward : rewards) reward -= mean_reward;
        auto result = compute_returns_with_gae(
            rewards, values, dones, last_value, gamma, gae_lambda);
        returns = std::move(result.returns);
        advantages = std::move(result.advantages);
        return;
    }
    if (method == RolloutBufferMethod::td) {
        (void)gamma;
        for (std::size_t reverse = 0U; reverse < size(); ++reverse) {
            const std::size_t step = size() - 1U - reverse;
            const double next_value = step + 1U == size() ? last_value : values[step + 1U];
            const double next_non_terminal = 1.0 - static_cast<double>(dones[step]);
            advantages[step] = rewards[step] - mean_reward
                + next_value * next_non_terminal - values[step];
            returns[step] = advantages[step] + values[step];
        }
        return;
    }
}

}  // namespace virne::solver::learning::rl_core
