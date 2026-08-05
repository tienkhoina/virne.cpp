#pragma once

#include "rollout_returns.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace virne::solver::learning::rl_core {

// Observations and tensor-valued extension items live in caller-owned arenas.
// The buffer stores stable IDs, so copying/splitting a transition never copies
// a graph or performs a dynamic Python-key lookup.
using ObservationId = std::uint64_t;
using TensorId = std::uint64_t;

enum class RolloutExtensionId : std::uint8_t {
    hidden_states = 0,
    cell_states = 1,
    action_mask = 2,
    entropies = 3,
    feasibility_flags = 4,
    feasibility_budgets = 5,
};

constexpr std::size_t kRolloutExtensionCount = 6U;

enum class RolloutBufferMethod : std::uint8_t {
    gae = 0,
    mc = 1,
    td = 2,
};

enum class RolloutBufferErrorCode : std::uint8_t {
    invalid_extension,
    invalid_method,
    invalid_ratio,
    length_mismatch,
    disabled_extension,
};

class RolloutBufferException final : public std::runtime_error {
public:
    RolloutBufferException(
        RolloutBufferErrorCode code,
        std::string message);

    [[nodiscard]] RolloutBufferErrorCode code() const noexcept { return code_; }

private:
    RolloutBufferErrorCode code_;
};

RolloutExtensionId rollout_extension_id_from_name(std::string_view name);
RolloutBufferMethod rollout_buffer_method_from_name(std::string_view name);
std::string_view rollout_extension_name(RolloutExtensionId id) noexcept;

class RolloutBuffer {
public:
    // Kept as a direct field for compatibility with Python callers.  Python
    // increments it on add(), but does not adjust it during merge/split.
    std::size_t curr_idx = 0U;

    // Fixed basic lanes.  IDs refer to caller-owned observation/tensor arenas.
    std::vector<ObservationId> observations;
    std::vector<std::int64_t> actions;
    std::vector<double> rewards;
    std::vector<std::uint8_t> dones;
    std::vector<ObservationId> next_observations;
    std::vector<std::uint8_t> has_next_observation;
    std::vector<double> logprobs;
    std::vector<double> values;

    // Fixed calculated lanes.
    std::vector<double> advantages;
    std::vector<double> returns;

    // Fixed extension lanes.  The enabled flags are resolved once by extend;
    // the copy/split/merge loops branch on these direct booleans only.
    bool hidden_states_enabled = true;
    bool cell_states_enabled = true;
    bool action_mask_enabled = true;
    bool entropies_enabled = true;
    bool feasibility_flags_enabled = false;
    bool feasibility_budgets_enabled = false;
    std::vector<TensorId> hidden_state_ids;
    std::vector<TensorId> cell_state_ids;
    std::vector<TensorId> action_mask_ids;
    std::vector<TensorId> entropy_ids;
    std::vector<double> feasibility_flags;
    std::vector<double> feasibility_budgets;

    RolloutBuffer() = default;
    virtual ~RolloutBuffer() = default;

    virtual void reset();
    void clear() { reset(); }
    virtual void reserve(std::size_t transition_count);

    [[nodiscard]] std::size_t size() const noexcept {
        return logprobs.size();
    }

    void add(
        ObservationId observation,
        std::int64_t action,
        double reward,
        bool done,
        double logprob,
        double value = 0.0,
        ObservationId next_observation = 0U,
        bool has_next = false);

    // String resolution is a cold configuration operation.  Production loops
    // should retain the enum overload and never call this overload per item.
    void extend(std::string_view name);
    void extend(RolloutExtensionId id);

    [[nodiscard]] bool is_extension_enabled(RolloutExtensionId id) const noexcept;

    [[nodiscard]] RolloutBuffer get_subbuffer(
        const std::vector<std::size_t>& indices) const;
    [[nodiscard]] std::vector<RolloutBuffer> split_with_ratios(
        const std::vector<double>& ratios) const;
    [[nodiscard]] std::vector<RolloutBuffer> split_with_ratios() const {
        return split_with_ratios({0.5, 0.5});
    }
    [[nodiscard]] std::vector<RolloutBuffer> split_with_instance() const;

    // Merge follows Python's list concatenation order.  curr_idx intentionally
    // remains unchanged, matching buffer.py's public side effect.
    virtual void merge(const RolloutBuffer& buffer);

    virtual void compute_returns_and_advantages(
        double last_value = 0.0,
        double gamma = 0.99,
        double gae_lambda = 0.98,
        RolloutBufferMethod method = RolloutBufferMethod::gae);

protected:
    void copy_indices_into(
        RolloutBuffer& destination,
        const std::vector<std::size_t>& indices) const;
    void copy_range_into(
        RolloutBuffer& destination,
        std::size_t begin,
        std::size_t count) const;
    void validate_transition_lengths() const;
    void append_extension_values(const RolloutBuffer& buffer);
};

class RolloutBufferWithCost : public RolloutBuffer {
public:
    std::vector<double> costs;
    std::vector<double> cost_values;
    std::vector<double> cost_returns;
    std::vector<double> baseline_cost_returns;

    void reset() override;
    void clear() { reset(); }
    void reserve(std::size_t transition_count) override;

    using RolloutBuffer::merge;
    void merge(const RolloutBufferWithCost& buffer);
    void merge(const RolloutBuffer& buffer) override;

    void compute_cost_returns(
        double gamma = 1.0,
        CostReturnMethod method = CostReturnMethod::reachability);

    [[nodiscard]] RolloutBufferWithCost get_cost_subbuffer(
        const std::vector<std::size_t>& indices) const;
};

class RolloutBufferWithAvgReturn : public RolloutBuffer {
public:
    void compute_returns_and_advantages(
        double last_value = 0.0,
        double gamma = 0.99,
        double gae_lambda = 0.98,
        RolloutBufferMethod method = RolloutBufferMethod::gae) override;
};

}  // namespace virne::solver::learning::rl_core
