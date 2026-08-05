#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace virne::core {
class Solution;
}

namespace virne::network {
class PhysicalNetwork;
class VirtualNetwork;
}

namespace virne::solver::learning::rl_core {

enum class RewardCalculatorKind : std::uint8_t {
    gradual_intermediate = 0,
    adaptive_intermediate = 1,
    fixed_intermediate = 2,
    vanilla = 3,
};

struct RewardCalculatorOptions {
    RewardCalculatorKind kind = RewardCalculatorKind::fixed_intermediate;
    double intermediate_reward = 0.1;
};

enum class RewardCalculatorErrorCode : std::uint8_t {
    invalid_kind,
    invalid_intermediate_reward,
    zero_virtual_nodes,
};

class RewardCalculatorException final : public std::runtime_error {
public:
    RewardCalculatorException(
        RewardCalculatorErrorCode code,
        std::string message);

    RewardCalculatorErrorCode code() const noexcept { return code_; }

private:
    RewardCalculatorErrorCode code_;
};

double current_place_progress(
    const network::VirtualNetwork& v_net,
    const core::Solution& solution);

class BaseRewardCalculator {
public:
    explicit BaseRewardCalculator(RewardCalculatorOptions options);
    virtual ~BaseRewardCalculator() = default;

    virtual double compute(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        core::Solution& solution) = 0;

    double v_net_reward() const noexcept { return v_net_reward_; }
    const RewardCalculatorOptions& options() const noexcept { return options_; }

protected:
    void commit_reward(core::Solution& solution, double reward) noexcept;
    RewardCalculatorOptions options_;
    double v_net_reward_ = 0.0;
};

class GradualIntermediateRewardCalculator final : public BaseRewardCalculator {
public:
    explicit GradualIntermediateRewardCalculator(RewardCalculatorOptions options)
        : BaseRewardCalculator(options) {}

    double compute(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        core::Solution& solution) override;
};

class AdaptiveWeightRewardCalculator final : public BaseRewardCalculator {
public:
    explicit AdaptiveWeightRewardCalculator(RewardCalculatorOptions options)
        : BaseRewardCalculator(options) {}

    double compute(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        core::Solution& solution) override;
};

class FixedWeightRewardCalculator final : public BaseRewardCalculator {
public:
    explicit FixedWeightRewardCalculator(RewardCalculatorOptions options)
        : BaseRewardCalculator(options) {}

    double compute(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        core::Solution& solution) override;
};

class VanillaRewardCalculator final : public BaseRewardCalculator {
public:
    explicit VanillaRewardCalculator(RewardCalculatorOptions options)
        : BaseRewardCalculator(options) {}

    double compute(
        const network::PhysicalNetwork& p_net,
        const network::VirtualNetwork& v_net,
        core::Solution& solution) override;
};

class RewardCalculatorRegistry final {
public:
    static constexpr std::size_t kSize = 4U;
    static constexpr std::size_t size() noexcept { return kSize; }

    static RewardCalculatorKind kind_from_name(std::string_view name);
    static std::string_view name(RewardCalculatorKind kind) noexcept;
    static std::array<RewardCalculatorKind, kSize> list_registered() noexcept;
    static std::unique_ptr<BaseRewardCalculator> create(
        RewardCalculatorOptions options);
};

}  // namespace virne::solver::learning::rl_core
