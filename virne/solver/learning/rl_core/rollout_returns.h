#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace virne::solver::learning::rl_core {

struct ReturnAdvantageResult {
    std::vector<double> returns;
    std::vector<double> advantages;
};

enum class CostReturnMethod : std::uint8_t {
    reachability = 0,
    cumulative = 1,
};

enum class RolloutReturnErrorCode : std::uint8_t {
    length_mismatch,
    invalid_method,
};

class RolloutReturnException final : public std::runtime_error {
public:
    RolloutReturnException(
        RolloutReturnErrorCode code,
        std::string message);

    RolloutReturnErrorCode code() const noexcept { return code_; }

private:
    RolloutReturnErrorCode code_;
};

ReturnAdvantageResult compute_returns_with_gae(
    const std::vector<double>& items,
    const std::vector<double>& values,
    const std::vector<std::uint8_t>& dones,
    double last_value = 0.0,
    double gamma = 0.99,
    double gae_lambda = 0.98);

std::vector<double> compute_returns_with_mc(
    const std::vector<double>& items,
    const std::vector<std::uint8_t>& dones,
    double gamma = 0.99);

std::vector<double> compute_returns_with_td(
    const std::vector<double>& items,
    const std::vector<double>& values,
    const std::vector<std::uint8_t>& dones,
    double last_value = 0.0,
    double gamma = 0.99);

CostReturnMethod cost_return_method_from_name(std::string_view name);

std::vector<double> compute_cost_returns(
    const std::vector<double>& costs,
    const std::vector<std::uint8_t>& dones,
    double gamma = 1.0,
    CostReturnMethod method = CostReturnMethod::reachability);

}  // namespace virne::solver::learning::rl_core
