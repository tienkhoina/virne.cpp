#include "../../virne/solver/learning/rl_core/rollout_returns.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rl = virne::solver::learning::rl_core;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " (length)");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require(std::abs(actual[index] - expected[index]) < 1e-12,
                message + " (value)");
    }
}

void test_returns() {
    const std::vector<double> items{1.2, -0.4, 0.7, 0.1, -0.3};
    const std::vector<double> values{0.2, 0.3, -0.1, 0.4, 0.05};
    const std::vector<std::uint8_t> dones{0U, 0U, 1U, 0U, 1U};
    const auto gae = rl::compute_returns_with_gae(
        items, values, dones, 0.25, 0.91, 0.83);
    require_close(
        gae.returns,
        {1.3319401719999997, 0.11323999999999984, 0.7,
         -0.11885499999999993, -0.3},
        "GAE returns");
    require_close(
        gae.advantages,
        {1.1319401719999997, -0.18676000000000015, 0.7999999999999999,
         -0.518855, -0.35},
        "GAE advantages");
    require_close(
        rl::compute_returns_with_mc(items, dones, 0.91),
        {1.41567, 0.237, 0.7, -0.17299999999999996, -0.3},
        "MC returns");
    require_close(
        rl::compute_returns_with_td(items, values, dones, 0.25, 0.91),
        {1.273, -0.791, 0.8, -0.2545, -0.35},
        "TD returns");
}

void test_cost_returns() {
    const std::vector<double> costs{1.0, -2.0, 0.5, 2.0, -1.0};
    const std::vector<std::uint8_t> dones{0U, 0U, 1U, 0U, 1U};
    require_close(
        rl::compute_cost_returns(costs, dones, 0.8, rl::CostReturnMethod::reachability),
        {1.0, 0.4, 0.5, 2.0, -1.0},
        "reachability cost returns");
    require_close(
        rl::compute_cost_returns(costs, dones, 0.8, rl::CostReturnMethod::cumulative),
        {1.32, 0.4, 0.5, 2.0, 0.0},
        "cumulative cost returns");
    require(rl::cost_return_method_from_name("cumulative") == rl::CostReturnMethod::cumulative,
            "cost method lookup");
}

void test_errors_and_empty() {
    require(rl::compute_returns_with_mc({}, {}, 0.9).empty(), "empty MC");
    bool length_threw = false;
    try {
        static_cast<void>(rl::compute_returns_with_gae({1.0}, {}, {0U}));
    } catch (const rl::RolloutReturnException& error) {
        length_threw = error.code() == rl::RolloutReturnErrorCode::length_mismatch;
    }
    require(length_threw, "length mismatch");
    bool method_threw = false;
    try {
        static_cast<void>(rl::cost_return_method_from_name("unknown"));
    } catch (const rl::RolloutReturnException& error) {
        method_threw = error.code() == rl::RolloutReturnErrorCode::invalid_method;
    }
    require(method_threw, "invalid method");
}

}  // namespace

int main() {
    try {
        test_returns();
        test_cost_returns();
        test_errors_and_empty();
        std::cout << "rollout_returns_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rollout_returns_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
