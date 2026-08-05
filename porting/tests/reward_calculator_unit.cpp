#include "../../virne/solver/learning/rl_core/reward_calculator.h"

#include "../../virne/core/solution.h"
#include "../../virne/network/physical_network.h"
#include "../../virne/network/virtual_network.h"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rl = virne::solver::learning::rl_core;
namespace core = virne::core;
namespace network = virne::network;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

network::VirtualNetwork make_virtual_network() {
    network::BaseNetworkConstruction construction;
    construction.incoming_graph.emplace(
        Graph(3U, std::vector<EdgeEndpoints>{{0U, 1U}, {1U, 2U}}));
    return network::VirtualNetwork(std::move(construction));
}

core::Solution make_solution() {
    core::SolutionMetadata metadata;
    metadata.v_net_num_nodes = 3U;
    metadata.v_net_num_edges = 2U;
    return core::Solution(metadata);
}

void test_registry() {
    const auto registered = rl::RewardCalculatorRegistry::list_registered();
    require(registered.size() == 4U, "registry size");
    require(
        rl::RewardCalculatorRegistry::kind_from_name("fixed_intermediate")
            == rl::RewardCalculatorKind::fixed_intermediate,
        "fixed registry lookup");
    require(
        rl::RewardCalculatorRegistry::name(registered[0])
            == "gradual_intermediate",
        "registry order");
    bool threw = false;
    try {
        static_cast<void>(rl::RewardCalculatorRegistry::kind_from_name("missing"));
    } catch (const rl::RewardCalculatorException& error) {
        threw = error.code() == rl::RewardCalculatorErrorCode::invalid_kind;
    }
    require(threw, "unknown registry name");
}

void test_rewards() {
    network::PhysicalNetwork physical;
    auto virtual_network = make_virtual_network();

    auto solution = make_solution();
    solution.node_slots.insert_or_assign(0, 4);
    require(
        std::abs(rl::current_place_progress(virtual_network, solution) - 0.5)
            < 1e-12,
        "placement progress");
    solution.v_net_r2c_ratio = 0.8;
    solution.place_result = true;
    solution.route_result = true;

    auto gradual = rl::RewardCalculatorRegistry::create({
        rl::RewardCalculatorKind::gradual_intermediate, 0.1});
    require(std::abs(gradual->compute(physical, virtual_network, solution) - 0.04) < 1e-12,
            "gradual partial reward");
    require(std::abs(solution.v_net_reward - 0.04) < 1e-12, "solution reward update");
    require(std::abs(gradual->v_net_reward() - 0.04) < 1e-12, "calculator reward update");

    solution.reset();
    solution.v_net_r2c_ratio = 0.8;
    auto fixed = rl::RewardCalculatorRegistry::create({
        rl::RewardCalculatorKind::fixed_intermediate, 0.25});
    require(std::abs(fixed->compute(physical, virtual_network, solution) - 0.25) < 1e-12,
            "fixed partial reward");

    solution.reset();
    solution.place_result = false;
    auto adaptive = rl::RewardCalculatorRegistry::create({
        rl::RewardCalculatorKind::adaptive_intermediate, 0.1});
    require(std::abs(adaptive->compute(physical, virtual_network, solution) + 1.0 / 3.0) < 1e-12,
            "adaptive failure reward");

    solution.reset();
    solution.result = true;
    solution.v_net_r2c_ratio = 0.8;
    auto vanilla = rl::RewardCalculatorRegistry::create({
        rl::RewardCalculatorKind::vanilla, 0.1});
    require(std::abs(vanilla->compute(physical, virtual_network, solution) - 0.8) < 1e-12,
            "vanilla result reward");

    bool threw = false;
    try {
        static_cast<void>(rl::RewardCalculatorRegistry::create({
            rl::RewardCalculatorKind::fixed_intermediate, -0.2}));
    } catch (const rl::RewardCalculatorException& error) {
        threw = error.code() == rl::RewardCalculatorErrorCode::invalid_intermediate_reward;
    }
    require(threw, "negative intermediate reward validation");
}

}  // namespace

int main() {
    try {
        test_registry();
        test_rewards();
        std::cout << "reward_calculator_unit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reward_calculator_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
