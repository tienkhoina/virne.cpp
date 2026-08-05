#include "../virne/solver/learning/rl_core/reward_calculator.h"

#include "../virne/core/solution.h"
#include "../virne/network/physical_network.h"
#include "../virne/network/virtual_network.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rl = virne::solver::learning::rl_core;
namespace core = virne::core;
namespace network = virne::network;

namespace {

struct BenchmarkOptions {
    bool enabled = false;
    int threads = 1;
    int warmups = 2;
    int repetitions = 5;
    int iterations = 1000;
};

struct RewardCase {
    bool result;
    bool place;
    bool route;
    std::size_t slots;
    double ratio;
};

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

int positive_argument(const std::string& value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return parsed;
}

BenchmarkOptions parse_arguments(const int argc, char** argv) {
    BenchmarkOptions result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--benchmark") {
            result.enabled = true;
        } else if (argument == "--threads" && index + 1 < argc) {
            result.threads = positive_argument(argv[++index], "--threads");
        } else if (argument == "--warmups" && index + 1 < argc) {
            result.warmups = positive_argument(argv[++index], "--warmups");
        } else if (argument == "--repetitions" && index + 1 < argc) {
            result.repetitions = positive_argument(argv[++index], "--repetitions");
        } else if (argument == "--iterations" && index + 1 < argc) {
            result.iterations = positive_argument(argv[++index], "--iterations");
        } else {
            throw std::runtime_error("unknown or incomplete harness argument: " + argument);
        }
    }
    return result;
}

void write_double_array(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << std::setprecision(17) << values[index];
    }
    std::cout << ']';
}

void prepare_solution(core::Solution& solution, const RewardCase& item) {
    solution.reset();
    solution.result = item.result;
    solution.place_result = item.place;
    solution.route_result = item.route;
    solution.v_net_r2c_ratio = item.ratio;
    for (std::size_t index = 0U; index < item.slots; ++index) {
        solution.node_slots.insert_or_assign(
            static_cast<core::SolutionNodeId>(index),
            static_cast<core::SolutionNodeId>(index + 10U));
    }
}

void write_calculator_case(
    const rl::RewardCalculatorKind kind,
    const network::PhysicalNetwork& physical,
    const network::VirtualNetwork& virtual_network,
    const std::vector<RewardCase>& cases) {
    auto calculator = rl::RewardCalculatorRegistry::create({kind, 0.1});
    std::vector<double> rewards;
    std::vector<double> totals;
    rewards.reserve(cases.size());
    totals.reserve(cases.size());
    for (const auto& item : cases) {
        auto solution = make_solution();
        prepare_solution(solution, item);
        rewards.push_back(calculator->compute(physical, virtual_network, solution));
        totals.push_back(solution.v_net_reward);
    }
    write_double_array(rewards);
    std::cout << ",\"solution_totals\":";
    write_double_array(totals);
    std::cout << ",\"calculator_total\":"
              << std::setprecision(17) << calculator->v_net_reward();
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto options = parse_arguments(argc, argv);
        const network::PhysicalNetwork physical;
        auto virtual_network = make_virtual_network();
        const std::vector<RewardCase> cases{
            {true, true, true, 0U, 0.75},
            {false, true, true, 1U, 0.75},
            {false, false, true, 1U, 0.75},
            {false, true, true, 0U, 0.0},
        };

        std::cout << "{\"component\":\"reward_calculator\",\"fixture_version\":1,\"outputs\":{";
        const auto registered = rl::RewardCalculatorRegistry::list_registered();
        for (std::size_t index = 0U; index < registered.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << '"' << rl::RewardCalculatorRegistry::name(registered[index]) << "\":{";
            std::cout << "\"rewards\":";
            write_calculator_case(registered[index], physical, virtual_network, cases);
            std::cout << '}';
        }
        std::cout << '}';

        if (options.enabled) {
            rl::RewardCalculatorOptions calculator_options{
                rl::RewardCalculatorKind::fixed_intermediate, 0.1};
            auto calculator = rl::RewardCalculatorRegistry::create(calculator_options);
            auto solution = make_solution();
            solution.result = true;
            solution.v_net_r2c_ratio = 0.75;
            for (int warmup = 0; warmup < options.warmups; ++warmup) {
                for (int iteration = 0; iteration < options.iterations; ++iteration) {
                    static_cast<void>(calculator->compute(physical, virtual_network, solution));
                }
            }
            std::vector<std::int64_t> samples;
            samples.reserve(static_cast<std::size_t>(options.repetitions));
            double last_reward = 0.0;
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < options.iterations; ++iteration) {
                    last_reward = calculator->compute(physical, virtual_network, solution);
                }
                samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
            }
            std::cout << ",\"benchmark\":{\"name\":\"fixed_reward_compute\",\"threads\":"
                      << options.threads << ",\"warmups\":" << options.warmups
                      << ",\"repetitions\":" << options.repetitions
                      << ",\"iterations\":" << options.iterations << ",\"samples_ns\":[";
            for (std::size_t index = 0U; index < samples.size(); ++index) {
                if (index != 0U) std::cout << ',';
                std::cout << samples[index];
            }
            std::cout << "],\"last_reward\":" << std::setprecision(17) << last_reward << '}';
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reward-calculator harness error: " << error.what() << '\n';
        return 1;
    }
}
