#include "meta_heuristic.h"

#include "../../../random/py_random.h"
#include "../../core/controller/controller.h"
#include "../../core/counter.h"
#include "../../network/physical_network.h"
#include "../../network/virtual_network.h"
#include "../../utils/deterministic_executor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

namespace virne::solver::meta {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

std::vector<Vertex> ordered_nodes(const Graph& graph) {
    std::vector<Vertex> result;
    result.reserve(graph.num_nodes());
    for (const Vertex node : graph.node_view()) {
        result.push_back(node);
    }
    return result;
}

std::size_t population_size(const MetaHeuristicOptions& options) {
    return std::max<std::size_t>(1U, options.population_size);
}

std::size_t iteration_count(const MetaHeuristicOptions& options) {
    return options.max_iterations;
}

} // namespace

MetaHeuristicSolver::MetaHeuristicSolver(
    SolverDependencies dependencies,
    SolverConfig config,
    MetaAlgorithm algorithm,
    PyRandom& random,
    MetaHeuristicOptions options)
    : Solver(dependencies, std::move(config)),
      algorithm_(algorithm),
      random_(&random),
      options_(options) {}

core::NodeSlots MetaHeuristicSolver::make_node_slots(
    const std::vector<Vertex>& virtual_nodes,
    const std::vector<core::SolutionNodeId>& position) const {
    core::NodeSlots slots;
    const std::size_t count = std::min(
        virtual_nodes.size(), position.size());
    for (std::size_t index = 0U; index < count; ++index) {
        slots.insert_or_assign(
            static_cast<core::SolutionNodeId>(virtual_nodes[index]),
            position[index]);
    }
    return slots;
}

MetaHeuristicSolver::Candidate MetaHeuristicSolver::evaluate(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes,
    const std::vector<core::SolutionNodeId>& position) const {
    network::PhysicalNetwork working_physical = physical_network.clone();
    core::Solution solution = core::Solution::from_v_net(virtual_network);
    auto prepared_controller = controller().prepare(
        virtual_network, working_physical);

    core::controller::DeployWithNodeSlotsOptions deploy_options;
    deploy_options.shortest_method = config().shortest_method;
    deploy_options.k = config().k_shortest;
    deploy_options.max_path_nodes = 1.0e6;
    deploy_options.workers = core::controller::ControllerWorkers{
        options_.topology_constraint_workers,
        options_.candidate_workers};
    (void)prepared_controller.deploy_with_node_slots(
        make_node_slots(virtual_nodes, position),
        solution,
        deploy_options);

    // Counter is prepared against the immutable virtual graph.  It fills the
    // fixed cost/revenue fields used by every meta objective without touching
    // the caller's Recorder or the live physical network.
    const auto prepared_counter = counter().prepare(virtual_network);
    prepared_counter.count_solution(solution);

    double fitness = kInfinity;
    if (solution.result && solution.v_net_revenue > 0.0 &&
        std::isfinite(solution.v_net_cost) &&
        std::isfinite(solution.v_net_revenue)) {
        fitness = solution.v_net_cost / solution.v_net_revenue;
    }
    return Candidate(position, std::move(solution), fitness);
}

MetaHeuristicSolver::Population MetaHeuristicSolver::evaluate_positions(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes,
    const PositionBatch& positions) const {
    if (positions.empty()) {
        return {};
    }

    // The coordinator owns all random draws. Each worker receives an already
    // materialized position and evaluates it against its own physical clone;
    // no shared Solution, network mutation, Recorder, or RNG is touched.
    // deterministic_parallel_blocks preserves index order and also collapses
    // nested controller parallelism on a worker to avoid oversubscription.
    std::vector<std::optional<Candidate>> slots(positions.size());
    utils::deterministic_parallel_blocks(
        positions.size(),
        options_.evaluation_workers,
        1U,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                slots[index].emplace(evaluate(
                    virtual_network,
                    physical_network,
                    virtual_nodes,
                    positions[index]));
            }
        });

    Population result;
    result.reserve(slots.size());
    for (auto& slot : slots) {
        result.push_back(std::move(*slot));
    }
    return result;
}

MetaHeuristicSolver::Population MetaHeuristicSolver::initial_population(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes) {
    const std::size_t count = population_size(options_);
    PositionBatch positions;
    positions.reserve(count);
    positions.push_back(greedy_position(
        virtual_nodes.size(), physical_network.num_nodes()));
    while (positions.size() < count) {
        positions.push_back(random_position(
            virtual_nodes.size(), physical_network.num_nodes()));
    }
    return evaluate_positions(
        virtual_network, physical_network, virtual_nodes, positions);
}

std::vector<core::SolutionNodeId> MetaHeuristicSolver::greedy_position(
    std::size_t virtual_node_count,
    std::size_t physical_node_count) const {
    std::vector<core::SolutionNodeId> result(
        virtual_node_count, core::SolutionNodeId{-1});
    const std::size_t count = std::min(
        virtual_node_count, physical_node_count);
    for (std::size_t index = 0U; index < count; ++index) {
        result[index] = static_cast<core::SolutionNodeId>(index);
    }
    return result;
}

std::vector<core::SolutionNodeId> MetaHeuristicSolver::random_position(
    std::size_t virtual_node_count,
    std::size_t physical_node_count) {
    std::vector<core::SolutionNodeId> result(
        virtual_node_count, core::SolutionNodeId{-1});
    std::vector<core::SolutionNodeId> available;
    available.reserve(physical_node_count);
    for (std::size_t index = 0U; index < physical_node_count; ++index) {
        available.push_back(static_cast<core::SolutionNodeId>(index));
    }
    const std::size_t count = std::min(
        virtual_node_count, physical_node_count);
    for (std::size_t index = 0U; index < count; ++index) {
        const auto selected = static_cast<std::size_t>(
            random_->randrange(static_cast<std::uint64_t>(available.size())));
        result[index] = available[selected];
        available[selected] = available.back();
        available.pop_back();
    }
    return result;
}

void MetaHeuristicSolver::repair_position(
    std::vector<core::SolutionNodeId>& position,
    std::size_t physical_node_count) const {
    std::vector<std::uint8_t> used(physical_node_count, 0U);
    for (core::SolutionNodeId& value : position) {
        if (value >= 0 &&
            static_cast<std::uint64_t>(value) < physical_node_count &&
            used[static_cast<std::size_t>(value)] == 0U) {
            used[static_cast<std::size_t>(value)] = 1U;
            continue;
        }

        value = core::SolutionNodeId{-1};
        for (std::size_t candidate = 0U;
             candidate < physical_node_count;
             ++candidate) {
            if (used[candidate] == 0U) {
                used[candidate] = 1U;
                value = static_cast<core::SolutionNodeId>(candidate);
                break;
            }
        }
    }
}

std::vector<core::SolutionNodeId> MetaHeuristicSolver::neighbor_position(
    const std::vector<core::SolutionNodeId>& position,
    std::size_t physical_node_count) {
    std::vector<core::SolutionNodeId> result = position;
    if (result.empty() || physical_node_count < 2U) {
        return result;
    }

    for (std::size_t attempt = 0U;
         attempt < std::max<std::size_t>(1U, options_.max_attempts);
         ++attempt) {
        const std::size_t virtual_index = static_cast<std::size_t>(
            random_->randrange(static_cast<std::uint64_t>(result.size())));
        std::vector<core::SolutionNodeId> available;
        available.reserve(physical_node_count);
        for (std::size_t candidate = 0U;
             candidate < physical_node_count;
             ++candidate) {
            const auto value = static_cast<core::SolutionNodeId>(candidate);
            if (value == result[virtual_index]) {
                continue;
            }
            bool used_elsewhere = false;
            for (std::size_t other = 0U; other < result.size(); ++other) {
                if (other != virtual_index && result[other] == value) {
                    used_elsewhere = true;
                    break;
                }
            }
            if (!used_elsewhere) {
                available.push_back(value);
            }
        }
        if (!available.empty()) {
            const auto selected = static_cast<std::size_t>(
                random_->randrange(
                    static_cast<std::uint64_t>(available.size())));
            result[virtual_index] = available[selected];
            return result;
        }
    }
    return result;
}

bool MetaHeuristicSolver::better(
    const Candidate& left,
    const Candidate& right) const {
    if (left.fitness < right.fitness) {
        return true;
    }
    if (left.fitness > right.fitness ||
        !std::isfinite(left.fitness) ||
        !std::isfinite(right.fitness)) {
        return false;
    }
    return left.position < right.position;
}

MetaHeuristicSolver::Candidate MetaHeuristicSolver::best_of(
    const Population& population) const {
    Candidate best = population.front();
    for (std::size_t index = 1U; index < population.size(); ++index) {
        if (better(population[index], best)) {
            best = population[index];
        }
    }
    return best;
}

core::Solution MetaHeuristicSolver::run_genetic(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes) {
    const std::size_t count = population_size(options_);
    Population population = initial_population(
        virtual_network, physical_network, virtual_nodes);
    Candidate global_best = best_of(population);
    const double mutation_probability = virtual_nodes.empty()
        ? 0.0
        : 1.0 / (static_cast<double>(virtual_nodes.size()) *
                 static_cast<double>(count) / 2.0);

    for (std::size_t iteration = 0U;
         iteration < iteration_count(options_);
         ++iteration) {
        PositionBatch next_positions;
        next_positions.reserve(count);
        while (next_positions.size() < count) {
            auto tournament = [&]() -> const Candidate& {
                const Candidate* selected = nullptr;
                for (std::size_t draw = 0U; draw < 3U; ++draw) {
                    const std::size_t index = static_cast<std::size_t>(
                        random_->randrange(static_cast<std::uint64_t>(
                            population.size())));
                    if (selected == nullptr ||
                        better(population[index], *selected)) {
                        selected = &population[index];
                    }
                }
                return *selected;
            };

            std::vector<core::SolutionNodeId> first = tournament().position;
            std::vector<core::SolutionNodeId> second = tournament().position;
            if (first.size() > 1U &&
                random_->random() < options_.crossover_probability) {
                std::size_t left = static_cast<std::size_t>(random_->randrange(
                    static_cast<std::uint64_t>(first.size())));
                std::size_t right = static_cast<std::size_t>(random_->randrange(
                    static_cast<std::uint64_t>(first.size())));
                if (left > right) {
                    std::swap(left, right);
                }
                for (std::size_t index = left; index <= right; ++index) {
                    std::swap(first[index], second[index]);
                }
            }

            auto mutate = [&](std::vector<core::SolutionNodeId>& position) {
                for (std::size_t index = 0U; index < position.size(); ++index) {
                    if (random_->random() < mutation_probability) {
                        const auto changed = neighbor_position(
                            position, physical_network.num_nodes());
                        position = changed;
                    }
                }
                repair_position(position, physical_network.num_nodes());
            };
            mutate(first);
            mutate(second);
            next_positions.push_back(std::move(first));
            if (next_positions.size() < count) {
                next_positions.push_back(std::move(second));
            }
        }
        population = evaluate_positions(
            virtual_network,
            physical_network,
            virtual_nodes,
            next_positions);
        const Candidate iteration_best = best_of(population);
        if (better(iteration_best, global_best)) {
            global_best = iteration_best;
        }
    }
    return global_best.solution;
}

core::Solution MetaHeuristicSolver::run_simulated_annealing(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes) {
    Population individuals = initial_population(
        virtual_network, physical_network, virtual_nodes);
    Candidate global_best = best_of(individuals);
    std::vector<double> temperatures(
        individuals.size(), options_.initial_temperature);
    for (std::size_t iteration = 0U;
         iteration < iteration_count(options_);
         ++iteration) {
        // Neighbours and acceptance draws stay on the coordinator. Only the
        // independent mapping/cost evaluations enter the deterministic batch.
        PositionBatch next_positions;
        next_positions.reserve(individuals.size());
        for (const Candidate& current : individuals) {
            next_positions.push_back(neighbor_position(
                current.position, physical_network.num_nodes()));
        }
        Population next_population = evaluate_positions(
            virtual_network,
            physical_network,
            virtual_nodes,
            next_positions);
        for (std::size_t index = 0U; index < individuals.size(); ++index) {
            Candidate& current = individuals[index];
            Candidate next = std::move(next_population[index]);
            const double temperature = temperatures[index];
            bool accept = better(next, current);
            if (!accept && std::isfinite(current.fitness) &&
                std::isfinite(next.fitness) && temperature > 0.0) {
                const double probability = std::exp(
                    -(next.fitness - current.fitness) / temperature);
                accept = random_->random() < probability;
            } else if (!std::isfinite(current.fitness) &&
                       std::isfinite(next.fitness)) {
                accept = true;
            }
            if (accept) {
                current = std::move(next);
                if (better(current, global_best)) {
                    global_best = current;
                }
            }
            temperatures[index] *= options_.attenuation_factor;
        }
    }
    return global_best.solution;
}

core::Solution MetaHeuristicSolver::run_tabu_search(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes) {
    const std::size_t count = population_size(options_);
    Population individuals = initial_population(
        virtual_network, physical_network, virtual_nodes);
    Candidate global_best = best_of(individuals);
    using Move = std::pair<std::size_t, core::SolutionNodeId>;
    std::vector<std::vector<Move>> tabu(count);

    for (std::size_t iteration = 0U;
         iteration < iteration_count(options_);
         ++iteration) {
        PositionBatch next_positions;
        next_positions.reserve(individuals.size());
        std::vector<Move> selected_moves(
            individuals.size(), Move{0U, core::SolutionNodeId{-1}});
        std::vector<std::uint8_t> found(individuals.size(), 0U);

        // Tabu decisions are serial and index ordered; expensive mapping is
        // deferred to one deterministic batch after every candidate is known.
        for (std::size_t individual_index = 0U;
             individual_index < individuals.size();
             ++individual_index) {
            const auto& current = individuals[individual_index];
            std::vector<core::SolutionNodeId> next_position = current.position;
            for (std::size_t attempt = 0U;
                 attempt < std::max<std::size_t>(1U, options_.max_attempts);
                 ++attempt) {
                if (next_position.empty()) {
                    break;
                }
                const auto candidate_position = neighbor_position(
                    next_position, physical_network.num_nodes());
                std::size_t virtual_index = next_position.size();
                for (std::size_t gene = 0U;
                     gene < next_position.size();
                     ++gene) {
                    if (candidate_position[gene] != next_position[gene]) {
                        virtual_index = gene;
                        break;
                    }
                }
                if (virtual_index == next_position.size()) {
                    continue;
                }
                const Move move{
                    virtual_index, candidate_position[virtual_index]};
                if (std::find(tabu[individual_index].begin(),
                              tabu[individual_index].end(),
                              move) != tabu[individual_index].end()) {
                    continue;
                }
                next_position = candidate_position;
                selected_moves[individual_index] = move;
                found[individual_index] = 1U;
                break;
            }
            next_positions.push_back(std::move(next_position));
        }

        Population next_population = evaluate_positions(
            virtual_network,
            physical_network,
            virtual_nodes,
            next_positions);
        for (std::size_t individual_index = 0U;
             individual_index < individuals.size();
             ++individual_index) {
            if (found[individual_index] == 0U) {
                continue;
            }
            Candidate& current = individuals[individual_index];
            Candidate next = std::move(next_population[individual_index]);
            tabu[individual_index].push_back(
                selected_moves[individual_index]);
            if (tabu[individual_index].size() > options_.tabu_tenure) {
                tabu[individual_index].erase(tabu[individual_index].begin());
            }
            if (better(next, current)) {
                current = std::move(next);
                if (better(current, global_best)) {
                    global_best = current;
                }
            }
        }
    }
    return global_best.solution;
}

core::Solution MetaHeuristicSolver::run_particle_swarm(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes) {
    Population particles;
    Population personal_best;
    particles = initial_population(
        virtual_network, physical_network, virtual_nodes);
    personal_best = particles;
    Candidate global_best = best_of(particles);

    for (std::size_t iteration = 0U;
         iteration < iteration_count(options_);
         ++iteration) {
        PositionBatch next_positions;
        next_positions.reserve(particles.size());
        for (std::size_t particle_index = 0U;
             particle_index < particles.size();
             ++particle_index) {
            Position next =
                particles[particle_index].position;
            for (std::size_t gene = 0U; gene < next.size(); ++gene) {
                const double draw = random_->random();
                if (draw < options_.pso_inertia) {
                    continue;
                }
                if (draw < options_.pso_inertia + options_.pso_cognition) {
                    next[gene] = personal_best[particle_index].position[gene];
                } else if (draw < options_.pso_inertia +
                                      options_.pso_cognition +
                                      options_.pso_social) {
                    next[gene] = global_best.position[gene];
                } else {
                    next = neighbor_position(next, physical_network.num_nodes());
                }
            }
            repair_position(next, physical_network.num_nodes());
            next_positions.push_back(std::move(next));
        }
        Population next_population = evaluate_positions(
            virtual_network,
            physical_network,
            virtual_nodes,
            next_positions);
        for (std::size_t particle_index = 0U;
             particle_index < next_population.size();
             ++particle_index) {
            particles[particle_index] = std::move(next_population[particle_index]);
            if (better(particles[particle_index], personal_best[particle_index])) {
                personal_best[particle_index] = particles[particle_index];
            }
            if (better(personal_best[particle_index], global_best)) {
                global_best = personal_best[particle_index];
            }
        }
    }
    return global_best.solution;
}

core::Solution MetaHeuristicSolver::run_ant_colony(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network,
    const std::vector<Vertex>& virtual_nodes) {
    const std::size_t count = population_size(options_);
    const std::size_t physical_count = physical_network.num_nodes();
    std::vector<std::vector<double>> pheromone(
        virtual_nodes.size(),
        std::vector<double>(physical_count, 1.0));
    PositionBatch initial_positions;
    initial_positions.push_back(greedy_position(
        virtual_nodes.size(), physical_count));
    Candidate global_best = evaluate_positions(
        virtual_network,
        physical_network,
        virtual_nodes,
        initial_positions).front();

    for (std::size_t iteration = 0U;
         iteration < iteration_count(options_);
         ++iteration) {
        PositionBatch ant_positions;
        ant_positions.reserve(count);
        for (std::size_t ant = 0U; ant < count; ++ant) {
            Position position(
                virtual_nodes.size(), core::SolutionNodeId{-1});
            std::vector<std::uint8_t> used(physical_count, 0U);
            for (std::size_t virtual_index = 0U;
                 virtual_index < virtual_nodes.size();
                 ++virtual_index) {
                std::vector<std::size_t> candidates;
                std::vector<double> weights;
                for (std::size_t physical_index = 0U;
                     physical_index < physical_count;
                     ++physical_index) {
                    if (used[physical_index] != 0U) {
                        continue;
                    }
                    candidates.push_back(physical_index);
                    weights.push_back(std::pow(
                        std::max(pheromone[virtual_index][physical_index],
                                 std::numeric_limits<double>::min()),
                        1.0));
                }
                if (candidates.empty()) {
                    break;
                }
                const double total = std::accumulate(
                    weights.begin(), weights.end(), 0.0);
                double target = random_->random() * total;
                std::size_t selected = candidates.back();
                for (std::size_t index = 0U; index < candidates.size(); ++index) {
                    target -= weights[index];
                    if (target <= 0.0) {
                        selected = candidates[index];
                        break;
                    }
                }
                position[virtual_index] = static_cast<core::SolutionNodeId>(
                    selected);
                used[selected] = 1U;
            }
            ant_positions.push_back(std::move(position));
        }
        Population ants = evaluate_positions(
            virtual_network,
            physical_network,
            virtual_nodes,
            ant_positions);
        const Candidate iteration_best = best_of(ants);
        if (better(iteration_best, global_best)) {
            global_best = iteration_best;
        }
        for (auto& row : pheromone) {
            for (double& value : row) {
                value *= options_.aco_evaporation;
            }
        }
        if (std::isfinite(global_best.fitness)) {
            const double deposit = options_.aco_deposit /
                std::max(global_best.fitness, std::numeric_limits<double>::min());
            for (std::size_t virtual_index = 0U;
                 virtual_index < global_best.position.size();
                 ++virtual_index) {
                const auto physical = global_best.position[virtual_index];
                if (physical >= 0 &&
                    static_cast<std::uint64_t>(physical) < physical_count) {
                    pheromone[virtual_index][static_cast<std::size_t>(physical)] +=
                        deposit;
                }
            }
        }
    }
    return global_best.solution;
}

core::Solution MetaHeuristicSolver::search(
    const network::VirtualNetwork& virtual_network,
    const network::PhysicalNetwork& physical_network) {
    const std::vector<Vertex> virtual_nodes = ordered_nodes(
        virtual_network.graph());
    switch (algorithm_) {
    case MetaAlgorithm::genetic:
        return run_genetic(virtual_network, physical_network, virtual_nodes);
    case MetaAlgorithm::simulated_annealing:
        return run_simulated_annealing(
            virtual_network, physical_network, virtual_nodes);
    case MetaAlgorithm::tabu_search:
        return run_tabu_search(virtual_network, physical_network, virtual_nodes);
    case MetaAlgorithm::particle_swarm:
        return run_particle_swarm(
            virtual_network, physical_network, virtual_nodes);
    case MetaAlgorithm::ant_colony:
        return run_ant_colony(virtual_network, physical_network, virtual_nodes);
    }
    return core::Solution::from_v_net(virtual_network);
}

core::Solution MetaHeuristicSolver::solve(
    const SolverInstance& instance) {
    return search(instance.virtual_network, instance.physical_network);
}

MutableSolverResult MetaHeuristicSolver::solve_mutable(
    const MutableSolverInstance& instance) {
    core::Solution solution = search(
        instance.virtual_network, instance.physical_network);
    if (!solution.result) {
        return MutableSolverResult{
            std::move(solution), SolverMutationState::detached};
    }
    if (!instance.mutation.deploy(solution)) {
        solution.result = false;
        solution.place_result = false;
        solution.route_result = false;
        return MutableSolverResult{
            std::move(solution), SolverMutationState::detached};
    }
    return MutableSolverResult{
        std::move(solution), SolverMutationState::committed};
}

MetaHeuristicSolverIds register_meta_heuristic_solvers(
    SolverRegistry& registry,
    PyRandom& random,
    MetaHeuristicOptions options) {
    const auto register_one = [&registry, &random, options](
        std::string name,
        MetaAlgorithm algorithm) {
        return registry.register_solver(
            std::move(name),
            SolverCategory::meta_heuristic,
            [algorithm, &random, options](
                SolverDependencies dependencies,
                SolverConfig config) {
                return std::make_unique<MetaHeuristicSolver>(
                    dependencies,
                    std::move(config),
                    algorithm,
                    random,
                    options);
            });
    };

    MetaHeuristicSolverIds ids{};
    ids.ga_meta = register_one("ga_meta", MetaAlgorithm::genetic);
    ids.sa_meta = register_one(
        "sa_meta", MetaAlgorithm::simulated_annealing);
    ids.ts_meta = register_one("ts_meta", MetaAlgorithm::tabu_search);
    ids.pso_meta = register_one(
        "pso_meta", MetaAlgorithm::particle_swarm);
    ids.aco_meta = register_one("aco_meta", MetaAlgorithm::ant_colony);
    return ids;
}

} // namespace virne::solver::meta
