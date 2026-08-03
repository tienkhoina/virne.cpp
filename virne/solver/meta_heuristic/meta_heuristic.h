#pragma once

#include "../base_solver.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class PyRandom;

namespace virne::core {
class PreparedCounter;
}

namespace virne::solver::meta {

enum class MetaAlgorithm : std::uint8_t {
    genetic,
    simulated_annealing,
    tabu_search,
    particle_swarm,
    ant_colony,
};

// The Python leaves use these fixed defaults.  They are kept in one typed
// options object so the registry and the benchmark use exactly one policy.
struct MetaHeuristicOptions {
    std::size_t population_size = 8U;
    std::size_t max_iterations = 12U;
    std::size_t max_attempts = 2U;
    // Evaluation is the only parallel section owned by this module. Random
    // transitions stay on the coordinator so the shared Python-compatible
    // stream remains deterministic and race-free.
    std::size_t evaluation_workers = 1U;
    std::size_t candidate_workers = 1U;
    std::size_t topology_constraint_workers = 1U;
    double crossover_probability = 0.8;
    double initial_temperature = 2.0;
    double attenuation_factor = 0.95;
    std::size_t tabu_tenure = 5U;
    double pso_inertia = 0.1;
    double pso_cognition = 0.2;
    double pso_social = 0.7;
    double aco_evaporation = 0.5;
    double aco_deposit = 1.0;
};

struct MetaHeuristicSolverIds {
    SolverId ga_meta;
    SolverId sa_meta;
    SolverId ts_meta;
    SolverId pso_meta;
    SolverId aco_meta;
};

class MetaHeuristicSolver final : public Solver {
public:
    MetaHeuristicSolver(
        SolverDependencies dependencies,
        SolverConfig config,
        MetaAlgorithm algorithm,
        PyRandom& random,
        MetaHeuristicOptions options = {});

    core::Solution solve(const SolverInstance& instance) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance& instance) override;

private:
    struct Candidate {
        std::vector<core::SolutionNodeId> position;
        core::Solution solution;
        double fitness = 0.0;

        Candidate(
            std::vector<core::SolutionNodeId> position_value,
            core::Solution solution_value,
            double fitness_value)
            : position(std::move(position_value)),
              solution(std::move(solution_value)),
              fitness(fitness_value) {}
    };

    using Population = std::vector<Candidate>;
    using Position = std::vector<core::SolutionNodeId>;
    using PositionBatch = std::vector<Position>;

    Candidate evaluate(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<core::SolutionNodeId>& position,
        const core::PreparedCounter& prepared_counter) const;

    Population evaluate_positions(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes,
        const PositionBatch& positions) const;

    Population initial_population(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes);

    core::NodeSlots make_node_slots(
        const std::vector<Vertex>& virtual_nodes,
        const std::vector<core::SolutionNodeId>& position) const;

    std::vector<core::SolutionNodeId> random_position(
        std::size_t virtual_node_count,
        std::size_t physical_node_count);
    std::vector<core::SolutionNodeId> greedy_position(
        std::size_t virtual_node_count,
        std::size_t physical_node_count) const;
    void repair_position(
        std::vector<core::SolutionNodeId>& position,
        std::size_t physical_node_count) const;
    std::vector<core::SolutionNodeId> neighbor_position(
        const std::vector<core::SolutionNodeId>& position,
        std::size_t physical_node_count);
    bool better(const Candidate& left, const Candidate& right) const;
    Candidate best_of(const Population& population) const;
    core::Solution run_genetic(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes);
    core::Solution run_simulated_annealing(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes);
    core::Solution run_tabu_search(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes);
    core::Solution run_particle_swarm(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes);
    core::Solution run_ant_colony(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network,
        const std::vector<Vertex>& virtual_nodes);
    core::Solution search(
        const network::VirtualNetwork& virtual_network,
        const network::PhysicalNetwork& physical_network);

    MetaAlgorithm algorithm_;
    PyRandom* random_;
    MetaHeuristicOptions options_;
};

MetaHeuristicSolverIds register_meta_heuristic_solvers(
    SolverRegistry& registry,
    PyRandom& random,
    MetaHeuristicOptions options = {});

} // namespace virne::solver::meta
