#include "solver_registry.h"

#include "../../network/physical_network.h"
#include "../../network/virtual_network.h"
#include "../../core/controller/controller.h"
#include "../../core/counter.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace virne::solver::learning {
namespace rl = rl_core;
namespace p = rl_policy;
namespace {
using Profile = LearningEnvironmentProfile;
using Algorithm = rl::UpdateAlgorithm;
const std::array<LearningSolverDescriptor, 34> descriptors{{
    {"pg_mlp", p::PolicyKind::mlp, Algorithm::pg, Profile::placement_vanilla, true},
    {"pg_mlp2", p::PolicyKind::mlp, Algorithm::pg, Profile::placement_vanilla_more_metrics},
    {"pg_mlp+", p::PolicyKind::mlp, Algorithm::pg, Profile::joint_node},
    {"ppo_mlp+", p::PolicyKind::mlp, Algorithm::ppo, Profile::joint_node},
    {"a2c_mlp+", p::PolicyKind::mlp, Algorithm::a2c, Profile::joint_node},
    {"pg_cnn", p::PolicyKind::cnn, Algorithm::pg, Profile::placement_vanilla, true},
    {"pg_cnn2", p::PolicyKind::cnn, Algorithm::pg, Profile::placement_vanilla_more_metrics},
    {"pg_cnn+", p::PolicyKind::cnn, Algorithm::pg, Profile::joint_node},
    {"a2c_cnn+", p::PolicyKind::cnn, Algorithm::a2c, Profile::joint_node},
    {"ppo_cnn+", p::PolicyKind::cnn, Algorithm::ppo, Profile::joint_node},
    // Despite its name, the source PpoAttSolver explicitly initializes PGSolver.
    {"ppo_att", p::PolicyKind::attention, Algorithm::pg, Profile::placement_adaptive, true},
    {"pg_att+", p::PolicyKind::attention, Algorithm::pg, Profile::placement_node},
    {"a2c_att+", p::PolicyKind::attention, Algorithm::a2c, Profile::placement_node},
    {"ppo_att+", p::PolicyKind::attention, Algorithm::ppo, Profile::placement_node},
    {"pg_gcn_mlp+", p::PolicyKind::gcn_mlp, Algorithm::pg, Profile::joint_node},
    {"a2c_gcn_mlp+", p::PolicyKind::gcn_mlp, Algorithm::a2c, Profile::joint_node},
    {"ppo_gcn_mlp+", p::PolicyKind::gcn_mlp, Algorithm::ppo, Profile::joint_node},
    {"pg_gat_mlp+", p::PolicyKind::gat_mlp, Algorithm::pg, Profile::joint_node},
    {"a2c_gat_mlp+", p::PolicyKind::gat_mlp, Algorithm::a2c, Profile::joint_node},
    {"ppo_gat_mlp+", p::PolicyKind::gat_mlp, Algorithm::ppo, Profile::joint_node},
    {"pg_deep_edge_gat_mlp+", p::PolicyKind::deep_edge_gat_mlp, Algorithm::pg, Profile::joint_node},
    {"a2c_deep_edge_gat_mlp+", p::PolicyKind::deep_edge_gat_mlp, Algorithm::a2c, Profile::joint_node},
    {"ppo_deep_edge_gat_mlp+", p::PolicyKind::deep_edge_gat_mlp, Algorithm::ppo, Profile::joint_node},
    {"a2c_dual_gcn", p::PolicyKind::dual_gcn, Algorithm::a2c, Profile::joint_dual_gradual},
    {"ppo_dual_gcn", p::PolicyKind::dual_gcn, Algorithm::ppo, Profile::joint_dual_gradual},
    {"pg_dual_gcn+", p::PolicyKind::dual_gcn, Algorithm::pg, Profile::joint_dual},
    {"a2c_dual_gcn+", p::PolicyKind::dual_gcn, Algorithm::a2c, Profile::joint_dual},
    {"ppo_dual_gcn+", p::PolicyKind::dual_gcn, Algorithm::ppo, Profile::joint_dual},
    {"pg_dual_gat+", p::PolicyKind::dual_gat, Algorithm::pg, Profile::joint_dual},
    {"a2c_dual_gat+", p::PolicyKind::dual_gat, Algorithm::a2c, Profile::joint_dual},
    {"ppo_dual_gat+", p::PolicyKind::dual_gat, Algorithm::ppo, Profile::joint_dual},
    {"pg_dual_deep_edge_gat+", p::PolicyKind::dual_deep_edge_gat, Algorithm::pg, Profile::joint_dual},
    {"a2c_dual_deep_edge_gat+", p::PolicyKind::dual_deep_edge_gat, Algorithm::a2c, Profile::joint_dual},
    {"ppo_dual_deep_edge_gat+", p::PolicyKind::dual_deep_edge_gat, Algorithm::ppo, Profile::joint_dual},
}};

// InstanceEnvironment currently derives these resource/latency selections from
// each request. Refuse custom service selections instead of silently ignoring
// the Controller/Counter supplied through the base Solver contract.
void validate_services(const Solver& solver, const network::VirtualNetwork& network) {
    namespace attr = network::attribute;
    const network::AttributeSelection resources{
        std::vector<attr::AttributeKind>{attr::AttributeKind::resource}, std::nullopt};
    const auto nodes = network.select_node_attributes(resources);
    const auto links = network.select_link_attributes(resources);
    const auto latency = network.select_link_attributes(network::AttributeSelection{
        std::vector<attr::AttributeKind>{attr::AttributeKind::latency}, std::nullopt});
    auto hard_links = links;
    hard_links.insert(hard_links.end(), latency.begin(), latency.end());
    const auto& selection = solver.controller().selection();
    if (selection.node_resources != nodes || selection.link_resources != links
        || selection.constraints.node_at_node != nodes || selection.constraints.link_at_link != links
        || selection.constraints.link_at_path != latency || !selection.constraints.graph.empty()
        || selection.hard_node_constraints != nodes || selection.hard_link_constraints != hard_links
        || selection.reusable != solver.config().reusable)
        throw std::invalid_argument("learning registry requires the instance environment resource/latency controller selection");
    const auto& counting = solver.counter().selection();
    if ((counting.node_resources && *counting.node_resources != nodes)
        || (counting.link_resources && *counting.link_resources != links))
        throw std::invalid_argument("learning registry requires all request resources in Counter selection");
}
}  // namespace

const std::array<LearningSolverDescriptor, 34>& registered_learning_solvers() noexcept { return descriptors; }
const LearningSolverDescriptor& learning_solver_descriptor(const std::string_view name) {
    const auto found = std::find_if(descriptors.begin(), descriptors.end(),
        [name](const auto& value) { return value.name == name; });
    if (found == descriptors.end()) throw SolverException(SolverErrorCode::unknown_solver,
        SolverOperation::resolve_solver, "unsupported learning solver: " + std::string(name));
    return *found;
}
LearningSolverOptions resolve_learning_solver_options(const std::string_view name, LearningSolverOptions options) {
    const auto& descriptor = learning_solver_descriptor(name);
    options.session.policy = descriptor.policy;
    options.session.training.algorithm = descriptor.algorithm;
    if (descriptor.force_successful_experience) options.session.include_failed = false;
    auto& environment = options.environment;
    environment.features = rl::FeatureConstructorKind::p_net_v_node;
    environment.mode = rl::InstanceStepMode::joint;
    switch (descriptor.environment) {
    case Profile::placement_vanilla:
    case Profile::placement_vanilla_more_metrics:
    case Profile::placement_adaptive:
        environment.mode = rl::InstanceStepMode::placement;
        environment.degree = true;
        environment.other_metrics = descriptor.environment == Profile::placement_vanilla_more_metrics;
        environment.aggregated_links = true;
        environment.node_status = true;
        environment.reward = descriptor.environment == Profile::placement_adaptive
            ? rl::InstanceEnvironmentOptions::Reward::adaptive : rl::InstanceEnvironmentOptions::Reward::vanilla;
        break;
    case Profile::placement_node:
        environment.mode = rl::InstanceStepMode::placement;
        break;
    case Profile::joint_node:
        break;
    case Profile::joint_dual_gradual:
        environment.degree = false;
        environment.other_metrics = false;
        environment.aggregated_links = true;
        environment.reward = rl::InstanceEnvironmentOptions::Reward::gradual;
        [[fallthrough]];
    case Profile::joint_dual:
        environment.features = rl::FeatureConstructorKind::p_net_v_net;
        break;
    }
    return options;
}

LearningSolver::LearningSolver(SolverDependencies dependencies, SolverConfig config,
    const std::string_view name, LearningSolverOptions options)
    : Solver(std::move(dependencies), std::move(config)), descriptor_(&learning_solver_descriptor(name)),
      options_(resolve_learning_solver_options(name, std::move(options))) {
    const auto& base = this->config();
    auto& environment = options_.environment;
    if (base.allow_rejection || base.allow_revocable || environment.allow_rejection || environment.allow_revocable)
        throw std::invalid_argument("registered learning policies require a physical-node action space");
    if (base.node_ranking_method == rank::NodeRankMethod::random && environment.node_order.empty())
        throw std::invalid_argument("random virtual-node ranking requires an explicit prepared node_order");
    environment.reusable = base.reusable;
    environment.shortest_method = base.shortest_method;
    environment.k_shortest = base.k_shortest;
    if (base.seed) options_.session.seed = *base.seed;
    session_ = std::make_unique<rl::LearningSession>(options_.session);
}
LearningSolver::~LearningSolver() = default;
rl::InstanceEnvironment LearningSolver::make_environment(const SolverInstance& instance) const {
    validate_services(*this, instance.virtual_network);
    auto options = options_.environment;
    if (options.node_order.empty()) {
        const auto ranked = rank::NodeRanker({}, options_.node_ranking).prepare(instance.virtual_network)
            .rank(config().node_ranking_method, rank::NodeRankOptions{true, options.workers, std::nullopt});
        options.node_order.reserve(ranked.size());
        for (const auto& entry : ranked) options.node_order.push_back(static_cast<std::int64_t>(entry.node_id));
    }
    return rl::InstanceEnvironment(instance.physical_network.clone(), instance.virtual_network.clone(), std::move(options));
}
core::Solution LearningSolver::solve(const SolverInstance& instance) { return evaluate(instance).solution; }
LearningSolverResult LearningSolver::evaluate(const SolverInstance& instance, const bool sample) {
    auto environment = make_environment(instance);
    auto episode = session_->solve(environment, sample);
    return {environment.solution_snapshot(), std::move(episode)};
}
LearningSolverResult LearningSolver::learn(const SolverInstance& instance) {
    auto environment = make_environment(instance);
    auto episode = session_->learn(environment);
    return {environment.solution_snapshot(), std::move(episode)};
}
std::optional<double> LearningSolver::update() { return session_->update(); }
std::string_view LearningSolver::canonical_name() const noexcept { return descriptor_->name; }
const LearningSolverOptions& LearningSolver::learning_options() const noexcept { return options_; }
std::int64_t LearningSolver::pending_steps() const noexcept { return session_->pending_steps(); }
std::int64_t LearningSolver::update_time() const noexcept { return session_->update_time(); }
void LearningSolver::set_sampling_seed(const std::uint64_t seed) { session_->set_sampling_seed(seed); }
std::vector<double> LearningSolver::parameter_values() const { return session_->parameter_values(); }
std::vector<std::uint8_t> LearningSolver::checkpoint() const { return session_->checkpoint(); }
void LearningSolver::restore_checkpoint(const std::vector<std::uint8_t>& bytes) { session_->restore_checkpoint(bytes); }

std::vector<SolverId> register_learning_solvers(SolverRegistry& registry, LearningSolverOptions options) {
    std::vector<SolverId> result;
    result.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        const auto name = descriptor.name;
        result.push_back(registry.register_solver(std::string(name), SolverCategory::reinforcement_learning,
            [name, options](SolverDependencies dependencies, SolverConfig config) -> std::unique_ptr<Solver> {
                return std::make_unique<LearningSolver>(std::move(dependencies), std::move(config), name, options);
            }));
    }
    return result;
}

}  // namespace virne::solver::learning
