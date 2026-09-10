#include "instance_environment.h"
#include "feature_constructor.h"
#include "reward_calculator.h"
#include "../../../core/controller/controller.h"
#include "../../../core/counter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace virne::solver::learning::rl_core {
namespace ctl = core::controller;
namespace attr = network::attribute;
namespace {
ctl::ControllerSelection selection_for(const network::VirtualNetwork& network, const bool reusable) {
    ctl::ControllerSelection result;
    const network::AttributeSelection resources{std::vector<attr::AttributeKind>{attr::AttributeKind::resource}, std::nullopt};
    result.node_resources = network.select_node_attributes(resources);
    result.link_resources = network.select_link_attributes(resources);
    result.constraints.node_at_node = result.node_resources;
    result.constraints.link_at_link = result.link_resources;
    const network::AttributeSelection latency{std::vector<attr::AttributeKind>{attr::AttributeKind::latency}, std::nullopt};
    result.constraints.link_at_path = network.select_link_attributes(latency);
    result.hard_node_constraints = result.node_resources;
    result.hard_link_constraints = result.link_resources;
    result.hard_link_constraints.insert(result.hard_link_constraints.end(), result.constraints.link_at_path.begin(), result.constraints.link_at_path.end());
    result.reusable = reusable;
    return result;
}
}

class InstanceEnvironment::Impl {
public:
    using Prefix = std::vector<std::pair<std::int64_t, std::int64_t>>;
    using RevokedKey = std::pair<Prefix, std::int64_t>;
    network::PhysicalNetwork physical;
    network::VirtualNetwork virtual_network;
    network::PhysicalNetwork backup;
    InstanceEnvironmentOptions options;
    ctl::ControllerSelection selection;
    ctl::Controller controller;
    ctl::NodeMapper node_mapper;
    ctl::LinkMapper link_mapper;
    ctl::ConstraintChecker constraint_checker;
    core::Counter counter;
    core::Solution solution;
    std::optional<ctl::PreparedController> prepared;
    std::optional<ctl::PreparedNodeMapper> nodes;
    std::optional<ctl::PreparedLinkMapper> links;
    std::optional<ctl::PreparedConstraintChecker> checks;
    std::optional<core::PreparedCounter> counting;
    FeatureConstructorPreparedData feature_prepared;
    std::unique_ptr<FeatureConstructor> features;
    std::unique_ptr<BaseRewardCalculator> reward;
    std::vector<std::vector<std::int64_t>> initial_candidates;
    std::vector<attr::AttributeRegistryId> physical_link_resources, physical_node_resources;
    std::optional<attr::AttributeRegistryId> balance_capacity, balance_originator;
    std::map<RevokedKey, std::vector<std::int64_t>> revoked;
    std::int64_t pair_current = 0;
    bool terminal = false;
    EpisodeStatus status = EpisodeStatus::running;

    Impl(network::PhysicalNetwork&& p, network::VirtualNetwork&& v, InstanceEnvironmentOptions o)
        : physical(std::move(p)), virtual_network(std::move(v)), backup(physical.clone()), options(std::move(o)),
          selection(selection_for(virtual_network, options.reusable)), controller(selection),
          node_mapper(ctl::NodeMapperSelection{selection.constraints.node_at_node, selection.node_resources, selection.hard_node_constraints}),
          link_mapper(ctl::LinkMapperSelection{selection.constraints.link_at_link, selection.constraints.link_at_path,
              selection.link_resources, selection.hard_link_constraints, selection.reusable}),
          constraint_checker(selection.constraints), counter(), solution(core::Solution::from_v_net(virtual_network)) {
        if (!physical.num_nodes() || !virtual_network.num_nodes() || !options.workers || options.k_shortest < 1) throw std::invalid_argument("invalid RL environment dimensions/options");
        if (options.node_order.empty()) for (std::size_t node = 0; node < virtual_network.num_nodes(); ++node) options.node_order.push_back(static_cast<std::int64_t>(node));
        for (auto id : selection.link_resources) {
            const auto binding = physical.bind_link_attribute(virtual_network.link_attributes().at(id).spec().name);
            if (!binding) throw std::invalid_argument("missing physical link resource");
            physical_link_resources.push_back(binding->registry_id);
        }
        physical_node_resources = physical.select_node_attributes(network::AttributeSelection{
            std::vector<attr::AttributeKind>{attr::AttributeKind::resource}, std::nullopt});
        const auto extrema = physical.select_node_attributes(network::AttributeSelection{
            std::vector<attr::AttributeKind>{attr::AttributeKind::extrema}, std::nullopt});
        if (physical_node_resources.size() == 1 && !extrema.empty()) {
            const auto& spec = physical.node_attributes().at(extrema.front()).spec();
            const auto originator = spec.originator ? physical.bind_node_attribute(*spec.originator) : std::nullopt;
            if (!originator) throw std::invalid_argument("invalid capacity extrema originator");
            balance_originator = originator->registry_id;
            balance_capacity = extrema.front();
        }
        auto order = options.node_order;
        std::sort(order.begin(), order.end());
        if (order.size() != virtual_network.num_nodes()) throw std::invalid_argument("virtual node order size mismatch");
        for (std::size_t i = 0; i < order.size(); ++i) if (order[i] != static_cast<std::int64_t>(i)) throw std::invalid_argument("virtual node order must be a dense permutation");
        network::BaseNetworkBenchmarkSelection benchmark_selection;
        benchmark_selection.workers = options.workers;
        feature_prepared.p_net_attribute_benchmarks = network::get_attribute_benchmarks(backup, benchmark_selection);
        network::TopologicalMetricOptions metrics;
        metrics.degree = options.degree; metrics.closeness = options.other_metrics;
        metrics.eigenvector = options.other_metrics; metrics.betweenness = options.other_metrics;
        metrics.worker_count = options.workers;
        feature_prepared.p_net_topological_metrics = network::TopologicalMetricCalculator::calculate(backup.graph(), metrics);
        feature_prepared.v_net_topological_metrics = network::TopologicalMetricCalculator::calculate(virtual_network.graph(), metrics);
        bind();
        for (std::size_t node = 0; node < virtual_network.num_nodes(); ++node) initial_candidates.push_back(candidates(static_cast<std::int64_t>(node), false));
    }
    void bind() {
        prepared.emplace(controller.prepare(virtual_network, physical));
        nodes.emplace(node_mapper.prepare(virtual_network, physical));
        links.emplace(link_mapper.prepare(virtual_network, physical));
        checks.emplace(constraint_checker.prepare(virtual_network, physical));
        counting.emplace(counter.prepare(virtual_network));
        FeatureConstructorOptions feature_options;
        feature_options.kind = options.features; feature_options.use_node_status_flags = options.node_status;
        feature_options.use_aggregated_link_attrs = options.aggregated_links; feature_options.use_degree_metric = options.degree;
        feature_options.use_more_topological_metrics = options.other_metrics; feature_options.workers = options.workers;
        features = std::make_unique<FeatureConstructor>(physical, virtual_network, feature_options, feature_prepared);
        reward = RewardCalculatorRegistry::create({static_cast<RewardCalculatorKind>(options.reward), options.intermediate_reward});
    }
    std::int64_t base_actions() const {
        const auto p = static_cast<std::int64_t>(physical.num_nodes()), v = static_cast<std::int64_t>(virtual_network.num_nodes());
        if (options.mode == InstanceStepMode::node_pair || options.mode == InstanceStepMode::node_slots) {
            if (p > std::numeric_limits<std::int64_t>::max() / v) throw std::overflow_error("RL action space overflow");
            return p * v;
        }
        return p;
    }
    std::int64_t actions() const { return base_actions() + options.allow_rejection + options.allow_revocable; }
    std::int64_t current() const {
        if (options.mode == InstanceStepMode::node_pair) return pair_current;
        return solution.node_slots.size() == virtual_network.num_nodes() ? 0 : options.node_order.at(solution.node_slots.size());
    }
    Prefix prefix() const {
        Prefix result;
        for (const auto& slot : solution.node_slots.entries()) result.emplace_back(slot.key, slot.value);
        return result;
    }
    std::vector<std::int64_t> candidates(const std::int64_t vnode, const bool filter_selected) const {
        std::vector<std::uint8_t> used(physical.num_nodes(), 0);
        if (filter_selected && !options.reusable) for (const auto& slot : solution.node_slots.entries()) used.at(static_cast<std::size_t>(slot.value)) = 1;
        std::vector<std::vector<double>> v_resources, p_resources;
        if (options.check_link_candidates) {
            v_resources = network::get_aggregation_attrs_data(virtual_network, selection.link_resources, attr::LinkAggregation::maximum);
            p_resources = network::get_aggregation_attrs_data(physical, physical_link_resources, attr::LinkAggregation::maximum);
        }
        std::vector<std::int64_t> result;
        const auto range = physical.graph().nodes();
        for (auto iterator = range.first; iterator != range.second; ++iterator) {
            const auto node = *iterator;
            if (used.at(node) || !checks->check_node_level_constraints(static_cast<Vertex>(vnode), node).feasible) continue;
            if (options.check_link_candidates) {
                if (physical.graph().degree(node) < virtual_network.graph().degree(static_cast<Vertex>(vnode))) continue;
                bool enough = true;
                for (std::size_t resource = 0; resource < v_resources.size(); ++resource) if (v_resources[resource].at(static_cast<std::size_t>(vnode)) > p_resources[resource].at(node)) { enough = false; break; }
                if (!enough) continue;
            }
            result.push_back(static_cast<std::int64_t>(node));
        }
        return result;
    }
    std::vector<std::uint8_t> mask() const {
        std::vector<std::uint8_t> result(static_cast<std::size_t>(actions()), 0);
        const auto vnode_count = static_cast<std::int64_t>(virtual_network.num_nodes());
        if (options.mode == InstanceStepMode::node_pair || options.mode == InstanceStepMode::node_slots) {
            for (std::int64_t vnode = 0; vnode < vnode_count; ++vnode) {
                if (options.mode == InstanceStepMode::node_pair && solution.node_slots.contains(vnode)) continue;
                for (const auto pnode : initial_candidates.at(static_cast<std::size_t>(vnode))) {
                    bool selected = false;
                    if (options.mode == InstanceStepMode::node_pair && !options.reusable) for (const auto& slot : solution.node_slots.entries()) if (slot.value == pnode) { selected = true; break; }
                    if (!selected) result.at(static_cast<std::size_t>(pnode * vnode_count + vnode)) = 1;
                }
            }
        } else {
            for (auto node : candidates(current(), true)) result.at(static_cast<std::size_t>(node)) = 1;
            if (options.allow_revocable) {
                const auto banned = revoked.find({prefix(), current()});
                if (banned != revoked.end()) for (auto node : banned->second) result.at(static_cast<std::size_t>(node)) = 0;
            }
        }
        if (options.allow_rejection) result.at(static_cast<std::size_t>(base_actions())) = 1;
        if (options.allow_revocable && !solution.node_slots.empty()) result.at(static_cast<std::size_t>(base_actions() + options.allow_rejection)) = 1;
        return result;
    }
    InstanceObservation observation() const {
        return {features->construct(solution, static_cast<std::size_t>(current())), mask(), current(),
            static_cast<std::int64_t>(virtual_network.num_nodes()), actions()};
    }
    EpisodeInfo info() const {
        EpisodeInfo result;
        result.status = status; result.result = solution.result; result.place_result = solution.place_result;
        result.route_result = solution.route_result; result.early_rejection = solution.early_rejection;
        result.request_id = solution.v_net_id; result.num_interactions = solution.num_interactions; result.revoke_times = solution.revoke_times;
        result.cumulative_reward = solution.v_net_reward; result.revenue = solution.v_net_revenue; result.cost = solution.v_net_cost;
        result.revenue_cost_ratio = solution.v_net_r2c_ratio; result.hard_constraint_violation = solution.v_net_total_hard_constraint_violation;
        result.selected_actions = solution.selected_actions;
        for (const auto& slot : solution.node_slots.entries()) result.node_slots.push_back({slot.key, slot.value});
        for (const auto& path : solution.link_paths.entries()) {
            RouteAssignment route{{path.key.source, path.key.target}, {}};
            for (const auto& edge : path.value) route.physical_edges.push_back({edge.source, edge.target});
            result.link_paths.push_back(std::move(route));
        }
        return result;
    }
    InstanceStep result(const bool done, const EpisodeStatus next_status) {
        terminal = done; status = next_status;
        const double value = options.mode == InstanceStepMode::solution ? 0.0 : reward->compute(physical, virtual_network, solution);
        return {observation(), value, terminal, info()};
    }
    void require_active() const { if (terminal) throw std::logic_error("reset RL environment before stepping a completed episode"); }
    InstanceStep reject() {
        require_active(); solution.early_rejection = true;
        return result(true, EpisodeStatus::rejected);
    }
    InstanceStep revoke() {
        require_active();
        if (solution.node_slots.empty()) throw std::logic_error("cannot revoke an empty node mapping");
        solution.place_result = true; solution.route_result = true; ++solution.revoke_times;
        const auto last = solution.node_slots.entries().back();
        prepared->undo_place_and_route(static_cast<Vertex>(last.key), solution);
        counting->count_partial_solution(solution);
        revoked[{prefix(), last.key}].push_back(last.value);
        return result(false, EpisodeStatus::running);
    }
    InstanceStep step(std::int64_t action) {
        require_active();
        if (options.mode == InstanceStepMode::node_slots || options.mode == InstanceStepMode::solution) throw std::invalid_argument("this RL mode requires a complete assignment/solution");
        if (action < 0 || action >= actions()) throw std::out_of_range("RL action outside action space");
        const bool joint = options.mode != InstanceStepMode::placement;
        if (joint) {
            ++solution.num_interactions;
            const auto recorded_action = options.mode == InstanceStepMode::node_pair
                ? (action < base_actions() ? action / static_cast<std::int64_t>(virtual_network.num_nodes())
                   : static_cast<std::int64_t>(physical.num_nodes()) + action - base_actions()) : action;
            solution.selected_actions.push_back(recorded_action);
            if (solution.num_interactions > static_cast<std::int64_t>(10 * virtual_network.num_nodes())) return reject();
        }
        if (options.allow_rejection && action == base_actions()) return reject();
        if (options.allow_revocable && action == base_actions() + options.allow_rejection) return revoke();
        if (options.mode == InstanceStepMode::node_pair) {
            pair_current = action % static_cast<std::int64_t>(virtual_network.num_nodes());
            action /= static_cast<std::int64_t>(virtual_network.num_nodes());
            if (solution.node_slots.contains(pair_current)) {
                solution.place_result = false; counting->count_solution(solution);
                return result(true, EpisodeStatus::placement_failed);
            }
        }
        if (!options.reusable) for (const auto& slot : solution.node_slots.entries()) if (slot.value == action) {
            solution.place_result = false;
            if (joint) counting->count_solution(solution);
            return result(true, EpisodeStatus::placement_failed);
        }
        if (!joint) {
            const auto placed = nodes->place(static_cast<Vertex>(current()), static_cast<Vertex>(action), solution,
                ctl::NodePlacementOptions{options.allow_constraint_violation, true});
            if (!placed.placed) { solution.place_result = false; counting->count_solution(solution); return result(true, EpisodeStatus::placement_failed); }
            if (solution.node_slots.size() < virtual_network.num_nodes()) return result(false, EpisodeStatus::running);
            ctl::LinkMappingOptions routing;
            routing.shortest_method = options.shortest_method; routing.k = options.k_shortest;
            routing.topology_constraint_workers = options.workers; routing.candidate_workers = options.workers;
            routing.allow_constraint_violation = options.allow_constraint_violation;
            solution.route_result = links->link_mapping(solution, routing);
            solution.result = solution.route_result;
        } else {
            ctl::PlaceAndRouteOptions routing;
            routing.shortest_method = options.shortest_method; routing.k = options.k_shortest;
            routing.workers.topology_constraint_workers = options.workers; routing.workers.candidate_workers = options.workers;
            routing.allow_constraint_violation = options.allow_constraint_violation;
            const auto placed = prepared->place_and_route(static_cast<Vertex>(current()), static_cast<Vertex>(action), solution, routing);
            if (!placed.succeeded) {
                if (options.allow_revocable && !solution.node_slots.empty()) {
                    solution.selected_actions.push_back(static_cast<std::int64_t>(physical.num_nodes()) + options.allow_rejection);
                    return revoke();
                }
                counting->count_solution(solution);
                return result(true, placed.failure_phase == ctl::ControllerFailurePhase::place ? EpisodeStatus::placement_failed : EpisodeStatus::routing_failed);
            }
            if (solution.node_slots.size() < virtual_network.num_nodes()) { counting->count_partial_solution(solution); return result(false, EpisodeStatus::running); }
            solution.result = true;
        }
        counting->count_solution(solution);
        return result(true, solution.result ? EpisodeStatus::success : EpisodeStatus::routing_failed);
    }
};

InstanceEnvironment::InstanceEnvironment(network::PhysicalNetwork&& p, network::VirtualNetwork&& v, InstanceEnvironmentOptions options)
    : impl_(std::make_unique<Impl>(std::move(p), std::move(v), std::move(options))) {}
InstanceEnvironment::InstanceEnvironment(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
InstanceEnvironment::~InstanceEnvironment() = default;
InstanceEnvironment::InstanceEnvironment(InstanceEnvironment&&) noexcept = default;
InstanceEnvironment& InstanceEnvironment::operator=(InstanceEnvironment&&) noexcept = default;
InstanceObservation InstanceEnvironment::reset() {
    impl_->features.reset(); impl_->prepared.reset(); impl_->nodes.reset(); impl_->links.reset(); impl_->checks.reset(); impl_->counting.reset();
    impl_->physical = impl_->backup.clone();
    impl_->solution = core::Solution::from_v_net(impl_->virtual_network);
    impl_->revoked.clear(); impl_->pair_current = 0; impl_->terminal = false; impl_->status = EpisodeStatus::running;
    impl_->bind(); return impl_->observation();
}
InstanceObservation InstanceEnvironment::observation() const { return impl_->observation(); }
std::vector<std::uint8_t> InstanceEnvironment::action_masks() const { return impl_->mask(); }
InstanceStep InstanceEnvironment::step(const std::int64_t action) { return impl_->step(action); }
InstanceStep InstanceEnvironment::reject() { return impl_->reject(); }
InstanceStep InstanceEnvironment::revoke() { return impl_->revoke(); }
EpisodeInfo InstanceEnvironment::info() const { return impl_->info(); }
std::int64_t InstanceEnvironment::num_actions() const { return impl_->actions(); }
std::int64_t InstanceEnvironment::current_virtual_node() const { return impl_->current(); }
bool InstanceEnvironment::done() const noexcept { return impl_->terminal; }
double InstanceEnvironment::placement_progress() const {
    return static_cast<double>(impl_->solution.node_slots.size()) / static_cast<double>(std::max<std::size_t>(1, impl_->virtual_network.num_nodes() - 1));
}
InstanceStep InstanceEnvironment::step_node_slots(const std::vector<NodeAssignment>& assignments) {
    impl_->require_active();
    core::NodeSlots slots;
    for (const auto& entry : assignments) slots.insert_or_assign(entry.virtual_node, entry.physical_node);
    if (slots.size() != impl_->virtual_network.num_nodes()) return impl_->result(true, EpisodeStatus::incomplete);
    ctl::DeployWithNodeSlotsOptions routing;
    routing.shortest_method = impl_->options.shortest_method; routing.k = impl_->options.k_shortest;
    routing.workers.topology_constraint_workers = impl_->options.workers;
    routing.workers.candidate_workers = impl_->options.workers;
    if (impl_->options.allow_constraint_violation) {
        bool placed = true;
        for (const auto& slot : slots.entries()) {
            if (!impl_->nodes->place(static_cast<Vertex>(slot.key), static_cast<Vertex>(slot.value), impl_->solution,
                    ctl::NodePlacementOptions{true, true}).placed) { placed = false; break; }
        }
        impl_->solution.place_result = placed;
        if (placed) {
            ctl::LinkMappingOptions link_options;
            link_options.shortest_method = routing.shortest_method; link_options.k = routing.k;
            link_options.allow_constraint_violation = true;
            link_options.topology_constraint_workers = impl_->options.workers;
            link_options.candidate_workers = impl_->options.workers;
            impl_->solution.route_result = impl_->links->link_mapping(impl_->solution, link_options);
        }
        impl_->solution.result = impl_->solution.place_result && impl_->solution.route_result;
    } else {
        impl_->prepared->deploy_with_node_slots(slots, impl_->solution, routing);
    }
    impl_->counting->count_solution(impl_->solution);
    return impl_->result(true, impl_->solution.result ? EpisodeStatus::success : EpisodeStatus::routing_failed);
}
InstanceStep InstanceEnvironment::step_solution(const core::Solution& solution) {
    impl_->require_active(); impl_->solution = solution;
    return impl_->result(true, solution.result ? EpisodeStatus::success : EpisodeStatus::incomplete);
}
core::Solution InstanceEnvironment::solution_snapshot() const { return impl_->solution; }
const network::PhysicalNetwork& InstanceEnvironment::physical_network() const noexcept { return impl_->physical; }
double InstanceEnvironment::node_load_balance(const std::int64_t node) const {
    if (node < 0 || node >= static_cast<std::int64_t>(impl_->physical.num_nodes())) throw std::out_of_range("invalid physical node");
    const auto values = network::get_node_attrs_data(impl_->physical, impl_->physical_node_resources);
    std::vector<double> amounts;
    for (const auto& column : values) amounts.push_back(std::visit([](const auto& value) -> double {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(value);
        else throw std::invalid_argument("node resource must be numeric");
    }, column.at(static_cast<std::size_t>(node))));
    if (amounts.empty()) throw std::invalid_argument("load balance requires node resources");
    if (amounts.size() == 1) {
        if (!impl_->balance_capacity || !impl_->balance_originator) throw std::invalid_argument("single-resource load balance needs capacity extrema");
        const auto original = network::get_node_attrs_data(impl_->physical, {*impl_->balance_capacity, *impl_->balance_originator});
        const double capacity = std::visit([](const auto& value) -> double {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(value);
            else throw std::invalid_argument("node capacity must be numeric");
        }, original.front().at(static_cast<std::size_t>(node)));
        const double available = std::visit([](const auto& value) -> double {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(value);
            else throw std::invalid_argument("node resource must be numeric");
        }, original.at(1).at(static_cast<std::size_t>(node)));
        return available / capacity;
    }
    double mean = 0; for (double amount : amounts) mean += amount; mean /= static_cast<double>(amounts.size());
    double variance = 0; for (double amount : amounts) variance += (amount - mean) * (amount - mean);
    return std::sqrt(variance / static_cast<double>(amounts.size()));
}
std::unique_ptr<InstanceEnvironment> InstanceEnvironment::clone() const {
    auto copied = std::make_unique<Impl>(impl_->physical.clone(), impl_->virtual_network.clone(), impl_->options);
    copied->backup = impl_->backup.clone(); copied->solution = impl_->solution;
    copied->feature_prepared = impl_->feature_prepared; copied->revoked = impl_->revoked;
    copied->initial_candidates = impl_->initial_candidates; copied->pair_current = impl_->pair_current;
    copied->terminal = impl_->terminal; copied->status = impl_->status; copied->bind();
    return std::unique_ptr<InstanceEnvironment>(new InstanceEnvironment(std::move(copied)));
}

}  // namespace virne::solver::learning::rl_core
