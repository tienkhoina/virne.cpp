#include "numeric_bytes.h"
#include "feature_constructor_fixture.h"
#include "../virne/solver/learning/rl_core/instance_environment.h"
#include <chrono>
#include <stdexcept>

namespace rl = virne::solver::learning::rl_core;
namespace fixture = feature_constructor_fixture;
using virne::porting::emit_numeric_bytes;
using virne::porting::emit_numeric_scalar;

namespace {
void network(const std::string& name, const rl::NetworkFeatures& features) {
    emit_numeric_bytes(name + ".x", "Float", features.x.values, {features.x.rows, features.x.columns});
    emit_numeric_bytes(name + ".edge_index", "Long", features.edge_index.values, {features.edge_index.rows, features.edge_index.columns});
    emit_numeric_bytes(name + ".edge_attr", "Float", features.edge_attr.values, {features.edge_attr.rows, features.edge_attr.columns});
}
void snapshot(const rl::InstanceEnvironment& env, const double reward = 0.0) {
    const auto obs = env.observation();
    const auto info = env.info();
    network("p_net", obs.features.p_net);
    if (obs.features.v_node_x) emit_numeric_bytes("v_node_x", "Float", obs.features.v_node_x->values, {obs.features.v_node_x->values.size()});
    if (obs.features.v_net) network("v_net", *obs.features.v_net);
    emit_numeric_bytes("mask", "Bool", obs.action_mask, {obs.action_mask.size()});
    emit_numeric_scalar("current", "Long", obs.curr_v_node_id);
    emit_numeric_scalar("v_net_size", "Long", obs.v_net_size);
    emit_numeric_scalar("num_actions", "Long", obs.num_actions);
    emit_numeric_scalar("reward", "Double", reward);
    emit_numeric_scalar("done", "Bool", static_cast<std::uint8_t>(env.done()));
    emit_numeric_scalar("result", "Bool", static_cast<std::uint8_t>(info.result));
    emit_numeric_scalar("place_result", "Bool", static_cast<std::uint8_t>(info.place_result));
    emit_numeric_scalar("route_result", "Bool", static_cast<std::uint8_t>(info.route_result));
    emit_numeric_scalar("rejection", "Bool", static_cast<std::uint8_t>(info.early_rejection));
    emit_numeric_scalar("interactions", "Long", info.num_interactions);
    emit_numeric_scalar("revokes", "Long", info.revoke_times);
    emit_numeric_scalar("total_reward", "Double", info.cumulative_reward);
    emit_numeric_scalar("revenue", "Double", info.revenue);
    emit_numeric_scalar("cost", "Double", info.cost);
    emit_numeric_scalar("ratio", "Double", info.revenue_cost_ratio);
    emit_numeric_scalar("violation", "Double", info.hard_constraint_violation);
    std::vector<std::int64_t> slots;
    for (const auto& slot : info.node_slots) { slots.push_back(slot.virtual_node); slots.push_back(slot.physical_node); }
    emit_numeric_bytes("slots", "Long", slots, {info.node_slots.size(), 2});
    emit_numeric_bytes("selected", "Long", info.selected_actions, {info.selected_actions.size()});
    emit_numeric_scalar("path_count", "Long", static_cast<std::int64_t>(info.link_paths.size()));
    for (std::size_t i = 0; i < info.link_paths.size(); ++i) {
        const auto& route = info.link_paths[i];
        const auto prefix = "path_" + std::to_string(i);
        emit_numeric_bytes(prefix + ".virtual", "Long", std::vector<std::int64_t>{route.virtual_edge.source, route.virtual_edge.target}, {2});
        std::vector<std::int64_t> edges;
        for (const auto& edge : route.physical_edges) { edges.push_back(edge.source); edges.push_back(edge.target); }
        emit_numeric_bytes(prefix + ".physical", "Long", edges, {route.physical_edges.size(), 2});
    }
}
rl::InstanceEnvironment make_env(rl::InstanceEnvironmentOptions options = {}, const int failure = 0) {
    auto p = fixture::physical_network(); auto v = fixture::virtual_network();
    if (failure == 1) v.set_node_attrs_data({fixture::dense_node(fixture::node_id(v, "cpu"), {200, 35, 15, 25})});
    if (failure == 2) p.set_link_attrs_data({fixture::sparse_link(fixture::link_id(p, "bw"),
        {{0,1},{1,2},{2,3},{3,4},{4,5},{5,0},{1,4},{2,5}}, {1,1,1,1,1,1,1,1})});
    return rl::InstanceEnvironment(std::move(p), std::move(v), options);
}
void trajectory(rl::InstanceEnvironmentOptions options, const std::vector<std::int64_t>& actions, const int failure = 0) {
    auto env = make_env(options, failure);
    snapshot(env);
    for (auto action : actions) {
        if (env.done()) break;
        const auto result = env.step(action);
        snapshot(env, result.reward);
    }
    env.reset(); snapshot(env);
}
void exact(const std::string& selected) {
    const auto run = [&](const std::string& name, const auto& callback) {
        if (!selected.empty() && selected != name) return;
        std::cout << "CASE " << name << '\n'; callback();
    };
    run("joint", [] { trajectory({}, {0,1,2,3}); });
    run("placement", [] { rl::InstanceEnvironmentOptions o; o.mode = rl::InstanceStepMode::placement; trajectory(o, {0,1,2,3}); });
    run("duplicate", [] { trajectory({}, {0,0}); });
    run("node_failure", [] { trajectory({}, {0}, 1); });
    run("link_failure", [] { trajectory({}, {0,1}, 2); });
    run("reject", [] { rl::InstanceEnvironmentOptions o; o.allow_rejection = true; trajectory(o, {0,6}); });
    run("revoke", [] { rl::InstanceEnvironmentOptions o; o.allow_revocable = true; trajectory(o, {0,1,6,4,2,3}); });
    run("node_pair", [] { rl::InstanceEnvironmentOptions o; o.mode = rl::InstanceStepMode::node_pair; o.features = rl::FeatureConstructorKind::p_net_v_net; trajectory(o, {0,5,10,15}); });
    run("slots", [] {
        rl::InstanceEnvironmentOptions o; o.mode = rl::InstanceStepMode::node_slots;
        auto env = make_env(o); snapshot(env);
        auto result = env.step_node_slots({{0,0},{1,1},{2,2},{3,3}}); snapshot(env, result.reward);
    });
    run("incomplete_slots", [] {
        rl::InstanceEnvironmentOptions o; o.mode = rl::InstanceStepMode::node_slots;
        auto env = make_env(o); snapshot(env);
        auto result = env.step_node_slots({{0,0},{1,1}}); snapshot(env, result.reward);
    });
    run("clone", [] {
        auto env = make_env(); env.step(0);
        auto branch = env.clone();
        const auto branch_step = branch->step(4); snapshot(*branch, branch_step.reward);
        snapshot(env);
        branch->reset(); snapshot(*branch);
        emit_numeric_scalar("balance", "Double", env.node_load_balance(0));
    });
    for (int reward = 0; reward < 4; ++reward) run("reward_" + std::to_string(reward), [=] {
        rl::InstanceEnvironmentOptions o; o.reward = static_cast<rl::InstanceEnvironmentOptions::Reward>(reward);
        trajectory(o, {0,1,2,3});
    });
}
}
int main(int argc, char** argv) {
    try {
        std::string selected;
        if (argc == 3 && std::string(argv[1]) == "--case") selected = argv[2];
        else if (argc != 1) throw std::invalid_argument("invalid environment harness argument");
        exact(selected);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
