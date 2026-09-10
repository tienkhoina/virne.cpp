#include "numeric_bytes.h"
#include "feature_constructor_fixture.h"
#include "../virne/solver/learning/rl_core/learning_session.h"
#include <chrono>
#include <stdexcept>
#include <string>

namespace rl = virne::solver::learning::rl_core;
namespace p = virne::solver::learning::rl_policy;
namespace fixture = feature_constructor_fixture;
using virne::porting::emit_numeric_bytes;
using virne::porting::emit_numeric_scalar;
namespace {
const std::vector<std::string> cases{"pg", "a2c", "ppo", "arppo", "source_ddpg", "cnn", "gcn", "gat", "attention", "dual_gcn", "failed", "dqn", "double_dqn", "double_dqn_gcn"};
rl::LearningSessionOptions options(const std::string& name) {
    rl::LearningSessionOptions o;
    o.seed = 401; o.embedding_dim = 8; o.num_layers = 2; o.target_steps = 8; o.intraop_threads = 1;
    o.learning_rates = {0.003, 0.007, 0.002, 0.004, 0.005}; o.weight_decay = 0.01;
    o.training.batch_size = 4; o.training.repeat_times = 1;
    o.training.normalize_returns = true; o.training.normalize_advantage = true;
    o.training.max_grad_norm = 0.25; o.training.mask_coefficient = 0.3;
    if (name == "pg") o.training.algorithm = rl::UpdateAlgorithm::pg;
    if (name == "a2c") o.training.algorithm = rl::UpdateAlgorithm::a2c;
    if (name == "arppo") o.training.algorithm = rl::UpdateAlgorithm::arppo;
    if (name == "source_ddpg") o.training.algorithm = rl::UpdateAlgorithm::source_ddpg;
    if (name == "cnn") o.policy = p::PolicyKind::cnn;
    if (name == "gcn") o.policy = p::PolicyKind::gcn_mlp;
    if (name == "gat") o.policy = p::PolicyKind::gat_mlp;
    if (name == "attention") o.policy = p::PolicyKind::attention;
    if (name == "dual_gcn") o.policy = p::PolicyKind::dual_gcn;
    if (name == "dqn" || name == "double_dqn" || name == "double_dqn_gcn") {
        o.training.algorithm = name == "dqn" ? rl::UpdateAlgorithm::dqn : rl::UpdateAlgorithm::double_dqn;
        o.target_steps = 4; o.include_failed = true;
        o.dqn = {2,.6,.1,7.,713,true,true};
        if (name == "double_dqn_gcn") o.policy = p::PolicyKind::gcn_mlp;
    }
    return o;
}
rl::InstanceEnvironment environment(const std::string& name) {
    auto physical = fixture::physical_network(); auto virtual_net = fixture::virtual_network();
    if (name == "failed") virtual_net.set_node_attrs_data({fixture::dense_node(fixture::node_id(virtual_net, "cpu"), {200, 35, 15, 25})});
    rl::InstanceEnvironmentOptions o;
    if (name == "dual_gcn") o.features = rl::FeatureConstructorKind::p_net_v_net;
    return rl::InstanceEnvironment(std::move(physical), std::move(virtual_net), o);
}
void snapshot(const rl::EpisodeLearningResult& result) {
    emit_numeric_bytes("actions", "Long", result.actions, {result.actions.size()});
    for (const auto& entry : std::vector<std::pair<std::string, const std::vector<double>*>>{
        {"rewards", &result.rewards}, {"values", &result.values}, {"logprobs", &result.log_probabilities}, {"returns", &result.returns}})
        emit_numeric_bytes(entry.first, "Double", *entry.second, {entry.second->size()});
    emit_numeric_bytes("dones", "Bool", result.dones, {result.dones.size()});
    emit_numeric_scalar("last_value", "Double", result.last_value);
    emit_numeric_scalar("success", "Bool", static_cast<std::uint8_t>(result.solution.result));
    emit_numeric_scalar("reward", "Double", result.solution.cumulative_reward);
    emit_numeric_scalar("ratio", "Double", result.solution.revenue_cost_ratio);
    emit_numeric_scalar("pending", "Long", result.pending_steps);
    emit_numeric_scalar("updates", "Long", result.update_time);
    if (result.update_loss) emit_numeric_scalar("loss", "Double", *result.update_loss);
    else std::cout << "loss|undefined\n";
    if (result.exploration_epsilon) {
        emit_numeric_scalar("epsilon", "Double", *result.exploration_epsilon);
        emit_numeric_scalar("exploration_steps", "Long", result.exploration_steps);
    }
}
void parameters(const rl::LearningSession& session) {
    const auto values = session.parameter_values();
    emit_numeric_bytes("parameters", "Double", values, {values.size()});
    const auto target = session.target_parameter_values();
    if (!target.empty()) emit_numeric_bytes("target_parameters", "Double", target, {target.size()});
}
void run(const std::string& name) {
    rl::LearningSession session(options(name)); auto env = environment(name);
    for (int episode = 0; episode < 3; ++episode) {
        if (episode) session.set_sampling_seed(600 + episode);
        std::cout << "EPISODE " << episode << '\n';
        snapshot(session.learn(env)); parameters(session);
    }
    const auto loss = session.update();
    if (loss) emit_numeric_scalar("flush_loss", "Double", *loss);
    else std::cout << "flush_loss|undefined\n";
    parameters(session);
    for (const bool sample : {false, true}) {
        session.set_sampling_seed(710);
        std::cout << "SOLVE " << static_cast<int>(sample) << '\n';
        snapshot(session.solve(env, sample)); parameters(session);
    }
}
void benchmark(const bool value_learning = false) {
    std::vector<std::int64_t> samples;
    std::vector<double> expected_parameters;
    std::vector<double> expected_target;
    std::vector<std::int64_t> last_actions;
    std::optional<double> final_loss;
    for (int repetition = -3; repetition < 5; ++repetition) {
        const std::string name = value_learning ? "double_dqn" : "ppo";
        rl::LearningSession session(options(name)); auto env = environment(name);
        session.solve(env);
        rl::EpisodeLearningResult result;
        const auto start = std::chrono::steady_clock::now();
        for (int episode = 0; episode < 3; ++episode) {
            session.set_sampling_seed(800 + episode);
            result = session.learn(env);
        }
        const auto loss = session.update();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        const auto current = session.parameter_values();
        const auto target = session.target_parameter_values();
        if (repetition == -3) { expected_parameters = current; expected_target = target; final_loss = loss; last_actions = result.actions; }
        else if (current != expected_parameters || target != expected_target || loss != final_loss || result.actions != last_actions)
            throw std::runtime_error("learning benchmark output changed between repetitions");
        if (repetition >= 0) samples.push_back(elapsed);
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
    std::cout << "]}\n";
    emit_numeric_bytes("actions", "Long", last_actions, {last_actions.size()});
    if (final_loss) emit_numeric_scalar("loss", "Double", *final_loss);
    else std::cout << "loss|undefined\n";
    emit_numeric_bytes("parameters", "Double", expected_parameters, {expected_parameters.size()});
    if (!expected_target.empty()) emit_numeric_bytes("target_parameters", "Double", expected_target, {expected_target.size()});
}
void checkpoint_case(const std::string& name) {
    const std::string kind = name == "resume_gcn" ? "gcn" : (name == "resume_dual" ? "dual_gcn"
        : (name == "resume_dqn" ? "dqn" : (name == "resume_double_dqn" ? "double_dqn" : "ppo")));
    auto o = options(kind); auto env = environment(kind);
    rl::LearningSession original(o);
    for (int episode = 0; episode < 3; ++episode) original.learn(env);
    const auto saved = original.checkpoint();
    const auto expected = original.learn(env);
    const auto expected_parameters = original.parameter_values();
    const auto expected_target = original.target_parameter_values();
    const auto after_update = original.checkpoint();
    rl::LearningSession restored(o);
    restored.restore_checkpoint(saved);
    const auto actual = restored.learn(env);
    const auto same = [](const rl::EpisodeLearningResult& a, const rl::EpisodeLearningResult& b) {
        return a.actions == b.actions && a.values == b.values && a.log_probabilities == b.log_probabilities
            && a.rewards == b.rewards && a.returns == b.returns && a.update_loss == b.update_loss
            && a.update_time == b.update_time && a.pending_steps == b.pending_steps
            && a.exploration_epsilon == b.exploration_epsilon && a.exploration_steps == b.exploration_steps;
    };
    if (!same(expected, actual) || expected_parameters != restored.parameter_values() || expected_target != restored.target_parameter_values())
        throw std::runtime_error("checkpoint resume differs from uninterrupted learning");
    std::cout << "CASE " << name << '\n'; snapshot(actual); parameters(restored);
    bool rejected = false;
    try { restored.restore_checkpoint({1, 2, 3}); } catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid checkpoint was accepted");
    const auto continued = restored.learn(env);
    const auto continued_parameters = restored.parameter_values();
    const auto continued_target = restored.target_parameter_values();
    original.restore_checkpoint(after_update);
    const auto baseline = original.learn(env);
    if (!same(continued, baseline) || original.parameter_values() != continued_parameters || original.target_parameter_values() != continued_target)
        throw std::runtime_error("failed checkpoint restore modified the live session or RNG");
    snapshot(continued); parameters(restored);
    emit_numeric_scalar("invalid_rejected", "Bool", static_cast<std::uint8_t>(rejected));
}
void initialization_failures() {
    for (const bool recurrent : {false,true}) {
        auto o = options("dqn");
        if (recurrent) o.policy = p::PolicyKind::gcn_seq2seq;
        else o.weight_decay = -.1;
        auto env = environment(recurrent ? "dual_gcn" : "ppo");
        rl::LearningSession session(o);
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool rejected = false;
            try { session.learn(env); } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected || !session.parameter_values().empty() || !session.target_parameter_values().empty()
                || session.update_time() || session.pending_steps()) throw std::runtime_error("failed initialization left a partial session");
        }
    }
    std::cout << "{\"status\":\"PASS\",\"failed_initialization_attempts\":4}\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--benchmark") { benchmark(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--benchmark-value") { benchmark(true); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--invariants") { initialization_failures(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--checkpoint-case") { checkpoint_case(argv[2]); return 0; }
        std::string selected;
        if (argc == 3 && std::string(argv[1]) == "--case") selected = argv[2];
        else if (argc != 1) throw std::invalid_argument("invalid learning harness argument");
        bool found = false;
        for (const auto& name : cases) if (selected.empty() || selected == name) {
            found = true; std::cout << "CASE " << name << '\n'; run(name);
        }
        if (!found) throw std::invalid_argument("unknown learning case");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
