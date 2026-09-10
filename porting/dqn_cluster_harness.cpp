#include "numeric_bytes.h"
#include "feature_constructor_fixture.h"
#include "../virne/solver/learning/rl_core/learning_session.h"
#include <iostream>
#include <stdexcept>
#include <string>

namespace rl = virne::solver::learning::rl_core;
namespace fixture = feature_constructor_fixture;
using virne::porting::emit_numeric_bytes;
using virne::porting::emit_numeric_scalar;
namespace {
rl::LearningSessionOptions options(const bool batch_norm) {
    rl::LearningSessionOptions o;
    o.seed = 401; o.embedding_dim = 8; o.num_layers = 2; o.target_steps = 7; o.intraop_threads = 1;
    o.learning_rates = {.003,.007,.002,.004,.005}; o.weight_decay = .01;
    o.training.algorithm = rl::UpdateAlgorithm::double_dqn; o.training.batch_size = 3;
    o.training.max_grad_norm = .25; o.batch_norm = batch_norm; o.include_failed = true;
    o.dqn = {3,.6,.1,7.,713,true,true};
    return o;
}
rl::InstanceEnvironment environment(const bool failed) {
    auto physical = fixture::physical_network(); auto virtual_net = fixture::virtual_network();
    if (failed) virtual_net.set_node_attrs_data({fixture::dense_node(fixture::node_id(virtual_net,"cpu"),{200,35,15,25})});
    return rl::InstanceEnvironment(std::move(physical),std::move(virtual_net));
}
void parameters(const rl::LearningSession& session) {
    const auto online = session.parameter_values(), target = session.target_parameter_values();
    emit_numeric_bytes("parameters","Double",online,{online.size()});
    emit_numeric_bytes("target_parameters","Double",target,{target.size()});
}
void episode(const rl::EpisodeLearningResult& r,const rl::LearningSession& session) {
    emit_numeric_bytes("actions","Long",r.actions,{r.actions.size()});
    for (const auto& entry : std::vector<std::pair<const char*,const std::vector<double>*>>{
        {"rewards",&r.rewards},{"values",&r.values},{"logprobs",&r.log_probabilities},{"returns",&r.returns}})
        emit_numeric_bytes(entry.first,"Double",*entry.second,{entry.second->size()});
    emit_numeric_bytes("dones","Bool",r.dones,{r.dones.size()});
    emit_numeric_scalar("last_value","Double",r.last_value);
    emit_numeric_scalar("success","Bool",static_cast<std::uint8_t>(r.solution.result));
    emit_numeric_scalar("pending","Long",r.pending_steps); emit_numeric_scalar("updates","Long",r.update_time);
    if (r.update_loss) emit_numeric_scalar("loss","Double",*r.update_loss); else std::cout << "loss|undefined\n";
    emit_numeric_scalar("epsilon","Double",*r.exploration_epsilon);
    emit_numeric_scalar("steps","Long",r.exploration_steps); parameters(session);
}
void run(const std::string& name) {
    auto o = options(name == "bn_target_resume");
    if (name == "bn_target_resume") o.target_steps = 1;
    if (name == "graph_pending_resume") { o.policy = virne::solver::learning::rl_policy::PolicyKind::gcn_mlp; o.target_steps = 1000; }
    auto session = std::make_unique<rl::LearningSession>(o);
    auto valid = environment(false), failed = environment(true);
    if (name == "fresh_seed_resume") session->set_exploration_seed(947);
    for (int i = 0; i < 6; ++i) {
        if (i == 2 && name == "reseed_resume") session->set_exploration_seed(947);
        if (i == 3) {
            if (name == "graph_pending_resume" && !session->pending_steps()) throw std::runtime_error("graph case requires a nonempty rollout");
            if (name == "bn_target_resume" && (session->update_time() != 3 || session->parameter_values() == session->target_parameter_values()))
                throw std::runtime_error("BatchNorm case requires updated and distinct online/target policies");
            const auto bytes = session->checkpoint();
            auto restored = std::make_unique<rl::LearningSession>(o);
            restored->restore_checkpoint(bytes);
            session = std::move(restored);
            if (name == "bn_target_resume") {
                std::cout << "EVAL\n";
                episode(session->solve(valid),*session);
            }
            if (name == "graph_pending_resume") {
                emit_numeric_scalar("pending_flush","Double",*session->update()); parameters(*session);
            }
        }
        if (i == 4 && name == "incompatible_checkpoint") {
            const auto saved = session->checkpoint();
            auto different_options = o; different_options.embedding_dim = 10;
            rl::LearningSession different(different_options); different.solve(valid);
            const auto incompatible = different.checkpoint();
            session->restore_checkpoint(saved);
            bool rejected = false;
            try { session->restore_checkpoint(incompatible); } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("incompatible checkpoint accepted");
            std::cout << "incompatible_rejected|Bool||01\n";
        }
        std::cout << "EPISODE " << i << '\n';
        const auto result = session->learn(i == 1 || i == 4 ? failed : valid);
        episode(result,*session);
    }
    const auto loss = session->update();
    if (loss) emit_numeric_scalar("flush_loss","Double",*loss); else std::cout << "flush_loss|undefined\n";
    parameters(*session);
}
}
int main(int argc,char** argv) {
    const std::string selected = argc == 3 && std::string(argv[1]) == "--case" ? argv[2] : "";
    if (argc != 1 && selected.empty()) return 2;
    int errors = 0;
    for (const std::string name : {"bn_target_resume","graph_pending_resume","fresh_seed_resume","reseed_resume","incompatible_checkpoint"}) {
        if (!selected.empty() && selected != name) continue;
        std::cout << "CASE " << name << '\n';
        try { run(name); }
        catch (const std::exception& error) { std::cout << "ERROR " << error.what() << '\n'; ++errors; }
    }
    return errors ? 1 : 0;
}
