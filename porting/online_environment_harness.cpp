#include "numeric_bytes.h"
#include "online_cluster_fixture.h"
#include <chrono>
#include <iostream>

namespace rl = virne::solver::learning::rl_core;
using virne::porting::emit_numeric_bytes;
using virne::porting::emit_numeric_scalar;
namespace {
void snapshot(const rl::OnlineEnvironment& env,const double reward=0.,const bool request_done=false) {
    const auto obs = env.observation(); const auto& state = env.lifecycle().recorder().state();
    emit_numeric_scalar("done","Bool",static_cast<std::uint8_t>(obs.done));
    emit_numeric_scalar("request_done","Bool",static_cast<std::uint8_t>(request_done));
    emit_numeric_scalar("reward","Double",reward); emit_numeric_scalar("cumulative","Double",env.cumulative_reward());
    emit_numeric_scalar("event_id","Long",obs.event ? static_cast<std::int64_t>(obs.event->event_id) : -1);
    emit_numeric_scalar("request_id","Long",obs.event ? obs.event->request_id : -1);
    emit_numeric_scalar("arrivals","Long",state.virtual_network_count);
    emit_numeric_scalar("successes","Long",state.success_count); emit_numeric_scalar("inservice","Long",state.inservice_count);
    emit_numeric_scalar("revenue","Double",state.total_revenue); emit_numeric_scalar("cost","Double",state.total_cost);
    emit_numeric_scalar("records","Long",static_cast<std::int64_t>(env.lifecycle().recorder().memory().size()));
    const auto& physical = env.physical_network();
    const auto resource_values = [&](const auto& data) {
        std::vector<double> result;
        for (const auto& column : data) for (const auto& v : column) result.push_back(attr_to_double(v));
        return result;
    };
    const auto nodes = resource_values(virne::network::get_node_attrs_data(physical,{0,1,2}));
    const auto links = resource_values(virne::network::get_link_attrs_data(physical,{0,1}));
    emit_numeric_bytes("node_resources","Double",nodes,{3,physical.num_nodes()});
    emit_numeric_bytes("link_resources","Double",links,{2,physical.num_links()});
    if (obs.instance) {
        const auto& o = *obs.instance;
        emit_numeric_bytes("p_x","Float",o.features.p_net.x.values,{o.features.p_net.x.rows,o.features.p_net.x.columns});
        if (o.features.v_node_x) emit_numeric_bytes("v_node_x","Float",o.features.v_node_x->values,{o.features.v_node_x->values.size()});
        if (o.features.v_net) emit_numeric_bytes("v_x","Float",o.features.v_net->x.values,{o.features.v_net->x.rows,o.features.v_net->x.columns});
        emit_numeric_bytes("mask","Bool",o.action_mask,{o.action_mask.size()});
        emit_numeric_scalar("current","Long",o.curr_v_node_id);
    }
}
void execute(rl::OnlineEnvironment& env,const std::string& name,const bool emit) {
    env.reset(); if (emit) snapshot(env);
    std::int64_t step_index = 0;
    while (!env.done()) {
        const auto current = env.instance().current_virtual_node();
        const auto request_id = env.lifecycle().current_event().request_id;
        rl::OnlineStep step;
        if (name == "slots") step = env.step_node_slots({{0,0},{1,1},{2,2},{3,3}});
        else if (name == "solution") {
            rl::InstanceEnvironment detached(env.lifecycle().physical_network().clone(),env.lifecycle().current_virtual_network().clone());
            for (std::int64_t node = 0; node < 4; ++node) if(detached.step(node).done) break;
            step = env.step_solution(detached.solution_snapshot());
        } else if (name == "reject" && request_id == 1) step = env.reject();
        else if (name == "failure" && request_id == 1) step = env.step(0);
        else if (name == "pair") {
            const auto placed = env.instance().info().node_slots.size();
            step = env.step(static_cast<std::int64_t>(placed*5));
        } else step = env.step(current);
        if (emit) snapshot(env,step.reward,step.request_done);
        if (++step_index > 30) throw std::runtime_error("online cluster did not terminate");
    }
}
void benchmark() {
    std::vector<std::int64_t> samples;
    auto env = online_cluster_fixture::environment("joint");
    for (int repetition=-3;repetition<5;++repetition) {
        const auto start = std::chrono::steady_clock::now(); execute(*env,"joint",false);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        if (repetition>=0) samples.push_back(elapsed);
    }
    std::cout << "{\"samples_ns\":[";
    for (std::size_t i=0;i<samples.size();++i) { if(i) std::cout << ','; std::cout << samples[i]; }
    std::cout << "]}\n"; snapshot(*env);
}
void external_finish_invariant() {
    auto env=online_cluster_fixture::environment("joint");env->reset();env->step(0);
    while(!env->instance().done()) env->instance().step(env->instance().current_virtual_node());
    const double expected=env->instance().info().cumulative_reward;
    env->finish_instance();
    if(env->cumulative_reward()!=expected) throw std::runtime_error("external finish counted intermediate reward twice");
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--benchmark") { benchmark(); return 0; }
        const std::string selected=argc==3 && std::string(argv[1])=="--cases" ? argv[2] : "";
        if(argc!=1 && selected.empty()) throw std::invalid_argument("unknown online harness argument");
        int errors=0;
        for(const std::string name:{"joint","placement","pair","slots","solution","reject","failure"}) {
            if(!selected.empty() && (","+selected+",").find(","+name+",")==std::string::npos) continue;
            std::cout << "CASE " << name << '\n';
            try { auto env=online_cluster_fixture::environment(name); execute(*env,name,true); if(name=="solution")external_finish_invariant(); }
            catch(const std::exception& error) { std::cout << "ERROR " << error.what() << '\n'; ++errors; }
        }
        return errors ? 1 : 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
