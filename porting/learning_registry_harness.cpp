#include "numeric_bytes.h"
#include "online_cluster_fixture.h"
#include "../virne/solver/learning/solver_registry.h"
#include "../virne/solver/learning/online_learning.h"
#include "../virne/core/logger.h"
#include <chrono>
#include <sstream>

namespace learn=virne::solver::learning;
namespace rl=learn::rl_core;
namespace core=virne::core;
namespace sol=virne::solver;
using virne::porting::emit_numeric_bytes;
using virne::porting::emit_numeric_scalar;
namespace {
learn::LearningSolverOptions options() {
    learn::LearningSolverOptions o;
    o.session.seed=401; o.session.embedding_dim=8; o.session.num_layers=3; o.session.intraop_threads=1;
    o.session.target_steps=4; o.session.training.batch_size=3; o.session.training.repeat_times=1;
    o.session.training.max_grad_norm=.25; o.session.learning_rates={.003,.007,.002,.004,.005};
    o.session.weight_decay=.01; o.session.include_failed=true; return o;
}
core::controller::ControllerSelection selection() {
    auto c=online_cluster_fixture::lifecycle_config("registry").controller;
    c.constraints.node_at_node=c.node_resources; c.constraints.link_at_link=c.link_resources;
    c.hard_node_constraints=c.node_resources; c.hard_link_constraints=c.link_resources; return c;
}
struct Services {
    core::controller::Controller controller{selection()};
    core::Counter counter;
    core::Recorder recorder;
    core::Logger logger;
    static core::LoggerConfig logger_config() {
        core::LoggerConfig c; c.backends={false,false}; c.solver_name="registry";c.run_id="run";return c;
    }
    Services():recorder(core::Counter{},online_cluster_fixture::lifecycle_config("registry").recorder),logger(logger_config()){}
    sol::SolverDependencies dependencies(){return {std::cref(controller),std::ref(recorder),std::cref(counter),std::ref(logger)};}
};
sol::SolverConfig base_options(){sol::SolverConfig c;c.seed=401;c.shortest_method=core::controller::ShortestPathMethod::bfs_shortest;return c;}
void state(const learn::LearningSolverResult& r,const learn::LearningSolver& solver) {
    const auto& e=r.episode;
    emit_numeric_bytes("actions","Long",e.actions,{e.actions.size()});
    emit_numeric_bytes("rewards","Double",e.rewards,{e.rewards.size()});
    emit_numeric_bytes("values","Double",e.values,{e.values.size()});
    emit_numeric_bytes("returns","Double",e.returns,{e.returns.size()});
    emit_numeric_scalar("result","Bool",static_cast<std::uint8_t>(r.solution.result));
    emit_numeric_scalar("ratio","Double",r.solution.v_net_r2c_ratio);
    emit_numeric_scalar("updates","Long",solver.update_time());
    if(e.update_loss) emit_numeric_scalar("loss","Double",*e.update_loss);else std::cout<<"loss|undefined\n";
    const auto p=solver.parameter_values();emit_numeric_bytes("parameters","Double",p,{p.size()});
}
void catalog(){
    sol::SolverRegistry registry;learn::register_learning_solvers(registry,options());registry.freeze();
    for(const auto& d:learn::registered_learning_solvers()){
        const auto o=learn::resolve_learning_solver_options(d.name,options());
        const auto id=registry.resolve(d.name);
        std::cout<<d.name<<'|'<<static_cast<int>(o.session.policy)<<'|'<<static_cast<int>(o.session.training.algorithm)<<'|'
            <<static_cast<int>(o.environment.mode)<<'|'<<static_cast<int>(o.environment.features)<<'|'
            <<o.environment.node_status<<'|'<<o.environment.aggregated_links<<'|'<<o.environment.degree<<'|'<<o.environment.other_metrics<<'|'
            <<static_cast<int>(o.environment.reward)<<'|'<<o.session.include_failed<<'|'
            <<static_cast<int>(registry.descriptor(id).category)<<'\n';
    }
}
void instance_case(const std::string& name){
    Services services;sol::SolverRegistry registry;learn::register_learning_solvers(registry,options());registry.freeze();
    auto base=registry.create(registry.resolve(name),services.dependencies(),base_options());
    auto* solver=dynamic_cast<learn::LearningSolver*>(base.get());if(!solver)throw std::runtime_error("registry returned wrong solver type");
    auto p=feature_constructor_fixture::physical_network();auto v=feature_constructor_fixture::virtual_network();
    for(int i=0;i<2;++i){std::cout<<"LEARN "<<i<<'\n';state(solver->learn({v,p}),*solver);}
    const auto saved=solver->checkpoint();
    std::cout<<"SOLVE\n";state(solver->evaluate({v,p}),*solver);
    solver->restore_checkpoint(saved);
    std::cout<<"RESUME\n";state(solver->learn({v,p}),*solver);
}
void online_case(const bool benchmark){
    Services services;sol::SolverRegistry registry;learn::register_learning_solvers(registry,options());registry.freeze();
    auto base=registry.create(registry.resolve("ppo_mlp+"),services.dependencies(),base_options());
    auto& solver=dynamic_cast<learn::LearningSolver&>(*base);
    auto o=solver.learning_options().environment;o.mode=rl::InstanceStepMode::solution;
    rl::OnlineEnvironment online(feature_constructor_fixture::physical_network(),online_cluster_fixture::simulator(),
        online_cluster_fixture::lifecycle_config("registry_online"),o);
    online.reset();
    if(benchmark) solver.evaluate({online.lifecycle().current_virtual_network(),online.lifecycle().physical_network()});
    const auto start=std::chrono::steady_clock::now();
    const auto epoch=learn::run_online_learning_epoch(online,solver,{true,false,true,true});
    const auto loss=epoch.final_loss;
    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    if(benchmark)std::cout<<"{\"elapsed_ns\":"<<elapsed<<"}\n";
    for(const auto& r:epoch.episodes){emit_numeric_bytes("actions","Long",r.actions,{r.actions.size()});emit_numeric_scalar("result","Bool",static_cast<std::uint8_t>(r.solution.result));}
    emit_numeric_bytes("inservice","Long",epoch.inservice_counts,{epoch.inservice_counts.size()});
    emit_numeric_scalar("records","Long",static_cast<std::int64_t>(online.lifecycle().recorder().memory().size()));
    emit_numeric_scalar("revenue","Double",online.lifecycle().recorder().state().total_revenue);
    emit_numeric_scalar("cost","Double",online.lifecycle().recorder().state().total_cost);
    if(loss)emit_numeric_scalar("loss","Double",*loss);else std::cout<<"loss|undefined\n";
    const auto p=solver.parameter_values();emit_numeric_bytes("parameters","Double",p,{p.size()});
}
void benchmark_epoch(){
    std::vector<std::int64_t> samples;std::string reference;
    for(int repetition=-3;repetition<5;++repetition){
        std::ostringstream capture;auto* previous=std::cout.rdbuf(capture.rdbuf());
        try{online_case(true);}catch(...){std::cout.rdbuf(previous);throw;}
        std::cout.rdbuf(previous);
        const auto output=capture.str();const auto newline=output.find('\n');
        const auto elapsed=std::stoll(output.substr(output.find(':')+1));
        const auto records=output.substr(newline+1);
        if(repetition==-3)reference=records;
        else if(reference!=records)throw std::runtime_error("registry benchmark output changed between samples");
        if(repetition>=0)samples.push_back(elapsed);
    }
    std::cout<<"{\"samples_ns\":[";
    for(std::size_t i=0;i<samples.size();++i){if(i)std::cout<<',';std::cout<<samples[i];}
    std::cout<<"]}\n"<<reference;
}
}
int main(int argc,char** argv){
    try{
        if(argc==2 && std::string(argv[1])=="--benchmark"){benchmark_epoch();return 0;}
        if(argc==3 && std::string(argv[1])=="--case" && std::string(argv[2])=="registry_online"){
            std::cout<<"CASE registry_online\n";online_case(false);return 0;
        }
        if(argc!=1)throw std::invalid_argument("invalid registry harness args");
        std::cout<<"CASE registry\n";catalog();
        int errors=0;
        for(const std::string name:{"ppo_mlp+","pg_mlp","ppo_att","a2c_dual_gcn+","registry_online"}){
            std::cout<<"CASE "<<name<<'\n';
            try{if(name=="registry_online")online_case(false);else instance_case(name);}
            catch(const std::exception& error){std::cout<<"ERROR "<<error.what()<<'\n';++errors;}
        }
        return errors?1:0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
