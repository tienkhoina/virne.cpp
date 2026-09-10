#!/usr/bin/env python3
"""Source-derived solver registration and grouped native learning comparisons."""
from __future__ import annotations
import __future__
import ast
import copy
import hashlib
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf,open_dict
import compare_learning_session as session

REGISTRY={}
SOURCE_HASHES={}
class Registry:
    @staticmethod
    def register(solver_name,solver_type):
        def attach(cls): REGISTRY[solver_name]=cls;return cls
        return attach
class Agent:
    def __init__(self,env): self.InstanceEnv=env
def capture_algorithm(name):
    def initialize(self,controller,recorder,counter,logger,config,builder,converter,**kwargs):
        self.algorithm=name;self.builder=builder;self.converter=converter
    return type(name,(),{"__init__":initialize})
ALGORITHMS={name:capture_algorithm(name) for name in ("PGSolver","A2CSolver","PPOSolver","A3CSolver")}
POLICIES={"mlp":4,"cnn":0,"att":5,"gcn_mlp":1,"gat_mlp":2,"deep_edge_gat_mlp":3,"dual_gcn":7,"dual_gat":8,"dual_deep_edge_gat":9}
builders=SimpleNamespace(**{"build_"+name+"_policy":name for name in POLICIES})
BASE=session.environment_oracle

def source(relative):
    path=session.ROOT/relative
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=session.training.PINNED[relative]:raise RuntimeError("source drift: "+relative)
    SOURCE_HASHES[relative]=digest
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    tree.body=[node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    return compile(tree,str(path),"exec",flags=__future__.annotations.compiler_flag)

maker={"SolverRegistry":Registry,"InstanceAgent":Agent}
exec(source("reinforcement_learning/solver_maker.py"),maker)
for filename in ("mlp_solver.py","cnn_solver.py","att_solver.py","gnn_mlp_solver.py","dual_gnn_solver.py"):
    namespace={"np":np,"torch":torch,"open_dict":open_dict,"SolverRegistry":Registry,"InstanceAgent":Agent,
        "PolicyBuilder":builders,"TensorConvertor":session.CONVERT,"make_solver_class":maker["make_solver_class"],
        "JointPRStepInstanceRLEnv":BASE.ENVS.JointPRStepInstanceRLEnv,"PlaceStepInstanceRLEnv":BASE.ENVS.PlaceStepInstanceRLEnv,**ALGORITHMS}
    exec(source("reinforcement_learning/"+filename),namespace)
NAMES=[name for name in REGISTRY if not name.startswith("a3")]

def environment(name,physical=None,virtual=None,reference=None):
    if reference is None:reference=BASE.environment()
    config=reference.config;config.rl.if_use_negative_sample=True
    descriptor=REGISTRY[name](None,None,None,None,config)
    env=descriptor.InstanceEnv(reference.p_net if physical is None else physical,reference.v_net if virtual is None else virtual,
        reference.controller,object(),reference.counter,object(),config)
    return env,descriptor

def catalog():
    result=["CASE registry"]
    for name in NAMES:
        env,d=environment(name)
        f=env.config.rl.feature_constructor
        mode=0 if isinstance(env,BASE.ENVS.PlaceStepInstanceRLEnv) else 1
        values=[POLICIES[d.builder],{"PGSolver":0,"A2CSolver":1,"PPOSolver":2}[d.algorithm],mode,
            {"p_net_v_node":1,"p_net_v_net":2}[f.name],int(f.if_use_node_status_flags),int(f.if_use_aggregated_link_attrs),
            int(f.if_use_degree_metric),int(f.if_use_more_topological_metrics),
            {"gradual_intermediate":0,"adaptive_intermediate":1,"fixed_intermediate":2,"vanilla":3}[env.config.rl.reward_calculator.name],
            int(env.config.rl.if_use_negative_sample),6]
        result.append(name+"|"+"|".join(map(str,values)))
    return result

def build_solver(name,env):
    d=REGISTRY[name](None,None,None,None,env.config)
    original=session.build_policy
    def model(kind,obs):
        common=dict(num_layers=3,embedding_dim=8,dropout_prob=0.,batch_norm=False)
        nodes,features=obs["p_net_x"].shape
        if d.builder=="dual_gcn":return session.policies.DUAL["BiGcnActorCritic"](nodes,features,obs["p_net_edge_attr"].shape[1],obs["v_net_x"].shape[1],obs["v_net_edge_attr"].shape[1],**common)
        vnode=len(obs["v_node_x"])
        if d.builder=="att":return session.policies.ATT["AttActorCritic"](nodes,features,vnode,**common)
        return session.training.mlp_namespace["MlpActorCritic"](features+vnode,nodes,**common)
    session.build_policy=model
    try:solver=session.build_solver("dual_gcn" if d.builder=="dual_gcn" else ("attention" if d.builder=="att" else "ppo"),env)
    finally:session.build_policy=original
    solver.config.rl.norm_advantage=False;solver.config.rl.norm_reward=False;solver.coef_mask_loss=0.
    solver.batch_size=3;solver.update=MethodType(session.training.namespace[d.algorithm].update,solver)
    solver.preprocess_obs=d.converter
    return solver

def snapshot(solver,solution,buffer,loss):
    record=BASE.record
    return [record("actions",buffer.actions,"int64"),record("rewards",buffer.rewards,"float64"),
        record("values",buffer.values,"float64"),record("returns",buffer.returns,"float64"),
        record("result",solution.result,"bool"),record("ratio",solution.v_net_r2c_ratio,"float64"),
        record("updates",solver.update_time,"int64"),"loss|undefined" if loss is None else record("loss",float(loss),"float64"),*session.parameters(solver)]

def collect(solver,env,learn=True):
    solver.policy.train(learn);solver.InstanceEnv=lambda *a,**k:env
    solution,buffer,last=session.collect(solver,env)
    if learn and (solution.result or env.config.rl.if_use_negative_sample):solver.buffer.merge(buffer)
    loss=solver.update() if learn and solver.buffer.size()>=4 else None
    return solution,buffer,loss

def instance_case(name):
    env,_=environment(name);solver=build_solver(name,env)
    lines=["CASE "+name]
    for i in range(2):
        solution,buffer,loss=collect(solver,env)
        lines += ["LEARN "+str(i),*snapshot(solver,solution,buffer,loss)]
    rng=torch.get_rng_state()
    solver.select_action=lambda obs,**kwargs:session.SOLVER.select_action(solver,obs,sample=False)
    solution,buffer,loss=collect(solver,env,False)
    lines += ["SOLVE",*snapshot(solver,solution,buffer,loss)]
    torch.set_rng_state(rng);solver.select_action=MethodType(session.SOLVER.select_action,solver)
    solution,buffer,loss=collect(solver,env)
    return lines+["RESUME",*snapshot(solver,solution,buffer,loss)]

def online_case(benchmark=False):
    import compare_online_environment as online
    env=online.environment("solution");env.reset()
    first,_=environment("ppo_mlp+",copy.deepcopy(env.p_net),copy.deepcopy(env.v_net))
    solver=build_solver("ppo_mlp+",first)
    if benchmark:
        solver.select_action=lambda obs,**kwargs:session.SOLVER.select_action(solver,obs,sample=False)
        collect(solver,first,False)
        solver.select_action=MethodType(session.SOLVER.select_action,solver)
    lines=["CASE registry_online"];inservice=[];episodes=[]
    start=time.perf_counter_ns()
    while not env.done:
        instance,_=environment("ppo_mlp+",copy.deepcopy(env.p_net),copy.deepcopy(env.v_net),reference=first)
        solution,buffer,loss=collect(solver,instance)
        env.step(solution)
        inservice.append(env.recorder.state["inservice_count"])
        episodes.append((buffer.actions,solution.result))
    loss=solver.update() if solver.buffer.size() else None
    elapsed=time.perf_counter_ns()-start
    for actions,success in episodes:
        lines += [BASE.record("actions",actions,"int64"),BASE.record("result",success,"bool")]
    lines += [BASE.record("inservice",inservice,"int64"),BASE.record("records",len(env.recorder.memory),"int64"),
        BASE.record("revenue",env.recorder.state["total_revenue"],"float64"),BASE.record("cost",env.recorder.state["total_cost"],"float64"),
        "loss|undefined" if loss is None else BASE.record("loss",float(loss),"float64"),*session.parameters(solver)]
    return lines,elapsed

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=["registry_online"]);parser.add_argument("--reuse-report",type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    functions={"registry":catalog,**{name:(lambda name=name:instance_case(name)) for name in ("ppo_mlp+","pg_mlp","ppo_att","a2c_dual_gcn+")},
        "registry_online":lambda:online_case()[0]}
    if args.case:functions={args.case:functions[args.case]}
    expected,errors={},{}
    for name,run in functions.items():
        try:expected[name]=run()
        except Exception as error:expected[name]=["CASE "+name];errors[name]=type(error).__name()+": "+str(error)
    env=dict(os.environ);env["LD_LIBRARY_PATH"]=str(Path(__file__).resolve().parents[1]/"libs/libtorch/lib")
    process=subprocess.run([str(args.native.resolve())]+(["--case",args.case] if args.case else []),capture_output=True,text=True,env=env)
    actual={};current=None
    for line in process.stdout.splitlines():
        if line.startswith("CASE "):current=line[5:];actual[current]=[]
        if current:actual[current].append(line)
    reports={}
    if args.reuse_report:
        saved=json.loads(args.reuse_report.read_bytes())
        for name,record in saved["cases"].items():
            if name not in functions and record["status"]=="PASS":reports[name]={**record,"reused_from":args.reuse_report.name}
    for name,lines in expected.items():
        other=actual.get(name,[]);mismatch=session.policies.base.first_difference(lines,other)
        if mismatch:
            i=mismatch["line"];mismatch["python"]=lines[i][:180] if i<len(lines) else "missing";mismatch["cpp"]=other[i][:180] if i<len(other) else "missing"
        reports[name]={"status":"PASS" if mismatch is None and name not in errors else "FAIL","records":len(lines),"mismatch":mismatch,
            "python_error":errors.get(name),"native_errors":[line for line in other if line.startswith("ERROR ")],
            "python_sha256":hashlib.sha256("\n".join(lines).encode()).hexdigest(),"native_sha256":hashlib.sha256("\n".join(other).encode()).hexdigest()}
        print(json.dumps({"case":name,**reports[name]}),flush=True)
    report={"component":"learning_solver_registry","status":"PASS" if process.returncode==0 and all(r["status"]=="PASS" for r in reports.values()) else "FAIL",
        "cases":reports,"source_sha256":SOURCE_HASHES,"native_sha256":hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "native_stderr":process.stderr,"registered_source_names":NAMES,"threads":1,"benchmark":"NOT_RUN"}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return 0 if report["status"]=="PASS" else 1

if __name__=="__main__":raise SystemExit(main())
