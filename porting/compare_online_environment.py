#!/usr/bin/env python3
"""Compare complete online RL trajectories with actual Python lifecycle classes.

The concrete adapter supplies the abstract observation/reward hooks.  All
transition methods come from the source OnlineRLEnv classes.  The simulator is
prepared input, and record saving is disabled by source configuration.  Every
case is collected even if a source class raises, so one grouped gate identifies
the complete cluster's failures before repairs are made.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
from omegaconf import OmegaConf

import compare_instance_environment as base

ROOT, LOADER, CORE, NET = base.ROOT, base.LOADER, base.CORE, base.NET
ONLINE = sys.modules["virne.solver.learning.rl_core.online_rl_environment"]
FEATURES = sys.modules["virne.solver.learning.rl_core.feature_constructor"]
REWARDS = sys.modules["virne.solver.learning.rl_core.reward_calculator"]
SIMULATOR = sys.modules["virne.network.virtual_network_request_simulator"]
# SourceLoader uses a placeholder Recorder for instance-only imports.  Load the
# actual implementation now: online count/add/release must exercise its state.
RECORDER = LOADER.load("core/recorder.py")
LOADER.export("virne.core", RECORDER, ["Recorder"])
CASES = ("joint", "placement", "pair", "slots", "solution", "reject", "failure")

# Added only after the first grouped gate recorded the source exceptions.
# These replace broken dispatch/glue; resource mutation, routing, counting and
# lifecycle operations still run their original implementations.
def pair_ready(self,event_id=0):
    ONLINE.OnlineRLEnvBase.ready(self,event_id)
    self._curr_v_node_id=0
    self.candidates_dict=self.controller.construct_candidates_dict(self.v_net,self.p_net)
def pair_step(self,action):
    self._curr_v_node_id=int(action)%self.v_net.num_nodes
    return ONLINE.JointPRStepRLEnv.step(self,int(action)//self.v_net.num_nodes)
def pair_mask(self):
    mask=np.zeros((self.p_net.num_nodes,self.v_net.num_nodes),dtype=bool)
    for vnode,candidates in self.candidates_dict.items():mask[candidates,vnode]=True
    for vnode,pnode in self.solution.node_slots.items():mask[:,vnode]=False;mask[pnode,:]=False
    return mask.reshape(-1)
ONLINE.NodePairStepRLEnv.ready=pair_ready
ONLINE.NodePairStepRLEnv.step=pair_step
ONLINE.NodePairStepRLEnv.generate_action_mask=pair_mask

class SlotsCorrection(ast.NodeTransformer):
    def visit_If(self,node):
        if ast.unparse(node.test)=="len(node_slots) == self.v_net.num_nodes":
            return ast.parse("self.controller.deploy_with_node_slots(self.v_net,self.p_net,node_slots,self.solution,inplace=True,shortest_method=self.shortest_method,k_shortest=self.k_shortest,if_allow_constraint_violation=self.if_allow_constraint_violation)\nif not self.solution.result:\n    self.rollback_for_failure(self.get_failure_reason(self.solution))").body
        return self.generic_visit(node)
base.patch_function(ONLINE.NodeSlotsStepRLEnv,"step",SlotsCorrection())
original_solution_step=ONLINE.SolutionStepRLEnv.step
def detached_solution_step(self,solution):
    if solution.result:self.controller.deploy(self.v_net,self.p_net,solution)
    return original_solution_step(self,solution)
ONLINE.SolutionStepRLEnv.step=detached_solution_step
original_place_and_route=CORE.Controller.place_and_route
def reject_duplicate(self,v_net,p_net,v_node_id,p_node_id,solution,*args,**kwargs):
    if not self.reusable and p_node_id in solution.node_slots.values():
        solution.place_result=False
        return False,{}
    return original_place_and_route(self,v_net,p_net,v_node_id,p_node_id,solution,*args,**kwargs)
CORE.Controller.place_and_route=reject_duplicate


class PreparedSimulator(SIMULATOR.VirtualNetworkRequestSimulator):
    """Keep exactly the same prepared requests/events on every timed reset."""

    def renew(self, v_nets=True, events=True, seed=None):
        # Graph generation is outside this lifecycle gate on both runtimes.
        return None


def simulator(node_specs, link_specs):
    requests = []
    for request_id, arrival in enumerate((0.0, 1.0, 6.0)):
        _, virtual, _, _ = base.networks()
        for key, value in (("id", request_id), ("arrival_time", arrival), ("lifetime", 5.0)):
            virtual.set_graph_attribute(key, value)
        requests.append(virtual)
    events = [SIMULATOR.VirtualNetworkEvent(*values) for values in (
        (0, 1, 0, 0.0), (1, 1, 1, 1.0), (2, 0, 0, 5.0),
        (3, 0, 1, 6.0), (4, 1, 2, 6.0), (5, 0, 2, 11.0))]
    settings = {"output": {"save_dir": "/tmp/virne-online-cluster/prepared"},
        "node_attrs_setting": node_specs, "link_attrs_setting": link_specs,
        "num_v_nets": 3, "v_net_size": {"low": 4, "high": 5},
        "topology": {"type": "path"},
        "lifetime": {"distribution": "uniform", "low": 5, "high": 6},
        "arrival_rate": {"distribution": "poisson", "lam": 1}}
    return PreparedSimulator(requests, events, settings)


def configuration(name):
    return OmegaConf.create({
        "simulation": {"p_net_dataset_dir": "/tmp/virne-online-cluster/physical",
            "v_nets_dataset_dir": "/tmp/virne-online-cluster/prepared"},
        "experiment": {"run_id": "run", "seed": None, "if_load_v_nets": False,
            "save_root_dir": "/tmp/virne-online-cluster"},
        "recorder": {"if_temp_save_records": False, "if_save_records": False,
            "record_dir_name": "records", "summary_file_name": "summary.csv"},
        "solver": {"solver_name": name, "reusable": False,
            "node_ranking_method": "order", "link_ranking_method": "order",
            "matching_mathod": "greedy", "shortest_method": "bfs_shortest", "k_shortest": 10},
        "rl": {"feature_constructor": {
            "name": "p_net_v_net" if name == "pair" else "p_net_v_node",
            "extracted_attr_types": ["resource"], "if_use_node_status_flags": True,
            "if_use_aggregated_link_attrs": True, "if_use_degree_metric": True,
            "if_use_more_topological_metrics": True},
            "reward_calculator": {"name": "fixed_intermediate", "intermediate_reward": 0.1}}})


class ConcreteObservationReward:
    """Only fills source abstract hooks and tracks its terminal observation."""

    def reset(self, seed=None):
        self.done = False
        return super().reset(seed)

    def ready(self, event_id=0):
        super().ready(event_id)
        NET.TopologicalMetricCalculator.add_to_cache("v_net",
            NET.TopologicalMetricCalculator.calculate(self.v_net))
        cls = FEATURES.FeatureConstructorRegistry.get(self.config.rl.feature_constructor.name)
        self.feature_constructor = cls(self.p_net, self.v_net, self.config)
        self.reward_calculator = REWARDS.RewardCalculatorRegistry.get("fixed_intermediate")(self.config)

    def get_observation(self):
        if self.done:
            return None
        obs = self.feature_constructor.construct(self.p_net, self.v_net, self.solution, self.curr_v_node_id)
        obs["action_mask"] = self.generate_action_mask()
        return obs

    def compute_reward(self, *args, **kwargs):
        reward = self.reward_calculator.compute(self.p_net, self.v_net, self.solution)
        self.v_net_reward += reward
        self.cumulative_reward += reward
        return reward

    def transit_obs(self):
        self.done = super().transit_obs()
        return self.done


def environment(name="joint"):
    p_net, _, node_specs, link_specs = base.networks()
    config = configuration(name)
    controller = CORE.Controller(node_specs, link_specs, [], config=config.solver)
    counter = CORE.Counter(node_specs, link_specs, [], config=config)
    recorder = RECORDER.Recorder(counter, config)
    logger = logging.Logger("online_cluster", level=logging.CRITICAL + 1)
    logger.addHandler(logging.NullHandler())
    parent = {"joint": ONLINE.JointPRStepRLEnv, "placement": ONLINE.PlaceStepRLEnv,
        "pair": ONLINE.NodePairStepRLEnv, "slots": ONLINE.NodeSlotsStepRLEnv,
        "solution": ONLINE.SolutionStepRLEnv,
        "reject": ONLINE.JointPRStepRLEnv, "failure": ONLINE.JointPRStepRLEnv}[name]
    cls = type("Concrete" + parent.__name__, (ConcreteObservationReward, parent), {})
    return cls(p_net, simulator(node_specs, link_specs), controller, recorder, counter,
        logger, config, allow_rejection=name == "reject", verbose=0)


def snapshot(env, reward=0.0, request_done=False):
    record, state = base.record, env.recorder.state
    event = None if env.done else env.curr_event
    lines = [record("done", env.done, "bool"), record("request_done", request_done, "bool"),
        record("reward", reward, "float64"), record("cumulative", env.cumulative_reward, "float64"),
        record("event_id", -1 if event is None else event["id"], "int64"),
        record("request_id", -1 if event is None else event["v_net_id"], "int64")]
    for label, key, dtype in (("arrivals", "v_net_count", "int64"),
        ("successes", "success_count", "int64"), ("inservice", "inservice_count", "int64"),
        ("revenue", "total_revenue", "float64"), ("cost", "total_cost", "float64")):
        lines.append(record(label, state[key], dtype))
    lines.append(record("records", len(env.recorder.memory), "int64"))
    node_values = [[env.p_net.nodes[node][name] for node in env.p_net.nodes]
        for name in ("cpu", "gpu", "ram")]
    link_values = [[env.p_net.edges[edge][name] for edge in env.p_net.edges]
        for name in ("bw", "latency")]
    lines += [record("node_resources", node_values, "float64"),
        record("link_resources", link_values, "float64")]
    if not env.done:
        obs = env.get_observation()
        lines.append(record("p_x", obs["p_net_x"], "float32"))
        if "v_node_x" in obs:
            lines.append(record("v_node_x", obs["v_node_x"], "float32"))
        if "v_net_x" in obs:
            lines.append(record("v_x", obs["v_net_x"], "float32"))
        lines += [record("mask", np.asarray(obs["action_mask"], dtype=bool).reshape(-1), "bool"),
            record("current", env.curr_v_node_id, "int64")]
    return lines


def detached_solution(env):
    detached = base.ENVS.JointPRStepInstanceRLEnv(copy.deepcopy(env.p_net), copy.deepcopy(env.v_net),
        env.controller, env.recorder, env.counter, env.logger, env.config)
    for node in range(4):
        if detached.step(node)[2]: break
    return detached.solution


def execute(env, name="joint", emit=True, lines=None):
    if lines is None: lines = []
    env.reset()
    if emit: lines.extend(snapshot(env))
    steps = 0
    while not env.done:
        previous_event = env.curr_event["id"]
        request_id = env.curr_event["v_net_id"]
        if name == "slots": action = {node: node for node in range(4)}
        elif name == "solution": action = detached_solution(env)
        elif name == "reject" and request_id == 1: action = env.p_net.num_nodes
        elif name == "failure" and request_id == 1: action = 0
        elif name == "pair": action = len(env.solution.node_slots) * 5
        else: action = env.curr_v_node_id
        _, reward, done, _ = env.step(action)
        env.done = done
        request_done = done or env.curr_event["id"] != previous_event
        if emit: lines.extend(snapshot(env, reward, request_done))
        steps += 1
        if steps > 30: raise RuntimeError("online cluster did not terminate")
    return lines


def source_hashes():
    # Includes files imported transitively after the initial SourceLoader pass.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("virne.") and getattr(module, "__file__", None):
            path = Path(module.__file__)
            if path.is_relative_to(ROOT):
                LOADER.hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(LOADER.hashes.items()))


def signature(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def first_difference(expected, actual):
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<missing>"
        right = actual[index] if index < len(actual) else "<missing>"
        if left != right:
            return {"line": index, "record": left.split("|")[0], "python": left[:250], "cpp": right[:250]}
    return None


def native_output(binary, arguments):
    result = subprocess.run([str(binary.resolve()), *arguments], capture_output=True, text=True)
    return result.stdout.splitlines(), result.returncode, result.stderr


def grouped(lines):
    result, name = {}, None
    for line in lines:
        if line.startswith("CASE "): name = line[5:]; result[name] = []
        if name is None: raise RuntimeError("unexpected native output before CASE: " + line)
        result[name].append(line)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-from", type=Path)
    parser.add_argument("--cases",nargs="+",choices=CASES)
    parser.add_argument("--reuse-report",type=Path)
    args = parser.parse_args()
    report = {"component": "online_environment", "source_sha256": source_hashes(),
        "reference_corrections": ["previously accepted computed link candidate feasibility and safe revoked-action membership",
            "node-pair ready uses writable internal current ID, initializes candidates and decodes scalar physical-major actions",
            "node-pair masks both selected virtual/physical nodes",
            "node-slots uses original deploy_with_node_slots to allocate and journal node resources before routing",
            "detached-solution adapter uses original Controller.deploy before online SolutionStep",
            "reject repeated physical nodes when reusable is false before source route raises NotImplementedError"],
        "adapter": "concrete source FeatureConstructor and FixedWeightRewardCalculator hooks, prepared simulator input, terminal observation sentinel; actual online transition/lifecycle/recorder/controller/counter methods",
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "platform": platform.platform(), "numpy": np.__version__,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.benchmark_from:
        parity_bytes = args.benchmark_from.read_bytes()
        parity = json.loads(parity_bytes)
        if parity.get("status") != "PASS" or parity.get("component") != report["component"]:
            raise RuntimeError("a passing online environment cluster report is required")
        if parity.get("source_sha256") != report["source_sha256"] or parity.get("native_sha256") != report["native_sha256"]:
            raise RuntimeError("source/binary changed after the accepted online gate")
        native, code, stderr = native_output(args.native, ["--benchmark"])
        if code: raise RuntimeError(stderr or "native benchmark failed")
        native_samples, actual = json.loads(native[0])["samples_ns"], native[1:]
        env, samples = environment("joint"), []
        for repetition in range(-3, 5):
            started = time.perf_counter_ns()
            execute(env, "joint", False)
            elapsed = time.perf_counter_ns() - started
            if repetition >= 0: samples.append(elapsed)
        expected = snapshot(env)
        mismatch = first_difference(expected, actual)
        report.update(status="PASS" if mismatch is None else "FAIL", mismatch=mismatch,
            workload="Reset plus complete three-request joint trajectory, including departure release, feature generation and recorder updates; initial construction and output serialization excluded",
            warmups=3, repetitions=5, differential_rerun=False,
            parity_reused_from=args.benchmark_from.name, parity_report_sha256=hashlib.sha256(parity_bytes).hexdigest(),
            native_samples_ns=native_samples, python_samples_ns=samples,
            native_median_ns=statistics.median(native_samples), python_median_ns=statistics.median(samples),
            native_p95_ns=float(np.percentile(native_samples,95)), python_p95_ns=float(np.percentile(samples,95)),
            python_over_native_speedup=statistics.median(samples)/statistics.median(native_samples),
            python_output_sha256=signature(expected), native_output_sha256=signature(actual))
    else:
        selected=args.cases or CASES
        native, code, stderr = native_output(args.native, ["--cases",",".join(selected)] if args.cases else [])
        actual, expected, failures = grouped(native), {}, {}
        for name in selected:
            expected[name] = ["CASE " + name]
            try:
                execute(environment(name), name, lines=expected[name])
            except Exception as error:
                expected[name].append("ERROR " + str(error))
                failures[name] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        results = {}
        if args.reuse_report:
            reused=json.loads(args.reuse_report.read_bytes())
            for name,value in reused["cases"].items():
                if name not in selected and value["status"]=="PASS":results[name]={**value,"reused_from":args.reuse_report.name}
        for name in selected:
            mismatch = first_difference(expected[name], actual.get(name, []))
            native_error = next((line[6:] for line in actual.get(name, []) if line.startswith("ERROR ")), None)
            results[name] = {"status": "FAIL" if mismatch or name in failures or native_error else "PASS",
                "records": len(expected[name]), "mismatch": mismatch,
                "python_error": failures.get(name), "native_error": native_error}
            results[name]["python_sha256"]=signature(expected[name])
            results[name]["native_sha256"]=signature(actual.get(name,[]))
        report.update(status="PASS" if code == 0 and list(actual) == list(selected) and all(value["status"] == "PASS" for value in results.values()) else "FAIL",
            cases=results, records=sum(value["records"] for value in results.values()), native_exit_code=code, native_stderr=stderr,
            native_case_order=list(actual),
            executed_cases=list(selected),reused_report=args.reuse_report.name if args.reuse_report else None,
            python_output_sha256=signature([line for block in expected.values() for line in block]),
            native_output_sha256=signature(native), benchmark="NOT_RUN")
    report["source_sha256"] = source_hashes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "records", "native_median_ns", "python_median_ns", "python_over_native_speedup") if key in report}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
