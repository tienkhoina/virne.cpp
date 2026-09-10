#!/usr/bin/env python3
"""Compare native RL transitions with actual Python environment/controller classes."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import networkx as nx
import numpy as np
from omegaconf import OmegaConf
from oracle.rl_source_loader import SourceLoader

ROOT = Path(__file__).resolve().parents[2] / "virne" / "virne"
LOADER = SourceLoader(ROOT)
ENVS = LOADER.environment()
NET = sys.modules["virne.network"]
CORE = sys.modules["virne.core"]


def patch_function(owner, name, transformer):
    original = getattr(owner, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, original.__code__.co_filename, "exec"), namespace)
    setattr(owner, name, namespace[name])


class CandidateCorrection(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "new_filter" for target in node.targets):
            node.value.right.args[0] = ast.Name(id="suitable_nodes", ctx=ast.Load())
        return self.generic_visit(node)


class RevokeCorrection(ast.NodeTransformer):
    def visit_comprehension(self, node):
        for condition in node.ifs:
            if isinstance(condition, ast.Compare):
                for i, value in enumerate(condition.comparators):
                    if isinstance(value, ast.Name) and value.id == "revoked_actions":
                        condition.comparators[i] = ast.Name(id="candidate_nodes", ctx=ast.Load())
        return self.generic_visit(node)


patch_function(CORE.Controller, "find_candidate_nodes", CandidateCorrection())
patch_function(ENVS.RLBaseEnv, "generate_action_mask", RevokeCorrection())


def pair_mask(self):
    mask = np.zeros((self.v_net.num_nodes, self.p_net.num_nodes), dtype=bool)
    for vnode, candidates in self.candidates_dict.items(): mask[vnode, candidates] = True
    for vnode, pnode in self.solution.node_slots.items():
        mask[vnode, :] = False
        if not self.reusable: mask[:, pnode] = False
    return mask.T.reshape(-1)


# The reference changes are in-memory and restricted to confirmed defects.
ENVS.NodePairStepInstanceRLEnv.generate_action_mask = pair_mask


def spec(name, owner, kind="resource", originator=None):
    result = {"name": name, "owner": owner, "type": kind}
    if originator is not None: result["originator"] = originator
    return result


def networks(failure=0):
    p_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4),(2,5)]
    v_edges = [(0,1),(0,2),(1,2),(2,3)]
    node_specs = [spec(name, "node") for name in ("cpu", "gpu", "ram")]
    link_specs = [spec(name, "link") for name in ("bw", "latency")]
    p_node_specs = node_specs + [spec("max_" + name, "node", "extrema", name) for name in ("cpu", "gpu", "ram")]
    p_link_specs = link_specs + [spec("max_" + name, "link", "extrema", name) for name in ("bw", "latency")]
    # NetworkX Graph(existing_graph) rebuilds adjacency in edge-iteration order.
    # Populate the prepared inputs directly to preserve the same neighbor order
    # as the native fixture, which is part of shortest-path tie behavior.
    physical = NET.PhysicalNetwork(config={"node_attrs_setting": p_node_specs, "link_attrs_setting": p_link_specs})
    virtual = NET.VirtualNetwork(config={"node_attrs_setting": node_specs, "link_attrs_setting": link_specs}, id=0, arrival_time=0.0, lifetime=100.0)
    physical.add_nodes_from(range(6)); physical.add_edges_from(p_edges)
    virtual.add_nodes_from(range(4)); virtual.add_edges_from(v_edges)
    for name, values in {"cpu": [100,80,60,90,70,50], "gpu": [40,60,20,80,50,30], "ram": [256,128,192,160,224,96]}.items():
        for node, value in enumerate(values): physical.nodes[node].update({name: value, "max_" + name: value})
    for name, values in {"bw": [100,80,60,90,70,50,75,65], "latency": [8,16,24,12,20,28,10,18]}.items():
        for edge, value in zip(p_edges, values): physical.edges[edge].update({name: value, "max_" + name: value})
    for name, values in {"cpu": [20,35,15,25], "gpu": [5,10,15,20], "ram": [32,48,64,40]}.items():
        for node, value in enumerate(values): virtual.nodes[node][name] = value
    for name, values in {"bw": [12,20,16,24], "latency": [4,6,8,10]}.items():
        for edge, value in zip(v_edges, values): virtual.edges[edge][name] = value
    if failure == 1: virtual.nodes[0]["cpu"] = 200
    if failure == 2:
        for edge in p_edges: physical.edges[edge]["bw"] = 1
    return physical, virtual, node_specs, link_specs


def environment(mode="joint", rejection=False, revocable=False, reward=2, failure=0):
    p, v, node_specs, link_specs = networks(failure)
    solver_options = {"reusable": False, "node_ranking_method": "order", "link_ranking_method": "order",
                      "matching_mathod": "greedy", "shortest_method": "bfs_shortest", "k_shortest": 10}
    config = OmegaConf.create({"solver": solver_options, "rl": {
        "feature_constructor": {"name": "p_net_v_net" if mode == "node_pair" else "p_net_v_node",
            "extracted_attr_types": ["resource"], "if_use_node_status_flags": True,
            "if_use_aggregated_link_attrs": True, "if_use_degree_metric": True, "if_use_more_topological_metrics": True},
        "reward_calculator": {"name": ["gradual_intermediate", "adaptive_intermediate", "fixed_intermediate", "vanilla"][reward], "intermediate_reward": 0.1}}})
    controller = CORE.Controller(node_specs, link_specs, [], config=solver_options)
    counter = CORE.Counter(node_specs, link_specs, [], config=config)
    cls = {"joint": ENVS.JointPRStepInstanceRLEnv, "placement": ENVS.PlaceStepInstanceRLEnv,
           "node_pair": ENVS.NodePairStepInstanceRLEnv, "slots": ENVS.NodeSlotsStepInstanceRLEnv}[mode]
    result = cls(p, v, controller, object(), counter, object(), config,
                 allow_rejection=rejection, allow_revocable=revocable)
    return result


def record(name, value, dtype):
    value = np.asarray(value, dtype=dtype)
    label = {"float32": "Float", "float64": "Double", "int64": "Long", "bool": "Bool"}[str(value.dtype)]
    return name + "|" + label + "|" + "".join(f"{i}," for i in value.shape) + "|" + np.ascontiguousarray(value).tobytes().hex()


def snapshot(env, reward=0.0, done=False):
    obs, solution = env.get_observation(), env.solution
    lines = []
    for prefix in ("p_net",):
        lines += [record(prefix + ".x", obs[prefix + "_x"], "float32"),
                  record(prefix + ".edge_index", obs[prefix + "_edge_index"], "int64"),
                  record(prefix + ".edge_attr", obs[prefix + "_edge_attr"], "float32")]
    if "v_node_x" in obs: lines.append(record("v_node_x", obs["v_node_x"], "float32"))
    if "v_net_x" in obs:
        lines += [record("v_net.x", obs["v_net_x"], "float32"), record("v_net.edge_index", obs["v_net_edge_index"], "int64"),
                  record("v_net.edge_attr", obs["v_net_edge_attr"], "float32")]
    mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
    lines += [record("mask", mask, "bool"), record("current", env.curr_v_node_id, "int64"),
              record("v_net_size", env.v_net.num_nodes, "int64"), record("num_actions", mask.size, "int64"),
              record("reward", reward, "float64"), record("done", done, "bool")]
    for name, source, dtype in (("result", "result", "bool"), ("place_result", "place_result", "bool"),
        ("route_result", "route_result", "bool"), ("rejection", "early_rejection", "bool"),
        ("interactions", "num_interactions", "int64"), ("revokes", "revoke_times", "int64"),
        ("total_reward", "v_net_reward", "float64"), ("revenue", "v_net_revenue", "float64"),
        ("cost", "v_net_cost", "float64"), ("ratio", "v_net_r2c_ratio", "float64"),
        ("violation", "v_net_total_hard_constraint_violation", "float64")):
        lines.append(record(name, solution[source], dtype))
    lines += [record("slots", np.array(list(solution.node_slots.items()), dtype=np.int64).reshape(-1, 2), "int64"),
              record("selected", solution.selected_actions, "int64"), record("path_count", len(solution.link_paths), "int64")]
    for i, (virtual_edge, path) in enumerate(solution.link_paths.items()):
        lines += [record(f"path_{i}.virtual", virtual_edge, "int64"), record(f"path_{i}.physical", np.asarray(path, dtype=np.int64).reshape(-1, 2), "int64")]
    return lines


def trajectory(actions, **kwargs):
    env = environment(**kwargs)
    lines = snapshot(env)
    for action in actions:
        _, reward, done, _ = env.step(action)
        lines.extend(snapshot(env, reward, done))
        if done: break
    env.reset()
    if kwargs.get("mode") == "node_pair": env._curr_v_node_id = 0
    return lines + snapshot(env)


def slots(incomplete=False):
    env = environment(mode="slots")
    lines = snapshot(env)
    assignments = {i: i for i in range(2 if incomplete else 4)}
    _, reward, done, _ = env.step(assignments)
    return lines + snapshot(env, reward, done)


def clone_case():
    env = environment()
    env.step(0)
    branch = copy.deepcopy(env)
    _, reward, done, _ = branch.step(4)
    lines = snapshot(branch, reward, done) + snapshot(env)
    branch.reset()
    return lines + snapshot(branch) + [record("balance", env.get_node_load_balance(0), "float64")]


def cases(selected=None):
    functions = {"joint": lambda: trajectory([0,1,2,3]), "placement": lambda: trajectory([0,1,2,3], mode="placement"),
        "duplicate": lambda: trajectory([0,0]), "node_failure": lambda: trajectory([0], failure=1),
        "link_failure": lambda: trajectory([0,1], failure=2), "reject": lambda: trajectory([0,6], rejection=True),
        "revoke": lambda: trajectory([0,1,6,4,2,3], revocable=True),
        "node_pair": lambda: trajectory([0,5,10,15], mode="node_pair"),
        "slots": slots, "incomplete_slots": lambda: slots(True), "clone": clone_case}
    for i in range(4): functions[f"reward_{i}"] = lambda kind=i: trajectory([0,1,2,3], reward=kind)
    if selected: functions = {selected: functions[selected]}
    result = {}
    for name, run in functions.items(): result[name] = ["CASE " + name, *run()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    expected = cases(args.case)
    command = [str(args.native.resolve())] + (["--case", args.case] if args.case else [])
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode: raise RuntimeError(process.stderr)
    actual, name = {}, None
    for line in process.stdout.splitlines():
        if line.startswith("CASE "): name = line[5:]; actual[name] = []
        actual[name].append(line)
    mismatches = []
    for name, lines in expected.items():
        if lines != actual.get(name):
            other = actual.get(name, [])
            for i in range(max(len(lines), len(other))):
                left, right = lines[i] if i < len(lines) else "<missing>", other[i] if i < len(other) else "<missing>"
                if left != right:
                    mismatches.append({"case": name, "line": i, "record": left.split("|")[0],
                        "python": left[:250], "cpp": right[:250]}); break
    signature = lambda blocks: hashlib.sha256("\n".join(line for lines in blocks.values() for line in lines).encode()).hexdigest()
    report = {"component": "instance_environment", "status": "FAIL" if mismatches else "PASS",
        "cases": list(expected), "records": sum(map(len, expected.values())), "mismatches": mismatches,
        "python_sha256": signature(expected), "cpp_sha256": signature(actual), "source_sha256": LOADER.hashes,
        "reference_corrections": ["apply computed link candidate feasibility", "check candidate membership before removing revoked actions",
            "physical-major node-pair mask without terminal fallback", "canonical bool mask and scalar size/action-count metadata",
            "reset node-pair current node to the initial ID"], "benchmark": "NOT_REQUESTED"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "records", "mismatches")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
