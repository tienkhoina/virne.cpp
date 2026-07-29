#!/usr/bin/env python3
"""Exact AST-isolated differential for the non-RL Environment lifecycle."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union


SOURCE_SHA256 = "6004cff2114e504e30c5490232763f71cb9e7799216c4f6ce8badc27a0e42b34"
WORKERS = (1, 2, 8)

BOUNDARIES = {
    "dependency_fakes":
        "Controller, Counter, Recorder, Logger, simulator, clock, and filesystem are narrow deterministic fakes; their production leaves retain independent frozen gates.",
    "dense_shared_ids":
        "The pinned Python ready/transit methods index vectors with request/event IDs. Shared cases use dense IDs; native sparse-ID preparation belongs to the unit gate.",
    "admission_logger":
        "The admission branch passes extra positional values to Logger.warning; the fake accepts them so the intended rejection/rollback lifecycle remains observable.",
    "deferred_surfaces":
        "Solver registries, JointPR, observations, reward implementations, action masks, RL, Torch, ML, logging backends, datasets, and persistence are excluded.",
}


def load_environment(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Environment source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    if set(classes) != {
            "BaseEnvironment", "SolutionStepEnvironment",
            "JointPRStepEnvironment"}:
        raise RuntimeError("Environment class inventory drift")

    base_names = [
        "ready", "reset", "release", "get_failure_reason",
        "rollback_for_failure", "transit_obs", "add_record",
        "count_and_add_record",
    ]
    base_methods = {
        node.name: node for node in classes["BaseEnvironment"].body
        if isinstance(node, ast.FunctionDef) and node.name in base_names
    }
    if list(base_methods) != base_names:
        raise RuntimeError(
            f"BaseEnvironment method inventory drift: {list(base_methods)}")
    step_methods = [
        node for node in classes["SolutionStepEnvironment"].body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    ]
    if len(step_methods) != 1:
        raise RuntimeError("SolutionStepEnvironment.step inventory drift")

    isolated_class = ast.ClassDef(
        name="OracleEnvironment",
        bases=[],
        keywords=[],
        body=[base_methods[name] for name in base_names] + step_methods,
        decorator_list=[],
        type_params=[],
    )
    isolated = ast.fix_missing_locations(ast.Module(
        body=[isolated_class], type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Union": Union,
        "Solution": FakeSolution,
        "copy": copy,
        "get_v_nets_dataset_dir_from_setting": lambda setting, seed=None:
            "unused-vnet-dataset",
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda path: False)),
        "time": SimpleNamespace(time=lambda: 1234.5),
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["OracleEnvironment"]


class FakeSolution(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @classmethod
    def from_v_net(cls, virtual_network):
        return cls({
            "v_net_id": virtual_network.id,
            "v_net_lifetime": virtual_network.lifetime,
            "v_net_arrival_time": virtual_network.arrival_time,
            "v_net_num_nodes": virtual_network.num_nodes,
            "v_net_num_edges": virtual_network.num_links,
            "result": False,
            "node_slots": {},
            "link_paths": {},
            "node_slots_info": {},
            "link_paths_info": {},
            "v_net_total_hard_constraint_violation": 0.0,
            "v_net_r2c_ratio": 0.0,
            "v_net_node_cost": 0.0,
            "v_net_link_cost": 0.0,
            "v_net_cost": 0.0,
            "v_net_revenue": 0.0,
            "v_net_time_cost": 0.0,
            "v_net_time_revenue": 0.0,
            "place_result": True,
            "route_result": True,
            "early_rejection": False,
            "description": "",
        })

    def to_dict(self):
        return copy.deepcopy(dict(self))


class FakeVirtualNetwork:
    def __init__(self, request_id: int, arrival_time: float):
        self.id = request_id
        self.arrival_time = arrival_time
        self.lifetime = 1.0
        self.num_nodes = 1
        self.num_links = 0


class FakePhysicalNetwork:
    def __init__(self):
        self.nodes = [{"cpu": 100.0}, {"cpu": 200.0}]
        self.links = {(0, 1): {"bandwidth": 300.0}}


class FakeSimulator:
    def __init__(self, request_count: int):
        self.v_nets = [
            FakeVirtualNetwork(index, index * 2.0)
            for index in range(request_count)
        ]
        self.events = []
        for index in range(request_count):
            self.events.extend([
                {"id": index * 2, "type": 1, "v_net_id": index,
                 "time": index * 2.0},
                {"id": index * 2 + 1, "type": 0, "v_net_id": index,
                 "time": index * 2.0 + 1.0},
            ])
        self.num_events = len(self.events)
        self.v_sim_setting = object()

    def renew(self, v_nets=True, events=True, seed=None):
        return None


class FakeCounter:
    @staticmethod
    def _node_cost(solution) -> float:
        return sum(
            float(value)
            for resources in solution["node_slots_info"].values()
            for value in resources.values()
        )

    def count_solution(self, virtual_network, solution):
        node_cost = self._node_cost(solution) if solution["result"] else 0.0
        solution["v_net_node_cost"] = node_cost
        solution["v_net_link_cost"] = 0.0
        solution["v_net_cost"] = node_cost
        solution["v_net_revenue"] = node_cost
        solution["v_net_time_cost"] = node_cost
        solution["v_net_time_revenue"] = node_cost
        solution["v_net_r2c_ratio"] = 1.0 if node_cost else 0.0
        return solution.to_dict()

    @staticmethod
    def calculate_sum_node_resource(physical_network):
        return sum(node["cpu"] for node in physical_network.nodes)

    @staticmethod
    def calculate_sum_link_resource(physical_network):
        return sum(link["bandwidth"] for link in physical_network.links.values())

    def calculate_sum_network_resource(
            self, physical_network, node=True, link=True):
        result = 0.0
        if node:
            result += self.calculate_sum_node_resource(physical_network)
        if link:
            result += self.calculate_sum_link_resource(physical_network)
        return result


class FakeController:
    @staticmethod
    def deploy(virtual_network, physical_network, solution):
        if not solution["result"]:
            return False
        for (_virtual_node, physical_node), resources in (
                solution["node_slots_info"].items()):
            for name, value in resources.items():
                physical_network.nodes[physical_node][name] -= value
        return True

    @staticmethod
    def release(virtual_network, physical_network, solution):
        if not solution["result"]:
            return False
        for virtual_node, physical_node in solution["node_slots"].items():
            resources = solution["node_slots_info"][(
                virtual_node, physical_node)]
            for name, value in resources.items():
                physical_network.nodes[physical_node][name] += value
        return True


class FakeRecorder:
    def __init__(self, counter):
        self.counter = counter
        self.if_temp_save_records = False
        self.temp_save_path = ""
        self.reset()

    def reset(self):
        self.state = {
            "v_net_count": 0,
            "success_count": 0,
            "inservice_count": 0,
        }
        self.memory = []
        self.v_net_event_dict = {}

    def count_init_p_net_info(self, physical_network):
        self.initial_resource = self.counter.calculate_sum_network_resource(
            physical_network)

    def update_state(self, values):
        self.state.update(values)

    def get_record(self, event_id=None, v_net_id=None):
        if event_id is None:
            event_id = self.v_net_event_dict[v_net_id]
        return self.memory[int(event_id)]

    def count(self, virtual_network, physical_network, solution):
        if self.state["event_type"] == 1:
            self.counter.count_solution(virtual_network, solution)
            self.v_net_event_dict[solution["v_net_id"]] = self.state["event_id"]
            self.state["v_net_count"] += 1
            if solution["result"]:
                self.state["success_count"] += 1
                self.state["inservice_count"] += 1
        elif self.state["event_type"] == 0:
            if self.get_record(v_net_id=solution["v_net_id"])["result"]:
                self.state["inservice_count"] -= 1
        else:
            raise RuntimeError("invalid fake event type")
        return {**copy.deepcopy(self.state), **solution.to_dict()}

    def add_record(self, record, extra_info):
        stored = copy.deepcopy(record)
        stored.update(copy.deepcopy(extra_info))
        self.memory.append(stored)
        return stored

    def summary_records(self, memory):
        return copy.deepcopy(self.state)


class FakeLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def critical(self, *args, **kwargs):
        return None


def make_environment(OracleEnvironment, request_count, admission=False):
    environment = object.__new__(OracleEnvironment)
    environment.controller = FakeController()
    environment.counter = FakeCounter()
    environment.recorder = FakeRecorder(environment.counter)
    environment.logger = FakeLogger()
    environment.p_net = FakePhysicalNetwork()
    environment.init_p_net = copy.deepcopy(environment.p_net)
    environment.v_net_simulator = FakeSimulator(request_count)
    environment.config = SimpleNamespace(
        experiment=SimpleNamespace(if_load_v_nets=False),
        simulation=SimpleNamespace(
            v_sim_setting_num_node_resource_attrs=1,
            v_sim_setting_num_link_resource_attrs=1,
        ),
    )
    environment.verbose = 0
    environment.p_net_dataset_dir = "unused-pnet-dataset"
    environment.v_nets_dataset_dir = "unused-vnet-dataset"
    environment.run_id = "run"
    environment.seed = None
    environment.solver_name = "oracle"
    environment.extra_summary_info = {}
    environment.extra_record_info = {}
    environment.r2c_ratio_threshold = 0.5 if admission else 0.0
    environment.vn_size_threshold = 0 if admission else 10000
    environment.get_observation = lambda: None
    environment.compute_reward = lambda: 0.0
    environment.get_info = lambda record={}: copy.deepcopy(record)
    environment.summary_records = lambda: environment.recorder.summary_records(
        environment.recorder.memory)
    environment.reset(seed=None)
    return environment


def accepted_solution(environment):
    solution = FakeSolution.from_v_net(environment.v_net)
    solution["result"] = True
    solution["node_slots"][0] = 0
    solution["node_slots_info"][(0, 0)] = {"cpu": 10.0}
    return solution


def rejected_solution(environment, reason):
    solution = FakeSolution.from_v_net(environment.v_net)
    if reason == "early":
        solution["early_rejection"] = True
        solution["place_result"] = False
        solution["route_result"] = False
    elif reason == "place":
        solution["place_result"] = False
        solution["route_result"] = False
    elif reason == "route":
        solution["route_result"] = False
    return solution


def double_token(value) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def record_token(record) -> str:
    return (
        f"{record['event_id']},{record['event_type']},"
        f"{record['v_net_id']},{1 if record['result'] else 0},"
        f"{record['description'].encode('utf-8').hex()}"
    )


def reason_token(environment, solution, accepted):
    if accepted:
        return "none"
    value = environment.get_failure_reason(solution)
    return {
        "reject": "early",
        "place": "place",
        "route": "route",
        "unknown": "unknown",
    }[value]


def run_step(environment, solution):
    memory_before = len(environment.recorder.memory)
    _observation, _reward, done, record = environment.step(solution)
    accepted = bool(record["result"])
    auto_released = len(environment.recorder.memory) - memory_before - 1
    step = (
        f"accepted={1 if accepted else 0},"
        f"reason={reason_token(environment, solution, accepted)},"
        f"done={1 if done else 0},"
        f"auto={auto_released},"
        f"summary={1 if done else 0},"
        f"record={record_token(record)}"
    )
    return step, done


def snapshot_token(environment, done):
    phase = "finished" if done else "active"
    current = "none"
    if not done:
        event = environment.curr_event
        current = (
            f"{event['id']},{event['id']},{event['type']},"
            f"{event['v_net_id']},{event['v_net_id']},"
            f"{double_token(event['time'])}"
        )
    state = environment.recorder.state
    records = "|".join(record_token(record)
                       for record in environment.recorder.memory)
    return (
        f"phase={phase},processed={environment.num_processed_v_nets},"
        f"current={current},"
        f"counts={state['v_net_count']},{state['success_count']},"
        f"{state['inservice_count']},"
        f"resources={double_token(environment.p_net.nodes[0]['cpu'])},"
        f"{double_token(environment.p_net.nodes[1]['cpu'])},"
        f"{double_token(environment.p_net.links[(0, 1)]['bandwidth'])},"
        f"memory={records}"
    )


def mixed_case(OracleEnvironment):
    environment = make_environment(OracleEnvironment, 2)
    first_step, first_done = run_step(
        environment, accepted_solution(environment))
    first = first_step + ";" + snapshot_token(environment, first_done)
    second_step, second_done = run_step(
        environment, rejected_solution(environment, "place"))
    second = second_step + ";" + snapshot_token(environment, second_done)
    return f"first{{{first}}};second{{{second}}}"


def rejection_case(OracleEnvironment, reason, hard=False, admission=False):
    environment = make_environment(
        OracleEnvironment, 1, admission=admission)
    if hard or admission:
        solution = FakeSolution.from_v_net(environment.v_net)
        solution["result"] = True
        if hard:
            solution["v_net_total_hard_constraint_violation"] = 1.0
    else:
        solution = rejected_solution(environment, reason)
    step, done = run_step(environment, solution)
    return step + ";" + snapshot_token(environment, done)


def python_cases(OracleEnvironment):
    mixed = mixed_case(OracleEnvironment)
    return {
        "mixed_workers_1": mixed,
        "mixed_workers_2": mixed_case(OracleEnvironment),
        "mixed_workers_8": mixed_case(OracleEnvironment),
        "reject_early": rejection_case(OracleEnvironment, "early"),
        "reject_place": rejection_case(OracleEnvironment, "place"),
        "reject_route": rejection_case(OracleEnvironment, "route"),
        "reject_unknown": rejection_case(OracleEnvironment, "unknown"),
        "reject_hard": rejection_case(
            OracleEnvironment, "unknown", hard=True),
        "reject_admission": rejection_case(
            OracleEnvironment, "unknown", admission=True),
    }


def cpp_cases(executable: pathlib.Path, root: pathlib.Path):
    process = subprocess.run(
        [str(executable), "differential", str(root)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Environment harness failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        name, payload = line.split("\t", 1)
        if name in result:
            raise RuntimeError(f"duplicate native case: {name}")
        result[name] = payload
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    OracleEnvironment = load_environment(args.source)
    expected = python_cases(OracleEnvironment)
    with tempfile.TemporaryDirectory(prefix="virne_environment_diff_") as tmp:
        actual = cpp_cases(args.harness, pathlib.Path(tmp) / "native")

    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, C++={list(actual)}")
    worker_payloads = [actual[f"mixed_workers_{value}"] for value in WORKERS]
    if len(set(worker_payloads)) != 1:
        raise RuntimeError("native Environment worker payload drift")
    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "Environment differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.BaseEnvironment",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "native_workers": list(WORKERS),
        "boundaries": BOUNDARIES,
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
