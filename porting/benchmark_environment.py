#!/usr/bin/env python3
"""Compact mixed lifecycle benchmark for the pinned non-RL Environment."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pathlib
import statistics
import struct
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union


SOURCE_SHA256 = "6004cff2114e504e30c5490232763f71cb9e7799216c4f6ce8badc27a0e42b34"
PHYSICAL_NODE_COUNT = 4096
PHYSICAL_LINK_COUNT = PHYSICAL_NODE_COUNT - 1
VIRTUAL_NODE_COUNT = 8
REQUEST_COUNT = 96
EVENT_COUNT = REQUEST_COUNT * 2
WARMUP_COUNT = 1
SAMPLE_COUNT = 3
WORKERS = (1, 2, 8)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1


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
    steps = [
        node for node in classes["SolutionStepEnvironment"].body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    ]
    if len(steps) != 1:
        raise RuntimeError("SolutionStepEnvironment.step inventory drift")

    isolated_class = ast.ClassDef(
        name="OracleEnvironment",
        bases=[],
        keywords=[],
        body=[base_methods[name] for name in base_names] + steps,
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
        "Solution": OracleSolution,
        "copy": copy,
        "get_v_nets_dataset_dir_from_setting": lambda setting, seed=None:
            "unused-vnet-dataset",
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda path: False)),
        "time": SimpleNamespace(time=lambda: 1234.5),
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["OracleEnvironment"]


def node_capacity(index: int) -> float:
    return 10000.0 + (index % 31) * 0.25


def link_capacity(index: int) -> float:
    return 20000.0 + (index % 29) * 0.5


def node_demand(request: int, index: int) -> float:
    return 0.5 + ((request + index) % 8) * 0.125


def link_demand(request: int, index: int) -> float:
    return 0.25 + ((request + index) % 8) * 0.0625


def request_is_accepted(request: int) -> bool:
    return request % 3 != 1


EXPECTED_ACCEPTED = sum(
    request_is_accepted(request) for request in range(REQUEST_COUNT))


class OracleSolution(dict):
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
            "v_net_node_revenue": 0.0,
            "v_net_link_revenue": 0.0,
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


class OracleVirtualNetwork:
    def __init__(self, request: int):
        self.id = request
        self.arrival_time = float(request)
        self.lifetime = 8.5
        self.num_nodes = VIRTUAL_NODE_COUNT
        self.num_links = VIRTUAL_NODE_COUNT - 1
        self.node_demands = [
            node_demand(request, index)
            for index in range(VIRTUAL_NODE_COUNT)
        ]
        self.link_demands = [
            link_demand(request, index)
            for index in range(VIRTUAL_NODE_COUNT - 1)
        ]


class OraclePhysicalNetwork:
    def __init__(self):
        self.nodes = [
            {"cpu": node_capacity(index)}
            for index in range(PHYSICAL_NODE_COUNT)
        ]
        self.links = {
            (index, index + 1): {
                "bandwidth": link_capacity(index)}
            for index in range(PHYSICAL_LINK_COUNT)
        }


class OracleSimulator:
    def __init__(self):
        self.v_nets = [
            OracleVirtualNetwork(request) for request in range(REQUEST_COUNT)
        ]
        scheduled = []
        for request in range(REQUEST_COUNT):
            scheduled.append((float(request), 1, request))
            scheduled.append((float(request) + 8.5, 0, request))
        scheduled.sort(key=lambda item: item[0])
        self.events = [
            {"id": index, "type": event_type, "v_net_id": request,
             "time": event_time}
            for index, (event_time, event_type, request)
            in enumerate(scheduled)
        ]
        self.num_events = len(self.events)
        self.v_sim_setting = object()

    def renew(self, v_nets=True, events=True, seed=None):
        return None


class OracleCounter:
    @staticmethod
    def calculate_sum_node_resource(physical_network):
        return sum(node["cpu"] for node in physical_network.nodes)

    @staticmethod
    def calculate_sum_link_resource(physical_network):
        return sum(
            physical_network.links[
                (index, index + 1)]["bandwidth"]
            for index in range(PHYSICAL_LINK_COUNT)
        )

    def calculate_sum_network_resource(
            self, physical_network, node=True, link=True):
        total = 0.0
        if node:
            total += self.calculate_sum_node_resource(physical_network)
        if link:
            total += self.calculate_sum_link_resource(physical_network)
        return total

    def count_solution(self, virtual_network, solution):
        node_cost = sum(
            float(value)
            for resources in solution["node_slots_info"].values()
            for value in resources.values()
        )
        link_cost = sum(
            float(value)
            for resources in solution["link_paths_info"].values()
            for value in resources.values()
        )
        node_revenue = sum(virtual_network.node_demands)
        link_revenue = sum(virtual_network.link_demands)
        cost = node_cost + link_cost
        revenue = node_revenue + link_revenue
        solution["v_net_node_cost"] = node_cost
        solution["v_net_link_cost"] = link_cost
        solution["v_net_cost"] = cost
        solution["v_net_node_revenue"] = node_revenue
        solution["v_net_link_revenue"] = link_revenue
        solution["v_net_revenue"] = revenue
        solution["v_net_time_cost"] = cost * virtual_network.lifetime
        solution["v_net_time_revenue"] = revenue * virtual_network.lifetime
        solution["v_net_r2c_ratio"] = revenue / cost if cost else 0.0
        return solution.to_dict()


class OracleController:
    @staticmethod
    def deploy(virtual_network, physical_network, solution):
        if not solution["result"]:
            return False
        for (_virtual_node, physical_node), resources in (
                solution["node_slots_info"].items()):
            for name, value in resources.items():
                physical_network.nodes[physical_node][name] -= value
        for (_virtual_link, physical_link), resources in (
                solution["link_paths_info"].items()):
            for name, value in resources.items():
                physical_network.links[physical_link][name] -= value
        return True

    @staticmethod
    def release(virtual_network, physical_network, solution):
        if not solution["result"]:
            return False
        for (_virtual_node, physical_node), resources in (
                solution["node_slots_info"].items()):
            for name, value in resources.items():
                physical_network.nodes[physical_node][name] += value
        for (_virtual_link, physical_link), resources in (
                solution["link_paths_info"].items()):
            for name, value in resources.items():
                physical_network.links[physical_link][name] += value
        return True


class OracleRecorder:
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
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "physical_available_resource": 0.0,
            "physical_node_available_resource": 0.0,
            "physical_link_available_resource": 0.0,
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
        self.state["physical_available_resource"] = (
            self.counter.calculate_sum_network_resource(physical_network))
        self.state["physical_node_available_resource"] = (
            self.counter.calculate_sum_node_resource(physical_network))
        self.state["physical_link_available_resource"] = (
            self.counter.calculate_sum_link_resource(physical_network))
        if self.state["event_type"] == 1:
            self.counter.count_solution(virtual_network, solution)
            self.v_net_event_dict[solution["v_net_id"]] = (
                self.state["event_id"])
            self.state["v_net_count"] += 1
            if solution["result"]:
                self.state["success_count"] += 1
                self.state["inservice_count"] += 1
                self.state["total_revenue"] += solution["v_net_revenue"]
                self.state["total_cost"] += solution["v_net_cost"]
        elif self.state["event_type"] == 0:
            if self.get_record(v_net_id=solution["v_net_id"])["result"]:
                self.state["inservice_count"] -= 1
        else:
            raise RuntimeError("invalid oracle event type")
        return {**copy.deepcopy(self.state), **solution.to_dict()}

    def add_record(self, record, extra_info):
        stored = copy.deepcopy(record)
        stored.update(copy.deepcopy(extra_info))
        self.memory.append(stored)
        return stored

    def summary_records(self, memory):
        return copy.deepcopy(self.state)


class OracleLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def critical(self, *args, **kwargs):
        return None


def make_environment(OracleEnvironment):
    environment = object.__new__(OracleEnvironment)
    environment.controller = OracleController()
    environment.counter = OracleCounter()
    environment.recorder = OracleRecorder(environment.counter)
    environment.logger = OracleLogger()
    environment.p_net = OraclePhysicalNetwork()
    environment.init_p_net = copy.deepcopy(environment.p_net)
    environment.v_net_simulator = OracleSimulator()
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
    environment.r2c_ratio_threshold = 0.0
    environment.vn_size_threshold = 10000
    environment.get_observation = lambda: None
    environment.compute_reward = lambda: 0.0
    environment.get_info = lambda record={}: copy.deepcopy(record)
    environment.summary_records = lambda: environment.recorder.summary_records(
        environment.recorder.memory)
    environment.reset(seed=None)
    return environment


def make_solution(virtual_network, request: int):
    solution = OracleSolution.from_v_net(virtual_network)
    if not request_is_accepted(request):
        solution["place_result"] = False
        solution["route_result"] = False
        return solution

    solution["result"] = True
    offset = (request * 13) % (PHYSICAL_NODE_COUNT - VIRTUAL_NODE_COUNT)
    for index in range(VIRTUAL_NODE_COUNT):
        physical_node = offset + index
        solution["node_slots"][index] = physical_node
        solution["node_slots_info"][(index, physical_node)] = {
            "cpu": node_demand(request, index)}
    for index in range(VIRTUAL_NODE_COUNT - 1):
        virtual_link = (index, index + 1)
        physical_link = (
            offset + index,
            offset + index + 1,
        )
        solution["link_paths"][virtual_link] = [physical_link]
        solution["link_paths_info"][(virtual_link, physical_link)] = {
            "bandwidth": link_demand(request, index)}
    return solution


def make_input_templates(environment):
    return [
        make_solution(environment.v_net_simulator.v_nets[request], request)
        for request in range(REQUEST_COUNT)
    ]


class OutputDigest:
    def __init__(self):
        self.checksum = FNV_OFFSET
        self.bytes = 0

    def append_bytes(self, value: bytes):
        for byte in value:
            self.checksum ^= byte
            self.checksum = (self.checksum * FNV_PRIME) & U64_MASK
            self.bytes += 1

    def append_byte(self, value: int):
        self.append_bytes(bytes((value,)))

    def append_u64(self, value: int):
        self.append_bytes(int(value & U64_MASK).to_bytes(8, "little"))

    def append_i64(self, value: int):
        self.append_u64(value)

    def append_double(self, value: float):
        self.append_bytes(struct.pack("<d", float(value)))

    def append_string(self, value: str):
        encoded = value.encode("utf-8")
        self.append_u64(len(encoded))
        self.append_bytes(encoded)


def append_physical(digest: OutputDigest, physical_network):
    for index in range(PHYSICAL_NODE_COUNT):
        digest.append_double(physical_network.nodes[index]["cpu"])
    for index in range(PHYSICAL_LINK_COUNT):
        digest.append_double(physical_network.links[
            (index, index + 1)]["bandwidth"])


def physical_digest(physical_network):
    digest = OutputDigest()
    append_physical(digest, physical_network)
    return digest


def episode_output(environment, accepted: int):
    state = environment.recorder.state
    if (environment.num_processed_v_nets != EVENT_COUNT or
            len(environment.recorder.memory) != EVENT_COUNT or
            state["v_net_count"] != REQUEST_COUNT or
            state["success_count"] != EXPECTED_ACCEPTED or
            state["inservice_count"] != 0 or
            accepted != EXPECTED_ACCEPTED):
        raise RuntimeError("Python Environment final state gate mismatch")
    for index in range(PHYSICAL_NODE_COUNT):
        if environment.p_net.nodes[index]["cpu"] != node_capacity(index):
            raise RuntimeError("Python final physical node drift")
        if index < PHYSICAL_LINK_COUNT:
            link = (index, index + 1)
            if (environment.p_net.links[link]["bandwidth"] !=
                    link_capacity(index)):
                raise RuntimeError("Python final physical link drift")

    physical = physical_digest(environment.p_net)
    output = OutputDigest()
    output.append_byte(2)
    output.append_u64(environment.num_processed_v_nets)
    output.append_i64(state["v_net_count"])
    output.append_i64(state["success_count"])
    output.append_i64(state["inservice_count"])
    output.append_u64(len(environment.recorder.memory))
    append_physical(output, environment.p_net)
    for record in environment.recorder.memory:
        output.append_i64(record["event_id"])
        output.append_byte(record["event_type"])
        output.append_i64(record["v_net_id"])
        output.append_byte(1 if record["result"] else 0)
        output.append_string(record["description"])
    return {
        "checksum": output.checksum,
        "physical_checksum": physical.checksum,
        "output_bytes": output.bytes,
        "records": len(environment.recorder.memory),
        "accepted": accepted,
    }


def run_episode(environment, inputs):
    accepted = 0
    for request, solution in enumerate(inputs):
        _observation, _reward, done, record = environment.step(solution)
        actual = bool(record["result"])
        expected = request_is_accepted(request)
        if actual != expected or done != (request + 1 == REQUEST_COUNT):
            raise RuntimeError("Python Environment step gate mismatch")
        accepted += int(actual)
    return accepted


def require_same_output(expected, actual, stage: str):
    if expected != actual:
        raise RuntimeError(
            f"Python Environment output drift at {stage}: "
            f"{expected!r} != {actual!r}")


def run_python_benchmark(OracleEnvironment):
    environment = make_environment(OracleEnvironment)
    templates = make_input_templates(environment)

    gate_inputs = copy.deepcopy(templates)
    expected = episode_output(
        environment, run_episode(environment, gate_inputs))

    for _ in range(WARMUP_COUNT):
        environment.reset(seed=None)
        inputs = copy.deepcopy(templates)
        require_same_output(
            expected,
            episode_output(environment, run_episode(environment, inputs)),
            "warmup",
        )

    samples = []
    for sample in range(SAMPLE_COUNT):
        environment.reset(seed=None)
        inputs = copy.deepcopy(templates)
        begin = time.perf_counter_ns()
        accepted = run_episode(environment, inputs)
        elapsed = time.perf_counter_ns() - begin
        actual = episode_output(environment, accepted)
        require_same_output(expected, actual, f"sample-{sample}")
        samples.append(elapsed / 1_000_000.0)
    return {**expected, "median_ms": statistics.median(samples)}


def parse_native_row(line: str):
    fields = {}
    for field in line.split(";"):
        name, value = field.split("=", 1)
        fields[name] = value
    expected_names = [
        "workers", "requests", "events", "accepted", "records",
        "checksum", "physical_checksum", "output_bytes", "median_ms",
    ]
    if list(fields) != expected_names:
        raise RuntimeError(f"native row schema drift: {list(fields)}")
    return {
        "workers": int(fields["workers"]),
        "requests": int(fields["requests"]),
        "events": int(fields["events"]),
        "accepted": int(fields["accepted"]),
        "records": int(fields["records"]),
        "checksum": int(fields["checksum"]),
        "physical_checksum": int(fields["physical_checksum"]),
        "output_bytes": int(fields["output_bytes"]),
        "median_ms": float(fields["median_ms"]),
    }


def run_native_benchmark(executable: pathlib.Path, root: pathlib.Path):
    process = subprocess.run(
        [str(executable), "benchmark", str(root)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Environment benchmark failed: {process.stderr.strip()}")
    rows = [parse_native_row(line) for line in process.stdout.splitlines()]
    if [row["workers"] for row in rows] != list(WORKERS):
        raise RuntimeError("native Environment worker inventory drift")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    OracleEnvironment = load_environment(args.source)
    python_result = run_python_benchmark(OracleEnvironment)
    with tempfile.TemporaryDirectory(prefix="virne_environment_bench_") as tmp:
        native_rows = run_native_benchmark(
            args.benchmark, pathlib.Path(tmp) / "native")

    exact_fields = [
        "requests", "events", "accepted", "records", "checksum",
        "physical_checksum", "output_bytes",
    ]
    expected = {
        "requests": REQUEST_COUNT,
        "events": EVENT_COUNT,
        **{name: python_result[name] for name in exact_fields[2:]},
    }
    for row in native_rows:
        actual = {name: row[name] for name in exact_fields}
        if actual != expected:
            raise RuntimeError(
                f"Environment workers={row['workers']} exact gate mismatch: "
                f"{actual!r} != {expected!r}")
        if not 0.0 < row["median_ms"] < python_result["median_ms"]:
            raise RuntimeError(
                f"Environment workers={row['workers']} did not beat Python")
        row["speedup"] = python_result["median_ms"] / row["median_ms"]

    report = {
        "component": "core.BaseEnvironment",
        "source_sha256": SOURCE_SHA256.upper(),
        "workload": "mixed_accepted_rejected_overlapping_leave_lifecycle",
        "physical_nodes": PHYSICAL_NODE_COUNT,
        "physical_links": PHYSICAL_LINK_COUNT,
        "virtual_nodes_per_request": VIRTUAL_NODE_COUNT,
        "virtual_links_per_request": VIRTUAL_NODE_COUNT - 1,
        "requests": REQUEST_COUNT,
        "events": EVENT_COUNT,
        "accepted": EXPECTED_ACCEPTED,
        "warmup": WARMUP_COUNT,
        "samples": SAMPLE_COUNT,
        "records": python_result["records"],
        "checksum": python_result["checksum"],
        "physical_checksum": python_result["physical_checksum"],
        "output_bytes": python_result["output_bytes"],
        "python_median_ms": python_result["median_ms"],
        "native": native_rows,
        "timing_boundary": (
            "Environment arrival admission/count/deploy/record/transit and "
            "automatic leave release/record only; fixture construction, "
            "reset/preparation, input cloning, process startup and all "
            "checksum/output gates are outside the timer."),
        "oracle_boundary": (
            "Exact SHA-locked AST-isolated Python Environment lifecycle over "
            "deterministic collaborators that perform node/link resource "
            "sums, deploy/release mutations, counter fields, record state and "
            "deep history snapshots. Logging, filesystem persistence, "
            "observation/reward, solver, system, RL and ML are excluded."),
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
