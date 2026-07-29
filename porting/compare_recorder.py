#!/usr/bin/env python3
"""Compact AST-isolated differential for the pinned non-ML Recorder leaf."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import tempfile
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from typing import Any, Dict


SOURCE_SHA256 = (
    "41e66702f29644e6f8a374ef909a1c41f87982558863be17b6fdf72ceb971f4a"
)

BOUNDARIES = {
    "rl_and_training": (
        "save_summary RL/feature/training naming and solver integration are "
        "intentionally outside this non-ML differential; the native core "
        "only exposes a cold typed summary extension seam."
    ),
    "reward": (
        "v_net_reward is covered only as an unchanged scalar carried by "
        "Solution/Recorder; no reward calculation or learning behavior is run."
    ),
    "csv_and_pandas": (
        "Pandas-backed arbitrary dynamic record schemas are a cold boundary; "
        "native fixed-schema CSV has its own unit gate."
    ),
    "native_workers": (
        "Python Recorder is scalar. Native workers 1/2/8 are required here "
        "to produce identical full state/history payloads."
    ),
}


class FakeOmegaConf:
    @staticmethod
    def create(value):
        return value


def load_recorder(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Recorder source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Recorder"
    ]
    if len(matches) != 1:
        raise RuntimeError("Recorder class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "recorder_oracle",
        "Any": Any,
        "Dict": Dict,
        "OmegaConf": FakeOmegaConf,
        "OrderedDict": OrderedDict,
        "PhysicalNetwork": object,
        "Solution": object,
        "VirtualNetwork": object,
        "copy": copy,
        "csv": csv,
        "defaultdict": defaultdict,
        "os": os,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Recorder"]


class FakePhysicalNetwork:
    node_total = 60.0
    link_total = 12.0


class FakeVirtualNetwork:
    node_total = 6.0
    link_total = 3.0
    lifetime = 1.0


class FakeSolution(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self):
        return dict(self)


class FakeCounter:
    """Pinned Counter leaf behavior used only as Recorder's dependency."""

    @staticmethod
    def calculate_sum_network_resource(network, node=True, link=True):
        return (
            (network.node_total if node else 0.0) +
            (network.link_total if link else 0.0)
        )

    @staticmethod
    def count_solution(v_net, solution):
        solution["num_placed_nodes"] = len(solution.node_slots)
        solution["num_routed_links"] = len(solution.link_paths)
        solution["v_net_node_demand"] = v_net.node_total
        solution["v_net_link_demand"] = v_net.link_total
        # This intentionally preserves the frozen Python Counter expression.
        solution["v_net_demand"] = (
            solution["v_net_node_demand"] + solution["v_net_demand"]
        )
        if solution["result"]:
            solution["place_result"] = True
            solution["route_result"] = True
            solution["early_rejection"] = False
            solution["v_net_node_revenue"] = solution["v_net_node_demand"]
            solution["v_net_link_revenue"] = solution["v_net_link_demand"]
            solution["v_net_node_cost"] = solution["v_net_node_revenue"]
            link_cost = 0.0
            for virtual_link, physical_links in solution["link_paths"].items():
                for physical_link in physical_links:
                    link_cost += solution["link_paths_info"][
                        (virtual_link, physical_link)
                    ]["bandwidth"]
            solution["v_net_link_cost"] = link_cost
            solution["v_net_path_cost"] = (
                solution["v_net_link_cost"] -
                solution["v_net_link_revenue"]
            )
            solution["v_net_revenue"] = (
                solution["v_net_node_revenue"] +
                solution["v_net_link_revenue"]
            )
            solution["v_net_cost"] = (
                solution["v_net_revenue"] +
                solution["v_net_path_cost"]
            )
            solution["v_net_r2c_ratio"] = (
                solution["v_net_revenue"] / solution["v_net_cost"]
                if solution["v_net_cost"] != 0 else 0.0
            )
        else:
            solution["v_net_node_revenue"] = 0.0
            solution["v_net_link_revenue"] = 0.0
            solution["v_net_revenue"] = 0.0
            solution["v_net_path_cost"] = 0.0
            solution["v_net_cost"] = 0.0
            solution["v_net_r2c_ratio"] = 0.0
        solution["v_net_time_revenue"] = (
            solution["v_net_revenue"] * v_net.lifetime
        )
        solution["v_net_time_cost"] = (
            solution["v_net_cost"] * v_net.lifetime
        )
        solution["v_net_time_rc_ratio"] = (
            solution["v_net_r2c_ratio"] * v_net.lifetime
        )
        return solution.to_dict()


def config(root: pathlib.Path, run_id: str):
    return SimpleNamespace(
        experiment=SimpleNamespace(
            save_root_dir=str(root),
            run_id=run_id,
        ),
        recorder=SimpleNamespace(
            if_temp_save_records=False,
            record_dir_name="records",
        ),
        solver=SimpleNamespace(solver_name="recorder-differential"),
    )


def seed_preserved_solution_fields(solution: FakeSolution) -> None:
    solution.update({
        "v_net_total_hard_constraint_violation": 0.25,
        "step_node": OrderedDict([(0, 1.25)]),
        "step_link": OrderedDict(),
        "step_path": OrderedDict(),
        "offset_node": [(0, OrderedDict([(0, -0.25)]))],
        "offset_link": [],
        "offset_path": [],
        "violation_node": [],
        "violation_link": [((0, 1), OrderedDict([(0, 0.5)]))],
        "violation_path": [],
        "v_net_single_step_violation_list": [True, -2, 1.5],
        "v_net_single_step_hard_constraint_offset": -0.5,
        "v_net_max_single_step_hard_constraint_violation": 2.25,
        "revoke_times": 2,
        "selected_actions": [4, 5],
        "num_interactions": 3,
        "num_attempt_times": 7,
    })


def base_solution(virtual_network_id: int, result: bool) -> FakeSolution:
    solution = FakeSolution({
        "v_net_id": virtual_network_id,
        "v_net_lifetime": 1.0,
        "v_net_arrival_time": float(virtual_network_id + 1),
        "v_net_num_nodes": 2,
        "v_net_num_edges": 1,
        "result": result,
        "node_slots": OrderedDict(),
        "link_paths": OrderedDict(),
        "node_slots_info": OrderedDict(),
        "link_paths_info": OrderedDict(),
        "v_net_cost": 0.0,
        "v_net_revenue": 0.0,
        "v_net_demand": 0.0,
        "v_net_node_demand": 0.0,
        "v_net_link_demand": 0.0,
        "v_net_node_revenue": 0.0,
        "v_net_link_revenue": 0.0,
        "v_net_node_cost": 0.0,
        "v_net_link_cost": 0.0,
        "v_net_path_cost": 0.0,
        "v_net_r2c_ratio": 0.0,
        "v_net_time_cost": 0.0,
        "v_net_time_revenue": 0.0,
        "v_net_time_rc_ratio": 0.0,
        "description": "original" if result else "",
        "place_result": True,
        "route_result": True,
        "early_rejection": False,
        "v_net_reward": float(virtual_network_id),
        "num_placed_nodes": None,
        "num_routed_links": None,
    })
    seed_preserved_solution_fields(solution)
    return solution


def make_success_solution(
    virtual_network_id: int,
    first_physical_node: int,
    second_physical_node: int,
    empty_route: bool = False,
) -> FakeSolution:
    solution = base_solution(virtual_network_id, True)
    solution["node_slots"][0] = first_physical_node
    solution["node_slots"][1] = second_physical_node
    virtual_link = (0, 1)
    if empty_route:
        solution["link_paths"][virtual_link] = []
        return solution
    physical_path = [(0, 1), (1, 2)]
    solution["link_paths"][virtual_link] = physical_path
    for physical_link in physical_path:
        solution["link_paths_info"][(virtual_link, physical_link)] = {
            "bandwidth": 3.0,
        }
    return solution


def make_failure_solution(virtual_network_id: int) -> FakeSolution:
    return base_solution(virtual_network_id, False)


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def number_token(value) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    return double_token(value)


def attribute_values_payload(values) -> str:
    return "[" + ",".join(
        f"{key}={number_token(value)}" for key, value in values.items()
    ) + "]"


def link_token(link) -> str:
    return f"{link[0]}:{link[1]}"


def node_slots_payload(slots) -> str:
    return "[" + ",".join(
        f"{virtual}={physical}"
        for virtual, physical in slots.items()
    ) + "]"


def link_paths_payload(paths) -> str:
    return "[" + ",".join(
        f"{link_token(virtual)}=[" + ",".join(
            link_token(physical) for physical in physical_path
        ) + "]"
        for virtual, physical_path in paths.items()
    ) + "]"


def link_info_payload(info) -> str:
    return "[" + ",".join(
        f"{link_token(key[0])}/{link_token(key[1])}=" +
        attribute_values_payload(OrderedDict([(0, values["bandwidth"])]))
        for key, values in info.items()
    ) + "]"


def table_payload(entries, link_keys: bool = False) -> str:
    return "[" + ",".join(
        f"{link_token(key) if link_keys else key}=" +
        attribute_values_payload(values)
        for key, values in entries
    ) + "]"


def optional_size_token(value) -> str:
    return "n" if value is None else f"i:{value}"


METRIC_FIELDS = [
    "v_net_cost",
    "v_net_revenue",
    "v_net_demand",
    "v_net_node_demand",
    "v_net_link_demand",
    "v_net_node_revenue",
    "v_net_link_revenue",
    "v_net_node_cost",
    "v_net_link_cost",
    "v_net_path_cost",
    "v_net_r2c_ratio",
    "v_net_time_cost",
    "v_net_time_revenue",
    "v_net_time_rc_ratio",
]


def solution_payload(value) -> str:
    return (
        f"meta={value['v_net_id']},"
        f"{double_token(value['v_net_lifetime'])},"
        f"{double_token(value['v_net_arrival_time'])},"
        f"{value['v_net_num_nodes']},{value['v_net_num_edges']}"
        f";result={1 if value['result'] else 0}"
        f";slots={node_slots_payload(value['node_slots'])}"
        f";paths={link_paths_payload(value['link_paths'])}"
        f";node_info={len(value['node_slots_info'])}"
        f";link_info={link_info_payload(value['link_paths_info'])}"
        ";metrics=" + ",".join(
            double_token(value[field]) for field in METRIC_FIELDS
        ) +
        f";description={value['description'].encode('utf-8').hex()}"
        f";hard={double_token(value['v_net_total_hard_constraint_violation'])}"
        f";step={attribute_values_payload(value['step_node'])}/"
        f"{attribute_values_payload(value['step_link'])}/"
        f"{attribute_values_payload(value['step_path'])}"
        f";offsets={table_payload(value['offset_node'])}/"
        f"{table_payload(value['offset_link'], True)}/"
        f"{table_payload(value['offset_path'], True)}"
        f";violations={table_payload(value['violation_node'])}/"
        f"{table_payload(value['violation_link'], True)}/"
        f"{table_payload(value['violation_path'], True)}"
        ";violation_list=[" + ",".join(
            number_token(item)
            for item in value["v_net_single_step_violation_list"]
        ) + "]"
        f";single={double_token(value['v_net_single_step_hard_constraint_offset'])},"
        f"{double_token(value['v_net_max_single_step_hard_constraint_violation'])}"
        f";flags={1 if value['place_result'] else 0}"
        f"{1 if value['route_result'] else 0}"
        f"{1 if value['early_rejection'] else 0}"
        f";revoke={value['revoke_times']}"
        ";actions=[" + ",".join(
            str(action) for action in value["selected_actions"]
        ) + "]"
        f";interactions={value['num_interactions']}"
        f";reward={double_token(value['v_net_reward'])}"
        f";counts={optional_size_token(value['num_placed_nodes'])},"
        f"{optional_size_token(value['num_routed_links'])},"
        f"{optional_size_token(value['num_attempt_times'])}"
    )


def state_payload(value) -> str:
    event = (
        f"{value['event_id']}:{value['event_type']}"
        if "event_id" in value else "n"
    )
    return (
        f"event={event}"
        f";counts={value['v_net_count']},{value['success_count']},"
        f"{value['inservice_count']}"
        f";totals={double_token(value['total_revenue'])},"
        f"{double_token(value['total_cost'])},"
        f"{double_token(value['total_time_revenue'])},"
        f"{double_token(value['total_time_cost'])}"
        f";ratios={double_token(value['long_term_r2c_ratio'])},"
        f"{double_token(value['long_term_time_r2c_ratio'])}"
        f";running={value['num_running_p_net_nodes']}"
        f";physical={double_token(value['p_net_available_resource'])},"
        f"{double_token(value['p_net_node_available_resource'])},"
        f"{double_token(value['p_net_link_available_resource'])}"
        f";utilization="
        f"{double_token(value['p_net_node_resource_utilization'])},"
        f"{double_token(value['p_net_link_resource_utilization'])}"
    )


def initial_payload(value) -> str:
    return (
        double_token(value["p_net_available_resource"]) + "," +
        double_token(value["p_net_node_available_resource"]) + "," +
        double_token(value["p_net_link_available_resource"])
    )


def nodes_payload(nodes) -> str:
    return "[" + ",".join(str(node) for node in nodes) + "]"


def record_payload(record) -> str:
    return "{" + state_payload(record) + "|" + solution_payload(record) + "}"


def flow_payload(Recorder, root: pathlib.Path, workers: int) -> str:
    del workers  # Python is scalar; the repeated oracle output is intentional.
    p_net = FakePhysicalNetwork()
    v_net = FakeVirtualNetwork()
    recorder = Recorder(FakeCounter(), config(root, "flow"))
    recorder.count_init_p_net_info(p_net)
    stages = []

    def append_stage(record):
        stages.append(
            record_payload(record) + "@" +
            nodes_payload(recorder.get_running_p_net_nodes())
        )

    recorder.update_state({"event_id": 0, "event_type": 1})
    first = make_success_solution(10, 2, 2)
    append_stage(recorder.add_record(recorder.count(v_net, p_net, first)))

    recorder.update_state({"event_id": 1, "event_type": 1})
    failure = make_failure_solution(20)
    append_stage(recorder.add_record(recorder.count(v_net, p_net, failure)))

    recorder.update_state({"event_id": 2, "event_type": 1})
    second = make_success_solution(30, 1, 2)
    append_stage(recorder.add_record(recorder.count(v_net, p_net, second)))

    lookup = (
        f"{recorder.get_record(event_id=-1)['v_net_id']},"
        f"{recorder.get_record(event_id=-3)['v_net_id']},"
        f"{1 if recorder.get_record(v_net_id=20)['result'] else 0}"
    )

    recorder.update_state({"event_id": 3, "event_type": 0})
    first_leave = make_success_solution(10, 2, 2)
    first_leave["result"] = False
    append_stage(recorder.add_record(
        recorder.count(v_net, p_net, first_leave)))

    recorder.update_state({"event_id": 4, "event_type": 0})
    failed_leave = make_failure_solution(20)
    failed_leave["result"] = True
    failed_leave["node_slots"][0] = 99
    append_stage(recorder.add_record(
        recorder.count(v_net, p_net, failed_leave)))

    recorder.update_state({"event_id": 5, "event_type": 0})
    second_leave = make_success_solution(30, 1, 2)
    second_leave["result"] = False
    append_stage(recorder.add_record(
        recorder.count(v_net, p_net, second_leave)))

    return (
        "initial=" + initial_payload(recorder.init_p_net_info) +
        ";stages=[" + "#".join(stages) + "]" +
        ";lookup=" + lookup +
        ";history=[" + "#".join(
            record_payload(record) for record in recorder.memory
        ) + "]" +
        ";final={" + state_payload(recorder.state) + "@" +
        nodes_payload(recorder.get_running_p_net_nodes()) + "}"
    )


def ratio_error_payload(Recorder, root: pathlib.Path, workers: int) -> str:
    del workers
    p_net = FakePhysicalNetwork()
    v_net = FakeVirtualNetwork()
    recorder = Recorder(FakeCounter(), config(root, "ratio"))
    recorder.count_init_p_net_info(p_net)
    recorder.update_state({"event_id": 0, "event_type": 1})
    solution = make_success_solution(53, 0, 1, empty_route=True)
    error = "none"
    try:
        recorder.count(v_net, p_net, solution)
    except AssertionError:
        error = "invalid_time_ratio"

    lookup = "record"
    try:
        recorder.get_record(v_net_id=53)
    except IndexError:
        lookup = "mapped_without_record"
    except KeyError:
        lookup = "unmapped"

    return (
        f"error={error};initial={initial_payload(recorder.init_p_net_info)}"
        f";state={{{state_payload(recorder.state)}}}"
        f";solution={{{solution_payload(solution)}}}"
        f";nodes={nodes_payload(recorder.get_running_p_net_nodes())}"
        f";memory={len(recorder.memory)};lookup={lookup}"
    )


def python_cases(Recorder):
    with tempfile.TemporaryDirectory(prefix="virne_recorder_oracle_") as temp:
        root = pathlib.Path(temp)
        cases = {}
        for workers in (1, 2, 8):
            cases[f"flow_workers_{workers}"] = flow_payload(
                Recorder, root, workers)
        for workers in (1, 2, 8):
            cases[f"ratio_error_workers_{workers}"] = ratio_error_payload(
                Recorder, root, workers)
        return cases


def cpp_cases(harness: pathlib.Path):
    process = subprocess.run(
        [str(harness.resolve()), "differential"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Recorder harness failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        name, payload = line.split("\t", 1)
        if name in result:
            raise RuntimeError(f"duplicate native case: {name}")
        result[name] = payload
    return result


def require_worker_parity(cases, prefix: str) -> None:
    values = [cases[f"{prefix}_workers_{workers}"] for workers in (1, 2, 8)]
    if len(set(values)) != 1:
        raise RuntimeError(f"native worker parity failed for {prefix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Recorder = load_recorder(args.source)
    expected = python_cases(Recorder)
    actual = cpp_cases(args.harness)
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, "
            f"C++={list(actual)}")
    require_worker_parity(actual, "flow")
    require_worker_parity(actual, "ratio_error")
    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected
        if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "Recorder differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.Recorder.non_ml",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "native_worker_counts": [1, 2, 8],
        "python_only_boundaries": BOUNDARIES,
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
