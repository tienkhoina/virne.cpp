#!/usr/bin/env python3
"""AST-isolated exact differential for the pinned Solution leaf."""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import hashlib
import json
import pathlib
import pprint
import struct
import subprocess
from types import SimpleNamespace
from typing import Any


SOURCE_SHA256 = "ec5d64ef6695350f718a818461fbf60934b5077d8add39bb7ee8be18cfe1e78b"
BOUNDARIES = {
    "arbitrary_reflection":
        "Python update/setattr can add or type-change arbitrary fields; native fixed fields are typed and direct.",
    "method_shadowing":
        "Python instance attributes may shadow methods and dunder behavior; native methods cannot be replaced by data.",
    "unbounded_python_scalars":
        "Python accepts arbitrary ID/integer/object payloads; native IDs are int64/size_t and attribute values use the frozen numeric lanes.",
    "dynamic_attribute_names":
        "Python resource and constraint dictionaries use names; native boundaries resolve each name once to a registry ID and hot access uses direct slots.",
}


class ClassDict:
    def __init__(self):
        pass

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def load_solution_class(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Solution source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Solution"
    ]
    if len(matches) != 1:
        raise RuntimeError("Solution class inventory drift")
    isolated = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "ClassDict": ClassDict,
        "OrderedDict": OrderedDict,
        "pprint": pprint,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Solution"]


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def number_token(value) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    return double_token(value)


ATTR_IDS = {"cpu": 0, "latency": 1, "bw": 2}


def attribute_values_payload(values) -> str:
    return "[" + ",".join(
        f"{ATTR_IDS[name]}={number_token(value)}"
        for name, value in sorted(values.items(), key=lambda item: ATTR_IDS[item[0]])
    ) + "]"


def link_token(link) -> str:
    return f"{link[0]}:{link[1]}"


def node_slots_payload(values) -> str:
    return "[" + ",".join(f"{key}={value}" for key, value in values.items()) + "]"


def link_paths_payload(values) -> str:
    return "[" + ",".join(
        f"{link_token(key)}=[{','.join(link_token(link) for link in path)}]"
        for key, path in values.items()
    ) + "]"


def node_info_payload(values) -> str:
    return "[" + ",".join(
        f"{key[0]}:{key[1]}={attribute_values_payload(value)}"
        for key, value in values.items()
    ) + "]"


def link_info_payload(values) -> str:
    return "[" + ",".join(
        f"{link_token(key[0])}/{link_token(key[1])}={attribute_values_payload(value)}"
        for key, value in values.items()
    ) + "]"


def constraint_table_payload(values, link_keys: bool) -> str:
    return "[" + ",".join(
        f"{link_token(key) if link_keys else key}={attribute_values_payload(value)}"
        for key, value in values.items()
    ) + "]"


def optional_int(solution, name: str) -> str:
    return f"i:{getattr(solution, name)}" if hasattr(solution, name) else "n"


def snapshot(solution) -> str:
    metrics = [
        solution.v_net_cost,
        solution.v_net_revenue,
        solution.v_net_demand,
        solution.v_net_node_demand,
        solution.v_net_link_demand,
        solution.v_net_node_revenue,
        solution.v_net_link_revenue,
        solution.v_net_node_cost,
        solution.v_net_link_cost,
        solution.v_net_path_cost,
        solution.v_net_r2c_ratio,
        solution.v_net_time_cost,
        solution.v_net_time_revenue,
        solution.v_net_time_rc_ratio,
    ]
    step = solution.v_net_single_step_constraint_offset
    offsets = solution.v_net_constraint_offsets
    violations = solution.v_net_constraint_violations
    return (
        f"id=i:{solution.v_net_id}"
        f";life={double_token(solution.v_net_lifetime)}"
        f";arrival={double_token(solution.v_net_arrival_time)}"
        f";nodes=i:{solution.v_net_num_nodes}"
        f";egdes=i:{solution.v_net_num_egdes}"
        f";result={'b:1' if solution.result else 'b:0'}"
        f";node_slots={node_slots_payload(solution.node_slots)}"
        f";link_paths={link_paths_payload(solution.link_paths)}"
        f";node_info={node_info_payload(solution.node_slots_info)}"
        f";link_info={link_info_payload(solution.link_paths_info)}"
        f";metrics=[{','.join(double_token(value) for value in metrics)}]"
        f";description=s:{solution.description.encode('utf-8').hex()}"
        f";total={double_token(solution.v_net_total_hard_constraint_violation)}"
        f";step={attribute_values_payload(step['node_level'])}/"
        f"{attribute_values_payload(step['link_level'])}/"
        f"{attribute_values_payload(step['path_level'])}"
        f";offsets={constraint_table_payload(offsets['node_level'], False)}/"
        f"{constraint_table_payload(offsets['link_level'], True)}/"
        f"{constraint_table_payload(offsets['path_level'], True)}"
        f";violations={constraint_table_payload(violations['node_level'], False)}/"
        f"{constraint_table_payload(violations['link_level'], True)}/"
        f"{constraint_table_payload(violations['path_level'], True)}"
        f";step_list=[{','.join(number_token(value) for value in solution.v_net_single_step_violation_list)}]"
        f";step_hard={double_token(solution.v_net_single_step_hard_constraint_offset)}"
        f";max_hard={double_token(solution.v_net_max_single_step_hard_constraint_violation)}"
        f";place={'b:1' if solution.place_result else 'b:0'}"
        f";route={'b:1' if solution.route_result else 'b:0'}"
        f";early={'b:1' if solution.early_rejection else 'b:0'}"
        f";revoke=i:{solution.revoke_times}"
        f";actions=[{','.join(number_token(value) for value in solution.selected_actions)}]"
        f";interactions=i:{solution.num_interactions}"
        f";reward={double_token(solution.v_net_reward)}"
        f";later={optional_int(solution, 'num_placed_nodes')},"
        f"{optional_int(solution, 'num_routed_links')},"
        f"{optional_int(solution, 'num_attempt_times')}"
    )


def vnet(**kwargs):
    return SimpleNamespace(**kwargs)


def missing_field_payload(Solution, field: str) -> str:
    values = {}
    if field != "id":
        values["id"] = 7
    if field == "arrival_time":
        values["lifetime"] = 3.5
    try:
        Solution(vnet(**values))
    except AttributeError:
        return f"field={field}"
    return "field=none"


def batch_payload(Solution) -> str:
    solutions = [
        Solution(vnet(
            id=index,
            lifetime=1.0 + index,
            arrival_time=index * 0.25,
            num_nodes=2 + index,
            num_links=1 + index,
        ))
        for index in range(64)
    ]
    return "[" + ",".join(
        f"{solution.v_net_id}:{double_token(solution.v_net_lifetime)}:"
        f"{double_token(solution.v_net_arrival_time)}:{solution.v_net_num_nodes}:"
        f"{solution.v_net_num_egdes}:{'1' if solution.is_feasible() else '0'}"
        for solution in solutions
    ) + "]"


def oracle_cases(Solution) -> list[tuple[str, str]]:
    ordinary = Solution(vnet(
        id=7, lifetime=3.5, arrival_time=-0.0, num_nodes=4, num_links=3,
    ))
    cases = [("construction", snapshot(ordinary)), ("repr", repr(ordinary))]
    for name, result, violation in [
        ("feasible_false", False, 0.0),
        ("feasible_zero", True, 0.0),
        ("feasible_negative", True, -1.0),
        ("feasible_positive", True, 0.25),
        ("feasible_nan", True, float("nan")),
    ]:
        ordinary.result = result
        ordinary.v_net_total_hard_constraint_violation = violation
        cases.append((name, "b:1" if ordinary.is_feasible() else "b:0"))
    ordinary.reset()

    ordinary.node_slots[9] = 90
    ordinary.node_slots[2] = 20
    ordinary.node_slots[9] = 91
    ordinary.link_paths[(0, 1)] = [(91, 12), (12, 20)]
    ordinary.node_slots_info[(9, 91)] = {"cpu": 7, "bw": 2.5}
    ordinary.link_paths_info[((0, 1), (91, 12))] = {"latency": True}
    ordinary.v_net_single_step_constraint_offset["node_level"]["cpu"] = -1.0
    ordinary.v_net_single_step_constraint_offset["link_level"]["bw"] = 3.0
    ordinary.v_net_constraint_offsets["node_level"][9] = {"cpu": 7, "bw": 2.5}
    ordinary.v_net_constraint_offsets["link_level"][(0, 1)] = {"latency": True}
    ordinary.v_net_constraint_offsets["path_level"][(0, 1)] = {"cpu": 7, "bw": 2.5}
    ordinary.v_net_constraint_violations["node_level"][9] = {"latency": True}
    ordinary.v_net_constraint_violations["link_level"][(0, 1)] = {"cpu": 7, "bw": 2.5}
    ordinary.v_net_constraint_violations["path_level"][(0, 1)] = {"latency": True}
    ordinary.v_net_single_step_violation_list = [-1, 0.5, True]
    ordinary.selected_actions = [4, 8]
    cases.append(("typed_mappings", snapshot(ordinary)))

    ordinary.result = True
    ordinary.v_net_cost = 9.5
    ordinary.description = "dirty"
    ordinary.place_result = False
    ordinary.num_placed_nodes = 2
    ordinary.num_routed_links = 1
    ordinary.num_attempt_times = 5
    ordinary.reset()
    cases.append(("reset_dirty", snapshot(ordinary)))

    ordinary.update({
        "result": True,
        "v_net_cost": 11.25,
        "description": "typed update",
        "early_rejection": True,
    })
    cases.append(("typed_update", snapshot(ordinary)))
    cases.extend([
        ("missing_id", missing_field_payload(Solution, "id")),
        ("missing_lifetime", missing_field_payload(Solution, "lifetime")),
        ("missing_arrival", missing_field_payload(Solution, "arrival_time")),
    ])
    payload = batch_payload(Solution)
    for workers in (0, 1, 2, 8):
        cases.append((f"batch_w{workers}", payload))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        [str(path)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Solution harness failed: {process.stderr.strip()}")
    result = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].startswith("case=") or fields[1] != "ok":
            raise RuntimeError(f"malformed Solution line: {line!r}")
        result.append((fields[0][5:], bytes.fromhex(fields[2]).decode("utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Solution = load_solution_class(args.source)
    expected = oracle_cases(Solution)
    actual = parse_cpp(args.harness.resolve())
    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"Solution mismatch {wanted[0]}: C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"Solution inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "shared_cases": [name for name, _ in expected],
        "native_extension_cases": [
            "compact_numeric_entry_ids",
            "direct_attribute_registry_slots",
        ],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + 2 + len(BOUNDARIES),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"Solution differential: PASS ({len(expected)} shared + 2 native + "
        f"{len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
