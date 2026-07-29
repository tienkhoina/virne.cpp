#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python NodeMapper leaf."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any


SOURCE_SHA256 = "8ad2ae077e61732dcb77ffa08269aab630bdb6df29969c983d09a3565ba860f9"

BOUNDARIES = {
    "dynamic_protocols":
        "Python accepts arbitrary mappings, custom numeric objects, monkey patches, and unbounded node labels; native preparation accepts frozen typed networks and values.",
    "invalid_mapping_inputs":
        "Python exposes assertion or unbound-local failures for invalid methods and empty/incomplete candidates; native reports deterministic typed errors.",
    "native_candidate_workers":
        "Configured candidate workers preserve Python candidate order but are a native throughput extension; Python evaluates candidates sequentially.",
    "unsupported_unsafe_mapping":
        "Both implementations reject allow-violation whole-network mapping; native reports a typed unsupported-operation error before mutation.",
}


def load_mapper(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"NodeMapper source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NodeMapper"
    ]
    if len(matches) != 1:
        raise RuntimeError("NodeMapper class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Callable": Any,
        "ConstraintChecker": Any,
        "Dict": dict,
        "List": list,
        "Optional": Any,
        "PhysicalNetwork": Any,
        "ResourceUpdator": Any,
        "Solution": Any,
        "Tuple": tuple,
        "Union": Any,
        "VirtualNetwork": Any,
        "copy": copy,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["NodeMapper"]


class NodeView:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, node):
        return self.values[node]


class FakeNetwork:
    def __init__(self, values):
        self.nodes = NodeView(values)
        self.num_nodes = len(values)


class FakeNodeResource:
    def __init__(self, name: str, soft: bool = False):
        self.name = name
        self.soft = soft

    def check_constraint_satisfiability(self, virtual, physical):
        offset = virtual[self.name] - physical[self.name]
        return self.soft or offset <= 0, offset


class FakeConstraintChecker:
    def __init__(self, attrs):
        self.attrs = list(attrs)

    def check_node_level_constraints(
            self, virtual, physical, virtual_node, physical_node):
        virtual_values = virtual.nodes[virtual_node]
        physical_values = physical.nodes[physical_node]
        feasible = True
        offsets = {}
        for attr in self.attrs:
            flag, offset = attr.check_constraint_satisfiability(
                virtual_values, physical_values)
            feasible = flag and feasible
            offsets[attr.name] = offset
        return feasible, offsets


class FakeResourceUpdator:
    @staticmethod
    def update_node_resources(
            physical, physical_node, resources, operator, safe=True):
        target = physical.nodes[physical_node]
        for name, value in resources.items():
            if operator in ("-", "sub"):
                if safe:
                    assert target[name] >= value
                target[name] -= value
            elif operator in ("+", "add"):
                target[name] += value
            else:
                raise NotImplementedError


CPU_ID = 0
QUALITY_ID = 1
CONSTRAINT_IDS = {"cpu": CPU_ID, "quality": QUALITY_ID}
RESOURCE_IDS = {"cpu": CPU_ID}


def fixture(NodeMapper):
    virtual = FakeNetwork({
        0: {"cpu": 4, "quality": 5.0},
        1: {"cpu": 6, "quality": 1.0},
    })
    physical = FakeNetwork({
        0: {"cpu": 5, "quality": 2.0},
        1: {"cpu": 3, "quality": 10.0},
        2: {"cpu": 10, "quality": 10.0},
        3: {"cpu": 1, "quality": 1.0},
    })
    cpu = FakeNodeResource("cpu")
    quality = FakeNodeResource("quality", soft=True)
    mapper = NodeMapper(
        FakeConstraintChecker([cpu, quality]),
        FakeResourceUpdator(),
        [cpu],
        ["cpu"])
    return mapper, virtual, physical


def make_solution():
    return {
        "node_slots": {},
        "node_slots_info": {},
        "v_net_constraint_offsets": {"node_level": {}},
        "v_net_constraint_violations": {"node_level": {}},
        "v_net_total_hard_constraint_violation": 0.0,
        "place_result": True,
        "result": True,
    }


def number_token(value) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    return "d:" + struct.pack(">d", float(value)).hex()


def values_payload(values, ids) -> str:
    return "{" + ",".join(
        f"{ids[name]}={number_token(value)}"
        for name, value in values.items()
        if name in ids
    ) + "}"


def solution_payload(physical, solution) -> str:
    phys = ",".join(
        number_token(physical.nodes.values[node]["cpu"])
        for node in range(physical.num_nodes)
    )
    slots = ",".join(
        f"{virtual}:{physical_node}"
        for virtual, physical_node in solution["node_slots"].items()
    )
    info = ",".join(
        f"{virtual}:{physical_node}{values_payload(values, RESOURCE_IDS)}"
        for (virtual, physical_node), values
        in solution["node_slots_info"].items()
    )
    offsets = ",".join(
        f"{virtual}{values_payload(values, CONSTRAINT_IDS)}"
        for virtual, values
        in solution["v_net_constraint_offsets"]["node_level"].items()
    )
    violations = ",".join(
        f"{virtual}{values_payload(values, CONSTRAINT_IDS)}"
        for virtual, values
        in solution["v_net_constraint_violations"]["node_level"].items()
    )
    total = number_token(
        float(solution["v_net_total_hard_constraint_violation"]))
    flags = (
        ("1" if solution["place_result"] else "0") + "," +
        ("1" if solution["result"] else "0"))
    return (
        f"phys=[{phys}];slots=[{slots}];info=[{info}];"
        f"offsets=[{offsets}];violations=[{violations}];"
        f"total={total};flags={flags}"
    )


def placement_payload(result, physical, solution) -> str:
    placed, offsets = result
    hard_feasible = offsets["cpu"] <= 0
    return (
        f"placed={'1' if placed else '0'};"
        f"feasible={'1' if hard_feasible else '0'};" +
        solution_payload(physical, solution)
    )


def mapping_payload(mapped, physical, solution) -> str:
    return (
        f"mapped={'1' if mapped else '0'};" +
        solution_payload(physical, solution)
    )


def run_greedy(NodeMapper, inplace=True):
    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    mapped = mapper.node_mapping(
        virtual,
        physical,
        [0, 1],
        [0, 1, 2],
        solution,
        reusable=False,
        inplace=inplace,
        matching_mathod="greedy")
    return mapping_payload(mapped, physical, solution)


def python_cases(NodeMapper):
    cases = {}

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    result = mapper.place(virtual, physical, 0, 0, solution)
    cases["safe_success"] = placement_payload(result, physical, solution)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    result = mapper.place(virtual, physical, 0, 1, solution)
    cases["safe_failure"] = placement_payload(result, physical, solution)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    result = mapper.place(
        virtual,
        physical,
        0,
        1,
        solution,
        if_allow_constraint_violation=True)
    cases["unsafe_failure_places"] = placement_payload(
        result, physical, solution)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    result = mapper.place(
        virtual,
        physical,
        0,
        0,
        solution,
        if_record_constraint_violation=False)
    cases["safe_success_no_record"] = placement_payload(
        result, physical, solution)

    mapper, _, physical = fixture(NodeMapper)
    solution = make_solution()
    offsets = {"cpu": 2, "quality": 3.0}
    mapper.record_place_constraint_violation(0, offsets, solution)
    mapper.record_place_constraint_violation(0, offsets, solution)
    cases["record_repeated"] = solution_payload(physical, solution)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    mapper.place(
        virtual,
        physical,
        0,
        0,
        solution,
        if_record_constraint_violation=False)
    undone = mapper.undo_place(0, physical, solution)
    cases["undo_success"] = (
        f"undone={'1' if undone else '0'};" +
        solution_payload(physical, solution))

    cases["mapping_greedy"] = run_greedy(NodeMapper)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    mapped = mapper.node_mapping(
        virtual,
        physical,
        [0, 1],
        [0, 1, 2],
        solution,
        matching_mathod="l2s2")
    cases["mapping_l2s2_failure"] = mapping_payload(
        mapped, physical, solution)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    mapped = mapper.node_mapping(
        virtual,
        physical,
        [0, 1],
        [2],
        solution,
        reusable=True)
    cases["mapping_reusable"] = mapping_payload(
        mapped, physical, solution)

    cases["mapping_inplace_false"] = run_greedy(
        NodeMapper, inplace=False)

    mapper, virtual, physical = fixture(NodeMapper)
    solution = make_solution()
    solution["v_net_total_hard_constraint_violation"] = 2.0
    mapped = mapper.node_mapping(
        virtual,
        physical,
        [0, 1],
        [0, 1, 2],
        solution,
        matching_mathod="l2s2")
    cases["mapping_failure_preserves_total"] = mapping_payload(
        mapped, physical, solution)

    canonical = run_greedy(NodeMapper)
    cases["mapping_workers_0_1_2_8"] = "|".join(
        [canonical, canonical, canonical, canonical])
    return cases


def cpp_cases(harness: pathlib.Path):
    process = subprocess.run(
        [str(harness), "differential"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"NodeMapper harness failed: {process.stderr.strip()}")
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

    NodeMapper = load_mapper(args.source)
    expected = python_cases(NodeMapper)
    actual = cpp_cases(args.harness)
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, C++={list(actual)}")
    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected
        if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "NodeMapper differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.controller.NodeMapper",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
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
