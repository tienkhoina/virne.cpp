#!/usr/bin/env python3
"""Exact Controller-class AST differential against real native production."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


SOURCE_SHA256 = "352552b0312d18b42eb6904629ed43c76df2d9c844df59fe51ab8653ffac02a5"
PHYSICAL_LINKS = ((0, 1), (1, 4), (0, 2), (2, 5))
WORKERS = (1, 2, 8)

BOUNDARIES = {
    "oracle_collaborators": (
        "The exact Controller class AST runs with narrow deterministic "
        "NodeMapper/LinkMapper/ResourceUpdator/network collaborators matching "
        "this fixture; native always runs the real prepared production stack."
    ),
    "native_workers": (
        "Python lifecycle mutation is sequential; native workers 1/2/8 must "
        "retain its output and partial-error order."
    ),
    "deferred": (
        "MCF, BFS solver deployment, candidate search, unsafe whole-link "
        "mapping, solver/system/RL/ML remain outside this differential."
    ),
}


class Attr:
    def __init__(self, name: str):
        self.name = name


class FakeVirtualNetwork:
    def __init__(self):
        self.nodes = {
            0: {"cpu": 4}, 1: {"cpu": 2}, 2: {"cpu": 2}}
        self.links = {
            (0, 1): {"bandwidth": 3},
            (0, 2): {"bandwidth": 6},
        }
        self.adj = {
            0: {1: {}, 2: {}},
            1: {0: {}},
            2: {0: {}},
        }
        self.num_nodes = 3
        self.num_links = 2
        self.id = 17
        self.lifetime = 4.0
        self.arrival_time = 1.25

    def get_link_attrs(self):
        return [Attr("bandwidth")]


class FakePhysicalNetwork:
    def __init__(self):
        self.nodes = {
            index: {"cpu": value}
            for index, value in enumerate((10, 10, 10, 3, 10, 10))
        }
        self.links = {
            link: {"bandwidth": value}
            for link, value in zip(PHYSICAL_LINKS, (5, 4, 5, 4))
        }


class FakeSolution(dict):
    @classmethod
    def from_v_net(cls, _virtual):
        return make_solution()


def canonical_link(link):
    return tuple(sorted(link))


class FakeNodeMapper:
    @staticmethod
    def place(
        virtual, physical, virtual_node, physical_node, solution,
        if_allow_constraint_violation=False,
    ):
        demand = virtual.nodes[virtual_node]["cpu"]
        available = physical.nodes[physical_node]["cpu"]
        offset = demand - available
        solution["v_net_constraint_offsets"]["node_level"][virtual_node] = {
            "cpu": offset}
        violation = max(offset, 0)
        solution["v_net_constraint_violations"]["node_level"][virtual_node] = {
            "cpu": violation}
        solution["v_net_total_hard_constraint_violation"] += violation
        feasible = offset <= 0
        if not feasible and not if_allow_constraint_violation:
            return False, {"cpu": offset}
        physical.nodes[physical_node]["cpu"] -= demand
        solution["node_slots"][virtual_node] = physical_node
        solution["node_slots_info"][(virtual_node, physical_node)] = {
            "cpu": demand}
        return True, {"cpu": offset}

    @staticmethod
    def undo_place(virtual_node, physical, solution):
        physical_node = solution["node_slots"][virtual_node]
        resources = solution["node_slots_info"][(virtual_node, physical_node)]
        physical.nodes[physical_node]["cpu"] += resources["cpu"]
        del solution["node_slots"][virtual_node]
        del solution["node_slots_info"][(virtual_node, physical_node)]
        return True


class FakeLinkMapper:
    @staticmethod
    def _path(physical_pair):
        if physical_pair == (0, 4):
            return [(0, 1), (1, 4)]
        if physical_pair == (4, 0):
            return [(4, 1), (1, 0)]
        if physical_pair == (0, 5):
            return [(0, 2), (2, 5)]
        if physical_pair == (5, 0):
            return [(5, 2), (2, 0)]
        return []

    def route(
        self, virtual, physical, virtual_link, physical_pair, solution,
        shortest_method="bfs_shortest", k=1,
        if_allow_constraint_violation=False,
    ):
        del shortest_method, k
        demand = virtual.links[canonical_link(virtual_link)]["bandwidth"]
        path = self._path(physical_pair)
        feasible = bool(path) and all(
            physical.links[canonical_link(link)]["bandwidth"] >= demand
            for link in path
        )
        if feasible or if_allow_constraint_violation:
            raw_offsets = [
                demand - physical.links[canonical_link(link)]["bandwidth"]
                for link in path
            ]
            solution["link_paths"][virtual_link] = list(path)
            for physical_link in path:
                key = canonical_link(physical_link)
                physical.links[key]["bandwidth"] -= demand
                solution["link_paths_info"][(virtual_link, physical_link)] = {
                    "bandwidth": demand}
            pooled = max(raw_offsets)
            solution["v_net_constraint_offsets"]["link_level"][virtual_link] = {
                "bandwidth": pooled}
            solution["v_net_constraint_offsets"]["path_level"][virtual_link] = {}
            solution["v_net_constraint_violations"]["link_level"][virtual_link] = {
                "bandwidth": max(pooled, 0)}
            solution["v_net_constraint_violations"]["path_level"][virtual_link] = {}
            return True, {"link_level": {}, "path_level": {}}

        solution["link_paths"][virtual_link] = []
        placeholder = 100.0
        solution["v_net_constraint_offsets"]["link_level"][virtual_link] = {
            "bandwidth": placeholder}
        solution["v_net_constraint_offsets"]["path_level"][virtual_link] = {}
        solution["v_net_constraint_violations"]["link_level"][virtual_link] = {
            "bandwidth": placeholder}
        solution["v_net_constraint_violations"]["path_level"][virtual_link] = {}
        solution["v_net_total_hard_constraint_violation"] += placeholder
        return False, {
            "link_level": {"bandwidth": placeholder}, "path_level": {}}

    @staticmethod
    def undo_route(virtual_link, physical, solution):
        path = solution["link_paths"][virtual_link]
        for physical_link in path:
            resources = solution["link_paths_info"][(
                virtual_link, physical_link)]
            physical.links[canonical_link(physical_link)]["bandwidth"] += (
                resources["bandwidth"])
            del solution["link_paths_info"][(virtual_link, physical_link)]
        del solution["link_paths"][virtual_link]
        return True


class FakeResourceUpdator:
    @staticmethod
    def update_node_resources(
        physical, physical_node, resources, operator, safe=True,
    ):
        for name, amount in resources.items():
            if operator == "-":
                if safe:
                    assert physical.nodes[physical_node][name] >= amount
                physical.nodes[physical_node][name] -= amount
            elif operator == "+":
                physical.nodes[physical_node][name] += amount
            else:
                raise NotImplementedError

    @staticmethod
    def update_link_resources(
        physical, physical_link, resources, operator, safe=True,
    ):
        target = physical.links[canonical_link(physical_link)]
        for name, amount in resources.items():
            if operator == "-":
                if safe:
                    assert target[name] >= amount
                target[name] -= amount
            elif operator == "+":
                target[name] += amount
            else:
                raise NotImplementedError


def load_controller(source: pathlib.Path):
    source_bytes = source.read_bytes()
    actual = hashlib.sha256(source_bytes).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Controller source hash drift: {actual}")
    tree = ast.parse(source_bytes.decode("utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Controller"
    ]
    if len(matches) != 1:
        raise RuntimeError("Controller class inventory drift")
    isolated = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    namespace = {
        "__name__": "_controller_differential_oracle",
        "Any": Any,
        "Callable": Callable,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
        "DictConfig": object,
        "VirtualNetwork": FakeVirtualNetwork,
        "PhysicalNetwork": FakePhysicalNetwork,
        "Solution": FakeSolution,
        "copy": copy,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Controller"]


def fixture(Controller):
    virtual = FakeVirtualNetwork()
    physical = FakePhysicalNetwork()
    value = Controller.__new__(Controller)
    value.node_resource_attrs = [Attr("cpu")]
    value.link_resource_attrs = [Attr("bandwidth")]
    value.hard_constraint_attrs_names = ["cpu", "bandwidth"]
    value.step_constraint_offset_placeholder = {
        "node_level": {"cpu": 0.0},
        "link_level": {"bandwidth": 0.0},
        "path_level": {},
    }
    value.node_mapper = FakeNodeMapper()
    value.link_mapper = FakeLinkMapper()
    value.resource_updator = FakeResourceUpdator()
    return value, virtual, physical


def make_solution(result=True):
    return FakeSolution({
        "result": result,
        "place_result": True,
        "route_result": True,
        "node_slots": {},
        "link_paths": {},
        "node_slots_info": {},
        "link_paths_info": {},
        "v_net_constraint_offsets": {
            "node_level": {}, "link_level": {}, "path_level": {}},
        "v_net_constraint_violations": {
            "node_level": {}, "link_level": {}, "path_level": {}},
        "v_net_total_hard_constraint_violation": 0.0,
        "v_net_single_step_constraint_offset": {
            "node_level": {}, "link_level": {}, "path_level": {}},
        "v_net_single_step_hard_constraint_offset": float("-inf"),
        "v_net_max_single_step_hard_constraint_violation": float("-inf"),
    })


def manual_solution(result=True, second_cpu=2):
    solution = make_solution(result)
    solution["node_slots"] = {0: 0, 1: 4}
    solution["node_slots_info"] = {
        (0, 0): {"cpu": 4},
        (1, 4): {"cpu": second_cpu},
    }
    solution["link_paths"] = {(0, 1): [(0, 1), (1, 4)]}
    solution["link_paths_info"] = {
        ((0, 1), (0, 1)): {"bandwidth": 3},
        ((0, 1), (1, 4)): {"bandwidth": 3},
    }
    return solution


def number_token(value):
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    return "d:" + struct.pack(">d", float(value)).hex()


def double_token(value):
    return "d:" + struct.pack(">d", float(value)).hex()


def values_payload(values, ids):
    return "{" + ",".join(
        f"{ids[name]}={number_token(value)}"
        for name, value in values.items() if name in ids
    ) + "}"


def link_token(link):
    return f"{link[0]}:{link[1]}"


def solution_payload(physical, solution):
    phys = "N[" + ",".join(
        number_token(physical.nodes[node]["cpu"])
        for node in range(6)
    ) + "]L[" + ",".join(
        number_token(physical.links[link]["bandwidth"])
        for link in PHYSICAL_LINKS
    ) + "]"
    slots = ",".join(
        f"{virtual}:{physical_node}"
        for virtual, physical_node in solution["node_slots"].items())
    paths = ",".join(
        f"{link_token(virtual)}{{" + ",".join(
            link_token(link) for link in path) + "}"
        for virtual, path in solution["link_paths"].items())
    node_info = ",".join(
        f"{virtual}:{physical_node}{values_payload(values, {'cpu': 0})}"
        for (virtual, physical_node), values
        in solution["node_slots_info"].items())
    link_info = ",".join(
        f"{link_token(virtual)}/{link_token(physical_link)}"
        f"{values_payload(values, {'bandwidth': 0})}"
        for (virtual, physical_link), values
        in solution["link_paths_info"].items())
    node_offsets = ",".join(
        f"{node}{values_payload(values, {'cpu': 0})}"
        for node, values
        in solution["v_net_constraint_offsets"]["node_level"].items())
    link_offsets = ",".join(
        f"{link_token(link)}{values_payload(values, {'bandwidth': 0})}"
        for link, values
        in solution["v_net_constraint_offsets"]["link_level"].items())
    node_violations = ",".join(
        f"{node}{values_payload(values, {'cpu': 0})}"
        for node, values
        in solution["v_net_constraint_violations"]["node_level"].items())
    link_violations = ",".join(
        f"{link_token(link)}{values_payload(values, {'bandwidth': 0})}"
        for link, values
        in solution["v_net_constraint_violations"]["link_level"].items())
    step = solution["v_net_single_step_constraint_offset"]
    flags = ",".join("1" if solution[key] else "0" for key in (
        "place_result", "route_result", "result"))
    return (
        f"{phys};slots=[{slots}];paths=[{paths}];ni=[{node_info}];"
        f"li=[{link_info}];no=[{node_offsets}];lo=[{link_offsets}];"
        f"nv=[{node_violations}];lv=[{link_violations}];"
        f"step=N{values_payload(step['node_level'], {'cpu': 0})}"
        f"L{values_payload(step['link_level'], {'bandwidth': 0})}P{{}};"
        f"total={double_token(solution['v_net_total_hard_constraint_violation'])};"
        f"single={double_token(solution['v_net_single_step_hard_constraint_offset'])};"
        f"maximum={double_token(solution['v_net_max_single_step_hard_constraint_violation'])};"
        f"flags={flags}"
    )


def place_payload(result, attempted, phase, physical, solution):
    succeeded, _ = result
    placed = phase != "place"
    last = "none" if attempted == 0 else ("1" if succeeded else "0")
    return (
        f"ok={'1' if succeeded else '0'};phase={phase};"
        f"placed={'1' if placed else '0'};routes={attempted};last={last};" +
        solution_payload(physical, solution)
    )


def python_cases(Controller):
    cases = {}

    value, virtual, physical = fixture(Controller)
    solution = make_solution()
    result = value.place_and_route(virtual, physical, 1, 4, solution)
    cases["safe_no_neighbor"] = place_payload(
        result, 0, "none", physical, solution)

    value, virtual, physical = fixture(Controller)
    solution = make_solution()
    value.place_and_route(virtual, physical, 1, 4, solution)
    result = value.place_and_route(virtual, physical, 0, 0, solution)
    cases["safe_one_route"] = place_payload(
        result, 1, "none", physical, solution)

    value, virtual, physical = fixture(Controller)
    solution = make_solution()
    result = value.place_and_route(virtual, physical, 0, 3, solution)
    cases["placement_failure"] = place_payload(
        result, 0, "place", physical, solution)

    value, virtual, physical = fixture(Controller)
    solution = make_solution()
    value.place_and_route(virtual, physical, 1, 4, solution)
    value.place_and_route(virtual, physical, 2, 5, solution)
    result = value.place_and_route(virtual, physical, 0, 0, solution)
    cases["route_partial_failure"] = place_payload(
        result, 2, "route", physical, solution)

    value, virtual, physical = fixture(Controller)
    solution = make_solution()
    value.place_and_route(virtual, physical, 1, 4, solution)
    value.place_and_route(virtual, physical, 0, 0, solution)
    undone = value.undo_place_and_route(virtual, physical, 0, 0, solution)
    cases["undo_place_route"] = (
        f"undone={'1' if undone else '0'};" +
        solution_payload(physical, solution))

    value, virtual, physical = fixture(Controller)
    solution = manual_solution(False)
    deployed = value.deploy(virtual, physical, solution)
    released = value.release(virtual, physical, solution)
    cases["unsuccessful_noop"] = (
        f"deploy={'1' if deployed else '0'};"
        f"release={'1' if released else '0'};" +
        solution_payload(physical, solution))

    rows = []
    for workers in WORKERS:
        value, virtual, physical = fixture(Controller)
        solution = manual_solution()
        deployed = value.deploy(virtual, physical, solution)
        after_deploy = solution_payload(physical, solution)
        released = value.release(virtual, physical, solution)
        rows.append(
            f"w={workers}{{d={'1' if deployed else '0'};{after_deploy};"
            f"r={'1' if released else '0'};{solution_payload(physical, solution)}}}")
    cases["roundtrip_workers_1_2_8"] = "|".join(rows)

    rows = []
    for workers in WORKERS:
        value, virtual, physical = fixture(Controller)
        solution = manual_solution(True, 99)
        error = "none"
        try:
            value.deploy(virtual, physical, solution)
        except AssertionError:
            error = "resource"
        rows.append(
            f"w={workers}{{error={error};{solution_payload(physical, solution)}}}")
    cases["deploy_partial_workers_1_2_8"] = "|".join(rows)

    value, virtual, physical = fixture(Controller)
    solution = manual_solution()
    value.deploy(virtual, physical, solution)
    del solution["node_slots_info"][(1, 4)]
    error = "none"
    try:
        value.release(virtual, physical, solution)
    except KeyError:
        error = "missing"
    cases["release_partial_missing_info"] = (
        f"error={error};" + solution_payload(physical, solution))

    value, virtual, physical = fixture(Controller)
    solution = manual_solution()
    value.deploy(virtual, physical, solution)
    undone = value.undo_deploy(virtual, physical, solution)
    cases["undo_deploy_quirk"] = (
        f"undone={'1' if undone else '0'};" +
        solution_payload(physical, solution))
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
            f"Controller harness failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        name, separator, payload = line.partition("\t")
        if not separator or name in result:
            raise RuntimeError(f"malformed native Controller case: {line!r}")
        result[name] = payload
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Controller = load_controller(args.source)
    expected = python_cases(Controller)
    actual = cpp_cases(args.harness)
    if list(actual) != list(expected):
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, C++={list(actual)}")
    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "Controller differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.controller.Controller",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "workers": list(WORKERS),
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
