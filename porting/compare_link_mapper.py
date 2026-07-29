#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python LinkMapper leaf."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import logging
import pathlib
import struct
import subprocess
from typing import Any


SOURCE_SHA256 = "e7e5fb542d6fca6a5c9a8bebace2c82e9bfedef1628cd8adbb065c4f95d40237"
UNEXPECTED_CONSTRAINT_VIOLATION = 100.0

BOUNDARIES = {
    "broken_unsafe_whole_mapping":
        "Pinned Python non-empty unsafe link_mapping passes an unsupported pruning_ratio keyword, followed by tuple arithmetic and a self-reference bug; native rejects it before mutation.",
    "deferred_mcf_solver":
        "route_v_links_with_mcf dynamically imports OR-Tools/SCIP and is a deferred solver surface outside this graph/controller leaf.",
    "dynamic_protocols":
        "Python accepts arbitrary mappings, callback objects, monkey patches, custom numerics, and unbounded labels; native uses frozen typed networks, values, and ranker input.",
    "native_candidate_workers":
        "Configured candidate workers preserve Python path order but are a native throughput extension; Python checks candidates sequentially.",
}


def path_to_links(path):
    if len(path) <= 1:
        raise ValueError("path requires at least two vertices")
    return list(zip(path[:-1], path[1:]))


def load_mapper(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"LinkMapper source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LinkMapper"
    ]
    if len(matches) != 1:
        raise RuntimeError("LinkMapper class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "BaseNetwork": Any,
        "Callable": Any,
        "ConstraintChecker": Any,
        "Dict": dict,
        "List": list,
        "Optional": Any,
        "PhysicalNetwork": Any,
        "ResourceUpdator": Any,
        "Solution": Any,
        "TopologyAnalyzer": Any,
        "Tuple": tuple,
        "UNEXPECTED_CONSTRAINT_VIOLATION": UNEXPECTED_CONSTRAINT_VIOLATION,
        "Union": Any,
        "VirtualNetwork": Any,
        "copy": copy,
        "logging": logging,
        "path_to_links": path_to_links,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["LinkMapper"]


class LinkView:
    def __init__(self, values):
        self.values = {
            tuple(sorted(link)): attributes
            for link, attributes in values.items()
        }

    def __getitem__(self, link):
        return self.values[tuple(sorted(link))]

    def __iter__(self):
        return iter(self.values)


class FakeNetwork:
    def __init__(self, links, num_links=None):
        self.links = LinkView(links)
        self.num_links = len(links) if num_links is None else num_links


class FakeLinkResource:
    def __init__(self, name):
        self.name = name


class FakeConstraintChecker:
    @staticmethod
    def check_path_level_constraints(
            virtual, physical, virtual_link, path):
        virtual_values = virtual.links[virtual_link]
        link_offsets = {}
        latency_sum = 0.0
        feasible = True
        for physical_link in path_to_links(path):
            physical_values = physical.links[physical_link]
            offset = (
                virtual_values["capacity"] -
                physical_values["capacity"])
            link_offsets[physical_link] = offset
            feasible = offset <= 0 and feasible
            latency_sum += physical_values["latency"]
        latency_offset = latency_sum - virtual_values["latency"]
        feasible = latency_offset <= 0 and feasible
        return feasible, {
            "link_level": {"capacity": link_offsets},
            "path_level": {"latency": latency_offset},
        }


class FakeResourceUpdator:
    @staticmethod
    def update_link_resources(
            physical, physical_link, resources, operator, safe=True):
        target = physical.links[physical_link]
        for name, value in resources.items():
            if operator in ("-", "sub"):
                if safe:
                    assert target[name] >= value
                target[name] -= value
            elif operator in ("+", "add"):
                target[name] += value
            else:
                raise NotImplementedError


PATHS = [
    [0, 1, 4],
    [0, 2, 4],
    [0, 3, 4],
]
PHYSICAL_LINKS = [
    (0, 1), (1, 4),
    (0, 2), (2, 4),
    (0, 3), (3, 4),
]


class FakeTopologyAnalyzer:
    @staticmethod
    def find_shortest_paths(
            virtual, physical, virtual_link, physical_pair,
            method="k_shortest", k=3):
        del virtual, physical, virtual_link
        if physical_pair != (0, 4):
            return []
        if method in ("first_shortest", "bfs_shortest", "available_shortest"):
            return copy.deepcopy(PATHS[:1])
        if method == "all_shortest":
            return copy.deepcopy(PATHS)
        if k <= 0:
            return []
        return copy.deepcopy(PATHS[:k])


CAPACITY_ID = 0
LATENCY_ID = 1
LINK_IDS = {"capacity": CAPACITY_ID}
PATH_IDS = {"latency": LATENCY_ID}
RESOURCE_IDS = {"capacity": CAPACITY_ID}


def fixture(LinkMapper, reusable=False):
    virtual = FakeNetwork({
        (0, 1): {"capacity": 5, "latency": 6.0},
    })
    physical = FakeNetwork({
        (0, 1): {"capacity": 4, "latency": 3.0},
        (1, 4): {"capacity": 4, "latency": 3.0},
        (0, 2): {"capacity": 5, "latency": 3.0},
        (2, 4): {"capacity": 5, "latency": 3.0},
        (0, 3): {"capacity": 5, "latency": 3.0},
        (3, 4): {"capacity": 5, "latency": 3.0},
    })
    capacity = FakeLinkResource("capacity")
    mapper = LinkMapper(
        FakeConstraintChecker(),
        FakeResourceUpdator(),
        FakeTopologyAnalyzer(),
        [capacity],
        ["capacity", "latency"],
        {
            "link_level": {"capacity": 0.0},
            "path_level": {"latency": 0.0},
        },
        reusable=reusable)
    return mapper, virtual, physical


def make_solution(with_slots=False):
    solution = {
        "node_slots": {},
        "link_paths": {},
        "link_paths_info": {},
        "v_net_constraint_offsets": {
            "link_level": {}, "path_level": {}},
        "v_net_constraint_violations": {
            "link_level": {}, "path_level": {}},
        "v_net_total_hard_constraint_violation": 0.0,
        "route_result": True,
        "result": True,
    }
    if with_slots:
        solution["node_slots"] = {0: 0, 1: 4}
    return solution


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


def link_token(link) -> str:
    return f"{link[0]}:{link[1]}"


def solution_payload(physical, solution) -> str:
    phys = ",".join(
        number_token(physical.links[link]["capacity"])
        for link in PHYSICAL_LINKS
    )
    paths = ",".join(
        f"{link_token(virtual_link)}{{" +
        ",".join(link_token(item) for item in physical_path) + "}"
        for virtual_link, physical_path in solution["link_paths"].items()
    )
    info = ",".join(
        f"{link_token(virtual_link)}/{link_token(physical_link)}" +
        values_payload(values, RESOURCE_IDS)
        for (virtual_link, physical_link), values
        in solution["link_paths_info"].items()
    )
    link_offsets = ",".join(
        f"{link_token(link)}{values_payload(values, LINK_IDS)}"
        for link, values
        in solution["v_net_constraint_offsets"]["link_level"].items()
    )
    path_offsets = ",".join(
        f"{link_token(link)}{values_payload(values, PATH_IDS)}"
        for link, values
        in solution["v_net_constraint_offsets"]["path_level"].items()
    )
    link_violations = ",".join(
        f"{link_token(link)}{values_payload(values, LINK_IDS)}"
        for link, values
        in solution["v_net_constraint_violations"]["link_level"].items()
    )
    path_violations = ",".join(
        f"{link_token(link)}{values_payload(values, PATH_IDS)}"
        for link, values
        in solution["v_net_constraint_violations"]["path_level"].items()
    )
    total = number_token(
        float(solution["v_net_total_hard_constraint_violation"]))
    flags = (
        ("1" if solution["route_result"] else "0") + "," +
        ("1" if solution["result"] else "0"))
    return (
        f"phys=[{phys}];paths=[{paths}];info=[{info}];"
        f"lo=[{link_offsets}];po=[{path_offsets}];"
        f"lv=[{link_violations}];pv=[{path_violations}];"
        f"total={total};flags={flags}"
    )


def is_placeholder(check_info) -> bool:
    return any(
        isinstance(value, float)
        for value in check_info["link_level"].values())


def route_payload(result, physical, solution) -> str:
    routed, check_info = result
    return (
        f"routed={'1' if routed else '0'};"
        f"placeholder={'1' if is_placeholder(check_info) else '0'};" +
        solution_payload(physical, solution)
    )


def mapping_payload(mapped, physical, solution) -> str:
    return (
        f"mapped={'1' if mapped else '0'};" +
        solution_payload(physical, solution)
    )


def route_options(mapper, virtual, physical, solution, **overrides):
    options = {
        "shortest_method": "k_shortest",
        "k": 3,
        "if_allow_constraint_violation": False,
        "if_record_constraint_violation": True,
    }
    options.update(overrides)
    return mapper.route(
        virtual, physical, (0, 1), options.pop("physical_pair"),
        solution, **options)


def run_safe(LinkMapper):
    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4))
    return route_payload(result, physical, solution)


def manual_check():
    return {
        "link_level": {
            "capacity": {
                (0, 1): -2,
                (1, 4): 3,
                (0, 2): 4,
            },
        },
        "path_level": {"latency": -1.0},
    }


def python_cases(LinkMapper):
    cases = {}

    mapper, virtual, physical = fixture(LinkMapper, reusable=True)
    solution = make_solution()
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 0))
    cases["same_node_reusable"] = route_payload(
        result, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    error = "none"
    try:
        route_options(
            mapper,
            virtual,
            physical,
            solution,
            physical_pair=(0, 0))
    except NotImplementedError:
        error = "same_node"
    cases["same_node_nonreusable"] = (
        f"error={error};" + solution_payload(physical, solution))

    cases["safe_success"] = run_safe(LinkMapper)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 5))
    cases["safe_no_path"] = route_payload(result, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    virtual.links[(0, 1)]["latency"] = 5.0
    solution = make_solution()
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4))
    cases["safe_all_infeasible"] = route_payload(
        result, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()

    def reverse_ranker(
            virtual_net, physical_net, virtual_link,
            physical_pair, paths):
        del virtual_net, physical_net, virtual_link, physical_pair
        return list(reversed(paths))

    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4),
        rank_path_func=reverse_ranker)
    cases["safe_ranker_reverse"] = route_payload(
        result, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4),
        if_record_constraint_violation=False)
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 5),
        if_record_constraint_violation=False)
    cases["reroute_resource_leak"] = route_payload(
        result, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4),
        if_allow_constraint_violation=True)
    cases["unsafe_first_feasible"] = route_payload(
        result, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    virtual.links[(0, 1)]["capacity"] = 6
    virtual.links[(0, 1)]["latency"] = 5.0
    solution = make_solution()
    result = route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4),
        if_allow_constraint_violation=True)
    cases["unsafe_least_violation_tie"] = route_payload(
        result, physical, solution)

    mapper, _, physical = fixture(LinkMapper)
    solution = make_solution()
    mapper.record_route_constraint_violation(
        (0, 1), manual_check(), solution)
    cases["pooling_mixed"] = solution_payload(physical, solution)

    mapper, _, physical = fixture(LinkMapper)
    solution = make_solution()
    mapper.record_route_constraint_violation(
        (0, 1), manual_check(), solution)
    mapper.record_route_constraint_violation(
        (0, 1), manual_check(), solution)
    cases["pooling_repeated"] = solution_payload(physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4),
        if_record_constraint_violation=False)
    undone = mapper.undo_route((0, 1), physical, solution)
    cases["undo_success"] = (
        f"undone={'1' if undone else '0'};" +
        solution_payload(physical, solution))

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution()
    route_options(
        mapper,
        virtual,
        physical,
        solution,
        physical_pair=(0, 4),
        if_record_constraint_violation=False)
    del solution["link_paths_info"][((0, 1), (2, 4))]
    error = "none"
    try:
        mapper.undo_route((0, 1), physical, solution)
    except KeyError:
        error = "missing_info"
    cases["undo_partial_missing_info"] = (
        f"error={error};" + solution_payload(physical, solution))

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution(with_slots=True)
    mapped = mapper.link_mapping(
        virtual,
        physical,
        solution,
        shortest_method="k_shortest",
        k=3)
    cases["mapping_success"] = mapping_payload(
        mapped, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    solution = make_solution(with_slots=True)
    mapped = mapper.link_mapping(
        virtual,
        physical,
        solution,
        shortest_method="k_shortest",
        k=3,
        inplace=False)
    cases["mapping_clone"] = mapping_payload(mapped, physical, solution)

    mapper, virtual, physical = fixture(LinkMapper)
    virtual.links[(0, 1)]["latency"] = 5.0
    solution = make_solution(with_slots=True)
    mapped = mapper.link_mapping(
        virtual,
        physical,
        solution,
        shortest_method="k_shortest",
        k=3)
    cases["mapping_failure"] = mapping_payload(
        mapped, physical, solution)

    canonical = run_safe(LinkMapper)
    cases["workers_0_1_2_8"] = "|".join(
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
            f"LinkMapper harness failed: {process.stderr.strip()}")
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

    LinkMapper = load_mapper(args.source)
    expected = python_cases(LinkMapper)
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
            "LinkMapper differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.controller.LinkMapper",
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
