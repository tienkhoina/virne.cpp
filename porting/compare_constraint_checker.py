#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python ConstraintChecker leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any


SOURCE_SHA256 = "ea41ee9226cef3f38cfff30d7abb276a7a9af58fee67ce78c3763413bdb9619a"

BOUNDARIES = {
    "dynamic_protocols":
        "Python permits monkey-patched attributes and arbitrary mapping protocols; native preparation accepts frozen concrete typed attributes.",
    "unbounded_labels":
        "Python node labels and integers are unbounded objects; native graph vertices and registry IDs use frozen bounded integer types.",
    "dynamic_names":
        "Python returns string-keyed offset dictionaries; native names resolve once during prepare and results use direct ID slots.",
    "native_batches":
        "Configurable deterministic worker batches are a native throughput extension; Python exposes scalar checks only.",
}


def path_to_links(path):
    if len(path) <= 1:
        raise ValueError("path_to_links requires at least two vertices")
    return list(zip(path[:-1], path[1:]))


def load_checker(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"ConstraintChecker source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConstraintChecker"
    ]
    if len(matches) != 1:
        raise RuntimeError("ConstraintChecker class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "path_to_links": path_to_links,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["ConstraintChecker"]


class NodeView:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, node):
        return self.values[node]


class LinkView:
    def __init__(self, values):
        self.values = {
            tuple(sorted(endpoints)): attributes
            for endpoints, attributes in values.items()
        }

    def __getitem__(self, endpoints):
        return self.values[tuple(sorted(endpoints))]


class FakeNetwork:
    def __init__(self, nodes, links, graph):
        self.nodes = NodeView(nodes)
        self.links = LinkView(links)
        self.graph = graph


class FakeAttribute:
    def __init__(self, name: str, family: str, soft: bool = False):
        self.name = name
        self.family = family
        self.soft = soft

    def check_constraint_satisfiability(self, virtual, physical):
        virtual_values = virtual.graph if isinstance(virtual, FakeNetwork) else virtual
        if isinstance(physical, FakeNetwork):
            physical_value = physical.graph[self.name]
        elif isinstance(physical, list):
            physical_value = sum(item[self.name] for item in physical)
        else:
            physical_value = physical[self.name]
        virtual_value = virtual_values[self.name]
        if self.family == "latency":
            offset = physical_value - virtual_value
        else:
            offset = virtual_value - physical_value
        return self.soft or offset <= 0, offset


NODE_IDS = {"node_hard": 0, "node_soft": 1}
LINK_IDS = {
    "link_hard": 0,
    "link_soft": 1,
    "latency_hard": 2,
    "latency_soft": 3,
}
GRAPH_IDS = {"graph_hard": 0, "graph_soft": 1, "graph_boolean": 2}


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def number_token(value) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    return double_token(value)


def offsets_payload(values, ids) -> str:
    ordered = sorted(values.items(), key=lambda item: ids[item[0]])
    return "[" + ",".join(
        f"{ids[name]}={number_token(value)}" for name, value in ordered
    ) + "]"


def check_payload(result, ids) -> str:
    flag, offsets = result
    return (
        f"flag={'b:1' if flag else 'b:0'};"
        f"offsets={offsets_payload(offsets, ids)}"
    )


def path_payload(result) -> str:
    flag, info = result
    link_groups = []
    for name, values in info["link_level"].items():
        encoded = ",".join(
            f"{link[0]}:{link[1]}={number_token(value)}"
            for link, value in values.items()
        )
        link_groups.append(f"{LINK_IDS[name]}={{{encoded}}}")
    return (
        f"flag={'b:1' if flag else 'b:0'};"
        f"links=[{','.join(link_groups)}];"
        f"path={offsets_payload(info['path_level'], LINK_IDS)}"
    )


def fixture(Checker):
    virtual = FakeNetwork(
        nodes={
            0: {"node_hard": 5, "node_soft": 9.5},
            1: {"node_hard": 7, "node_soft": 2.0},
            2: {"node_hard": 1, "node_soft": 1.0},
        },
        links={
            (0, 1): {
                "link_hard": 4, "link_soft": 9.5,
                "latency_hard": 10.0, "latency_soft": 5.0,
            },
            (1, 2): {
                "link_hard": 2, "link_soft": 1.0,
                "latency_hard": 4.0, "latency_soft": 3.0,
            },
        },
        graph={
            "graph_hard": 7,
            "graph_soft": 20.5,
            "graph_boolean": False,
        },
    )
    physical = FakeNetwork(
        nodes={
            0: {"node_hard": 8, "node_soft": 4.0},
            1: {"node_hard": 3, "node_soft": 10.0},
            2: {"node_hard": 9, "node_soft": 3.0},
            3: {"node_hard": 6, "node_soft": 1.5},
        },
        links={
            (0, 1): {
                "link_hard": 6, "link_soft": 3.0,
                "latency_hard": 3.0, "latency_soft": 4.0,
            },
            (1, 2): {
                "link_hard": 2, "link_soft": 10.0,
                "latency_hard": 4.0, "latency_soft": 5.0,
            },
            (1, 3): {
                "link_hard": 7, "link_soft": 4.0,
                "latency_hard": 9.0, "latency_soft": 8.0,
            },
            (0, 3): {
                "link_hard": 7, "link_soft": 4.0,
                "latency_hard": 6.0, "latency_soft": 7.0,
            },
        },
        graph={
            "graph_hard": 10,
            "graph_soft": 5.0,
            "graph_boolean": True,
        },
    )
    node_attrs = [
        FakeAttribute("node_hard", "resource"),
        FakeAttribute("node_soft", "resource", soft=True),
    ]
    link_attrs = [
        FakeAttribute("link_hard", "resource"),
        FakeAttribute("link_soft", "resource", soft=True),
    ]
    path_attrs = [
        FakeAttribute("latency_hard", "latency"),
        FakeAttribute("latency_soft", "latency", soft=True),
    ]
    graph_attrs = [
        FakeAttribute("graph_hard", "resource"),
        FakeAttribute("graph_soft", "resource", soft=True),
        FakeAttribute("graph_boolean", "resource"),
    ]
    checker = Checker(node_attrs, link_attrs, path_attrs, graph_attrs)
    empty = Checker([], [], [], [])
    return checker, empty, virtual, physical


def python_cases(Checker):
    checker, empty, virtual, physical = fixture(Checker)
    cases = {
        "graph": check_payload(
            checker.check_graph_constraints(virtual, physical), GRAPH_IDS),
        "node_pass": check_payload(
            checker.check_node_level_constraints(virtual, physical, 0, 0),
            NODE_IDS),
        "node_fail": check_payload(
            checker.check_node_level_constraints(virtual, physical, 0, 1),
            NODE_IDS),
        "node_mixed": check_payload(
            checker.check_node_level_constraints(virtual, physical, 1, 2),
            NODE_IDS),
        "link_pass": check_payload(
            checker.check_link_level_constraints(
                virtual, physical, (0, 1), (0, 1)), LINK_IDS),
        "link_fail": check_payload(
            checker.check_link_level_constraints(
                virtual, physical, (0, 1), (1, 2)), LINK_IDS),
        "link_reversed": check_payload(
            checker.check_link_level_constraints(
                virtual, physical, (1, 0), (1, 0)), LINK_IDS),
        "path_direct": path_payload(
            checker.check_path_level_constraints(
                virtual, physical, (0, 1), [0, 3])),
        "path_link_fail": path_payload(
            checker.check_path_level_constraints(
                virtual, physical, (0, 1), [0, 1, 2])),
        "path_latency_fail": path_payload(
            checker.check_path_level_constraints(
                virtual, physical, (0, 1), [0, 1, 3])),
        "empty_graph": check_payload(
            empty.check_graph_constraints(virtual, physical), {}),
        "empty_node": check_payload(
            empty.check_node_level_constraints(virtual, physical, 0, 0), {}),
        "empty_link": check_payload(
            empty.check_link_level_constraints(
                virtual, physical, (0, 1), (0, 1)), {}),
        "empty_path": path_payload(
            empty.check_path_level_constraints(
                virtual, physical, (0, 1), [0, 3])),
    }
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
            f"ConstraintChecker harness failed: {process.stderr.strip()}")
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

    Checker = load_checker(args.source)
    expected = python_cases(Checker)
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
            "ConstraintChecker differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.controller.ConstraintChecker",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "native_unit_case_groups": 7,
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
