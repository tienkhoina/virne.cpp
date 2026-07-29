#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python TopologyAnalyzer leaf."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pathlib
import subprocess
from collections import deque
from itertools import islice
from typing import Any

import networkx as nx


SOURCE_SHA256 = "665519c5c4bf50c2318e2d22d30679881dda0edcbc0b618c4fb2055ac5a01b28"

BOUNDARIES = {
    "broken_python_direct_bfs":
        "The pinned direct Python helper raises on no-path and mishandles identical endpoints; the native typed helper deliberately returns nullopt or {source}, while high-level no-path remains empty.",
    "dynamic_methods":
        "Python validates method spellings with assert; native public methods are an enum and invalid underlying enum values produce a typed error.",
    "dynamic_names":
        "Python resource dictionaries hash names in every edge predicate; native prepare resolves names once and hot loops use graph-local IDs and direct typed pointers.",
    "native_parallelism":
        "Configurable deterministic mask and request workers are a native extension; Python exposes scalar traversal only.",
}


def load_analyzer(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"TopologyAnalyzer source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TopologyAnalyzer"
    ]
    if len(matches) != 1:
        raise RuntimeError("TopologyAnalyzer class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Callable": Any,
        "Dict": dict,
        "List": list,
        "Optional": Any,
        "PhysicalNetwork": Any,
        "Tuple": tuple,
        "Union": Any,
        "VirtualNetwork": Any,
        "copy": copy,
        "deque": deque,
        "islice": islice,
        "nx": nx,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["TopologyAnalyzer"]


class FakeNetwork(nx.Graph):
    @property
    def links(self):
        return self.edges

    @property
    def num_nodes(self):
        return self.number_of_nodes()


class FakeLinkResource:
    def __init__(self, name: str, soft: bool = False):
        self.name = name
        self.soft = soft

    def check_constraint_satisfiability(self, virtual, physical):
        offset = virtual[self.name] - physical[self.name]
        return self.soft or offset <= 0, offset


class FakeConstraintChecker:
    def __init__(self, link_attrs):
        self.link_attrs = list(link_attrs)

    @staticmethod
    def check_constraint_satisfiability(virtual, physical, attrs):
        feasible = True
        offsets = {}
        for link_attr in attrs:
            flag, offset = link_attr.check_constraint_satisfiability(
                virtual, physical)
            feasible = flag and feasible
            offsets[link_attr.name] = offset
        return feasible, offsets

    def check_link_level_constraints(
            self, virtual, physical, virtual_link, physical_link):
        return self.check_constraint_satisfiability(
            virtual.links[virtual_link],
            physical.links[physical_link],
            self.link_attrs)


PHYSICAL_LINKS = [
    (0, 1),
    (1, 5),
    (0, 2),
    (2, 5),
    (0, 3),
    (3, 4),
    (4, 5),
]


def fixture(TopologyAnalyzer, constraints=(), resources=()):
    virtual = FakeNetwork()
    virtual.add_nodes_from(range(2))
    virtual.add_edge(0, 1, capacity_hard=5, capacity_soft=100.0)

    physical = FakeNetwork()
    physical.add_nodes_from(range(7))
    hard_values = [4, 12, 8, 12, 12, 12, 12]
    for link, hard in zip(PHYSICAL_LINKS, hard_values):
        physical.add_edge(
            *link,
            capacity_hard=hard,
            capacity_soft=1.0)

    hard = FakeLinkResource("capacity_hard")
    soft = FakeLinkResource("capacity_soft", soft=True)
    by_name = {"hard": hard, "soft": soft}
    checker = FakeConstraintChecker(by_name[name] for name in constraints)
    analyzer = TopologyAnalyzer(
        checker, [by_name[name] for name in resources])
    return analyzer, virtual, physical


def paths_payload(paths) -> str:
    return "[" + ",".join(
        "[" + ",".join(str(vertex) for vertex in path) + "]"
        for path in paths
    ) + "]"


def mask_payload(view) -> str:
    return "".join(
        "1" if view.has_edge(*link) else "0"
        for link in PHYSICAL_LINKS
    )


def find(
        analyzer,
        virtual,
        physical,
        method,
        k=10,
        max_hop=1.0e6,
        source=0,
        target=5):
    return analyzer.find_shortest_paths(
        virtual,
        physical,
        (0, 1),
        (source, target),
        method=method,
        k=k,
        max_hop=max_hop)


def python_cases(TopologyAnalyzer):
    analyzer, virtual, physical = fixture(TopologyAnalyzer)
    cases = {
        "mode_first_shortest": paths_payload(find(
            analyzer, virtual, physical, "first_shortest")),
        "mode_k_shortest": paths_payload(find(
            analyzer, virtual, physical, "k_shortest", k=3)),
        "mode_k_shortest_length": paths_payload(find(
            analyzer, virtual, physical, "k_shortest_length", k=3)),
        "mode_all_shortest": paths_payload(find(
            analyzer, virtual, physical, "all_shortest")),
        "mode_bfs_shortest": paths_payload(find(
            analyzer, virtual, physical, "bfs_shortest")),
        "mode_available_shortest": paths_payload(find(
            analyzer, virtual, physical, "available_shortest")),
        "k_shortest_zero": paths_payload(find(
            analyzer, virtual, physical, "k_shortest", k=0)),
        "k_shortest_negative": paths_payload(find(
            analyzer, virtual, physical, "k_shortest", k=-1)),
        "length_zero": paths_payload(find(
            analyzer, virtual, physical, "k_shortest_length", k=0)),
        "length_cutoff_four": paths_payload(find(
            analyzer, virtual, physical, "k_shortest_length", k=4)),
        "max_first_only": paths_payload(find(
            analyzer, virtual, physical, "k_shortest", k=3, max_hop=3.0)),
        "max_reject": paths_payload(find(
            analyzer, virtual, physical, "k_shortest", k=3, max_hop=2.5)),
        "no_path": paths_payload(find(
            analyzer,
            virtual,
            physical,
            "first_shortest",
            source=0,
            target=6)),
    }

    tie_virtual = FakeNetwork()
    tie_virtual.add_nodes_from(range(2))
    tie_virtual.add_edge(0, 1)
    tie_physical = FakeNetwork()
    tie_physical.add_nodes_from(range(4))
    tie_physical.add_edges_from([
        (0, 1), (2, 3), (0, 2), (0, 3), (1, 2)])
    tie_analyzer = TopologyAnalyzer(FakeConstraintChecker([]), [])
    cases["first_dijkstra_fifo_tie"] = paths_payload(find(
        tie_analyzer,
        tie_virtual,
        tie_physical,
        "first_shortest",
        source=1,
        target=3))
    cases["available_dijkstra_fifo_tie"] = paths_payload(find(
        tie_analyzer,
        tie_virtual,
        tie_physical,
        "available_shortest",
        source=1,
        target=3))

    hard_soft, virtual, physical = fixture(
        TopologyAnalyzer, constraints=("hard", "soft"))
    cases["available_hard_soft"] = paths_payload(find(
        hard_soft, virtual, physical, "available_shortest"))
    cases["bfs_hard_soft"] = paths_payload(find(
        hard_soft, virtual, physical, "bfs_shortest"))

    soft_only, virtual, physical = fixture(
        TopologyAnalyzer, constraints=("soft",))
    cases["available_soft_only"] = paths_payload(find(
        soft_only, virtual, physical, "available_shortest"))
    cases["bfs_soft_only"] = paths_payload(find(
        soft_only, virtual, physical, "bfs_shortest"))

    prune_single, virtual, physical = fixture(
        TopologyAnalyzer, resources=("hard",))
    cases["prune_ratio_then_div"] = mask_payload(
        prune_single.create_pruned_network(
            virtual, physical, (0, 1), ratio=2.0, div=3.0))
    cases["prune_equal_boundary"] = mask_payload(
        prune_single.create_pruned_network(
            virtual, physical, (0, 1), ratio=1.0, div=1.0))

    prune_duplicate, virtual, physical = fixture(
        TopologyAnalyzer, resources=("hard", "hard"))
    cases["prune_duplicate_resource"] = mask_payload(
        prune_duplicate.create_pruned_network(
            virtual, physical, (0, 1), ratio=2.0, div=3.0))

    prune_soft, virtual, physical = fixture(
        TopologyAnalyzer, resources=("soft",))
    cases["prune_soft_resource"] = mask_payload(
        prune_soft.create_pruned_network(
            virtual, physical, (0, 1), ratio=100.0, div=0.0))

    prune_empty, virtual, physical = fixture(TopologyAnalyzer)
    cases["prune_empty_resources"] = mask_payload(
        prune_empty.create_pruned_network(
            virtual, physical, (0, 1), ratio=99.0, div=77.0))
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
            f"TopologyAnalyzer harness failed: {process.stderr.strip()}")
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

    TopologyAnalyzer = load_analyzer(args.source)
    expected = python_cases(TopologyAnalyzer)
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
            "TopologyAnalyzer differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.controller.TopologyAnalyzer",
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
