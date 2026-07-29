#!/usr/bin/env python3
"""Compact checksum-gated safe LinkMapper benchmark; not run on import."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import logging
import pathlib
import platform
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx


SOURCE_SHA256 = (
    "e7e5fb542d6fca6a5c9a8bebace2c82e9bfedef1628cd8adbb065c4f95d40237"
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def path_to_links(path: list[int]) -> list[tuple[int, int]]:
    if len(path) < 2:
        raise ValueError("path must contain at least two vertices")
    return list(zip(path[:-1], path[1:]))


def load_link_mapper(source: pathlib.Path):
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"LinkMapper source hash drift: {actual}")
    tree = ast.parse(raw.decode("utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LinkMapper"
    ]
    if len(matches) != 1:
        raise RuntimeError("LinkMapper class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Callable": Callable,
        "ConstraintChecker": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "PhysicalNetwork": Any,
        "ResourceUpdator": Any,
        "Solution": Any,
        "TopologyAnalyzer": Any,
        "Tuple": Tuple,
        "Union": Union,
        "VirtualNetwork": Any,
        "UNEXPECTED_CONSTRAINT_VIOLATION": 100.0,
        "copy": copy,
        "logging": logging,
        "path_to_links": path_to_links,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["LinkMapper"]


class FakeNetwork:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.links = graph.edges
        self.num_links = graph.number_of_edges()
        self.id = 0


class FakeLinkResource:
    def __init__(self, name: str):
        self.name = name


class FakeTopologyAnalyzer:
    def find_shortest_paths(
        self,
        _v_net,
        p_net,
        _v_link,
        physical_pair,
        method="all_shortest",
        k=1,
    ):
        del k
        if method != "all_shortest":
            raise RuntimeError("benchmark requires all_shortest")
        return [
            list(path)
            for path in nx.all_shortest_paths(
                p_net.graph, physical_pair[0], physical_pair[1]
            )
        ]


class FakeConstraintChecker:
    def check_path_level_constraints(
        self, v_net, p_net, virtual_link, physical_path
    ):
        demand = v_net.links[virtual_link]["bw"]
        offsets: dict[tuple[int, int], int] = {}
        for physical_link in path_to_links(physical_path):
            offsets[physical_link] = (
                demand - p_net.links[physical_link]["bw"]
            )
        feasible = all(offset <= 0 for offset in offsets.values())
        return feasible, {
            "link_level": {"bw": offsets},
            "path_level": {},
        }


class FakeResourceUpdator:
    def update_link_resources(
        self,
        p_net,
        physical_link,
        resources,
        operator="-",
        safe=True,
    ):
        for name, amount in resources.items():
            current = p_net.links[physical_link][name]
            if operator == "-":
                if safe and current < amount:
                    raise RuntimeError("insufficient benchmark resource")
                p_net.links[physical_link][name] = current - amount
            elif operator == "+":
                p_net.links[physical_link][name] = current + amount
            else:
                raise RuntimeError("unsupported benchmark operator")


@dataclass
class PythonFixture:
    mapper: Any
    virtual_network: FakeNetwork
    physical_network: FakeNetwork
    solution: dict[str, Any]
    physical_edges: list[tuple[int, int]]
    target: int


def make_python_fixture(LinkMapper, candidate_count: int) -> PythonFixture:
    virtual_graph = nx.Graph()
    virtual_graph.add_edge(0, 1, bw=2)
    virtual_network = FakeNetwork(virtual_graph)

    target = candidate_count + 1
    physical_graph = nx.Graph()
    physical_graph.add_nodes_from(range(target + 1))
    physical_edges: list[tuple[int, int]] = []
    for index in range(candidate_count):
        intermediate = index + 1
        capacity = 2 if index + 1 == candidate_count else 1
        physical_edges.append((0, intermediate))
        physical_edges.append((intermediate, target))
        physical_graph.add_edge(0, intermediate, bw=capacity)
        physical_graph.add_edge(intermediate, target, bw=capacity)
    physical_network = FakeNetwork(physical_graph)

    mapper = LinkMapper(
        FakeConstraintChecker(),
        FakeResourceUpdator(),
        FakeTopologyAnalyzer(),
        [FakeLinkResource("bw")],
        ["bw"],
        {"link_level": {"bw": 0}, "path_level": {}},
        False,
    )
    solution: dict[str, Any] = {
        "result": False,
        "place_result": True,
        "route_result": True,
        "link_paths": {},
        "link_paths_info": {},
        "v_net_constraint_offsets": {
            "node_level": {},
            "link_level": {},
            "path_level": {},
        },
        "v_net_constraint_violations": {
            "node_level": {},
            "link_level": {},
            "path_level": {},
        },
        "v_net_total_hard_constraint_violation": 0.0,
        "v_net_single_step_constraint_offset": {
            "node_level": {},
            "link_level": {},
            "path_level": {},
        },
    }
    return PythonFixture(
        mapper,
        virtual_network,
        physical_network,
        solution,
        physical_edges,
        target,
    )


def double_token(value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"d:{bits:016x}"


def number_token(value: Any) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return double_token(value)
    raise RuntimeError("LinkMapper benchmark entered a non-numeric lane")


def link_token(link: tuple[int, int]) -> str:
    return f"({link[0]},{link[1]})"


def values_payload(values: dict[str, Any]) -> str:
    if not values:
        return "{}"
    if set(values) != {"bw"}:
        raise RuntimeError("unexpected Python Solution value")
    return "{bw=" + number_token(values["bw"]) + "}"


def constraint_table_payload(table: dict[Any, dict[str, Any]]) -> str:
    return "[" + ",".join(
        f"{link_token(link)}:{values_payload(values)}"
        for link, values in table.items()
    ) + "]"


def state_payload(
    fixture: PythonFixture,
    routed: bool,
    check_info: dict[str, Any],
) -> str:
    solution = fixture.solution
    link_offsets = check_info["link_level"]["bw"]
    feasible = all(offset <= 0 for offset in link_offsets.values())
    physical = ",".join(
        f"{link_token(link)}="
        f"{number_token(fixture.physical_network.links[link]['bw'])}"
        for link in fixture.physical_edges
    )
    paths = ",".join(
        f"{link_token(virtual_link)}:["
        + ",".join(link_token(link) for link in path)
        + "]"
        for virtual_link, path in solution["link_paths"].items()
    )
    info = ",".join(
        f"{link_token(key[0])}@{link_token(key[1])}:"
        f"{values_payload(values)}"
        for key, values in solution["link_paths_info"].items()
    )
    checked_links = ",".join(
        f"{link_token(link)}:{values_payload({'bw': offset})}"
        for link, offset in link_offsets.items()
    )
    offsets = solution["v_net_constraint_offsets"]
    violations = solution["v_net_constraint_violations"]
    flags = ",".join(
        "1" if solution[name] else "0"
        for name in ("result", "place_result", "route_result")
    )
    return (
        f"routed={1 if routed else 0};placeholder=0;"
        f"feasible={1 if feasible else 0};phys=[{physical}];"
        f"paths=[{paths}];info=[{info}];"
        f"check_link=[{checked_links}];"
        f"check_path={values_payload(check_info['path_level'])};"
        f"offset_link={constraint_table_payload(offsets['link_level'])};"
        f"offset_path={constraint_table_payload(offsets['path_level'])};"
        f"violation_link="
        f"{constraint_table_payload(violations['link_level'])};"
        f"violation_path="
        f"{constraint_table_payload(violations['path_level'])};"
        f"total="
        f"{double_token(solution['v_net_total_hard_constraint_violation'])};"
        f"flags={flags}"
    )


def fingerprint(value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def run_python_once(
    LinkMapper, candidate_count: int
) -> tuple[int, str]:
    fixture = make_python_fixture(LinkMapper, candidate_count)
    begin = time.perf_counter_ns()
    routed, check_info = fixture.mapper.route(
        fixture.virtual_network,
        fixture.physical_network,
        (0, 1),
        (0, fixture.target),
        fixture.solution,
        shortest_method="all_shortest",
        k=candidate_count,
        rank_path_func=None,
        if_allow_constraint_violation=False,
        if_record_constraint_violation=True,
    )
    elapsed = time.perf_counter_ns() - begin
    expected_path = [
        (0, candidate_count),
        (candidate_count, fixture.target),
    ]
    if not routed or fixture.solution["link_paths"].get((0, 1)) != expected_path:
        raise RuntimeError("Python all-shortest candidate order drifted")
    return elapsed, state_payload(fixture, routed, check_info)


def parse_cpp(
    benchmark: pathlib.Path,
    candidate_count: int,
    candidate_workers: int,
) -> dict[str, int]:
    process = subprocess.run(
        [str(benchmark), str(candidate_count), str(candidate_workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"C++ LinkMapper benchmark failed: {process.stderr.strip()}"
        )
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed C++ benchmark output: {line!r}")
        fields[key] = value
    required = {
        "protocol",
        "kind",
        "candidate_paths",
        "physical_edges",
        "candidate_workers",
        "type_tag",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "status",
    }
    if (
        set(fields) != required
        or fields["protocol"] != "1"
        or fields["kind"] != "link_mapper_safe_all_shortest"
        or int(fields["candidate_paths"]) != candidate_count
        or int(fields["physical_edges"]) != candidate_count * 2
        or int(fields["candidate_workers"]) != candidate_workers
        or fields["type_tag"] != "ordered_link_route_state_v1"
        or fields["status"] != "PASS"
    ):
        raise RuntimeError(f"invalid C++ benchmark protocol: {fields!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
    }


def validate_output(
    current: dict[str, int],
    expected_checksum: int,
    expected_bytes: int,
    label: str,
) -> None:
    if (
        current["checksum"] != expected_checksum
        or current["output_bytes"] != expected_bytes
    ):
        raise RuntimeError(f"{label} complete route-state drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-paths", type=int, default=1024)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if (
        args.candidate_paths < 2
        or not args.workers
        or any(worker < 0 for worker in args.workers)
        or args.warmups < 0
        or args.repetitions <= 0
    ):
        raise RuntimeError("invalid LinkMapper benchmark protocol")

    LinkMapper = load_link_mapper(args.source)
    _reference_ns, reference_payload = run_python_once(
        LinkMapper, args.candidate_paths
    )
    expected_checksum, expected_bytes = fingerprint(reference_payload)

    for workers in args.workers:
        validate_output(
            parse_cpp(args.benchmark, args.candidate_paths, workers),
            expected_checksum,
            expected_bytes,
            f"worker {workers} pre-timing gate",
        )

    for _ in range(args.warmups):
        _elapsed, payload = run_python_once(LinkMapper, args.candidate_paths)
        if fingerprint(payload) != (expected_checksum, expected_bytes):
            raise RuntimeError("Python warm-up output drift")
    python_samples: list[int] = []
    for _ in range(args.repetitions):
        elapsed, payload = run_python_once(LinkMapper, args.candidate_paths)
        if fingerprint(payload) != (expected_checksum, expected_bytes):
            raise RuntimeError("Python timed output drift")
        python_samples.append(elapsed)
    python_ns = int(statistics.median(python_samples))

    rows = []
    for workers in args.workers:
        for _ in range(args.warmups):
            validate_output(
                parse_cpp(args.benchmark, args.candidate_paths, workers),
                expected_checksum,
                expected_bytes,
                f"worker {workers} warm-up",
            )
        cpp_samples: list[int] = []
        for _ in range(args.repetitions):
            current = parse_cpp(
                args.benchmark, args.candidate_paths, workers
            )
            validate_output(
                current,
                expected_checksum,
                expected_bytes,
                f"worker {workers} sample",
            )
            cpp_samples.append(current["elapsed_ns"])
        cpp_ns = int(statistics.median(cpp_samples))
        speedup = python_ns / cpp_ns
        if speedup <= 1.0:
            raise RuntimeError(
                f"worker {workers} C++ did not beat Python: {speedup:.3f}x"
            )
        row = {
            "kind": "link_mapper_safe_all_shortest",
            "candidate_paths": args.candidate_paths,
            "physical_edges": args.candidate_paths * 2,
            "candidate_workers": workers,
            "type_tag": "ordered_link_route_state_v1",
            "python_median_ms": python_ns / 1e6,
            "cpp_median_ms": cpp_ns / 1e6,
            "speedup": speedup,
            "checksum": expected_checksum,
            "output_bytes": expected_bytes,
        }
        rows.append(row)
        print(
            f"link_mapper_safe_all_shortest/w{workers}: "
            f"python={row['python_median_ms']:.6f} ms, "
            f"cpp={row['cpp_median_ms']:.6f} ms, "
            f"speedup={speedup:.3f}x"
        )

    report = {
        "source_sha256": SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(
            args.benchmark.read_bytes()
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "candidate_paths": args.candidate_paths,
            "physical_edges": args.candidate_paths * 2,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "shortest_method": "all_shortest",
            "allow_constraint_violation": False,
            "record_constraint_violation": True,
            "topology_constraint_workers": 1,
            "fixture_and_preparation_excluded": True,
            "cpp_process_startup_excluded": True,
            "fingerprint_excluded": True,
            "complete_route_state_gate_before_timing": True,
            "caller_configured_candidate_workers": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(f"LinkMapper benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
