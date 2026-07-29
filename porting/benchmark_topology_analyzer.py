#!/usr/bin/env python3
"""One checksum-gated frozen benchmark for TopologyAnalyzer."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
from collections import deque
from itertools import islice
from typing import Any

import networkx as nx


SOURCE_SHA256 = "665519c5c4bf50c2318e2d22d30679881dda0edcbc0b618c4fb2055ac5a01b28"
NODE_COUNT = 512
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def load_analyzer(source: pathlib.Path):
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


def make_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(NODE_COUNT))
    graph.add_edges_from((node, node + 1) for node in range(NODE_COUNT - 1))
    return graph


def make_queries(count: int) -> list[tuple[int, int]]:
    result = []
    for index in range(count):
        source = (index * 37 + 11) % NODE_COUNT
        target = (index * 91 + 257) % NODE_COUNT
        if target == source:
            target = (target + 1) % NODE_COUNT
        result.append((source, target))
    return result


def mix_u64(value: int, current: int) -> int:
    for _ in range(8):
        current ^= value & 0xFF
        current = (current * FNV_PRIME) & MASK64
        value >>= 8
    return current


def fingerprint(results: list[list[list[int]]]) -> dict[str, int]:
    checksum = mix_u64(len(results), FNV_OFFSET)
    path_count = 0
    vertex_count = 0
    for request_paths in results:
        checksum = mix_u64(len(request_paths), checksum)
        path_count += len(request_paths)
        for path in request_paths:
            checksum = mix_u64(len(path), checksum)
            vertex_count += len(path)
            for vertex in path:
                checksum = mix_u64(vertex, checksum)
    return {
        "checksum": checksum,
        "path_count": path_count,
        "vertex_count": vertex_count,
    }


def python_run(analyzer, graph, queries):
    return [
        analyzer.find_shortest_paths(
            None,
            graph,
            (0, 1),
            pair,
            method="first_shortest")
        for pair in queries
    ]


def run_cpp(binary: pathlib.Path, workers: int, count: int,
            warmups: int, repetitions: int) -> dict[str, int]:
    process = subprocess.run(
        [str(binary), str(workers), str(count), str(warmups), str(repetitions)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    return json.loads(process.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    TopologyAnalyzer = load_analyzer(args.source)
    analyzer = TopologyAnalyzer(None, [])
    graph = make_graph()
    queries = make_queries(args.count)

    expected = None
    for _ in range(args.warmups):
        current = fingerprint(python_run(analyzer, graph, queries))
        expected = current if expected is None else expected
        if current != expected:
            raise RuntimeError("Python warm-up output changed")

    samples = []
    for _ in range(args.repetitions):
        begin = time.perf_counter_ns()
        results = python_run(analyzer, graph, queries)
        elapsed = time.perf_counter_ns() - begin
        current = fingerprint(results)
        expected = current if expected is None else expected
        if current != expected:
            raise RuntimeError("Python measured output changed")
        samples.append(elapsed)
    python_median = int(statistics.median(samples))

    cpp_rows = []
    for workers in (1, 2, 8):
        row = run_cpp(
            args.binary,
            workers,
            args.count,
            args.warmups,
            args.repetitions)
        row_fingerprint = {
            key: row[key]
            for key in ("checksum", "path_count", "vertex_count")
        }
        if row_fingerprint != expected:
            raise RuntimeError(
                f"C++ workers={workers} fingerprint mismatch: "
                f"{row_fingerprint} != {expected}")
        row["speedup_vs_python"] = python_median / row["cpp_median_ns"]
        cpp_rows.append(row)

    report = {
        "case": "first_shortest_batch",
        "component": "core.controller.TopologyAnalyzer",
        "count": args.count,
        "cpp": cpp_rows,
        "fingerprint": expected,
        "python_median_ns": python_median,
        "repetitions": args.repetitions,
        "status": "PASS",
        "warmups": args.warmups,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
