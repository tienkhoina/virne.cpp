#!/usr/bin/env python3
"""Single compact checksum-gated benchmark for the frozen NodeRank leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import statistics
import struct
import subprocess
import time
from typing import Any, Dict, Type, Union

import networkx as nx
import numpy as np


SOURCE_SHA256 = "428f0deb188e685a9f3eb6177a70df533519f62c560284063fc4cf6ec7624fff"
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def load_ffd(source: pathlib.Path):
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"NodeRank source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    wanted = {
        "NodeRankRegistry", "NodeRank", "OrderNodeRank", "RandomNodeRank",
        "FFDNodeRank", "NRMNodeRank", "DegreeWeightedResoureNodeRank",
        "GRCNodeRank", "RWNodeRank", "NPSNodeRank",
    }
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "abc": __import__("abc"),
        "Any": Any,
        "BaseNetwork": Any,
        "Dict": Dict,
        "Type": Type,
        "Union": Union,
        "np": np,
        "nx": nx,
    }
    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[]))
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["FFDNodeRank"]


def fixture_value(resource: int, node: int) -> float:
    return ((resource + 1) * 4 + (node % 1024)) / 4.0


def expected_score(resource_count: int, node: int) -> float:
    numerator = sum(
        (resource + 1) * 4 + (node % 1024)
        for resource in range(resource_count)
    )
    return numerator / 4.0


class BenchmarkNetwork:
    def __init__(self, node_count: int, resource_count: int):
        self.nodes = range(node_count)
        self.num_nodes = node_count
        self._token = object()
        self._rows = [
            [fixture_value(resource, node) for node in range(node_count)]
            for resource in range(resource_count)
        ]

    def get_node_attrs(self, _kind):
        return self._token

    def get_node_attrs_data(self, token):
        if token is not self._token:
            raise RuntimeError("unexpected benchmark attribute token")
        return self._rows


def bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def fnv_mix(value: int, checksum: int) -> int:
    for shift in range(0, 64, 8):
        checksum ^= (value >> shift) & 0xFF
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def validate_and_checksum(
    ranking,
    node_count: int,
    resource_count: int,
) -> int:
    if len(ranking) != node_count:
        raise AssertionError("Python NodeRank benchmark length drift")
    checksum = FNV_OFFSET
    for node, value in ranking.items():
        expected = expected_score(resource_count, int(node))
        if bits(value) != bits(expected):
            raise AssertionError("Python NodeRank benchmark value drift")
        checksum = fnv_mix(int(node), checksum)
        checksum = fnv_mix(bits(value), checksum)
    return checksum


def native_run(
    executable: pathlib.Path,
    workers: int,
    warmups: int,
    samples: int,
    nodes: int,
    resources: int,
):
    completed = subprocess.run(
        [
            str(executable), str(workers), str(warmups), str(samples),
            str(nodes), str(resources),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--nodes", type=int, default=131072)
    parser.add_argument("--resources", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    args = parser.parse_args()

    ranker = load_ffd(args.source)()
    network = BenchmarkNetwork(args.nodes, args.resources)
    for _ in range(args.warmups):
        validate_and_checksum(
            ranker.rank(network, sort=False), args.nodes, args.resources
        )

    python_samples: list[int] = []
    python_checksum = 0
    for _ in range(args.samples):
        start = time.perf_counter_ns()
        ranking = ranker.rank(network, sort=False)
        stop = time.perf_counter_ns()
        python_samples.append(stop - start)
        current = validate_and_checksum(ranking, args.nodes, args.resources)
        if python_checksum and current != python_checksum:
            raise AssertionError("Python checksum changed across samples")
        python_checksum = current
    python_median_ns = int(statistics.median(python_samples))

    native: dict[str, Any] = {}
    expected_bytes = args.nodes * 16
    for workers in args.workers:
        row = native_run(
            args.benchmark.resolve(), workers, args.warmups, args.samples,
            args.nodes, args.resources,
        )
        if (
            row["entries"] != args.nodes
            or row["bytes"] != expected_bytes
            or row["checksum"] != python_checksum
        ):
            raise AssertionError(f"native output gate failed at workers={workers}")
        row["median_ms"] = row["median_ns"] / 1_000_000.0
        row["speedup_vs_python"] = python_median_ns / row["median_ns"]
        if row["speedup_vs_python"] <= 1.0:
            raise AssertionError(
                f"native NodeRank is not faster at workers={workers}: {row}")
        native[str(workers)] = row

    result = {
        "component": "solver.rank.NodeRank",
        "source_sha256": SOURCE_SHA256.upper(),
        "fixture": {
            "nodes": args.nodes,
            "resources": args.resources,
            "sort": False,
            "warmups": args.warmups,
            "samples": args.samples,
        },
        "output": {
            "entries": args.nodes,
            "bytes": expected_bytes,
            "checksum": python_checksum,
        },
        "python": {
            "samples_ns": python_samples,
            "median_ns": python_median_ns,
            "median_ms": python_median_ns / 1_000_000.0,
        },
        "cpp": native,
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
