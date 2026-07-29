#!/usr/bin/env python3
"""One-shot frozen runtime comparison for the pinned LinkRank leaf."""

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
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


SOURCE_SHA256 = "c4d52bd389a004a91d8fcc7e0827bb9620252d9619d4f9121d11ba4e17ff0880"
EDGE_COUNT = 131072
RESOURCE_COUNT = 8
WARMUPS = 1
SAMPLES = 3
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def load_ffd(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"LinkRank source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if [node.name for node in classes] != [
            "LinkRank", "OrderLinkRank", "FFDLinkRank"]:
        raise RuntimeError("LinkRank class inventory drift")
    isolated = ast.fix_missing_locations(ast.Module(
        body=classes, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "ABC": ABC,
        "Any": Any,
        "BaseNetwork": Any,
        "Dict": Dict,
        "abstractmethod": abstractmethod,
        "np": np,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["FFDLinkRank"]


def resource_value(resource: int, edge: int) -> int:
    return (edge * (resource * 2 + 3) + resource * 17) % 100003 - 50000


class IntendedNetwork:
    def __init__(self):
        self.links = [(edge, edge + 1) for edge in range(EDGE_COUNT)]
        self.rows = [
            [resource_value(resource, edge) for edge in range(EDGE_COUNT)]
            for resource in range(RESOURCE_COUNT)
        ]
        self.token = object()

    def get_attrs(self, owner, kind):
        if owner != "link" or kind != "resource":
            raise RuntimeError("link rank benchmark selection drift")
        return self.token

    def get_link_attrs_data(self, attrs):
        if attrs is not self.token:
            raise RuntimeError("link rank benchmark binding drift")
        return self.rows


def digest(ranking) -> dict[str, int]:
    checksum = FNV_OFFSET
    byte_count = 0

    def append_u64(value: int):
        nonlocal checksum, byte_count
        for shift in range(0, 64, 8):
            checksum ^= (value >> shift) & 0xFF
            checksum = (checksum * FNV_PRIME) & MASK64
            byte_count += 1

    for (source, target), score in ranking.items():
        append_u64(source)
        append_u64(target)
        bits = struct.unpack("<Q", struct.pack("<d", float(score)))[0]
        append_u64(bits)
    return {
        "checksum": checksum,
        "bytes": byte_count,
        "entries": len(ranking),
    }


def python_benchmark(ffd_class) -> tuple[float, dict[str, int]]:
    network = IntendedNetwork()
    ranker = ffd_class()
    baseline = digest(ranker.rank(network, sort=True))
    for _ in range(WARMUPS):
        if digest(ranker.rank(network, sort=True)) != baseline:
            raise RuntimeError("Python LinkRank warm-up output drift")
    samples = []
    for _ in range(SAMPLES):
        begin = time.perf_counter_ns()
        ranking = ranker.rank(network, sort=True)
        elapsed = time.perf_counter_ns() - begin
        samples.append(elapsed / 1_000_000.0)
        if digest(ranking) != baseline:
            raise RuntimeError("Python LinkRank sample output drift")
    return statistics.median(samples), baseline


def native_benchmark(binary: pathlib.Path):
    process = subprocess.run(
        [str(binary.resolve()), "benchmark"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"native LinkRank benchmark failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        fields = dict(item.split("=", 1) for item in line.split("\t"))
        workers = int(fields.pop("workers"))
        result[workers] = {
            "median_ms": float(fields["median_ms"]),
            "checksum": int(fields["checksum"]),
            "bytes": int(fields["bytes"]),
            "entries": int(fields["entries"]),
        }
    if list(result) != [1, 2, 8]:
        raise RuntimeError(f"native worker inventory drift: {list(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    python_ms, expected = python_benchmark(load_ffd(args.source))
    native = native_benchmark(args.binary)
    for workers, record in native.items():
        actual = {key: record[key] for key in ("checksum", "bytes", "entries")}
        if actual != expected:
            raise RuntimeError(
                f"native LinkRank output drift at workers={workers}: "
                f"expected={expected}, actual={actual}")
        record["speedup_vs_python"] = python_ms / record["median_ms"]

    report = {
        "component": "solver.rank.LinkRank",
        "source_sha256": SOURCE_SHA256.upper(),
        "fixture": {
            "edges": EDGE_COUNT,
            "resources": RESOURCE_COUNT,
            "sort": True,
            "warmups": WARMUPS,
            "samples": SAMPLES,
        },
        "output": expected,
        "python_median_ms": python_ms,
        "cpp": {str(workers): native[workers] for workers in (1, 2, 8)},
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
