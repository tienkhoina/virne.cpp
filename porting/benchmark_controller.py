#!/usr/bin/env python3
"""Compact deploy+release benchmark for the pinned Controller lifecycle."""

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
from typing import Any

import networkx as nx


SOURCE_SHA256 = "352552b0312d18b42eb6904629ed43c76df2d9c844df59fe51ab8653ffac02a5"
RESOURCE_SOURCE_SHA256 = "9b7c14f8c6eaa5e8bc50a723b727fec7eff8f1c7d05f7bef0da0cc941c15ac85"
ITEM_COUNT = 8192
WARMUP_COUNT = 1
SAMPLE_COUNT = 3
WORKERS = (1, 2, 8)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1
OUTPUT_BYTES = ITEM_COUNT * 2 * 8


def load_lifecycle(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Controller source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Controller"
    ]
    if len(classes) != 1:
        raise RuntimeError("Controller class inventory drift")
    methods = [
        node for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        node.name in {"deploy", "release"}
    ]
    if [method.name for method in methods] != ["deploy", "release"]:
        raise RuntimeError("Controller lifecycle method inventory drift")
    isolated_class = classes[0]
    isolated_class.name = "OracleController"
    isolated_class.body = methods
    isolated = ast.fix_missing_locations(ast.Module(
        body=[isolated_class], type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "PhysicalNetwork": Any,
        "Solution": Any,
        "VirtualNetwork": Any,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["OracleController"]


def load_resource_updator(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != RESOURCE_SOURCE_SHA256:
        raise RuntimeError(f"ResourceUpdator source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResourceUpdator"
    ]
    if len(classes) != 1:
        raise RuntimeError("ResourceUpdator class inventory drift")
    selected_names = {
        "__init__", "update_resource", "update_node_resources",
        "update_link_resources",
    }
    methods = [
        node for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        node.name in selected_names
    ]
    if [method.name for method in methods] != [
            "__init__", "update_resource", "update_node_resources",
            "update_link_resources"]:
        raise RuntimeError("ResourceUpdator lifecycle inventory drift")
    isolated_class = classes[0]
    isolated_class.name = "OracleResourceUpdator"
    isolated_class.body = methods
    isolated = ast.fix_missing_locations(ast.Module(
        body=[isolated_class], type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "BaseNetwork": Any,
        "PhysicalNetwork": Any,
        "Optional": __import__("typing").Optional,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["OracleResourceUpdator"]


def node_capacity(index: int) -> float:
    return 1000.0 + (index % 97) * 0.25


def link_capacity(index: int) -> float:
    return 2000.0 + (index % 89) * 0.5


def node_demand(index: int) -> float:
    return 0.5 + (index % 13) * 0.125


def link_demand(index: int) -> float:
    return 0.25 + (index % 11) * 0.0625


class OraclePhysicalNetwork(nx.Graph):
    @property
    def links(self):
        return self.edges


def make_physical_network():
    physical = OraclePhysicalNetwork()
    physical.add_nodes_from(
        (index, {"cpu": node_capacity(index)})
        for index in range(ITEM_COUNT))
    physical.add_edges_from(
        (index, (index + 1) % ITEM_COUNT,
         {"bandwidth": link_capacity(index)})
        for index in range(ITEM_COUNT))
    return physical


def make_solution():
    solution = {
        "result": True,
        "node_slots": {},
        "node_slots_info": {},
        "link_paths": {},
        "link_paths_info": {},
    }
    for index in range(ITEM_COUNT):
        solution["node_slots"][index] = index
        solution["node_slots_info"][(index, index)] = {
            "cpu": node_demand(index)}
        virtual_link = (index, (index + 1) % ITEM_COUNT)
        physical_link = virtual_link
        solution["link_paths"][virtual_link] = [physical_link]
        solution["link_paths_info"][(virtual_link, physical_link)] = {
            "bandwidth": link_demand(index)}
    return solution


def restore_baseline(physical_network):
    for index in range(ITEM_COUNT):
        physical_network.nodes[index]["cpu"] = node_capacity(index)
    for index in range(ITEM_COUNT):
        physical_network.links[(index, (index + 1) % ITEM_COUNT)][
            "bandwidth"] = link_capacity(index)


def hash_double(checksum: int, value: float) -> int:
    for byte in struct.pack("<d", value):
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & U64_MASK
    return checksum


def output_checksum(physical_network) -> int:
    checksum = FNV_OFFSET
    for index in range(ITEM_COUNT):
        checksum = hash_double(
            checksum, physical_network.nodes[index]["cpu"])
    for index in range(ITEM_COUNT):
        checksum = hash_double(
            checksum,
            physical_network.links[(index, (index + 1) % ITEM_COUNT)][
                "bandwidth"],
        )
    return checksum


def expected_checksum(deployed: bool) -> int:
    checksum = FNV_OFFSET
    for index in range(ITEM_COUNT):
        value = node_capacity(index)
        if deployed:
            value -= node_demand(index)
        checksum = hash_double(checksum, value)
    for index in range(ITEM_COUNT):
        value = link_capacity(index)
        if deployed:
            value -= link_demand(index)
        checksum = hash_double(checksum, value)
    return checksum


def run_python_benchmark(OracleController, OracleResourceUpdator):
    lifecycle = object.__new__(OracleController)
    lifecycle.resource_updator = OracleResourceUpdator([])
    physical_network = make_physical_network()
    solution = make_solution()
    expected_deployed = expected_checksum(True)
    expected_restored = expected_checksum(False)

    restore_baseline(physical_network)
    if not lifecycle.deploy(None, physical_network, solution):
        raise RuntimeError("Python Controller deploy returned false")
    if output_checksum(physical_network) != expected_deployed:
        raise RuntimeError("Python deployed checksum mismatch")
    if not lifecycle.release(None, physical_network, solution):
        raise RuntimeError("Python Controller release returned false")
    if output_checksum(physical_network) != expected_restored:
        raise RuntimeError("Python restored checksum mismatch")

    for _ in range(WARMUP_COUNT):
        restore_baseline(physical_network)
        lifecycle.deploy(None, physical_network, solution)
        lifecycle.release(None, physical_network, solution)
        if output_checksum(physical_network) != expected_restored:
            raise RuntimeError("Python warmup checksum mismatch")

    samples = []
    for _ in range(SAMPLE_COUNT):
        restore_baseline(physical_network)
        begin = time.perf_counter_ns()
        deployed = lifecycle.deploy(None, physical_network, solution)
        released = lifecycle.release(None, physical_network, solution)
        elapsed = time.perf_counter_ns() - begin
        if not deployed or not released:
            raise RuntimeError("Python lifecycle returned false")
        if output_checksum(physical_network) != expected_restored:
            raise RuntimeError("Python timed checksum mismatch")
        samples.append(elapsed / 1_000_000.0)
    return {
        "median_ms": statistics.median(samples),
        "deployed_checksum": expected_deployed,
        "checksum": expected_restored,
    }


def parse_native_row(line: str):
    result = {}
    for field in line.split(";"):
        name, value = field.split("=", 1)
        result[name] = value
    expected_fields = [
        "workers", "items", "operations", "deployed_checksum",
        "checksum", "output_bytes", "median_ms",
    ]
    if list(result) != expected_fields:
        raise RuntimeError(f"native row schema drift: {list(result)}")
    return {
        "workers": int(result["workers"]),
        "items": int(result["items"]),
        "operations": int(result["operations"]),
        "deployed_checksum": int(result["deployed_checksum"]),
        "checksum": int(result["checksum"]),
        "output_bytes": int(result["output_bytes"]),
        "median_ms": float(result["median_ms"]),
    }


def run_native_benchmark(executable: pathlib.Path):
    process = subprocess.run(
        [str(executable), "benchmark"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Controller benchmark failed: {process.stderr.strip()}")
    rows = [parse_native_row(line) for line in process.stdout.splitlines()]
    if [row["workers"] for row in rows] != list(WORKERS):
        raise RuntimeError("native worker inventory drift")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--resource-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    OracleController = load_lifecycle(args.source)
    OracleResourceUpdator = load_resource_updator(args.resource_source)
    python_result = run_python_benchmark(
        OracleController, OracleResourceUpdator)
    native_rows = run_native_benchmark(args.benchmark)

    for row in native_rows:
        if row["items"] != ITEM_COUNT or row["operations"] != ITEM_COUNT * 4:
            raise RuntimeError("native workload cardinality drift")
        if row["output_bytes"] != OUTPUT_BYTES:
            raise RuntimeError("native output byte gate mismatch")
        if row["deployed_checksum"] != python_result["deployed_checksum"]:
            raise RuntimeError("native deployed checksum mismatch")
        if row["checksum"] != python_result["checksum"]:
            raise RuntimeError("native restored checksum mismatch")
        if not 0.0 < row["median_ms"] < python_result["median_ms"]:
            raise RuntimeError(
                f"Controller workers={row['workers']} did not beat Python")
        row["speedup"] = python_result["median_ms"] / row["median_ms"]

    report = {
        "component": "core.controller.Controller",
        "source_sha256": SOURCE_SHA256.upper(),
        "resource_source_sha256": RESOURCE_SOURCE_SHA256.upper(),
        "workload": "deploy_release",
        "items": ITEM_COUNT,
        "operations": ITEM_COUNT * 4,
        "warmup": WARMUP_COUNT,
        "samples": SAMPLE_COUNT,
        "output_bytes": OUTPUT_BYTES,
        "deployed_checksum": python_result["deployed_checksum"],
        "checksum": python_result["checksum"],
        "python_median_ms": python_result["median_ms"],
        "native": native_rows,
        "oracle_boundary": (
            "AST-isolated exact Controller.deploy/release and exact "
            "ResourceUpdator scalar lifecycle methods over the same one-node/"
            "one-link-resource schema; no routing, solver, system, RL or ML."),
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
