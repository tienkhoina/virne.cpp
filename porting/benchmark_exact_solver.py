#!/usr/bin/env python3
"""Focused one-resource Python/C++ timing for the original MIP leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import statistics
import subprocess
import time

import networkx as nx
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


SOURCE_SHA256 = (
    "7e68586641c81aa556110a8ceaf3ac1a96addc564890d6b6fbe3f1d984693b2a"
)


class _Solver:
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        del recorder, counter, logger, config, kwargs
        self.controller = controller


class _Registry:
    @classmethod
    def register(cls, **kwargs):
        del kwargs
        return lambda value: value


class _Solution:
    @classmethod
    def from_v_net(cls, virtual_network):
        del virtual_network
        return {
            "result": False,
            "place_result": True,
            "route_result": True,
            "node_slots": {},
            "link_paths": {},
            "node_slots_info": {},
            "link_paths_info": {},
        }


class _QuietPprint:
    @staticmethod
    def pprint(value):
        del value


class _Network(nx.Graph):
    @property
    def links(self):
        return self.edges


class _Controller:
    @staticmethod
    def construct_candidates_dict(virtual_network, physical_network):
        return {
            virtual_node: [
                physical_node
                for physical_node in physical_network.nodes
                if virtual_network.nodes[virtual_node]["cpu"]
                <= physical_network.nodes[physical_node]["cpu"]
            ]
            for virtual_node in virtual_network.nodes
        }


def _load_original_mip(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"exact MIP source drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    mip_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MipSolver"
    )
    namespace = {
        "Solver": _Solver,
        "SolverRegistry": _Registry,
        "Solution": _Solution,
        "SolutionStepEnvironment": object,
        "pywraplp": pywraplp,
        "cp_model": cp_model,
        "pprint": _QuietPprint,
    }
    isolated = ast.fix_missing_locations(
        ast.Module(body=[mip_class], type_ignores=[])
    )
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["MipSolver"]


def _fixture():
    physical = _Network()
    physical.add_nodes_from((node, {"cpu": 10}) for node in range(4))
    physical.add_edges_from(
        (
            (0, 1, {"bw": 10}),
            (1, 2, {"bw": 10}),
            (2, 3, {"bw": 10}),
            (0, 3, {"bw": 10}),
        )
    )
    virtual = _Network()
    virtual.add_node(0, cpu=4)
    virtual.add_node(1, cpu=5)
    virtual.add_edge(0, 1, bw=4)
    return virtual, physical


def _python_benchmark(mip_class, samples: int):
    virtual, physical = _fixture()
    solver = mip_class(_Controller(), None, None, None, None)

    def run_once():
        begin = time.perf_counter_ns()
        solution = solver.solve({"v_net": virtual, "p_net": physical})
        elapsed = time.perf_counter_ns() - begin
        path_edges = sum(len(path) for path in solution["link_paths"].values())
        signature = {
            "accepted": bool(solution["result"]),
            "node_slots": len(solution["node_slots"]),
            "path_edges": path_edges,
        }
        return elapsed, signature

    run_once()
    rows = [run_once() for _ in range(samples)]
    signatures = {json.dumps(row[1], sort_keys=True) for row in rows}
    if len(signatures) != 1:
        raise RuntimeError("Python MIP structural output is not stable")
    durations = [row[0] for row in rows]
    return {
        "median_ns": int(statistics.median(durations)),
        "min_ns": min(durations),
        **rows[0][1],
    }


def _cpp_benchmark(executable: pathlib.Path):
    completed = subprocess.run(
        [str(executable.resolve()), "--benchmark"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=root.parent / "virne" / "virne" / "solver" / "exact" / "mip.py",
    )
    parser.add_argument(
        "--cpp",
        type=pathlib.Path,
        default=root
        / "build-clangcl-exact-unit"
        / "porting"
        / "vne_exact_solver_unit.exe",
    )
    parser.add_argument("--samples", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.samples <= 0:
        raise ValueError("samples must be positive")

    python_result = _python_benchmark(
        _load_original_mip(arguments.source), arguments.samples
    )
    cpp_result = _cpp_benchmark(arguments.cpp)
    for field in ("accepted", "node_slots", "path_edges"):
        if python_result[field] != cpp_result[field]:
            raise RuntimeError(
                f"structural differential failed for {field}: "
                f"Python={python_result[field]!r}, C++={cpp_result[field]!r}"
            )
    report = {
        "fixture": "4-node-ring_2-vnodes_1-cpu_1-bw",
        "samples": arguments.samples,
        "python_ortools": getattr(__import__("ortools"), "__version__", "unknown"),
        "python": python_result,
        "cpp": cpp_result,
        "cpp_speedup": python_result["median_ns"] / cpp_result["median_ns"],
    }
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
