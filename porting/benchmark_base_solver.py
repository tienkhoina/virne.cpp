#!/usr/bin/env python3
"""Compact Python/C++ benchmark for the BaseSolver startup holder."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import pathlib
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Type


SOURCE_SHA256 = (
    "196573e631654a6d14888685b93977601412f110a9b5620398850cce121a2806"
)
DEFAULT_WORKERS = (1, 2, 8)


def load_leaf(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"BaseSolver source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if [node.name for node in classes] != ["Solver", "SolverRegistry"]:
        raise RuntimeError("BaseSolver class inventory drift")

    class Dummy:
        pass

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "abc": __import__("abc"),
        "os": os,
        "Controller": Dummy,
        "Recorder": Dummy,
        "Counter": Dummy,
        "Logger": Dummy,
        "Solution": Dummy,
        "SolutionStepEnvironment": Dummy,
        "Optional": Optional,
        "Dict": Dict,
        "Type": Type,
    }
    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[]))
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Solver"], namespace["SolverRegistry"]


def make_inputs(count: int):
    result = []
    for index in range(count):
        result.append(SimpleNamespace(
            experiment=SimpleNamespace(
                seed=index,
                save_root_dir="/benchmark",
                run_id=f"run-{index}",
            ),
            solver=SimpleNamespace(
                solver_name=f"solver-{index % 32}",
                reusable=index % 2 != 0,
                node_ranking_method="nps",
                link_ranking_method="ffd",
                matching_mathod="l2s2",
                shortest_method="all_shortest",
                k_shortest=10 + index % 7,
                allow_rejection=index % 3 == 0,
                allow_revocable=index % 5 == 0,
            ),
            verbose=index % 5,
        ))
    return result


class HashWriter:
    def __init__(self):
        self.hash = 14695981039346656037
        self.bytes = 0

    def update(self, data: bytes):
        for value in data:
            self.hash ^= value
            self.hash = (self.hash * 1099511628211) & 0xffffffffffffffff
            self.bytes += 1

    def u8(self, value: int):
        self.update(value.to_bytes(1, "little", signed=False))

    def u32(self, value: int):
        self.update(value.to_bytes(4, "little", signed=False))

    def u64(self, value: int):
        self.update(value.to_bytes(8, "little", signed=False))

    def i64(self, value: int):
        self.update(value.to_bytes(8, "little", signed=True))

    def string(self, value: str):
        encoded = value.encode("utf-8")
        self.u32(len(encoded))
        self.update(encoded)


def checksum_solvers(solvers, identities):
    controller, recorder, counter, logger = identities
    writer = HashWriter()
    for solver in solvers:
        writer.u32(solver.seed)
        writer.i64(solver.verbose)
        writer.u64(solver.num_arrived_v_nets)
        writer.string(solver.save_dir)
        writer.u8(int(solver.reusable))
        writer.string(solver.node_ranking_method)
        writer.string(solver.link_ranking_method)
        writer.string(solver.matching_mathod)
        writer.string(solver.shortest_method)
        writer.i64(solver.k_shortest)
        writer.u8(int(solver.allow_rejection))
        writer.u8(int(solver.allow_revocable))
        same_identity = (
            solver.controller is controller
            and solver.recorder is recorder
            and solver.counter is counter
            and solver.logger is logger
        )
        writer.u8(int(same_identity))
    return writer.hash, writer.bytes


def benchmark_python(Solver, SolverRegistry, count, warmups, repetitions):
    SolverRegistry._registry = {}
    handlers = []
    for index in range(32):
        handler = type(f"Handler{index}", (Solver,), {})
        handlers.append(SolverRegistry.register(
            f"solver-{index}", "heuristic")(handler))

    inputs = make_inputs(count)
    direct_handlers = [handlers[index % 32] for index in range(count)]
    identities = (object(), object(), object(), object())

    def construct():
        return [
            direct_handlers[index](
                identities[0],
                identities[1],
                identities[2],
                identities[3],
                inputs[index],
                verbose=inputs[index].verbose,
            )
            for index in range(count)
        ]

    for _ in range(warmups):
        checksum_solvers(construct(), identities)

    samples = []
    expected = None
    output_bytes = None
    for _ in range(repetitions):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter_ns()
            solvers = construct()
            finish = time.perf_counter_ns()
        finally:
            gc.enable()
        output = checksum_solvers(solvers, identities)
        if expected is not None and output != expected:
            raise RuntimeError("Python benchmark checksum drift")
        expected = output
        output_bytes = output[1]
        samples.append(finish - start)

    return {
        "entries": count,
        "bytes": output_bytes,
        "checksum": expected[0],
        "median_ns": sorted(samples)[len(samples) // 2],
        "samples_ns": samples,
    }


def benchmark_native(binary, workers, count, warmups, repetitions):
    completed = subprocess.run(
        [
            str(binary),
            "--workers", str(workers),
            "--count", str(count),
            "--warmups", str(warmups),
            "--samples", str(repetitions),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", required=True, type=pathlib.Path)
    parser.add_argument("--workers", nargs="+", type=int,
                        default=list(DEFAULT_WORKERS))
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if tuple(args.workers) != DEFAULT_WORKERS:
        raise RuntimeError("the frozen-ready sweep requires workers 1 2 8")
    if args.count <= 0 or args.repetitions <= 0 or args.warmups < 0:
        raise RuntimeError("invalid benchmark dimensions")

    Solver, SolverRegistry = load_leaf(args.source)
    python_result = benchmark_python(
        Solver, SolverRegistry, args.count, args.warmups, args.repetitions)
    native = {}
    for workers in args.workers:
        row = benchmark_native(
            args.benchmark.resolve(),
            workers,
            args.count,
            args.warmups,
            args.repetitions,
        )
        for key in ("entries", "bytes", "checksum"):
            if row[key] != python_result[key]:
                raise RuntimeError(
                    f"workers={workers} output mismatch for {key}: "
                    f"Python={python_result[key]}, C++={row[key]}")
        row["median_ms"] = row["median_ns"] / 1_000_000
        row["speedup_vs_python"] = (
            python_result["median_ns"] / row["median_ns"])
        if row["speedup_vs_python"] <= 1.0:
            raise RuntimeError(
                f"workers={workers} did not beat Python: {row}")
        native[str(workers)] = row

    python_result["median_ms"] = python_result["median_ns"] / 1_000_000
    result = {
        "component": "solver.base_solver",
        "status": "PASS",
        "source_sha256": SOURCE_SHA256.upper(),
        "fixture": {
            "registered_descriptors": 32,
            "holders": args.count,
            "warmups": args.warmups,
            "samples": args.repetitions,
        },
        "output": {
            "entries": python_result["entries"],
            "bytes": python_result["bytes"],
            "checksum": python_result["checksum"],
        },
        "python": python_result,
        "cpp": native,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
