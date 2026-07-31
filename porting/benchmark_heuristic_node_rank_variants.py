"""One frozen runtime comparison for all non-ML node-rank solver variants."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import time
from typing import Any

import numpy as np

import compare_heuristic_node_rank_variants as oracle


SOLVER_NAMES = oracle.SOLVER_NAMES
WORKERS = (1, 2, 8)
SEED = 77


def benchmark_virtual_network():
    node_count = 8
    edges = [(index, index + 1) for index in range(node_count - 1)]
    node_resources = [5 + (index * 3) % 7 for index in range(node_count)]
    link_resources = [2 + (index * 5) % 9 for index in range(node_count - 1)]
    network = oracle.FakeNetwork(
        node_resources, edges, link_resources, request=True
    )
    network.request_id = 41
    network.arrival_time = 3.0
    network.lifetime = 20.0
    return network


def benchmark_physical_network():
    node_count = 48
    edges = [(index, index + 1) for index in range(node_count - 1)]
    node_resources = [100 + (index * 17) % 31 for index in range(node_count)]
    link_resources = [100 + (index * 13) % 29 for index in range(node_count - 1)]
    return oracle.FakeNetwork(node_resources, edges, link_resources)


def solver_instance(solvers, name):
    return solvers[name](
        oracle.FakeController(),
        object(),
        object(),
        object(),
        oracle.solver_config(),
        matching_mathod="greedy",
        shortest_method="bfs_shortest",
        k_shortest=10,
    )


def python_sample(solvers, name, iterations):
    np.random.seed(SEED)
    instance = solver_instance(solvers, name)
    virtual = benchmark_virtual_network()
    physical_inputs = [benchmark_physical_network() for _ in range(iterations)]
    solutions = []
    begin = time.perf_counter_ns()
    for physical in physical_inputs:
        solutions.append(
            instance.solve({"v_net": virtual, "p_net": physical})
        )
    elapsed = time.perf_counter_ns() - begin
    rng_next = None
    if name == "random_rank":
        rng_next = int(np.random.randint(0, 2**32, dtype=np.uint32))
    verification = {
        "solutions": [oracle.canonical_solution(solution) for solution in solutions],
        "rng_next": rng_next,
    }
    return elapsed, verification


def native_result(harness, workers, warmups, repetitions, iterations):
    completed = subprocess.run(
        [
            str(harness),
            "--benchmark",
            "--workers",
            str(workers),
            "--warmups",
            str(warmups),
            "--repetitions",
            str(repetitions),
            "--iterations",
            str(iterations),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native benchmark workers={workers} failed:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def percentile95(values):
    ordered = sorted(values)
    position = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[position]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-source", required=True, type=pathlib.Path)
    parser.add_argument("--rank-source", required=True, type=pathlib.Path)
    parser.add_argument("--harness", required=True, type=pathlib.Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.iterations <= 0:
        raise ValueError("warmups must be non-negative; repetitions/iterations positive")

    rankers = oracle.load_rankers(args.rank_source)
    solvers = oracle.load_solvers(args.solver_source, rankers)
    python_rows: dict[str, dict[str, Any]] = {}
    for name in SOLVER_NAMES:
        samples = []
        expected = None
        for sample in range(args.warmups + args.repetitions):
            elapsed, verification = python_sample(
                solvers, name, args.iterations
            )
            if sample >= args.warmups:
                samples.append(elapsed)
                if expected is None:
                    expected = verification
                elif verification != expected:
                    raise RuntimeError(
                        f"Python output changed between samples for {name}"
                    )
        python_rows[name] = {
            "samples_ns": samples,
            "verification": expected,
        }

    worker_rows = {}
    for workers in WORKERS:
        native = native_result(
            args.harness,
            workers,
            args.warmups,
            args.repetitions,
            args.iterations,
        )
        native_by_name = {row["solver"]: row for row in native["solvers"]}
        rows = []
        for name in SOLVER_NAMES:
            expected = python_rows[name]["verification"]
            actual = native_by_name[name]["verification"]
            if actual != expected:
                raise RuntimeError(
                    f"benchmark differential mismatch for {name}, workers={workers}\n"
                    f"python={json.dumps(expected, sort_keys=True)}\n"
                    f"native={json.dumps(actual, sort_keys=True)}"
                )
            native_samples = native_by_name[name]["samples_ns"]
            rows.append(
                {
                    "solver": name,
                    "python_median_ns_per_solve": statistics.median(
                        python_rows[name]["samples_ns"]
                    )
                    / args.iterations,
                    "python_p95_ns_per_solve": percentile95(
                        python_rows[name]["samples_ns"]
                    )
                    / args.iterations,
                    "native_median_ns_per_solve": statistics.median(
                        native_samples
                    )
                    / args.iterations,
                    "native_p95_ns_per_solve": percentile95(native_samples)
                    / args.iterations,
                    "speedup_median": statistics.median(
                        python_rows[name]["samples_ns"]
                    )
                    / statistics.median(native_samples),
                    "native_samples_ns": native_samples,
                }
            )
        worker_rows[str(workers)] = rows

    result = {
        "component": "solver.heuristic.node_rank.variants",
        "status": "PASS",
        "fixture": {"physical_nodes": 48, "virtual_nodes": 8},
        "seed": SEED,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "iterations_per_sample": args.iterations,
        "workers": worker_rows,
        "solvers": list(SOLVER_NAMES),
        "gate": "exact verification payload matched before timing rows",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
