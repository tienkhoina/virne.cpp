#!/usr/bin/env python3
"""Robust worker sweep for deterministic topology-generator batches."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform
import statistics
import sys

import compare_topology_generator as compare


def parse_workers(text: str) -> list[int]:
    workers: list[int] = []
    for item in text.split(","):
        value = int(item)
        if value < 1:
            raise ValueError("worker counts must be positive")
        if value not in workers:
            workers.append(value)
    if not workers:
        raise ValueError("worker sweep cannot be empty")
    return workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--python-source", required=True, type=Path)
    parser.add_argument("--workers", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = parse_workers(args.workers)
    if args.warmups < 1 or args.repetitions < 1 or args.rounds < 1:
        raise ValueError("warmups, repetitions and rounds must be positive")

    oracle = compare.load_original(args.python_source)
    actual = compare.parse_parity(
        compare.run_cpp([str(args.cpp), "--parity"])
    )
    compare.require_parity(actual, compare.python_parity(oracle))
    print(f"differential: PASS ({len(actual)} exact cases)")

    metadata: dict[str, int] | None = None
    collected: dict[
        int, list[dict[str, tuple[int, list[int], int]]]
    ] = {worker: [] for worker in workers}
    all_rows: list[dict[str, tuple[int, list[int], int]]] = []

    for round_index in range(args.rounds):
        offset = round_index % len(workers)
        order = workers[offset:] + workers[:offset]
        if round_index % 2 == 1:
            order.reverse()
        for worker in order:
            lines = compare.run_cpp(
                [
                    str(args.cpp),
                    "--benchmark",
                    str(args.warmups),
                    str(args.repetitions),
                    str(worker),
                ]
            )
            current_metadata, rows = compare.parse_cpp_benchmark(lines)
            if metadata is None:
                metadata = current_metadata
            elif current_metadata != metadata:
                raise AssertionError("benchmark metadata changed during sweep")
            collected[worker].append(rows)
            all_rows.append(rows)
            print(
                f"collected round={round_index + 1}/{args.rounds} "
                f"worker={worker}"
            )

    assert metadata is not None
    python_rows = compare.python_benchmark(
        oracle, metadata, args.warmups, args.repetitions
    )
    print(
        f"runtime: Python {platform.python_version()}, "
        f"NetworkX {sys.modules['networkx'].__version__}"
    )
    print(
        f"protocol: warmups={args.warmups}, repetitions={args.repetitions}, "
        f"rounds={args.rounds}, workers={workers}; checksum and destruction "
        "excluded from timed regions"
    )

    first_rows = collected[workers[0]][0]
    batch_names = sorted(
        name[:-3] for name in first_rows if name.endswith(".mt")
    )
    recommendations: dict[str, int] = {}
    minimum_round_wins = math.ceil(args.rounds * 2 / 3)

    for operation in batch_names:
        python_samples, python_checksum = python_rows[operation + ".py"]
        python_median, _, _ = compare.stats(python_samples)

        all_st_samples: list[int] = []
        for rows in all_rows:
            _, samples, checksum = rows[operation + ".st"]
            if checksum != python_checksum:
                raise AssertionError(f"{operation}.st checksum mismatch")
            all_st_samples.extend(samples)
        st_median, _, _ = compare.stats(all_st_samples)

        print()
        print(
            f"{operation}: Python={python_median / 1e6:.3f} ms, "
            f"C++ ST={st_median / 1e6:.3f} ms"
        )
        print(
            "workers   median ms     MAD ms     p95 ms   vs ST   "
            "vs Python   round wins"
        )
        print("-" * 78)

        best_worker = 1
        best_median = st_median
        for worker in workers:
            aggregate: list[int] = []
            round_wins = 0
            for rows in collected[worker]:
                _, mt_samples, mt_checksum = rows[operation + ".mt"]
                _, st_samples, st_checksum = rows[operation + ".st"]
                if mt_checksum != python_checksum or st_checksum != python_checksum:
                    raise AssertionError(
                        f"{operation} worker={worker} checksum mismatch"
                    )
                aggregate.extend(mt_samples)
                if statistics.median(mt_samples) < statistics.median(st_samples):
                    round_wins += 1
            median, mad, p95 = compare.stats(aggregate)
            print(
                f"{worker:7d} {median / 1e6:11.3f} "
                f"{mad / 1e6:10.3f} {p95 / 1e6:10.3f} "
                f"{st_median / median:7.2f}x "
                f"{python_median / median:9.2f}x "
                f"{round_wins:5d}/{args.rounds}"
            )
            reliable = round_wins >= minimum_round_wins
            if reliable and median < best_median:
                best_worker = worker
                best_median = median

        recommendations[operation] = best_worker
        if best_worker == 1:
            print("selected: sequential (no reliable parallel improvement)")
        else:
            print(
                f"selected: {best_worker} workers, "
                f"{best_median / 1e6:.3f} ms, "
                f"{st_median / best_median:.2f}x versus C++ ST"
            )

    # The sweep repeatedly measured original single-call rows too. Validate
    # their checksums and require their aggregate C++ median to beat Python.
    for cpp_name in sorted(name for name in first_rows if name.endswith(".cpp")):
        py_name = compare.python_name(cpp_name)
        python_samples, python_checksum = python_rows[py_name]
        cpp_samples: list[int] = []
        for rows in all_rows:
            _, samples, checksum = rows[cpp_name]
            if checksum != python_checksum:
                raise AssertionError(f"{cpp_name} checksum mismatch")
            cpp_samples.extend(samples)
        cpp_median, _, _ = compare.stats(cpp_samples)
        python_median, _, _ = compare.stats(python_samples)
        if cpp_median >= python_median:
            raise AssertionError(f"{cpp_name} did not beat Python")

    print()
    print("recommended worker counts:")
    for operation, worker in recommendations.items():
        print(f"  {operation}: {worker}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"sweep_topology_generator_workers: FAIL: {error}", file=sys.stderr)
        raise
