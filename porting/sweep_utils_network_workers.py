#!/usr/bin/env python3
"""Find the fastest deterministic worker count for utils.network batches."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform
import sys

import compare_utils_network as compare


def parse_workers(text: str) -> list[int]:
    workers = []
    for part in text.split(","):
        value = int(part)
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
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--workers", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = parse_workers(args.workers)
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    oracle = compare.load_original(args.python_source)

    cpp_parity = compare.parse_parity(
        compare.run_cpp([str(args.cpp), "--parity"])
    )
    compare.require_parity(cpp_parity, compare.python_parity(oracle))
    print(f"differential: PASS ({len(cpp_parity)} cases)")

    sweep_rows: dict[int, dict[str, tuple[int, list[int], int]]] = {}
    all_rows: list[dict[str, tuple[int, list[int], int]]] = []
    metadata: dict[str, int] | None = None
    for round_index in range(args.rounds):
        run_order = workers[round_index % len(workers) :] + workers[
            : round_index % len(workers)
        ]
        if round_index % 2 == 1:
            run_order = list(reversed(run_order))
        for worker_count in run_order:
            lines = compare.run_cpp(
                [
                    str(args.cpp),
                    "--benchmark",
                    str(args.fixture),
                    str(args.warmups),
                    str(args.repetitions),
                    str(worker_count),
                ]
            )
            current_metadata, rows = compare.parse_cpp_benchmark(lines)
            if metadata is None:
                metadata = current_metadata
            elif metadata != current_metadata:
                raise AssertionError(
                    "C++ benchmark metadata changed during sweep"
                )
            all_rows.append(rows)
            if worker_count not in sweep_rows:
                sweep_rows[worker_count] = {
                    name: (reported_workers, list(samples), checksum)
                    for name, (reported_workers, samples, checksum) in rows.items()
                }
            else:
                aggregate = sweep_rows[worker_count]
                for name, (reported_workers, samples, checksum) in rows.items():
                    old_workers, old_samples, old_checksum = aggregate[name]
                    if old_workers != reported_workers or old_checksum != checksum:
                        raise AssertionError(
                            f"{name}: row metadata/checksum changed across rounds"
                        )
                    old_samples.extend(samples)
            print(
                f"collected round={round_index + 1}/{args.rounds} "
                f"C++ worker={worker_count}"
            )

    assert metadata is not None
    python_rows = compare.python_benchmark(
        oracle,
        args.fixture,
        metadata,
        args.warmups,
        args.repetitions,
    )
    print(
        f"runtime: Python {platform.python_version()}, "
        f"NetworkX {sys.modules['networkx'].__version__}"
    )
    print(
        f"protocol: warmups={args.warmups}, repetitions={args.repetitions}, "
        f"rounds={args.rounds}, worker sweep={workers}; "
        "fixture loading and checksums excluded"
    )

    first_rows = sweep_rows[workers[0]]
    operations = sorted(
        name[:-3] for name in first_rows if name.endswith(".mt")
    )
    recommendations: dict[str, int] = {}
    for operation in operations:
        python_name = operation + ".py"
        python_samples, python_checksum = python_rows[python_name]
        python_median, _, _ = compare.stats(python_samples)
        st_samples: list[int] = []
        for rows in all_rows:
            _, current_samples, st_checksum = rows[operation + ".st"]
            if st_checksum != python_checksum:
                raise AssertionError(f"{operation}.st checksum mismatch")
            st_samples.extend(current_samples)
        st_median, _, _ = compare.stats(st_samples)

        print()
        print(
            f"{operation}: Python={python_median / 1e6:.3f} ms, "
            f"C++ ST={st_median / 1e6:.3f} ms"
        )
        print("workers   median ms     MAD ms     p95 ms   vs ST   vs Python")
        print("-" * 66)
        best_worker = workers[0]
        best_median = math.inf
        for worker_count in workers:
            _, samples, checksum = sweep_rows[worker_count][operation + ".mt"]
            if checksum != python_checksum:
                raise AssertionError(
                    f"{operation} worker={worker_count} checksum mismatch"
                )
            median, mad, p95 = compare.stats(samples)
            if median < best_median:
                best_median = median
                best_worker = worker_count
            print(
                f"{worker_count:7d} {median / 1e6:11.3f} "
                f"{mad / 1e6:10.3f} {p95 / 1e6:10.3f} "
                f"{st_median / median:7.2f}x "
                f"{python_median / median:9.2f}x"
            )
        if best_median < st_median:
            recommendations[operation] = best_worker
            print(
                f"best: {best_worker} worker(s), {best_median / 1e6:.3f} ms, "
                f"{st_median / best_median:.2f}x versus C++ ST"
            )
        else:
            recommendations[operation] = 1
            print(
                f"best: sequential, {st_median / 1e6:.3f} ms; "
                "parallel fan-out does not improve this workload"
            )

    print()
    print("recommended worker counts:")
    for operation, worker_count in recommendations.items():
        print(f"  {operation}: {worker_count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"sweep_utils_network_workers: FAIL: {error}", file=sys.stderr)
        raise
