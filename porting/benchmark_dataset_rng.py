#!/usr/bin/env python3
"""Checksum/state-gated Python/C++ timing and cast-worker sweep for dataset RNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import time
import zlib
from dataclasses import dataclass

import compare_dataset_core
import compare_dataset_rng


@dataclass(frozen=True)
class Row:
    kind: str
    count: int
    distribution: str
    dtype: str
    kwargs: dict
    cast: bool = False


ROWS = (
    Row("normal_float", 300_000, "normal", "float", {"loc": -1.25, "scale": 4.5}),
    Row("uniform_float", 300_000, "uniform", "float", {"low": -2, "high": 7.5}),
    Row("uniform_int", 300_000, "uniform", "int", {"low": -1000, "high": 1000}),
    Row(
        "uniform_int64_max",
        300_000,
        "uniform",
        "int",
        {"low": 0, "high": 2**63 - 1},
    ),
    Row("exponential_float", 300_000, "exponential", "float", {"scale": 0.5}),
    Row("poisson_int", 300_000, "poisson", "int", {"lam": 20.0}),
    Row(
        "normal_int",
        300_000,
        "normal",
        "int",
        {"loc": -1.25, "scale": 4.5},
        True,
    ),
    Row(
        "normal_bool",
        300_000,
        "normal",
        "bool",
        {"loc": -1.25, "scale": 4.5},
        True,
    ),
    Row(
        "exponential_int",
        300_000,
        "exponential",
        "int",
        {"scale": 0.5},
        True,
    ),
    Row(
        "exponential_bool",
        300_000,
        "exponential",
        "bool",
        {"scale": 0.5},
        True,
    ),
    Row("poisson_float", 300_000, "poisson", "float", {"lam": 20.0}, True),
    Row("poisson_bool", 300_000, "poisson", "bool", {"lam": 20.0}, True),
)


def scaled_count(value: int, scale: float) -> int:
    return max(1, int(round(value * scale)))


def python_benchmark(module, row: Row, seed: int) -> dict[str, int | str]:
    np = module.np
    np.random.seed(seed)
    start = time.perf_counter_ns()
    values = module.generate_data_with_distribution(
        row.count,
        row.distribution,
        row.dtype,
        **row.kwargs,
    )
    elapsed_ns = time.perf_counter_ns() - start

    # Verification and continuation are deliberately outside the timed region.
    random_value = float(np.random.random_sample())
    normal_value = float(np.random.normal())
    array = np.ascontiguousarray(values)
    expected_dtype = {
        "float": np.dtype("float64"),
        "int": np.dtype("int64"),
        "bool": np.dtype("bool"),
    }[row.dtype]
    if array.dtype != expected_dtype or array.ndim != 1 or array.size != row.count:
        raise RuntimeError(
            f"Python output representation drift for {row.kind}: "
            f"dtype={array.dtype}, shape={array.shape}"
        )
    payload = array.tobytes(order="C")
    return {
        "elapsed_ns": elapsed_ns,
        "checksum": zlib.adler32(payload) & 0xFFFFFFFF,
        "output_bytes": len(payload),
        "random_bits": compare_dataset_rng.float_bits(random_value),
        "normal_bits": compare_dataset_rng.float_bits(normal_value),
    }


def cpp_benchmark(
    harness: pathlib.Path,
    row: Row,
    workers: int,
    seed: int,
) -> dict[str, int | str]:
    process = subprocess.run(
        [
            str(harness),
            "benchmark",
            row.kind,
            str(row.count),
            str(workers),
            str(seed),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ RNG benchmark failed: {process.stderr.strip()}")
    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError(f"malformed C++ RNG benchmark line: {line!r}")
        key, value = line.split("=", 1)
        values[key] = value
    required = {
        "benchmark_version",
        "kind",
        "workers",
        "count",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "random_bits",
        "normal_bits",
        "status",
    }
    if (
        set(values) != required
        or values["benchmark_version"] != "1"
        or values["status"] != "PASS"
        or values["kind"] != row.kind
        or int(values["workers"]) != workers
        or int(values["count"]) != row.count
    ):
        raise RuntimeError(f"invalid C++ RNG benchmark response: {values!r}")
    return {
        "elapsed_ns": int(values["elapsed_ns"]),
        "checksum": int(values["checksum"]),
        "output_bytes": int(values["output_bytes"]),
        "random_bits": values["random_bits"],
        "normal_bits": values["normal_bits"],
    }


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def timing_summary(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = [sample / 1_000_000.0 for sample in samples_ns]
    median = statistics.median(samples_ms)
    deviations = [abs(value - median) for value in samples_ms]
    return {
        "median_ms": median,
        "mad_ms": statistics.median(deviations),
        "p95_ms": percentile95(samples_ms),
    }


def benchmark_widths(
    module,
    harness: pathlib.Path,
    row: Row,
    widths: list[int],
    seed: int,
    warmups: int,
    repetitions: int,
) -> list[dict]:
    python_samples = {workers: [] for workers in widths}
    cpp_samples = {workers: [] for workers in widths}
    semantics: dict[int, tuple[int, int, str, str]] = {}
    width_index = {workers: index for index, workers in enumerate(widths)}
    for sample in range(warmups + repetitions):
        offset = sample % len(widths)
        ordered_widths = widths[offset:] + widths[:offset]
        if (sample // len(widths)) % 2 != 0:
            ordered_widths.reverse()
        for workers in ordered_widths:
            if (sample + width_index[workers]) % 2 == 0:
                python = python_benchmark(module, row, seed)
                cpp = cpp_benchmark(harness, row, workers, seed)
            else:
                cpp = cpp_benchmark(harness, row, workers, seed)
                python = python_benchmark(module, row, seed)
            python_semantic = (
                int(python["checksum"]),
                int(python["output_bytes"]),
                str(python["random_bits"]),
                str(python["normal_bits"]),
            )
            cpp_semantic = (
                int(cpp["checksum"]),
                int(cpp["output_bytes"]),
                str(cpp["random_bits"]),
                str(cpp["normal_bits"]),
            )
            if python_semantic != cpp_semantic:
                raise RuntimeError(
                    f"{row.kind}/w{workers} semantic/state mismatch: "
                    f"python={python_semantic}, cpp={cpp_semantic}"
                )
            semantic = semantics.get(workers)
            if semantic is None:
                semantics[workers] = python_semantic
            elif semantic != python_semantic:
                raise RuntimeError(f"{row.kind}/w{workers} checksum/state drift")
            if sample >= warmups:
                python_samples[workers].append(int(python["elapsed_ns"]))
                cpp_samples[workers].append(int(cpp["elapsed_ns"]))

    results = []
    for workers in widths:
        semantic = semantics[workers]
        python_stats = timing_summary(python_samples[workers])
        cpp_stats = timing_summary(cpp_samples[workers])
        results.append(
            {
                "kind": row.kind,
                "workers": workers,
                "count": row.count,
                "checksum": semantic[0],
                "output_bytes": semantic[1],
                "random_bits": semantic[2],
                "normal_bits": semantic[3],
                "python": python_stats,
                "cpp": cpp_stats,
                "speedup": python_stats["median_ms"] / cpp_stats["median_ms"],
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--workers", type=int, nargs="+", default=list(range(1, 9)) + [0])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=[row.kind for row in ROWS],
        default=[row.kind for row in ROWS],
    )
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--performance-gate", action="store_true")
    parser.add_argument("--worker-policy-gate", action="store_true")
    args = parser.parse_args()
    if (
        args.warmups < 0
        or args.repetitions < 1
        or args.scale <= 0
        or args.seed < 0
        or args.seed > 0xFFFFFFFF
        or not args.workers
        or any(worker < 0 for worker in args.workers)
        or len(set(args.workers)) != len(args.workers)
    ):
        raise ValueError("invalid dataset RNG benchmark protocol")
    if args.performance_gate:
        if (
            set(args.kinds) != {row.kind for row in ROWS}
            or set(args.workers) != set(range(9))
            or args.warmups < 5
            or args.repetitions < 31
            or args.scale != 1.0
        ):
            raise ValueError(
                f"canonical performance gate requires all {len(ROWS)} kinds, workers "
                "0..8, scale 1, at least 5 warmups and 31 repetitions"
            )
    if args.worker_policy_gate:
        if (
            set(args.kinds) != {"exponential_int", "exponential_bool"}
            or len(args.kinds) != 2
            or set(args.workers) != set(range(9))
            or args.warmups < 5
            or args.repetitions < 31
            or args.seed != 123
            or args.scale not in {0.64, 2.0}
        ):
            raise ValueError(
                "canonical worker-policy gate requires exponential_int and "
                "exponential_bool, workers 0..8, seed 123, scale 0.64 or 2, "
                "at least 5 warmups and 31 repetitions"
            )

    module = compare_dataset_core.load_oracle(args.python_source)
    rows = tuple(
        Row(
            row.kind,
            scaled_count(row.count, args.scale),
            row.distribution,
            row.dtype,
            row.kwargs,
            row.cast,
        )
        for row in ROWS
        if row.kind in args.kinds
    )
    results: list[dict] = []
    for row in rows:
        widths = args.workers if row.cast else [1]
        row_results = benchmark_widths(
            module,
            args.harness,
            row,
            list(widths),
            args.seed,
            args.warmups,
            args.repetitions,
        )
        for result in row_results:
            results.append(result)
            workers = result["workers"]
            worker_text = "auto" if workers == 0 else str(workers)
            print(
                f"{row.kind}/w{worker_text}: "
                f"python={result['python']['median_ms']:.6f} ms, "
                f"cpp={result['cpp']['median_ms']:.6f} ms, "
                f"speedup={result['speedup']:.3f}x"
            )

    if args.performance_gate or args.worker_policy_gate:
        slower = [
            row
            for row in results
            if row["speedup"] <= 1.0
        ]
        if slower:
            labels = ", ".join(
                f"{row['kind']}/w{row['workers']}={row['speedup']:.3f}x"
                for row in slower
            )
            raise RuntimeError(f"C++-faster performance gate failed: {labels}")
        by_key = {(row["kind"], row["workers"]): row for row in results}
        for kind in ("exponential_int", "exponential_bool"):
            exponential_row = next(
                (row for row in rows if row.kind == kind), None
            )
            minimum_parallel_count = 131_072
            if (
                exponential_row is not None
                and exponential_row.count >= minimum_parallel_count
                and (kind, 0) in by_key
            ):
                automatic_ms = by_key[(kind, 0)]["cpp"]["median_ms"]
                sequential_ms = by_key[(kind, 1)]["cpp"]["median_ms"]
                if automatic_ms >= sequential_ms:
                    raise RuntimeError(
                        f"automatic {kind} mode did not beat sequential"
                    )
                explicit_ms = [
                    by_key[(kind, workers)]["cpp"]["median_ms"]
                    for workers in range(1, 9)
                    if (kind, workers) in by_key
                ]
                if explicit_ms and automatic_ms > min(explicit_ms) * 1.15:
                    raise RuntimeError(
                        f"automatic {kind} mode is more than 15% slower than "
                        "the best explicit width"
                    )

    payload = {
        "source_sha256": compare_dataset_core.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": module.np.__version__,
        "protocol": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "paired_alternating_order": True,
            "worker_widths_interleaved": True,
            "worker_width_order_rotated_and_reversed": True,
            "process_startup_excluded": True,
            "rng_construction_and_seed_excluded": True,
            "checksum_and_continuation_excluded": True,
            "allocation_generation_and_cast_included": True,
            "thread_creation_and_join_included": True,
            "workers": args.workers,
            "seed": args.seed,
            "scale": args.scale,
            "performance_gate": args.performance_gate,
            "worker_policy_gate": args.worker_policy_gate,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpus_visible": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "rows": results,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"dataset RNG benchmark: PASS ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
