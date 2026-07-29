#!/usr/bin/env python3
"""Canonical worker sweep for the production double satisfiability batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import struct
import sys
import time
from typing import Any

import benchmark_attribute_method
import compare_attribute_method


COUNTS = (10_000, 100_000, 1_000_000, 4_000_000)
KINDS = (
    "hard_ge_double",
    "hard_le_double",
    "hard_eq_double",
    "soft_ge_double",
    "soft_le_double",
    "soft_eq_double",
)
WIDTHS = (1, 0, 2, 3, 4, 5, 6, 7, 8)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = 2**64 - 1
MATERIAL_SPEEDUP = 1.15
NEAR_BEST_RATIO = 1.05


def fnv_byte(checksum: int, value: int) -> int:
    return ((checksum ^ value) * FNV_PRIME) & UINT64_MASK


def fnv_u64(checksum: int, value: int) -> int:
    value &= UINT64_MASK
    for _ in range(8):
        checksum = fnv_byte(checksum, value & 0xFF)
        value >>= 8
    return checksum


def double_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def python_oracle_checksum(module, kind: str, count: int) -> dict[str, int]:
    """Run the pinned Python scalar method once and stream its exact checksum."""
    restriction, operation, value_lane = kind.split("_")
    if value_lane != "double" or restriction not in {"hard", "soft"}:
        raise ValueError(f"unsupported double batch kind: {kind}")
    calculator = object.__new__(module.ConstraintAttributeMethod)
    calculator.constraint_restrictions = restriction
    checksum = FNV_OFFSET
    start = time.perf_counter_ns()
    for index in range(count):
        virtual_value = (index % 97 - 48) * 0.25
        physical_value = (index % 89 - 44) * 0.5
        flag, offset = calculator._calculate_satisfiability_values(
            virtual_value, physical_value, operation
        )
        if type(flag) is not bool or type(offset) is not float:
            raise RuntimeError(
                "Python oracle representation drift at "
                f"index {index}: flag={type(flag).__name__}, "
                f"offset={type(offset).__name__}"
            )
        checksum = fnv_byte(checksum, int(flag))
        checksum = fnv_byte(checksum, ord("f"))
        checksum = fnv_u64(checksum, double_bits(offset))
    elapsed_ns = time.perf_counter_ns() - start
    return {
        "checksum": checksum,
        "output_bytes": count * 10,
        "elapsed_ns": elapsed_ns,
    }


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def timing_summary(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = [sample / 1_000_000.0 for sample in samples_ns]
    median = statistics.median(samples_ms)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(sample - median) for sample in samples_ms),
        "p95_ms": percentile95(samples_ms),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
    }


def semantic_tuple(result: dict[str, int | str]) -> tuple[int, int, str]:
    return (
        int(result["checksum"]),
        int(result["output_bytes"]),
        str(result["first_failure"]),
    )


def run_cpp(
    harness: pathlib.Path, kind: str, count: int, workers: int
) -> dict[str, int | str]:
    return benchmark_attribute_method.cpp_benchmark(
        harness,
        kind,
        count,
        workers,
        "benchmark_double_batch",
    )


def rotated_widths(sample: int) -> list[int]:
    widths = list(WIDTHS)
    offset = sample % len(widths)
    ordered = widths[offset:] + widths[:offset]
    if (sample // len(widths)) % 2:
        ordered.reverse()
    return ordered


def sweep_count(
    module,
    harness: pathlib.Path,
    kind: str,
    count: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    oracle = python_oracle_checksum(module, kind, count)
    expected = (oracle["checksum"], oracle["output_bytes"], "none")
    samples = {workers: [] for workers in WIDTHS}
    effective_workers: dict[int, int] = {}

    # Every requested width passes a full-size oracle gate before timing.
    for workers in WIDTHS:
        cpp = run_cpp(harness, kind, count, workers)
        actual = semantic_tuple(cpp)
        if actual != expected:
            raise RuntimeError(
                f"preflight mismatch at count={count}, workers={workers}: "
                f"python={expected}, cpp={actual}"
            )
        effective_workers[workers] = int(cpp["effective_workers"])

    for sample in range(warmups + repetitions):
        for workers in rotated_widths(sample):
            cpp = run_cpp(harness, kind, count, workers)
            actual = semantic_tuple(cpp)
            if actual != expected:
                raise RuntimeError(
                    f"sample mismatch at count={count}, workers={workers}, "
                    f"sample={sample}: python={expected}, cpp={actual}"
                )
            if int(cpp["effective_workers"]) != effective_workers[workers]:
                raise RuntimeError(
                    f"effective worker drift at count={count}, "
                    f"workers={workers}"
                )
            if sample >= warmups:
                samples[workers].append(int(cpp["elapsed_ns"]))

    rows = []
    for workers in WIDTHS:
        rows.append(
            {
                "count": count,
                "workers": workers,
                "effective_workers": effective_workers[workers],
                "checksum": oracle["checksum"],
                "output_bytes": oracle["output_bytes"],
                "timing": timing_summary(samples[workers]),
            }
        )
    return {
        "kind": kind,
        "count": count,
        "python_oracle": oracle,
        "rows": rows,
    }


def count_observation(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    sequential = next(row for row in rows if row["workers"] == 1)
    default_config = next(row for row in rows if row["workers"] == 0)
    explicit = [row for row in rows if row["workers"] != 0]
    best = min(explicit, key=lambda row: row["timing"]["median_ms"])
    near_best = [
        row
        for row in explicit
        if row["timing"]["median_ms"]
        <= best["timing"]["median_ms"] * NEAR_BEST_RATIO
    ]
    economical = min(
        near_best,
        key=lambda row: (row["effective_workers"], row["workers"]),
    )
    sequential_ms = sequential["timing"]["median_ms"]
    best_ms = best["timing"]["median_ms"]
    best_speedup = sequential_ms / best_ms
    parallel_worthwhile = best_speedup >= MATERIAL_SPEEDUP
    recommended = economical if parallel_worthwhile else sequential
    return {
        "kind": result["kind"],
        "count": result["count"],
        "sequential_median_ms": sequential_ms,
        "best_requested_workers": best["workers"],
        "best_effective_workers": best["effective_workers"],
        "best_median_ms": best_ms,
        "best_vs_sequential": best_speedup,
        "parallel_worthwhile": parallel_worthwhile,
        "recommended_requested_workers": recommended["workers"],
        "recommended_effective_workers": recommended["effective_workers"],
        "recommended_median_ms": recommended["timing"]["median_ms"],
        "default_zero_effective_workers": default_config["effective_workers"],
        "default_zero_median_ms": default_config["timing"]["median_ms"],
        "default_zero_vs_sequential": sequential_ms
        / default_config["timing"]["median_ms"],
        "default_zero_within_15_percent_of_best": default_config["timing"]["median_ms"]
        <= best_ms * MATERIAL_SPEEDUP,
    }


def measurement_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    kind_results = []
    for kind in KINDS:
        observations = [
            count_observation(result)
            for result in results
            if result["kind"] == kind
        ]
        if not observations:
            continue
        observations.sort(key=lambda row: row["count"])
        kind_results.append(
            {
                "kind": kind,
                "counts": observations,
            }
        )
    return {
        "configuration_policy": (
            "caller supplies a typed worker count; zero/one are sequential; "
            "production has no machine-specific automatic threshold"
        ),
        "material_speedup_threshold": MATERIAL_SPEEDUP,
        "economical_width_rule": (
            "report-only smallest effective width within 5% of the best explicit median"
        ),
        "kinds": kind_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=compare_attribute_method.default_source(),
    )
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print compact medians/summary while retaining full --output JSON",
    )
    parser.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    arguments = parser.parse_args()
    if (
        not arguments.counts
        or any(count <= 0 for count in arguments.counts)
        or not arguments.kinds
        or any(kind not in KINDS for kind in arguments.kinds)
        or arguments.warmups < 0
        or arguments.repetitions < 1
    ):
        raise SystemExit("counts/warmups/repetitions are invalid")
    if len(set(arguments.counts)) != len(arguments.counts):
        raise SystemExit("counts must be unique")
    if len(set(arguments.kinds)) != len(arguments.kinds):
        raise SystemExit("kinds must be unique")

    source = arguments.source.resolve()
    harness = arguments.harness.resolve()
    module = compare_attribute_method.load_original(source)
    info = benchmark_attribute_method.harness_info(harness)
    results = []
    for kind in arguments.kinds:
        for count in sorted(arguments.counts):
            print(
                f"sweeping kind={kind}, count={count}",
                file=sys.stderr,
                flush=True,
            )
            results.append(
                sweep_count(
                    module,
                    harness,
                    kind,
                    count,
                    arguments.warmups,
                    arguments.repetitions,
                )
            )

    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    result = {
        "status": "PASS",
        "python_source_sha256": compare_attribute_method.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "numpy": module.np.__version__,
        "networkx": module.nx.__version__,
        "platform": platform.platform(),
        "compiler": info["compiler"],
        "affinity_cpus": affinity,
        "harness_affinity_cpus": info["affinity_cpus"],
        "warmups": arguments.warmups,
        "repetitions": arguments.repetitions,
        "widths": list(WIDTHS),
        "kinds": list(arguments.kinds),
        "results": results,
        "all_python_checksums_exact": True,
        "measurement_summary": measurement_summary(results),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary_only:
        summary = {
            "status": result["status"],
            "harness_sha256": result["harness_sha256"],
            "measurement_summary": result["measurement_summary"],
            "medians": [
                {
                    "kind": count_result["kind"],
                    "count": count_result["count"],
                    "checksum": count_result["python_oracle"]["checksum"],
                    "rows": [
                        [
                            row["workers"],
                            row["effective_workers"],
                            row["timing"]["median_ms"],
                            row["timing"]["mad_ms"],
                            row["timing"]["p95_ms"],
                        ]
                        for row in count_result["rows"]
                    ],
                }
                for count_result in result["results"]
            ],
        }
        print(json.dumps(summary, sort_keys=True))
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
