#!/usr/bin/env python3
"""Checksum-gated Python/C++ timing and exploratory worker sweep."""

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
import sys
import time
from dataclasses import dataclass
from typing import Any

import compare_attribute_method


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = 2**64 - 1


@dataclass(frozen=True)
class Row:
    kind: str
    count_divisor: int = 1
    exception_path: bool = False


ROWS = (
    Row("resource_add_int"),
    Row("resource_sub_double"),
    Row("guarded_subtract_int"),
    Row("guard_failure_int", count_divisor=10, exception_path=True),
    Row("hard_ge_int"),
    Row("hard_le_double"),
    Row("hard_eq_double"),
    Row("soft_ge_int"),
)

SWEEP_KINDS = ("hard_le_double",)
WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def fnv_byte(checksum: int, value: int) -> int:
    return ((checksum ^ value) * FNV_PRIME) & UINT64_MASK


def fnv_u64(checksum: int, value: int) -> int:
    value &= UINT64_MASK
    for _ in range(8):
        checksum = fnv_byte(checksum, value & 0xFF)
        value >>= 8
    return checksum


def checksum_number(checksum: int, value: bool | int | float) -> int:
    if type(value) is bool:
        checksum = fnv_byte(checksum, ord("b"))
        return fnv_u64(checksum, int(value))
    if type(value) is int:
        checksum = fnv_byte(checksum, ord("i"))
        return fnv_u64(checksum, value)
    if type(value) is float:
        checksum = fnv_byte(checksum, ord("f"))
        return fnv_u64(checksum, int(compare_attribute_method.float_bits(value), 16))
    raise TypeError(f"unexpected benchmark output type: {type(value).__name__}")


def benchmark_checksum(flags: list[bool], outputs: list[bool | int | float]) -> int:
    checksum = FNV_OFFSET
    for flag, output in zip(flags, outputs, strict=True):
        checksum = fnv_byte(checksum, int(flag))
        checksum = checksum_number(checksum, output)
    return checksum


def resource_instance(module):
    class Resource(module.ResourceAttributeMethod):
        pass

    result = Resource()
    result.type = "resource"
    result.name = "cpu"
    return result


def python_fixture(kind: str, count: int):
    virtual_values: list[bool | int | float] = []
    physical_values: list[bool | int | float] = []
    for index in range(count):
        if kind == "resource_add_int":
            virtual_values.append(index % 17 - 8)
            physical_values.append(index % 101 - 50)
        elif kind == "resource_sub_double":
            virtual_values.append((index % 29 - 14) * 0.125)
            physical_values.append((index % 113 - 50) * 0.25)
        elif kind == "guarded_subtract_int":
            virtual_values.append(index % 7)
            physical_values.append(100 + index % 31)
        elif kind == "guard_failure_int":
            virtual_values.append(index % 17)
            physical_values.append(index % 11)
        elif kind in {"hard_ge_int", "soft_ge_int"}:
            virtual_values.append(index % 97 - 48)
            physical_values.append(index % 89 - 44)
        elif kind == "hard_le_double":
            virtual_values.append((index % 97 - 48) * 0.25)
            physical_values.append((index % 89 - 44) * 0.5)
        elif kind == "hard_eq_double":
            virtual_value = (index % 97 - 48) * 0.25
            virtual_values.append(virtual_value)
            physical_values.append(
                virtual_value if index % 2 == 0 else virtual_value + 0.25
            )
        else:
            raise ValueError(f"unknown Python benchmark kind: {kind}")
    return virtual_values, physical_values


def python_benchmark(module, kind: str, count: int) -> dict[str, int | str]:
    virtual_values, physical_values = python_fixture(kind, count)
    flags: list[bool] = [False] * count
    outputs: list[bool | int | float] = list(physical_values)
    first_failure: str | int = "none"

    if kind.startswith("resource_") or kind.startswith("guard"):
        instance = resource_instance(module)
        virtual_mappings = [{"cpu": value} for value in virtual_values]
        physical_mappings = [{"cpu": value} for value in physical_values]
        method = "+" if kind == "resource_add_int" else "-"
        safe = kind not in {"resource_sub_double"}
        start = time.perf_counter_ns()
        for index in range(count):
            try:
                flags[index] = instance.update(
                    virtual_mappings[index], physical_mappings[index], method, safe
                )
            except ValueError:
                if kind != "guard_failure_int":
                    raise
                if first_failure == "none":
                    first_failure = index
                flags[index] = False
        elapsed_ns = time.perf_counter_ns() - start
        outputs = [mapping["cpu"] for mapping in physical_mappings]
    else:
        calculator = object.__new__(module.ConstraintAttributeMethod)
        calculator.constraint_restrictions = (
            "soft" if kind == "soft_ge_int" else "hard"
        )
        method = {
            "hard_ge_int": "ge",
            "hard_le_double": "le",
            "hard_eq_double": "eq",
            "soft_ge_int": "ge",
        }[kind]
        start = time.perf_counter_ns()
        for index in range(count):
            flag, offset = calculator._calculate_satisfiability_values(
                virtual_values[index], physical_values[index], method
            )
            if type(flag) is not bool:
                raise RuntimeError(f"Python scalar flag type drift at {index}")
            flags[index] = flag
            outputs[index] = offset
        elapsed_ns = time.perf_counter_ns() - start

    # Representation checks and checksum are deliberately outside the timer.
    checksum = benchmark_checksum(flags, outputs)
    return {
        "elapsed_ns": elapsed_ns,
        "checksum": checksum,
        "output_bytes": count * 10,
        "first_failure": str(first_failure),
    }


def parse_lines(process: subprocess.CompletedProcess[str], context: str) -> dict[str, str]:
    if process.returncode != 0:
        raise RuntimeError(f"{context} failed: {process.stderr.strip()}")
    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError(f"malformed {context} line: {line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate {context} key: {key}")
        values[key] = value
    return values


def cpp_benchmark(
    harness: pathlib.Path,
    kind: str,
    count: int,
    workers: int,
    command: str = "benchmark",
) -> dict[str, int | str]:
    process = subprocess.run(
        [str(harness), command, kind, str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    values = parse_lines(process, "C++ attribute-method benchmark")
    required = {
        "benchmark_version",
        "kind",
        "count",
        "workers",
        "effective_workers",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "first_failure",
        "status",
    }
    if (
        set(values) != required
        or values["benchmark_version"] != "1"
        or values["status"] != "PASS"
        or values["kind"] != kind
        or int(values["count"]) != count
        or int(values["workers"]) != workers
    ):
        raise RuntimeError(f"invalid C++ benchmark response: {values!r}")
    return {
        "elapsed_ns": int(values["elapsed_ns"]),
        "checksum": int(values["checksum"]),
        "output_bytes": int(values["output_bytes"]),
        "first_failure": values["first_failure"],
        "effective_workers": int(values["effective_workers"]),
    }


def harness_info(harness: pathlib.Path) -> dict[str, str]:
    process = subprocess.run(
        [str(harness), "info"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    values = parse_lines(process, "C++ attribute-method info")
    required = {
        "info_version",
        "compiler",
        "hardware_concurrency",
        "available_cpus",
        "affinity_cpus",
        "status",
    }
    if set(values) != required or values["info_version"] != "1" or values["status"] != "PASS":
        raise RuntimeError(f"invalid C++ info response: {values!r}")
    return values


def semantic_tuple(result: dict[str, int | str]) -> tuple[int, int, str]:
    return (
        int(result["checksum"]),
        int(result["output_bytes"]),
        str(result["first_failure"]),
    )


def assert_semantics(
    python: dict[str, int | str],
    cpp: dict[str, int | str],
    context: str,
) -> tuple[int, int, str]:
    python_semantics = semantic_tuple(python)
    cpp_semantics = semantic_tuple(cpp)
    if python_semantics != cpp_semantics:
        raise RuntimeError(
            f"{context} semantic mismatch: "
            f"python={python_semantics}, cpp={cpp_semantics}"
        )
    return python_semantics


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def timing_summary(samples_ns: list[int]) -> dict[str, float]:
    values = [sample / 1_000_000.0 for sample in samples_ns]
    median = statistics.median(values)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(value - median) for value in values),
        "p95_ms": percentile95(values),
    }


def benchmark_pair(
    module,
    harness: pathlib.Path,
    kind: str,
    count: int,
    workers: int,
    warmups: int,
    repetitions: int,
) -> dict:
    # A full-size checksum/type/bit preflight is mandatory before timing.
    preflight_python = python_benchmark(module, kind, count)
    preflight_cpp = cpp_benchmark(harness, kind, count, workers)
    semantics = assert_semantics(
        preflight_python, preflight_cpp, f"preflight {kind}/w{workers}"
    )

    python_samples: list[int] = []
    cpp_samples: list[int] = []
    effective_workers = int(preflight_cpp["effective_workers"])
    for sample in range(warmups + repetitions):
        if sample % 2 == 0:
            python = python_benchmark(module, kind, count)
            cpp = cpp_benchmark(harness, kind, count, workers)
        else:
            cpp = cpp_benchmark(harness, kind, count, workers)
            python = python_benchmark(module, kind, count)
        if assert_semantics(python, cpp, f"sample {kind}/w{workers}") != semantics:
            raise RuntimeError(f"{kind}/w{workers} checksum drift")
        if int(cpp["effective_workers"]) != effective_workers:
            raise RuntimeError(f"{kind}/w{workers} effective-width drift")
        if sample >= warmups:
            python_samples.append(int(python["elapsed_ns"]))
            cpp_samples.append(int(cpp["elapsed_ns"]))

    python_stats = timing_summary(python_samples)
    cpp_stats = timing_summary(cpp_samples)
    return {
        "kind": kind,
        "count": count,
        "workers": workers,
        "effective_workers": effective_workers,
        "checksum": semantics[0],
        "output_bytes": semantics[1],
        "first_failure": semantics[2],
        "python": python_stats,
        "cpp": cpp_stats,
        "speedup": python_stats["median_ms"] / cpp_stats["median_ms"],
    }


def benchmark_widths(
    module,
    harness: pathlib.Path,
    kind: str,
    count: int,
    warmups: int,
    repetitions: int,
) -> list[dict]:
    widths = list(WIDTHS)
    python_samples: list[int] = []
    cpp_samples = {workers: [] for workers in widths}
    semantics: dict[int, tuple[int, int, str]] = {}
    effective: dict[int, int] = {}

    # Validate all requested widths before collecting the first timing sample.
    python_preflight = python_benchmark(module, kind, count)
    cpp_command = (
        "benchmark_double_batch" if kind == "hard_le_double" else "benchmark"
    )
    for workers in widths:
        cpp = cpp_benchmark(harness, kind, count, workers, cpp_command)
        semantics[workers] = assert_semantics(
            python_preflight, cpp, f"worker preflight {kind}/w{workers}"
        )
        effective[workers] = int(cpp["effective_workers"])

    for sample in range(warmups + repetitions):
        offset = sample % len(widths)
        ordered = widths[offset:] + widths[:offset]
        if (sample // len(widths)) % 2:
            ordered.reverse()
        cpp_results: dict[int, dict[str, Any]] = {}
        if sample % 2 == 0:
            python = python_benchmark(module, kind, count)
        for workers in ordered:
            cpp_results[workers] = cpp_benchmark(
                harness, kind, count, workers, cpp_command
            )
        if sample % 2 != 0:
            python = python_benchmark(module, kind, count)
        for workers in widths:
            cpp = cpp_results[workers]
            if (
                assert_semantics(python, cpp, f"worker sample {kind}/w{workers}")
                != semantics[workers]
            ):
                raise RuntimeError(f"{kind}/w{workers} worker checksum drift")
            if int(cpp["effective_workers"]) != effective[workers]:
                raise RuntimeError(f"{kind}/w{workers} effective-width drift")
            if sample >= warmups:
                cpp_samples[workers].append(int(cpp["elapsed_ns"]))
        if sample >= warmups:
            python_samples.append(int(python["elapsed_ns"]))

    results = []
    python_stats = timing_summary(python_samples)
    for workers in widths:
        cpp_stats = timing_summary(cpp_samples[workers])
        results.append(
            {
                "kind": kind,
                "count": count,
                "workers": workers,
                "effective_workers": effective[workers],
                "checksum": semantics[workers][0],
                "output_bytes": semantics[workers][1],
                "first_failure": semantics[workers][2],
                "python": python_stats,
                "cpp": cpp_stats,
                "speedup": python_stats["median_ms"] / cpp_stats["median_ms"],
            }
        )
    return results


def worker_measurement(results: list[dict]) -> dict:
    decisions = []
    for kind in SWEEP_KINDS:
        rows = [row for row in results if row["kind"] == kind]
        sequential = next(row for row in rows if row["workers"] == 1)
        explicit = [row for row in rows if 2 <= row["workers"] <= 8]
        best = min(explicit, key=lambda row: row["cpp"]["median_ms"])
        improvement = (
            sequential["cpp"]["median_ms"] / best["cpp"]["median_ms"]
        )
        decisions.append(
            {
                "kind": kind,
                "production_batch_api": True,
                "sequential_median_ms": sequential["cpp"]["median_ms"],
                "best_requested_workers": best["workers"],
                "best_effective_workers": best["effective_workers"],
                "best_median_ms": best["cpp"]["median_ms"],
                "best_vs_sequential": improvement,
                "material_15_percent_win": improvement >= 1.15,
            }
        )
    return {
        "configuration_policy": (
            "caller supplies a typed worker count; zero/one are sequential; "
            "there is no automatic production threshold"
        ),
        "threshold": "15% is report-only for measured configurations",
        "rows": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=pathlib.Path, default=compare_attribute_method.default_source()
    )
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    arguments = parser.parse_args()
    if arguments.count <= 0 or arguments.warmups < 0 or arguments.repetitions < 1:
        raise SystemExit("count/repetition arguments are invalid")

    source = arguments.source.resolve()
    harness = arguments.harness.resolve()
    module = compare_attribute_method.load_original(source)
    info = harness_info(harness)

    canonical = []
    for row in ROWS:
        count = max(1, arguments.count // row.count_divisor)
        result = benchmark_pair(
            module,
            harness,
            row.kind,
            count,
            1,
            arguments.warmups,
            arguments.repetitions,
        )
        result["exception_path"] = row.exception_path
        canonical.append(result)

    worker_results = []
    for kind in SWEEP_KINDS:
        worker_results.extend(
            benchmark_widths(
                module,
                harness,
                kind,
                arguments.count,
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
        "canonical_results": canonical,
        "worker_results": worker_results,
        "all_checksums_exact": True,
        "all_canonical_cpp_faster": all(row["speedup"] > 1.0 for row in canonical),
        "performance_gate": all(
            row["speedup"] > 1.0
            for row in canonical
            if not row["exception_path"]
        ),
        "exception_path_is_report_only": True,
        "worker_measurement": worker_measurement(worker_results),
        "production_batch_api_present": True,
        "deferred_rows": {
            "extrema_lookup_gather": "belongs to the future registry adapter",
            "generation_delegation": "belongs to the future BaseAttribute adapter",
        },
    }
    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
