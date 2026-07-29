#!/usr/bin/env python3
"""Checksum-gated Python/C++ timing and worker sweep for dataset core."""

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
import tempfile
import time
from dataclasses import dataclass

import compare_dataset_core


@dataclass(frozen=True)
class Row:
    kind: str
    operations: int
    batch_size: int
    batch: bool = False


ROWS = (
    Row("parameters", 100_000, 4),
    Row("filename", 30_000, 1024),
    Row("physical", 20_000, 17),
    Row("physical_existing", 10_000, 17),
    Row("virtual", 20_000, 17),
    Row("filename_batch", 4, 8192, True),
    Row("physical_batch", 3, 4096, True),
    Row("virtual_batch", 3, 4096, True),
)


def scaled_count(value: int, scale: float) -> int:
    return max(1, int(round(value * scale)))


def summarize(values: list[str]) -> tuple[int, int]:
    text = compare_dataset_core.fnv_summary(values)
    checksum, output_bytes = text.split(":")
    return int(checksum), int(output_bytes)


def build_inputs(module, root: pathlib.Path, row: Row):
    physical = compare_dataset_core.physical_setting(root)
    virtual = compare_dataset_core.virtual_setting(root)
    if row.kind == "physical_existing":
        physical["topology"]["file_path"] = str(root / "Topo.multi.part.gml")
    if row.kind == "parameters":
        return [
            {},
            {"distribution": "exponential", "scale": 500},
            {"distribution": "poisson", "lam": 0.04},
            {"distribution": "uniform", "low": 0, "high": 100},
        ]
    if row.kind in ("filename", "filename_batch"):
        return [
            (
                {"solver_name": "solver" if index % 2 == 0 else "solver-x"},
                index,
                {
                    "index": index,
                    "even": index % 2 == 0,
                    "raw": "x=y/z",
                },
            )
            for index in range(row.batch_size)
        ]
    if row.kind in ("physical", "physical_existing", "physical_batch"):
        return physical
    if row.kind in ("virtual", "virtual_batch"):
        return virtual
    raise ValueError(f"unsupported row: {row.kind}")


def python_benchmark(module, root: pathlib.Path, row: Row) -> dict[str, int]:
    inputs = build_inputs(module, root, row)
    outputs: list[str] = []
    start = time.perf_counter_ns()
    if row.kind == "parameters":
        outputs = [
            module.get_parameters_string(
                module.get_distribution_parameters(inputs[index % len(inputs)])
            )
            for index in range(row.operations)
        ]
    elif row.kind == "filename":
        outputs = [
            module.generate_file_name(
                inputs[index % len(inputs)][0],
                epoch_id=inputs[index % len(inputs)][1],
                **inputs[index % len(inputs)][2],
            )
            for index in range(row.operations)
        ]
    elif row.kind in ("physical", "physical_existing"):
        outputs = [
            module.get_p_net_dataset_dir_from_setting(inputs, index % 17)
            for index in range(row.operations)
        ]
    elif row.kind == "virtual":
        outputs = [
            module.get_v_nets_dataset_dir_from_setting(inputs, index % 17)
            for index in range(row.operations)
        ]
    elif row.kind == "filename_batch":
        for _ in range(row.operations):
            outputs = [
                module.generate_file_name(config, epoch_id=epoch, **items)
                for config, epoch, items in inputs
            ]
    elif row.kind == "physical_batch":
        for _ in range(row.operations):
            outputs = [
                module.get_p_net_dataset_dir_from_setting(inputs, index % 17)
                for index in range(row.batch_size)
            ]
    elif row.kind == "virtual_batch":
        for _ in range(row.operations):
            outputs = [
                module.get_v_nets_dataset_dir_from_setting(inputs, index % 17)
                for index in range(row.batch_size)
            ]
    else:
        raise ValueError(f"unsupported Python row: {row.kind}")
    elapsed_ns = time.perf_counter_ns() - start
    checksum, output_bytes = summarize(outputs)
    return {
        "elapsed_ns": elapsed_ns,
        "checksum": checksum,
        "output_bytes": output_bytes,
    }


def cpp_benchmark(
    harness: pathlib.Path,
    root: pathlib.Path,
    row: Row,
    workers: int,
) -> dict[str, int]:
    process = subprocess.run(
        [
            str(harness),
            "benchmark",
            row.kind,
            str(row.operations),
            str(row.batch_size),
            str(workers),
            str(root),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ benchmark failed: {process.stderr.strip()}")
    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError(f"malformed C++ benchmark line: {line!r}")
        key, value = line.split("=", 1)
        values[key] = value
    required = {
        "benchmark_version",
        "kind",
        "workers",
        "operations",
        "batch_size",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "status",
    }
    if set(values) != required or values["benchmark_version"] != "1" or values["status"] != "PASS":
        raise RuntimeError(f"invalid C++ benchmark response: {values!r}")
    if values["kind"] != row.kind:
        raise RuntimeError("C++ benchmark kind drift")
    return {
        "elapsed_ns": int(values["elapsed_ns"]),
        "checksum": int(values["checksum"]),
        "output_bytes": int(values["output_bytes"]),
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


def benchmark_row(
    module,
    harness: pathlib.Path,
    root: pathlib.Path,
    row: Row,
    workers: int,
    warmups: int,
    repetitions: int,
) -> dict:
    python_samples: list[int] = []
    cpp_samples: list[int] = []
    semantic: tuple[int, int] | None = None
    total = warmups + repetitions
    for sample in range(total):
        if sample % 2 == 0:
            python = python_benchmark(module, root, row)
            cpp = cpp_benchmark(harness, root, row, workers)
        else:
            cpp = cpp_benchmark(harness, root, row, workers)
            python = python_benchmark(module, root, row)
        python_semantic = (python["checksum"], python["output_bytes"])
        cpp_semantic = (cpp["checksum"], cpp["output_bytes"])
        if python_semantic != cpp_semantic:
            raise RuntimeError(
                f"{row.kind}/w{workers} semantic mismatch: "
                f"python={python_semantic}, cpp={cpp_semantic}"
            )
        if semantic is None:
            semantic = python_semantic
        elif semantic != python_semantic:
            raise RuntimeError(f"{row.kind}/w{workers} checksum drift")
        if sample >= warmups:
            python_samples.append(python["elapsed_ns"])
            cpp_samples.append(cpp["elapsed_ns"])
    python_stats = timing_summary(python_samples)
    cpp_stats = timing_summary(cpp_samples)
    speedup = python_stats["median_ms"] / cpp_stats["median_ms"]
    assert semantic is not None
    return {
        "kind": row.kind,
        "workers": workers,
        "operations": row.operations,
        "batch_size": row.batch_size,
        "checksum": semantic[0],
        "output_bytes": semantic[1],
        "python": python_stats,
        "cpp": cpp_stats,
        "speedup": speedup,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 0])
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=[row.kind for row in ROWS],
        default=[row.kind for row in ROWS],
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1 or args.scale <= 0:
        raise ValueError("invalid benchmark protocol")
    if not args.workers or any(worker < 0 for worker in args.workers):
        raise ValueError("invalid worker list")

    module = compare_dataset_core.load_oracle(args.python_source)
    rows = tuple(
        Row(
            row.kind,
            scaled_count(row.operations, args.scale),
            scaled_count(row.batch_size, args.scale),
            row.batch,
        )
        for row in ROWS
        if row.kind in args.kinds
    )
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="virne_dataset_core_bench_") as text_root:
        root = pathlib.Path(text_root)
        (root / "Topo.multi.part.gml").write_bytes(b"fixture")
        for row in rows:
            widths = args.workers if row.batch else [1]
            for workers in widths:
                result = benchmark_row(
                    module,
                    args.harness,
                    root,
                    row,
                    workers,
                    args.warmups,
                    args.repetitions,
                )
                results.append(result)
                worker_text = "auto" if workers == 0 else str(workers)
                print(
                    f"{row.kind}/w{worker_text}: "
                    f"python={result['python']['median_ms']:.6f} ms, "
                    f"cpp={result['cpp']['median_ms']:.6f} ms, "
                    f"speedup={result['speedup']:.3f}x"
                )

    payload = {
        "source_sha256": compare_dataset_core.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "protocol": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "paired_alternating_order": True,
            "process_startup_excluded": True,
            "fixture_setup_and_verification_excluded": True,
            "thread_creation_included_for_batch_api": True,
            "workers": args.workers,
            "scale": args.scale,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpus_visible": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        },
        "rows": results,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"dataset core benchmark: PASS ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
