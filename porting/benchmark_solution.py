#!/usr/bin/env python3
"""One compact accepted lifecycle benchmark for the pinned Solution leaf."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import time
from types import SimpleNamespace

from compare_solution import load_solution_class


MASK64 = (1 << 64) - 1
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211


def mix_u64(checksum: int, value: int) -> int:
    value &= MASK64
    for _ in range(8):
        checksum ^= value & 0xFF
        checksum = (checksum * FNV_PRIME) & MASK64
        value >>= 8
    return checksum


def double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def python_lifecycle(Solution, metadata):
    started = time.perf_counter_ns()
    solutions = [Solution(item) for item in metadata]
    for index, solution in enumerate(solutions):
        solution.result = True
        solution.v_net_cost = index * 0.5
        solution.description = "dirty"
        solution.node_slots[index % 7] = index % 11
        solution.num_placed_nodes = index % 13
    for solution in solutions:
        solution.reset()
    feasible_count = 0
    for index, solution in enumerate(solutions):
        solution.result = index % 3 != 0
        solution.v_net_total_hard_constraint_violation = (
            0.5 if index % 5 == 0 else -0.25
        )
        feasible_count += int(solution.is_feasible())
    elapsed_ns = time.perf_counter_ns() - started

    checksum = FNV_OFFSET
    for solution in solutions:
        for value in (
            solution.v_net_id,
            double_bits(solution.v_net_lifetime),
            double_bits(solution.v_net_arrival_time),
            solution.v_net_num_nodes,
            solution.v_net_num_egdes,
            int(solution.result),
            double_bits(solution.v_net_total_hard_constraint_violation),
            solution.num_placed_nodes,
        ):
            checksum = mix_u64(checksum, value)
    return {
        "elapsed_ns": elapsed_ns,
        "entry_count": len(solutions),
        "output_bytes": len(solutions) * 64,
        "feasible_count": feasible_count,
        "checksum": checksum,
    }


def cpp_lifecycle(harness: pathlib.Path, count: int, workers: int):
    process = subprocess.run(
        [str(harness), "benchmark", str(count), str(workers)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Solution benchmark harness failed: {process.stderr.strip()}")
    fields = {}
    for item in process.stdout.strip().split(";"):
        key, value = item.split("=", 1)
        fields[key] = int(value)
    required = {
        "elapsed_ns", "entry_count", "output_bytes", "feasible_count", "checksum"
    }
    if fields.keys() != required:
        raise RuntimeError(f"malformed Solution benchmark output: {process.stdout!r}")
    return fields


def median_sample(callable_, warmups: int, repetitions: int):
    for _ in range(warmups):
        callable_()
    samples = [callable_() for _ in range(repetitions)]
    fingerprint = {
        key: samples[0][key]
        for key in ("entry_count", "output_bytes", "feasible_count", "checksum")
    }
    for sample in samples[1:]:
        if any(sample[key] != value for key, value in fingerprint.items()):
            raise RuntimeError("Solution benchmark output changed between repetitions")
    return statistics.median(sample["elapsed_ns"] for sample in samples), fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=32768)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.count <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("invalid benchmark dimensions")

    Solution = load_solution_class(args.source)
    metadata = [
        SimpleNamespace(
            id=index,
            lifetime=1.0 + (index % 17) * 0.25,
            arrival_time=index * 0.125,
            num_nodes=2 + index % 31,
            num_links=1 + index % 37,
        )
        for index in range(args.count)
    ]
    python_ns, python_fingerprint = median_sample(
        lambda: python_lifecycle(Solution, metadata),
        args.warmups, args.repetitions,
    )

    rows = []
    for workers in args.workers:
        cpp_ns, cpp_fingerprint = median_sample(
            lambda workers=workers: cpp_lifecycle(
                args.harness.resolve(), args.count, workers
            ),
            args.warmups, args.repetitions,
        )
        if cpp_fingerprint != python_fingerprint:
            raise RuntimeError(
                f"Solution benchmark fingerprint mismatch w{workers}: "
                f"C++={cpp_fingerprint}, Python={python_fingerprint}"
            )
        speedup = python_ns / cpp_ns
        if speedup <= 1.0:
            raise RuntimeError(f"Solution benchmark slower at w{workers}: {speedup:.6f}x")
        rows.append({
            "kind": "solution_lifecycle",
            "workers": workers,
            "count": args.count,
            "entry_count": cpp_fingerprint["entry_count"],
            "output_bytes": cpp_fingerprint["output_bytes"],
            "feasible_count": cpp_fingerprint["feasible_count"],
            "checksum": cpp_fingerprint["checksum"],
            "python_median_ms": python_ns / 1_000_000,
            "cpp_median_ms": cpp_ns / 1_000_000,
            "speedup": speedup,
        })

    payload = {
        "result": "PASS",
        "source_sha256": "ec5d64ef6695350f718a818461fbf60934b5077d8add39bb7ee8be18cfe1e78b",
        "protocol": {
            "count": args.count,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "caller_configured_workers": True,
            "worker_auto_tuning": False,
            "fixture_creation_excluded": True,
            "process_startup_excluded": True,
            "fingerprint_excluded": True,
            "binary_output_gate_before_acceptance": True,
        },
        "rows": rows,
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for row in rows:
        print(
            f"Solution lifecycle w{row['workers']}: "
            f"Python {row['python_median_ms']:.6f} ms, "
            f"C++ {row['cpp_median_ms']:.6f} ms, {row['speedup']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
