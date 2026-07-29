#!/usr/bin/env python3
"""One compact checksum-gated ConstraintChecker benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import time

from compare_constraint_checker import fixture, load_checker


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


def number_bits(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value & MASK64
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def python_batch(checker, virtual, physical, requests):
    started = time.perf_counter_ns()
    checksum = FNV_OFFSET
    feasible_count = 0
    for virtual_node, physical_node in requests:
        feasible, offsets = checker.check_node_level_constraints(
            virtual, physical, virtual_node, physical_node)
        feasible_count += int(feasible)
        checksum = mix_u64(checksum, int(feasible))
        checksum = mix_u64(checksum, number_bits(offsets["node_hard"]))
        checksum = mix_u64(checksum, number_bits(offsets["node_soft"]))
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "elapsed_ns": elapsed_ns,
        "entry_count": len(requests),
        "feasible_count": feasible_count,
        "checksum": checksum,
    }


def cpp_batch(harness: pathlib.Path, count: int, workers: int):
    process = subprocess.run(
        [str(harness), "benchmark", str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ConstraintChecker benchmark harness failed: {process.stderr.strip()}")
    fields = {}
    for item in process.stdout.strip().split(";"):
        key, value = item.split("=", 1)
        fields[key] = int(value)
    required = {
        "elapsed_ns", "entry_count", "feasible_count", "checksum"
    }
    if fields.keys() != required:
        raise RuntimeError(f"malformed benchmark output: {process.stdout!r}")
    return fields


def median_sample(callable_, warmups: int, repetitions: int):
    for _ in range(warmups):
        callable_()
    samples = [callable_() for _ in range(repetitions)]
    fingerprint = {
        key: samples[0][key]
        for key in ("entry_count", "feasible_count", "checksum")
    }
    for sample in samples[1:]:
        if any(sample[key] != value for key, value in fingerprint.items()):
            raise RuntimeError("benchmark fingerprint changed between samples")
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
    if args.count < 0 or args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("count/warmups/repetitions are out of range")

    Checker = load_checker(args.source)
    full_checker, _, virtual, physical = fixture(Checker)
    checker = Checker(
        full_checker.node_constraint_attrs_checking_at_node,
        [],
        [],
        [],
    )
    requests = [
        (index % 3, (index * 3 + 1) % 4)
        for index in range(args.count)
    ]

    python_ns, fingerprint = median_sample(
        lambda: python_batch(checker, virtual, physical, requests),
        args.warmups,
        args.repetitions,
    )
    rows = []
    for workers in args.workers:
        cpp_ns, cpp_fingerprint = median_sample(
            lambda workers=workers: cpp_batch(
                args.harness, args.count, workers),
            args.warmups,
            args.repetitions,
        )
        if cpp_fingerprint != fingerprint:
            raise RuntimeError(
                f"C++ fingerprint mismatch at workers={workers}: "
                f"Python={fingerprint}, C++={cpp_fingerprint}")
        rows.append({
            "workers": workers,
            "cpp_median_ns": cpp_ns,
            "speedup_vs_python": python_ns / cpp_ns,
        })

    report = {
        "component": "core.controller.ConstraintChecker",
        "case": "node_batch",
        "count": args.count,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "python_median_ns": python_ns,
        "fingerprint": fingerprint,
        "cpp": rows,
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
