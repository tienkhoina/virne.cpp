#!/usr/bin/env python3
"""One compact checksum-gated ResourceUpdator batch benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import time

from compare_resource_updator import FakeNetwork, load_updator


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


def python_batch(ResourceUpdator, count: int):
    physical = FakeNetwork(
        nodes={
            index: {"cpu": 100 + index % 17}
            for index in range(count)
        },
        links={},
    )
    updator = ResourceUpdator([])
    started = time.perf_counter_ns()
    for index in range(count):
        updator.update_node_resources(
            physical,
            index,
            {"cpu": 1 + index % 7},
            "-",
            True,
        )
    checksum = FNV_OFFSET
    value_sum = 0
    for index in range(count):
        value = physical.nodes[index]["cpu"]
        value_sum += value
        checksum = mix_u64(checksum, value)
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "elapsed_ns": elapsed_ns,
        "entry_count": count,
        "value_sum": value_sum,
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
            f"ResourceUpdator benchmark harness failed: {process.stderr.strip()}")
    fields = {}
    for item in process.stdout.strip().split(";"):
        key, value = item.split("=", 1)
        fields[key] = int(value)
    required = {"elapsed_ns", "entry_count", "value_sum", "checksum"}
    if fields.keys() != required:
        raise RuntimeError(f"malformed benchmark output: {process.stdout!r}")
    return fields


def median_sample(callable_, warmups: int, repetitions: int):
    for _ in range(warmups):
        callable_()
    samples = [callable_() for _ in range(repetitions)]
    fingerprint = {
        key: samples[0][key]
        for key in ("entry_count", "value_sum", "checksum")
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

    ResourceUpdator = load_updator(args.source)
    python_ns, fingerprint = median_sample(
        lambda: python_batch(ResourceUpdator, args.count),
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
        "component": "core.controller.ResourceUpdator",
        "case": "disjoint_node_batch",
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
