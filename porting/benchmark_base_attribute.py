#!/usr/bin/env python3
"""Compact semantic-gated BaseAttribute benchmark and worker smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_base_attribute


KINDS = ("customized", "exponential_int")


def fnv1a(payload: bytes) -> int:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def python_run(module, BaseNetwork, kind: str, count: int, seed: int) -> dict:
    class Probe(module.BaseAttribute):
        def set_data(self, network, attribute_data):
            return None

        def get_data(self, network):
            return []

    if kind == "customized":
        value = Probe(
            "benchmark", "node", "resource", True,
            distribution="customized", dtype="bool", min=-2.5, max=7.25,
        )
        dtype = module.np.dtype("float64")
    elif kind == "exponential_int":
        value = Probe(
            "benchmark", "node", "resource", True,
            distribution="exponential", dtype="int", scale=0.5,
        )
        dtype = module.np.dtype("int64")
    else:
        raise ValueError(f"unknown benchmark kind: {kind}")
    network = BaseNetwork(count, count // 2)
    module.np.random.seed(seed)
    started = time.perf_counter_ns()
    generated = value._generate_data(network)
    elapsed_ns = time.perf_counter_ns() - started
    array = module.np.ascontiguousarray(generated, dtype=dtype)
    payload = array.tobytes(order="C")
    next_bits = compare_base_attribute.float_bits(module.np.random.random_sample())
    return {
        "elapsed_ns": elapsed_ns,
        "checksum": fnv1a(payload),
        "output_bytes": len(payload),
        "next_bits": next_bits,
    }


def cpp_run(
    benchmark: pathlib.Path,
    kind: str,
    count: int,
    workers: int,
    seed: int,
) -> dict:
    process = subprocess.run(
        [str(benchmark), kind, str(count), str(workers), str(seed)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ benchmark failed: {process.stderr.strip()}")
    values = dict(line.split("=", 1) for line in process.stdout.splitlines())
    required = {
        "protocol", "kind", "count", "workers", "elapsed_ns", "checksum",
        "output_bytes", "next_bits", "status",
    }
    if (
        set(values) != required
        or values["protocol"] != "1"
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
        "next_bits": f"{int(values['next_bits']):016x}",
    }


def median_ms(samples: list[int]) -> float:
    return statistics.median(samples) / 1_000_000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=300_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if (
        args.count < 1 or args.warmups < 0 or args.repetitions < 1
        or not args.workers or any(worker < 0 for worker in args.workers)
        or len(set(args.workers)) != len(args.workers)
        or args.seed < 0 or args.seed > 0xFFFFFFFF
    ):
        raise ValueError("invalid compact BaseAttribute benchmark protocol")

    module, dataset, BaseNetwork, previous, module_name = (
        compare_base_attribute.load_base_oracle(args.source, args.dataset_source)
    )
    rows = []
    try:
        for kind in KINDS:
            python_samples: list[int] = []
            cpp_samples = {worker: [] for worker in args.workers}
            locked_semantic = None
            for sample in range(args.warmups + args.repetitions):
                python = python_run(module, BaseNetwork, kind, args.count, args.seed)
                for worker in args.workers:
                    cpp = cpp_run(
                        args.benchmark, kind, args.count, worker, args.seed
                    )
                    python_semantic = (
                        python["checksum"], python["output_bytes"], python["next_bits"]
                    )
                    cpp_semantic = (
                        cpp["checksum"], cpp["output_bytes"], cpp["next_bits"]
                    )
                    if cpp_semantic != python_semantic:
                        raise RuntimeError(
                            f"{kind}/w{worker} semantic mismatch: "
                            f"python={python_semantic}, cpp={cpp_semantic}"
                        )
                    if locked_semantic is None:
                        locked_semantic = python_semantic
                    elif locked_semantic != python_semantic:
                        raise RuntimeError(f"{kind} semantic drift between samples")
                    if sample >= args.warmups:
                        cpp_samples[worker].append(cpp["elapsed_ns"])
                if sample >= args.warmups:
                    python_samples.append(python["elapsed_ns"])

            python_ms = median_ms(python_samples)
            for worker in args.workers:
                cpp_ms = median_ms(cpp_samples[worker])
                row = {
                    "kind": kind,
                    "workers": worker,
                    "count": args.count,
                    "checksum": locked_semantic[0],
                    "output_bytes": locked_semantic[1],
                    "next_bits": locked_semantic[2],
                    "python_median_ms": python_ms,
                    "cpp_median_ms": cpp_ms,
                    "speedup": python_ms / cpp_ms,
                }
                rows.append(row)
                print(
                    f"{kind}/w{worker}: python={python_ms:.6f} ms, "
                    f"cpp={cpp_ms:.6f} ms, speedup={row['speedup']:.3f}x"
                )
    finally:
        compare_base_attribute.restore_oracle(previous, module_name)

    slower = [row for row in rows if row["speedup"] <= 1.0]
    if slower:
        labels = ", ".join(
            f"{row['kind']}/w{row['workers']}={row['speedup']:.3f}x"
            for row in slower
        )
        raise RuntimeError(f"C++-faster representative gate failed: {labels}")

    payload = {
        "source_sha256": compare_base_attribute.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "process_startup_excluded": True,
            "checksum_and_rng_continuation_excluded": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"base_attribute benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
