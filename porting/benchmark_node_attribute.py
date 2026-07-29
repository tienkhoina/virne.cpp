#!/usr/bin/env python3
"""Compact semantic-gated benchmark for the NodeAttribute leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_node_attribute


KINDS = ("dense_roundtrip", "position")


def fnv1a(payload: bytes) -> int:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def python_run(node, BaseNetwork, kind: str, count: int, seed: int) -> dict:
    if kind == "dense_roundtrip":
        graph = BaseNetwork(count)
        item = node.NodeAttribute("load", "node", "status")
        input_values = [index % 1009 - 504 for index in range(count)]
        started = time.perf_counter_ns()
        item.set_data(graph, input_values)
        output = item.get_data(graph)
        elapsed_ns = time.perf_counter_ns() - started
        array = node.np.ascontiguousarray(output, dtype=node.np.int64)
        next_bits = "0000000000000000"
    elif kind == "position":
        graph = BaseNetwork(count)
        item = compare_node_attribute.configure_position(
            node, "uniform", "float", -0.25, 0.75, low=-2.0, high=3.0
        )
        node.np.random.seed(seed)
        started = time.perf_counter_ns()
        output = item.generate_data(graph)
        elapsed_ns = time.perf_counter_ns() - started
        array = node.np.ascontiguousarray(output, dtype=node.np.float64)
        next_bits = compare_node_attribute.bits(node.np.random.random_sample())
    else:
        raise ValueError(f"unknown benchmark kind: {kind}")
    payload = array.tobytes(order="C")
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
        raise RuntimeError(f"C++ node benchmark failed: {process.stderr.strip()}")
    values = dict(line.split("=", 1) for line in process.stdout.splitlines())
    required = {
        "protocol", "kind", "count", "workers", "elapsed_ns", "checksum",
        "output_bytes", "next_bits", "status",
    }
    if (
        set(values) != required or values["protocol"] != "1"
        or values["status"] != "PASS" or values["kind"] != kind
        or int(values["count"]) != count or int(values["workers"]) != workers
    ):
        raise RuntimeError(f"invalid C++ node benchmark response: {values!r}")
    return {
        "elapsed_ns": int(values["elapsed_ns"]),
        "checksum": int(values["checksum"]),
        "output_bytes": int(values["output_bytes"]),
        "next_bits": f"{int(values['next_bits']):016x}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--method-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=100_000)
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
        raise ValueError("invalid compact node benchmark protocol")

    node, base, method, dataset, BaseNetwork, previous, names = (
        compare_node_attribute.load_oracle(
            args.source, args.base_source, args.method_source, args.dataset_source
        )
    )
    rows = []
    try:
        for kind in KINDS:
            python_samples = []
            cpp_samples = {worker: [] for worker in args.workers}
            semantic = None
            for sample in range(args.warmups + args.repetitions):
                python = python_run(node, BaseNetwork, kind, args.count, args.seed)
                for worker in args.workers:
                    cpp = cpp_run(
                        args.benchmark, kind, args.count, worker, args.seed
                    )
                    python_gate = (
                        python["checksum"], python["output_bytes"], python["next_bits"]
                    )
                    cpp_gate = (
                        cpp["checksum"], cpp["output_bytes"], cpp["next_bits"]
                    )
                    if cpp_gate != python_gate:
                        raise RuntimeError(
                            f"{kind}/w{worker} semantic mismatch: "
                            f"python={python_gate}, cpp={cpp_gate}"
                        )
                    if semantic is None:
                        semantic = python_gate
                    elif semantic != python_gate:
                        raise RuntimeError(f"{kind} output drift between samples")
                    if sample >= args.warmups:
                        cpp_samples[worker].append(cpp["elapsed_ns"])
                if sample >= args.warmups:
                    python_samples.append(python["elapsed_ns"])

            python_ms = statistics.median(python_samples) / 1_000_000.0
            for worker in args.workers:
                cpp_ms = statistics.median(cpp_samples[worker]) / 1_000_000.0
                row = {
                    "kind": kind,
                    "workers": worker,
                    "count": args.count,
                    "checksum": semantic[0],
                    "output_bytes": semantic[1],
                    "next_bits": semantic[2],
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
        compare_node_attribute.restore(previous, names)

    slower = [row for row in rows if row["speedup"] <= 1.0]
    if slower:
        labels = ", ".join(
            f"{row['kind']}/w{row['workers']}={row['speedup']:.3f}x"
            for row in slower
        )
        raise RuntimeError(f"C++-faster representative gate failed: {labels}")

    payload = {
        "source_sha256": compare_node_attribute.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "networkx_version": node.nx.__version__,
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
            "fixture_creation_and_checksum_excluded": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"node_attribute benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
