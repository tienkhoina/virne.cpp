#!/usr/bin/env python3
"""One compact caller-configured BaseNetwork benchmark; run only after diff."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from typing import Any

import compare_base_network as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fingerprint(value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def make_operation(module, operation: str, count: int):
    network = differential.benchmark_fixture(module, count)
    node_attrs = network.get_node_attrs(names=["cpu", "peak"])
    link_attrs = network.get_link_attrs(names=["bw", "peak_bw"])
    entries = 2 * network.number_of_nodes() + 2 * network.number_of_edges()

    if operation == "get":
        def run():
            return (
                network.get_node_attrs_data(node_attrs),
                network.get_link_attrs_data(link_attrs),
            )

        def encode(value) -> str:
            node_rows, link_rows = value
            return (
                "node=" + differential.canonical_rows(node_rows)
                + ";link=" + differential.canonical_rows(link_rows)
            )

        return run, encode, entries

    if operation == "set":
        nodes = list(network.nodes)
        edges = list(network.edges)
        node_updates = {
            network.node_attrs["cpu"]: [
                (index * 3) % 997 for index, _node in enumerate(nodes)
            ],
            network.node_attrs["peak"]: [
                float((index * 5) % 991) + 0.75
                for index, _node in enumerate(nodes)
            ],
        }
        link_updates = {
            network.link_attrs["bw"]: [
                (index * 7) % 983 for index, _edge in enumerate(edges)
            ],
            network.link_attrs["peak_bw"]: [
                float((index * 11) % 977) + 0.125
                for index, _edge in enumerate(edges)
            ],
        }

        def run():
            network.set_node_attrs_data(node_updates)
            network.set_link_attrs_data(link_updates)
            return None

        def encode(_value) -> str:
            return (
                "node=" + differential.canonical_rows(
                    network.get_node_attrs_data(node_attrs)
                )
                + ";link=" + differential.canonical_rows(
                    network.get_link_attrs_data(link_attrs)
                )
            )

        return run, encode, entries

    if operation == "manager":
        manager = module._benchmark_oracle.AttributeBenchmarkManager

        def run():
            return manager.get_benchmarks(network)

        return run, differential.canonical_benchmarks, 3

    raise RuntimeError(f"unknown operation: {operation}")


def parse_cpp(
    path: pathlib.Path,
    operation: str,
    count: int,
    workers: int,
) -> dict[str, int]:
    process = subprocess.run(
        [str(path), operation, str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"C++ BaseNetwork benchmark failed: {process.stderr.strip()}"
        )
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed C++ benchmark output: {line!r}")
        fields[key] = value
    required = {
        "protocol",
        "kind",
        "operation",
        "count",
        "workers",
        "type_tag",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "entry_count",
        "status",
    }
    if (
        set(fields) != required
        or fields["protocol"] != "1"
        or fields["kind"] != "base_network"
        or fields["operation"] != operation
        or fields["type_tag"] != "attrvalue_bits_order_v1"
        or fields["status"] != "PASS"
        or int(fields["count"]) != count
        or int(fields["workers"]) != workers
    ):
        raise RuntimeError(f"invalid C++ benchmark protocol: {fields!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
        "entry_count": int(fields["entry_count"]),
    }


def validate_output(
    current: dict[str, int],
    checksum: int,
    output_bytes: int,
    entries: int,
    label: str,
) -> None:
    if (
        current["checksum"] != checksum
        or current["output_bytes"] != output_bytes
        or current["entry_count"] != entries
    ):
        raise RuntimeError(f"{label} raw-type/bit/order output drift")


def time_python(
    run: Callable[[], Any], warmups: int, repetitions: int
) -> tuple[int, Any]:
    value = None
    for _ in range(warmups):
        value = run()
    samples: list[int] = []
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        value = run()
        end = time.perf_counter_ns()
        samples.append(end - begin)
    return int(statistics.median(samples)), value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument(
        "--operations", nargs="+", default=["get", "set", "manager"]
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    allowed_operations = {"get", "set", "manager"}
    if (
        args.count < 4
        or args.warmups < 0
        or args.repetitions <= 0
        or not args.operations
        or any(operation not in allowed_operations for operation in args.operations)
        or len(set(args.operations)) != len(args.operations)
        or not args.workers
        or any(worker < 0 for worker in args.workers)
    ):
        raise RuntimeError("invalid BaseNetwork benchmark protocol")

    module = differential.load_oracle(args.source)
    rows = []
    try:
        for operation in args.operations:
            run, encode, entries = make_operation(module, operation, args.count)

            # Exact types, raw floating bits, ordering, and cardinality gate
            # complete before either runtime is sampled.
            reference_value = run()
            reference = encode(reference_value)
            expected_checksum, expected_bytes = fingerprint(reference)
            for workers in args.workers:
                validate_output(
                    parse_cpp(args.benchmark, operation, args.count, workers),
                    expected_checksum,
                    expected_bytes,
                    entries,
                    f"{operation}/w{workers} pre-timing gate",
                )

            python_ns, python_value = time_python(
                run, args.warmups, args.repetitions
            )
            final_checksum, final_bytes = fingerprint(encode(python_value))
            if (
                final_checksum != expected_checksum
                or final_bytes != expected_bytes
            ):
                raise RuntimeError(f"Python {operation} timed output drift")

            for workers in args.workers:
                for _ in range(args.warmups):
                    validate_output(
                        parse_cpp(args.benchmark, operation, args.count, workers),
                        expected_checksum,
                        expected_bytes,
                        entries,
                        f"{operation}/w{workers} warm-up",
                    )
                samples = []
                for _ in range(args.repetitions):
                    current = parse_cpp(
                        args.benchmark, operation, args.count, workers
                    )
                    validate_output(
                        current,
                        expected_checksum,
                        expected_bytes,
                        entries,
                        f"{operation}/w{workers} sample",
                    )
                    samples.append(current["elapsed_ns"])
                cpp_ns = int(statistics.median(samples))
                speedup = python_ns / cpp_ns
                if speedup <= 1.0:
                    raise RuntimeError(
                        f"{operation}/w{workers} C++ did not beat Python: "
                        f"{speedup:.3f}x"
                    )
                row = {
                    "operation": operation,
                    "count": args.count,
                    "workers": workers,
                    "type_tag": "attrvalue_bits_order_v1",
                    "python_median_ms": python_ns / 1e6,
                    "cpp_median_ms": cpp_ns / 1e6,
                    "speedup": speedup,
                    "checksum": expected_checksum,
                    "output_bytes": expected_bytes,
                    "entry_count": entries,
                }
                rows.append(row)
                print(
                    f"{operation}/w{workers}: "
                    f"python={row['python_median_ms']:.6f} ms, "
                    f"cpp={row['cpp_median_ms']:.6f} ms, "
                    f"speedup={speedup:.3f}x"
                )
        numpy_version = module.np.__version__
        networkx_version = module.nx.__version__
    finally:
        differential.unload_oracle()

    payload = {
        "source_sha256": differential.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "numpy_version": numpy_version,
        "networkx_version": networkx_version,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count,
            "operations": args.operations,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_and_input_creation_excluded": True,
            "cpp_process_startup_excluded": True,
            "fingerprint_excluded": True,
            "raw_types_bits_order_gate_before_timing": True,
            "caller_configured_workers": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"BaseNetwork benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
