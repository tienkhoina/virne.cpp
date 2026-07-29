#!/usr/bin/env python3
"""Compact exact benchmark for the completed LinkAttribute leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_link_attribute
import compare_node_attribute


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv1a(payload: bytes) -> int:
    result = FNV_OFFSET
    for byte in payload:
        result ^= byte
        result = (result * FNV_PRIME) & MASK64
    return result


def parse_cpp(path: pathlib.Path, kind: str, count: int, workers: int):
    process = subprocess.run(
        [str(path), kind, str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ benchmark failed: {process.stderr.strip()}")
    fields = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed C++ benchmark output: {line!r}")
        fields[key] = value
    required = {
        "protocol", "kind", "count", "workers", "elapsed_ns",
        "checksum", "output_bytes", "status",
    }
    if set(fields) != required or fields["protocol"] != "1" \
            or fields["status"] != "PASS" or fields["kind"] != kind:
        raise RuntimeError(f"invalid C++ protocol: {fields!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
    }


def dense_fixture(link, BaseNetwork, count: int):
    graph = BaseNetwork(count + 1)
    graph.add_edges_from((index, index + 1) for index in range(count))
    item = link.LinkAttribute("load", "link", "resource")
    values = [index % 1009 for index in range(count)]

    def run():
        item.set_data(graph, values)
        return item.get_data(graph)

    return run, lambda output: link.np.asarray(output, dtype=link.np.int64).tobytes()


def position_fixture(link, BaseNetwork, count: int):
    graph = BaseNetwork(count + 1)
    graph.add_edges_from((index, index + 1) for index in range(count))
    for index in range(count + 1):
        graph.nodes[index]["pos"] = (index % 997, (index * 17) % 991)
    latency = link.LinkLatencyAttribute(
        "latency",
        config={
            "generative": True,
            "distribution": "position",
            "min": 0.0,
            "max": 1.0,
        },
    )

    def run():
        return latency.generate_data(graph)

    return run, lambda output: link.np.asarray(output, dtype=link.np.float64).tobytes()


def time_python(run, encode, warmups: int, repetitions: int):
    for _ in range(warmups):
        output = run()
    samples = []
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        output = run()
        end = time.perf_counter_ns()
        samples.append(end - begin)
    payload = encode(output)
    return statistics.median(samples), fnv1a(payload), len(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--node-source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--method-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.count <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise RuntimeError("invalid benchmark protocol")

    link, dataset, BaseNetwork, _, previous, names = \
        compare_link_attribute.load_oracle(args)
    try:
        fixtures = {
            "dense_roundtrip": dense_fixture(link, BaseNetwork, args.count),
            "position_latency": position_fixture(link, BaseNetwork, args.count),
        }
        python_results = {}
        for kind, (run, encode) in fixtures.items():
            python_results[kind] = time_python(
                run, encode, args.warmups, args.repetitions
            )

        rows = []
        for kind in fixtures:
            python_ns, expected_checksum, expected_bytes = python_results[kind]
            for workers in args.workers:
                for _ in range(args.warmups):
                    warm = parse_cpp(args.benchmark, kind, args.count, workers)
                    if warm["checksum"] != expected_checksum \
                            or warm["output_bytes"] != expected_bytes:
                        raise RuntimeError(f"{kind}/w{workers} warm-up drift")
                samples = []
                for _ in range(args.repetitions):
                    current = parse_cpp(args.benchmark, kind, args.count, workers)
                    if current["checksum"] != expected_checksum \
                            or current["output_bytes"] != expected_bytes:
                        raise RuntimeError(f"{kind}/w{workers} output drift")
                    samples.append(current["elapsed_ns"])
                cpp_ns = statistics.median(samples)
                speedup = python_ns / cpp_ns
                if speedup <= 1.0:
                    raise RuntimeError(
                        f"{kind}/w{workers} C++ did not beat Python: {speedup:.3f}x"
                    )
                row = {
                    "kind": kind,
                    "count": args.count,
                    "workers": workers,
                    "python_median_ms": python_ns / 1e6,
                    "cpp_median_ms": cpp_ns / 1e6,
                    "speedup": speedup,
                    "checksum": expected_checksum,
                    "output_bytes": expected_bytes,
                }
                rows.append(row)
                print(
                    f"{kind}/w{workers}: python={row['python_median_ms']:.6f} ms, "
                    f"cpp={row['cpp_median_ms']:.6f} ms, speedup={speedup:.3f}x"
                )
    finally:
        compare_node_attribute.restore(previous, names)

    payload = {
        "source_sha256": compare_link_attribute.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "networkx_version": link.nx.__version__,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "protocol": {
            "count": args.count,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_creation_and_checksum_excluded": True,
            "process_startup_excluded": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"link_attribute benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
