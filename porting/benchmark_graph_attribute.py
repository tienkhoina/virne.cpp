#!/usr/bin/env python3
"""One compact exact benchmark for the GraphAttribute batch extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_graph_attribute


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv1a(payload: bytes) -> int:
    result = FNV_OFFSET
    for byte in payload:
        result ^= byte
        result = (result * FNV_PRIME) & MASK64
    return result


def input_value(index: int) -> float:
    if index % 4093 == 0:
        return compare_graph_attribute.from_bits(
            0x7FF8000000000000 | ((index & 0xFFFF) + 1)
        )
    if index % 257 == 0:
        return -0.0
    return float((index % 1009) - 504) * 0.25


def parse_cpp(path: pathlib.Path, count: int, workers: int):
    process = subprocess.run(
        [str(path), str(count), str(workers)],
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
        "protocol", "kind", "count", "workers", "type_tag", "bits",
        "elapsed_ns", "checksum", "output_bytes", "status",
    }
    if set(fields) != required or fields["protocol"] != "1" \
            or fields["kind"] != "independent_roundtrip" \
            or fields["type_tag"] != "double" or fields["bits"] != "raw64" \
            or fields["status"] != "PASS" \
            or int(fields["count"]) != count \
            or int(fields["workers"]) != workers:
        raise RuntimeError(f"invalid C++ protocol: {fields!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
    }


def python_fixture(graph_module, BaseNetwork, dataset, count: int):
    networks = [BaseNetwork() for _ in range(count)]
    for index, network in enumerate(networks):
        if index & 1:
            network.graph["padding"] = 0
        if index % 3 == 0:
            network.graph["other_padding"] = 0
    values = [input_value(index) for index in range(count)]
    item = graph_module.GraphAttribute(
        "roundtrip_value", "graph", "status"
    )

    def run():
        for network, value in zip(networks, values):
            item.set_data(network, value)
        return [item.get_data(network) for network in networks]

    def encode(output):
        if any(type(value) is not float for value in output):
            raise RuntimeError("Python graph benchmark output lane drift")
        return dataset.np.asarray(output, dtype=dataset.np.float64).tobytes()

    return run, encode


def time_python(run, encode, warmups: int, repetitions: int):
    output = None
    for _ in range(warmups):
        output = run()
    samples = []
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        output = run()
        end = time.perf_counter_ns()
        samples.append(end - begin)
    if output is None:
        raise RuntimeError("Python graph benchmark produced no output")
    payload = encode(output)
    return statistics.median(samples), fnv1a(payload), len(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--method-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.count <= 0 or args.warmups < 0 or args.repetitions <= 0 \
            or any(worker < 0 for worker in args.workers):
        raise RuntimeError("invalid benchmark protocol")

    (
        graph,
        _,
        _,
        dataset,
        nx,
        BaseNetwork,
        _,
        previous,
        names,
    ) = compare_graph_attribute.load_oracle(args)
    try:
        run, encode = python_fixture(graph, BaseNetwork, dataset, args.count)
        python_ns, expected_checksum, expected_bytes = time_python(
            run, encode, args.warmups, args.repetitions
        )
        rows = []
        for workers in args.workers:
            for _ in range(args.warmups):
                warm = parse_cpp(args.benchmark, args.count, workers)
                if warm["checksum"] != expected_checksum \
                        or warm["output_bytes"] != expected_bytes:
                    raise RuntimeError(f"worker {workers} warm-up output drift")
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(args.benchmark, args.count, workers)
                if current["checksum"] != expected_checksum \
                        or current["output_bytes"] != expected_bytes:
                    raise RuntimeError(f"worker {workers} output drift")
                samples.append(current["elapsed_ns"])
            cpp_ns = statistics.median(samples)
            speedup = python_ns / cpp_ns
            if speedup <= 1.0:
                raise RuntimeError(
                    f"worker {workers} C++ did not beat Python: {speedup:.3f}x"
                )
            row = {
                "kind": "independent_roundtrip",
                "count": args.count,
                "workers": workers,
                "type_tag": "double",
                "bits": "raw64",
                "python_median_ms": python_ns / 1e6,
                "cpp_median_ms": cpp_ns / 1e6,
                "speedup": speedup,
                "checksum": expected_checksum,
                "output_bytes": expected_bytes,
            }
            rows.append(row)
            print(
                f"independent_roundtrip/w{workers}: "
                f"python={row['python_median_ms']:.6f} ms, "
                f"cpp={row['cpp_median_ms']:.6f} ms, speedup={speedup:.3f}x"
            )
    finally:
        compare_graph_attribute.restore(previous, names)

    payload = {
        "source_sha256": compare_graph_attribute.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "networkx_version": nx.__version__,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_creation_and_checksum_excluded": True,
            "process_startup_excluded": True,
            "single_workload": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"graph_attribute benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
