#!/usr/bin/env python3
"""Single compact VirtualNetwork demand benchmark; freeze after first PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_virtual_network as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fingerprint(value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def make_fixture(module, count: int):
    graph = module.nx.path_graph(count)
    value = module.VirtualNetwork(
        graph,
        config={
            "node_attrs_setting": [
                {"name": "cpu", "owner": "node", "type": "resource"},
                {"name": "peak", "owner": "node", "type": "resource"},
            ],
            "link_attrs_setting": [
                {"name": "bw", "owner": "link", "type": "resource"},
                {"name": "peak_bw", "owner": "link", "type": "resource"},
            ],
        },
    )
    module.nx.set_node_attributes(
        value, {index: (index * 17) % 1009 for index in range(count)}, "cpu"
    )
    module.nx.set_node_attributes(
        value,
        {index: float((index * 19) % 1013) + 0.25 for index in range(count)},
        "peak",
    )
    module.nx.set_edge_attributes(
        value,
        {(index, index + 1): (index * 23) % 1019 for index in range(count - 1)},
        "bw",
    )
    module.nx.set_edge_attributes(
        value,
        {
            (index, index + 1): float((index * 29) % 1021) + 0.125
            for index in range(count - 1)
        },
        "peak_bw",
    )
    return value


def output(value) -> str:
    return (
        "node=" + differential.double_token(value.total_node_resource_demand)
        + ";link=" + differential.double_token(value.total_link_resource_demand)
        + ";total=" + differential.double_token(value.total_resource_demand)
    )


def time_python(value, warmups: int, repetitions: int) -> tuple[int, str]:
    for _ in range(warmups):
        value.__dict__.pop("total_resource_demand", None)
        _ = value.total_resource_demand
    samples: list[int] = []
    for _ in range(repetitions):
        value.__dict__.pop("total_resource_demand", None)
        begin = time.perf_counter_ns()
        _ = value.total_resource_demand
        end = time.perf_counter_ns()
        samples.append(end - begin)
    return int(statistics.median(samples)), output(value)


def parse_cpp(
    path: pathlib.Path, count: int, workers: int
) -> dict[str, int | str]:
    process = subprocess.run(
        [str(path), str(count), str(workers)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(f"VirtualNetwork benchmark failed: {process.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1",
        "kind": "virtual_network_demand",
        "count": str(count),
        "workers": str(workers),
        "type_tag": "raw64_order_v1",
        "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(f"benchmark field {key}: {fields.get(key)!r} != {value!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
        "entry_count": int(fields["entry_count"]),
    }


def validate(
    current: dict[str, int | str], checksum: int, size: int, entries: int, label: str
) -> None:
    if (
        current["checksum"] != checksum
        or current["output_bytes"] != size
        or current["entry_count"] != entries
    ):
        raise RuntimeError(f"{label} output mismatch: {current!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=32768)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.count < 4 or args.warmups < 0 or args.repetitions <= 0:
        raise RuntimeError("invalid VirtualNetwork benchmark protocol")

    module = differential.load_oracle(args.source, args.base_source)
    try:
        value = make_fixture(module, args.count)
        python_ns, encoded = time_python(value, args.warmups, args.repetitions)
        checksum, size = fingerprint(encoded)
        entries = 2 * args.count + 2 * (args.count - 1)
        rows = []
        for workers in args.workers:
            for _ in range(args.warmups):
                validate(
                    parse_cpp(args.benchmark, args.count, workers),
                    checksum, size, entries, f"w{workers} warm-up",
                )
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(args.benchmark, args.count, workers)
                validate(current, checksum, size, entries, f"w{workers} sample")
                samples.append(int(current["elapsed_ns"]))
            cpp_ns = int(statistics.median(samples))
            speedup = python_ns / cpp_ns
            if speedup <= 1.0:
                raise RuntimeError(
                    f"VirtualNetwork w{workers} did not beat Python: {speedup:.3f}x"
                )
            row = {
                "kind": "virtual_resource_demand",
                "count": args.count,
                "workers": workers,
                "python_median_ms": python_ns / 1e6,
                "cpp_median_ms": cpp_ns / 1e6,
                "speedup": speedup,
                "checksum": checksum,
                "output_bytes": size,
                "entry_count": entries,
            }
            rows.append(row)
            print(
                f"virtual_resource_demand/w{workers}: "
                f"python={row['python_median_ms']:.6f} ms, "
                f"cpp={row['cpp_median_ms']:.6f} ms, speedup={speedup:.3f}x"
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
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_and_input_creation_excluded": True,
            "cpp_process_startup_excluded": True,
            "fingerprint_excluded": True,
            "raw64_output_gate_before_timing": True,
            "caller_configured_workers": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"VirtualNetwork benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
