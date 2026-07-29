#!/usr/bin/env python3
"""Single compact PhysicalNetwork orchestration benchmark; freeze after PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import random
import statistics
import subprocess
import time

import compare_physical_network as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def add_unsigned(checksum: int, value: int, size: int) -> int:
    for byte in int(value).to_bytes(size, "little", signed=False):
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum


def make_config(count: int):
    config = differential.generated_config()
    config["topology"]["num_nodes"] = count
    config["node_attrs_setting"][0]["high"] = 1009
    config["link_attrs_setting"][0]["high"] = 1021
    return config


def fingerprint(value, module) -> tuple[int, int, int]:
    checksum = FNV_OFFSET
    checksum = add_unsigned(checksum, value.number_of_nodes(), 8)
    checksum = add_unsigned(checksum, value.number_of_edges(), 8)
    entries = 0
    for node in value.nodes:
        checksum = add_unsigned(checksum, int(value.nodes[node]["cpu"]) & MASK64, 8)
        entries += 1
    for edge in value.edges:
        checksum = add_unsigned(
            checksum, int(value.edges[edge]["bandwidth"]) & MASK64, 8
        )
        entries += 1
    checksum = add_unsigned(checksum, random.getrandbits(32), 4)
    checksum = add_unsigned(
        checksum,
        int(module.np.random.randint(0, 2**32, dtype=module.np.uint32)),
        4,
    )
    return checksum, 16 + entries * 8 + 8, entries


def time_python(module, config, warmups: int, repetitions: int):
    for _ in range(warmups):
        module.PhysicalNetwork.from_setting(config, seed=77)
    samples: list[int] = []
    value = None
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        value = module.PhysicalNetwork.from_setting(config, seed=77)
        end = time.perf_counter_ns()
        samples.append(end - begin)
    if value is None:
        raise RuntimeError("PhysicalNetwork benchmark has no sample")
    return int(statistics.median(samples)), fingerprint(value, module)


def parse_cpp(path: pathlib.Path, count: int, workers: int) -> dict[str, int]:
    process = subprocess.run(
        [str(path), str(count), str(workers)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"PhysicalNetwork benchmark failed: {process.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed PhysicalNetwork benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1", "kind": "physical_from_setting",
        "count": str(count), "workers": str(workers),
        "type_tag": "le64_order_v1", "status": "PASS",
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
    current: dict[str, int], expected: tuple[int, int, int], label: str
) -> None:
    checksum, output_bytes, entries = expected
    if (
        current["checksum"] != checksum
        or current["output_bytes"] != output_bytes
        or current["entry_count"] != entries
    ):
        raise RuntimeError(f"{label} output mismatch: {current!r} != {expected!r}")


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
        raise RuntimeError("invalid PhysicalNetwork benchmark protocol")

    module = differential.load_oracle(args.source, args.base_source)
    try:
        config = make_config(args.count)
        python_ns, expected = time_python(
            module, config, args.warmups, args.repetitions
        )
        rows = []
        for workers in args.workers:
            for _ in range(args.warmups):
                validate(
                    parse_cpp(args.benchmark, args.count, workers),
                    expected, f"w{workers} warm-up",
                )
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(args.benchmark, args.count, workers)
                validate(current, expected, f"w{workers} sample")
                samples.append(current["elapsed_ns"])
            cpp_ns = int(statistics.median(samples))
            speedup = python_ns / cpp_ns
            if speedup <= 1.0:
                raise RuntimeError(
                    f"PhysicalNetwork w{workers} did not beat Python: {speedup:.3f}x"
                )
            checksum, output_bytes, entries = expected
            row = {
                "kind": "physical_from_setting", "count": args.count,
                "workers": workers, "python_median_ms": python_ns / 1e6,
                "cpp_median_ms": cpp_ns / 1e6, "speedup": speedup,
                "checksum": checksum, "output_bytes": output_bytes,
                "entry_count": entries,
            }
            rows.append(row)
            print(
                f"physical_from_setting/w{workers}: "
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
        "numpy_version": numpy_version, "networkx_version": networkx_version,
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count, "workers": args.workers,
            "warmups": args.warmups, "repetitions": args.repetitions,
            "fixture_and_input_creation_excluded": True,
            "cpp_process_startup_excluded": True,
            "fingerprint_excluded": True,
            "binary_output_gate_before_acceptance": True,
            "caller_configured_workers": True,
        },
        "rows": rows, "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"PhysicalNetwork benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
