#!/usr/bin/env python3
"""One compact simulator timing; freeze the artifacts after acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

import compare_virtual_network_request_simulator as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def add_bytes(checksum: int, encoded: bytes) -> int:
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum


def arrangement_fingerprint(simulator, next_numpy: int) -> tuple[int, int, int]:
    checksum = FNV_OFFSET
    for value in simulator.v_nets_size:
        checksum = add_bytes(checksum, struct.pack("<q", int(value)))
    for values in (simulator.v_nets_lifetime, simulator.v_nets_arrival_time):
        for value in values:
            checksum = add_bytes(checksum, struct.pack("<d", float(value)))
    checksum = add_bytes(checksum, struct.pack("<I", next_numpy))
    count = len(simulator.v_nets_size)
    return checksum, count * 24 + 4, count


def event_fingerprint(events) -> tuple[int, int, int]:
    checksum = FNV_OFFSET
    for event in events:
        checksum = add_bytes(
            checksum,
            struct.pack(
                "<QBqd",
                int(event.id),
                int(event.type),
                int(event.v_net_id),
                float(event.time),
            ),
        )
    return checksum, len(events) * 25, len(events)


def make_schedule_networks(count: int):
    return [
        differential.StubVirtualNetwork(
            id=index,
            arrival_time=float((index * 37) % 4096)
            + (0.25 if index % 3 == 0 else 0.0),
            lifetime=float((index * 13) % 127) + 1.0,
        )
        for index in range(count)
    ]


def arrangement_sample(module, count: int) -> tuple[int, tuple[int, int, int]]:
    simulator = module.VirtualNetworkRequestSimulator.from_setting(
        differential.make_config("float", False, count)
    )
    np.random.seed(991)
    begin = time.perf_counter_ns()
    simulator.arrange_v_nets()
    end = time.perf_counter_ns()
    next_numpy = differential.next_numpy_word()
    return end - begin, arrangement_fingerprint(simulator, next_numpy)


def schedule_sample(
    module, networks
) -> tuple[int, tuple[int, int, int]]:
    simulator = module.VirtualNetworkRequestSimulator(
        v_nets=networks, events=[], v_sim_setting={}
    )
    begin = time.perf_counter_ns()
    simulator._renew_events()
    end = time.perf_counter_ns()
    return end - begin, event_fingerprint(simulator.events)


def time_python(module, count: int, warmups: int, repetitions: int):
    schedule_networks = make_schedule_networks(count)
    for _ in range(warmups):
        arrangement_sample(module, count)
        schedule_sample(module, schedule_networks)

    samples = {"arrange": [], "schedule": []}
    fingerprints: dict[str, tuple[int, int, int]] = {}
    for _ in range(repetitions):
        elapsed, current = arrangement_sample(module, count)
        samples["arrange"].append(elapsed)
        previous = fingerprints.setdefault("arrange", current)
        if current != previous:
            raise RuntimeError("Python arrangement fingerprint drift")

        elapsed, current = schedule_sample(module, schedule_networks)
        samples["schedule"].append(elapsed)
        previous = fingerprints.setdefault("schedule", current)
        if current != previous:
            raise RuntimeError("Python schedule fingerprint drift")

    return {
        kind: int(statistics.median(values)) for kind, values in samples.items()
    }, fingerprints


def parse_cpp(
    path: pathlib.Path, kind: str, count: int, workers: int
) -> dict[str, int]:
    process = subprocess.run(
        [str(path), kind, str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "VirtualNetworkRequestSimulator benchmark failed: "
            + process.stderr.strip()
        )
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed simulator benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1",
        "kind": f"simulator_{kind}",
        "count": str(count),
        "workers": str(workers),
        "type_tag": "simulator_le_v1",
        "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(
                f"simulator benchmark field {key}: "
                f"{fields.get(key)!r} != {value!r}"
            )
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
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=65536)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.count <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise RuntimeError("invalid simulator benchmark protocol")

    torch_before = "torch" in sys.modules
    module = differential.load_oracle(args.source)
    try:
        python_ns, expected = time_python(
            module, args.count, args.warmups, args.repetitions
        )
        if not torch_before and "torch" in sys.modules:
            raise RuntimeError("simulator benchmark unexpectedly imported Torch")
    finally:
        differential.unload_oracle()

    rows = []
    for kind in ("arrange", "schedule"):
        for workers in args.workers:
            for _ in range(args.warmups):
                validate(
                    parse_cpp(args.benchmark, kind, args.count, workers),
                    expected[kind],
                    f"{kind}/w{workers} warm-up",
                )
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(args.benchmark, kind, args.count, workers)
                validate(current, expected[kind], f"{kind}/w{workers} sample")
                samples.append(current["elapsed_ns"])
            cpp_ns = int(statistics.median(samples))
            speedup = python_ns[kind] / cpp_ns
            if speedup <= 1.0:
                raise RuntimeError(
                    f"simulator {kind}/w{workers} did not beat Python: "
                    f"{speedup:.3f}x"
                )
            checksum, output_bytes, entries = expected[kind]
            row = {
                "kind": f"simulator_{kind}",
                "count": args.count,
                "workers": workers,
                "python_median_ms": python_ns[kind] / 1e6,
                "cpp_median_ms": cpp_ns / 1e6,
                "speedup": speedup,
                "checksum": checksum,
                "output_bytes": output_bytes,
                "entry_count": entries,
            }
            rows.append(row)
            print(
                f"simulator_{kind}/w{workers}: "
                f"python={row['python_median_ms']:.6f} ms, "
                f"cpp={row['cpp_median_ms']:.6f} ms, "
                f"speedup={speedup:.3f}x"
            )

    payload = {
        "source_sha256": differential.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_creation_excluded": True,
            "cpp_process_startup_excluded": True,
            "fingerprint_excluded": True,
            "binary_output_gate_before_acceptance": True,
            "caller_configured_workers": True,
            "worker_auto_tuning": False,
        },
        "rows": rows,
        "result": "PASS",
        "torch_imported": False,
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"VirtualNetworkRequestSimulator benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
