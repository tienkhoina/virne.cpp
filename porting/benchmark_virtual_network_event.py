#!/usr/bin/env python3
"""Single construct/sort timing for VirtualNetworkEvent; freeze after PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import struct
import subprocess
import time

import compare_virtual_network_event as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def make_specs(count: int):
    return [
        (
            index, 1 if index % 2 == 0 else 0, index // 2,
            float((index * 37) % 1009) + (0.25 if index % 3 == 0 else 0.0),
        )
        for index in range(count)
    ]


def fingerprint(events) -> tuple[int, int, int]:
    checksum = FNV_OFFSET
    for event in events:
        encoded = struct.pack(
            "<QBqd", event.id, event.type, event.v_net_id, float(event.time)
        )
        for byte in encoded:
            checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(events) * 25, len(events)


def materialize(Event, specs):
    events = [
        Event(id=item[0], type=item[1], v_net_id=item[2], time=item[3])
        for item in specs
    ]
    return sorted(events, key=lambda event: event.time)


def time_python(Event, specs, warmups: int, repetitions: int):
    for _ in range(warmups):
        materialize(Event, specs)
    samples: list[int] = []
    events = None
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        events = materialize(Event, specs)
        end = time.perf_counter_ns()
        samples.append(end - begin)
    if events is None:
        raise RuntimeError("event benchmark has no sample")
    return int(statistics.median(samples)), fingerprint(events)


def parse_cpp(path: pathlib.Path, count: int, workers: int) -> dict[str, int]:
    process = subprocess.run(
        [str(path), str(count), str(workers)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"VirtualNetworkEvent benchmark failed: {process.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed event benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1", "kind": "virtual_network_event_construct_sort",
        "count": str(count), "workers": str(workers),
        "type_tag": "event_le_v1", "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(f"event benchmark field {key}: {fields.get(key)!r} != {value!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
        "entry_count": int(fields["entry_count"]),
    }


def validate(current: dict[str, int], expected: tuple[int, int, int], label: str):
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
    parser.add_argument("--count", type=int, default=131072)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.count <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise RuntimeError("invalid VirtualNetworkEvent benchmark protocol")

    Event = differential.load_event_class(args.source)
    specs = make_specs(args.count)
    python_ns, expected = time_python(
        Event, specs, args.warmups, args.repetitions
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
                f"VirtualNetworkEvent w{workers} did not beat Python: {speedup:.3f}x"
            )
        checksum, output_bytes, entries = expected
        row = {
            "kind": "virtual_network_event_construct_sort",
            "count": args.count, "workers": workers,
            "python_median_ms": python_ns / 1e6,
            "cpp_median_ms": cpp_ns / 1e6, "speedup": speedup,
            "checksum": checksum, "output_bytes": output_bytes,
            "entry_count": entries,
        }
        rows.append(row)
        print(
            f"virtual_network_event_construct_sort/w{workers}: "
            f"python={row['python_median_ms']:.6f} ms, "
            f"cpp={row['cpp_median_ms']:.6f} ms, speedup={speedup:.3f}x"
        )

    payload = {
        "source_sha256": differential.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count, "workers": args.workers,
            "warmups": args.warmups, "repetitions": args.repetitions,
            "fixture_creation_excluded": True,
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
    print(f"VirtualNetworkEvent benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
