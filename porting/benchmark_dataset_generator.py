#!/usr/bin/env python3
"""Single compact Generator timing; freeze after the first accepted PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import numpy as np

import compare_dataset_generator as differential
import compare_virtual_network_request_simulator as simulator_diff


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fingerprint(payload: str, count: int) -> tuple[int, int, int]:
    checksum = FNV_OFFSET
    encoded = payload.encode("utf-8")
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded), count


def python_sample(module, kind: str, config, count: int):
    simulator_diff.set_seed(0)
    if kind == "ordinary":
        begin = time.perf_counter_ns()
        generated = module.Generator.generate_dataset(
            config, p_net=True, v_nets=True, save=False
        )
        end = time.perf_counter_ns()
    elif kind == "changeable":
        begin = time.perf_counter_ns()
        simulator = module.Generator.generate_changeable_v_nets_dataset_from_config(
            config, save=False
        )
        end = time.perf_counter_ns()
        generated = (None, simulator)
    else:
        raise RuntimeError(f"unknown Generator benchmark kind: {kind}")
    payload = differential.generated_payload(generated)
    return end - begin, fingerprint(payload, count)


def time_python(module, count: int, warmups: int, repetitions: int):
    config = differential.make_config(67, count)
    for _ in range(warmups):
        python_sample(module, "ordinary", config, count)
        python_sample(module, "changeable", config, count)
    samples = {"ordinary": [], "changeable": []}
    expected: dict[str, tuple[int, int, int]] = {}
    for _ in range(repetitions):
        for kind in ("ordinary", "changeable"):
            elapsed, current = python_sample(module, kind, config, count)
            samples[kind].append(elapsed)
            previous = expected.setdefault(kind, current)
            if current != previous:
                raise RuntimeError(f"Python Generator {kind} fingerprint drift")
    medians = {
        kind: int(statistics.median(values)) for kind, values in samples.items()
    }
    return medians, expected


def parse_cpp(
    path: pathlib.Path, kind: str, count: int, workers: int
) -> dict[str, int]:
    process = subprocess.run(
        [str(path), "benchmark", kind, str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Generator benchmark failed: {process.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed Generator benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1",
        "kind": f"dataset_generator_{kind}",
        "count": str(count),
        "workers": str(workers),
        "type_tag": "generator_text_v1",
        "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(
                f"Generator benchmark field {key}: "
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
    parser.add_argument("--simulator-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if (
        args.count <= 0
        or args.count % 4 != 0
        or args.warmups < 0
        or args.repetitions <= 0
    ):
        raise RuntimeError("invalid Generator benchmark protocol")

    module = differential.load_generator_oracle(
        args.source, args.simulator_source
    )
    try:
        python_ns, expected = time_python(
            module, args.count, args.warmups, args.repetitions
        )
        if differential._TORCH_SEED_CALLS != 0:
            raise RuntimeError("Generator benchmark called torch.seed")
    finally:
        differential.unload_generator_oracle()

    rows = []
    for kind in ("ordinary", "changeable"):
        for workers in args.workers:
            for _ in range(args.warmups):
                validate(
                    parse_cpp(args.benchmark, kind, args.count, workers),
                    expected[kind],
                    f"{kind}/w{workers} warm-up",
                )
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(
                    args.benchmark, kind, args.count, workers
                )
                validate(current, expected[kind], f"{kind}/w{workers} sample")
                samples.append(current["elapsed_ns"])
            cpp_ns = int(statistics.median(samples))
            speedup = python_ns[kind] / cpp_ns
            if speedup <= 1.0:
                raise RuntimeError(
                    f"Generator {kind}/w{workers} did not beat Python: "
                    f"{speedup:.3f}x"
                )
            checksum, output_bytes, entries = expected[kind]
            row = {
                "kind": f"dataset_generator_{kind}",
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
                f"dataset_generator_{kind}/w{workers}: "
                f"python={row['python_median_ms']:.6f} ms, "
                f"cpp={row['cpp_median_ms']:.6f} ms, "
                f"speedup={speedup:.3f}x"
            )

    payload = {
        "source_sha256": differential.SOURCE_SHA256,
        "simulator_source_sha256": simulator_diff.SOURCE_SHA256,
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
            "worker_fields": [
                "virtual_simulation.arrangement_workers",
                "virtual_simulation.event_workers",
            ],
            "per_request_factory_workers": 1,
            "per_request_attribute_workers": 1,
        },
        "rows": rows,
        "result": "PASS",
        "torch_seed_calls": 0,
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"dataset Generator benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
