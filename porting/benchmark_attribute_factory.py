#!/usr/bin/env python3
"""One compact, caller-configured benchmark for typed AttributeFactory batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_attribute_factory as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fingerprint(values) -> tuple[int, int]:
    encoded = differential.canonical_collection(values).encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def python_fixture(module, count: int):
    settings = []
    for index in range(count):
        setting = {"name": f"metric_{index}"}
        branch = index % 8
        if branch == 0:
            setting.update(owner="node", type="status")
        elif branch == 1:
            setting.update(owner="link", type="status")
        elif branch == 2:
            setting.update(
                owner="node", type="extrema", originator=f"origin_{index}"
            )
        elif branch == 3:
            setting.update(
                owner="link", type="extrema", originator=f"origin_{index}"
            )
        elif branch == 4:
            setting.update(owner="node", type="resource")
        elif branch == 5:
            setting.update(owner="link", type="resource")
        elif branch == 6:
            setting.update(owner="node", type="position")
        else:
            setting.update(owner="link", type="latency")

        settings.append(setting)

    def run():
        return module.create_attrs_from_setting(settings)

    return run


def parse_cpp(
    path: pathlib.Path, count: int, workers: int
) -> dict[str, int]:
    process = subprocess.run(
        [str(path), str(count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ factory benchmark failed: {process.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed C++ benchmark output: {line!r}")
        fields[key] = value
    required = {
        "protocol",
        "kind",
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
        or fields["kind"] != "typed_attribute_factory"
        or fields["type_tag"] != "ordered_factory_v1"
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
        raise RuntimeError(f"{label} ordered factory output drift")


def time_python(run, warmups: int, repetitions: int):
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
        raise RuntimeError("Python factory benchmark produced no output")
    return statistics.median(samples), output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=32768)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if (
        args.count <= 0
        or args.warmups < 0
        or args.repetitions <= 0
        or not args.workers
        or any(worker < 0 for worker in args.workers)
    ):
        raise RuntimeError("invalid benchmark protocol")

    module = differential.load_oracle(args.source)
    try:
        run = python_fixture(module, args.count)

        # The full ordered/direct-field gate is established before timing.
        reference = run()
        expected_checksum, expected_bytes = fingerprint(reference)
        expected_entries = len(reference)
        if expected_entries != args.count:
            raise RuntimeError("Python benchmark entry inventory drift")
        for index, name in enumerate(reference):
            if name != f"metric_{index}":
                raise RuntimeError("Python benchmark insertion order drift")
        for workers in args.workers:
            validate_output(
                parse_cpp(args.benchmark, args.count, workers),
                expected_checksum,
                expected_bytes,
                expected_entries,
                f"worker {workers} pre-timing gate",
            )

        python_ns, python_output = time_python(
            run, args.warmups, args.repetitions
        )
        final_checksum, final_bytes = fingerprint(python_output)
        if (
            final_checksum != expected_checksum
            or final_bytes != expected_bytes
            or len(python_output) != expected_entries
        ):
            raise RuntimeError("Python timed factory output drift")

        rows = []
        for workers in args.workers:
            for _ in range(args.warmups):
                warm = parse_cpp(args.benchmark, args.count, workers)
                validate_output(
                    warm,
                    expected_checksum,
                    expected_bytes,
                    expected_entries,
                    f"worker {workers} warm-up",
                )
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(args.benchmark, args.count, workers)
                validate_output(
                    current,
                    expected_checksum,
                    expected_bytes,
                    expected_entries,
                    f"worker {workers} sample",
                )
                samples.append(current["elapsed_ns"])
            cpp_ns = statistics.median(samples)
            speedup = python_ns / cpp_ns
            if speedup <= 1.0:
                raise RuntimeError(
                    f"worker {workers} C++ did not beat Python: {speedup:.3f}x"
                )
            row = {
                "kind": "typed_attribute_factory",
                "count": args.count,
                "workers": workers,
                "type_tag": "ordered_factory_v1",
                "python_median_ms": python_ns / 1e6,
                "cpp_median_ms": cpp_ns / 1e6,
                "speedup": speedup,
                "checksum": expected_checksum,
                "output_bytes": expected_bytes,
                "entry_count": expected_entries,
            }
            rows.append(row)
            print(
                f"typed_attribute_factory/w{workers}: "
                f"python={row['python_median_ms']:.6f} ms, "
                f"cpp={row['cpp_median_ms']:.6f} ms, speedup={speedup:.3f}x"
            )
        numpy_version = module.np.__version__
    finally:
        differential.unload_oracle()

    payload = {
        "source_sha256": differential.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "numpy_version": numpy_version,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "count": args.count,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_creation_and_fingerprint_excluded": True,
            "cpp_process_startup_excluded": True,
            "ordered_direct_field_gate_before_timing": True,
            "single_workload": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"attribute_factory benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
