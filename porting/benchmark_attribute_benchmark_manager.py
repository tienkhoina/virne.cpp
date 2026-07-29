#!/usr/bin/env python3
"""One compact exact benchmark for prepared attribute row maxima."""

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

import compare_attribute_benchmark_manager as differential


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv_byte(checksum: int, byte: int) -> int:
    return ((checksum ^ byte) * FNV_PRIME) & MASK64


def fingerprint(values) -> tuple[int, int]:
    checksum = FNV_OFFSET
    output_bytes = 0
    for name, value in values.items():
        encoded_name = name.encode("utf-8")
        for byte in encoded_name:
            checksum = fnv_byte(checksum, byte)
        checksum = fnv_byte(checksum, 0)
        for byte in struct.pack(">d", float(value)):
            checksum = fnv_byte(checksum, byte)
        output_bytes += len(encoded_name) + 1 + 8
    return checksum, output_bytes


def parse_cpp(
    path: pathlib.Path,
    rows: int,
    columns: int,
    workers: int,
):
    process = subprocess.run(
        [str(path), str(rows), str(columns), str(workers)],
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
        "protocol",
        "kind",
        "rows",
        "columns",
        "column_repetitions",
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
        or fields["kind"] != "prepared_row_maxima"
        or fields["column_repetitions"] != "2"
        or fields["type_tag"] != "ordered_utf8_raw64"
        or fields["status"] != "PASS"
        or int(fields["rows"]) != rows
        or int(fields["columns"]) != columns
        or int(fields["workers"]) != workers
    ):
        raise RuntimeError(f"invalid C++ benchmark protocol: {fields!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
        "entry_count": int(fields["entry_count"]),
    }


def python_fixture(module, rows: int, columns: int):
    np = module.np
    attrs = [
        differential.OracleAttribute("status", f"metric_{row}")
        for row in range(rows)
    ]
    row_indices = np.arange(rows, dtype=np.int64).reshape(rows, 1)
    column_indices = np.arange(columns, dtype=np.int64).reshape(1, columns)
    residues = (row_indices * 131 + column_indices * 17) % 2048
    matrix = ((residues - 1024) * 0.125).astype(np.float32)
    repeated = np.concatenate([matrix, matrix], axis=1)

    def run():
        return module.get_attr_benchmarks(["status"], attrs, repeated)

    return run


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
        raise RuntimeError("Python attribute benchmark produced no output")
    return statistics.median(samples), output


def validate_output(
    current,
    expected_checksum: int,
    expected_bytes: int,
    expected_entries: int,
    label: str,
) -> None:
    if (
        current["checksum"] != expected_checksum
        or current["output_bytes"] != expected_bytes
        or current["entry_count"] != expected_entries
    ):
        raise RuntimeError(f"{label} ordered output drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if (
        args.rows <= 0
        or args.columns <= 0
        or args.warmups < 0
        or args.repetitions <= 0
        or any(worker < 0 for worker in args.workers)
    ):
        raise RuntimeError("invalid benchmark protocol")

    module = differential.load_oracle(args.source)
    try:
        run = python_fixture(module, args.rows, args.columns)

        # Exact ordered output is established before any retained timing sample.
        reference = run()
        expected_checksum, expected_bytes = fingerprint(reference)
        expected_entries = len(reference)
        if expected_entries != args.rows:
            raise RuntimeError("Python benchmark key inventory drift")
        for index, name in enumerate(reference):
            if name != f"metric_{index}":
                raise RuntimeError("Python benchmark key order drift")
        for workers in args.workers:
            validate_output(
                parse_cpp(args.benchmark, args.rows, args.columns, workers),
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
            raise RuntimeError("Python timed output drift")

        rows = []
        for workers in args.workers:
            for _ in range(args.warmups):
                warm = parse_cpp(
                    args.benchmark, args.rows, args.columns, workers
                )
                validate_output(
                    warm,
                    expected_checksum,
                    expected_bytes,
                    expected_entries,
                    f"worker {workers} warm-up",
                )
            samples = []
            for _ in range(args.repetitions):
                current = parse_cpp(
                    args.benchmark, args.rows, args.columns, workers
                )
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
                "kind": "prepared_row_maxima",
                "rows": args.rows,
                "columns": args.columns,
                "column_repetitions": 2,
                "workers": workers,
                "type_tag": "ordered_utf8_raw64",
                "python_median_ms": python_ns / 1e6,
                "cpp_median_ms": cpp_ns / 1e6,
                "speedup": speedup,
                "checksum": expected_checksum,
                "output_bytes": expected_bytes,
                "entry_count": expected_entries,
            }
            rows.append(row)
            print(
                f"prepared_row_maxima/w{workers}: "
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
            "rows": args.rows,
            "columns": args.columns,
            "column_repetitions": 2,
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "fixture_creation_and_checksum_excluded": True,
            "process_startup_excluded": True,
            "ordered_output_gate_before_timing": True,
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
    print(f"attribute_benchmark_manager benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
