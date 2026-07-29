#!/usr/bin/env python3
"""One exact-byte Logger benchmark; do not broaden after acceptance.

The Python side executes the exact ``Logger`` class AST from the frozen source.
It uses the real stdlib logging/file backend and replaces only colorlog's
formatter constructor; empty messages and a non-triggering display interval
make that formatter observationally irrelevant. WandB, TensorBoard, Torch, RL,
solver execution, and system orchestration are never imported.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import logging
import os
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time
import types
from typing import Any, Dict, List, Optional, Union


SOURCE_SHA256 = "274189af7211aba134c4696b2da8190f743b2f59d4287cf56843510f0b2c7083"
METRIC_COUNT = 8
ENTRY_COUNT = 4096
CSV_ROWS = ENTRY_COUNT + 1
WORKERS = (1, 2, 8)
WARMUPS = 1
REPETITIONS = 3
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
METRIC_NAMES = tuple(f"metric_{index}" for index in range(METRIC_COUNT))


class _NoopColoredFormatter(logging.Formatter):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__("%(message)s")


def load_logger(source_path: pathlib.Path):
    source_bytes = source_path.read_bytes()
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(
            f"logger.py hash drift: {actual_hash} != {SOURCE_SHA256}")
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source, filename=str(source_path))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Logger"
    ]
    if len(classes) != 1:
        raise RuntimeError("expected exactly one Logger class in logger.py")

    isolated = ast.Module(body=classes, type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {
        "__name__": "_virne_logger_benchmark_oracle",
        "os": os,
        "logging": logging,
        "colorlog": types.SimpleNamespace(
            ColoredFormatter=_NoopColoredFormatter),
        "csv": csv,
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Union": Union,
        "DictConfig": type("DictConfig", (), {}),
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["Logger"]


def make_config(root: pathlib.Path, run_id: str):
    return types.SimpleNamespace(
        logger=types.SimpleNamespace(
            backends=["file"],
            level="CRITICAL",
            log_file_name="run.log",
            project_name="",
            experiment_name="",
            log_dir_name="logs",
            log_show_interval=ENTRY_COUNT + 1,
        ),
        experiment=types.SimpleNamespace(
            save_root_dir=str(root),
            run_id=run_id,
        ),
        solver=types.SimpleNamespace(solver_name="logger-benchmark"),
    )


def metric_value(entry: int, metric_id: int) -> float:
    return float((entry * (metric_id + 3) + metric_id * 17) % 1009)


def make_fixture() -> list[dict[str, float]]:
    return [
        {
            name: metric_value(entry, metric_id)
            for metric_id, name in enumerate(METRIC_NAMES)
        }
        for entry in range(ENTRY_COUNT)
    ]


def fingerprint(value: bytes) -> int:
    checksum = FNV_OFFSET
    for byte in value:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum


def validate_csv(value: bytes, label: str) -> None:
    header = ("update_time," + ",".join(METRIC_NAMES) + "\r\n").encode()
    if not value.startswith(header):
        raise RuntimeError(f"{label} CSV header drifted")
    if value.count(b"\n") != CSV_ROWS or value.count(b"\r\n") != CSV_ROWS:
        raise RuntimeError(f"{label} CSV row framing drifted")
    if not value.endswith(b"\r\n"):
        raise RuntimeError(f"{label} CSV is not CRLF terminated")


def run_python_once(Logger, fixture, label: str) -> tuple[int, bytes]:
    with tempfile.TemporaryDirectory(prefix="virne_logger_python_") as root:
        root_path = pathlib.Path(root)
        logger = Logger(make_config(root_path, label))
        started = time.perf_counter_ns()
        for step, data in enumerate(fixture):
            logger.log(message="", level="INFO", data=data, step=step)
        stopped = time.perf_counter_ns()
        logger.close()

        log_dir = root_path / "logger-benchmark" / label / "logs"
        csv_bytes = (log_dir / "training_info.csv").read_bytes()
        validate_csv(csv_bytes, f"Python {label}")
        if (log_dir / "run.log").read_bytes():
            raise RuntimeError(f"Python {label} empty message changed run.log")
        del logger
        return stopped - started, csv_bytes


def time_python(Logger, fixture) -> tuple[int, bytes]:
    warm_ns, expected = run_python_once(Logger, fixture, "warmup")
    if warm_ns <= 0:
        raise RuntimeError("Python Logger warm-up timer did not advance")
    samples: list[int] = []
    for repetition in range(REPETITIONS):
        elapsed, actual = run_python_once(
            Logger, fixture, f"sample-{repetition}")
        if actual != expected:
            raise RuntimeError(
                f"Python Logger CSV changed at repetition {repetition}")
        samples.append(elapsed)
    return int(statistics.median(samples)), expected


def parse_fields(output: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise RuntimeError(f"malformed Logger benchmark line: {line!r}")
        fields[key] = value
    return fields


def first_difference(left: bytes, right: bytes) -> int:
    for index, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return index
    return min(len(left), len(right))


def run_cpp_once(
    executable: pathlib.Path,
    workers: int,
    expected_csv: bytes,
    label: str,
) -> int:
    with tempfile.TemporaryDirectory(prefix="virne_logger_cpp_") as root:
        root_path = pathlib.Path(root)
        process = subprocess.run(
            [str(executable), str(root_path), str(workers)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Logger benchmark {label} failed: {process.stderr.strip()}")
        fields = parse_fields(process.stdout)
        expected_fields = {
            "protocol": "1",
            "kind": "logger_dense_metric_batch_csv",
            "semantics": "exact_python_ordered_csv_v1",
            "backend": "file",
            "message": "empty",
            "entry_count": str(ENTRY_COUNT),
            "metric_count": str(METRIC_COUNT),
            "workers": str(workers),
            "status": "PASS",
        }
        numeric_fields = {
            "elapsed_ns", "checksum", "output_bytes", "csv_rows", "log_bytes"
        }
        if set(fields) != set(expected_fields) | numeric_fields:
            raise RuntimeError(
                f"Logger benchmark field inventory drift: {sorted(fields)}")
        for key, expected in expected_fields.items():
            if fields[key] != expected:
                raise RuntimeError(
                    f"Logger benchmark {key}: {fields[key]!r} != {expected!r}")
        try:
            numeric = {key: int(fields[key]) for key in numeric_fields}
        except ValueError as error:
            raise RuntimeError("Logger benchmark numeric field is invalid") from error
        if numeric["elapsed_ns"] <= 0:
            raise RuntimeError("Logger benchmark timer did not advance")

        log_dir = (
            root_path / "logger-benchmark" / f"workers-{workers}" / "logs"
        )
        actual_csv = (log_dir / "training_info.csv").read_bytes()
        validate_csv(actual_csv, f"C++ {label}")
        if actual_csv != expected_csv:
            offset = first_difference(actual_csv, expected_csv)
            raise RuntimeError(
                f"C++ {label} CSV byte mismatch at offset {offset}; "
                f"native={len(actual_csv)} bytes oracle={len(expected_csv)} bytes")
        actual_log = (log_dir / "run.log").read_bytes()
        gates = {
            "checksum": fingerprint(expected_csv),
            "output_bytes": len(expected_csv),
            "csv_rows": CSV_ROWS,
            "log_bytes": 0,
        }
        mismatches = {
            key: {"expected": expected, "actual": numeric[key]}
            for key, expected in gates.items()
            if numeric[key] != expected
        }
        if actual_log or mismatches:
            raise RuntimeError(
                f"C++ {label} final file gate failed: {mismatches}")
        return numeric["elapsed_ns"]


def time_cpp(
    executable: pathlib.Path,
    workers: int,
    expected_csv: bytes,
) -> int:
    run_cpp_once(executable, workers, expected_csv, f"workers={workers} warm-up")
    samples = [
        run_cpp_once(
            executable,
            workers,
            expected_csv,
            f"workers={workers} repetition={repetition}",
        )
        for repetition in range(REPETITIONS)
    ]
    return int(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Logger = load_logger(args.source)
    fixture = make_fixture()
    python_ns, expected_csv = time_python(Logger, fixture)
    expected_checksum = fingerprint(expected_csv)

    native_rows = []
    for workers in WORKERS:
        cpp_ns = time_cpp(args.benchmark, workers, expected_csv)
        speedup = python_ns / cpp_ns
        if speedup <= 1.0:
            raise RuntimeError(
                f"Logger C++ workers={workers} did not beat Python: "
                f"{speedup:.3f}x")
        native_rows.append({
            "workers": workers,
            "cpp_median_ns": cpp_ns,
            "speedup_vs_python": speedup,
        })

    report = {
        "component": "core.Logger",
        "case": "dense_metric_batch_file_csv",
        "source_sha256": SOURCE_SHA256.upper(),
        "native_executable_sha256": hashlib.sha256(
            args.benchmark.read_bytes()).hexdigest().upper(),
        "oracle": {
            "loader": "exact single-class AST isolation",
            "backend": "real stdlib logging FileHandler plus csv.writer",
            "fake_boundary": (
                "colorlog.ColoredFormatter constructor only; empty messages "
                "and disabled progress output make it observationally cold"
            ),
            "excluded": ["wandb", "tensorboard", "torch", "rl", "solver", "system"],
        },
        "protocol": {
            "metric_count": METRIC_COUNT,
            "entry_count": ENTRY_COUNT,
            "csv_rows": CSV_ROWS,
            "workers": list(WORKERS),
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "construction_excluded": True,
            "schema_registration_excluded": True,
            "fixture_construction_excluded": True,
            "process_startup_excluded": True,
            "file_read_excluded": True,
            "checksum_excluded": True,
            "cleanup_excluded": True,
            "native_hot_loop_uses_dense_metric_ids": True,
            "result_gate": "exact CSV bytes plus rows/size/FNV64 and empty run.log",
        },
        "fingerprint": {
            "sha256": hashlib.sha256(expected_csv).hexdigest().upper(),
            "fnv64": expected_checksum,
            "output_bytes": len(expected_csv),
        },
        "python_median_ns": python_ns,
        "cpp": native_rows,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
