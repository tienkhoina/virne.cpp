#!/usr/bin/env python3
"""Canonical sequential runtime comparison for ``virne.utils.manager``."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import compare_manager


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1

KINDS = (
    "temp_empty",
    "temp_legacy",
    "clean_retained",
    "clean_delete_missing",
    "clean_delete_empty",
    "empty_retained",
    "empty_remove",
)

DEFAULT_OPERATIONS = {
    "temp_empty": 20_000,
    "temp_legacy": 5_000,
    "clean_retained": 50,
    "clean_delete_missing": 64,
    "clean_delete_empty": 64,
    "empty_retained": 10_000,
    "empty_remove": 256,
}


@dataclass(frozen=True)
class Metrics:
    kind: str
    operations: int
    elapsed_ns: int
    semantic_checksum: int
    output_checksum: int
    output_bytes: int


@dataclass(frozen=True)
class Timing:
    operations: int
    python_median_ms: float
    python_mad_ms: float
    python_p95_ms: float
    cpp_median_ms: float
    cpp_mad_ms: float
    cpp_p95_ms: float
    python_ns_per_operation: float
    cpp_ns_per_operation: float
    speedup: float
    semantic_checksum: int
    output_checksum: int
    output_bytes: int


def fnv_update(hash_value: int, data: bytes) -> int:
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & MASK64
    return hash_value


def fnv_update_u64(hash_value: int, value: int) -> int:
    return fnv_update(hash_value, value.to_bytes(8, "little"))


def semantic_checksum(operations: int, marker: int) -> int:
    result = FNV_OFFSET
    for index in range(operations):
        result = fnv_update_u64(result, index)
        result = fnv_update_u64(result, marker)
    return result


@contextlib.contextmanager
def fresh_sandbox() -> Iterator[pathlib.Path]:
    root = pathlib.Path(tempfile.mkdtemp(prefix="virne_manager_bench_")).resolve()
    temporary = pathlib.Path(tempfile.gettempdir()).resolve()
    if root.parent != temporary or not root.name.startswith("virne_manager_bench_"):
        raise RuntimeError(f"unsafe manager benchmark sandbox: {root}")
    try:
        yield root
    finally:
        resolved = root.resolve()
        if resolved.parent != temporary or not resolved.name.startswith("virne_manager_bench_"):
            raise RuntimeError(f"refusing unsafe manager benchmark cleanup: {resolved}")
        shutil.rmtree(resolved)


def write(path: pathlib.Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def normalized_output(output: str, sandbox: pathlib.Path) -> bytes:
    return output.replace(str(sandbox), "<ROOT>").encode(
        "utf-8", errors="surrogateescape"
    )


def config(record: pathlib.Path, log: pathlib.Path, save: pathlib.Path) -> Any:
    return type(
        "Config",
        (),
        {
            "record_dir": str(record),
            "log_dir": str(log),
            "save_dir": str(save),
        },
    )()


def python_benchmark(
    module: Any,
    kind: str,
    operations: int,
) -> Metrics:
    with fresh_sandbox() as sandbox:
        output = io.StringIO()
        marker = 0
        if kind == "temp_empty":
            target = sandbox / "target"
            target.mkdir()
            started = time.perf_counter_ns()
            for _ in range(operations):
                module.delete_temp_files(str(target))
            elapsed = time.perf_counter_ns() - started
            if not target.is_dir() or any(target.iterdir()):
                raise RuntimeError("Python temp-empty invariant failed")
            marker = 1
        elif kind == "temp_legacy":
            target = sandbox / "target"
            write(target / "file_temp", b"payload")
            caught = 0
            started = time.perf_counter_ns()
            for _ in range(operations):
                try:
                    module.delete_temp_files(str(target))
                except TypeError:
                    caught += 1
            elapsed = time.perf_counter_ns() - started
            if caught != operations or not (target / "file_temp").exists():
                raise RuntimeError("Python temp-legacy invariant failed")
            marker = 2
        elif kind == "clean_retained":
            save = sandbox / "save"
            for algorithm in range(8):
                for run in range(16):
                    write(
                        save
                        / f"algorithm_{algorithm}"
                        / f"run_{run}"
                        / "records"
                        / "result"
                    )
            with contextlib.redirect_stdout(output):
                started = time.perf_counter_ns()
                for _ in range(operations):
                    module.clean_save_dir(str(save))
                elapsed = time.perf_counter_ns() - started
            if output.getvalue():
                raise RuntimeError("Python retained scan emitted output")
            marker = 128
        elif kind in ("clean_delete_missing", "clean_delete_empty"):
            saves: list[pathlib.Path] = []
            for index in range(operations):
                save = sandbox / f"case_{index}" / "save"
                run = save / "algorithm" / "run"
                if kind == "clean_delete_empty":
                    (run / "records").mkdir(parents=True)
                else:
                    write(run / "models" / "weights")
                saves.append(save)
            with contextlib.redirect_stdout(output):
                started = time.perf_counter_ns()
                for save in saves:
                    module.clean_save_dir(str(save))
                elapsed = time.perf_counter_ns() - started
            if any((save / "algorithm" / "run").exists() for save in saves):
                raise RuntimeError("Python destructive clean retained a run")
            marker = 4 if kind == "clean_delete_empty" else 3
        elif kind == "empty_retained":
            record = sandbox / "record"
            log = sandbox / "log"
            save = sandbox / "save"
            write(record / "keep")
            write(log / "keep")
            write(save / "keep")
            value = config(record, log, save)
            started = time.perf_counter_ns()
            for _ in range(operations):
                module.delete_empty_dir(value)
            elapsed = time.perf_counter_ns() - started
            if not all((path / "keep").exists() for path in (record, log, save)):
                raise RuntimeError("Python retained empty-dir fixture changed")
            marker = 3
        elif kind == "empty_remove":
            configs: list[Any] = []
            paths: list[tuple[pathlib.Path, pathlib.Path, pathlib.Path]] = []
            for index in range(operations):
                root = sandbox / f"case_{index}"
                record = root / "record"
                log = root / "log"
                save = root / "save"
                record.mkdir(parents=True)
                log.mkdir()
                save.mkdir()
                configs.append(config(record, log, save))
                paths.append((record, log, save))
            started = time.perf_counter_ns()
            for value in configs:
                module.delete_empty_dir(value)
            elapsed = time.perf_counter_ns() - started
            if any(path.exists() for group in paths for path in group):
                raise RuntimeError("Python empty-remove retained a directory")
            marker = 3
        else:
            raise AssertionError(kind)

        normalized = normalized_output(output.getvalue(), sandbox)
        return Metrics(
            kind=kind,
            operations=operations,
            elapsed_ns=elapsed,
            semantic_checksum=semantic_checksum(operations, marker),
            output_checksum=fnv_update(FNV_OFFSET, normalized),
            output_bytes=len(normalized),
        )


def parse_cpp_output(text: str) -> Metrics:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise RuntimeError(f"malformed manager benchmark line: {line!r}")
        values[key] = value
    required = {
        "benchmark_version",
        "kind",
        "operations",
        "elapsed_ns",
        "semantic_checksum",
        "output_checksum",
        "output_bytes",
        "status",
    }
    if set(values) != required or values["benchmark_version"] != "1" or values["status"] != "PASS":
        raise RuntimeError(f"invalid manager benchmark response: {values!r}")
    return Metrics(
        kind=values["kind"],
        operations=int(values["operations"]),
        elapsed_ns=int(values["elapsed_ns"]),
        semantic_checksum=int(values["semantic_checksum"]),
        output_checksum=int(values["output_checksum"]),
        output_bytes=int(values["output_bytes"]),
    )


def cpp_benchmark(
    harness: pathlib.Path,
    kind: str,
    operations: int,
) -> Metrics:
    with fresh_sandbox() as sandbox:
        process = subprocess.run(
            [str(harness), str(sandbox), kind, str(operations)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"C++ manager benchmark failed: {process.stderr!r}"
            )
        return parse_cpp_output(process.stdout)


def validate_pair(python: Metrics, cpp: Metrics) -> None:
    for field in (
        "kind",
        "operations",
        "semantic_checksum",
        "output_checksum",
        "output_bytes",
    ):
        if getattr(python, field) != getattr(cpp, field):
            raise RuntimeError(
                f"manager benchmark {field} mismatch: "
                f"Python={getattr(python, field)!r}, C++={getattr(cpp, field)!r}"
            )


def distribution(values: list[int]) -> tuple[float, float, float]:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    ordered = sorted(values)
    p95 = float(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)])
    return median, mad, p95


def benchmark_kind(
    module: Any,
    harness: pathlib.Path,
    kind: str,
    operations: int,
    warmups: int,
    repetitions: int,
) -> Timing:
    python_samples: list[int] = []
    cpp_samples: list[int] = []
    reference: Metrics | None = None
    for sample in range(warmups + repetitions):
        if sample % 2 == 0:
            python = python_benchmark(module, kind, operations)
            cpp = cpp_benchmark(harness, kind, operations)
        else:
            cpp = cpp_benchmark(harness, kind, operations)
            python = python_benchmark(module, kind, operations)
        validate_pair(python, cpp)
        if reference is None:
            reference = python
        if sample >= warmups:
            python_samples.append(python.elapsed_ns)
            cpp_samples.append(cpp.elapsed_ns)

    assert reference is not None
    python_median, python_mad, python_p95 = distribution(python_samples)
    cpp_median, cpp_mad, cpp_p95 = distribution(cpp_samples)
    return Timing(
        operations=operations,
        python_median_ms=python_median / 1_000_000,
        python_mad_ms=python_mad / 1_000_000,
        python_p95_ms=python_p95 / 1_000_000,
        cpp_median_ms=cpp_median / 1_000_000,
        cpp_mad_ms=cpp_mad / 1_000_000,
        cpp_p95_ms=cpp_p95 / 1_000_000,
        python_ns_per_operation=python_median / operations,
        cpp_ns_per_operation=cpp_median / operations,
        speedup=python_median / cpp_median,
        semantic_checksum=reference.semantic_checksum,
        output_checksum=reference.output_checksum,
        output_bytes=reference.output_bytes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--json-output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.scale <= 0:
        raise ValueError("invalid benchmark protocol")
    harness = args.harness.resolve()
    source = args.python_source.resolve()
    if not harness.is_file():
        raise FileNotFoundError(harness)
    module = compare_manager.load_oracle(source)

    timings: dict[str, Timing] = {}
    for kind in KINDS:
        operations = max(1, round(DEFAULT_OPERATIONS[kind] * args.scale))
        timing = benchmark_kind(
            module,
            harness,
            kind,
            operations,
            args.warmups,
            args.repetitions,
        )
        timings[kind] = timing
        print(
            f"{kind}: operations={operations}, "
            f"python={timing.python_median_ms:.6f} ms "
            f"(MAD {timing.python_mad_ms:.6f}, p95 {timing.python_p95_ms:.6f}), "
            f"cpp={timing.cpp_median_ms:.6f} ms "
            f"(MAD {timing.cpp_mad_ms:.6f}, p95 {timing.cpp_p95_ms:.6f}), "
            f"speedup={timing.speedup:.3f}x"
        )

    required_faster = ("temp_empty", "clean_retained", "empty_retained")
    slower = [kind for kind in required_faster if timings[kind].speedup <= 1.0]
    if slower:
        raise RuntimeError(f"C++ failed required performance rows: {slower}")

    report = {
        "component": "virne.utils.manager",
        "status": "pass",
        "source_sha256": compare_manager.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpus_visible": os.cpu_count(),
        },
        "protocol": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "paired_alternating_order": True,
            "process_startup_excluded": True,
            "fixture_setup_and_verification_excluded": True,
            "worker_policy": "sequential-only",
        },
        "rows": {kind: asdict(timing) for kind, timing in timings.items()},
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json_report={args.json_output}")
    print("manager benchmark: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
