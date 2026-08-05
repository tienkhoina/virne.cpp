#!/usr/bin/env python3
"""Exact differential and one focused benchmark for RunningMeanStd."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import time
from typing import Any

import numpy as np


FIXTURE_VERSION = 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def update_moments(
    mean: list[float],
    variance: list[float],
    count: float,
    batch_mean: list[float],
    batch_variance: list[float],
    batch_count: float,
) -> tuple[list[float], list[float], float]:
    if len(batch_mean) == 1 and len(mean) != 1:
        batch_mean = [batch_mean[0]] * len(mean)
    if len(batch_variance) == 1 and len(mean) != 1:
        batch_variance = [batch_variance[0]] * len(mean)
    if len(mean) != len(variance) or len(batch_mean) != len(mean):
        raise ValueError("moment lane length mismatch")
    if len(batch_variance) != len(mean):
        raise ValueError("moment lane length mismatch")
    total_count = count + batch_count
    new_mean: list[float] = []
    new_variance: list[float] = []
    for old_mean, old_variance, current_mean, current_variance in zip(
        mean, variance, batch_mean, batch_variance
    ):
        delta = current_mean - old_mean
        new_mean.append(old_mean + delta * batch_count / total_count)
        m_a = old_variance * count
        m_b = current_variance * batch_count
        m2 = (
            m_a
            + m_b
            + delta * delta * count * batch_count / total_count
        )
        new_variance.append(m2 / total_count)
    return new_mean, new_variance, total_count


class RunningMeanStdPy:
    def __init__(self, feature_count: int = 1, epsilon: float = 1e-4):
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        self.mean = [0.0] * feature_count
        self.variance = [1.0] * feature_count
        self.count = epsilon

    @property
    def feature_count(self) -> int:
        return len(self.mean)

    def update(self, samples: list[float], sample_count: int) -> None:
        features = self.feature_count
        if len(samples) != sample_count * features:
            raise ValueError("sample lane mismatch")
        if sample_count == 0:
            # The C++ boundary defines an empty row batch as zero moments and
            # count zero; normal training never sends an empty batch.
            batch_mean = [float("nan")] * features
            batch_variance = [float("nan")] * features
            batch_count = 0.0
        else:
            batch_mean = []
            batch_variance = []
            for feature in range(features):
                values = [
                    samples[row * features + feature]
                    for row in range(sample_count)
                ]
                current_mean = sum(values) / sample_count
                current_variance = sum(
                    (value - current_mean) ** 2 for value in values
                ) / sample_count
                batch_mean.append(current_mean)
                batch_variance.append(current_variance)
            batch_count = float(sample_count)
        self.mean, self.variance, self.count = update_moments(
            self.mean,
            self.variance,
            self.count,
            batch_mean,
            batch_variance,
            batch_count,
        )

    def update_from_moments(
        self,
        batch_mean: list[float],
        batch_variance: list[float],
        batch_count: float,
    ) -> None:
        self.mean, self.variance, self.count = update_moments(
            self.mean,
            self.variance,
            self.count,
            batch_mean,
            batch_variance,
            batch_count,
        )


def summary(stats: RunningMeanStdPy) -> dict[str, Any]:
    checksum = FNV_OFFSET

    def update_bytes(payload: bytes) -> None:
        nonlocal checksum
        for byte in payload:
            checksum ^= byte
            checksum = (checksum * FNV_PRIME) & MASK64

    update_bytes(struct.pack("<Q", stats.feature_count))
    for value in stats.mean:
        update_bytes(struct.pack("<d", value))
    for value in stats.variance:
        update_bytes(struct.pack("<d", value))
    update_bytes(struct.pack("<d", stats.count))
    return {
        "feature_count": stats.feature_count,
        "mean": stats.mean,
        "variance": stats.variance,
        "count": stats.count,
        "checksum": str(checksum),
    }


def summary_from_parts(
    feature_count: int,
    mean: list[float],
    variance: list[float],
    count: float,
) -> dict[str, Any]:
    stats = RunningMeanStdPy(feature_count, 0.0)
    stats.mean = list(mean)
    stats.variance = list(variance)
    stats.count = count
    return summary(stats)


def exact_cases() -> list[dict[str, Any]]:
    two_batch = RunningMeanStdPy(2, 0.0)
    two_batch.update([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4)
    two_batch.update([9.0, 10.0, 11.0, 12.0], 2)

    initial_default = RunningMeanStdPy(3)

    moments_broadcast = RunningMeanStdPy(3, 0.0)
    moments_broadcast.update_from_moments([2.0], [1.0], 4.0)
    moments_broadcast.update_from_moments(
        [4.0, 6.0, 8.0], [0.5, 1.0, 2.0], 2.0
    )

    free_mean, free_variance, free_count = update_moments(
        [0.0, 0.0], [1.0, 1.0], 2.0, [4.0], [0.0], 2.0
    )
    free_function = {
        "feature_count": 2,
        "mean": free_mean,
        "variance": free_variance,
        "count": free_count,
    }

    return [
        {"name": "two_batch", "output": summary(two_batch)},
        {"name": "initial_default", "output": summary(initial_default)},
        {
            "name": "moments_broadcast",
            "output": summary(moments_broadcast),
        },
        {
            "name": "free_function",
            "output": summary_from_parts(
                free_function["feature_count"],
                free_function["mean"],
                free_function["variance"],
                free_function["count"],
            ),
        },
    ]


def error_cases() -> dict[str, bool]:
    result = {
        "invalid_feature_count": False,
        "sample_length_mismatch": False,
        "moment_length_mismatch": False,
    }
    try:
        RunningMeanStdPy(0)
    except ValueError:
        result["invalid_feature_count"] = True
    try:
        stats = RunningMeanStdPy(2)
        stats.update([1.0, 2.0, 3.0], 2)
    except ValueError:
        result["sample_length_mismatch"] = True
    try:
        stats = RunningMeanStdPy(2)
        stats.update_from_moments([1.0, 2.0, 3.0], [1.0], 1.0)
    except ValueError:
        result["moment_length_mismatch"] = True
    return result


def benchmark_samples() -> tuple[list[float], int, int]:
    rows, columns = 4096, 8
    samples = [
        (((row * 37 + column * 13) % 257) - 128) * 0.03125
        for row in range(rows)
        for column in range(columns)
    ]
    return samples, rows, columns


def benchmark_python(
    threads: int, warmups: int, repetitions: int, iterations: int
) -> dict[str, Any]:
    samples, rows, columns = benchmark_samples()
    # Run the actual NumPy path from Virne's RunningMeanStd.update: input
    # conversion, population mean/variance, then vectorized moment merging.
    sample_array = np.asarray(samples, dtype=np.float64).reshape(rows, columns)
    mean = np.zeros(columns, dtype=np.float64)
    variance = np.ones(columns, dtype=np.float64)
    count = 1e-4

    def update_numpy() -> None:
        nonlocal mean, variance, count
        values = np.array(sample_array)
        batch_mean = np.mean(values, axis=0)
        batch_variance = np.var(values, axis=0)
        batch_count = values.shape[0] if len(values.shape) > 1 else 1
        delta = batch_mean - mean
        total_count = count + batch_count
        new_mean = mean + delta * batch_count / total_count
        m_a = variance * count
        m_b = batch_variance * batch_count
        m2 = (
            m_a
            + m_b
            + np.square(delta) * count * batch_count / total_count
        )
        mean = new_mean
        variance = m2 / total_count
        count = total_count

    for _ in range(warmups):
        for _ in range(iterations):
            update_numpy()
    samples_ns: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            update_numpy()
        samples_ns.append(time.perf_counter_ns() - started)
    return {
        "name": "running_mean_std_4096x8",
        "threads": threads,
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": summary_from_parts(
            columns,
            mean.tolist(),
            variance.tolist(),
            float(count),
        ),
    }


def compare(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and expected == actual
    ):
        return []
    if type(expected) is not type(actual):
        return [{"path": path, "python": expected, "cpp": actual, "reason": "type"}]
    if isinstance(expected, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                result.extend(compare(expected[key], actual[key], f"{path}.{key}"))
            if len(result) >= 32:
                return result[:32]
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path, "python_length": len(expected),
                     "cpp_length": len(actual), "reason": "length"}]
        result: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare(left, right, f"{path}[{index}]"))
            if len(result) >= 32:
                return result[:32]
        return result
    return [] if expected == actual else [
        {"path": path, "python": expected, "cpp": actual, "reason": "value"}
    ]


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    command = [str(args.native)]
    if args.benchmark:
        command.extend([
            "--benchmark", "--threads", str(args.threads),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            "--iterations", str(args.iterations),
        ])
    process = subprocess.run(command, check=False, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"native failed: {process.stderr.strip()}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines")
    return json.loads(lines[0])


def compare_benchmark(
    native: dict[str, Any], python: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native benchmark report is missing")
    for key in ("name", "threads", "warmups", "repetitions", "iterations"):
        if native_benchmark.get(key) != python[key]:
            raise RuntimeError(
                f"benchmark parameter mismatch for {key}: "
                f"native={native_benchmark.get(key)!r}, python={python[key]!r}"
            )
    mismatches = compare(
        python["output"], native_benchmark.get("output"), "benchmark.output"
    )
    if mismatches:
        raise RuntimeError(f"benchmark output differs: {mismatches[:4]}")
    iterations = int(python["iterations"])
    native_samples = [int(value) / iterations
                      for value in native_benchmark["samples_ns"]]
    python_samples = [int(value) / iterations
                      for value in python["samples_ns"]]
    native_median = statistics.median(native_samples)
    python_median = statistics.median(python_samples)
    return {
        "name": python["name"],
        "threads": python["threads"],
        "warmups": python["warmups"],
        "repetitions": python["repetitions"],
        "iterations": iterations,
        "native_samples_ns_per_call": native_samples,
        "python_samples_ns_per_call": python_samples,
        "native_median_ns_per_call": native_median,
        "python_median_ns_per_call": python_median,
        "python_over_native_speedup": (
            python_median / native_median if native_median > 0 else None
        ),
        "output_checksum": python["output"]["checksum"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")

    native = run_native(args)
    expected = {
        "component": "running_mean_std",
        "fixture_version": FIXTURE_VERSION,
        "cases": exact_cases(),
        "errors": error_cases(),
    }
    actual_exact = {key: value for key, value in native.items()
                    if key != "benchmark"}
    mismatches = compare(expected, actual_exact)
    benchmark_report: Any = "DEFERRED_UNTIL_EXACT"
    if args.benchmark:
        benchmark_report = compare_benchmark(
            native,
            benchmark_python(args.threads, args.warmups,
                             args.repetitions, args.iterations),
            args,
        )
    report = {
        "component": "running_mean_std",
        "fixture_version": FIXTURE_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "exact_cases": f"{len(expected['cases'])}/{len(expected['cases'])}"
        if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark": benchmark_report,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    compact = {
        "component": report["component"],
        "status": report["status"],
        "exact_cases": report["exact_cases"],
        "benchmark": report["benchmark"],
    }
    if mismatches:
        compact["mismatches"] = mismatches[:4]
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
