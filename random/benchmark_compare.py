#!/usr/bin/env python3
"""Compare the opt-in C++ Random benchmark with pinned Python oracles."""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


def parse_cpp(stdout: str) -> dict[str, tuple[int, float]]:
    result: dict[str, tuple[int, float]] = {}
    for line in stdout.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        fields = [field.strip() for field in line.strip("|").split("|")]
        if fields[0] == "API":
            continue
        result[fields[0]] = (int(fields[1]), float(fields[2]))
    return result


def cpp_once(executable: Path, name: str) -> tuple[int, float]:
    completed = subprocess.run(
        [str(executable), "--api", name],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = parse_cpp(completed.stdout)
    if set(parsed) != {name}:
        raise RuntimeError(f"unexpected C++ benchmark rows: {sorted(parsed)}")
    return parsed[name]


def time_once(factory: Callable[[], Callable[[], object]]) -> float:
    operation = factory()
    gc.collect()
    begin = time.perf_counter_ns()
    result = operation()
    elapsed = time.perf_counter_ns() - begin
    # Keep the eager result alive through the end timestamp.
    if result is None:
        raise RuntimeError("benchmark operation unexpectedly returned None")
    return float(elapsed)


def python_random_factory(count: int):
    def factory():
        rng = random.Random(42)
        draw = rng.random

        def run():
            total = 0.0
            for _ in range(count):
                total += draw()
            return total

        return run

    return factory


def python_randint_factory(count: int):
    def factory():
        rng = random.Random(42)
        draw = rng.randint

        def run():
            total = 0
            for _ in range(count):
                total += draw(-1000, 1000)
            return total

        return run

    return factory


def numpy_factory(operation: str, count: int):
    def factory():
        rng = np.random.RandomState(42)
        if operation == "random":
            return lambda: rng.random_sample(count)
        if operation == "randint":
            return lambda: rng.randint(-1000, 1001, count, dtype=np.int64)
        if operation == "normal":
            return lambda: rng.normal(0.0, 1.0, count)
        if operation == "exponential":
            return lambda: rng.exponential(1.0, count)
        if operation == "poisson":
            return lambda: rng.poisson(20.0, count)
        raise AssertionError(operation)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    default_cpp = (
        Path(__file__).resolve().parent.parent
        / "build"
        / "random"
        / "benchmark_random"
    )
    parser.add_argument("--cpp", type=Path, default=default_cpp)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--api",
        action="append",
        help="benchmark only this exact API row (repeatable)",
    )
    args = parser.parse_args()

    if np.__version__ != "1.26.4":
        raise RuntimeError(
            f"benchmark baseline requires NumPy 1.26.4, found {np.__version__}"
        )
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "benchmark baseline requires CPython 3.10, found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    if not args.cpp.is_file():
        raise FileNotFoundError(f"build benchmark_random first: {args.cpp}")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")

    factories = {
        "PyRandom.random": lambda count: python_random_factory(count),
        "PyRandom.randint(-1000,1000)": lambda count: python_randint_factory(count),
        "NumpyRandomState.random": lambda count: numpy_factory("random", count),
        "NumpyRandomState.randint(-1000,1001)": lambda count: numpy_factory("randint", count),
        "NumpyRandomState.normal(0,1)": lambda count: numpy_factory("normal", count),
        "NumpyRandomState.exponential(1)": lambda count: numpy_factory("exponential", count),
        "NumpyRandomState.poisson(20)": lambda count: numpy_factory("poisson", count),
    }
    selected_names = args.api or list(factories)
    unknown = sorted(set(selected_names) - set(factories))
    if unknown:
        raise ValueError(f"unknown benchmark API(s): {unknown}")

    print("| API | samples | C++ median ns/value | CPython/NumPy median ns/value | C++ speedup |")
    print("|---|---:|---:|---:|---:|")
    failures: list[str] = []
    for name in selected_names:
        # One untimed call warms code/data paths. Each measured repetition is
        # paired by API, and order alternates to remove the old C++-first bias.
        warm_count, _ = cpp_once(args.cpp, name)
        time_once(factories[name](warm_count))
        cpp_samples: list[float] = []
        baseline_samples: list[float] = []
        count = warm_count
        for repetition in range(args.repetitions):
            if repetition % 2 == 0:
                measured_count, cpp_ns = cpp_once(args.cpp, name)
                baseline_total_ns = time_once(factories[name](count))
            else:
                baseline_total_ns = time_once(factories[name](count))
                measured_count, cpp_ns = cpp_once(args.cpp, name)
            if measured_count != count:
                raise RuntimeError(f"sample count changed for {name}")
            cpp_samples.append(cpp_ns)
            baseline_samples.append(baseline_total_ns / count)

        cpp_ns = statistics.median(cpp_samples)
        baseline_ns = statistics.median(baseline_samples)
        speedup = baseline_ns / cpp_ns
        print(
            f"| {name} | {count} | {cpp_ns:.2f} | "
            f"{baseline_ns:.2f} | {speedup:.2f}x |"
        )
        if speedup <= 1.0:
            failures.append(
                f"{name}: C++ {cpp_ns:.2f} ns/value, "
                f"baseline {baseline_ns:.2f} ns/value"
            )

    print(
        "\nCPython rows use scalar loops; NumPy rows use one vectorized "
        "RandomState call, matching the corresponding C++ vector overload. "
        "Each cell is the median of paired, alternating-order repetitions "
        "after one warm-up; vector benchmark calls are timed once per repetition."
    )
    if failures:
        raise RuntimeError(
            "strict Random benchmark gate requires C++ < CPython/NumPy:\n"
            + "\n".join(failures)
        )


if __name__ == "__main__":
    main()
