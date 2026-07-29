#!/usr/bin/env python3
"""Runtime/checksum benchmark for independent Python and C++ stats wrappers."""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import math
import statistics
import subprocess
import sys
import threading
import time as standard_time
from dataclasses import dataclass
from typing import Any

import compare_stats


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
OUTPUT_LINE = b"Running time of bench: 1.0000s\n"


def fnv_update(hash_value: int, data: bytes) -> int:
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & MASK64
    return hash_value


def fnv_update_u64(hash_value: int, value: int) -> int:
    return fnv_update(hash_value, value.to_bytes(8, "little"))


def payload_mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def worker_input(worker: int, iteration: int) -> int:
    return ((worker << 32) ^ iteration) & MASK64


class HashSink:
    encoding = "utf-8"

    def __init__(self) -> None:
        self.hash_value = FNV_OFFSET
        self.byte_count = 0

    def write(self, text: str) -> int:
        encoded = text.encode(self.encoding)
        self.hash_value = fnv_update(self.hash_value, encoded)
        self.byte_count += len(encoded)
        return len(text)

    def flush(self) -> None:
        return None


class ThreadRoutingSink:
    encoding = "utf-8"

    def __init__(self) -> None:
        self.local = threading.local()

    def bind(self, sink: HashSink) -> None:
        self.local.sink = sink

    def write(self, text: str) -> int:
        sink = getattr(self.local, "sink", None)
        if sink is None:
            raise RuntimeError("benchmark thread has no bound output sink")
        return sink.write(text)

    def flush(self) -> None:
        sink = getattr(self.local, "sink", None)
        if sink is not None:
            sink.flush()


class ThreadLocalAlternatingClock:
    def __init__(self) -> None:
        self.local = threading.local()

    def __call__(self) -> float:
        calls = getattr(self.local, "calls", 0)
        self.local.calls = calls + 1
        return 0.0 if calls % 2 == 0 else 1.0

    def current_calls(self) -> int:
        return getattr(self.local, "calls", 0)


@dataclass
class WorkerResult:
    return_hash: int = FNV_OFFSET
    output_hash: int = FNV_OFFSET
    output_bytes: int = 0
    calls: int = 0
    clock_calls: int = 0


@dataclass
class RunResult:
    elapsed_ns: int
    workers: list[WorkerResult]


@dataclass
class BenchmarkMetrics:
    workers: int
    iterations_per_worker: int
    baseline_elapsed_ns: int
    wrapped_elapsed_ns: int
    baseline_return_checksum: str
    return_checksum: str
    output_checksum: str
    expected_output_checksum: str
    output_bytes: int
    expected_output_bytes: int
    calls: int
    clock_calls: int
    status: str


def run_python_workers(
    module: Any,
    worker_count: int,
    iterations: int,
    *,
    wrapped: bool,
) -> RunResult:
    results = [WorkerResult() for _ in range(worker_count)]
    errors: list[BaseException | None] = [None] * worker_count
    ready_condition = threading.Condition()
    ready_workers = 0
    finished_workers = 0
    start_event = threading.Event()
    router = ThreadRoutingSink()
    clock = ThreadLocalAlternatingClock()

    def worker_main(worker: int) -> None:
        nonlocal ready_workers, finished_workers
        result = WorkerResult()
        timed = None
        sink = None
        calls = 0
        try:
            if wrapped:
                sink = HashSink()
                router.bind(sink)

                def bench(input_value: int) -> int:
                    nonlocal calls
                    calls += 1
                    return payload_mix(input_value)

                timed = module.test_running_time(bench)
        except BaseException as error:
            errors[worker] = error
        finally:
            with ready_condition:
                ready_workers += 1
                ready_condition.notify_all()

        start_event.wait()
        try:
            if errors[worker] is None:
                for iteration in range(iterations):
                    input_value = worker_input(worker, iteration)
                    if wrapped:
                        if timed is None:
                            raise AssertionError("timed callable was not initialized")
                        value = timed(input_value)
                    else:
                        value = payload_mix(input_value)
                        calls += 1
                    result.return_hash = fnv_update_u64(result.return_hash, value)

                result.calls = calls
                if wrapped:
                    if sink is None:
                        raise AssertionError("hash sink was not initialized")
                    result.output_hash = sink.hash_value
                    result.output_bytes = sink.byte_count
                    result.clock_calls = clock.current_calls()
                results[worker] = result
        except BaseException as error:  # surfaced in the controlling thread
            errors[worker] = error
        finally:
            with ready_condition:
                finished_workers += 1
                ready_condition.notify_all()

    threads = [
        threading.Thread(target=worker_main, args=(worker,))
        for worker in range(worker_count)
    ]

    original_time = standard_time.time
    output_context = contextlib.redirect_stdout(router) if wrapped else contextlib.nullcontext()
    try:
        if wrapped:
            standard_time.time = clock
        with output_context:
            for thread in threads:
                thread.start()
            with ready_condition:
                ready_condition.wait_for(lambda: ready_workers == worker_count)
            started = standard_time.perf_counter_ns()
            start_event.set()
            with ready_condition:
                ready_condition.wait_for(lambda: finished_workers == worker_count)
            stopped = standard_time.perf_counter_ns()
            for thread in threads:
                thread.join()
    finally:
        standard_time.time = original_time

    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError(f"Python benchmark worker failed: {failures[0]!r}")
    return RunResult(elapsed_ns=stopped - started, workers=results)


def combine_hashes(workers: list[WorkerResult], *, output: bool) -> int:
    combined = FNV_OFFSET
    for index, worker in enumerate(workers):
        combined = fnv_update_u64(combined, index)
        combined = fnv_update_u64(
            combined,
            worker.output_hash if output else worker.return_hash,
        )
    return combined


def expected_local_output_hash(iterations: int) -> int:
    hash_value = FNV_OFFSET
    for _ in range(iterations):
        hash_value = fnv_update(hash_value, OUTPUT_LINE)
    return hash_value


def python_benchmark(
    module: Any,
    worker_count: int,
    iterations: int,
) -> BenchmarkMetrics:
    baseline = run_python_workers(
        module, worker_count, iterations, wrapped=False
    )
    wrapped = run_python_workers(module, worker_count, iterations, wrapped=True)

    baseline_return_checksum = combine_hashes(baseline.workers, output=False)
    return_checksum = combine_hashes(wrapped.workers, output=False)
    output_checksum = combine_hashes(wrapped.workers, output=True)

    local_expected = expected_local_output_hash(iterations)
    expected_workers = [
        WorkerResult(output_hash=local_expected) for _ in range(worker_count)
    ]
    expected_output_checksum = combine_hashes(expected_workers, output=True)
    output_bytes = sum(worker.output_bytes for worker in wrapped.workers)
    expected_output_bytes = worker_count * iterations * len(OUTPUT_LINE)
    calls = sum(worker.calls for worker in wrapped.workers)
    clock_calls = sum(worker.clock_calls for worker in wrapped.workers)
    expected_calls = worker_count * iterations

    passed = (
        baseline_return_checksum == return_checksum
        and output_checksum == expected_output_checksum
        and output_bytes == expected_output_bytes
        and calls == expected_calls
        and clock_calls == expected_calls * 2
    )
    return BenchmarkMetrics(
        workers=worker_count,
        iterations_per_worker=iterations,
        baseline_elapsed_ns=baseline.elapsed_ns,
        wrapped_elapsed_ns=wrapped.elapsed_ns,
        baseline_return_checksum=f"{baseline_return_checksum:016x}",
        return_checksum=f"{return_checksum:016x}",
        output_checksum=f"{output_checksum:016x}",
        expected_output_checksum=f"{expected_output_checksum:016x}",
        output_bytes=output_bytes,
        expected_output_bytes=expected_output_bytes,
        calls=calls,
        clock_calls=clock_calls,
        status="PASS" if passed else "FAIL",
    )


def parse_cpp_metrics(output: str) -> BenchmarkMetrics:
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise RuntimeError(f"malformed C++ benchmark line: {line!r}")
        values[key] = value

    required = {
        "benchmark_version",
        "workers",
        "iterations_per_worker",
        "baseline_elapsed_ns",
        "wrapped_elapsed_ns",
        "baseline_return_checksum",
        "return_checksum",
        "output_checksum",
        "expected_output_checksum",
        "output_bytes",
        "expected_output_bytes",
        "calls",
        "clock_calls",
        "status",
    }
    if set(values) != required:
        raise RuntimeError(
            "unexpected C++ benchmark fields: "
            f"missing={sorted(required - set(values))}, "
            f"extra={sorted(set(values) - required)}"
        )
    if values["benchmark_version"] != "1":
        raise RuntimeError(
            f"unsupported C++ benchmark version: {values['benchmark_version']}"
        )
    return BenchmarkMetrics(
        workers=int(values["workers"]),
        iterations_per_worker=int(values["iterations_per_worker"]),
        baseline_elapsed_ns=int(values["baseline_elapsed_ns"]),
        wrapped_elapsed_ns=int(values["wrapped_elapsed_ns"]),
        baseline_return_checksum=values["baseline_return_checksum"],
        return_checksum=values["return_checksum"],
        output_checksum=values["output_checksum"],
        expected_output_checksum=values["expected_output_checksum"],
        output_bytes=int(values["output_bytes"]),
        expected_output_bytes=int(values["expected_output_bytes"]),
        calls=int(values["calls"]),
        clock_calls=int(values["clock_calls"]),
        status=values["status"],
    )


def cpp_benchmark(
    harness: pathlib.Path,
    worker_count: int,
    iterations: int,
) -> BenchmarkMetrics:
    process = subprocess.run(
        [str(harness), "benchmark", str(worker_count), str(iterations)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return parse_cpp_metrics(process.stdout)


def validate_pair(
    python_metrics: BenchmarkMetrics,
    cpp_metrics: BenchmarkMetrics,
) -> list[str]:
    failures: list[str] = []
    if python_metrics.status != "PASS":
        failures.append("Python benchmark checksum gate failed")
    if cpp_metrics.status != "PASS":
        failures.append("C++ benchmark checksum gate failed")

    exact_fields = (
        "workers",
        "iterations_per_worker",
        "baseline_return_checksum",
        "return_checksum",
        "output_checksum",
        "expected_output_checksum",
        "output_bytes",
        "expected_output_bytes",
        "calls",
        "clock_calls",
    )
    for field in exact_fields:
        python_value = getattr(python_metrics, field)
        cpp_value = getattr(cpp_metrics, field)
        if python_value != cpp_value:
            failures.append(
                f"{field}: Python={python_value!r}, C++={cpp_value!r}"
            )
    return failures


def benchmark_case(
    module: Any,
    harness: pathlib.Path,
    worker_count: int,
    iterations: int,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    python_runs: list[BenchmarkMetrics] = []
    cpp_runs: list[BenchmarkMetrics] = []
    for sample in range(warmups + repeats):
        if sample % 2 == 0:
            python_metrics = python_benchmark(module, worker_count, iterations)
            cpp_metrics = cpp_benchmark(harness, worker_count, iterations)
        else:
            cpp_metrics = cpp_benchmark(harness, worker_count, iterations)
            python_metrics = python_benchmark(module, worker_count, iterations)
        failures = validate_pair(python_metrics, cpp_metrics)
        if failures:
            raise RuntimeError("; ".join(failures))
        if sample >= warmups:
            python_runs.append(python_metrics)
            cpp_runs.append(cpp_metrics)

    def distribution(values: list[int]) -> tuple[int, int, int]:
        median = int(statistics.median(values))
        mad = int(statistics.median(abs(value - median) for value in values))
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return median, mad, ordered[p95_index]

    python_baseline, python_baseline_mad, python_baseline_p95 = distribution(
        [run.baseline_elapsed_ns for run in python_runs]
    )
    python_wrapped, python_wrapped_mad, python_wrapped_p95 = distribution(
        [run.wrapped_elapsed_ns for run in python_runs]
    )
    cpp_baseline, cpp_baseline_mad, cpp_baseline_p95 = distribution(
        [run.baseline_elapsed_ns for run in cpp_runs]
    )
    cpp_wrapped, cpp_wrapped_mad, cpp_wrapped_p95 = distribution(
        [run.wrapped_elapsed_ns for run in cpp_runs]
    )
    reference = python_runs[0]
    total_calls = worker_count * iterations
    python_overhead = python_wrapped - python_baseline
    cpp_overhead = cpp_wrapped - cpp_baseline
    overhead_speedup = (
        python_overhead / cpp_overhead
        if python_overhead > 0 and cpp_overhead > 0
        else float("nan")
    )
    return {
        "workers": worker_count,
        "iterations_per_worker": iterations,
        "warmups": warmups,
        "repeats": repeats,
        "total_calls": total_calls,
        "python_baseline_ns": python_baseline,
        "python_baseline_mad_ns": python_baseline_mad,
        "python_baseline_p95_ns": python_baseline_p95,
        "python_wrapped_ns": python_wrapped,
        "python_wrapped_mad_ns": python_wrapped_mad,
        "python_wrapped_p95_ns": python_wrapped_p95,
        "cpp_baseline_ns": cpp_baseline,
        "cpp_baseline_mad_ns": cpp_baseline_mad,
        "cpp_baseline_p95_ns": cpp_baseline_p95,
        "cpp_wrapped_ns": cpp_wrapped,
        "cpp_wrapped_mad_ns": cpp_wrapped_mad,
        "cpp_wrapped_p95_ns": cpp_wrapped_p95,
        "python_wrapper_overhead_ns": python_overhead,
        "cpp_wrapper_overhead_ns": cpp_overhead,
        "python_wrapped_ns_per_call": python_wrapped / total_calls,
        "cpp_wrapped_ns_per_call": cpp_wrapped / total_calls,
        "python_overhead_ns_per_call": python_overhead / total_calls,
        "cpp_overhead_ns_per_call": cpp_overhead / total_calls,
        "wrapped_speedup": python_wrapped / cpp_wrapped,
        "baseline_speedup": python_baseline / cpp_baseline,
        "overhead_speedup": overhead_speedup,
        "return_checksum": reference.return_checksum,
        "output_checksum": reference.output_checksum,
        "output_bytes": reference.output_bytes,
    }


def default_python_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "virne"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-root", type=pathlib.Path, default=default_python_root())
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.workers <= 0
        or args.iterations <= 0
        or args.warmups < 0
        or args.repeats <= 0
    ):
        raise ValueError(
            "workers/iterations/repeats must be positive and warmups non-negative"
        )
    harness = args.harness.resolve()
    if not harness.is_file():
        raise FileNotFoundError(f"C++ harness not found: {harness}")

    module = compare_stats.load_oracle(args.python_root.resolve())
    result = benchmark_case(
        module,
        harness,
        args.workers,
        args.iterations,
        args.warmups,
        args.repeats,
    )
    print("stats benchmark: PASS")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
