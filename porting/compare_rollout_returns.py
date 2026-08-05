#!/usr/bin/env python3
"""Exact rollout-return parity and focused GAE benchmark gate.

The oracle formulas below mirror ``virne/solver/learning/rl_core/buffer.py``.
They stay local so this scalar leaf does not import the rest of Virne or Torch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import time
from typing import Any


FIXTURE_VERSION = 1
BENCHMARK_STEPS = 4096
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def compute_returns_with_gae(
    items: list[float],
    values: list[float],
    dones: list[int],
    last_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.98,
) -> tuple[list[float], list[float]]:
    buffer_size = len(items)
    returns = [0.0] * buffer_size
    advantages = [0.0] * buffer_size
    last_gae_lam = 0.0
    for step in reversed(range(buffer_size)):
        next_values = last_value if step == buffer_size - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = items[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )
        advantages[step] = last_gae_lam
        returns[step] = advantages[step] + values[step]
    return returns, advantages


def compute_returns_with_mc(
    items: list[float], dones: list[int], gamma: float = 0.99
) -> list[float]:
    returns: list[float] = []
    discounted_item = 0.0
    for item, is_terminal in zip(reversed(items), reversed(dones)):
        if is_terminal:
            discounted_item = 0.0
        discounted_item = item + gamma * discounted_item
        # Keep the original Python oracle's insertion semantics.
        returns.insert(0, discounted_item)
    return returns


def compute_returns_with_td(
    items: list[float],
    values: list[float],
    dones: list[int],
    last_value: float = 0.0,
    gamma: float = 0.99,
) -> list[float]:
    returns: list[float] = []
    buffer_size = len(items)
    for step in reversed(range(buffer_size)):
        next_values = last_value if step == buffer_size - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        discounted_item = (
            items[step] + gamma * next_values * next_non_terminal - values[step]
        )
        returns.insert(0, discounted_item)
    return returns


def compute_cost_returns(
    costs: list[float],
    dones: list[int],
    gamma: float,
    method: str,
) -> list[float]:
    result: list[float] = []
    if method == "reachability":
        state_cost = 0.0
        for cost, is_terminal in zip(reversed(costs), reversed(dones)):
            if is_terminal:
                state_cost = -float("inf")
            state_cost = max(cost, gamma * state_cost)
            result.insert(0, state_cost)
        return result
    if method == "cumulative":
        nonnegative_costs = [cost if cost >= 0.0 else 0.0 for cost in costs]
        state_cost = 0.0
        for cost, is_terminal in zip(reversed(nonnegative_costs), reversed(dones)):
            if is_terminal:
                state_cost = 0.0
            state_cost = cost + gamma * state_cost
            result.insert(0, state_cost)
        return result
    raise ValueError(f"unsupported cost method: {method}")


def exact_outputs() -> dict[str, Any]:
    items = [1.2, -0.4, 0.7, 0.1, -0.3]
    values = [0.2, 0.3, -0.1, 0.4, 0.05]
    dones = [0, 0, 1, 0, 1]
    costs = [1.0, -2.0, 0.5, 2.0, -1.0]
    returns, advantages = compute_returns_with_gae(
        items, values, dones, 0.25, 0.91, 0.83
    )
    return {
        "gae": {"returns": returns, "advantages": advantages},
        "mc": compute_returns_with_mc(items, dones, 0.91),
        "td": compute_returns_with_td(items, values, dones, 0.25, 0.91),
        "cost_reachability": compute_cost_returns(
            costs, dones, 0.8, "reachability"
        ),
        "cost_cumulative": compute_cost_returns(costs, dones, 0.8, "cumulative"),
    }


def make_benchmark_fixture() -> tuple[list[float], list[float], list[int]]:
    items = [((index * 37) % 101 - 50) * 0.03125 for index in range(BENCHMARK_STEPS)]
    values = [((index * 19) % 79 - 39) * 0.015625 for index in range(BENCHMARK_STEPS)]
    dones = [1 if (index + 1) % 127 == 0 else 0 for index in range(BENCHMARK_STEPS)]
    dones[-1] = 1
    return items, values, dones


def output_summary(
    returns: list[float], advantages: list[float]
) -> dict[str, Any]:
    checksum = FNV_OFFSET
    for value in (*returns, *advantages):
        for byte in struct.pack("<d", value):
            checksum ^= byte
            checksum = (checksum * FNV_PRIME) & MASK64
    return {
        "length": len(returns),
        "checksum": str(checksum),
        "last_return": returns[-1] if returns else 0.0,
        "last_advantage": advantages[-1] if advantages else 0.0,
    }


def benchmark_python(
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    del threads  # One rollout is a reverse recurrence; episodes parallelize outside it.
    items, values, dones = make_benchmark_fixture()
    output: tuple[list[float], list[float]] = ([], [])
    for _ in range(warmups):
        for _ in range(iterations):
            output = compute_returns_with_gae(items, values, dones, 0.25, 0.99, 0.98)
    samples_ns: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            output = compute_returns_with_gae(items, values, dones, 0.25, 0.99, 0.98)
        samples_ns.append(time.perf_counter_ns() - start)
    return {
        "name": "gae_returns_4096",
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": output_summary(*output),
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
            if len(result) >= 48:
                return result[:48]
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{
                "path": path,
                "python_length": len(expected),
                "cpp_length": len(actual),
                "reason": "length",
            }]
        result = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare(left, right, f"{path}[{index}]"))
            if len(result) >= 48:
                return result[:48]
        return result
    return [] if expected == actual else [{
        "path": path,
        "python": expected,
        "cpp": actual,
        "reason": "value",
    }]


def compare_benchmark(
    native: dict[str, Any],
    python: dict[str, Any],
    threads: int,
) -> dict[str, Any]:
    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native benchmark report is missing")
    expected_parameters = {
        "name": python["name"],
        "threads": threads,
        "warmups": python["warmups"],
        "repetitions": python["repetitions"],
        "iterations": python["iterations"],
    }
    for key, expected in expected_parameters.items():
        if native_benchmark.get(key) != expected:
            raise RuntimeError(
                f"benchmark parameter mismatch for {key}: "
                f"native={native_benchmark.get(key)!r}, python={expected!r}"
            )
    output_mismatches = compare(
        python["output"], native_benchmark.get("output"), "benchmark.output"
    )
    if output_mismatches:
        raise RuntimeError(f"benchmark output differs: {output_mismatches[:4]}")
    iterations = int(python["iterations"])
    native_samples = [int(value) / iterations for value in native_benchmark["samples_ns"]]
    python_samples = [int(value) / iterations for value in python["samples_ns"]]
    native_median = statistics.median(native_samples)
    python_median = statistics.median(python_samples)
    return {
        "name": python["name"],
        "threads": threads,
        "warmups": python["warmups"],
        "repetitions": python["repetitions"],
        "iterations": iterations,
        "native_samples_ns_per_call": native_samples,
        "python_samples_ns_per_call": python_samples,
        "native_median_ns_per_call": native_median,
        "python_median_ns_per_call": python_median,
        "python_over_native_speedup": (
            python_median / native_median if native_median > 0.0 else None
        ),
        "output_checksum": python["output"]["checksum"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")

    native_command = [str(args.native)]
    if args.benchmark:
        native_command.extend([
            "--benchmark", "--threads", str(args.threads),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            "--iterations", str(args.iterations),
        ])
    process = subprocess.run(
        native_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode != 0:
        raise RuntimeError(f"native failed: {process.stderr.strip()}")
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines: {lines!r}")
    actual = json.loads(lines[0])
    expected = {
        "component": "rollout_returns",
        "fixture_version": FIXTURE_VERSION,
        "outputs": exact_outputs(),
    }
    actual_exact = {key: value for key, value in actual.items() if key != "benchmark"}
    mismatches = compare(expected, actual_exact)
    benchmark_report: Any = "DEFERRED_UNTIL_EXACT"
    if args.benchmark:
        benchmark_report = compare_benchmark(
            actual,
            benchmark_python(
                args.threads, args.warmups, args.repetitions, args.iterations
            ),
            args.threads,
        )
    report = {
        "component": "rollout_returns",
        "fixture_version": FIXTURE_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "exact_outputs": "5/5" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark": benchmark_report,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded, encoding="utf-8")
    compact = {
        key: report[key]
        for key in ("component", "status", "exact_outputs", "benchmark")
    }
    if mismatches:
        compact["mismatches"] = mismatches[:4]
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
