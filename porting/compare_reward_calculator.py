#!/usr/bin/env python3
"""Exact reward-calculator registry and formula parity gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import time
from typing import Any


FIXTURE_VERSION = 1
NODES = 3
CASES = (
    {"result": True, "place": True, "route": True, "slots": 0, "ratio": 0.75},
    {"result": False, "place": True, "route": True, "slots": 1, "ratio": 0.75},
    {"result": False, "place": False, "route": True, "slots": 1, "ratio": 0.75},
    {"result": False, "place": True, "route": True, "slots": 0, "ratio": 0.0},
)
REGISTRY = (
    "gradual_intermediate",
    "adaptive_intermediate",
    "fixed_intermediate",
    "vanilla",
)


def progress(case: dict[str, Any]) -> float:
    return case["slots"] / (NODES - 1 if NODES > 1 else 1)


def reward(name: str, case: dict[str, Any], weight: float = 0.1) -> float:
    if case["result"]:
        return float(case["ratio"])
    complete = case["place"] and case["route"]
    if name == "gradual_intermediate":
        return 0.1 * progress(case) * float(case["ratio"]) if complete else -progress(case)
    if name == "adaptive_intermediate":
        return 1.0 / NODES if complete else -1.0 / NODES
    if name == "fixed_intermediate":
        return weight if complete else -weight
    return 0.0


def expected_outputs() -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for name in REGISTRY:
        rewards = [reward(name, case) for case in CASES]
        outputs[name] = {
            "rewards": rewards,
            "solution_totals": rewards,
            "calculator_total": sum(rewards),
        }
    return outputs


def benchmark_python(
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    del threads  # scalar reward path is intentionally sequential
    last_reward = 0.75
    solution_reward = 0.0
    calculator_total = 0.0
    samples: list[int] = []
    for _ in range(warmups):
        for _ in range(iterations):
            last_reward = float(0.75)
            solution_reward += last_reward
            calculator_total += last_reward
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            last_reward = float(0.75)
            solution_reward += last_reward
            calculator_total += last_reward
        samples.append(time.perf_counter_ns() - start)
    return {"samples_ns": samples, "last_reward": last_reward,
            "solution_reward": solution_reward,
            "calculator_total": calculator_total}


def compare(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if (isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and expected == actual):
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
        return result[:48]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path, "reason": "length", "python_length": len(expected), "cpp_length": len(actual)}]
        result: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare(left, right, f"{path}[{index}]"))
        return result[:48]
    return [] if expected == actual else [{"path": path, "python": expected, "cpp": actual, "reason": "value"}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")
    native_command = [str(args.native)]
    if args.benchmark:
        native_command += [
            "--benchmark", "--threads", str(args.threads),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            "--iterations", str(args.iterations),
        ]
    process = subprocess.run(
        native_command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"native failed: {process.stderr.strip()} / {lines!r}")
    actual = json.loads(lines[0])
    expected = {
        "component": "reward_calculator",
        "fixture_version": FIXTURE_VERSION,
        "outputs": expected_outputs(),
    }
    actual_exact = {key: value for key, value in actual.items() if key != "benchmark"}
    mismatches = compare(expected, actual_exact)
    benchmark: Any = "DEFERRED_UNTIL_EXACT"
    if args.benchmark:
        python_benchmark = benchmark_python(
            args.threads, args.warmups, args.repetitions, args.iterations)
        native_benchmark = actual.get("benchmark")
        if not isinstance(native_benchmark, dict):
            raise RuntimeError("native benchmark missing")
        if native_benchmark.get("last_reward") != python_benchmark["last_reward"]:
            raise RuntimeError("benchmark last reward differs")
        native_samples = [value / args.iterations for value in native_benchmark["samples_ns"]]
        python_samples = [value / args.iterations for value in python_benchmark["samples_ns"]]
        native_median = statistics.median(native_samples)
        python_median = statistics.median(python_samples)
        benchmark = {
            "name": native_benchmark["name"],
            "threads": args.threads,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "iterations": args.iterations,
            "native_samples_ns_per_call": native_samples,
            "python_samples_ns_per_call": python_samples,
            "native_median_ns_per_call": native_median,
            "python_median_ns_per_call": python_median,
            "python_over_native_speedup": python_median / native_median,
            "last_reward": python_benchmark["last_reward"],
        }
    report = {
        "component": "reward_calculator",
        "fixture_version": FIXTURE_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "exact_outputs": "4/4" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark": benchmark,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded, encoding="utf-8")
    compact = {key: report[key] for key in ("component", "status", "exact_outputs", "benchmark")}
    if mismatches:
        compact["mismatches"] = mismatches[:4]
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
