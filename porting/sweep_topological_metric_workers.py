#!/usr/bin/env python3
"""Sweep deterministic worker policies for topological metrics.

The C++ harness owns the samples inside each invocation.  This driver repeats
the harness in an interleaved order and summarizes the median reported by each
round, so one unusually favorable invocation cannot select the automatic
worker policy by itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import subprocess
import sys


@dataclass(frozen=True)
class CaseMeasurement:
    index: int
    name: str
    nodes: int
    edges: int
    median_ms: float
    mad_ms: float
    p95_ms: float
    checksum: int


@dataclass(frozen=True)
class HarnessRun:
    workers: int
    warmups: int
    repetitions: int
    cases: dict[str, CaseMeasurement]


def parse_workers(text: str) -> list[int]:
    workers: list[int] = []
    for token in text.split(","):
        try:
            worker = int(token.strip())
        except ValueError as error:
            raise ValueError(f"invalid worker count: {token!r}") from error
        if worker < 1:
            raise ValueError("worker counts must be positive")
        if worker not in workers:
            workers.append(worker)
    if not workers:
        raise ValueError("worker sweep cannot be empty")
    if 1 not in workers:
        raise ValueError("worker sweep must include 1 as the sequential baseline")
    return workers


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise RuntimeError(f"malformed harness line: {line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate harness key: {key}")
        values[key] = value
    return values


def require_key(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as error:
        raise RuntimeError(f"harness output is missing {key}") from error


def parse_nonnegative_int(values: dict[str, str], key: str) -> int:
    try:
        value = int(require_key(values, key))
    except ValueError as error:
        raise RuntimeError(f"{key} is not an integer") from error
    if value < 0:
        raise RuntimeError(f"{key} must be non-negative")
    return value


def parse_finite_float(values: dict[str, str], key: str) -> float:
    try:
        value = float(require_key(values, key))
    except ValueError as error:
        raise RuntimeError(f"{key} is not a float") from error
    if not math.isfinite(value):
        raise RuntimeError(f"{key} must be finite")
    return value


def parse_harness_output(text: str) -> HarnessRun:
    values = parse_key_values(text)
    count = parse_nonnegative_int(values, "bench_count")
    workers = parse_nonnegative_int(values, "bench_workers")
    warmups = parse_nonnegative_int(values, "bench_warmups")
    repetitions = parse_nonnegative_int(values, "bench_repetitions")
    if count == 0:
        raise RuntimeError("benchmark harness returned no cases")
    if repetitions == 0:
        raise RuntimeError("benchmark harness returned zero repetitions")

    cases: dict[str, CaseMeasurement] = {}
    for index in range(count):
        prefix = f"bench[{index}]"
        name = require_key(values, prefix + ".name")
        if not name:
            raise RuntimeError(f"{prefix}.name is empty")
        if name in cases:
            raise RuntimeError(f"duplicate benchmark case name: {name}")
        median_ms = parse_finite_float(values, prefix + ".cpp_median_ms")
        mad_ms = parse_finite_float(values, prefix + ".cpp_mad_ms")
        p95_ms = parse_finite_float(values, prefix + ".cpp_p95_ms")
        if median_ms <= 0.0:
            raise RuntimeError(f"{name}: median must be positive")
        if mad_ms < 0.0 or p95_ms < 0.0:
            raise RuntimeError(f"{name}: MAD and p95 must be non-negative")
        if p95_ms < median_ms:
            raise RuntimeError(f"{name}: p95 is smaller than the median")
        cases[name] = CaseMeasurement(
            index=index,
            name=name,
            nodes=parse_nonnegative_int(values, prefix + ".nodes"),
            edges=parse_nonnegative_int(values, prefix + ".edges"),
            median_ms=median_ms,
            mad_ms=mad_ms,
            p95_ms=p95_ms,
            checksum=parse_nonnegative_int(values, prefix + ".checksum"),
        )
    return HarnessRun(workers, warmups, repetitions, cases)


def run_harness(
    executable: Path,
    workers: int,
    warmups: int,
    repetitions: int,
) -> HarnessRun:
    completed = subprocess.run(
        [
            str(executable),
            "--benchmark",
            "--workers",
            str(workers),
            "--warmups",
            str(warmups),
            "--repetitions",
            str(repetitions),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"harness worker={workers} failed with exit "
            f"{completed.returncode}:\n{completed.stderr}"
        )
    run = parse_harness_output(completed.stdout)
    if run.workers != workers:
        raise AssertionError(
            f"harness reported workers={run.workers}, requested {workers}"
        )
    if run.warmups != warmups or run.repetitions != repetitions:
        raise AssertionError(
            "harness warm-up/repetition metadata differs from the request"
        )
    return run


def metadata_signature(run: HarnessRun) -> tuple[tuple[int, str, int, int], ...]:
    return tuple(
        (item.index, item.name, item.nodes, item.edges)
        for item in sorted(run.cases.values(), key=lambda value: value.index)
    )


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take a median of an empty sequence")
    return float(statistics.median(values))


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def measurement_medians(
    rounds: list[dict[int, HarnessRun]],
    worker: int,
    case_name: str,
) -> tuple[float, float, float]:
    measurements = [item[worker].cases[case_name] for item in rounds]
    return (
        median([item.median_ms for item in measurements]),
        median([item.mad_ms for item in measurements]),
        median([item.p95_ms for item in measurements]),
    )


def wins_against_baseline(
    rounds: list[dict[int, HarnessRun]],
    worker: int,
    case_name: str,
) -> int:
    if worker == 1:
        return 0
    return sum(
        item[worker].cases[case_name].median_ms
        < item[1].cases[case_name].median_ms
        for item in rounds
    )


def fastest_round_wins(
    rounds: list[dict[int, HarnessRun]],
    worker: int,
    case_name: str,
) -> int:
    wins = 0
    for item in rounds:
        fastest = min(
            run.cases[case_name].median_ms for run in item.values()
        )
        if item[worker].cases[case_name].median_ms == fastest:
            wins += 1
    return wins


def policy_cases(case_names: list[str]) -> dict[str, list[str]]:
    policies = {
        "closeness": [name for name in case_names if name.startswith("closeness_")],
        "betweenness": [
            name for name in case_names if name.startswith("betweenness_")
        ],
        "all": [name for name in case_names if name.startswith("all_metrics_")],
    }
    missing = [name for name, cases in policies.items() if not cases]
    if missing:
        raise AssertionError(
            "benchmark has no cases for auto policies: " + ", ".join(missing)
        )
    return policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--workers", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = parse_workers(args.workers)
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.repetitions < 1 or args.rounds < 1:
        raise ValueError("repetitions and rounds must be positive")
    executable = args.cpp.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    rounds: list[dict[int, HarnessRun]] = []
    expected_metadata: tuple[tuple[int, str, int, int], ...] | None = None
    expected_checksums: dict[str, int] = {}
    for round_index in range(args.rounds):
        offset = round_index % len(workers)
        order = workers[offset:] + workers[:offset]
        if round_index % 2 == 1:
            order.reverse()
        current_round: dict[int, HarnessRun] = {}
        for worker in order:
            run = run_harness(
                executable, worker, args.warmups, args.repetitions
            )
            signature = metadata_signature(run)
            if expected_metadata is None:
                expected_metadata = signature
                expected_checksums = {
                    name: item.checksum for name, item in run.cases.items()
                }
            elif signature != expected_metadata:
                raise AssertionError(
                    "benchmark case metadata changed across workers/rounds"
                )
            for name, item in run.cases.items():
                expected = expected_checksums[name]
                if item.checksum != expected:
                    raise AssertionError(
                        f"{name}: checksum changed at round={round_index + 1}, "
                        f"worker={worker}: {item.checksum} != {expected}"
                    )
            current_round[worker] = run
            print(
                f"collected round={round_index + 1}/{args.rounds} "
                f"worker={worker}",
                flush=True,
            )
        rounds.append(current_round)

    assert expected_metadata is not None
    case_names = [name for _, name, _, _ in expected_metadata]
    print(
        f"\nprotocol: warmups={args.warmups}, "
        f"repetitions={args.repetitions}, rounds={args.rounds}, "
        f"workers={workers}; exact checksums invariant across every run"
    )

    case_medians: dict[str, dict[int, float]] = {}
    for case_name in case_names:
        case_medians[case_name] = {}
        baseline, _, _ = measurement_medians(rounds, 1, case_name)
        print(f"\n{case_name}: worker-1 median={baseline:.6f} ms")
        print(
            "workers   median-of-medians ms   median MAD   median p95   "
            "vs w1   wins vs w1   fastest wins"
        )
        print("-" * 96)
        for worker in workers:
            worker_median, worker_mad, worker_p95 = measurement_medians(
                rounds, worker, case_name
            )
            case_medians[case_name][worker] = worker_median
            baseline_wins = wins_against_baseline(
                rounds, worker, case_name
            )
            fastest_wins = fastest_round_wins(
                rounds, worker, case_name
            )
            print(
                f"{worker:7d} {worker_median:22.6f} "
                f"{worker_mad:12.6f} {worker_p95:12.6f} "
                f"{baseline / worker_median:7.2f}x "
                f"{baseline_wins:7d}/{args.rounds:<3d} "
                f"{fastest_wins:8d}/{args.rounds}"
            )

    print("\noverall geometric mean versus worker 1:")
    overall_geomeans: dict[int, float] = {}
    for worker in workers:
        speedups = [
            case_medians[name][1] / case_medians[name][worker]
            for name in case_names
        ]
        overall_geomeans[worker] = geometric_mean(speedups)
        print(f"  worker {worker}: {overall_geomeans[worker]:.3f}x")

    minimum_wins = math.ceil(args.rounds * 2 / 3)
    recommendations: dict[str, int] = {}
    print(
        "\ncandidate auto policies "
        f"(reliable = beats worker 1 in >= {minimum_wins}/{args.rounds} "
        "rounds for every policy case):"
    )
    for policy, names in policy_cases(case_names).items():
        candidates: list[tuple[float, int]] = [(1.0, 1)]
        print(f"\n{policy} cases: {', '.join(names)}")
        print("workers   geometric mean   wins vs w1   reliable")
        print("-" * 54)
        for worker in workers:
            speedup = geometric_mean(
                [
                    case_medians[name][1] / case_medians[name][worker]
                    for name in names
                ]
            )
            wins = [
                wins_against_baseline(rounds, worker, name) for name in names
            ]
            reliable = worker == 1 or all(
                value >= minimum_wins for value in wins
            )
            print(
                f"{worker:7d} {speedup:14.3f}x "
                f"{sum(wins):5d}/{args.rounds * len(names):<5d} "
                f"{'yes' if reliable else 'no'}"
            )
            if worker != 1 and reliable and speedup > 1.0:
                candidates.append((speedup, worker))
        # Prefer higher aggregate speedup; on an exact tie use fewer workers.
        _, selected = max(candidates, key=lambda item: (item[0], -item[1]))
        recommendations[policy] = selected

    print("\nrecommended candidate auto policy:")
    for policy in ("closeness", "betweenness", "all"):
        worker = recommendations[policy]
        if worker == 1:
            print(f"  {policy}: 1 worker (no reliable parallel win)")
        else:
            speedup = geometric_mean(
                [
                    case_medians[name][1] / case_medians[name][worker]
                    for name in policy_cases(case_names)[policy]
                ]
            )
            print(f"  {policy}: {worker} workers ({speedup:.3f}x vs worker 1)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"sweep_topological_metric_workers: FAIL: {error}",
            file=sys.stderr,
        )
        raise
