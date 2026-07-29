#!/usr/bin/env python3
"""Sweep deterministic ClassDict batch worker counts and audit auto policy.

The C++ harness owns the samples in each invocation.  This driver rotates and
reverses invocation order across rounds, then compares median-of-round-medians
so a single unusually favorable run cannot select the production policy.
Every harness checksum must remain invariant for every explicit worker count,
the automatic policy, and every round before timing results are reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import subprocess
import sys


TARGET_CASES = ("batch_from_dict", "batch_to_dict")
AUTO_WORKERS = 0


@dataclass(frozen=True)
class CaseMeasurement:
    index: int
    name: str
    items: int
    fields: int
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
        token = token.strip()
        try:
            worker = int(token)
        except ValueError as error:
            raise ValueError(f"invalid worker count: {token!r}") from error
        if worker < 1:
            raise ValueError("explicit worker counts must be positive")
        if worker not in workers:
            workers.append(worker)
    if not workers:
        raise ValueError("worker sweep cannot be empty")
    if 1 not in workers:
        raise ValueError("worker sweep must include 1 as the baseline")
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
        if not key:
            raise RuntimeError(f"empty harness key in line: {line!r}")
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

        median_ms = parse_finite_float(
            values, prefix + ".cpp_median_ms"
        )
        mad_ms = parse_finite_float(values, prefix + ".cpp_mad_ms")
        p95_ms = parse_finite_float(values, prefix + ".cpp_p95_ms")
        if median_ms <= 0.0:
            raise RuntimeError(f"{name}: median must be positive")
        if mad_ms < 0.0 or p95_ms < 0.0:
            raise RuntimeError(
                f"{name}: MAD and p95 must be non-negative"
            )
        if p95_ms < median_ms:
            raise RuntimeError(f"{name}: p95 is smaller than the median")

        cases[name] = CaseMeasurement(
            index=index,
            name=name,
            items=parse_nonnegative_int(values, prefix + ".items"),
            fields=parse_nonnegative_int(values, prefix + ".fields"),
            median_ms=median_ms,
            mad_ms=mad_ms,
            p95_ms=p95_ms,
            checksum=parse_nonnegative_int(values, prefix + ".checksum"),
        )

    missing = [name for name in TARGET_CASES if name not in cases]
    if missing:
        raise RuntimeError(
            "benchmark harness is missing ClassDict batch cases: "
            + ", ".join(missing)
        )
    return HarnessRun(workers, warmups, repetitions, cases)


def run_harness(
    executable: Path,
    workers: int,
    warmups: int,
    repetitions: int,
    batch_items: int,
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
            "--batch-items",
            str(batch_items),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        label = "auto" if workers == AUTO_WORKERS else str(workers)
        raise RuntimeError(
            f"harness workers={label} failed with exit "
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


def metadata_signature(
    run: HarnessRun,
) -> tuple[tuple[int, str, int, int], ...]:
    return tuple(
        (item.index, item.name, item.items, item.fields)
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


def round_summary(
    rounds: list[dict[int, HarnessRun]],
    workers: int,
    case_name: str,
) -> tuple[float, float, float]:
    measurements = [item[workers].cases[case_name] for item in rounds]
    return (
        median([item.median_ms for item in measurements]),
        median([item.mad_ms for item in measurements]),
        median([item.p95_ms for item in measurements]),
    )


def wins_against_worker_one(
    rounds: list[dict[int, HarnessRun]],
    workers: int,
    case_name: str,
) -> int:
    if workers == 1:
        return 0
    return sum(
        item[workers].cases[case_name].median_ms
        < item[1].cases[case_name].median_ms
        for item in rounds
    )


def mode_label(workers: int) -> str:
    return "auto" if workers == AUTO_WORKERS else str(workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--workers", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-items", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = parse_workers(args.workers)
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.repetitions < 1 or args.rounds < 1:
        raise ValueError("repetitions and rounds must be positive")
    if args.batch_items < 1:
        raise ValueError("batch-items must be positive")

    executable = args.harness.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    # Zero is the harness spelling for automatic policy.  Keep it in the
    # rotation so auto is not always measured in the hottest or coldest slot.
    modes = workers + [AUTO_WORKERS]
    rounds: list[dict[int, HarnessRun]] = []
    expected_metadata: tuple[tuple[int, str, int, int], ...] | None = None
    expected_checksums: dict[str, int] = {}

    for round_index in range(args.rounds):
        offset = round_index % len(modes)
        order = modes[offset:] + modes[:offset]
        if round_index % 2 == 1:
            order.reverse()

        current_round: dict[int, HarnessRun] = {}
        for mode in order:
            run = run_harness(
                executable,
                mode,
                args.warmups,
                args.repetitions,
                args.batch_items,
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
                        f"{name}: checksum changed at "
                        f"round={round_index + 1}, "
                        f"workers={mode_label(mode)}: "
                        f"{item.checksum} != {expected}"
                    )
            current_round[mode] = run
            print(
                f"collected round={round_index + 1}/{args.rounds} "
                f"workers={mode_label(mode)}",
                flush=True,
            )
        rounds.append(current_round)

    assert expected_metadata is not None
    print(
        f"\nprotocol: warmups={args.warmups}, "
        f"repetitions={args.repetitions}, rounds={args.rounds}, "
        f"batch-items={args.batch_items}, "
        f"explicit workers={workers}, plus auto; rotating/reversed order; "
        "checksums invariant across every harness row and run"
    )

    medians: dict[str, dict[int, float]] = {
        name: {} for name in TARGET_CASES
    }
    recommendations: dict[str, int] = {}

    for case_name in TARGET_CASES:
        baseline, _, _ = round_summary(rounds, 1, case_name)
        print(
            f"\n{case_name}: worker-1 median-of-round-medians="
            f"{baseline:.6f} ms"
        )
        print(
            "workers   median-of-round-medians ms   median MAD   "
            "median p95   vs w1   round wins"
        )
        print("-" * 88)

        for mode in workers + [AUTO_WORKERS]:
            current, mad, p95 = round_summary(rounds, mode, case_name)
            medians[case_name][mode] = current
            wins = wins_against_worker_one(rounds, mode, case_name)
            print(
                f"{mode_label(mode):>7s} {current:27.6f} "
                f"{mad:12.6f} {p95:12.6f} "
                f"{baseline / current:7.2f}x "
                f"{wins:5d}/{args.rounds}"
            )

        best = min(
            workers,
            key=lambda worker: (medians[case_name][worker], worker),
        )
        recommendations[case_name] = best
        print(
            f"selected explicit: {best} worker(s), "
            f"{medians[case_name][best]:.6f} ms, "
            f"{baseline / medians[case_name][best]:.3f}x vs worker 1"
        )

    print("\ngeometric aggregate across both batch operations:")
    aggregate_speedups: dict[int, float] = {}
    for mode in workers + [AUTO_WORKERS]:
        speedup = geometric_mean(
            [
                medians[name][1] / medians[name][mode]
                for name in TARGET_CASES
            ]
        )
        aggregate_speedups[mode] = speedup
        print(f"  {mode_label(mode):>7s}: {speedup:.3f}x vs worker 1")

    aggregate_best = max(
        workers,
        key=lambda worker: (aggregate_speedups[worker], -worker),
    )
    tuned_speedup = geometric_mean(
        [
            medians[name][1] / medians[name][recommendations[name]]
            for name in TARGET_CASES
        ]
    )
    print(
        f"best single explicit width: {aggregate_best} "
        f"({aggregate_speedups[aggregate_best]:.3f}x aggregate)"
    )
    print(f"per-operation tuned aggregate: {tuned_speedup:.3f}x")

    print("\nautomatic-policy assessment:")
    auto_statuses: list[str] = []
    for case_name in TARGET_CASES:
        baseline = medians[case_name][1]
        best = recommendations[case_name]
        best_time = medians[case_name][best]
        auto_time = medians[case_name][AUTO_WORKERS]
        slowdown = auto_time / best_time
        wins = wins_against_worker_one(
            rounds, AUTO_WORKERS, case_name
        )

        # Five percent is effectively optimal for noisy wall-clock sweeps;
        # ten percent is acceptable only when auto is not slower than the
        # sequential baseline by more than two percent.
        if slowdown <= 1.05:
            status = "OPTIMAL"
        elif slowdown <= 1.10 and auto_time <= baseline * 1.02:
            status = "ACCEPTABLE"
        else:
            status = "REVIEW"
        auto_statuses.append(status)
        print(
            f"  {case_name}: {status}; auto={auto_time:.6f} ms, "
            f"best={best} workers/{best_time:.6f} ms, "
            f"slowdown={slowdown:.3f}x, "
            f"speedup-vs-w1={baseline / auto_time:.3f}x, "
            f"round-wins={wins}/{args.rounds}"
        )

    aggregate_status = (
        "REVIEW" if "REVIEW" in auto_statuses else
        "ACCEPTABLE" if "ACCEPTABLE" in auto_statuses else
        "OPTIMAL"
    )
    print(
        f"AUTO POLICY: {aggregate_status}; geometric speedup="
        f"{aggregate_speedups[AUTO_WORKERS]:.3f}x vs worker 1"
    )

    print("\nrecommended explicit worker counts:")
    for case_name in TARGET_CASES:
        print(f"  {case_name}: {recommendations[case_name]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"sweep_class_dict_workers: FAIL: {error}", file=sys.stderr)
        raise
