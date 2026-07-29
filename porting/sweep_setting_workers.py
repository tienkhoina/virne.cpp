#!/usr/bin/env python3
"""Sweep setting batch workers with rotated order and invariant checksums.

The harness performs all timed work internally, so subprocess startup is not
part of a sample.  Each outer round warms every mode, then rotates and reverses
the measured 1..N plus automatic-mode order.  Run ``--help`` for options.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


AUTO = 0
ROWS = ("batch_parse", "batch_dump")
EXPECTED_PYYAML = "6.0.1"


@dataclass(frozen=True)
class Summary:
    median_ms: float
    mad_ms: float
    p95_ms: float
    speedup_vs_one: float


def parse_workers(text: str) -> list[int]:
    output: list[int] = []
    for raw in text.split(","):
        token = raw.strip()
        try:
            value = int(token)
        except ValueError as error:
            raise ValueError(f"invalid worker count: {token!r}") from error
        if value < 1:
            raise ValueError("explicit worker counts must be positive")
        if value not in output:
            output.append(value)
    if 1 not in output:
        raise ValueError("worker sweep must include 1 as its baseline")
    return output


def mode_label(mode: int) -> str:
    return "auto" if mode == AUTO else str(mode)


def benchmark_inputs(path: Path, requested: str) -> dict[str, bytes]:
    source = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        source_format = "yaml"
        value = yaml.load(source.decode("utf-8"), Loader=yaml.Loader)
    elif suffix == ".json":
        source_format = "json"
        value = json.loads(source)
    else:
        raise ValueError("fixture extension must be .yaml, .yml, or .json")
    generated = {
        source_format: source,
        "json": json.dumps(value).encode("utf-8"),
        "yaml": yaml.dump(value).encode("utf-8"),
    }
    if requested == "both":
        return {"yaml": generated["yaml"], "json": generated["json"]}
    return {requested: generated[requested]}


def run_harness(
    executable: Path,
    format_name: str,
    source: bytes,
    workers: int,
    operations: int,
    batch_size: int,
    timeout: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(executable),
            "benchmark",
            format_name,
            str(workers),
            str(operations),
            str(batch_size),
            "1",
        ],
        input=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    try:
        response = json.loads(completed.stdout)
    except Exception as error:
        raise RuntimeError(
            f"workers={mode_label(workers)} emitted invalid JSON: "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        ) from error
    if completed.returncode != 0 or not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(
            f"workers={mode_label(workers)} failed with exit {completed.returncode}: "
            f"{response!r}, stderr={completed.stderr!r}"
        )
    if response.get("workers") != workers or response.get("format") != format_name:
        raise AssertionError("harness benchmark metadata differs from request")
    for row in ROWS:
        item = response.get(row)
        if not isinstance(item, dict):
            raise AssertionError(f"harness omitted {row}")
        total = item.get("total_ns")
        count = item.get("operations")
        expected_count = operations * batch_size
        if (
            not isinstance(total, int)
            or not isinstance(count, int)
            or total <= 0
            or count != expected_count
        ):
            raise AssertionError(f"invalid {row} timing row: {item!r}")
    if not isinstance(response.get("checksum"), int):
        raise AssertionError("harness omitted integer checksum")
    if not isinstance(response.get("output_size"), int):
        raise AssertionError("harness omitted integer output_size")
    return response


def rotated_modes(modes: list[int], index: int) -> list[int]:
    offset = index % len(modes)
    output = modes[offset:] + modes[:offset]
    if index % 2:
        output.reverse()
    return output


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty timing sample")
    return float(statistics.median(values))


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def collect_format(
    executable: Path,
    format_name: str,
    source: bytes,
    modes: list[int],
    warmups: int,
    repetitions: int,
    rounds: int,
    operations: int,
    batch_size: int,
    timeout: float,
) -> tuple[dict[str, dict[int, Summary]], dict[str, int]]:
    round_medians: dict[str, dict[int, list[float]]] = {
        row: {mode: [] for mode in modes} for row in ROWS
    }
    expected_checksum: int | None = None
    expected_output_size: int | None = None

    for round_index in range(rounds):
        warm_order = rotated_modes(modes, round_index)
        for warmup in range(warmups):
            for mode in rotated_modes(warm_order, warmup):
                response = run_harness(
                    executable,
                    format_name,
                    source,
                    mode,
                    operations,
                    batch_size,
                    timeout,
                )
                checksum = response["checksum"]
                output_size = response["output_size"]
                if expected_checksum is None:
                    expected_checksum = checksum
                    expected_output_size = output_size
                elif checksum != expected_checksum or output_size != expected_output_size:
                    raise AssertionError(
                        f"{format_name}: checksum/output size changed during warm-up "
                        f"at workers={mode_label(mode)}"
                    )

        current: dict[str, dict[int, list[float]]] = {
            row: {mode: [] for mode in modes} for row in ROWS
        }
        for sample_index in range(repetitions):
            order = rotated_modes(modes, round_index + sample_index)
            for mode in order:
                response = run_harness(
                    executable,
                    format_name,
                    source,
                    mode,
                    operations,
                    batch_size,
                    timeout,
                )
                checksum = response["checksum"]
                output_size = response["output_size"]
                if checksum != expected_checksum or output_size != expected_output_size:
                    raise AssertionError(
                        f"{format_name}: checksum/output size changed at "
                        f"round={round_index + 1}, sample={sample_index + 1}, "
                        f"workers={mode_label(mode)}"
                    )
                for row in ROWS:
                    item = response[row]
                    current[row][mode].append(
                        item["total_ns"] / item["operations"] / 1_000_000.0
                    )
        for row in ROWS:
            for mode in modes:
                round_medians[row][mode].append(median(current[row][mode]))
        print(
            f"collected format={format_name} round={round_index + 1}/{rounds}",
            flush=True,
        )

    assert expected_checksum is not None and expected_output_size is not None
    summaries: dict[str, dict[int, Summary]] = {row: {} for row in ROWS}
    for row in ROWS:
        baseline = median(round_medians[row][1])
        for mode in modes:
            values = round_medians[row][mode]
            row_median = median(values)
            deviations = [abs(value - row_median) for value in values]
            summaries[row][mode] = Summary(
                row_median,
                median(deviations),
                percentile95(values),
                baseline / row_median,
            )
    return summaries, {
        "checksum": expected_checksum,
        "output_size": expected_output_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness", "--cpp", dest="harness", required=True, type=Path
    )
    parser.add_argument(
        "--fixture", required=True, type=Path, help="representative YAML or JSON document"
    )
    parser.add_argument("--format", choices=("yaml", "json", "both"), default="both")
    parser.add_argument("--workers", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--operations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if yaml.__version__ != EXPECTED_PYYAML:
        raise RuntimeError(
            f"worker sweep requires PyYAML {EXPECTED_PYYAML}, found {yaml.__version__}"
        )
    executable = args.harness.resolve()
    fixture = args.fixture.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    workers = parse_workers(args.workers)
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if min(args.repetitions, args.rounds, args.operations, args.batch_size) < 1:
        raise ValueError("repetitions, rounds, operations, and batch-size must be positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")

    modes = workers + [AUTO]
    inputs = benchmark_inputs(fixture, args.format)
    all_summaries: dict[str, dict[str, dict[int, Summary]]] = {}
    invariants: dict[str, dict[str, int]] = {}
    for format_name, source in inputs.items():
        summaries, invariant = collect_format(
            executable,
            format_name,
            source,
            modes,
            args.warmups,
            args.repetitions,
            args.rounds,
            args.operations,
            args.batch_size,
            args.timeout,
        )
        all_summaries[format_name] = summaries
        invariants[format_name] = invariant

    print(
        f"\nprotocol: warmups={args.warmups}, samples={args.repetitions}, "
        f"outer-rounds={args.rounds}, inner-operations={args.operations}, "
        f"batch-size={args.batch_size}; modes={workers}+auto; "
        "rotated/reversed sample order; process startup excluded; "
        "checksum and output size invariant in every invocation"
    )

    aggregate_speedups: dict[int, list[float]] = {mode: [] for mode in modes}
    selected: dict[str, int] = {}
    for format_name, summaries in all_summaries.items():
        print(
            f"\nformat={format_name}; checksum={invariants[format_name]['checksum']}; "
            f"output_size={invariants[format_name]['output_size']}"
        )
        for row in ROWS:
            print(f"\n{format_name}_{row}")
            print("workers   median ms/op   MAD ms   p95 ms   speedup vs w1")
            print("-" * 64)
            for mode in modes:
                item = summaries[row][mode]
                aggregate_speedups[mode].append(item.speedup_vs_one)
                print(
                    f"{mode_label(mode):>7s} {item.median_ms:14.9f} "
                    f"{item.mad_ms:8.6f} {item.p95_ms:9.6f} "
                    f"{item.speedup_vs_one:13.3f}x"
                )
            best = min(workers, key=lambda mode: (summaries[row][mode].median_ms, mode))
            selected[f"{format_name}_{row}"] = best
            auto = summaries[row][AUTO]
            best_item = summaries[row][best]
            slowdown = auto.median_ms / best_item.median_ms
            status = (
                "OPTIMAL"
                if slowdown <= 1.05
                else "ACCEPTABLE"
                if slowdown <= 1.10 and auto.median_ms <= summaries[row][1].median_ms * 1.02
                else "REVIEW"
            )
            print(
                f"selected explicit={best}; auto={status}; "
                f"auto/best={slowdown:.3f}x; "
                f"auto speedup={auto.speedup_vs_one:.3f}x"
            )

    aggregate = {
        mode: geometric_mean(values) for mode, values in aggregate_speedups.items()
    }
    best_aggregate = max(workers, key=lambda mode: (aggregate[mode], -mode))
    print("\naggregate geometric speedup across formats and batch operations:")
    for mode in modes:
        print(f"  {mode_label(mode):>7s}: {aggregate[mode]:.3f}x vs worker 1")
    print(
        f"best_single_explicit={best_aggregate} "
        f"({aggregate[best_aggregate]:.3f}x); "
        f"auto={aggregate[AUTO]:.3f}x"
    )
    print("recommended explicit worker counts:")
    for name, worker in selected.items():
        print(f"  {name}: {worker}")

    if args.json_output is not None:
        payload = {
            "component": "virne.utils.setting",
            "protocol": {
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "rounds": args.rounds,
                "operations": args.operations,
                "batch_size": args.batch_size,
                "workers": workers,
                "includes_auto": True,
            },
            "invariants": invariants,
            "summaries": {
                format_name: {
                    row: {
                        mode_label(mode): asdict(item)
                        for mode, item in modes_summary.items()
                    }
                    for row, modes_summary in rows.items()
                }
                for format_name, rows in all_summaries.items()
            },
            "aggregate_speedup": {
                mode_label(mode): value for mode, value in aggregate.items()
            },
            "selected": selected,
            "best_single_explicit": best_aggregate,
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"json_report={args.json_output}")
    print("result=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"sweep_setting_workers: FAIL: {error}", file=sys.stderr)
        raise
