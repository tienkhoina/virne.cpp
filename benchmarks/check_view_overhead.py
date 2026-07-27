#!/usr/bin/env python3
"""Robust median gate for the C++ public-view versus indexed/raw benchmark."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
from pathlib import Path


TRAVERSALS = ("node attributes", "edge attributes", "adjacency")


def run_once(binary: Path) -> dict[str, float]:
    completed = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    ratios: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split("|")[1:-1]]
        if len(columns) != 4 or columns[0] not in TRAVERSALS:
            continue
        ratios[columns[0]] = float(columns[3].removesuffix("x"))
    if set(ratios) != set(TRAVERSALS):
        raise RuntimeError(
            "view benchmark output did not contain all traversal rows:\n"
            + completed.stdout
        )
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build/graph/graph_view_overhead_benchmark"),
    )
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--max-median", type=float, default=1.05)
    parser.add_argument(
        "--cpu",
        type=int,
        help="Linux CPU index to pin this process and inherited benchmark runs",
    )
    args = parser.parse_args()

    if args.runs < 3 or args.runs % 2 == 0:
        raise ValueError("--runs must be an odd integer of at least 3")
    if args.max_median <= 0.0:
        raise ValueError("--max-median must be positive")
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpu requires Linux sched_setaffinity")
        os.sched_setaffinity(0, {args.cpu})

    samples = {name: [] for name in TRAVERSALS}
    for _ in range(args.runs):
        for name, ratio in run_once(args.binary).items():
            samples[name].append(ratio)

    failed: list[str] = []
    print("| traversal | median view/raw | observed range | status |")
    print("|---|---:|---:|:---:|")
    for name in TRAVERSALS:
        values = samples[name]
        median = statistics.median(values)
        passed = median <= args.max_median
        if not passed:
            failed.append(name)
        print(
            f"| {name} | {median:.3f}x | "
            f"{min(values):.3f}x–{max(values):.3f}x | "
            f"{'PASS' if passed else 'FAIL'} |"
        )

    if failed:
        raise SystemExit(
            "view overhead median exceeded "
            f"{args.max_median:.3f}x for: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
