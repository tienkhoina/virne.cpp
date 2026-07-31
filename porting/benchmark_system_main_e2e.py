"""One frozen end-to-end runtime comparison for online node-rank systems."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any

import compare_system_main_e2e as oracle


def percentile95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def timing_summary(python_values: list[int], native_values: list[int]):
    python_median = statistics.median(python_values)
    native_median = statistics.median(native_values)
    return {
        "python_median_ns": python_median,
        "python_p95_ns": percentile95(python_values),
        "native_median_ns": native_median,
        "native_p95_ns": percentile95(native_values),
        "speedup_median": python_median / native_median,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-settings", required=True, type=pathlib.Path)
    parser.add_argument("--native", required=True, type=pathlib.Path)
    parser.add_argument("--native-config", required=True, type=pathlib.Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions positive")

    python_rows: dict[str, dict[str, Any]] = {}
    for solver_name in oracle.SOLVER_NAMES:
        setup_samples = []
        run_samples = []
        total_samples = []
        expected = None
        for sample in range(args.warmups + args.repetitions):
            report = oracle.run_python(args.python_settings, solver_name)
            observed = oracle.comparable(report)
            if expected is None:
                expected = observed
            elif observed != expected:
                raise RuntimeError(
                    f"Python E2E output changed between samples for {solver_name}"
                )
            if sample >= args.warmups:
                setup = int(report["setup_time_ns"])
                run = int(report["run_time_ns"])
                setup_samples.append(setup)
                run_samples.append(run)
                total_samples.append(setup + run)
        python_rows[solver_name] = {
            "verification": expected,
            "setup": setup_samples,
            "run": run_samples,
            "total": total_samples,
        }

    worker_rows = {}
    for workers in oracle.DEFAULT_WORKERS:
        rows = []
        for solver_name in oracle.SOLVER_NAMES:
            setup_samples = []
            run_samples = []
            total_samples = []
            for sample in range(args.warmups + args.repetitions):
                report = oracle.run_native(
                    args.native,
                    args.native_config,
                    solver_name,
                    workers,
                )
                if oracle.comparable(report) != python_rows[solver_name][
                    "verification"
                ]:
                    raise RuntimeError(
                        f"benchmark E2E mismatch for {solver_name}, "
                        f"workers={workers}"
                    )
                if sample >= args.warmups:
                    setup = int(report["setup_time_ns"])
                    run = int(report["run_time_ns"])
                    setup_samples.append(setup)
                    run_samples.append(run)
                    total_samples.append(setup + run)
            rows.append(
                {
                    "solver": solver_name,
                    "setup": timing_summary(
                        python_rows[solver_name]["setup"], setup_samples
                    ),
                    "run": timing_summary(
                        python_rows[solver_name]["run"], run_samples
                    ),
                    "total": timing_summary(
                        python_rows[solver_name]["total"], total_samples
                    ),
                    "native_setup_samples_ns": setup_samples,
                    "native_run_samples_ns": run_samples,
                }
            )
        worker_rows[str(workers)] = rows

    result = {
        "component": "system.main_config.online.node_rank",
        "status": "PASS",
        "fixture": {
            "physical_nodes": oracle.PHYSICAL_NODES,
            "virtual_requests": oracle.VIRTUAL_REQUESTS,
        },
        "seed": oracle.SEED,
        "solvers": list(oracle.SOLVER_NAMES),
        "workers": worker_rows,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "gate": "full canonical main report matched before accepting timings",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
