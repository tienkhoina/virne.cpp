#!/usr/bin/env python3
"""Canonical 1/2/4/8-worker stats benchmark with exact checksum gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys

import benchmark_stats
import compare_stats


def default_python_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "virne"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-root", type=pathlib.Path, default=default_python_root())
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="independent worker counts; canonical value is 1 2 4 8",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise ValueError(
            "iterations/repeats must be positive and warmups non-negative"
        )
    if any(worker <= 0 for worker in args.workers):
        raise ValueError("worker counts must be positive")
    if len(set(args.workers)) != len(args.workers):
        raise ValueError("worker counts must be unique")

    harness = args.harness.resolve()
    if not harness.is_file():
        raise FileNotFoundError(f"C++ harness not found: {harness}")
    module = compare_stats.load_oracle(args.python_root.resolve())

    results = [
        benchmark_stats.benchmark_case(
            module,
            harness,
            worker_count,
            args.iterations,
            args.warmups,
            args.repeats,
        )
        for worker_count in args.workers
    ]

    print("stats worker sweep: PASS")
    print(
        "workers\ttotal_calls\tpython_ms\tcpp_ms\tcpp_speedup"
        "\tpython_ns_call\tcpp_ns_call\tpython_overhead_ns_call"
        "\tcpp_overhead_ns_call\toverhead_speedup\treturn_checksum"
        "\toutput_checksum\toutput_bytes"
    )
    for result in results:
        print(
            f"{result['workers']}\t"
            f"{result['total_calls']}\t"
            f"{result['python_wrapped_ns'] / 1_000_000:.3f}\t"
            f"{result['cpp_wrapped_ns'] / 1_000_000:.3f}\t"
            f"{result['wrapped_speedup']:.3f}\t"
            f"{result['python_wrapped_ns_per_call']:.3f}\t"
            f"{result['cpp_wrapped_ns_per_call']:.3f}\t"
            f"{result['python_overhead_ns_per_call']:.3f}\t"
            f"{result['cpp_overhead_ns_per_call']:.3f}\t"
            f"{result['overhead_speedup']:.3f}\t"
            f"{result['return_checksum']}\t"
            f"{result['output_checksum']}\t"
            f"{result['output_bytes']}"
        )

    canonical_lines = [
        "|".join(
            (
                str(result["workers"]),
                str(result["iterations_per_worker"]),
                str(result["total_calls"]),
                result["return_checksum"],
                result["output_checksum"],
                str(result["output_bytes"]),
            )
        )
        for result in results
    ]
    sweep_digest = hashlib.sha256(
        ("\n".join(canonical_lines) + "\n").encode("ascii")
    ).hexdigest()
    print(f"sweep_checksum_sha256={sweep_digest}")
    print(
        "timing_note=medians; thread setup/teardown excluded; "
        "payload and output checksums exact"
    )
    if args.json_output is not None:
        report = {
            "component": "virne.utils.stats",
            "status": "pass",
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
            "protocol": {
                "workers": args.workers,
                "iterations_per_worker": args.iterations,
                "warmups": args.warmups,
                "repetitions": args.repeats,
                "thread_setup_teardown_excluded": True,
                "paired_alternating_order": True,
            },
            "results": results,
            "sweep_checksum_sha256": sweep_digest,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json_report={args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
