#!/usr/bin/env python3
"""Compact checksum-gated timing for the completed utils.config leaf."""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time

import compare_utils_config as differential


FEATURE_RESULT_FIELDS = (
    "num_extracted_p_node_attrs",
    "num_extracted_p_link_attrs",
    "num_extracted_v_node_attrs",
    "num_extracted_v_link_attrs",
    "p_num_nodes",
)


def reset_python_outputs(configs) -> None:
    for config in configs:
        config.pop("simulation", None)
        feature = config.rl.feature_constructor
        for field in FEATURE_RESULT_FIELDS:
            feature.pop(field, None)


def aggregate_checksum(configs) -> int:
    checksum = differential.hash_u64(differential.FNV_OFFSET, len(configs))
    for variant, config in enumerate(configs):
        summary = differential.serialize_summary(config, variant)
        checksum = differential.hash_u64(checksum, summary["checksum"])
    return checksum


def time_python(add_simulation, configs, repetitions: int):
    warmup = [differential.simulation_config(index) for index in range(len(configs))]
    for config in warmup:
        add_simulation(config)

    elapsed_ns = []
    for _ in range(repetitions):
        reset_python_outputs(configs)
        started = time.perf_counter_ns()
        for config in configs:
            add_simulation(config)
        elapsed_ns.append(time.perf_counter_ns() - started)
    return elapsed_ns, aggregate_checksum(configs)


def run_cpp(
    harness: pathlib.Path,
    count: int,
    workers: int,
    repetitions: int,
) -> dict:
    process = subprocess.run(
        [
            str(harness),
            "benchmark",
            str(count),
            str(workers),
            str(repetitions),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"C++ benchmark workers={workers} failed: {process.stderr.strip()}"
        )
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"malformed C++ benchmark JSON: {process.stdout!r}"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.repetitions <= 0:
        raise ValueError("count and repetitions must be positive")

    oracle = differential.load_oracle(
        args.config_source,
        args.dataset_source,
    )
    configs = [
        differential.simulation_config(index)
        for index in range(args.count)
    ]
    python_elapsed, expected_checksum = time_python(
        oracle["add_simulation_into_config"],
        configs,
        args.repetitions,
    )
    python_median = statistics.median(python_elapsed)

    rows = []
    for workers in (1, 2, 8):
        native = run_cpp(
            args.harness.resolve(),
            args.count,
            workers,
            args.repetitions,
        )
        if native["status"] != "PASS":
            raise RuntimeError(f"C++ workers={workers} status drift")
        if native["entry_count"] != args.count:
            raise RuntimeError(f"C++ workers={workers} entry-count drift")
        if native["checksum"] != expected_checksum:
            raise RuntimeError(
                f"C++ workers={workers} checksum mismatch: "
                f"{native['checksum']} != {expected_checksum}"
            )
        cpp_median = statistics.median(native["elapsed_ns"])
        speedup = python_median / cpp_median
        if speedup <= 1.0:
            raise RuntimeError(
                f"C++ workers={workers} is not faster: {speedup:.6f}x"
            )
        rows.append(
            {
                "workers": workers,
                "cpp_elapsed_ns": native["elapsed_ns"],
                "cpp_median_ns": cpp_median,
                "python_median_ns": python_median,
                "speedup": speedup,
            }
        )

    record = {
        "component": "virne.utils.config",
        "date": "2026-07-29",
        "workload": "typed simulation summary derivation",
        "count": args.count,
        "warmups": 1,
        "repetitions": args.repetitions,
        "workers": [1, 2, 8],
        "python_elapsed_ns": python_elapsed,
        "checksum": expected_checksum,
        "entry_count": args.count,
        "rows": rows,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_source_sha256": differential.CONFIG_SOURCE_SHA256,
        "dataset_source_sha256": differential.DATASET_SOURCE_SHA256,
        "result": "PASS",
    }
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "utils config benchmark: PASS; "
        + ", ".join(
            f"w{row['workers']}={row['speedup']:.3f}x" for row in rows
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
