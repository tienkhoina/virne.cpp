#!/usr/bin/env python3
"""Benchmark a validated new RL group without rerunning its differential cases."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("policies", "training"), required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module_name, component_name = {
        "policies": ("compare_rl_policy_remaining", "rl_policy_remaining"),
        "training": ("compare_training_update", "training_update"),
    }[args.component]
    reference = importlib.import_module(module_name)
    parity_bytes = args.parity_report.read_bytes()
    parity = json.loads(parity_bytes)
    if parity.get("status") != "PASS" or parity.get("component") != component_name:
        raise RuntimeError("a passing matching parity report is required")
    expected_sources = reference.EXPECTED if args.component == "policies" else {
        key: reference.PINNED[key] for key in parity["source_sha256"]}
    if parity["source_sha256"] != expected_sources:
        raise RuntimeError("parity source hashes differ from the loaded oracle")
    if parity.get("threads", 1) != 1:
        raise RuntimeError("this recorded workload uses one Torch thread")
    reference.torch.set_num_threads(1)
    reference.torch.set_num_interop_threads(1)
    native_arguments = ["--benchmark"] + (["--threads", 1] if args.component == "policies" else [])
    native = reference.base.run_native(args.native, native_arguments)
    native_samples = json.loads(native[0])["samples_ns"]
    python_samples, outputs = reference.benchmark_python()
    mismatch = reference.base.first_difference(outputs, native[1:])
    result = {
        "component": component_name,
        "status": "PASS" if mismatch is None else "FAIL",
        "parity_reused_from": args.parity_report.name,
        "parity_report_sha256": hashlib.sha256(parity_bytes).hexdigest(),
        "differential_rerun": False,
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "source_sha256": expected_sources,
        "reference_corrections": parity.get("reference_corrections", {}),
        "torch": reference.torch.__version__, "threads": 1,
        "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "warmups": 3, "repetitions": 5,
        "iterations_per_sample": 5 if args.component == "policies" else 1,
        "native_samples_ns": native_samples, "python_samples_ns": python_samples,
        "native_median_ns": statistics.median(native_samples),
        "python_median_ns": statistics.median(python_samples),
        "python_over_native_speedup": statistics.median(python_samples) / statistics.median(native_samples),
        "python_output_sha256": hashlib.sha256("\n".join(outputs).encode()).hexdigest(),
        "native_output_sha256": hashlib.sha256("\n".join(native[1:]).encode()).hexdigest(),
        "mismatch": mismatch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
        ("component", "status", "native_median_ns", "python_median_ns", "python_over_native_speedup", "mismatch")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
