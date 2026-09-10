#!/usr/bin/env python3
"""Time three PPO episodes and a final update using an already passing gate."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import compare_learning_session as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-learning", action="store_true")
    args = parser.parse_args()
    parity_bytes = args.parity_report.read_bytes()
    parity = json.loads(parity_bytes)
    algorithm = "double_dqn" if args.value_learning else "ppo"
    if parity.get("component") != "learning_session" or parity.get("status") != "PASS" or parity["cases"].get(algorithm, {}).get("status") != "PASS":
        raise RuntimeError("a passing matching learning-session parity report is required")
    for relative, expected in parity["source_sha256"].items():
        if hashlib.sha256((reference.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("source changed after parity: " + relative)
    reference.torch.set_num_threads(1); reference.torch.set_num_interop_threads(1)
    native = reference.native_command(args.native, ["--benchmark-value" if args.value_learning else "--benchmark"])
    native_samples = json.loads(native[0])["samples_ns"]
    python_samples, expected = reference.benchmark_python(args.value_learning)
    mismatch = reference.policies.base.first_difference(expected, native[1:])
    signature = lambda lines: hashlib.sha256("\n".join(lines).encode()).hexdigest()
    percentile = lambda values: float(reference.np.percentile(values, 95))
    report = {"component": "learning_session", "status": "PASS" if mismatch is None else "FAIL",
        "algorithm": algorithm,
        "workload": "Three sequential " + algorithm + " episodes followed by flushing remaining rollout; model/environment construction, one warm solve and output serialization excluded",
        "parity_reused_from": args.parity_report.name, "parity_report_sha256": hashlib.sha256(parity_bytes).hexdigest(),
        "differential_rerun": False, "source_sha256": parity["source_sha256"],
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "python_output_sha256": signature(expected), "native_output_sha256": signature(native[1:]),
        "native_samples_ns": native_samples, "python_samples_ns": python_samples,
        "native_median_ns": statistics.median(native_samples), "python_median_ns": statistics.median(python_samples),
        "native_p95_ns": percentile(native_samples), "python_p95_ns": percentile(python_samples),
        "python_over_native_speedup": statistics.median(python_samples) / statistics.median(native_samples),
        "mismatch": mismatch, "warmups": 3, "repetitions": 5,
        "threads": 1, "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch": reference.torch.__version__,
        "platform": platform.platform(), "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "native_median_ns", "python_median_ns", "python_over_native_speedup", "mismatch")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
