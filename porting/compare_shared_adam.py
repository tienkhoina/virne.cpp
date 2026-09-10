#!/usr/bin/env python3
"""Native shared gradient packets against the pinned, corrected SharedAdam class."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import compare_training_update as reference
from compare_learning_session import native_command

PATH = reference.ROOT / "rl_core/shared_adam.py"
DIGEST = hashlib.sha256(PATH.read_bytes()).hexdigest()
if DIGEST != reference.PINNED["rl_core/shared_adam.py"]: raise RuntimeError("SharedAdam source drift")
MODULE = reference.base.neural.load_file_module("shared_adam_reference", PATH)


def optimizer(maximum=False):
    a = torch.arange(6, dtype=torch.float32).reshape(2,3).div(8).requires_grad_()
    b = torch.arange(3, dtype=torch.float64).div(8).requires_grad_()
    result = MODULE.SharedAdam([
        {"params": [a], "lr": .003, "betas": (.9,.99), "eps": 1e-8, "weight_decay": .01, "amsgrad": maximum, "foreach": False},
        {"params": [b], "lr": .007, "betas": (.85,.98), "eps": 1e-7, "weight_decay": .02, "amsgrad": maximum, "foreach": False}])
    for p in (a,b):
        # Original integer steps are rejected by Torch 2.6; AMSGrad also needs
        # the maximum-variance lane absent from the source's eager initializer.
        result.state[p]["step"] = torch.tensor(0.)
        if maximum: result.state[p]["max_exp_avg_sq"] = torch.zeros_like(p)
    result.version = 0
    return result


def packet(step=0, missing=False):
    return [torch.full((2,3), .1 + step*.03, dtype=torch.float32),
            None if missing else torch.full((3,), -.2 + step*.02, dtype=torch.float64)]


def apply(optim, gradients):
    parameters = [p for group in optim.param_groups for p in group["params"]]
    copied = [None if grad is None else grad.detach().clone() for grad in gradients]
    optim.zero_grad(set_to_none=True)
    for p,g in zip(parameters, copied): p.grad = g
    norm = torch.nn.utils.clip_grad_norm_(parameters, .25)
    optim.step(); optim.version += 1
    return norm


def state(optim):
    record = reference.tensor_line
    lines = [record("version", torch.tensor(optim.version))]
    for i,p in enumerate(p for group in optim.param_groups for p in group["params"]):
        name, values = f"parameter_{i}", optim.state[p]
        lines.append(record(name, p))
        for suffix, key in (("step", "step"), ("mean", "exp_avg"), ("variance", "exp_avg_sq"), ("maximum", "max_exp_avg_sq")):
            lines.append(record(name + "." + suffix, values.get(key)))
    return lines


def sequential(maximum=False):
    optim = optimizer(maximum)
    parameters = [p for group in optim.param_groups for p in group["params"]]
    lines = state(optim)
    for step in range(4):
        if step == 2:
            optim.load_state_dict(saved_state); optim.version = saved_version
            with torch.no_grad():
                for p, old in zip(parameters, saved_parameters): p.copy_(old)
        norm = apply(optim, packet(step, step == 1))
        lines += [reference.tensor_line("norm", norm), *state(optim)]
        if step == 0:
            saved_state = copy.deepcopy(optim.state_dict()); saved_version = optim.version
            saved_parameters = [p.detach().clone() for p in parameters]
    return lines + state(optim) + [reference.tensor_line("invalid_rejected", torch.tensor(True))]


def exact():
    lines = ["CASE shared", *sequential(), "CASE shared_amsgrad", *sequential(True), "CASE shared_threads"]
    optim = optimizer()
    gradients = packet()
    for _ in range(64): apply(optim, gradients)
    return lines + state(optim)


def benchmark():
    samples, expected = [], None
    for repetition in range(-3, 5):
        optim = optimizer(); gradients = packet()
        start = time.perf_counter_ns()
        for _ in range(32): apply(optim, gradients)
        elapsed = time.perf_counter_ns() - start
        actual = state(optim)
        if expected is None: expected = actual
        elif actual != expected: raise RuntimeError("unstable shared benchmark")
        if repetition >= 0: samples.append(elapsed)
    return samples, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    report = {"component": "shared_adam", "source_sha256": {"rl_core/shared_adam.py": DIGEST},
        "reference_corrections": ["scalar tensor step counters required by Torch 2.6", "initialize AMSGrad maximum-variance state"],
        "native_concurrency": "four worker threads, 64 identical gradient packets, unique versions and coherent snapshots",
        "python_concurrency": "sequential equivalent; no Python IPC compatibility claimed", "threads": 1, "torch": torch.__version__}
    if args.benchmark_from:
        parity = json.loads(args.benchmark_from.read_text())
        if parity.get("status") != "PASS" or parity.get("component") != "shared_adam" or parity["source_sha256"] != report["source_sha256"]:
            raise RuntimeError("matching passing shared optimizer report required")
        native = native_command(args.native, ["--benchmark"])
        expected_samples, expected = benchmark()
        native_samples, actual = json.loads(native[0])["samples_ns"], native[1:]
        report.update(differential_rerun=False, parity_reused_from=args.benchmark_from.name,
            native_samples_ns=native_samples, python_samples_ns=expected_samples,
            native_median_ns=statistics.median(native_samples), python_median_ns=statistics.median(expected_samples),
            python_over_native_speedup=statistics.median(expected_samples)/statistics.median(native_samples), warmups=3, repetitions=5)
    else:
        expected, actual = exact(), native_command(args.native, [])
        report.update(cases=[line[5:] for line in expected if line.startswith("CASE ")], records=len(expected))
    mismatch = reference.base.first_difference(expected, actual)
    report.update(status="PASS" if mismatch is None else "FAIL", mismatch=mismatch,
        python_output_sha256=hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        native_output_sha256=hashlib.sha256("\n".join(actual).encode()).hexdigest(),
        native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "mismatch", "python_over_native_speedup") if key in report}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
