#!/usr/bin/env python3
"""Differential and focused timing gate for Virne action-mask helpers."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
FIXTURE_VERSION = 1


def tensor_summary(value: torch.Tensor, include_bits: bool = True) -> dict[str, Any]:
    array = value.detach().to("cpu").contiguous().numpy()
    payload = array.tobytes(order="C")
    checksum = FNV_OFFSET
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    if array.dtype == np.dtype("float32"):
        dtype, width, view = "float32", 8, np.uint32
    elif array.dtype == np.dtype("float64"):
        dtype, width, view = "float64", 16, np.uint64
    elif array.dtype == np.dtype("bool"):
        dtype, width, view = "bool", 2, np.uint8
    else:
        raise TypeError(f"unexpected action-mask output dtype: {array.dtype}")
    bits: list[str] = []
    if include_bits:
        flat = array.view(view).reshape(-1)
        bits = [format(int(item), f"0{width}x") for item in flat]
    return {
        "shape": list(array.shape),
        "dtype": dtype,
        "checksum": str(checksum),
        "bytes": len(payload),
        "bits": bits,
    }


def apply_mask_to_logit(values: torch.Tensor, mask: Any = None) -> torch.Tensor:
    if mask is None:
        return values
    if not isinstance(mask, torch.Tensor):
        mask = torch.BoolTensor(mask)
    mask = mask.bool().to(values.device).reshape(values.size())
    fill = torch.tensor(-1e8).float().type_as(values).to(values.device)
    return torch.where(mask, values, fill)


def apply_mask_to_prob(values: torch.Tensor, mask: Any = None) -> torch.Tensor:
    if mask is None:
        return values
    if not isinstance(mask, torch.Tensor):
        mask = torch.BoolTensor(mask)
    mask = mask.bool().to(values.device).reshape(values.size())
    fill = torch.tensor(0.0).float().type_as(values).to(values.device)
    return torch.where(mask, values, fill)


def expected_cases(device: torch.device) -> list[dict[str, Any]]:
    f32 = torch.tensor(
        [1.5, -2.25, 0.0, 4.75], dtype=torch.float32, device=device
    ).reshape(2, 2)
    f64 = torch.tensor(
        [1.25, -3.5, 2.75, 8.0], dtype=torch.float64, device=device
    ).reshape(2, 2)
    mixed_bool = torch.tensor(
        [1, 0, 1, 0], dtype=torch.bool, device=device
    ).reshape(1, 4)
    integer_mask = torch.tensor([2, 0, -1, 0], dtype=torch.int64, device=device)
    numeric_mask = torch.tensor(
        [1.0, 0.0, -2.0, 0.25], dtype=torch.float64, device=device
    )
    all_true = torch.ones((2, 2), dtype=torch.bool, device=device)
    all_false = torch.zeros((2, 2), dtype=torch.bool, device=device)
    vector_mask = [1, 0, 1, 0]
    values = [
        ("f32_logit_bool_reshape", apply_mask_to_logit(f32, mixed_bool)),
        ("f32_prob_int", apply_mask_to_prob(f32, integer_mask)),
        ("f32_logit_vector", apply_mask_to_logit(f32, vector_mask)),
        ("f32_prob_all_false", apply_mask_to_prob(f32, all_false)),
        ("f32_logit_all_true", apply_mask_to_logit(f32, all_true)),
        ("f32_logit_identity", apply_mask_to_logit(f32)),
        ("f32_prob_identity", apply_mask_to_prob(f32)),
        ("f64_logit_float", apply_mask_to_logit(f64, numeric_mask)),
        ("f64_prob_float", apply_mask_to_prob(f64, numeric_mask)),
    ]
    return [
        {"name": name, "output": tensor_summary(output)}
        for name, output in values
    ]


def benchmark_python(
    device: torch.device,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    rows, columns = 256, 1024
    values = torch.arange(
        rows * columns, dtype=torch.float32, device=device
    ).reshape(rows, columns) * 0.0007 - 0.3
    mask = (
        torch.arange(rows * columns, dtype=torch.int64, device=device)
        .remainder(5)
        .ne(0)
        .reshape(rows, columns)
    )
    with torch.inference_mode():
        output: torch.Tensor | None = None
        for _ in range(warmups):
            for _ in range(iterations):
                output = apply_mask_to_logit(values, mask)
        samples: list[int] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            for _ in range(iterations):
                output = apply_mask_to_logit(values, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter_ns() - started)
    if output is None:
        raise RuntimeError("Python action-mask benchmark produced no output")
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples,
        "output": tensor_summary(output, include_bits=False),
    }


def compare_exact(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [{"path": path, "python": expected, "cpp": actual, "reason": "type"}]
    if isinstance(expected, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                result.extend(compare_exact(expected[key], actual[key], f"{path}.{key}"))
            if len(result) >= 32:
                return result[:32]
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path, "python_length": len(expected),
                     "cpp_length": len(actual), "reason": "length"}]
        result: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare_exact(left, right, f"{path}[{index}]"))
            if len(result) >= 32:
                return result[:32]
        return result
    return [] if expected == actual else [
        {"path": path, "python": expected, "cpp": actual, "reason": "value"}
    ]


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        str(args.native), "--benchmark", "--device", args.device,
        "--threads", str(args.threads), "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions), "--iterations", str(args.iterations),
    ]
    process = subprocess.run(command, check=False, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"native action-mask harness failed: {process.stderr.strip()}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines")
    return json.loads(lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    expected = {
        "component": "action_mask",
        "device": device.type,
        "threads": args.threads,
        "cases": expected_cases(device),
    }
    native = run_native(args)
    actual_exact = {key: value for key, value in native.items() if key != "benchmark"}
    mismatches = compare_exact(expected, actual_exact)
    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native benchmark report is missing")
    python_benchmark = benchmark_python(
        device, args.warmups, args.repetitions, args.iterations
    )
    benchmark_mismatches = compare_exact(
        {key: value for key, value in python_benchmark.items()
         if key not in ("samples_ns",)},
        {key: value for key, value in native_benchmark.items()
         if key not in ("samples_ns",)},
        "benchmark",
    )
    native_samples = [int(value) / args.iterations
                      for value in native_benchmark["samples_ns"]]
    python_samples = [int(value) / args.iterations
                      for value in python_benchmark["samples_ns"]]
    native_median = statistics.median(native_samples)
    python_median = statistics.median(python_samples)
    status = "PASS" if not mismatches and not benchmark_mismatches else "FAIL"
    report = {
        "component": "action_mask",
        "fixture_version": FIXTURE_VERSION,
        "status": status,
        "exact_cases": f"{len(expected['cases']) if not mismatches else 0}/{len(expected['cases'])}",
        "mismatches": mismatches,
        "benchmark_mismatches": benchmark_mismatches,
        "benchmark": {
            "name": "mask_logits_256x1024",
            "python_samples_ns_per_call": python_samples,
            "native_samples_ns_per_call": native_samples,
            "python_median_ns_per_call": python_median,
            "native_median_ns_per_call": native_median,
            "python_over_native_speedup": python_median / native_median,
            "output_checksum": python_benchmark["output"]["checksum"],
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "component": report["component"], "status": status,
        "exact_cases": report["exact_cases"], "benchmark": report["benchmark"],
    }, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
