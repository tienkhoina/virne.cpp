#!/usr/bin/env python3
"""Exact Python/LibTorch gate and one timing row for ResNetBlock."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import compare_mlp_policy as tensor_oracle


EXPECTED_CLASS_SHA256 = (
    "2482c62844c394c01b8f52c56eab5cb816c66aa7f5a1e4d620a51b0ff04a5eeb"
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def source_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
        / "neural_network"
        / "res.py"
    )


def load_oracle() -> tuple[type[nn.Module], dict[str, str]]:
    path = source_path()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResNetBlock"
    ]
    if len(classes) != 1:
        raise RuntimeError("expected exactly one ResNetBlock class")
    segment = ast.get_source_segment(source, classes[0])
    if segment is None:
        raise RuntimeError("cannot isolate ResNetBlock source")
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    if digest != EXPECTED_CLASS_SHA256:
        raise RuntimeError(
            f"ResNetBlock source drift: expected {EXPECTED_CLASS_SHA256}, got {digest}"
        )
    module = ast.Module(body=[classes[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ResNetBlock"], {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "class_sha256": digest,
    }


ORACLE, SOURCE_METADATA = load_oracle()


def summary(value: torch.Tensor, include_bits: bool = True) -> dict[str, Any]:
    raw = tensor_oracle.tensor_summary(value, include_bits=include_bits)
    return {
        "shape": raw["shape"],
        "dtype": raw["dtype"],
        "checksum": raw["checksum"],
        "bits": raw["bits"],
    }


def input_tensor(batch: int, channels: int, height: int, width: int) -> torch.Tensor:
    return (
        torch.arange(batch * channels * height * width, dtype=torch.float32)
        .reshape(batch, channels, height, width)
        .remainder(37.0)
        .sub(18.0)
        .div(11.0)
    )


def state_summary(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "value": summary(value)}
        for name, value in sorted(model.state_dict().items())
    ]


def run_case(name: str, channels: int, batch: int, height: int,
             width: int) -> dict[str, Any]:
    torch.manual_seed({"single": 31, "rgb": 37, "wide": 41}[name])
    model = ORACLE(channels)
    model.eval()
    with torch.inference_mode():
        output = model(input_tensor(batch, channels, height, width))
    return {"name": name, "state": state_summary(model), "output": summary(output)}


def exact_cases() -> list[dict[str, Any]]:
    return [
        run_case("single", 1, 2, 3, 4),
        run_case("rgb", 3, 2, 4, 5),
        run_case("wide", 8, 1, 2, 2),
    ]


def fnv_mix(checksum: int, value: torch.Tensor) -> int:
    payload = bytes(
        value.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).tolist()
    )
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def benchmark_python(warmups: int, repetitions: int) -> dict[str, Any]:
    torch.manual_seed(101)
    model = ORACLE(64)
    model.eval()
    value = input_tensor(16, 64, 16, 16)
    checksum: int | None = None
    for _ in range(warmups):
        with torch.inference_mode():
            output = model(value)
        current = fnv_mix(FNV_OFFSET, output)
        checksum = current if checksum is None else checksum
        if current != checksum:
            raise RuntimeError("Python ResNet block checksum drift")
    samples: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(value)
        samples.append(time.perf_counter_ns() - started)
        if fnv_mix(FNV_OFFSET, output) != checksum:
            raise RuntimeError("Python ResNet block checksum drift")
    return {"samples_ns": samples, "checksum": str(checksum)}


def run_native(path: pathlib.Path, benchmark: bool, warmups: int,
               repetitions: int) -> dict[str, Any]:
    command = [str(path.resolve()), "--warmups", str(warmups),
               "--repetitions", str(repetitions)]
    if benchmark:
        command.append("--benchmark")
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups <= 0 or args.repetitions <= 0:
        parser.error("warmups and repetitions must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    expected = {"component": "res_net_block", "cases": exact_cases()}
    native = run_native(args.native, False, args.warmups, args.repetitions)
    mismatches = [] if native == expected else [{
        "path": "exact", "python": expected, "cpp": native,
    }]
    report: dict[str, Any] = {
        "component": "res_net_block",
        "fixture_version": 1,
        "oracle_backend": "ast_isolated_torch26_res_net_block",
        "python_source": SOURCE_METADATA,
        "exact_cases": "3/3" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark_mismatches": [],
        "benchmark": "NOT_REQUESTED",
        "status": "PASS" if not mismatches else "FAIL",
    }
    if mismatches:
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps({"component": report["component"], "status": "FAIL"}))
        return 1

    if args.benchmark:
        native_timed = run_native(args.native, True, args.warmups, args.repetitions)[
            "benchmark"
        ]
        python_timed = benchmark_python(args.warmups, args.repetitions)
        if str(native_timed["checksum"]) != python_timed["checksum"]:
            report["benchmark_mismatches"] = [{
                "path": "checksum", "python": python_timed["checksum"],
                "cpp": native_timed["checksum"],
            }]
            report["status"] = "FAIL"
        native_median = statistics.median(native_timed["samples_ns"])
        python_median = statistics.median(python_timed["samples_ns"])
        report["benchmark"] = {
            "name": native_timed["name"],
            "shape": [16, 64, 16, 16],
            "native_samples_ns": native_timed["samples_ns"],
            "python_samples_ns": python_timed["samples_ns"],
            "native_median_ns_per_call": native_median,
            "python_median_ns_per_call": python_median,
            "python_over_native_speedup": python_median / native_median,
            "output_checksum": native_timed["checksum"],
        }

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "component": report["component"],
        "exact_cases": report["exact_cases"],
        "benchmark": report["benchmark"],
        "status": report["status"],
    }, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
