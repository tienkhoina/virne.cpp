#!/usr/bin/env python3
"""Exact Python/LibTorch gate and one timing row for NeuralTensorNetwork."""

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

import compare_mlp_policy as tensor_oracle


EXPECTED_CLASS_SHA256 = (
    "d31b524ac423d344c4731bb3bc8b5a16ed9669e03923053ac7748a55aaef360a"
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
        / "neural_tensor.py"
    )


def load_oracle() -> tuple[type[nn.Module], dict[str, str]]:
    path = source_path()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NeuralTensorNetwork"
    ]
    if len(classes) != 1:
        raise RuntimeError("expected exactly one NeuralTensorNetwork class")
    segment = ast.get_source_segment(source, classes[0])
    if segment is None:
        raise RuntimeError("cannot isolate NeuralTensorNetwork source")
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    if digest != EXPECTED_CLASS_SHA256:
        raise RuntimeError(
            f"NeuralTensorNetwork source drift: expected {EXPECTED_CLASS_SHA256}, got {digest}"
        )
    module = ast.Module(body=[classes[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["NeuralTensorNetwork"], {
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


def input_tensor(batch: int, features: int, offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(batch * features, dtype=torch.float32)
        .reshape(batch, features)
        .remainder(19.0)
        .sub(9.0)
        .div(7.0)
        .add(offset)
    )


def run_case(
    name: str, input_features: int, tensor_neurons: int, batch: int
) -> dict[str, Any]:
    torch.manual_seed({"small": 17, "wide": 19, "scalar": 23}[name])
    model = ORACLE(input_features, tensor_neurons)
    model.eval()
    first = input_tensor(batch, input_features)
    second = input_tensor(batch, input_features, 0.25)
    with torch.inference_mode():
        output = model(first, second)
    return {
        "name": name,
        "weight_matrix": summary(model.weight_matrix),
        "weight_matrix_block": summary(model.weight_matrix_block),
        "bias": summary(model.bias),
        "output": summary(output),
    }


def exact_cases() -> list[dict[str, Any]]:
    return [
        run_case("small", 3, 2, 2),
        run_case("wide", 4, 3, 3),
        run_case("scalar", 1, 1, 1),
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
    model = ORACLE(64, 32)
    model.eval()
    first = input_tensor(32, 64)
    second = input_tensor(32, 64, 0.25)
    checksum: int | None = None
    for _ in range(warmups):
        with torch.inference_mode():
            output = model(first, second)
        current = fnv_mix(FNV_OFFSET, output)
        checksum = current if checksum is None else checksum
        if current != checksum:
            raise RuntimeError("Python neural tensor checksum drift")
    samples: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(first, second)
        samples.append(time.perf_counter_ns() - started)
        if fnv_mix(FNV_OFFSET, output) != checksum:
            raise RuntimeError("Python neural tensor checksum drift")
    return {"samples_ns": samples, "checksum": str(checksum)}


def run_native(
    path: pathlib.Path, benchmark: bool, warmups: int, repetitions: int
) -> dict[str, Any]:
    command = [
        str(path.resolve()),
        "--warmups",
        str(warmups),
        "--repetitions",
        str(repetitions),
    ]
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

    expected = {"component": "neural_tensor_network", "cases": exact_cases()}
    native = run_native(args.native, False, args.warmups, args.repetitions)
    mismatches = [] if native == expected else [{
        "path": "exact", "python": expected, "cpp": native,
    }]
    report: dict[str, Any] = {
        "component": "neural_tensor_network",
        "fixture_version": 1,
        "oracle_backend": "ast_isolated_torch26_neural_tensor_network",
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
                "path": "checksum",
                "python": python_timed["checksum"],
                "cpp": native_timed["checksum"],
            }]
            report["status"] = "FAIL"
        native_median = statistics.median(native_timed["samples_ns"])
        python_median = statistics.median(python_timed["samples_ns"])
        report["benchmark"] = {
            "name": native_timed["name"],
            "shape": [32, 64, 32],
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
