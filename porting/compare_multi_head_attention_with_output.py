#!/usr/bin/env python3
"""Exact Python/LibTorch gate and one timing row for output attention."""

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
    "3a7ee8f14233259fed983a2c2d67558e2cac2e2649802fc6db0679e31b4684bf"
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
        / "att.py"
    )


def load_oracle() -> tuple[type[nn.Module], dict[str, str]]:
    path = source_path()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MultiHeadAttentionWithOutput"
    ]
    if len(classes) != 1:
        raise RuntimeError("expected exactly one MultiHeadAttentionWithOutput class")
    segment = ast.get_source_segment(source, classes[0])
    if segment is None:
        raise RuntimeError("cannot isolate MultiHeadAttentionWithOutput source")
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    if digest != EXPECTED_CLASS_SHA256:
        raise RuntimeError(
            "MultiHeadAttentionWithOutput source drift: "
            f"expected {EXPECTED_CLASS_SHA256}, got {digest}"
        )
    module = ast.Module(body=[classes[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["MultiHeadAttentionWithOutput"], {
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


def input_tensor(
    batch: int, length: int, features: int, offset: float = 0.0
) -> torch.Tensor:
    return (
        torch.arange(batch * length * features, dtype=torch.float32)
        .reshape(batch, length, features)
        .remainder(47.0)
        .sub(23.0)
        .div(16.0)
        .add(offset)
    )


def make_mask(
    batch: int, query_length: int, key_length: int, broadcast: bool
) -> torch.Tensor:
    if broadcast:
        mask = torch.ones((query_length, key_length), dtype=torch.float32)
        mask[1, 0] = 0.0
    else:
        mask = torch.ones(
            (batch, 1, query_length, key_length), dtype=torch.float32
        )
        mask[0, 0, 1, 2] = 0.0
        mask[1, 0, 0, 1] = 0.0
    return mask


def run_case(
    name: str,
    embedding_dim: int,
    num_heads: int,
    d_k: int | None,
    d_v: int | None,
    dropout: float,
    batch: int,
    query_length: int,
    key_length: int,
    value_length: int,
    masked: bool,
    broadcast_mask: bool,
) -> dict[str, Any]:
    torch.manual_seed({"default": 71, "masked": 73, "broadcast": 79}[name])
    kwargs: dict[str, Any] = {
        "num_heads": num_heads,
        "dropout": dropout,
    }
    if d_k is not None:
        kwargs["d_k"] = d_k
    if d_v is not None:
        kwargs["d_v"] = d_v
    model = ORACLE(embedding_dim, **kwargs)
    model.eval()
    query = input_tensor(batch, query_length, embedding_dim)
    key = input_tensor(batch, key_length, embedding_dim, 0.25)
    value = input_tensor(batch, value_length, embedding_dim, -0.5)
    mask = make_mask(batch, query_length, key_length, broadcast_mask) if masked else None
    with torch.inference_mode():
        output, weights = model(query, key, value, mask)
    return {
        "name": name,
        "query_weight": summary(model.W_q.weight),
        "key_weight": summary(model.W_k.weight),
        "value_weight": summary(model.W_v.weight),
        "output_weight": summary(model.fc.weight),
        "output": summary(output),
        "weights": summary(weights),
    }


def exact_cases() -> list[dict[str, Any]]:
    return [
        run_case("default", 4, 2, None, None, 0.1, 2, 3, 4, 4, False, False),
        run_case("masked", 5, 2, 3, 2, 0.0, 2, 2, 3, 3, True, False),
        run_case("broadcast", 6, 3, 2, 1, 0.0, 1, 3, 2, 2, True, True),
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
    model = ORACLE(64, num_heads=4, d_k=16, d_v=16, dropout=0.1)
    model.eval()
    query = input_tensor(32, 64, 64)
    key = input_tensor(32, 64, 64, 0.25)
    value = input_tensor(32, 64, 64, -0.5)
    checksum: int | None = None
    for _ in range(warmups):
        with torch.inference_mode():
            output, _ = model(query, key, value)
        current = fnv_mix(FNV_OFFSET, output)
        checksum = current if checksum is None else checksum
        if current != checksum:
            raise RuntimeError("Python output attention checksum drift")
    samples: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output, _ = model(query, key, value)
        samples.append(time.perf_counter_ns() - started)
        if fnv_mix(FNV_OFFSET, output) != checksum:
            raise RuntimeError("Python output attention checksum drift")
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

    expected = {
        "component": "multi_head_attention_with_output",
        "cases": exact_cases(),
    }
    native = run_native(args.native, False, args.warmups, args.repetitions)
    mismatches = [] if native == expected else [{
        "path": "exact", "python": expected, "cpp": native,
    }]
    report: dict[str, Any] = {
        "component": "multi_head_attention_with_output",
        "fixture_version": 1,
        "oracle_backend": "ast_isolated_torch26_multi_head_attention_with_output",
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
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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
            "shape": [32, 64, 64],
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
