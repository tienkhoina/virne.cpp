#!/usr/bin/env python3
"""Exact Python/PyG and LibTorch gate for the typed GCN wrapper."""

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
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

import compare_mlp_policy as tensor_oracle


EXPECTED_CLASS_SHA256 = {
    "GCNConvNet": "dfba2acc980bbb339901fbabcd0f0c7bd1038efb0cf68213218fab6b87c5a035",
    "GraphPooling": "8581024adbc4609b34c629d9340e31900f6155e65531fce29d405d937de399c9",
}
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
        / "gnn.py"
    )


def load_oracle() -> tuple[type[nn.Module], dict[str, str]]:
    path = source_path()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in EXPECTED_CLASS_SHA256
    ]
    if {node.name for node in classes} != set(EXPECTED_CLASS_SHA256):
        raise RuntimeError("expected GCNConvNet and GraphPooling classes")
    class_nodes = {node.name: node for node in classes}
    digests = {
        name: hashlib.sha256(
            ast.get_source_segment(source, class_nodes[name]).encode("utf-8")
        ).hexdigest()
        for name in EXPECTED_CLASS_SHA256
    }
    for name, expected in EXPECTED_CLASS_SHA256.items():
        if digests[name] != expected:
            raise RuntimeError(
                f"{name} source drift: expected {expected}, got {digests[name]}"
            )
    module = ast.Module(
        body=[class_nodes["GraphPooling"], class_nodes["GCNConvNet"]],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "GCNConv": GCNConv,
        "to_dense_batch": to_dense_batch,
        "global_add_pool": global_add_pool,
        "global_mean_pool": global_mean_pool,
        "global_max_pool": global_max_pool,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["GCNConvNet"], {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "class_sha256": json.dumps(digests, sort_keys=True),
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


def node_input(count: int, features: int, offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(count * features, dtype=torch.float32)
        .reshape(count, features)
        .remainder(29.0)
        .sub(14.0)
        .div(9.0)
        .add(offset)
    )


def make_batch(node_counts: list[int]) -> Batch:
    graphs: list[Data] = []
    for graph_id, nodes in enumerate(node_counts):
        rows: list[int] = []
        cols: list[int] = []
        for node in range(nodes - 1):
            rows.extend((node, node + 1))
            cols.extend((node + 1, node))
        edge_index = torch.tensor([rows, cols], dtype=torch.int64)
        graphs.append(
            Data(
                x=node_input(nodes, 3, graph_id * 0.25),
                edge_index=edge_index,
            )
        )
    return Batch.from_data_list(graphs)


def state_summary(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "value": summary(value)}
        for name, value in sorted(model.state_dict().items())
    ]


def run_case(
    name: str, options: dict[str, Any], node_counts: list[int]
) -> dict[str, Any]:
    torch.manual_seed({"sparse": 31, "dense_bn": 37, "sum": 41,
                       "mean": 43, "max": 47}[name])
    model = ORACLE(**options)
    model.eval()
    batch = make_batch(node_counts)
    with torch.inference_mode():
        output = model(batch)
    return {"name": name, "state": state_summary(model), "output": summary(output)}


def exact_cases() -> list[dict[str, Any]]:
    common = {"input_dim": 3, "output_dim": 4, "embedding_dim": 6}
    return [
        run_case("sparse", {**common, "num_layers": 1, "batch_norm": False,
                             "dropout_prob": 1.0, "return_batch": False,
                             "pooling": None}, [3, 5]),
        run_case("dense_bn", {**common, "num_layers": 3, "batch_norm": True,
                               "dropout_prob": 1.0, "return_batch": True,
                               "pooling": None}, [3, 5]),
        run_case("sum", {**common, "num_layers": 3, "batch_norm": False,
                          "dropout_prob": 1.0, "return_batch": False,
                          "pooling": "sum"}, [3, 5]),
        run_case("mean", {**common, "num_layers": 3, "batch_norm": False,
                           "dropout_prob": 1.0, "return_batch": False,
                           "pooling": "mean"}, [3, 5]),
        run_case("max", {**common, "num_layers": 3, "batch_norm": False,
                          "dropout_prob": 1.0, "return_batch": False,
                          "pooling": "max"}, [3, 5]),
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
    model = ORACLE(
        input_dim=3,
        output_dim=32,
        embedding_dim=64,
        num_layers=3,
        batch_norm=False,
        dropout_prob=1.0,
        return_batch=False,
        pooling=None,
    )
    model.eval()
    batch = make_batch([16] * 32)
    checksum: int | None = None
    for _ in range(warmups):
        with torch.inference_mode():
            output = model(batch)
        current = fnv_mix(FNV_OFFSET, output)
        checksum = current if checksum is None else checksum
        if current != checksum:
            raise RuntimeError("Python GCN wrapper checksum drift")
    samples: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(batch)
        samples.append(time.perf_counter_ns() - started)
        if fnv_mix(FNV_OFFSET, output) != checksum:
            raise RuntimeError("Python GCN wrapper checksum drift")
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

    expected = {"component": "gcn_conv_net", "cases": exact_cases()}
    native = run_native(args.native, False, args.warmups, args.repetitions)
    mismatches = [] if native == expected else [{
        "path": "exact", "python": expected, "cpp": native,
    }]
    report: dict[str, Any] = {
        "component": "gcn_conv_net",
        "fixture_version": 1,
        "oracle_backend": "ast_isolated_torch26_pyg27_gcn_conv_net",
        "python_source": SOURCE_METADATA,
        "exact_cases": "5/5" if not mismatches else "FAIL",
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
            "shape": [512, 32],
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
