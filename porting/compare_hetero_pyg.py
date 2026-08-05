#!/usr/bin/env python3
"""Exact PyG heterogeneous-data/GAT parity gate.

The fixture intentionally uses fixed node/relation IDs.  Dynamic PyG dict keys
are resolved once while constructing the boundary fixture; summaries and hot
forward calls use tensors directly.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GATConv, HeteroConv


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
FIXTURE_VERSION = 1


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().cpu().contiguous().numpy()
    payload = array.tobytes(order="C")
    checksum = FNV_OFFSET
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    if array.dtype == np.dtype("float32"):
        bits = array.view(np.uint32).reshape(-1)
        dtype = "float32"
        width = 8
    elif array.dtype == np.dtype("float64"):
        bits = array.view(np.uint64).reshape(-1)
        dtype = "float64"
        width = 16
    elif array.dtype == np.dtype("int64"):
        bits = array.view(np.uint64).reshape(-1)
        dtype = "int64"
        width = 16
    elif array.dtype == np.dtype("bool"):
        bits = array.astype(np.uint8).reshape(-1)
        dtype = "bool"
        width = 2
    else:
        raise TypeError(f"unsupported dtype {array.dtype}")
    return {
        "shape": list(array.shape),
        "dtype": dtype,
        "checksum": str(checksum),
        "bytes": len(payload),
        "bits": [format(int(item), f"0{width}x") for item in bits],
    }


def edge_index(values: list[list[int]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def set_relation(
    data: HeteroData,
    source: str,
    relation: str,
    target: str,
    indices: list[list[int]],
    attributes: list[list[float]],
) -> None:
    data[source, relation, target].edge_index = edge_index(indices)
    data[source, relation, target].edge_attr = torch.tensor(
        attributes, dtype=torch.float32)


def graph_zero() -> HeteroData:
    data = HeteroData()
    data["p"].x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    data["v"].x = torch.tensor(
        [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=torch.float32)
    set_relation(data, "p", "connect", "p", [[0, 1, 2], [1, 2, 0]],
                 [[0.5], [1.0], [1.5]])
    set_relation(data, "v", "connect", "v", [[0, 1], [1, 0]],
                 [[0.25], [0.75]])
    set_relation(data, "v", "mapping", "p", [[0, 1], [1, 2]],
                 [[1.0], [1.0]])
    set_relation(data, "p", "mapping", "v", [[1, 2], [0, 1]],
                 [[1.0], [1.0]])
    set_relation(data, "v", "imaginary", "p", [[0, 1], [2, 0]],
                 [[0.2], [0.4]])
    set_relation(data, "p", "imaginary", "v", [[0, 2], [1, 1]],
                 [[0.2], [0.4]])
    return data


def graph_one() -> HeteroData:
    data = HeteroData()
    data["p"].x = torch.tensor(
        [[-1.0, -2.0], [-3.0, -4.0]], dtype=torch.float32)
    data["v"].x = torch.tensor(
        [[-0.5, -1.5, -2.5], [-3.5, -4.5, -5.5],
         [-6.5, -7.5, -8.5]], dtype=torch.float32)
    set_relation(data, "p", "connect", "p", [[0], [1]], [[0.6]])
    set_relation(data, "v", "connect", "v", [[0, 1], [1, 2]],
                 [[0.3], [0.9]])
    set_relation(data, "v", "mapping", "p", [[0, 2], [1, 0]],
                 [[1.0], [1.0]])
    set_relation(data, "p", "mapping", "v", [[0, 1], [2, 1]],
                 [[1.0], [1.0]])
    set_relation(data, "v", "imaginary", "p", [[1], [1]], [[0.7]])
    set_relation(data, "p", "imaginary", "v", [[0], [2]], [[0.7]])
    return data


def fill_parameter(value: torch.Tensor, first: float, step: float) -> None:
    with torch.no_grad():
        value.copy_(torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) * step + first)


def fill_gat(module: GATConv, offset: float = 0.0) -> None:
    # The native fixture computes these expressions as float literals.  Keep
    # the same rounding before handing values to a float32 Torch tensor.
    offset32 = np.float32(offset)
    f32 = lambda value: float(np.float32(value))
    fill_parameter(module.lin_src.weight, f32(np.float32(-0.12) + offset32), 0.011)
    fill_parameter(module.lin_dst.weight, f32(np.float32(0.08) - offset32), -0.009)
    fill_parameter(module.lin_edge.weight, f32(np.float32(0.03) + offset32), 0.006)
    fill_parameter(module.att_src, f32(np.float32(-0.02) + offset32), 0.004)
    fill_parameter(module.att_dst, f32(np.float32(0.015) - offset32), -0.003)
    fill_parameter(module.att_edge, f32(np.float32(0.01) + offset32), 0.002)
    fill_parameter(module.bias, f32(np.float32(0.02) - offset32), -0.004)


def make_hetero(first: HeteroData) -> HeteroConv:
    channels = [(2, 2), (3, 3), (3, 2), (2, 3), (3, 2), (2, 3)]
    relations = [
        ("p", "connect", "p"), ("v", "connect", "v"),
        ("v", "mapping", "p"), ("p", "mapping", "v"),
        ("v", "imaginary", "p"), ("p", "imaginary", "v"),
    ]
    modules: dict[tuple[str, str, str], GATConv] = {}
    for index, (key, channel) in enumerate(zip(relations, channels)):
        module = GATConv(channel, 2, heads=1, edge_dim=1,
                         add_self_loops=index < 2)
        fill_gat(module, np.float32(index) * np.float32(0.01))
        modules[key] = module
    hetero = HeteroConv(modules, aggr="sum")
    hetero.eval()
    return hetero


def build_outputs() -> dict[str, dict[str, Any]]:
    first = graph_zero()
    second = graph_one()
    packed = Batch.from_data_list([first, second])

    bipartite = GATConv(
        (3, 2), 2, heads=2, edge_dim=1, add_self_loops=False)
    fill_parameter(bipartite.lin_src.weight, -0.20, 0.03)
    fill_parameter(bipartite.lin_dst.weight, 0.10, -0.02)
    fill_parameter(bipartite.lin_edge.weight, 0.05, 0.01)
    fill_parameter(bipartite.att_src, -0.03, 0.007)
    fill_parameter(bipartite.att_dst, 0.02, -0.005)
    fill_parameter(bipartite.att_edge, 0.01, 0.004)
    fill_parameter(bipartite.bias, 0.04, -0.006)
    bipartite.eval()
    relation = first["v", "mapping", "p"]
    with torch.no_grad():
        bipartite_output, (_, bipartite_attention) = bipartite(
            (first["v"].x, first["p"].x), relation.edge_index,
            relation.edge_attr, return_attention_weights=True)

    self_loop = GATConv((2, 2), 2, heads=1, edge_dim=1)
    fill_parameter(self_loop.lin_src.weight, -0.11, 0.02)
    fill_parameter(self_loop.lin_dst.weight, -0.11, 0.02)
    fill_parameter(self_loop.lin_edge.weight, 0.04, 0.01)
    fill_parameter(self_loop.att_src, -0.02, 0.005)
    fill_parameter(self_loop.att_dst, 0.01, -0.003)
    fill_parameter(self_loop.att_edge, 0.005, 0.002)
    fill_parameter(self_loop.bias, 0.02, -0.004)
    self_loop.eval()
    p_connect = first["p", "connect", "p"]
    with torch.no_grad():
        self_output = self_loop(
            (first["p"].x, first["p"].x), p_connect.edge_index,
            p_connect.edge_attr)

    hetero = make_hetero(first)
    with torch.no_grad():
        hetero_output = hetero(
            first.x_dict, first.edge_index_dict,
            edge_attr_dict=first.edge_attr_dict)

    return {
        "batch_p_x": tensor_summary(packed["p"].x),
        "batch_v_x": tensor_summary(packed["v"].x),
        "batch_p_batch": tensor_summary(packed["p"].batch),
        "batch_v_batch": tensor_summary(packed["v"].batch),
        "batch_p_ptr": tensor_summary(packed["p"].ptr),
        "batch_v_ptr": tensor_summary(packed["v"].ptr),
        "batch_v_mapping_p_edge_index": tensor_summary(
            packed["v", "mapping", "p"].edge_index),
        "bipartite_output": tensor_summary(bipartite_output),
        "bipartite_attention": tensor_summary(bipartite_attention),
        "self_loop_output": tensor_summary(self_output),
        "hetero_p_output": tensor_summary(hetero_output["p"]),
        "hetero_v_output": tensor_summary(hetero_output["v"]),
    }


def compare(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [{"path": path, "python": expected, "cpp": actual, "reason": "type"}]
    if isinstance(expected, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                result.extend(compare(expected[key], actual[key], f"{path}.{key}"))
            if len(result) >= 48:
                return result[:48]
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path, "python_length": len(expected),
                     "cpp_length": len(actual), "reason": "length"}]
        result = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare(left, right, f"{path}[{index}]"))
            if len(result) >= 48:
                return result[:48]
        return result
    return [] if expected == actual else [{"path": path, "python": expected,
                                           "cpp": actual, "reason": "value"}]


def benchmark_python(
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    first = graph_zero()
    hetero = make_hetero(first)
    samples_ns: list[int] = []
    final_output: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(warmups):
            for _ in range(iterations):
                hetero(first.x_dict, first.edge_index_dict,
                       edge_attr_dict=first.edge_attr_dict)
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                output = hetero(
                    first.x_dict, first.edge_index_dict,
                    edge_attr_dict=first.edge_attr_dict)
                final_output = output["p"]
            samples_ns.append(time.perf_counter_ns() - start)
    assert final_output is not None
    return {
        "name": "hetero_conv_forward",
        "threads": threads,
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": tensor_summary(final_output),
    }


def compare_benchmark(
    native: dict[str, Any],
    python: dict[str, Any],
) -> dict[str, Any]:
    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native benchmark report is missing")
    for key in ("name", "threads", "warmups", "repetitions", "iterations"):
        if native_benchmark.get(key) != python[key]:
            raise RuntimeError(
                f"benchmark parameter mismatch for {key}: "
                f"native={native_benchmark.get(key)!r}, python={python[key]!r}")
    output_mismatches = compare(python["output"], native_benchmark.get("output"),
                                "benchmark.output")
    if output_mismatches:
        raise RuntimeError(f"benchmark output differs: {output_mismatches[:4]}")
    iterations = int(python["iterations"])
    native_samples = [int(value) / iterations for value in native_benchmark["samples_ns"]]
    python_samples = [int(value) / iterations for value in python["samples_ns"]]
    native_median = statistics.median(native_samples)
    python_median = statistics.median(python_samples)
    return {
        "name": python["name"],
        "threads": python["threads"],
        "warmups": python["warmups"],
        "repetitions": python["repetitions"],
        "iterations": iterations,
        "native_samples_ns_per_call": native_samples,
        "python_samples_ns_per_call": python_samples,
        "native_median_ns_per_call": native_median,
        "python_median_ns_per_call": python_median,
        "python_over_native_speedup": (
            python_median / native_median if native_median > 0.0 else None),
        "output_checksum": python["output"]["checksum"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    expected = {
        "component": "hetero_pyg",
        "fixture_version": FIXTURE_VERSION,
        "outputs": build_outputs(),
    }
    native_command = [str(args.native)]
    if args.benchmark:
        native_command.extend([
            "--benchmark", "--threads", str(args.threads),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            "--iterations", str(args.iterations),
        ])
    process = subprocess.run(
        native_command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={
            **os.environ,
            "LD_LIBRARY_PATH": ":".join(filter(None, [
                "/usr/local/lib/python3.10/site-packages/torch/lib",
                os.environ.get("LD_LIBRARY_PATH", ""),
            ])),
        })
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode != 0:
        raise RuntimeError(f"native failed: {process.stderr.strip()}")
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines: {lines!r}")
    actual = json.loads(lines[0])
    actual_exact = {key: value for key, value in actual.items() if key != "benchmark"}
    mismatches = compare(expected, actual_exact)
    benchmark_report: Any = "DEFERRED_UNTIL_EXACT"
    if args.benchmark:
        benchmark_report = compare_benchmark(
            actual,
            benchmark_python(args.threads, args.warmups,
                             args.repetitions, args.iterations),
        )
    report = {
        "component": "hetero_pyg",
        "status": "PASS" if not mismatches else "FAIL",
        "fixture_version": FIXTURE_VERSION,
        "exact_outputs": f"{len(expected['outputs']) - len(mismatches)}/{len(expected['outputs'])}",
        "mismatches": mismatches,
        "torch_version": torch.__version__,
        "benchmark": benchmark_report,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded, encoding="utf-8")
    compact = {"component": "hetero_pyg", "status": report["status"],
               "fixture_version": FIXTURE_VERSION,
               "exact_outputs": report["exact_outputs"],
               "benchmark": benchmark_report}
    if mismatches:
        compact["mismatches"] = mismatches[:4]
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
