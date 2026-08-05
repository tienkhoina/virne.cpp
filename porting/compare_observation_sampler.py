#!/usr/bin/env python3
"""Exact parity and one focused benchmark for typed observation sampling."""

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

import numpy as np
import torch


FIXTURE_VERSION = 2
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


class _OracleGraph:
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


class _OracleBatch:
    """Minimal Batch boundary used to execute Virne's original function."""

    def __init__(self, graphs: list[_OracleGraph]) -> None:
        self.graphs = list(graphs)
        xs: list[torch.Tensor] = []
        edges: list[torch.Tensor] = []
        edge_attrs: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        ptr = [0]
        edge_ptr = [0]
        node_offset = 0
        edge_offset = 0
        for graph_index, graph in enumerate(self.graphs):
            xs.append(graph.x)
            edges.append(graph.edge_index + node_offset)
            edge_attrs.append(graph.edge_attr)
            batches.append(torch.full(
                (graph.x.shape[0],), graph_index, dtype=torch.int64,
                device=graph.x.device,
            ))
            node_offset += int(graph.x.shape[0])
            edge_offset += int(graph.edge_index.shape[1])
            ptr.append(node_offset)
            edge_ptr.append(edge_offset)
        reference = self.graphs[0].x
        self.x = torch.cat(xs, dim=0)
        self.edge_index = torch.cat(edges, dim=1)
        self.edge_attr = torch.cat(edge_attrs, dim=0)
        self.batch = torch.cat(batches, dim=0)
        self.ptr = torch.tensor(ptr, dtype=torch.int64, device=reference.device)
        self.edge_ptr = torch.tensor(
            edge_ptr, dtype=torch.int64, device=reference.device
        )
        self.max_num_nodes = max(int(graph.x.shape[0]) for graph in self.graphs)

    @classmethod
    def from_data_list(cls, graphs: list[_OracleGraph]) -> "_OracleBatch":
        return cls(graphs)

    def index_select(self, indices: torch.Tensor) -> list[_OracleGraph]:
        return [
            self.graphs[int(index)]
            for index in indices.detach().to("cpu").tolist()
        ]

    def to(self, device: torch.device | str) -> "_OracleBatch":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.batch = self.batch.to(device)
        self.ptr = self.ptr.to(device)
        self.edge_ptr = self.edge_ptr.to(device)
        return self


def load_oracle_get_observations_sample() -> tuple[Any, pathlib.Path, str]:
    source_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
        / "utils.py"
    )
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text, source_path)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_observations_sample"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"Batch": _OracleBatch}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    function_text = ast.get_source_segment(source_text, function)
    if function_text is None:
        raise RuntimeError("cannot isolate get_observations_sample source")
    source_hash = hashlib.sha256(function_text.encode("utf-8")).hexdigest()
    return namespace["get_observations_sample"], source_path, source_hash


ORACLE_GET_OBSERVATIONS_SAMPLE, ORACLE_SOURCE, ORACLE_SOURCE_SHA256 = (
    load_oracle_get_observations_sample()
)


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
    elif array.dtype == np.dtype("int64"):
        dtype, width, view = "int64", 16, np.uint64
    elif array.dtype == np.dtype("bool"):
        dtype, width, view = "bool", 2, np.uint8
    else:
        raise TypeError(f"unexpected sampler dtype: {array.dtype}")
    bits: list[str] = []
    if include_bits:
        bits = [
            format(int(item), f"0{width}x")
            for item in array.view(view).reshape(-1)
        ]
    return {
        "shape": list(array.shape),
        "dtype": dtype,
        "device": str(value.device),
        "checksum": str(checksum),
        "bytes": len(payload),
        "bits": bits,
    }


def exact_cases(device: torch.device) -> list[dict[str, Any]]:
    selected_cpu = torch.tensor([3, 1, 3, -4], dtype=torch.int64)
    selected = selected_cpu.to(device)
    plain = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
    p_net_x = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 3, 2)
    v_node_x = torch.arange(12, dtype=torch.float64, device=device).reshape(4, 3)
    curr = torch.tensor([10, 11, 12, 13], dtype=torch.int64, device=device)
    mask = torch.tensor(
        [1, 0, 0, 1, 1, 1, 0, 0], dtype=torch.bool, device=device
    ).reshape(4, 2)
    size = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device=device)

    dense_sample = ORACLE_GET_OBSERVATIONS_SAMPLE(
        {
            "p_net_x": p_net_x,
            "v_node_x": v_node_x,
            "curr_v_node_id": curr,
            "action_mask": mask,
            "v_net_size": size,
        },
        selected,
        device,
    )
    mlp_sample = ORACLE_GET_OBSERVATIONS_SAMPLE(
        {"p_net_x": p_net_x, "curr_v_node_id": curr}, selected, device
    )
    first = _OracleGraph(
        torch.tensor([1.0, 2.0], dtype=torch.float32, device=device).reshape(2, 1),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device),
        torch.tensor(
            [0.25, 0.5], dtype=torch.float32, device=device
        ).reshape(2, 1),
    )
    second = _OracleGraph(
        torch.tensor(
            [5.0, 6.0, 7.0], dtype=torch.float32, device=device
        ).reshape(3, 1),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=device),
        torch.tensor(
            [0.75, 1.0], dtype=torch.float32, device=device
        ).reshape(2, 1),
    )
    graph_sample = ORACLE_GET_OBSERVATIONS_SAMPLE(
        {"graph": _OracleBatch.from_data_list([first, second])},
        torch.tensor([1, 0, 1], dtype=torch.int64, device=device),
        device,
    )["graph"]

    values = [
        ("tensor", ORACLE_GET_OBSERVATIONS_SAMPLE(plain, selected, device)),
        ("dense_p_net_x", dense_sample["p_net_x"]),
        ("dense_v_node_x", dense_sample["v_node_x"]),
        ("dense_curr_v_node_id", dense_sample["curr_v_node_id"]),
        ("dense_action_mask", dense_sample["action_mask"]),
        ("dense_v_net_size", dense_sample["v_net_size"]),
        ("mlp_p_net_x", mlp_sample["p_net_x"]),
        ("mlp_curr_v_node_id", mlp_sample["curr_v_node_id"]),
        ("graph_x", graph_sample.x),
        ("graph_edge_index", graph_sample.edge_index),
        ("graph_edge_attr", graph_sample.edge_attr),
        ("graph_batch", graph_sample.batch),
        ("graph_ptr", graph_sample.ptr),
        ("graph_edge_ptr", graph_sample.edge_ptr),
        ("graph_max_num_nodes", torch.tensor(
            [graph_sample.max_num_nodes], dtype=torch.int64
        )),
    ]
    return [
        {"name": name, "output": tensor_summary(output)}
        for name, output in values
    ]


def benchmark_fixture(device: torch.device) -> dict[str, torch.Tensor]:
    rows = 4096
    return {
        "p_net_x": (
            torch.arange(rows * 64, dtype=torch.float32, device=device)
            .reshape(rows, 64)
            .mul_(0.001)
        ),
        "v_node_x": (
            torch.arange(rows * 16, dtype=torch.float32, device=device)
            .reshape(rows, 16)
            .mul_(0.002)
        ),
        "curr_v_node_id": torch.arange(rows, dtype=torch.int64, device=device),
        "action_mask": (
            torch.arange(rows * 128, dtype=torch.int64, device=device)
            .remainder(3)
            .ne(0)
            .reshape(rows, 128)
        ),
        "v_net_size": torch.arange(
            rows, dtype=torch.float32, device=device
        ).mul_(0.25),
    }


def dense_summary(
    values: dict[str, torch.Tensor], include_bits: bool = False
) -> dict[str, Any]:
    return {
        key: tensor_summary(values[key], include_bits=include_bits)
        for key in (
            "p_net_x",
            "v_node_x",
            "curr_v_node_id",
            "action_mask",
            "v_net_size",
        )
    }


def benchmark_python(
    device: torch.device,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    source = benchmark_fixture(device)
    indices = (
        torch.arange(1024, dtype=torch.int64, device=device)
        .mul(37)
        .remainder(4096)
    )

    def sample() -> dict[str, torch.Tensor]:
        return ORACLE_GET_OBSERVATIONS_SAMPLE(source, indices, device)

    output: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for _ in range(warmups):
            for _ in range(iterations):
                output = sample()
        samples_ns: list[int] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            for _ in range(iterations):
                output = sample()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ns.append(time.perf_counter_ns() - started)
    return {
        "name": "dense_sample_4096_to_1024",
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": dense_summary(output),
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
            return [{
                "path": path,
                "python_length": len(expected),
                "cpp_length": len(actual),
                "reason": "length",
            }]
        result: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare(left, right, f"{path}[{index}]"))
            if len(result) >= 48:
                return result[:48]
        return result
    return [] if expected == actual else [
        {"path": path, "python": expected, "cpp": actual, "reason": "value"}
    ]


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        str(args.native),
        "--device", args.device,
        "--threads", str(args.threads),
        "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions),
        "--iterations", str(args.iterations),
    ]
    if args.benchmark:
        command.append("--benchmark")
    process = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"native observation sampler failed: {process.stderr.strip()}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines")
    return json.loads(lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    native = run_native(args)
    expected = {
        "component": "observation_sampler",
        "device": device.type,
        "threads": args.threads,
        "cases": exact_cases(device),
    }
    actual_exact = {key: value for key, value in native.items() if key != "benchmark"}
    mismatches = compare(expected, actual_exact)
    benchmark_report: Any = "DEFERRED_UNTIL_EXACT"
    benchmark_mismatches: list[dict[str, Any]] = []
    if args.benchmark:
        native_benchmark = native.get("benchmark")
        if not isinstance(native_benchmark, dict):
            raise RuntimeError("native benchmark report is missing")
        python_benchmark = benchmark_python(
            device, args.warmups, args.repetitions, args.iterations
        )
        benchmark_mismatches = compare(
            {key: value for key, value in python_benchmark.items()
             if key != "samples_ns"},
            {key: value for key, value in native_benchmark.items()
             if key != "samples_ns"},
            "benchmark",
        )
        native_samples = [
            int(value) / args.iterations
            for value in native_benchmark["samples_ns"]
        ]
        python_samples = [
            int(value) / args.iterations
            for value in python_benchmark["samples_ns"]
        ]
        native_median = statistics.median(native_samples)
        python_median = statistics.median(python_samples)
        benchmark_report = {
            "name": python_benchmark["name"],
            "device": device.type,
            "threads": args.threads,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "iterations": args.iterations,
            "native_samples_ns_per_call": native_samples,
            "python_samples_ns_per_call": python_samples,
            "native_median_ns_per_call": native_median,
            "python_median_ns_per_call": python_median,
            "python_over_native_speedup": python_median / native_median,
            "output_checksums": {
                key: value["checksum"]
                for key, value in python_benchmark["output"].items()
            },
        }
    status = "PASS" if not mismatches and not benchmark_mismatches else "FAIL"
    report = {
        "component": "observation_sampler",
        "fixture_version": FIXTURE_VERSION,
        "python_source": str(ORACLE_SOURCE),
        "python_source_sha256": ORACLE_SOURCE_SHA256,
        "status": status,
        "exact_cases": f"{len(expected['cases'])}/{len(expected['cases'])}"
        if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark_mismatches": benchmark_mismatches,
        "benchmark": benchmark_report,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "component": report["component"],
        "status": status,
        "exact_cases": report["exact_cases"],
        "benchmark": benchmark_report,
    }, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
