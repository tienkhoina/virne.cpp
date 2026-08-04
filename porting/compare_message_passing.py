"""Differential and compact timing gate for Virne's custom PyG layers."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any

import torch


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_layers(source: pathlib.Path, model_source: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("virne_graph_conv", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tree = ast.parse(model_source.read_text(encoding="utf-8"))
    graph_nodes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GraphViNEConv"
    ]
    if len(graph_nodes) != 1:
        raise RuntimeError(f"GraphViNEConv not found in {model_source}")
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "Linear": torch.nn.Linear,
        "MessagePassing": __import__(
            "torch_geometric.nn.conv", fromlist=["MessagePassing"]
        ).MessagePassing,
    }
    exec(compile(ast.Module(body=graph_nodes, type_ignores=[]), str(model_source), "exec"), namespace)
    module.GraphViNEConv = namespace["GraphViNEConv"]
    return module


def fill_linear(tensor: torch.Tensor, first: float, step: float) -> None:
    with torch.no_grad():
        sequence = torch.arange(
            tensor.numel(), dtype=tensor.dtype, device=tensor.device
        ).reshape(tensor.shape)
        tensor.copy_(sequence * step + first)


def fixture(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    x = torch.arange(12, dtype=dtype, device=device).reshape(4, 3) * 0.07 - 0.31
    edge_index = torch.tensor(
        [[0, 2, 1, 3, 2, 0], [1, 1, 2, 0, 3, 3]],
        dtype=torch.long,
        device=device,
    )
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": torch.arange(6, dtype=dtype, device=device) * 0.13 + 0.4,
        "edge_attr": torch.arange(12, dtype=dtype, device=device).reshape(6, 2)
        * 0.03
        - 0.2,
    }


def configure_graph_vine(layer: Any) -> None:
    fill_linear(layer.lin_rel.weight, -0.23, 0.037)
    fill_linear(layer.lin_rel.bias, 0.11, -0.021)
    fill_linear(layer.lin_root.weight, 0.31, -0.029)


def configure_edge_fusion(layer: Any) -> None:
    fill_linear(layer.lin_src.weight, -0.17, 0.013)
    if layer.lin_edge is not None:
        fill_linear(layer.lin_edge.weight, 0.07, -0.019)
    sequence = layer.pre_nns[0]
    fill_linear(sequence[0].weight, -0.08, 0.017)
    fill_linear(sequence[0].bias, 0.03, -0.011)
    fill_linear(sequence[2].weight, 0.14, -0.009)
    fill_linear(sequence[2].bias, -0.02, 0.007)
    if layer.bias is not None:
        fill_linear(layer.bias, 0.05, 0.004)


def python_records(layers: Any, device: torch.device) -> list[dict[str, Any]]:
    dtype = torch.float64
    data = fixture(device, dtype)
    records: list[dict[str, Any]] = []

    graph = layers.GraphViNEConv(3, 4, normalize=False, bias=True).to(
        device=device, dtype=dtype
    )
    configure_graph_vine(graph)
    records.append({"name": "graph_vine/output", "value": graph(
        data["x"], data["edge_index"], data["edge_weight"]
    )})

    normalized = layers.GraphViNEConv(3, 4, normalize=True, bias=True).to(
        device=device, dtype=dtype
    )
    configure_graph_vine(normalized)
    records.append({"name": "graph_vine/normalized", "value": normalized(
        data["x"], data["edge_index"], data["edge_weight"]
    )})

    edge = layers.EdgeFusionGATConv(
        3, 4, heads=1, edge_dim=2, add_self_loops=True,
        fill_value="mean", bias=True
    ).to(device=device, dtype=dtype)
    configure_edge_fusion(edge)
    records.append({"name": "edge_fusion/output", "value": edge(
        data["x"], data["edge_index"], data["edge_attr"]
    )})
    return records


def tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu().contiguous()
    return {"name": name, "shape": list(value.shape), "values": value.reshape(-1).tolist()}


def benchmark_fixture(device: torch.device) -> dict[str, torch.Tensor]:
    nodes, edges, channels = 512, 8192, 16
    dtype = torch.float32
    ids = torch.arange(edges, dtype=torch.long, device=device)
    return {
        "x": torch.arange(nodes * channels, dtype=dtype, device=device).reshape(
            nodes, channels
        ) * 0.0007 - 0.3,
        "edge_index": torch.stack((ids.remainder(nodes), (ids * 17 + 3).remainder(nodes))),
        "edge_weight": torch.arange(edges, dtype=dtype, device=device) * 0.00003 + 0.4,
        "edge_attr": torch.arange(edges * 2, dtype=dtype, device=device).reshape(
            edges, 2
        ) * 0.00002 - 0.1,
    }


def python_benchmark(
    layers: Any,
    device: torch.device,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> list[dict[str, float | str]]:
    data = benchmark_fixture(device)
    graph = layers.GraphViNEConv(16, 16, normalize=False, bias=True).to(device=device)
    configure_graph_vine(graph)
    edge = layers.EdgeFusionGATConv(
        16, 16, heads=1, edge_dim=2, add_self_loops=True,
        fill_value="mean", bias=True
    ).to(device=device)
    configure_edge_fusion(edge)
    cases = [
        ("graph_vine", lambda: graph(
            data["x"], data["edge_index"], data["edge_weight"]
        )),
        ("edge_fusion", lambda: edge(
            data["x"], data["edge_index"], data["edge_attr"]
        )),
    ]
    result: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for name, call in cases:
            for _ in range(warmups):
                output = call()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            samples: list[float] = []
            output = None
            for _ in range(repetitions):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                begin = time.perf_counter_ns()
                for _ in range(iterations):
                    output = call()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples.append((time.perf_counter_ns() - begin) / iterations)
            assert output is not None
            result.append({
                "name": name,
                "median_ns": statistics.median(samples),
                "signature": float(output.sum().item()),
            })
    return result


def run_native(
    harness: pathlib.Path,
    mode: str,
    device: str,
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    command = [
        str(harness), mode, "--device", device, "--threads", str(threads),
        "--warmups", str(warmups), "--repetitions", str(repetitions),
        "--iterations", str(iterations),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("native harness produced no JSON")
    return json.loads(lines[-1])


def compare_records(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_by_name = {record["name"]: tensor_record(record["name"], record["value"]) for record in expected}
    actual_by_name = {record["name"]: record for record in actual}
    errors: list[dict[str, Any]] = []
    max_error = 0.0
    for name, left in expected_by_name.items():
        right = actual_by_name.get(name)
        if right is None or left["shape"] != right.get("shape"):
            errors.append({"name": name, "reason": "missing_or_shape"})
            continue
        left_values = torch.tensor(left["values"], dtype=torch.float64)
        right_values = torch.tensor(right["values"], dtype=torch.float64)
        error = float((left_values - right_values).abs().max().item()) if left_values.numel() else 0.0
        max_error = max(max_error, error)
        if not torch.allclose(left_values, right_values, rtol=2e-10, atol=2e-11):
            errors.append({"name": name, "reason": "values", "max_abs_error": error})
    return {
        "total": len(expected_by_name),
        "passed": len(expected_by_name) - len(errors),
        "max_abs_error": max_error,
        "errors": errors,
        "status": "PASS" if not errors and len(expected_by_name) == len(actual_by_name) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--python-model-source", type=pathlib.Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--warmups", type=positive, default=2)
    parser.add_argument("--repetitions", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=3)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    layers = load_layers(args.python_source, args.python_model_source)
    expected = python_records(layers, device)
    native = run_native(args.harness, "parity", args.device, args.threads,
                        args.warmups, args.repetitions, args.iterations)
    parity = compare_records(expected, native["records"])
    python_timing = python_benchmark(
        layers, device, args.warmups, args.repetitions, args.iterations
    )
    native_timing = run_native(
        args.harness, "benchmark", args.device, args.threads,
        args.warmups, args.repetitions, args.iterations
    )["records"]
    native_by_name = {record["name"]: record for record in native_timing}
    benchmark: dict[str, Any] = {}
    for record in python_timing:
        native_record = native_by_name[record["name"]]
        benchmark[record["name"]] = {
            "python_median_ns_per_iteration": record["median_ns"],
            "cpp_median_ns_per_iteration": native_record["median_ns"],
            "cpp_speedup": record["median_ns"] / native_record["median_ns"],
            "python_signature": record["signature"],
            "cpp_signature": native_record["signature"],
            "output": "PASS" if abs(record["signature"] - native_record["signature"]) < 2e-3 else "FAIL",
        }
    result = {
        "component": "message_passing",
        "device": device.type,
        "parity": parity,
        "benchmark": benchmark,
        "python_torch": torch.__version__,
        "status": "PASS" if parity["status"] == "PASS" and all(
            row["output"] == "PASS" for row in benchmark.values()
        ) else "FAIL",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
