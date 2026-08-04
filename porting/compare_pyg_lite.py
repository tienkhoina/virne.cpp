#!/usr/bin/env python3
"""PyG 2.7 differential oracle and compact benchmark for PyG-lite."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
    scatter,
    softmax,
    sort_edge_index,
    to_dense_batch,
)


EXPECTED_TORCH = "2.6.0"
EXPECTED_PYG = "2.7.0"
PARITY_RTOL = 2e-9
PARITY_ATOL = 2e-11
BENCHMARK_RTOL = 3e-5
BENCHMARK_ATOL = 3e-5


@dataclass(frozen=True)
class BenchmarkFixture:
    x: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    gcn: GCNConv
    gat: GATConv


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def core_version(value: str) -> str:
    return value.split("+", maxsplit=1)[0]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.bool: "bool",
        torch.int64: "int64",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    if dtype not in names:
        raise RuntimeError(f"unsupported comparator dtype: {dtype}")
    return names[dtype]


def tensor_record(name: str, source: torch.Tensor) -> dict[str, Any]:
    value = source.detach().to("cpu").contiguous()
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": dtype_name(value.dtype),
        "values": value.reshape(-1).tolist(),
    }


def fill_linear(tensor: torch.Tensor, first: float, step: float) -> None:
    with torch.no_grad():
        sequence = torch.arange(
            tensor.numel(), dtype=tensor.dtype, device=tensor.device
        ).reshape(tensor.shape)
        tensor.copy_(sequence * step + first)


def loss_scale(output: torch.Tensor) -> torch.Tensor:
    return torch.arange(
        output.numel(), dtype=output.dtype, device=output.device
    ).reshape(output.shape) * 0.019 - 0.17


def parity_records(device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    floating = {"dtype": torch.float64, "device": device}
    integer = {"dtype": torch.int64, "device": device}
    add = lambda name, value: records.append(tensor_record(name, value))

    first = Data(
        x=torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], **floating),
        edge_index=torch.tensor([[2, 0, 1], [0, 1, 2]], **integer),
        edge_attr=torch.tensor([[0.5], [1.0], [1.5]], **floating),
    )
    second = Data(
        x=torch.tensor([[7.0, 8.0], [9.0, -10.0]], **floating),
        edge_index=torch.tensor([[0, 1], [1, 0]], **integer),
        edge_attr=torch.tensor([[2.0], [2.5]], **floating),
    )
    packed = Batch.from_data_list([first, second])
    add("batch/x", packed.x)
    add("batch/edge_index", packed.edge_index)
    add("batch/edge_attr", packed.edge_attr)
    add("batch/batch", packed.batch)
    add("batch/ptr", packed.ptr)
    selected = packed.index_select(torch.tensor([1, 0], **integer))
    add("batch/select1_x", selected[0].x)
    add("batch/select1_edge_index", selected[0].edge_index)
    add("batch/select0_x", selected[1].x)
    add("batch/select0_edge_index", selected[1].edge_index)

    scatter_source = torch.tensor(
        [[2.0, -1.0], [4.0, 3.0], [-6.0, 5.0], [1.5, -2.0], [-2.0, 4.0]],
        **floating,
    )
    scatter_index = torch.tensor([0, 1, 0, 2, 1], **integer)
    add("scatter/sum", scatter(scatter_source, scatter_index, dim=0,
                               dim_size=4, reduce="sum"))
    add("scatter/mean", scatter(scatter_source, scatter_index, dim=0,
                                dim_size=4, reduce="mean"))
    add("scatter/min", scatter(scatter_source, scatter_index, dim=0,
                               dim_size=4, reduce="min"))
    add("scatter/max", scatter(scatter_source, scatter_index, dim=0,
                               dim_size=4, reduce="max"))
    add("scatter/prod", scatter(scatter_source, scatter_index, dim=0,
                                dim_size=4, reduce="mul"))
    flags = torch.tensor([False, True, True, False], dtype=torch.bool, device=device)
    add("scatter/any", scatter(
        flags, torch.tensor([0, 1, 2, 3], **integer), dim=0,
        dim_size=5, reduce="any"))

    sortable = torch.tensor(
        [[2, 0, 1, 0, 2, 1], [0, 2, 0, 1, 1, 2]], **integer
    )
    sortable_attr = torch.tensor([20, 2, 10, 1, 21, 12], **integer)
    row_key = sortable[0] * 3 + sortable[1]
    row_permutation = torch.argsort(row_key)
    row_edges, row_attrs = sort_edge_index(
        sortable, sortable_attr, num_nodes=3, sort_by_row=True
    )
    add("sort/row_edge_index", row_edges)
    add("sort/row_edge_attr", row_attrs)
    add("sort/row_permutation", row_permutation)
    column_key = sortable[1] * 3 + sortable[0]
    column_permutation = torch.argsort(column_key)
    column_edges, column_attrs = sort_edge_index(
        sortable, sortable_attr, num_nodes=3, sort_by_row=False
    )
    add("sort/column_edge_index", column_edges)
    add("sort/column_edge_attr", column_attrs)
    add("sort/column_permutation", column_permutation)
    duplicate_edges = torch.zeros((2, 100), **integer)
    duplicate_attr = torch.arange(100, **integer)
    _, duplicate_attrs = sort_edge_index(
        duplicate_edges, duplicate_attr, num_nodes=1, sort_by_row=True
    )
    add("sort/duplicate_edge_attr", duplicate_attrs)
    add("sort/duplicate_permutation", torch.argsort(duplicate_edges[0]))

    loop_edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [0, 0, 2, 2, 3, 1]], **integer
    )
    loop_attrs = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],
         [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        **floating,
    )
    removed_edges, removed_attrs = remove_self_loops(loop_edges, loop_attrs)
    add("loops/remove_edge_index", removed_edges)
    add("loops/remove_edge_attr", removed_attrs)
    scalar_edges, scalar_attrs = add_self_loops(
        loop_edges, loop_attrs, fill_value=2.25, num_nodes=4
    )
    add("loops/add_scalar_edge_index", scalar_edges)
    add("loops/add_scalar_edge_attr", scalar_attrs)
    _, mean_attrs = add_self_loops(
        loop_edges, loop_attrs, fill_value="mean", num_nodes=4
    )
    add("loops/add_mean_edge_attr", mean_attrs)
    remaining_edges, remaining_attrs = add_remaining_self_loops(
        loop_edges, loop_attrs, fill_value="mean", num_nodes=4
    )
    add("loops/remaining_edge_index", remaining_edges)
    add("loops/remaining_edge_attr", remaining_attrs)

    logits = torch.tensor(
        [[1.0, -1.0], [2.0, 2.0], [-0.5, 3.0], [4.0, 0.5], [1.5, -2.0]],
        **floating,
        requires_grad=True,
    )
    groups = torch.tensor([0, 0, 1, 0, 1], **integer)
    probabilities = softmax(logits, groups, num_nodes=3)
    add("softmax/output", probabilities)
    probability_weight = torch.tensor(
        [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0],
         [0.125, -0.5], [1.0, 0.75]],
        **floating,
    )
    (probabilities * probability_weight).sum().backward()
    add("softmax/input_grad", logits.grad)

    dense_x = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0],
         [5.0, 50.0], [6.0, 60.0], [7.0, 70.0]],
        **floating,
    )
    dense_batch = torch.tensor([0, 0, 0, 1, 1, 2, 2], **integer)
    dense, mask = to_dense_batch(
        dense_x, dense_batch, fill_value=-3.0, max_num_nodes=3, batch_size=4
    )
    add("dense/values", dense)
    add("dense/mask", mask)
    truncated, truncated_mask = to_dense_batch(
        dense_x, dense_batch, fill_value=0.0, max_num_nodes=2, batch_size=3
    )
    add("dense/truncated_values", truncated)
    add("dense/truncated_mask", truncated_mask)
    add("pool/add", global_add_pool(dense_x, dense_batch, size=4))
    add("pool/mean", global_mean_pool(dense_x, dense_batch, size=4))
    add("pool/max", global_max_pool(dense_x, dense_batch, size=4))

    norm_edges = torch.tensor(
        [[0, 1, 1, 2, 3], [1, 0, 2, 2, 1]], **integer
    )
    unit_edges, unit_weight = gcn_norm(
        norm_edges, None, num_nodes=4, improved=False,
        add_self_loops=True, flow="source_to_target", dtype=torch.float64
    )
    add("gcn_norm/unit_edge_index", unit_edges)
    add("gcn_norm/unit_weight", unit_weight)
    norm_weight = torch.tensor([0.5, 1.25, 2.0, 0.75, 1.5], **floating)
    weighted_edges, weighted_weight = gcn_norm(
        norm_edges, norm_weight, num_nodes=4, improved=True,
        add_self_loops=True, flow="source_to_target", dtype=torch.float64
    )
    add("gcn_norm/weighted_edge_index", weighted_edges)
    add("gcn_norm/weighted_weight", weighted_weight)

    gcn = GCNConv(4, 3, improved=False, cached=False, add_self_loops=True,
                  normalize=True, bias=True).to(device=device, dtype=torch.float64)
    fill_linear(gcn.lin.weight, -0.28, 0.035)
    fill_linear(gcn.bias, -0.09, 0.07)
    gcn_x = torch.tensor(
        [[1.0, -1.0, 0.5, 2.0], [0.25, 1.5, -2.0, 0.75],
         [-1.5, 0.0, 2.5, -0.5], [2.0, 1.0, -0.75, 1.25],
         [0.5, -2.0, 1.0, 3.0]],
        **floating,
        requires_grad=True,
    )
    gcn_edges = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 3], [1, 2, 3, 4, 0, 0, 1]], **integer
    )
    gcn_weight = torch.tensor(
        [0.5, 1.0, 1.5, 0.75, 1.25, 2.0, 0.625], **floating
    )
    gcn_output = gcn(gcn_x, gcn_edges, gcn_weight)
    add("gcn/output", gcn_output)
    (gcn_output * loss_scale(gcn_output)).sum().backward()
    add("gcn/input_grad", gcn_x.grad)
    add("gcn/lin_weight_grad", gcn.lin.weight.grad)
    add("gcn/bias_grad", gcn.bias.grad)

    gat = GATConv(
        4, 2, heads=2, concat=True, negative_slope=0.2, dropout=0.0,
        add_self_loops=True, edge_dim=2, fill_value="mean", bias=True,
        residual=True,
    ).to(device=device, dtype=torch.float64)
    fill_linear(gat.lin.weight, -0.21, 0.017)
    fill_linear(gat.att_src, -0.13, 0.029)
    fill_linear(gat.att_dst, 0.07, -0.023)
    fill_linear(gat.lin_edge.weight, -0.11, 0.031)
    fill_linear(gat.att_edge, 0.19, -0.027)
    fill_linear(gat.res.weight, -0.08, 0.021)
    fill_linear(gat.bias, 0.03, -0.025)
    gat_x = gcn_x.detach().clone().requires_grad_(True)
    gat_edges = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 3, 4], [1, 2, 3, 4, 0, 0, 1, 4]], **integer
    )
    gat_edge_attr = torch.tensor(
        [[0.5, 1.0], [1.5, -0.25], [0.75, 2.0], [-1.0, 0.5],
         [2.0, 1.25], [-0.5, 0.75], [1.0, -1.5], [0.25, 0.125]],
        **floating,
        requires_grad=True,
    )
    gat_output, (gat_result_edges, attention) = gat(
        gat_x, gat_edges, gat_edge_attr, return_attention_weights=True
    )
    add("gat/output", gat_output)
    add("gat/edge_index", gat_result_edges)
    add("gat/attention", attention)
    (gat_output * loss_scale(gat_output)).sum().backward()
    add("gat/input_grad", gat_x.grad)
    add("gat/edge_attr_grad", gat_edge_attr.grad)
    add("gat/lin_weight_grad", gat.lin.weight.grad)
    add("gat/att_src_grad", gat.att_src.grad)
    add("gat/att_dst_grad", gat.att_dst.grad)
    add("gat/lin_edge_weight_grad", gat.lin_edge.weight.grad)
    add("gat/att_edge_grad", gat.att_edge.grad)
    add("gat/res_weight_grad", gat.res.weight.grad)
    add("gat/bias_grad", gat.bias.grad)
    synchronize(device)
    return records


def make_benchmark_fixture(device: torch.device) -> BenchmarkFixture:
    graphs = 8
    nodes_per_graph = 500
    nodes = graphs * nodes_per_graph
    edges_per_graph = 7000
    edges = graphs * edges_per_graph
    node_ids = torch.arange(nodes, dtype=torch.float32, device=device).view(-1, 1)
    feature_ids = torch.arange(25, dtype=torch.float32, device=device).view(1, -1)
    x = torch.sin(node_ids * 0.013 + feature_ids * 0.071)
    edge_ids = torch.arange(edges, dtype=torch.int64, device=device)
    graph_ids = torch.div(edge_ids, edges_per_graph, rounding_mode="floor")
    local_ids = edge_ids.remainder(edges_per_graph)
    source = graph_ids * nodes_per_graph + local_ids.remainder(nodes_per_graph)
    target = graph_ids * nodes_per_graph + (
        local_ids * 37 + torch.div(local_ids, nodes_per_graph,
                                   rounding_mode="floor") + 1
    ).remainder(nodes_per_graph)
    edge_index = torch.stack((source, target), dim=0)
    edge_values = edge_ids.to(torch.float32)
    edge_attr = torch.stack(
        (torch.sin(edge_values * 0.017), torch.cos(edge_values * 0.011)), dim=1
    )
    batch = torch.div(
        torch.arange(nodes, dtype=torch.int64, device=device),
        nodes_per_graph,
        rounding_mode="floor",
    )
    gcn = GCNConv(25, 32, cached=True).to(device=device, dtype=torch.float32)
    fill_linear(gcn.lin.weight, -0.04, 0.00011)
    fill_linear(gcn.bias, -0.02, 0.001)
    gcn.eval()
    gat = GATConv(
        25, 8, heads=4, concat=True, dropout=0.0, edge_dim=2,
        residual=True,
    ).to(device=device, dtype=torch.float32)
    fill_linear(gat.lin.weight, -0.035, 0.00009)
    fill_linear(gat.att_src, -0.025, 0.0013)
    fill_linear(gat.att_dst, 0.019, -0.0011)
    fill_linear(gat.lin_edge.weight, -0.03, 0.0007)
    fill_linear(gat.att_edge, 0.02, -0.0012)
    fill_linear(gat.res.weight, -0.025, 0.00008)
    fill_linear(gat.bias, 0.01, -0.0005)
    gat.eval()
    return BenchmarkFixture(x, batch, edge_index, edge_attr, gcn, gat)


def signature(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten()
    count = min(16, flat.numel())
    as_double = value.to(torch.float64)
    return torch.cat((
        flat[:count].to(torch.float64),
        as_double.sum().reshape(1),
        as_double.square().sum().reshape(1),
    )).cpu().contiguous()


def measure(
    name: str,
    function: Callable[[], torch.Tensor],
    warmups: int,
    repetitions: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    output: torch.Tensor | None = None
    for _ in range(warmups):
        for _ in range(iterations):
            output = function()
    synchronize(device)
    samples: list[int] = []
    for _ in range(repetitions):
        synchronize(device)
        started = time.perf_counter_ns()
        for _ in range(iterations):
            output = function()
        synchronize(device)
        samples.append(time.perf_counter_ns() - started)
    if output is None:
        raise RuntimeError(f"benchmark {name} did not produce output")
    return {
        "name": name,
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "signature": tensor_record("signature", signature(output)),
    }


def benchmark_records(
    device: torch.device,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> list[dict[str, Any]]:
    fixture = make_benchmark_fixture(device)

    def utility_pipeline() -> torch.Tensor:
        sorted_edges, sorted_attr = sort_edge_index(
            fixture.edge_index, fixture.edge_attr,
            num_nodes=fixture.x.shape[0], sort_by_row=True
        )
        loop_edges, loop_attr = add_remaining_self_loops(
            sorted_edges, sorted_attr, fill_value="mean",
            num_nodes=fixture.x.shape[0]
        )
        probabilities = softmax(
            loop_attr, loop_edges[1], num_nodes=fixture.x.shape[0]
        )
        dense, mask = to_dense_batch(
            fixture.x, fixture.batch, fill_value=0.0,
            max_num_nodes=500, batch_size=8
        )
        pooled = global_mean_pool(fixture.x, fixture.batch, size=8)
        return torch.cat((
            probabilities.flatten()[:32],
            dense.flatten()[:32],
            mask.to(torch.float32).sum().reshape(1),
            pooled.flatten(),
        ))

    with torch.inference_mode():
        return [
            measure("utility_pipeline", utility_pipeline, warmups,
                    repetitions, iterations, device),
            measure("gcn", lambda: fixture.gcn(fixture.x, fixture.edge_index),
                    warmups, repetitions, iterations, device),
            measure(
                "gat",
                lambda: fixture.gat(
                    fixture.x, fixture.edge_index, fixture.edge_attr
                ),
                warmups,
                repetitions,
                iterations,
                device,
            ),
        ]


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
        str(harness), mode, "--device", device, "--threads", str(threads)
    ]
    if mode == "benchmark":
        command.extend([
            "--warmups", str(warmups),
            "--repetitions", str(repetitions),
            "--iterations", str(iterations),
        ])
    process = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"PyG-lite harness {mode} failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one native JSON line, got {lines!r}")
    report = json.loads(lines[0])
    if not isinstance(report, dict) or report.get("mode") != mode:
        raise RuntimeError(f"invalid native {mode} report")
    return report


def record_tensor(record: dict[str, Any]) -> torch.Tensor:
    dtypes = {
        "bool": torch.bool,
        "int64": torch.int64,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtypes.get(record.get("dtype"))
    if dtype is None:
        raise RuntimeError(f"invalid native tensor dtype: {record.get('dtype')!r}")
    shape = record.get("shape")
    values = record.get("values")
    if not isinstance(shape, list) or not isinstance(values, list):
        raise RuntimeError("invalid native tensor record")
    return torch.tensor(values, dtype=dtype).reshape(shape)


def is_close_case(name: str) -> bool:
    if name.startswith("softmax/") or name.startswith("gcn/"):
        return True
    if name.startswith("gat/") and name != "gat/edge_index":
        return True
    return name in {"gcn_norm/unit_weight", "gcn_norm/weighted_weight"}


def compare_tensor_records(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    require_exact: bool,
) -> tuple[list[dict[str, Any]], int, int, float]:
    mismatches: list[dict[str, Any]] = []
    if len(expected) != len(actual):
        mismatches.append({
            "name": "records",
            "reason": "count",
            "python": len(expected),
            "cpp": len(actual),
        })
    passed = 0
    bit_exact = 0
    max_abs_error = 0.0
    for expected_record, actual_record in zip(expected, actual):
        name = expected_record["name"]
        if actual_record.get("name") != name:
            mismatches.append({
                "name": name,
                "reason": "order_or_name",
                "cpp": actual_record.get("name"),
            })
            continue
        expected_tensor = record_tensor(expected_record)
        actual_tensor = record_tensor(actual_record)
        if expected_tensor.shape != actual_tensor.shape:
            mismatches.append({
                "name": name,
                "reason": "shape",
                "python": list(expected_tensor.shape),
                "cpp": list(actual_tensor.shape),
            })
            continue
        if expected_tensor.dtype != actual_tensor.dtype:
            mismatches.append({
                "name": name,
                "reason": "dtype",
                "python": str(expected_tensor.dtype),
                "cpp": str(actual_tensor.dtype),
            })
            continue
        close_case = is_close_case(name)
        exact = torch.equal(expected_tensor, actual_tensor)
        if exact:
            bit_exact += 1
        if expected_tensor.is_floating_point() and expected_tensor.numel():
            max_abs_error = max(
                max_abs_error,
                float((expected_tensor - actual_tensor).abs().max().item()),
            )
        equal = exact or (
            close_case
            and not require_exact
            and torch.allclose(
                expected_tensor, actual_tensor,
                rtol=PARITY_RTOL, atol=PARITY_ATOL,
            )
        )
        if equal:
            passed += 1
            continue
        difference = 0.0
        if expected_tensor.is_floating_point():
            difference = float((expected_tensor - actual_tensor).abs().max().item())
        mismatches.append({
            "name": name,
            "reason": "allclose" if close_case else "exact",
            "max_abs_error": difference,
        })
    return mismatches, passed, bit_exact, max_abs_error


def compare_benchmarks(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if len(expected) != len(actual):
        mismatches.append({"reason": "benchmark_count"})
    for python_record, cpp_record in zip(expected, actual):
        name = python_record["name"]
        if cpp_record.get("name") != name:
            mismatches.append({
                "name": name,
                "reason": "benchmark_order_or_name",
                "cpp": cpp_record.get("name"),
            })
            continue
        python_signature = record_tensor(python_record["signature"])
        cpp_signature = record_tensor(cpp_record["signature"])
        signature_ok = (
            python_signature.shape == cpp_signature.shape
            and torch.allclose(
                python_signature, cpp_signature,
                rtol=BENCHMARK_RTOL, atol=BENCHMARK_ATOL,
            )
        )
        if not signature_ok:
            mismatches.append({
                "name": name,
                "reason": "benchmark_signature",
                "max_abs_error": float(
                    (python_signature - cpp_signature).abs().max().item()
                ) if python_signature.shape == cpp_signature.shape else None,
            })
        python_median = float(python_record["median_ns"])
        cpp_median = float(cpp_record["median_ns"])
        summary[name] = {
            "python_median_ns_per_iteration": python_median / iterations,
            "cpp_median_ns_per_iteration": cpp_median / iterations,
            "cpp_speedup": python_median / cpp_median,
            "output": "PASS" if signature_ok else "FAIL",
        }
    return mismatches, summary


def interleaved_benchmarks(
    harness: pathlib.Path,
    device: torch.device,
    native_device: str,
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rotate Python/native sample order so thermal drift cannot pick a winner."""
    python_samples: dict[str, list[int]] = {}
    native_samples: dict[str, list[int]] = {}
    python_signatures: dict[str, dict[str, Any]] = {}
    native_signatures: dict[str, dict[str, Any]] = {}

    def collect(
        records: list[dict[str, Any]],
        samples: dict[str, list[int]],
        signatures: dict[str, dict[str, Any]],
    ) -> None:
        for record in records:
            name = str(record["name"])
            values = record.get("samples_ns", [])
            if len(values) != 1:
                raise RuntimeError("interleaved benchmark requires one native sample")
            samples.setdefault(name, []).append(int(values[0]))
            signature_record = record["signature"]
            if name in signatures:
                if not torch.equal(
                    record_tensor(signatures[name]),
                    record_tensor(signature_record),
                ):
                    raise RuntimeError(f"benchmark signature drift: {name}")
            else:
                signatures[name] = signature_record

    for repetition in range(repetitions):
        if repetition % 2 == 0:
            collect(
                benchmark_records(device, warmups, 1, iterations),
                python_samples,
                python_signatures,
            )
            native = run_native(
                harness, "benchmark", native_device, threads,
                warmups, 1, iterations,
            )
            collect(native.get("benchmarks", []), native_samples, native_signatures)
        else:
            native = run_native(
                harness, "benchmark", native_device, threads,
                warmups, 1, iterations,
            )
            collect(native.get("benchmarks", []), native_samples, native_signatures)
            collect(
                benchmark_records(device, warmups, 1, iterations),
                python_samples,
                python_signatures,
            )

    def aggregate(
        samples: dict[str, list[int]],
        signatures: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "samples_ns": values,
                "median_ns": statistics.median(values),
                "signature": signatures[name],
            }
            for name, values in samples.items()
        ]

    return (
        aggregate(python_samples, python_signatures),
        aggregate(native_samples, native_signatures),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("parity", "benchmark", "all"),
                        default="all")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                        default="cpu")
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--warmups", type=positive, default=2)
    parser.add_argument("--repetitions", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=3)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    python_torch = core_version(torch.__version__)
    python_pyg = core_version(torch_geometric.__version__)
    if python_torch != EXPECTED_TORCH or python_pyg != EXPECTED_PYG:
        raise RuntimeError(
            f"oracle version drift: torch={torch.__version__}, "
            f"pyg={torch_geometric.__version__}; expected "
            f"torch={EXPECTED_TORCH}, pyg={EXPECTED_PYG}"
        )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.manual_seed(0)
    device = resolve_device(args.device)
    resolved_name = device.type

    report: dict[str, Any] = {
        "component": "pyg_lite",
        "status": "PASS",
        "device": resolved_name,
        "threads": args.threads,
        "python_torch": torch.__version__,
        "python_pyg": torch_geometric.__version__,
    }
    all_mismatches: list[dict[str, Any]] = []

    if args.mode in {"parity", "all"}:
        expected = parity_records(device)
        native = run_native(
            args.harness, "parity", args.device, args.threads,
            args.warmups, args.repetitions, args.iterations,
        )
        native_version = core_version(str(native.get("libtorch_version", "unknown")))
        if native_version != "unknown" and native_version != EXPECTED_TORCH:
            all_mismatches.append({
                "name": "libtorch_version",
                "python": EXPECTED_TORCH,
                "cpp": native_version,
            })
        if native.get("device") != resolved_name:
            all_mismatches.append({
                "name": "device",
                "python": resolved_name,
                "cpp": native.get("device"),
            })
        parity_mismatches, passed, bit_exact, max_abs_error = compare_tensor_records(
            expected,
            native.get("tensors", []),
            require_exact=device.type == "cpu",
        )
        all_mismatches.extend(parity_mismatches)
        report["parity"] = {
            "passed": passed,
            "bit_exact": bit_exact,
            "max_abs_error": max_abs_error,
            "exact_required": device.type == "cpu",
            "total": len(expected),
            "status": "PASS" if not parity_mismatches else "FAIL",
        }

    if args.mode in {"benchmark", "all"}:
        python_benchmarks, native_benchmarks = interleaved_benchmarks(
            args.harness,
            device,
            args.device,
            args.threads,
            args.warmups,
            args.repetitions,
            args.iterations,
        )
        benchmark_mismatches, benchmark_summary = compare_benchmarks(
            python_benchmarks,
            native_benchmarks,
            args.iterations,
        )
        all_mismatches.extend(benchmark_mismatches)
        report["benchmark"] = benchmark_summary

    if all_mismatches:
        report["status"] = "FAIL"
        report["mismatches"] = all_mismatches[:32]
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(encoded)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
