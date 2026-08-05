#!/usr/bin/env python3
"""Differential oracle and one prepared graph-sampler benchmark."""

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


FIXTURE_VERSION = 1
EXPECTED_ORACLE_SHA256 = (
    "349ad2c398a1a89c1717202a4d7b424da9c3491d51193e7c86ef7d1550f6e699"
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1

NODE_NAMES = ("p", "v")
RELATION_KEYS = (
    ("p", "connect", "p"),
    ("v", "connect", "v"),
    ("v", "mapping", "p"),
    ("p", "mapping", "v"),
    ("v", "imaginary", "p"),
    ("p", "imaginary", "v"),
)
RELATION_NAMES = (
    "p_connect_p",
    "v_connect_v",
    "v_mapping_p",
    "p_mapping_v",
    "v_imaginary_p",
    "p_imaginary_v",
)
RELATION_ENDPOINTS = ((0, 0), (1, 1), (1, 0), (0, 1), (1, 0), (0, 1))


class _StubData:
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def to(self, device: torch.device | str) -> "_StubData":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        return self


class _StubRelation:
    def __init__(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> None:
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def to(self, device: torch.device | str) -> "_StubRelation":
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        return self


class _StubHeteroData:
    def __init__(
        self,
        x: list[torch.Tensor],
        relations: list[_StubRelation],
    ) -> None:
        self.x = x
        self.relations = relations

    def to(self, device: torch.device | str) -> "_StubHeteroData":
        self.x = [value.to(device) for value in self.x]
        for relation in self.relations:
            relation.to(device)
        return self


def _collate_homogeneous(graphs: list[_StubData]) -> dict[str, Any]:
    xs: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    edge_attrs: list[torch.Tensor] = []
    batches: list[torch.Tensor] = []
    ptr = [0]
    edge_ptr = [0]
    node_offset = 0
    for graph_id, graph in enumerate(graphs):
        xs.append(graph.x)
        edges.append(graph.edge_index + node_offset)
        edge_attrs.append(graph.edge_attr)
        batches.append(torch.full(
            (graph.x.shape[0],), graph_id,
            dtype=torch.int64, device=graph.x.device,
        ))
        node_offset += int(graph.x.shape[0])
        ptr.append(node_offset)
        edge_ptr.append(edge_ptr[-1] + int(graph.edge_index.shape[1]))
    device = graphs[0].x.device
    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batches, dim=0),
        "ptr": torch.tensor(ptr, dtype=torch.int64, device=device),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.int64, device=device),
        "num_graphs": len(graphs),
        "max_num_nodes": max(int(graph.x.shape[0]) for graph in graphs),
    }


def _collate_heterogeneous(graphs: list[_StubHeteroData]) -> dict[str, Any]:
    node_ptr: list[list[int]] = [[0], [0]]
    node_x: list[list[torch.Tensor]] = [[], []]
    node_batch: list[list[torch.Tensor]] = [[], []]
    for graph_id, graph in enumerate(graphs):
        for node_type in range(2):
            value = graph.x[node_type]
            node_x[node_type].append(value)
            node_batch[node_type].append(torch.full(
                (value.shape[0],), graph_id,
                dtype=torch.int64, device=value.device,
            ))
            node_ptr[node_type].append(node_ptr[node_type][-1] + int(value.shape[0]))

    nodes: list[dict[str, torch.Tensor | int]] = []
    for node_type in range(2):
        device = graphs[0].x[node_type].device
        ptr = torch.tensor(node_ptr[node_type], dtype=torch.int64, device=device)
        nodes.append({
            "x": torch.cat(node_x[node_type], dim=0),
            "batch": torch.cat(node_batch[node_type], dim=0),
            "ptr": ptr,
            "node_ptr": ptr,
            "max_num_nodes": max(
                int(graph.x[node_type].shape[0]) for graph in graphs
            ),
        })

    relations: list[dict[str, torch.Tensor]] = []
    for relation_id, (source_type, target_type) in enumerate(RELATION_ENDPOINTS):
        edges: list[torch.Tensor] = []
        attrs: list[torch.Tensor] = []
        for graph_id, graph in enumerate(graphs):
            relation = graph.relations[relation_id]
            offset = torch.tensor(
                [[node_ptr[source_type][graph_id]],
                 [node_ptr[target_type][graph_id]]],
                dtype=torch.int64,
                device=relation.edge_index.device,
            )
            edges.append(relation.edge_index + offset)
            attrs.append(relation.edge_attr)
        relations.append({
            "edge_index": torch.cat(edges, dim=1),
            "edge_attr": torch.cat(attrs, dim=0),
        })
    return {"nodes": nodes, "relations": relations, "num_graphs": len(graphs)}


def _move_nested(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    return value


class _StubBatch:
    """Minimal PyG Batch boundary required by the extracted Virne function."""

    def __init__(self, graphs: list[_StubData | _StubHeteroData]) -> None:
        if not graphs:
            raise ValueError("oracle Batch requires at least one graph")
        self.graphs = list(graphs)
        if isinstance(graphs[0], _StubHeteroData):
            self.collated = _collate_heterogeneous(graphs)  # type: ignore[arg-type]
        else:
            self.collated = _collate_homogeneous(graphs)  # type: ignore[arg-type]

    @classmethod
    def from_data_list(
        cls, graphs: list[_StubData | _StubHeteroData]
    ) -> "_StubBatch":
        return cls(graphs)

    def index_select(
        self, indices: torch.Tensor
    ) -> list[_StubData | _StubHeteroData]:
        return [
            self.graphs[int(index)]
            for index in indices.detach().to("cpu").tolist()
        ]

    def to(self, device: torch.device | str) -> "_StubBatch":
        for graph in self.graphs:
            graph.to(device)
        self.collated = _move_nested(self.collated, device)
        return self


try:
    from torch_geometric.data import Batch as _PygBatch
    from torch_geometric.data import Data as _PygData
    from torch_geometric.data import HeteroData as _PygHeteroData
except ImportError:
    _PygBatch = None
    _PygData = None
    _PygHeteroData = None

HAS_PYG = _PygBatch is not None
ORACLE_BACKEND = "torch_geometric" if HAS_PYG else "minimal_batch_stub"
ORACLE_BATCH = _PygBatch if HAS_PYG else _StubBatch


def load_oracle_get_observations_sample() -> tuple[Any, pathlib.Path, str, str]:
    source_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
        / "utils.py"
    )
    source_bytes = source_path.read_bytes()
    # Path.read_text() normalizes CRLF, so the pinned AST segment is stable
    # across the Windows host and the Linux comparison container.
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text, source_path)
    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_observations_sample"
    ]
    if len(functions) != 1:
        raise RuntimeError("expected exactly one get_observations_sample function")
    function = functions[0]
    function_text = ast.get_source_segment(source_text, function)
    if function_text is None:
        raise RuntimeError("cannot isolate get_observations_sample source")
    function_hash = hashlib.sha256(function_text.encode("utf-8")).hexdigest()
    if function_hash != EXPECTED_ORACLE_SHA256:
        raise RuntimeError(
            "get_observations_sample oracle SHA256 changed: "
            f"expected {EXPECTED_ORACLE_SHA256}, got {function_hash}"
        )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"Batch": ORACLE_BATCH}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return (
        namespace["get_observations_sample"],
        source_path,
        function_hash,
        hashlib.sha256(source_bytes).hexdigest(),
    )


(
    ORACLE_GET_OBSERVATIONS_SAMPLE,
    ORACLE_SOURCE,
    ORACLE_SOURCE_SHA256,
    ORACLE_FILE_SHA256,
) = load_oracle_get_observations_sample()


def make_graph(
    seed: int,
    nodes: int,
    device: torch.device,
) -> Any:
    x = torch.arange(
        seed * 100, seed * 100 + nodes * 2,
        dtype=torch.float32, device=device,
    ).reshape(nodes, 2)
    edge_values: list[int] = []
    for node in range(nodes - 1):
        edge_values.extend((node, node + 1, node + 1, node))
    edge_index = (
        torch.tensor(edge_values, dtype=torch.int64, device=device)
        .reshape(-1, 2).t().contiguous()
        if edge_values
        else torch.empty((2, 0), dtype=torch.int64, device=device)
    )
    edge_attr = torch.arange(
        edge_index.shape[1], dtype=torch.float32, device=device
    ).reshape(-1, 1).add_(float(seed))
    if HAS_PYG:
        return _PygData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return _StubData(x, edge_index, edge_attr)


def make_relation(
    edges: list[int], seed: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = (
        torch.tensor(edges, dtype=torch.int64, device=device)
        .reshape(-1, 2).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.int64, device=device)
    )
    edge_attr = torch.arange(
        edge_index.shape[1], dtype=torch.float32, device=device
    ).reshape(-1, 1).add_(seed)
    return edge_index, edge_attr


def make_hetero_graph(
    seed: int,
    p_nodes: int,
    v_nodes: int,
    augment_virtual: bool,
    augment_physical: bool,
    device: torch.device,
) -> Any:
    node_x = [
        torch.arange(
            seed * 100, seed * 100 + p_nodes * 2,
            dtype=torch.float32, device=device,
        ).reshape(p_nodes, 2),
        torch.arange(
            seed * 10, seed * 10 + v_nodes,
            dtype=torch.float32, device=device,
        ).reshape(v_nodes, 1),
    ]
    p_edges: list[int] = []
    for node in range(p_nodes - 1):
        p_edges.extend((node, node + 1, node + 1, node))
    if augment_physical and p_nodes:
        p_edges.extend((0, 0))
    v_edges: list[int] = []
    for node in range(v_nodes - 1):
        v_edges.extend((node, node + 1, node + 1, node))
    if augment_virtual and v_nodes:
        v_edges.extend((0, 0))
    mapping = [0, 0] if p_nodes and v_nodes and seed != 2 else []
    imaginary = [v_nodes - 1, p_nodes - 1] if (
        p_nodes and v_nodes and seed != 1
    ) else []
    reverse_mapping = [value for edge in zip(mapping[1::2], mapping[::2]) for value in edge]
    reverse_imaginary = [
        value for edge in zip(imaginary[1::2], imaginary[::2]) for value in edge
    ]
    edge_values = (
        p_edges, v_edges, mapping, reverse_mapping, imaginary, reverse_imaginary
    )
    edge_seeds = (
        seed + 0.1, seed + 0.2, seed + 0.3,
        seed + 0.3, seed + 0.4, seed + 0.4,
    )
    relation_values = [
        make_relation(list(edges), float(relation_seed), device)
        for edges, relation_seed in zip(edge_values, edge_seeds)
    ]
    if HAS_PYG:
        result = _PygHeteroData()
        for name, value in zip(NODE_NAMES, node_x):
            result[name].x = value
        for key, (edge_index, edge_attr) in zip(RELATION_KEYS, relation_values):
            result[key].edge_index = edge_index
            result[key].edge_attr = edge_attr
        return result
    return _StubHeteroData(
        node_x,
        [_StubRelation(edge_index, edge_attr)
         for edge_index, edge_attr in relation_values],
    )


def make_batch(graphs: list[Any]) -> Any:
    return ORACLE_BATCH.from_data_list(graphs)


def homogeneous_values(batch: Any) -> dict[str, Any]:
    if not HAS_PYG:
        return batch.collated
    graph_list = batch.to_data_list()
    edge_ptr = [0]
    max_nodes = 0
    for graph in graph_list:
        edge_ptr.append(edge_ptr[-1] + int(graph.edge_index.shape[1]))
        max_nodes = max(max_nodes, int(graph.x.shape[0]))
    return {
        "x": batch.x,
        "edge_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "batch": batch.batch,
        "ptr": batch.ptr,
        "edge_ptr": torch.tensor(
            edge_ptr, dtype=torch.int64, device=batch.x.device
        ),
        "num_graphs": int(batch.num_graphs),
        "max_num_nodes": max_nodes,
    }


def heterogeneous_values(batch: Any) -> dict[str, Any]:
    if not HAS_PYG:
        return batch.collated
    nodes: list[dict[str, Any]] = []
    for name in NODE_NAMES:
        store = batch[name]
        ptr = store.ptr
        nodes.append({
            "x": store.x,
            "batch": store.batch,
            "ptr": ptr,
            "node_ptr": ptr,
            "max_num_nodes": int((ptr[1:] - ptr[:-1]).max().item()),
        })
    relations = [
        {
            "edge_index": batch[key].edge_index,
            "edge_attr": batch[key].edge_attr,
        }
        for key in RELATION_KEYS
    ]
    return {"nodes": nodes, "relations": relations, "num_graphs": int(batch.num_graphs)}


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
        raise TypeError(f"unexpected graph sampler dtype: {array.dtype}")
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


def append_case(
    output: list[dict[str, Any]],
    name: str,
    value: torch.Tensor,
    include_bits: bool = True,
) -> None:
    output.append({"name": name, "output": tensor_summary(value, include_bits)})


def append_batch_cases(
    prefix: str,
    batch: Any,
    output: list[dict[str, Any]],
    include_bits: bool = True,
) -> None:
    values = homogeneous_values(batch)
    for field in ("x", "edge_index", "edge_attr", "batch", "ptr", "edge_ptr"):
        append_case(output, f"{prefix}.{field}", values[field], include_bits)
    device = values["x"].device
    append_case(output, f"{prefix}.num_graphs", torch.tensor(
        [values["num_graphs"]], dtype=torch.int64, device=device
    ), include_bits)
    append_case(output, f"{prefix}.max_num_nodes", torch.tensor(
        [values["max_num_nodes"]], dtype=torch.int64, device=device
    ), include_bits)


def append_hetero_cases(
    prefix: str,
    batch: Any,
    output: list[dict[str, Any]],
) -> None:
    values = heterogeneous_values(batch)
    for node_name, node in zip(NODE_NAMES, values["nodes"]):
        for field in ("x", "batch", "ptr", "node_ptr"):
            append_case(output, f"{prefix}.{node_name}.{field}", node[field])
    for relation_name, relation in zip(RELATION_NAMES, values["relations"]):
        append_case(
            output, f"{prefix}.relation.{relation_name}.edge_index",
            relation["edge_index"],
        )
        append_case(
            output, f"{prefix}.relation.{relation_name}.edge_attr",
            relation["edge_attr"],
        )
    device = values["nodes"][0]["x"].device
    append_case(output, f"{prefix}.num_graphs", torch.tensor(
        [values["num_graphs"]], dtype=torch.int64, device=device
    ))
    append_case(output, f"{prefix}.max_num_nodes.p", torch.tensor(
        [values["nodes"][0]["max_num_nodes"]], dtype=torch.int64, device=device
    ))
    append_case(output, f"{prefix}.max_num_nodes.v", torch.tensor(
        [values["nodes"][1]["max_num_nodes"]], dtype=torch.int64, device=device
    ))


def exact_cases(device: torch.device) -> list[dict[str, Any]]:
    indices = torch.tensor([2, 0, 2, -2], dtype=torch.int64, device=device)
    p_batch = make_batch([make_graph(1, 2, device), make_graph(2, 4, device),
                          make_graph(3, 3, device)])
    homogeneous = ORACLE_GET_OBSERVATIONS_SAMPLE(
        {"graph": p_batch}, indices, device
    )["graph"]

    v_batch = make_batch([make_graph(11, 1, device), make_graph(12, 3, device),
                          make_graph(13, 2, device)])
    dual = ORACLE_GET_OBSERVATIONS_SAMPLE({
        "p_net": p_batch,
        "v_net": v_batch,
        "curr_v_node_id": torch.tensor([5, 6, 7], dtype=torch.int64, device=device),
        "action_mask": torch.tensor(
            [[1, 0], [0, 1], [1, 1]], dtype=torch.bool, device=device
        ),
        "v_net_size": torch.tensor([2.0, 4.0, 3.0], dtype=torch.float32, device=device),
    }, indices, device)

    p_counts = (2, 4, 3)
    v_counts = (1, 3, 2)
    base = make_batch([
        make_hetero_graph(i, p_counts[i], v_counts[i], False, False, device)
        for i in range(3)
    ])
    virtual_augmented = make_batch([
        make_hetero_graph(i, p_counts[i], v_counts[i], True, False, device)
        for i in range(3)
    ])
    physical_augmented = make_batch([
        make_hetero_graph(i, p_counts[i], v_counts[i], False, True, device)
        for i in range(3)
    ])
    hetero = ORACLE_GET_OBSERVATIONS_SAMPLE({
        "hetero_data": base,
        "aug_hetero_data_a": virtual_augmented,
        "aug_hetero_data_b": physical_augmented,
        "curr_v_node_id": torch.tensor([0, 1, 2], dtype=torch.int64, device=device),
        "action_mask": torch.ones((3, 4), dtype=torch.bool, device=device),
        "v_net_size": torch.tensor(
            [1.0, 3.0, 2.0], dtype=torch.float32, device=device
        ).reshape(3, 1),
    }, indices, device)

    output: list[dict[str, Any]] = []
    append_batch_cases("homogeneous", homogeneous, output)
    append_batch_cases("dual.p_net", dual["p_net"], output)
    append_batch_cases("dual.v_net", dual["v_net"], output)
    append_case(output, "dual.general.curr_v_node_id", dual["curr_v_node_id"])
    append_case(output, "dual.general.action_mask", dual["action_mask"])
    append_case(output, "dual.general.v_net_size", dual["v_net_size"])
    append_hetero_cases("hetero.base", hetero["hetero_data"], output)
    append_hetero_cases(
        "hetero.virtual_augmented", hetero["aug_hetero_data_a"], output
    )
    append_hetero_cases(
        "hetero.physical_augmented", hetero["aug_hetero_data_b"], output
    )
    append_case(output, "hetero.general.curr_v_node_id", hetero["curr_v_node_id"])
    append_case(output, "hetero.general.action_mask", hetero["action_mask"])
    append_case(output, "hetero.general.v_net_size", hetero["v_net_size"])
    return output


def benchmark_python(
    device: torch.device,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    graph_count = 256
    source = make_batch([
        make_graph(graph_id + 1, 24 + graph_id % 9, device)
        for graph_id in range(graph_count)
    ])
    indices = torch.tensor(
        [(index * 37) % graph_count for index in range(128)],
        dtype=torch.int64,
        device=device,
    )

    def sample() -> Any:
        return ORACLE_GET_OBSERVATIONS_SAMPLE(
            {"graph": source}, indices, device
        )["graph"]

    sampled: Any = None
    with torch.inference_mode():
        for _ in range(warmups):
            for _ in range(iterations):
                sampled = sample()
        samples_ns: list[int] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            for _ in range(iterations):
                sampled = sample()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ns.append(time.perf_counter_ns() - started)
    output: list[dict[str, Any]] = []
    append_batch_cases("sample", sampled, output, include_bits=False)
    return {
        "name": "prepared_homogeneous_256_to_128",
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": output,
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


def run_native(args: argparse.Namespace, benchmark: bool) -> dict[str, Any]:
    command = [
        str(args.native),
        "--device", args.device,
        "--threads", str(args.threads),
        "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions),
        "--iterations", str(args.iterations),
    ]
    if benchmark:
        command.append("--benchmark")
    process = subprocess.run(
        command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"native graph observation sampler failed: {process.stderr.strip()}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines")
    return json.loads(lines[0])


def write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "component": report["component"],
        "status": report["status"],
        "exact_cases": report["exact_cases"],
        "oracle_backend": report["oracle_backend"],
        "benchmark": report["benchmark"],
    }, sort_keys=True, separators=(",", ":")))


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

    expected = {
        "component": "graph_observation_sampler",
        "device": device.type,
        "threads": args.threads,
        "cases": exact_cases(device),
    }
    # Exact is a distinct native run. No benchmark code is reached on failure.
    native_exact = run_native(args, benchmark=False)
    mismatches = compare(expected, native_exact)
    report: dict[str, Any] = {
        "component": "graph_observation_sampler",
        "fixture_version": FIXTURE_VERSION,
        "python_source": str(ORACLE_SOURCE),
        "python_oracle_sha256": ORACLE_SOURCE_SHA256,
        "python_file_sha256": ORACLE_FILE_SHA256,
        "oracle_backend": ORACLE_BACKEND,
        "status": "FAIL" if mismatches else "PASS",
        "exact_cases": "FAIL" if mismatches else f"{len(expected['cases'])}/{len(expected['cases'])}",
        "mismatches": mismatches,
        "benchmark_mismatches": [],
        "benchmark": "DEFERRED_UNTIL_EXACT",
    }
    if mismatches:
        write_report(args, report)
        return 1

    if args.benchmark:
        native_with_benchmark = run_native(args, benchmark=True)
        repeated_exact = {
            key: value for key, value in native_with_benchmark.items()
            if key != "benchmark"
        }
        repeated_mismatches = compare(expected, repeated_exact, "repeat_exact")
        if repeated_mismatches:
            report["status"] = "FAIL"
            report["mismatches"] = repeated_mismatches
            write_report(args, report)
            return 1
        native_benchmark = native_with_benchmark.get("benchmark")
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
        report["benchmark_mismatches"] = benchmark_mismatches
        report["benchmark"] = {
            "name": python_benchmark["name"],
            "native_scope": "sample(prepared_selection)",
            "python_scope": "AST oracle get_observations_sample",
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
                item["name"]: item["output"]["checksum"]
                for item in python_benchmark["output"]
            },
        }
        if benchmark_mismatches:
            report["status"] = "FAIL"

    write_report(args, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
