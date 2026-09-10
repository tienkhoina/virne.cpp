#!/usr/bin/env python3
"""One exact gate and one aggregate benchmark for remaining neural leaves."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    NNConv,
    PNAConv,
    SAGEConv,
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


EXPECTED_FILE_SHA256 = {
    "multipler.py": "66b017324fec8493d7570d53224c2d8bfbfa846a0befa78b191b15e00fafecd6",
    "loss.py": "ba5eafe4fee8d5483360210ceaf87ab966f4df1e2ce80a32ca2d6be95e7e1e56",
    "res.py": "365b78be46f90ce2a0c2acc3dba8a1e2cc99628348446b5d6d5b2069a4e8206c",
    "sinkhorn.py": "e98026d712043e9e1b0353772799f04a5707588f4db65b7702fa9690920a5606",
    "gnn.py": "4ba777bafa5a104bfad113440e62d1153e63ee12dc78144374575c0e2202863c",
    "graph_conv.py": "5d787b98c535cacdfb10b60e5f779f55d9fa05819862b03895d6bde437dca315",
}
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def source_dir() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
        / "neural_network"
    )


def verify_sources() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, expected in EXPECTED_FILE_SHA256.items():
        path = source_dir() / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"{name} source drift: expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def load_file_module(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gnn_oracle(graph_conv: ModuleType) -> dict[str, Any]:
    path = source_dir() / "gnn.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "scatter": scatter,
        "MessagePassing": MessagePassing,
        "GCNConv": GCNConv,
        "GATConv": GATConv,
        "PNAConv": PNAConv,
        "NNConv": NNConv,
        "SAGEConv": SAGEConv,
        "global_add_pool": global_add_pool,
        "global_max_pool": global_max_pool,
        "global_mean_pool": global_mean_pool,
        "Data": Data,
        "Batch": Batch,
        "to_dense_batch": to_dense_batch,
        "EdgeFusionGATConv": graph_conv.EdgeFusionGATConv,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


SOURCE_HASHES = verify_sources()
MULTIPLER_MODULE = load_file_module("virne_oracle_multipler", source_dir() / "multipler.py")
LOSS_MODULE = load_file_module("virne_oracle_loss", source_dir() / "loss.py")
RES_MODULE = load_file_module("virne_oracle_res", source_dir() / "res.py")
SINKHORN_MODULE = load_file_module("virne_oracle_sinkhorn", source_dir() / "sinkhorn.py")
GRAPH_CONV_MODULE = load_file_module("virne_oracle_graph_conv", source_dir() / "graph_conv.py")
GNN = load_gnn_oracle(GRAPH_CONV_MODULE)


def values(shape: tuple[int, ...], offset: float = 0.0) -> torch.Tensor:
    count = math.prod(shape)
    return (
        torch.arange(count, dtype=torch.float32)
        .reshape(shape)
        .remainder(29.0)
        .sub(14.0)
        .div(9.0)
        .add(offset)
    )


def graph_batch() -> Batch:
    first = Data(
        x=values((3, 3)),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64
        ),
        edge_attr=values((4, 2), 0.1),
    )
    second = Data(
        x=values((4, 3), 0.25),
        edge_index=torch.tensor(
            [[0, 1, 2, 2, 3], [1, 0, 1, 3, 2]], dtype=torch.int64
        ),
        edge_attr=values((5, 2), -0.2),
    )
    return Batch.from_data_list([first, second])


def mix_tensor(checksum: int, value: torch.Tensor) -> int:
    payload = value.detach().to("cpu").contiguous().numpy().tobytes(order="C")
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "checksum": str(mix_tensor(FNV_OFFSET, value)),
    }


def state_records(model: nn.Module) -> list[dict[str, Any]]:
    return [
        tensor_record(name, value)
        for name, value in sorted(model.state_dict().items())
    ]


def fill_parameters(model: nn.Module, scale: float = 0.01) -> None:
    with torch.no_grad():
        for parameter_id, parameter in enumerate(model.parameters(), start=1):
            parameter.copy_(
                torch.arange(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                .reshape(parameter.shape)
                .add(float(parameter_id))
                .mul(scale)
            )


def case(name: str, model: nn.Module | None, outputs: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        "name": name,
        "state": [] if model is None else state_records(model),
        "outputs": [tensor_record(label, value) for label, value in outputs],
    }


def multiplier_case() -> dict[str, Any]:
    torch.manual_seed(41)
    model = MULTIPLER_MODULE.MultiplerNet(5).eval()
    output = model(values((7, 5)))
    return case(
        "multipler",
        model,
        [("output", output), ("rng_tail", torch.rand(4))],
    )


def loss_case() -> dict[str, Any]:
    h1 = values((8, 4))
    h2 = values((8, 4), 0.2).flip((0,))
    default_loss = LOSS_MODULE.BarlowTwinsContrastiveLoss()
    explicit_loss = LOSS_MODULE.BarlowTwinsContrastiveLoss(
        lambda_=0.125, batch_norm=False, eps=1e-4
    )
    return case(
        "barlow_twins",
        None,
        [("default", default_loss(h1, h2)), ("explicit", explicit_loss(h1, h2))],
    )


def residual_case() -> dict[str, Any]:
    torch.manual_seed(43)
    model = RES_MODULE.ResNetBlock(3).eval()
    return case(
        "res_net_block",
        model,
        [("output", model(values((2, 3, 4, 5)))), ("rng_tail", torch.rand(4))],
    )


def sinkhorn_case() -> dict[str, Any]:
    scores = torch.tensor(
        [[0.2, -0.1, 0.7], [0.4, 0.3, -0.5], [-0.2, 0.8, 0.1]],
        dtype=torch.float32,
    )
    simple = SINKHORN_MODULE.Sinkhorn(sinkhorn_iters=7, tau=0.7)
    sequential = SINKHORN_MODULE.SinkhornNetwork(
        max_iter=8, tau=0.8, epsilon=1e-4, log_forward=True,
        batched_operation=False,
    )
    batched = SINKHORN_MODULE.SinkhornNetwork(
        max_iter=8, tau=0.8, epsilon=1e-4, log_forward=True,
        batched_operation=True,
    )
    return case(
        "sinkhorn",
        None,
        [
            ("simple", simple(scores.unsqueeze(0))),
            ("sequential", sequential(scores)),
            ("batched", batched(scores)),
            (
                "padded",
                sequential(
                    torch.stack((scores, scores + 0.25)),
                    nrows=[3, 2],
                    ncols=[3, 2],
                ),
            ),
            ("dummy", sequential(scores, dummy=True)),
            (
                "logsumexp",
                SINKHORN_MODULE.logsumexp(
                    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
                ),
            ),
        ],
    )


def graph_helpers_case() -> dict[str, Any]:
    dense = values((2, 3, 4))
    mask = torch.tensor(
        [[True, False, True], [False, True, True]], dtype=torch.bool
    )
    adjacency = torch.tensor(
        [
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        ],
        dtype=torch.float32,
    )
    converted = GNN["DenseToSparse"]()(
        values((2, 3, 2)), adjacency, values((2, 3, 3), 1.0)
    )
    return case(
        "graph_helpers",
        None,
        [
            ("to_sparse_batch", GNN["to_sparse_batch"](dense, mask)),
            ("x", converted[0]),
            ("edge_index", converted[1]),
            ("edge_weight", converted[2]),
            ("batch", converted[3]),
        ],
    )


def attention_pool_case() -> dict[str, Any]:
    torch.manual_seed(47)
    model = GNN["GraphAttentionPooling"](4).eval()
    batch = graph_batch()
    x = values((7, 4))
    return case(
        "graph_attention_pool",
        model,
        [
            ("output", model(x, batch.batch, size=batch.num_graphs)),
            ("coefs", model.get_coefs(x)),
            ("rng_tail", torch.rand(4)),
        ],
    )


def gat_case() -> dict[str, Any]:
    torch.manual_seed(53)
    model = GNN["GATConvNet"](
        3, 4, num_layers=2, embedding_dim=5, num_heads=1, edge_dim=2,
        batch_norm=False, dropout_prob=1.0, return_batch=False,
        pooling="attention",
    ).eval()
    return case(
        "gat_conv_net",
        model,
        [("output", model(graph_batch())), ("rng_tail", torch.rand(4))],
    )


def edge_fusion_case() -> dict[str, Any]:
    torch.manual_seed(59)
    model = GNN["EdgeFusionGATConvNet"](
        3, 4, num_layers=2, embedding_dim=5, num_heads=1, edge_dim=2,
        batch_norm=False, dropout_prob=1.0, return_batch=True, pooling=None,
    ).eval()
    return case(
        "edge_fusion_gat_conv_net",
        model,
        [("output", model(graph_batch())), ("rng_tail", torch.rand(4))],
    )


def deep_edge_case() -> dict[str, Any]:
    torch.manual_seed(61)
    model = GNN["DeepEdgeFeatureGAT"](
        3, 4, 2, num_layers=3, alpha=0.2, theta=0.2,
        embedding_dim=5, num_heads=1, batch_norm=False,
        dropout_prob=1.0, return_batch=False, pooling="mean",
    )
    fill_parameters(model)
    model.eval()
    return case(
        "deep_edge_feature_gat",
        model,
        [("output", model(graph_batch())), ("rng_tail", torch.rand(4))],
    )


def gcn_attention_case() -> dict[str, Any]:
    torch.manual_seed(67)
    model = GNN["GCNConvNet"](
        3, 4, embedding_dim=5, num_layers=2, batch_norm=False,
        dropout_prob=1.0, return_batch=False, pooling="attention",
    ).eval()
    return case(
        "gcn_attention_pool",
        model,
        [("output", model(graph_batch())), ("rng_tail", torch.rand(4))],
    )


def verify_error_boundaries() -> dict[str, str]:
    checks = {
        "res_linear_net": lambda: RES_MODULE.ResLinearNet(4, 8),
        "nn_conv_net": lambda: GNN["NNConvNet"](3, 4, edge_dim=2),
        "pna_conv_net": lambda: GNN["PNAConvNet"](3, 4, edge_dim=2),
    }
    result: dict[str, str] = {}
    for name, constructor in checks.items():
        try:
            constructor()
        except Exception:
            result[name] = "python_constructor_error"
        else:
            raise RuntimeError(f"expected Python {name} construction to fail")
    return result


def exact_report() -> dict[str, Any]:
    with torch.inference_mode():
        cases = [
            multiplier_case(),
            loss_case(),
            residual_case(),
            sinkhorn_case(),
            graph_helpers_case(),
            attention_pool_case(),
            gat_case(),
            edge_fusion_case(),
            deep_edge_case(),
            gcn_attention_case(),
        ]
    cases.extend(lifecycle_cases())
    return {
        "component": "neural_network_remaining",
        "cases": cases,
        "boundaries": verify_error_boundaries(),
    }


def backward_case(name, model, output, inputs):
    output.square().sum().backward()
    outputs = [("output", output)]
    outputs.extend((f"input_grad_{i}", x.grad) for i, x in enumerate(inputs))
    outputs.extend(
        (f"grad.{name}", parameter.grad)
        for name, parameter in sorted(model.named_parameters())
        if parameter.grad is not None
    )
    outputs.append(("rng_tail", torch.rand(4)))
    return case(name, model, outputs)


def lifecycle_cases():
    results = []
    torch.manual_seed(71)
    model = RES_MODULE.ResNetBlock(3).double()
    x = values((3, 4, 5)).double().requires_grad_()
    results.append(backward_case("res_unbatched_backward", model, model(x), [x]))

    torch.manual_seed(73)
    model = MULTIPLER_MODULE.MultiplerNet(5)
    x = values((2, 3, 5)).requires_grad_()
    results.append(backward_case("multipler_backward", model, model(x), [x]))

    for norm in (False, True):
        torch.manual_seed(79)
        model = LOSS_MODULE.BarlowTwinsContrastiveLoss(lambda_=0.125, batch_norm=norm, eps=1e-4)
        x = values((8, 4)).requires_grad_()
        y = values((8, 4), 0.2).flip((0,)).requires_grad_()
        results.append(backward_case("loss_norm_backward" if norm else "loss_raw_backward", model, model(x, y), [x, y]))

    for batched in (False, True):
        torch.manual_seed(83)
        model = SINKHORN_MODULE.SinkhornNetwork(max_iter=5, tau=0.8, log_forward=False, batched_operation=batched)
        x = values((1, 2, 3)).requires_grad_()
        output = model.forward_log(x, r=torch.tensor([[0.5, 1.5]]), c=torch.tensor([[0.5, 0.75, 0.75]]), dummy_row=True)
        results.append(backward_case("sinkhorn_batched_backward" if batched else "sinkhorn_sequential_backward", model, output, [x]))

    torch.manual_seed(89)
    model = SINKHORN_MODULE.Sinkhorn(sinkhorn_iters=3, tau=0.7)
    x = values((2, 3, 3)).requires_grad_()
    result = backward_case("sinkhorn_simple_backward", model, model(x, eps=0.02), [x])
    result["outputs"].extend([
        tensor_record("row_norm", model.row_norm(x)),
        tensor_record("col_norm", model.col_norm(x)),
        tensor_record("logsumexp_dim", SINKHORN_MODULE.logsumexp(x, dim=1, keepdim=True)),
    ])
    results.append(result)

    outputs = []
    for batched in (False, True):
        model = SINKHORN_MODULE.SinkhornNetwork(max_iter=4, tau=0.8, batched_operation=batched)
        prefix = "batched_" if batched else "sequential_"
        x = values((1, 3, 3))
        outputs.append((prefix + "square", model.forward_log(x, nrows=[2], ncols=[3], dummy_row=True)))
        outputs.append((prefix + "dummy", model.forward_log_dummy(x, nrows=[2], ncols=[3], dummy_row=True)))
        outputs.append((prefix + "empty", model.forward_log(x, nrows=[0], ncols=[3])))
    results.append(case("sinkhorn_dummy_boundaries", None, outputs))

    for edges in (False, True):
        torch.manual_seed(97)
        model = GRAPH_CONV_MODULE.EdgeFusionGATConv(3, 4, edge_dim=2 if edges else None)
        model.reset_parameters()
        batch = graph_batch()
        batch.x.requires_grad_()
        inputs = [batch.x]
        if edges:
            batch.edge_attr.requires_grad_()
            inputs.append(batch.edge_attr)
        else:
            batch.edge_attr = None
        assert model.lin_src.weight is model.lin_dst.weight
        output = model(batch.x, batch.edge_index, batch.edge_attr)
        results.append(backward_case("edge_fusion_reset_edges_backward" if edges else "edge_fusion_reset_no_edges_backward", model, output, inputs))

    torch.manual_seed(101)
    model = GNN["GATConvNet"](3, 4, num_layers=3, embedding_dim=5, num_heads=7, edge_dim=2, batch_norm=True, dropout_prob=0.25, pooling="mean")
    batch = graph_batch()
    batch.x.requires_grad_()
    batch.edge_attr.requires_grad_()
    results.append(backward_case("gat_training_backward", model, model(batch), [batch.x, batch.edge_attr]))

    torch.manual_seed(103)
    model = GNN["DeepEdgeFeatureGAT"](3, 4, 2, num_layers=4, embedding_dim=5, batch_norm=True, dropout_prob=0.25, pooling="mean")
    fill_parameters(model)
    batch = graph_batch()
    batch.x.requires_grad_()
    batch.edge_attr.requires_grad_()
    results.append(backward_case("deep_gat_training_backward", model, model(batch), [batch.x, batch.edge_attr]))

    torch.manual_seed(107)
    model = GNN["GraphAttentionPooling"](4).double()
    x = values((7, 4)).double().requires_grad_()
    results.append(backward_case("attention_pool_backward", model, model(x, graph_batch().batch, size=3), [x]))

    torch.manual_seed(109)
    model = GNN["DenseToSparse"]()
    x = values((2, 3, 2)).requires_grad_()
    weights = values((2, 3, 3)).double()
    converted = model(x, values((2, 3, 3)), weights)
    output = torch.cat((converted[0].flatten(), converted[2]))
    results.append(backward_case("dense_to_sparse_x_backward", model, output, [x]))
    for scalar in (False, True):
        torch.manual_seed(113)
        model = GNN["GATConvNet"](3, 4, num_layers=2, embedding_dim=5, edge_dim=1 if scalar else 2, pooling="mean").eval()
        batch = graph_batch()
        batch.edge_attr = batch.edge_attr[:, 0].contiguous() if scalar else None
        results.append(case("gat_scalar_edges" if scalar else "gat_optional_edges", model,
                            [("output", model(batch)), ("rng_tail", torch.rand(4))]))
    torch.manual_seed(127)
    model = GNN["DeepEdgeFeatureGAT"](3, 4, None, num_layers=3, theta=0.0, embedding_dim=5, pooling="mean")
    fill_parameters(model)
    model.eval()
    batch = graph_batch()
    batch.edge_attr = None
    results.append(case("deep_no_edges_theta_zero", model,
                        [("output", model(batch)), ("rng_tail", torch.rand(4))]))
    return results


class AggregateWorkload:
    def __init__(self) -> None:
        self.batch = graph_batch()
        self.multiplier = MULTIPLER_MODULE.MultiplerNet(5)
        self.contrastive = LOSS_MODULE.BarlowTwinsContrastiveLoss()
        self.residual = RES_MODULE.ResNetBlock(3)
        self.sinkhorn_network = SINKHORN_MODULE.SinkhornNetwork(
            max_iter=6, tau=0.8, epsilon=1e-4, log_forward=True,
            batched_operation=True,
        )
        self.sinkhorn = SINKHORN_MODULE.Sinkhorn(sinkhorn_iters=5, tau=0.7)
        self.attention_pool = GNN["GraphAttentionPooling"](4)
        self.gat = GNN["GATConvNet"](
            3, 4, num_layers=2, embedding_dim=5, num_heads=1, edge_dim=2,
            batch_norm=False, dropout_prob=1.0, return_batch=False,
            pooling="attention",
        )
        self.edge_fusion = GNN["EdgeFusionGATConvNet"](
            3, 4, num_layers=2, embedding_dim=5, num_heads=1, edge_dim=2,
            batch_norm=False, dropout_prob=1.0, return_batch=False,
            pooling="mean",
        )
        self.deep = GNN["DeepEdgeFeatureGAT"](
            3, 4, 2, num_layers=3, alpha=0.2, theta=0.2,
            embedding_dim=5, num_heads=1, batch_norm=False,
            dropout_prob=1.0, return_batch=False, pooling="mean",
        )
        self.gcn = GNN["GCNConvNet"](
            3, 4, embedding_dim=5, num_layers=2, batch_norm=False,
            dropout_prob=1.0, return_batch=False, pooling="attention",
        )
        for model in (
            self.multiplier,
            self.residual,
            self.attention_pool,
            self.gat,
            self.edge_fusion,
            self.deep,
            self.gcn,
        ):
            fill_parameters(model)
            model.eval()

    def run(self) -> list[torch.Tensor]:
        mask = torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        )
        adjacency = torch.tensor(
            [
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            ],
            dtype=torch.float32,
        )
        converted = GNN["DenseToSparse"]()(
            values((2, 3, 2)), adjacency, values((2, 3, 3), 1.0)
        )
        scores = torch.stack((values((3, 3)), values((3, 3), 0.25)))
        return [
            self.multiplier(values((32, 5))),
            self.contrastive(values((32, 4)), values((32, 4), 0.2)),
            self.residual(values((4, 3, 8, 8))),
            self.sinkhorn_network(scores),
            self.sinkhorn(scores),
            GNN["to_sparse_batch"](values((2, 3, 4)), mask),
            converted[0],
            converted[1],
            converted[2],
            converted[3],
            self.attention_pool(
                values((7, 4)), self.batch.batch, size=self.batch.num_graphs
            ),
            self.gat(self.batch),
            self.edge_fusion(self.batch),
            self.deep(self.batch),
            self.gcn(self.batch),
        ]


def workload_checksum(outputs: list[torch.Tensor]) -> int:
    checksum = FNV_OFFSET
    for output in outputs:
        checksum = mix_tensor(checksum, output)
    return checksum


def benchmark_python(warmups: int, repetitions: int) -> dict[str, Any]:
    workload = AggregateWorkload()
    checksum: int | None = None
    with torch.inference_mode():
        for _ in range(warmups):
            current = workload_checksum(workload.run())
            checksum = current if checksum is None else checksum
            if current != checksum:
                raise RuntimeError("Python aggregate warmup checksum drift")
        samples: list[int] = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            outputs = workload.run()
            samples.append(time.perf_counter_ns() - started)
            if workload_checksum(outputs) != checksum:
                raise RuntimeError("Python aggregate benchmark checksum drift")
    return {"samples_ns": samples, "checksum": str(checksum)}


def run_native(
    path: pathlib.Path,
    benchmark: bool,
    warmups: int,
    repetitions: int,
    threads: int,
) -> dict[str, Any]:
    command = [
        str(path.resolve()),
        "--warmups",
        str(warmups),
        "--repetitions",
        str(repetitions),
        "--threads",
        str(threads),
    ]
    if benchmark:
        command.append("--benchmark")
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"native harness exited {process.returncode}: {process.stderr.strip()}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def first_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {"path": path, "python": expected, "cpp": actual}
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return {
                "path": path + ".keys",
                "python": sorted(expected),
                "cpp": sorted(actual),
            }
        for key in expected:
            mismatch = first_difference(expected[key], actual[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {"path": path + ".length", "python": len(expected), "cpp": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = first_difference(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return {"path": path, "python": expected, "cpp": actual}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups <= 0 or args.repetitions <= 0 or args.threads <= 0:
        parser.error("warmups and repetitions must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    expected = exact_report()
    native = run_native(args.native, False, args.warmups, args.repetitions, args.threads)
    mismatch = first_difference(expected, native)
    mismatches = [] if mismatch is None else [mismatch]
    if len(expected["cases"]) == len(native.get("cases", [])):
        mismatches = [difference for index, (left, right) in enumerate(zip(expected["cases"], native["cases"]))
                      if (difference := first_difference(left, right, f"$.cases[{index}]")) is not None]
        boundary_difference = first_difference(expected["boundaries"], native["boundaries"], "$.boundaries")
        if boundary_difference is not None:
            mismatches.append(boundary_difference)
    report: dict[str, Any] = {
        "component": "neural_network_remaining",
        "fixture_version": 2,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cases": expected["cases"],
        "oracle_backend": "ast_isolated_torch26_pyg27",
        "python_sources": SOURCE_HASHES,
        "exact_cases": f"{len(expected['cases'])}/{len(expected['cases'])}" if mismatch is None else "FAIL",
        "mismatches": mismatches,
        "benchmark_mismatches": [],
        "benchmark": "NOT_REQUESTED",
        "status": "PASS" if mismatch is None else "FAIL",
    }

    if mismatch is None and args.benchmark:
        native_timed = run_native(
            args.native, True, args.warmups, args.repetitions, args.threads
        )["benchmark"]
        python_timed = benchmark_python(args.warmups, args.repetitions)
        if native_timed["checksum"] != python_timed["checksum"]:
            report["benchmark_mismatches"] = [
                {
                    "path": "benchmark.checksum",
                    "python": python_timed["checksum"],
                    "cpp": native_timed["checksum"],
                }
            ]
            report["status"] = "FAIL"
        native_median = statistics.median(native_timed["samples_ns"])
        python_median = statistics.median(python_timed["samples_ns"])
        report["benchmark"] = {
            "name": native_timed["name"],
            "covered_valid_leaves": 10,
            "native_samples_ns": native_timed["samples_ns"],
            "python_samples_ns": python_timed["samples_ns"],
            "native_median_ns_per_pipeline": native_median,
            "python_median_ns_per_pipeline": python_median,
            "python_over_native_speedup": python_median / native_median,
            "output_checksum": native_timed["checksum"],
        }

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "component": report["component"],
                "exact_cases": report["exact_cases"],
                "mismatches": report["mismatches"],
                "benchmark": report["benchmark"],
                "status": report["status"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
