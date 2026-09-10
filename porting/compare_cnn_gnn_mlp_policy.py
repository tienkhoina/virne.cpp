#!/usr/bin/env python3
"""Exact tensor bytes, state, gradients and paired timing for new policy leaves."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import time

import torch
from torch_geometric.data import Batch, Data

import compare_neural_network_remaining as neural


SOURCE_ROOT = neural.source_dir().parent
EXPECTED = {
    "rl_policy/cnn_policy.py": "8cc2ea694880370c7923a31235d624fd7fddf0c56635ac5a025b1747791fc0fc",
    "rl_policy/gnn_mlp_policy.py": "f60090e3e722175d9da60819353f7b1f86d8fca896e0ea295cf14d448fb1fb59",
    "rl_policy/base_policy.py": "8fff0492a113f34d2a3b8a6bc7ee9c665635b7c921c47450f6f9d7c44b8c52d4",
    "neural_network/mlp.py": "4f6d24dde669ec310fd2961bb1a6c0945f5a2d93f2042402025acf094eba89cb",
}


def load_oracles():
    for relative, expected in EXPECTED.items():
        actual = hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source drift: {relative}: {actual}")
    base = neural.load_file_module("oracle_policy_base", SOURCE_ROOT / "rl_policy/base_policy.py")
    mlp = neural.load_file_module("oracle_policy_mlp_leaf", SOURCE_ROOT / "neural_network/mlp.py")
    result = []
    for relative in ("rl_policy/cnn_policy.py", "rl_policy/gnn_mlp_policy.py"):
        tree = ast.parse((SOURCE_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = {
            "torch": torch, "nn": torch.nn, "F": torch.nn.functional,
            "BaseActorCritic": base.BaseActorCritic, "ActorCriticRegistry": base.ActorCriticRegistry,
            "MLPNet": mlp.MLPNet, "ResNetBlock": neural.RES_MODULE.ResNetBlock,
            **{name: neural.GNN[name] for name in ("GCNConvNet", "GATConvNet", "DeepEdgeFeatureGAT")},
        }
        exec(compile(tree, relative, "exec"), namespace)
        result.append(namespace)
    return result


CNN, GNN = load_oracles()
KINDS = ("GcnMlpActorCritic", "GatMlpActorCritic", "DeepEdgeFeatureGATActorCritic")


def values(shape):
    return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).remainder(17).sub(8).div(8)


def observation():
    a = Data(x=values((3, 3)), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), edge_attr=values((4, 2)))
    b = a.clone()
    b.x = values((4, 3))
    return {"p_net": Batch.from_data_list([a, b]), "v_node_x": values((2, 2))}


def tensor_line(name, tensor):
    if tensor is None:
        return name + "|undefined"
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"non-finite oracle tensor: {name}")
    dtype = {torch.float32: "Float", torch.float64: "Double", torch.int64: "Long", torch.bool: "Bool"}[tensor.dtype]
    shape = "".join(f"{size}," for size in tensor.shape)
    return f"{name}|{dtype}|{shape}|{tensor.detach().cpu().contiguous().numpy().tobytes().hex()}"


def module_lines(model, gradients):
    tensors = {"state." + name: value for name, value in model.state_dict().items()}
    if gradients:
        tensors.update({"grad." + name: value.grad for name, value in model.named_parameters()})
    return [tensor_line(name, value) for name, value in sorted(tensors.items())]


def cnn_case(index, backward, unbatched=False):
    torch.manual_seed(211)
    model = CNN["CnnActorCritic"](3, -7 if index == 1 else 5, 1 if index == 1 else 128)
    x = values((1, 5, 3) if unbatched else (2, 1, 5, 3))
    if backward:
        model.double()
        x = x.double().requires_grad_()
    else:
        model.eval()
    obs = {"p_net_x": x}
    logits, value = model.act(obs), model.evaluate(obs)
    if backward:
        (logits.square().sum() + value.square().sum()).backward()
    lines = [f"CASE cnn_{index}", tensor_line("logits", logits), tensor_line("value", value)]
    if backward:
        lines.append(tensor_line("input_grad", x.grad))
    return lines + module_lines(model, backward) + [tensor_line("rng_tail", torch.rand(4))]


def gnn_case(kind, backward, batch_norm=False):
    torch.manual_seed(223)
    model = GNN[KINDS[kind]](4, 3, 2, num_layers=3 if backward else 2, embedding_dim=8,
                             p_net_edge_dim=2, dropout_prob=0.25 if backward else 0.0, batch_norm=batch_norm)
    if kind == 2 and backward:
        neural.fill_parameters(model)
    if not backward:
        model.eval()
    obs = observation()
    if backward:
        obs["p_net"].x.requires_grad_()
        obs["p_net"].edge_attr.requires_grad_()
        obs["v_node_x"].requires_grad_()
    logits, value = model.act(obs), model.evaluate(obs)
    if backward:
        (logits.square().sum() + value.square().sum()).backward()
    lines = [f"CASE gnn_{kind}_{int(backward)}" + ("_bn" if batch_norm else ""), tensor_line("logits", logits), tensor_line("value", value)]
    if backward:
        lines.extend([tensor_line("p_grad", obs["p_net"].x.grad),
                      tensor_line("edge_grad", obs["p_net"].edge_attr.grad),
                      tensor_line("v_grad", obs["v_node_x"].grad)])
    return lines + module_lines(model, backward) + [tensor_line("rng_tail", torch.rand(4))]


def exact_lines():
    lines = []
    for index in range(4):
        lines.extend(cnn_case(index, index == 2, index == 3))
    for kind in range(3):
        for backward in (False, True):
            lines.extend(gnn_case(kind, backward))
    lines.extend(gnn_case(0, True, True))
    try:
        model = GNN[KINDS[0]](4, 3, 2, num_layers=2, embedding_dim=10, p_net_edge_dim=2, dropout_prob=0.0, batch_norm=True)
        model.act(observation())
    except RuntimeError as error:
        if "running_mean" not in str(error):
            raise
    else:
        raise RuntimeError("expected Python GNN/MLP BatchNorm dimension error")
    lines.extend(["CASE gnn_batch_norm_boundary", "error|batch_norm_shape"])
    return lines


def run_native(path, arguments):
    process = subprocess.run([str(path.resolve()), *map(str, arguments)], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"native harness exited {process.returncode}: {process.stderr}")
    return process.stdout.splitlines()


def benchmark_python():
    with torch.no_grad():
        torch.manual_seed(227)
        cnn = CNN["CnnActorCritic"](3, 5, 128).eval()
        gat = GNN[KINDS[1]](4, 3, 2, num_layers=2, embedding_dim=8, p_net_edge_dim=2, dropout_prob=0.0, batch_norm=False).eval()
        image, obs = {"p_net_x": values((16, 1, 32, 3))}, observation()
        def run():
            return [cnn.act(image), cnn.evaluate(image), gat.act(obs), gat.evaluate(obs)]
        reference = run()
        for _ in range(3):
            run()
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(10):
                output = run()
            samples.append((time.perf_counter_ns() - start) // 10)
            assert all(torch.equal(a, b) for a, b in zip(reference, output))
        return samples, [tensor_line(f"output_{index}", tensor) for index, tensor in enumerate(reference)]


def first_difference(expected, actual):
    context = ""
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<missing>"
        right = actual[index] if index < len(actual) else "<missing>"
        if left.startswith("CASE "):
            context = left[5:]
        if left != right:
            return {"case": context, "record": left.split("|")[0], "line": index,
                    "python_sha256": hashlib.sha256(left.encode()).hexdigest(),
                    "cpp_sha256": hashlib.sha256(right.encode()).hexdigest()}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    expected = exact_lines()
    actual = run_native(args.native, [args.threads])
    mismatch = first_difference(expected, actual)
    report = {"component": "cnn_gnn_mlp_policy", "status": "PASS" if mismatch is None else "FAIL",
              "python_sources": EXPECTED, "neural_sources": neural.SOURCE_HASHES,
              "torch_version": torch.__version__, "threads": args.threads,
              "exact_bytes": mismatch is None, "mismatch": mismatch,
              "cases": [line[5:] for line in expected if line.startswith("CASE ")],
              "record_count": len(expected), "raw_fixture_bytes": len("\n".join(expected)),
              "python_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest(),
              "cpp_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest(),
              "benchmark": "NOT_REQUESTED"}
    if mismatch is None and args.benchmark:
        native_lines = run_native(args.native, ["--benchmark", args.threads])
        native_samples = json.loads(native_lines[0])["samples_ns"]
        python_samples, outputs = benchmark_python()
        mismatch = first_difference(outputs, native_lines[1:])
        if mismatch is not None:
            report["status"] = "FAIL"
        report["benchmark"] = {"name": "cnn_and_gat_mlp_actor_critic", "warmups": 3, "repetitions": 5,
                               "iterations": 10, "native_samples_ns": native_samples,
                               "python_samples_ns": python_samples, "output_mismatch": mismatch,
                               "output_sha256": hashlib.sha256("\n".join(outputs).encode()).hexdigest(),
                               "native_median_ns": statistics.median(native_samples),
                               "python_median_ns": statistics.median(python_samples),
                               "python_over_native_speedup": statistics.median(python_samples) / statistics.median(native_samples)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "exact_bytes", "mismatch", "benchmark")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
