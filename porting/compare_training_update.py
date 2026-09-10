#!/usr/bin/env python3
"""Compare real Python update methods and optimizer state to the native trainer."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import pathlib
import statistics
import time
from types import SimpleNamespace, MethodType
import numpy as np
import torch
from torch.distributions import Categorical
import compare_cnn_gnn_mlp_policy as base

ROOT = base.SOURCE_ROOT
if torch.__version__ != "2.6.0+cpu": raise RuntimeError("this training oracle requires Torch 2.6.0+cpu")
INVENTORY = json.loads((pathlib.Path(__file__).parent / "results/learning_api_inventory_2026-09-08.json").read_text())
PINNED = {entry["path"]: entry["sha256"] for entry in INVENTORY["modules"]}

def source(relative):
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED[relative]: raise RuntimeError("source drift: " + relative)
    return ast.parse(path.read_text(encoding="utf-8"), filename=relative)

namespace = {"torch": torch, "F": torch.nn.functional, "np": np, "copy": copy,
             "Categorical": Categorical, "Batch": base.Batch}
utility_nodes = [node for node in source("utils.py").body
                 if (isinstance(node, ast.FunctionDef)
                     and node.name in {"apply_mask_to_logit", "get_observations_sample"})
                 or (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in {"NEG_TENSER", "ZERO_TENSER"}
                             for target in node.targets))]
exec(compile(ast.Module(utility_nodes, type_ignores=[]), "utils.py", "exec"), namespace)
for cls in source("rl_core/rl_solver.py").body:
    if not isinstance(cls, ast.ClassDef): continue
    wanted = {"evaluate_actions", "update_grad"} if cls.name == "RLSolver" else {"update"}
    if cls.name not in {"RLSolver", "PGSolver", "A2CSolver", "PPOSolver", "ARPPOSolver", "DDPGSolver"}: continue
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    exec(compile(ast.Module([cls], type_ignores=[]), "rl_solver.py", "exec"), namespace)

tree = source("rl_policy/mlp_policy.py")
tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
mlp_namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "MLPNet": base.GNN["MLPNet"],
                 "BaseActorCritic": base.CNN["BaseActorCritic"], "ActorCriticRegistry": base.CNN["ActorCriticRegistry"]}
exec(compile(tree, "mlp_policy.py", "exec"), mlp_namespace)
source("rl_core/buffer.py")
buffer_module = base.neural.load_file_module("training_oracle_buffer", ROOT / "rl_core/buffer.py")

def values(shape):
    return torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).remainder(17).sub(8).div(16)

def tensor_line(name, tensor):
    if tensor is None: return name + "|undefined"
    names = {torch.float32: "Float", torch.float64: "Double", torch.int64: "Long", torch.bool: "Bool",
             torch.complex64: "ComplexFloat", torch.complex128: "ComplexDouble"}
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()
    return name + "|" + names[tensor.dtype] + "|" + "".join(f"{i}," for i in tensor.shape) + "|" + raw

def optimizer_state(optimizer):
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    lines = []
    for i, parameter in enumerate(parameters):
        name, state = f"optimizer_{i}.", optimizer.state.get(parameter, {})
        for suffix, tensor in (("parameter", parameter), ("gradient", parameter.grad), ("step", state.get("step")),
                               ("mean", state.get("exp_avg")), ("variance", state.get("exp_avg_sq")), ("maximum", state.get("max_exp_avg_sq"))):
            lines.append(tensor_line(name + suffix, tensor))
    return lines

def adam_case(variant):
    a, b = values((2, 3)), values((3,))
    if variant == 1: a, b = a.double(), b.double()
    if variant == 2: a, b = a.t(), b.double()
    if variant == 3: a, b = torch.complex(a, a * 0.25), torch.complex(b, b * -0.5)
    a.requires_grad_(); b.requires_grad_()
    optimizer = torch.optim.Adam([
        {"params": [a], "lr": 0.003, "betas": (0.8, 0.95), "eps": 1e-7, "weight_decay": 0.02, "amsgrad": variant >= 2, "maximize": variant == 3},
        {"params": [b], "lr": 0.007, "betas": (0.85, 0.98), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": variant >= 2}], foreach=False)
    lines, saved = [], None
    for step in range(4):
        optimizer.zero_grad(set_to_none=step != 2)
        a.grad = torch.full_like(a, 0.1 + step * 0.03)
        if step != 1: b.grad = torch.full_like(b, -0.2 + step * 0.02)
        if step == 2:
            optimizer.load_state_dict(saved)
            optimizer.param_groups[0]["lr"] = 0.002
        norm = torch.nn.utils.clip_grad_norm_([a, b], 0.25)
        optimizer.step()
        lines += [f"STEP {step}", tensor_line("norm", norm), *optimizer_state(optimizer)]
        if step == 0: saved = copy.deepcopy(optimizer.state_dict())
    return lines

ALGORITHMS = ["PGSolver", "A2CSolver", "PPOSolver", "ARPPOSolver", "DDPGSolver"]
def make_solver(algorithm, masked, stop=False):
    torch.manual_seed(401)
    policy = mlp_namespace["MlpActorCritic"](3, 4, num_layers=2, embedding_dim=8, dropout_prob=0.0, batch_norm=False).train()
    optimizer = torch.optim.Adam([{"params": policy.actor.parameters(), "lr": 0.003},
                                  {"params": policy.critic.parameters(), "lr": 0.007}], weight_decay=0.01, foreach=False)
    instance = SimpleNamespace(policy=policy, optimizer=optimizer, device=torch.device("cpu"),
        config=SimpleNamespace(rl=SimpleNamespace(norm_advantage=True, norm_reward=True, clip_grad=True,
            max_grad_norm=0.25, target_kl=1e-12 if stop else None, mask_actions=True, maskable_policy=True),
            training=SimpleNamespace(distributed_training=False, log_interval=1)),
        repeat_times=1, batch_size=4, eps_clip=0.2, softmax_temp=1.0, coef_critic_loss=0.5,
        coef_entropy_loss=0.01, coef_mask_loss=0.3, criterion_critic=torch.nn.MSELoss(), update_time=0,
        logger=SimpleNamespace(log=lambda *a, **k: None), lr_scheduler=None)
    instance.evaluate_actions = MethodType(namespace["RLSolver"].evaluate_actions, instance)
    instance.update_grad = MethodType(namespace["RLSolver"].update_grad, instance)
    instance.update = MethodType(namespace[ALGORITHMS[algorithm]].update, instance)
    instance.preprocess_obs = lambda observations, device: {key: torch.stack([item[key] for item in observations]).to(device) for key in observations[0]}
    observations = values((6, 4, 3))
    masks = torch.ones((6, 4), dtype=torch.bool)
    masks[:, 3] = False; masks[3, 3] = True
    def reset_buffer():
        instance.buffer = buffer_module.RolloutBuffer()
        for i, action in enumerate([0, 1, 2, 3, 0, 1]):
            obs = {"p_net_x": observations[i]}
            if masked: obs["action_mask"] = masks[i]
            instance.buffer.add(obs, action, [0.1, -0.2, 0.4, 0.3, -0.1, 0.7][i], False,
                                np.array([3.0 if stop else -1.4], dtype=np.float32))
        instance.buffer.returns = [1.0, -0.5, 2.0, 0.3, -1.0, 1.2]
    instance.reset_buffer = reset_buffer
    return instance

def training_case(algorithm, masked=False, stop=False):
    solver = make_solver(algorithm, masked, stop)
    scheduler = []
    solver.lr_scheduler = SimpleNamespace(step=lambda: scheduler.extend([solver.buffer.size(), solver.update_time]))
    lines = []
    for update in range(2):
        solver.reset_buffer()
        loss = solver.update()
        lines += [f"UPDATE {update}", tensor_line("loss", loss), tensor_line("update_time", torch.tensor(solver.update_time)),
                  tensor_line("buffer_size", torch.tensor(solver.buffer.size())), *optimizer_state(solver.optimizer),
                  tensor_line("rng_tail", torch.rand(4))]
    return lines + [tensor_line("scheduler", torch.tensor(scheduler))]

def exact():
    lines = []
    for i in range(4): lines.extend([f"CASE adam_{i}", *adam_case(i)])
    for i in range(5): lines.extend([f"CASE training_{i}", *training_case(i)])
    lines.extend(["CASE training_mask", *training_case(2, True)])
    lines.extend(["CASE training_early_stop", *training_case(2, False, True)])
    return lines

def benchmark_python():
    samples, reference, loss = [], None, None
    for repetition in range(-3, 5):
        solver = make_solver(2, True)
        solver.reset_buffer()
        start = time.perf_counter_ns()
        loss = solver.update()
        elapsed = time.perf_counter_ns() - start
        parameters = list(solver.policy.parameters())
        if reference is None: reference = [parameter.detach().clone() for parameter in parameters]
        else: assert all(torch.equal(a, b) for a, b in zip(reference, parameters))
        if repetition >= 0: samples.append(elapsed)
    return samples, [tensor_line("loss", loss), *[tensor_line(f"parameter_{i}", value) for i, value in enumerate(reference)]]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    expected, actual = exact(), base.run_native(args.native, [])
    mismatch = base.first_difference(expected, actual)
    result = {"component": "training_update", "status": "PASS" if mismatch is None else "FAIL", "mismatch": mismatch,
        "cases": [line[5:] for line in expected if line.startswith("CASE ")], "records": len(expected),
        "python_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "cpp_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest(),
        "source_sha256": {key: PINNED[key] for key in ("rl_core/rl_solver.py", "rl_core/buffer.py", "rl_policy/mlp_policy.py", "utils.py")},
        "torch": torch.__version__, "benchmark": "NOT_REQUESTED"}
    if mismatch is None and args.benchmark:
        measured = base.run_native(args.native, ["--benchmark"])
        native_samples = json.loads(measured[0])["samples_ns"]
        samples, outputs = benchmark_python()
        mismatch = base.first_difference(outputs, measured[1:])
        result["benchmark"] = {"native_samples_ns": native_samples, "python_samples_ns": samples,
            "native_median_ns": statistics.median(native_samples), "python_median_ns": statistics.median(samples),
            "speedup": statistics.median(samples) / statistics.median(native_samples),
            "output_sha256": hashlib.sha256("\n".join(outputs).encode()).hexdigest(), "mismatch": mismatch}
        if mismatch: result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "records", "mismatch", "benchmark")}))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
