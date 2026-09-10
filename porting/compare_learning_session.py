#!/usr/bin/env python3
"""End-to-end native learning against the original agent, converter and updates."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import compare_instance_environment as environment_oracle
import compare_training_update as training
import compare_rl_policy_remaining as policies

ROOT = training.ROOT
CASES = ["pg", "a2c", "ppo", "arppo", "source_ddpg", "cnn", "gcn", "gat", "attention", "dual_gcn", "failed", "dqn", "double_dqn", "double_dqn_gcn"]
CHECKPOINT_CASES = ["resume_mlp", "resume_gcn", "resume_dual", "resume_dqn", "resume_double_dqn"]
VALUE_CASES = {"dqn", "double_dqn", "double_dqn_gcn"}
SOURCE_HASHES = {}


class ConstantDistanceCorrection(ast.NodeTransformer):
    def visit_If(self, node):
        if isinstance(node.test, ast.Compare) and ast.unparse(node.test) == "np.max(avg_distance) == 0":
            node.test = ast.parse("np.max(avg_distance) == np.min(avg_distance)", mode="eval").body
            node.body = ast.parse("avg_distance = np.zeros_like(avg_distance)").body
        return self.generic_visit(node)


environment_oracle.patch_function(sys.modules["virne.solver.learning.obs_handler"].ObservationHandler,
    "get_average_distance", ConstantDistanceCorrection())


def selected_source(relative, predicate, namespace):
    path = ROOT / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != training.PINNED[relative]: raise RuntimeError("source drift: " + relative)
    SOURCE_HASHES[relative] = digest
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if predicate(node)]
    exec(compile(tree, str(path), "exec"), namespace)


converter_namespace = {"torch": torch, "np": np, "Data": Data, "Batch": Batch}
selected_source("utils.py", lambda n: isinstance(n, ast.FunctionDef) and n.name == "get_pyg_data", converter_namespace)
selected_source("rl_core/tensor_convertor.py", lambda n: isinstance(n, ast.ClassDef) and n.name == "TensorConvertor", converter_namespace)
CONVERT = converter_namespace["TensorConvertor"]


def methods(relative, owner, names, namespace):
    path = ROOT / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != training.PINNED[relative]: raise RuntimeError("source drift: " + relative)
    SOURCE_HASHES[relative] = digest
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module([cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[owner]


agent_namespace = {"torch": torch, "RolloutBuffer": training.buffer_module.RolloutBuffer}
AGENT = methods("rl_core/instance_agent.py", "InstanceAgent", {"learn_with_instance"}, agent_namespace)
SOLVER = methods("rl_core/rl_solver.py", "RLSolver", {"select_action", "estimate_value"}, dict(training.namespace))
DQN = methods("rl_core/rl_solver.py", "DQNSolver", {"update", "select_action", "_calc_epsilon"}, dict(training.namespace))
DOUBLE_DQN = methods("rl_core/rl_solver.py", "DoubleDQNSolver", {"update"}, dict(training.namespace))


def environment(name):
    env = environment_oracle.environment(failure=int(name == "failed"))
    if name == "dual_gcn":
        env.config.rl.feature_constructor.name = "p_net_v_net"
        registry = environment_oracle.ENVS.FeatureConstructorRegistry
        env.feature_constructor = registry.get("p_net_v_net")(env.p_net, env.v_net, env.config)
    return env


def build_policy(name, obs):
    if name == "double_dqn_gcn": name = "gcn"
    nodes, features = obs["p_net_x"].shape
    common = dict(num_layers=2, embedding_dim=8, dropout_prob=0.0, batch_norm=False)
    if name == "dual_gcn":
        return policies.DUAL["BiGcnActorCritic"](nodes, features, obs["p_net_edge_attr"].shape[1],
            obs["v_net_x"].shape[1], obs["v_net_edge_attr"].shape[1], **common)
    virtual = len(obs["v_node_x"])
    if name == "cnn": return policies.base.CNN["CnnActorCritic"](features + virtual, nodes, embedding_dim=8)
    if name in ("gcn", "gat"):
        return policies.base.GNN["GcnMlpActorCritic" if name == "gcn" else "GatMlpActorCritic"](nodes, features, virtual, **common)
    if name == "attention": return policies.ATT["AttActorCritic"](nodes, features, virtual, **common)
    return training.mlp_namespace["MlpActorCritic"](features + virtual, nodes, **common)


def build_solver(name, env):
    torch.manual_seed(401)
    policy = build_policy(name, env.get_observation())
    rates = {"actor": .003, "critic": .007, "encoder": .002, "cost_critic": .004, "lambda_net": .005}
    optimizer = torch.optim.Adam([{"params": child.parameters(), "lr": rates[key]}
                                  for key, child in policy.named_children()], weight_decay=.01, foreach=False)
    solver = SimpleNamespace(policy=policy, optimizer=optimizer, device=torch.device("cpu"),
        config=SimpleNamespace(rl=SimpleNamespace(norm_advantage=True, norm_reward=True, clip_grad=True,
            max_grad_norm=.25, target_kl=None, mask_actions=True, maskable_policy=True),
            training=SimpleNamespace(distributed_training=False, log_interval=1)),
        repeat_times=1, batch_size=4, eps_clip=.2, softmax_temp=1., coef_critic_loss=.5,
        coef_entropy_loss=.01, coef_mask_loss=.3, criterion_critic=torch.nn.MSELoss(), update_time=0,
        logger=SimpleNamespace(log=lambda *a, **k: None), lr_scheduler=None,
        buffer=training.buffer_module.RolloutBuffer(), controller=None, recorder=None, counter=None)
    algorithm = CASES.index(name) if name in CASES[:5] else 2
    for method in ("evaluate_actions", "update_grad"):
        setattr(solver, method, MethodType(getattr(training.namespace["RLSolver"], method), solver))
    solver.update = MethodType(training.namespace[training.ALGORITHMS[algorithm]].update, solver)
    solver.select_action = MethodType(SOLVER.select_action, solver)
    solver.estimate_value = MethodType(SOLVER.estimate_value, solver)
    converter_name = {"cnn": "cnn", "gcn": "gnn_mlp", "gat": "gnn_mlp", "attention": "att", "dual_gcn": "dual_gnn"}.get(name, "mlp")
    solver.preprocess_obs = getattr(CONVERT, "obs_as_tensor_for_" + converter_name)
    solver.InstanceEnv = lambda *a, **k: env
    if name in VALUE_CASES:
        solver.target_policy = copy.deepcopy(policy)
        solver.target_update_interval = 2
        solver.config.rl.gamma = .99
        solver.epsilon_start,solver.epsilon_end,solver.epsilon_decay = .6,.1,7.
        solver.epsilon,solver.steps_done = .6,0
        solver.p_net_setting_num_nodes = env.p_net.num_nodes
        solver._calc_epsilon = MethodType(DQN._calc_epsilon,solver)
        solver.select_action = MethodType(DQN.select_action,solver)
        solver.update = MethodType((DQN if name == "dqn" else DOUBLE_DQN).update,solver)
        if name == "double_dqn_gcn": solver.preprocess_obs = CONVERT.obs_as_tensor_for_gnn_mlp
        np.random.seed(713)
    return solver


def snapshot(solver, solution, buffer, last_value, loss):
    record = environment_oracle.record
    lines = [record("actions", buffer.actions, "int64")]
    for key, value in (("rewards", buffer.rewards), ("values", buffer.values),
                       ("logprobs", np.asarray(buffer.logprobs).reshape(-1)), ("returns", buffer.returns)):
        lines.append(record(key, value, "float64"))
    lines += [record("dones", buffer.dones, "bool"), record("last_value", last_value, "float64"),
        record("success", solution.result, "bool"), record("reward", solution.v_net_reward, "float64"),
        record("ratio", solution.v_net_r2c_ratio, "float64"), record("pending", solver.buffer.size(), "int64"),
        record("updates", solver.update_time, "int64")]
    lines.append("loss|undefined" if loss is None else record("loss", float(loss), "float64"))
    if hasattr(solver,"steps_done"):
        lines += [record("epsilon",solver.epsilon,"float64"),record("exploration_steps",solver.steps_done,"int64")]
    return lines


def parameters(solver):
    values = np.concatenate([p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in solver.policy.parameters()])
    lines = [environment_oracle.record("parameters", values, "float64")]
    if hasattr(solver,"target_policy"):
        target = np.concatenate([p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in solver.target_policy.parameters()])
        lines.append(environment_oracle.record("target_parameters",target,"float64"))
    return lines


def collect(solver, env):
    solution, buffer, last_value = AGENT.learn_with_instance(solver, {"p_net": env.p_net, "v_net": env.v_net})
    buffer.compute_returns_and_advantages(last_value, gamma=.99, gae_lambda=.98, method="gae")
    return solution, buffer, last_value


def run(name):
    env = environment(name)
    solver = build_solver(name, env)
    lines = ["CASE " + name]
    for episode in range(3):
        if episode: torch.manual_seed(600 + episode)
        solver.policy.train()
        solution, buffer, last_value = collect(solver, env)
        if solution.result or name in VALUE_CASES: solver.buffer.merge(buffer)
        loss = solver.update() if solver.buffer.size() >= (4 if name in VALUE_CASES else 8) else None
        lines += ["EPISODE " + str(episode), *snapshot(solver, solution, buffer, last_value, loss), *parameters(solver)]
    loss = solver.update() if solver.buffer.size() else None
    lines += ["flush_loss|undefined" if loss is None else environment_oracle.record("flush_loss", float(loss), "float64"), *parameters(solver)]
    for sample in (False, True):
        torch.manual_seed(710)
        solver.policy.eval()
        solver.select_action = lambda obs, **kwargs: (DQN if name in VALUE_CASES else SOLVER).select_action(solver, obs, sample=sample)
        solution, buffer, last_value = collect(solver, env)
        lines += ["SOLVE " + str(int(sample)), *snapshot(solver, solution, buffer, last_value, None), *parameters(solver)]
    return lines


def benchmark_python(value_learning=False):
    samples, expected = [], None
    for repetition in range(-3, 5):
        name = "double_dqn" if value_learning else "ppo"
        selection = DQN if value_learning else SOLVER
        env = environment(name)
        solver = build_solver(name, env)
        solver.policy.eval()
        solver.select_action = lambda obs, **kwargs: selection.select_action(solver, obs, sample=False)
        collect(solver, env)
        solver.select_action = MethodType(selection.select_action, solver)
        start = time.perf_counter_ns()
        for episode in range(3):
            torch.manual_seed(800 + episode)
            solver.policy.train()
            solution, buffer, last_value = collect(solver, env)
            if solution.result or value_learning: solver.buffer.merge(buffer)
            if solver.buffer.size() >= (4 if value_learning else 8): solver.update()
        loss = solver.update() if solver.buffer.size() else None
        elapsed = time.perf_counter_ns() - start
        output = [environment_oracle.record("actions", buffer.actions, "int64"),
            "loss|undefined" if loss is None else environment_oracle.record("loss", float(loss), "float64"), *parameters(solver)]
        if expected is None: expected = output
        elif expected != output: raise RuntimeError("Python benchmark output changed between repetitions")
        if repetition >= 0: samples.append(elapsed)
    return samples, expected


def checkpoint_case(name):
    kind = {"resume_mlp": "ppo", "resume_gcn": "gcn", "resume_dual": "dual_gcn", "resume_dqn":"dqn", "resume_double_dqn":"double_dqn"}[name]
    env = environment(kind)
    solver = build_solver(kind, env)
    lines = ["CASE " + name]
    for episode in range(5):
        solver.policy.train()
        solution, buffer, last_value = collect(solver, env)
        if solution.result or kind in VALUE_CASES: solver.buffer.merge(buffer)
        loss = solver.update() if solver.buffer.size() >= (4 if kind in VALUE_CASES else 8) else None
        if episode >= 3: lines += [*snapshot(solver, solution, buffer, last_value, loss), *parameters(solver)]
    return lines + [environment_oracle.record("invalid_rejected", True, "bool")]


def native_command(path, arguments):
    native_env = dict(os.environ)
    native_env["LD_LIBRARY_PATH"] = str(Path(__file__).resolve().parents[1] / "libs/libtorch/lib")
    process = subprocess.run([str(path.resolve()), *arguments], capture_output=True, text=True, env=native_env)
    if process.returncode: raise RuntimeError(process.stderr)
    return process.stdout.splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES + CHECKPOINT_CASES, default=CASES)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    reports = {}
    for name in args.cases:
        resume = name in CHECKPOINT_CASES
        expected = checkpoint_case(name) if resume else run(name)
        actual = native_command(args.native, ["--checkpoint-case" if resume else "--case", name])
        mismatch = policies.base.first_difference(expected, actual)
        signature = lambda lines: hashlib.sha256("\n".join(lines).encode()).hexdigest()
        reports[name] = {"status": "PASS" if mismatch is None else "FAIL", "records": len(expected),
            "python_sha256": signature(expected), "cpp_sha256": signature(actual), "mismatch": mismatch}
        print(json.dumps({"case": name, **reports[name]}), flush=True)
    report = {"component": "learning_session", "status": "PASS" if all(r["status"] == "PASS" for r in reports.values()) else "FAIL",
        "cases": reports, "source_sha256": {**SOURCE_HASHES, **policies.EXPECTED},
        "environment_source_sha256": environment_oracle.LOADER.hashes,
        "torch": torch.__version__, "threads": 1,
        "reference": "Original InstanceAgent.learn_with_instance, TensorConvertor, RLSolver action/value and update methods; native environment corrections are inherited from compare_instance_environment.",
        "additional_correction": "Constant average distances normalize to zeros; original nonzero constant distances divide by zero and Categorical rejects NaN logits.",
        "benchmark": "NOT_REQUESTED"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
