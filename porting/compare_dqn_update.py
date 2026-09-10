#!/usr/bin/env python3
"""DQN/DoubleDQN update and target-state differential against original methods."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import MethodType, SimpleNamespace
import torch
import compare_training_update as reference
from compare_learning_session import native_command

namespace = dict(reference.namespace)
tree = reference.source("rl_core/rl_solver.py")
for cls in tree.body:
    if not isinstance(cls, ast.ClassDef) or cls.name not in {"DQNSolver", "DoubleDQNSolver"}: continue
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update"]
    exec(compile(ast.Module([cls], type_ignores=[]), "rl_core/rl_solver.py", "exec"), namespace)


def feature_batch_norm(self, value):
    if value.dim() <= 2: return torch.nn.BatchNorm1d.forward(self, value)
    return torch.nn.BatchNorm1d.forward(self, value.reshape(-1, value.shape[-1])).reshape(value.shape)


def make_solver(double_q=False, batch_norm=False):
    torch.manual_seed(503)
    policy = reference.mlp_namespace["MlpActorCritic"](3,4,num_layers=2,embedding_dim=8,dropout_prob=0.,batch_norm=batch_norm)
    if batch_norm:
        for module in policy.modules():
            if isinstance(module, torch.nn.BatchNorm1d): module.forward = MethodType(feature_batch_norm, module)
    target = copy.deepcopy(policy)
    optimizer = torch.optim.Adam([{"params": policy.actor.parameters(), "lr": .003},
        {"params": policy.critic.parameters(), "lr": .007}], weight_decay=.01, foreach=False)
    solver = SimpleNamespace(policy=policy,target_policy=target,optimizer=optimizer,device=torch.device("cpu"),
        config=SimpleNamespace(rl=SimpleNamespace(gamma=.97,max_grad_norm=.25)),target_update_interval=2,
        update_time=0,batch_size=4,epsilon=.4,logging_order=[])
    solver.logger = SimpleNamespace(log=lambda info,step: solver.logging_order.extend([solver.buffer.size(),step]))
    solver.update = MethodType(namespace["DoubleDQNSolver" if double_q else "DQNSolver"].update, solver)
    solver.preprocess_obs = lambda records,device: {"p_net_x": torch.stack([obs["p_net_x"] for obs in records]).to(device)}
    observations = reference.values((6,4,3))
    next_observations = observations.flip((1,)) + .125
    def reset():
        solver.buffer = reference.buffer_module.RolloutBuffer()
        for i,action in enumerate([0,1,2,3,0,1]):
            solver.buffer.add({"p_net_x": observations[i]}, action, [.1,-.2,.4,.3,-.1,.7][i],
                [False,False,True,False,True,True][i], 0., next_obs={"p_net_x": next_observations[i]})
    solver.reset_buffer = reset
    return solver


def run(double_q,batch_norm):
    solver = make_solver(double_q,batch_norm)
    lines = []
    for update in range(4):
        solver.reset_buffer()
        loss = solver.update()
        lines += ["UPDATE " + str(update), reference.tensor_line("loss",loss),
            reference.tensor_line("update_time",torch.tensor(solver.update_time)),
            reference.tensor_line("buffer_size",torch.tensor(solver.buffer.size())),
            reference.tensor_line("target_synchronized",torch.tensor(update % 2 == 0)),
            *reference.base.module_lines(solver.policy,True), "TARGET", *reference.base.module_lines(solver.target_policy,False),
            *reference.optimizer_state(solver.optimizer), reference.tensor_line("rng_tail",torch.rand(4))]
    return lines + [reference.tensor_line("logging_order",torch.tensor(solver.logging_order))]


def exact():
    lines = []
    for batch_norm in (False,True):
        for double_q in (False,True):
            name = ("double_dqn" if double_q else "dqn") + ("_bn" if batch_norm else "")
            lines += ["CASE " + name, *run(double_q,batch_norm)]
    return lines


def benchmark():
    samples,expected = [],None
    for repetition in range(-3,5):
        solver = make_solver(True,False); solver.reset_buffer()
        start = time.perf_counter_ns()
        loss = solver.update()
        elapsed = time.perf_counter_ns()-start
        output = [reference.tensor_line("loss",loss),
            *[reference.tensor_line(f"parameter_{i}",p) for i,p in enumerate(solver.policy.parameters())]]
        if expected is None: expected = output
        elif expected != output: raise RuntimeError("DQN benchmark output drift")
        if repetition >= 0: samples.append(elapsed)
    return samples,expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--benchmark-from",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    report = {"component":"dqn_update","threads":1,"torch":torch.__version__,
        "source_sha256": {key:reference.PINNED[key] for key in ("rl_core/rl_solver.py","rl_core/buffer.py","rl_policy/mlp_policy.py","neural_network/mlp.py")},
        "reference_corrections": ["Feature-last BatchNorm flattens leading dimensions in the two BN cases; original rank-3 input has an incompatible channel dimension"],
        "preserved_source_behavior": ["Target sync on update zero before increment", "Unmasked target max and source buffer clear", "Double DQN online argmax followed by target gather", "Target BatchNorm remains in the source training mode"]}
    if args.benchmark_from:
        parity_bytes = args.benchmark_from.read_bytes(); parity = json.loads(parity_bytes)
        if parity.get("status") != "PASS" or parity.get("component") != "dqn_update" or parity["source_sha256"] != report["source_sha256"]:
            raise RuntimeError("matching passing DQN report required")
        measured = native_command(args.native,["--benchmark"])
        native_samples,actual = json.loads(measured[0])["samples_ns"],measured[1:]
        python_samples,expected = benchmark()
        report.update(differential_rerun=False,parity_reused_from=args.benchmark_from.name,
            parity_report_sha256=hashlib.sha256(parity_bytes).hexdigest(),warmups=3,repetitions=5,
            native_samples_ns=native_samples,python_samples_ns=python_samples,
            native_median_ns=statistics.median(native_samples),python_median_ns=statistics.median(python_samples),
            python_over_native_speedup=statistics.median(python_samples)/statistics.median(native_samples))
    else:
        expected,actual = exact(),native_command(args.native,[])
        report.update(cases=[line[5:] for line in expected if line.startswith("CASE ")],records=len(expected))
    mismatch = reference.base.first_difference(expected,actual)
    report.update(status="PASS" if mismatch is None else "FAIL",mismatch=mismatch,
        python_output_sha256=hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        native_output_sha256=hashlib.sha256("\n".join(actual).encode()).hexdigest(),
        native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","records","mismatch","python_over_native_speedup") if key in report}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
