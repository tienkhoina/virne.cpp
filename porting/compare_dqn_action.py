#!/usr/bin/env python3
"""Original epsilon-greedy policy selection and NumPy RNG state restoration."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import torch
import compare_training_update as reference
from compare_learning_session import native_command

tree = reference.source("rl_core/rl_solver.py")
cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == "DQNSolver")
cls.bases,cls.keywords,cls.decorator_list = [],[],[]
cls.body = [node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in {"_calc_epsilon","select_action"}]
namespace = dict(reference.namespace)
exec(compile(ast.Module([cls],type_ignores=[]),"rl_core/rl_solver.py","exec"),namespace)
SELECTOR = namespace["DQNSolver"]


def rng_state(state):
    _,key,position,has_gaussian,gaussian = state
    return [reference.tensor_line("rng_key",torch.tensor(key.astype(np.int64))),
        reference.tensor_line("rng_position",torch.tensor(position)),
        reference.tensor_line("rng_has_gaussian",torch.tensor(bool(has_gaussian))),
        reference.tensor_line("rng_gaussian",torch.tensor(gaussian,dtype=torch.float64))]


def run(variant):
    np.random.seed(713)
    solver = SimpleNamespace(epsilon_start=.9,epsilon_end=.1,epsilon_decay=7.,epsilon=.9,steps_done=0,
        p_net_setting_num_nodes=4,config=SimpleNamespace(rl=SimpleNamespace(mask_actions=variant != 1,maskable_policy=variant != 2)))
    solver._calc_epsilon = MethodType(SELECTOR._calc_epsilon,solver)
    calls = 0
    def forward(obs):
        nonlocal calls
        calls += 1
        return torch.tensor([[.2,.6,-.4,.8]],dtype=torch.float32)
    solver.policy = SimpleNamespace(act=forward)
    actions,steps,counts,epsilons = [],[],[],[]
    obs = {"action_mask":torch.tensor([[True,True,False,False]])}
    for i in range(40):
        if i == 18:
            random_state,solver.steps_done,solver.epsilon = saved
            np.random.set_state(random_state)
        action,logprob = SELECTOR.select_action(solver,obs,sample=i not in (8,23))
        actions.append(action); steps.append(solver.steps_done); counts.append(calls); epsilons.append(solver.epsilon)
        if i == 10: saved = (copy.deepcopy(np.random.get_state()),solver.steps_done,solver.epsilon)
    return [reference.tensor_line("actions",torch.tensor(actions)),reference.tensor_line("steps",torch.tensor(steps)),
        reference.tensor_line("calls",torch.tensor(counts)),reference.tensor_line("epsilon",torch.tensor(epsilons,dtype=torch.float64)),
        reference.tensor_line("logprob",torch.from_numpy(logprob)),*rng_state(np.random.get_state())]


def gaussian_resume():
    random = np.random.RandomState(917)
    random.normal(); saved = random.get_state(); lines = rng_state(saved)
    random.random(); random.normal(); random.set_state(saved)
    values = [random.normal(),random.normal(),random.normal()]
    lines += [reference.tensor_line("normal",torch.tensor(values,dtype=torch.float64)),*rng_state(random.get_state())]
    return lines + [reference.tensor_line("invalid_rejected",torch.tensor(True)),*rng_state(random.get_state())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    expected = []
    for variant in range(3): expected += [f"CASE epsilon_{variant}",*run(variant)]
    expected += ["CASE rng_gaussian_resume",*gaussian_resume()]
    actual = native_command(args.native,[])
    mismatch = reference.base.first_difference(expected,actual)
    if mismatch and mismatch["record"] == "epsilon":
        index = mismatch["line"]
        decode = lambda line: np.frombuffer(bytes.fromhex(line.split("|")[-1]),dtype=np.float64)
        left,right = decode(expected[index]),decode(actual[index])
        mismatch["differing_values"] = [{"index":int(i),"python":float(left[i]),"cpp":float(right[i])} for i in np.flatnonzero(left != right)]
    report = {"component":"dqn_action","status":"PASS" if mismatch is None else "FAIL","mismatch":mismatch,
        "cases":[line[5:] for line in expected if line.startswith("CASE ")],"records":len(expected),
        "source_sha256":{"rl_core/rl_solver.py":reference.PINNED["rl_core/rl_solver.py"]},
        "python_output_sha256":hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "native_output_sha256":hashlib.sha256("\n".join(actual).encode()).hexdigest(),
        "native_sha256":hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "reference_corrections":[],"threads":1,"torch":torch.__version__,"numpy":np.__version__}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","records","mismatch")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
